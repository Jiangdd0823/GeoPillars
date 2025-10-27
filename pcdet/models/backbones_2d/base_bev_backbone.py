import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, Identity
from ..model_utils.attention_utils import SE, CoordAtt, SplitAttentionImage, SplitAttentionLLC


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm=False,
                 bias=False,
                 activation=False,
                 ):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=1, padding=0)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = Swish()
            # self.swish = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneSplitAttn(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        # revise 1123
        self.attn = SplitAttentionImage(128)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # revise 1123
        x = self.attn(x)
        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneLargeKernel(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        # REVISE 1123 多加一层largekernel 不下采样
        cur_layer = []
        for i in range(layer_nums[0]):
            cur_layer.extend([
                # nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[0], num_filters[0], kernel_size=7,
                    stride=layer_strides[0], padding=3, bias=False
                ),
                nn.BatchNorm2d(num_filters[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        # revise 1123
        x = self.largekernel(spatial_features)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneDecouple(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        data_dict['spatial_features_2d'] = ups

        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class LargeKernel(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 3,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=7, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, input_channel, output_channel, attn: bool=False):
        super().__init__()
        mid_channel = output_channel // 2
        self.main_conv = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, 1, 1),
            nn.BatchNorm2d(mid_channel),
            Swish(),
        )
        self.short_conv = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, 1, 1),
            nn.BatchNorm2d(mid_channel),
            Swish(),
        )
        self.block = BasicBlock(mid_channel, mid_channel, 1)
        # fixme mmdet里这里用1x1，我这里用3x3
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.BatchNorm2d(output_channel),
            Swish(),
        )
        self.attention = None
        if attn:
            self.attention = CoordAtt(output_channel, output_channel)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.block(x_main)
        x = torch.cat([x_short, x_main], dim=1)
        if self.attention is not None:
            x = self.attention(x)
        x = self.final_conv(x)
        return x


class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


# fixme 1114 可以将全部block改为largekernel
class BaseBEVResBackbonePlus(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        # revise 1116 7x7和3x3交叉使用在第二层开始
        assert num_levels == 4
        cur_layer = []
        for i in range(layer_nums[0]):
            cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            # cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        # revise 1114 全为largekernel
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
                # LargeKernel(c_in_list[idx], num_filters[idx], layer_strides[idx], 3, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                    # LargeKernel(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        # revise 1203
        # if self.model_cfg.get('ATTN', 'SplitAttentionLLC'):
        #     self.attn = SplitAttentionLLC(128, k=3)
        #     self.num_bev_features = 384

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        # revise 1112 先经过largekel
        x = self.largekernel(spatial_features)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # revise 1203
        # if self.model_cfg.get('ATTN', 'SplitAttentionLLC'):
        #     x = torch.stack(ups, dim=1)
        #     x = self.attn(x)
        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVResBackboneDualFPN(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        # assert num_levels == 4
        cur_layer = []
        for i in range(layer_nums[0]):
            # cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            # if len(upsample_strides) > 0:
            #     stride = upsample_strides[idx]
            #     if stride >= 1:
            #         self.deblocks.append(nn.Sequential(
            #             nn.ConvTranspose2d(
            #                 num_filters[idx], num_upsample_filters[idx],
            #                 upsample_strides[idx],
            #                 stride=upsample_strides[idx], bias=False
            #             ),

            #             nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
            #             nn.ReLU()
            #         ))
            #     else:
            #         stride = np.round(1 / stride).astype(np.int)
            #         self.deblocks.append(nn.Sequential(
            #             nn.Conv2d(
            #                 num_filters[idx], num_upsample_filters[idx],
            #                 stride,
            #                 stride=stride, bias=False
            #             ),
            #             nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
            #             nn.ReLU()
            #         ))

        # c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        # if len(upsample_strides) > num_levels:
        #     self.deblocks.append(nn.Sequential(
        #         nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
        #         nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
        #         nn.ReLU(),
        #     ))

        # self.num_bev_features = c_in


        # self.trans_channel54 = nn.Sequential(
        #     nn.Conv2d(512, 256, 1, 1, bias=False),
        #     nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        # )
        self.deblocks_54 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        # self.trans_channel53 = nn.Sequential(
        #     nn.Conv2d(256, 128, 1, 1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        # )
        self.deblocks_53 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        # self.trans_channel43 = nn.Sequential(
        #     nn.Conv2d(256, 128, 1, 1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        # )
        self.deblocks_43 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        # self.trans_channel42 = nn.Sequential(
        #     nn.Conv2d(256, 128, 1, 1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        # )
        self.deblocks_42 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deblocks_32 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        self.deblocks_31 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.deblocks_21 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.trans_channel11 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.num_bev_features = 384
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     init.constant_(m.weight, 1)
            #     init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        # revise 1112 先经过largekel
        c1 = self.largekernel(spatial_features)

        c2 = self.blocks[0](c1)
        c3 = self.blocks[1](c2)
        c4 = self.blocks[2](c3)
        c5 = self.blocks[3](c4)
        # revise 1113 舍弃不同通道特征之间的1x1卷积
        # p5 = self.trans_channel54(c5)
        # p54 = self.deblocks_54(p5)
        p54 = self.deblocks_54(c5)
        p4 = c4 + p54
        # p4 = self.trans_channel43(p4)
        p43 = self.deblocks_43(p4)
        # p53 = self.trans_channel53(p5)
        # p53 = self.deblocks_53(p53)
        p53 = self.deblocks_53(c5)
        p3 = c3 + p43 + p53
        # p42 = self.trans_channel42(p4)
        p42 = self.deblocks_42(p4)
        p32 = self.deblocks_32(p3)
        p2 = c2 + p42 + p32
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        p21 = self.deblocks_21(p2)
        p31 = self.deblocks_31(p3)
        p1 = self.trans_channel11(c1)
        p1 = p1 + p21 + p31

        t2 = self.smooth2(p2)
        t3 = self.smooth3(p32)
        t4 = self.smooth4(p42)
        x = torch.cat([t2, t3, t4], dim=1)
        # x = torch.cat([p1, p2], dim=1)

        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #
        #     stride = int(spatial_features.shape[2] / x.shape[2])
        #     ret_dict['spatial_features_%dx' % stride] = x
        #     if len(self.deblocks) > 0:
        #         ups.append(self.deblocks[i](x))
        #     else:
        #         ups.append(x)
        #
        # if len(ups) > 1:
        #     x = torch.cat(ups, dim=1)
        # elif len(ups) == 1:
        #     x = ups[0]
        #
        # if len(self.deblocks) > len(self.blocks):
        #     x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


# todo 这是特征相加的版本
class BaseBEVResBackboneDualFPN1114(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.act = Swish()
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        cur_layer = []
        for i in range(layer_nums[0]):
            # cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
        # revise 1114 1x1卷积后加swish(), relu改为swish()
        self.trans_channel54 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_54 = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                self.act,
            )
        self.trans_channel53 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_53 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel43 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # revise 1114
        self.deblocks_43 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel42 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_42 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel32 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_32 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        self.deblocks_31 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel22 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_21 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel11 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.num_bev_features = 384

        # revise 1114 因为relu换成swish，kaiming_normal_不适合
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     init.constant_(m.weight, 1)
            #     init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        c1 = self.largekernel(spatial_features)

        c2 = self.blocks[0](c1)
        c3 = self.blocks[1](c2)
        c4 = self.blocks[2](c3)
        c5 = self.blocks[3](c4)
        # revise 1113 舍弃不同通道特征之间的1x1卷积
        # revise 1114 使用不同通道特征之间的1x1卷积,相加后的连接处用conv3x3平滑
        p5 = self.trans_channel54(c5)
        p54 = self.deblocks_54(p5)
        # p54 = self.deblocks_54(c5)
        p4 = c4 + p54
        p4 = self.trans_channel43(p4)
        p43 = self.deblocks_43(p4)
        p53 = self.trans_channel53(p5)
        p53 = self.deblocks_53(p53)
        # p53 = self.deblocks_53(c5)
        p3 = c3 + p43 + p53
        p3 = self.trans_channel32(p3)
        p32 = self.deblocks_32(p3)
        p42 = self.trans_channel42(p4)
        p42 = self.deblocks_42(p42)
        p2 = c2 + p42 + p32
        p2 = self.trans_channel22(p2)
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        # p21 = self.deblocks_21(p2)
        # p31 = self.deblocks_31(p3)
        # p1 = self.trans_channel11(c1)
        # p1 = p1 + p21 + p31
        # p1 = self.reblock_12(p1)

        t2 = self.smooth2(p2)
        t3 = self.smooth3(p32)
        t4 = self.smooth4(p42)
        x = torch.cat([t2, t3, t4], dim=1)

        data_dict['spatial_features_2d'] = x

        return data_dict


# todo 特征cat版本
# revise 1115 将全部block改为largekernel
# revise 1115 全largekernel版本不好
class BaseBEVResBackboneDualFPN1115(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.act = Swish()
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        cur_layer = []
        for i in range(layer_nums[0]):
            cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            # cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                # BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
                LargeKernel(c_in_list[idx], num_filters[idx], layer_strides[idx], 3, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    # BasicBlock(num_filters[idx], num_filters[idx])
                    LargeKernel(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
        # revise 1114 1x1卷积后加swish(), relu改为swish()
        # revise 1114 在反卷积后加入1x1平滑再bn act；特征cat后经过bottleneck特征提取
        self.trans_channel54 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_54 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.Conv2d(256, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
            )
        self.trans_channel53 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_53 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel43 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck4 = BottleNeck(512, 128, attn=False)
        self.deblocks_43 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel42 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_42 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel32 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck3 = BottleNeck(384, 128, attn=False)
        self.deblocks_32 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        self.deblocks_31 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel22 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck2 = BottleNeck(384, 128, attn=False)
        self.deblocks_21 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.num_bev_features = 384

        # revise 1114 因为relu换成swish，kaiming_normal_不适合
        # revise 1114 kaiming_normal_适合
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        c1 = self.largekernel(spatial_features)
        c2 = self.blocks[0](c1)
        c3 = self.blocks[1](c2)
        c4 = self.blocks[2](c3)
        c5 = self.blocks[3](c4)
        # revise 1113 舍弃不同通道特征之间的1x1卷积
        # revise 1114 使用不同通道特征之间的1x1卷积,相加后的连接处用conv3x3平滑
        # revise 1115 特征cat
        p5 = self.trans_channel54(c5)
        p54 = self.deblocks_54(p5)
        # p54 = self.deblocks_54(c5)
        # p4 = c4 + p54
        # p4 = self.trans_channel43(p4)
        p4 = torch.cat([p54, c4], dim=1)
        p4 = self.bottleneck4(p4)
        p43 = self.deblocks_43(p4)
        p53 = self.trans_channel53(p5)
        p53 = self.deblocks_53(p53)
        # p53 = self.deblocks_53(c5)
        # p3 = c3 + p43 + p53
        # p3 = self.trans_channel32(p3)
        p3 = torch.cat([c3, p43, p53], dim=1)
        p3 = self.bottleneck3(p3)
        p32 = self.deblocks_32(p3)
        p42 = self.trans_channel42(p4)
        p42 = self.deblocks_42(p42)
        # p2 = c2 + p42 + p32
        # p2 = self.trans_channel22(p2)
        p2 = torch.cat([c2, p42, p32], dim=1)
        p2 = self.bottleneck2(p2)
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        # p21 = self.deblocks_21(p2)
        # p31 = self.deblocks_31(p3)
        # p1 = self.trans_channel11(c1)
        # p1 = p1 + p21 + p31
        # p1 = self.reblock_12(p1)

        t2 = self.smooth2(p2)
        t3 = self.smooth3(p32)
        t4 = self.smooth4(p42)
        x = torch.cat([t2, t3, t4], dim=1)

        data_dict['spatial_features_2d'] = x

        return data_dict


# todo 标准fpn改法
class BaseBEVResBackboneFPN1115(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.act = Swish()
        self.attn = False
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        cur_layer = []
        for i in range(layer_nums[0]):
            cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            # cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                # BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
                LargeKernel(c_in_list[idx], num_filters[idx], layer_strides[idx], 3, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                    # LargeKernel(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
        # revise 1114 1x1卷积后加swish(), relu改为swish()
        # revise 1114 在反卷积后加入1x1平滑再bn act；特征cat后经过bottleneck特征提取
        self.trans_channel54 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_54 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.Conv2d(256, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
            )
        self.trans_channel53 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_53 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel43 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck4 = BottleNeck(512, 128, attn=self.attn)
        self.deblocks_43 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel42 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_42 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel32 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck3 = BottleNeck(256, 128, attn=self.attn)
        self.deblocks_32 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        self.deblocks_31 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel22 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck2 = BottleNeck(256, 128, attn=self.attn)
        self.deblocks_21 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.num_bev_features = 384

        # revise 1114 因为relu换成swish，kaiming_normal_不适合
        # revise 1114 kaiming_normal_适合
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        c1 = self.largekernel(spatial_features)
        c2 = self.blocks[0](c1)
        c3 = self.blocks[1](c2)
        c4 = self.blocks[2](c3)
        c5 = self.blocks[3](c4)
        # revise 1113 舍弃不同通道特征之间的1x1卷积
        # revise 1114 使用不同通道特征之间的1x1卷积,相加后的连接处用conv3x3平滑
        # revise 1115 特征cat
        p5 = self.trans_channel54(c5)
        p54 = self.deblocks_54(p5)
        # p54 = self.deblocks_54(c5)
        # p4 = c4 + p54
        # p4 = self.trans_channel43(p4)
        p4 = torch.cat([p54, c4], dim=1)
        p4 = self.bottleneck4(p4)
        p43 = self.deblocks_43(p4)
        # p53 = self.trans_channel53(p5)
        # p53 = self.deblocks_53(p53)
        # p53 = self.deblocks_53(c5)
        # p3 = c3 + p43 + p53
        # p3 = self.trans_channel32(p3)
        # p3 = torch.cat([c3, p43, p53], dim=1)
        p3 = torch.cat([c3, p43], dim=1)
        p3 = self.bottleneck3(p3)
        p32 = self.deblocks_32(p3)
        # p42 = self.trans_channel42(p4)
        # p42 = self.deblocks_42(p42)
        # p2 = c2 + p42 + p32
        # p2 = self.trans_channel22(p2)
        p2 = torch.cat([c2, p32], dim=1)
        p2 = self.bottleneck2(p2)
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        # p21 = self.deblocks_21(p2)
        # p31 = self.deblocks_31(p3)
        # p1 = self.trans_channel11(c1)
        # p1 = p1 + p21 + p31
        # p1 = self.reblock_12(p1)

        t2 = self.smooth2(p2)
        t3 = self.smooth3(p32)
        p42 = self.deblocks_42(p43)
        t4 = self.smooth4(p42)
        x = torch.cat([t2, t3, t4], dim=1)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVResBackboneFPN1116(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.act = Swish()
        self.attn = True
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        cur_layer = []
        for i in range(layer_nums[0]):
            cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            # cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
                # LargeKernel(c_in_list[idx], num_filters[idx], layer_strides[idx], 3, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                    # LargeKernel(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
        # revise 1114 1x1卷积后加swish(), relu改为swish()
        # revise 1114 在反卷积后加入1x1平滑再bn act；特征cat后经过bottleneck特征提取
        self.trans_channel54 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_54 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.Conv2d(256, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            self.act,
            )
        self.trans_channel53 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_53 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel43 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck4 = BottleNeck(512, 128, attn=self.attn)
        self.deblocks_43 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.trans_channel42 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_42 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel32 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck3 = BottleNeck(256, 128, attn=self.attn)
        self.deblocks_32 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        self.deblocks_31 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.trans_channel22 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.bottleneck2 = BottleNeck(256, 128, attn=self.attn)
        self.deblocks_21 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        # self.smooth2 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        # self.smooth3 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        # self.smooth4 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     self.act,
        # )
        self.redeblocks_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.redeblocks_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.conv_fuse2 = SeparableConvBlock(128, 128)
        self.conv_fuse3 = SeparableConvBlock(128, 128)
        self.conv_fuse4 = SeparableConvBlock(128, 128)
        self.num_bev_features = 384

        # revise 1114 因为relu换成swish，kaiming_normal_不适合
        # revise 1114 kaiming_normal_适合
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        c1 = self.largekernel(spatial_features)
        c2 = self.blocks[0](c1)
        c3 = self.blocks[1](c2)
        c4 = self.blocks[2](c3)
        c5 = self.blocks[3](c4)
        # revise 1113 舍弃不同通道特征之间的1x1卷积
        # revise 1114 使用不同通道特征之间的1x1卷积,相加后的连接处用conv3x3平滑
        # revise 1115 特征cat
        p5 = self.trans_channel54(c5)
        p54 = self.deblocks_54(p5)
        # p54 = self.deblocks_54(c5)
        # p4 = c4 + p54
        # p4 = self.trans_channel43(p4)
        p4 = torch.cat([p54, c4], dim=1)
        p4 = self.bottleneck4(p4)
        p43 = self.deblocks_43(p4)
        # p53 = self.trans_channel53(p5)
        # p53 = self.deblocks_53(p53)
        # p53 = self.deblocks_53(c5)
        # p3 = c3 + p43 + p53
        # p3 = self.trans_channel32(p3)
        # p3 = torch.cat([c3, p43, p53], dim=1)
        p3 = torch.cat([c3, p43], dim=1)
        p3 = self.bottleneck3(p3)
        p32 = self.deblocks_32(p3)
        # p42 = self.trans_channel42(p4)
        # p42 = self.deblocks_42(p42)
        # p2 = c2 + p42 + p32
        # p2 = self.trans_channel22(p2)
        p2 = torch.cat([c2, p32], dim=1)
        p2 = self.bottleneck2(p2)
        # revise 1113 使用第一和第二层cat，行人和车掉点严重，需cat后面层特征。所以舍弃第一层使用后面层cat
        # p21 = self.deblocks_21(p2)
        # p31 = self.deblocks_31(p3)
        # p1 = self.trans_channel11(c1)
        # p1 = p1 + p21 + p31
        # p1 = self.reblock_12(p1)
        t2 = self.conv_fuse2(p2 * 0.5 + c2 * 0.5)
        # t2 = self.smooth2(p2)
        c_t3 = self.redeblocks_3(c3)
        t3 = self.conv_fuse3(p32 * 0.5 + c_t3 * 0.5)
        # t3 = self.smooth3(p32)
        p42 = self.deblocks_42(p43)
        c_t4 = self.redeblocks_4(c4)
        t4 = self.conv_fuse4(p42 * 0.5 + c_t4 * 0.5)
        # t4 = self.smooth4(p42)
        x = torch.cat([t2, t3, t4], dim=1)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVResBackboneSSD1118(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.act = Swish()
        self.attn = True
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        # if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
        #     assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
        #     num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        #     upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        # else:
        #     upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # revise 1112 加入一层不下采样的
        cur_layer = []
        for i in range(layer_nums[0]):
            cur_layer.extend([LargeKernel(num_filters[0], num_filters[0], layer_strides[0], 3, False)])
            # cur_layer.extend([BasicBlock(num_filters[0], num_filters[0], layer_strides[0], 1, False)])
        self.largekernel = nn.Sequential(*cur_layer)
        layer_nums = layer_nums[1:]
        layer_strides = layer_strides[1:]
        num_filters = num_filters[1:]

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels-1):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, False)
                # LargeKernel(c_in_list[idx], num_filters[idx], layer_strides[idx], 3, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                    # LargeKernel(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

        # revise 1118 修改整体backbone
        self.largekernel_block1 = nn.Sequential(
            LargeKernel(inplanes=64, planes=128, stride=4, downsample=True),
            BasicBlock(128, 128, 1, False),
        )
        self.largekernel_block2 = nn.Sequential(
            LargeKernel(inplanes=128, planes=256, stride=4, downsample=True),
            BasicBlock(256, 256, 1, False),
        )

        # revise 1114 1x1卷积后加swish(), relu改为swish()
        # revise 1114 在反卷积后加入1x1平滑再bn act；特征cat后经过bottleneck特征提取

        self.deblocks_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )

        self.deblocks_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.deblocks_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 1, stride=1, bias=False),
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            self.act,
        )
        self.num_bev_features = 384

    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        # down
        c1 = self.largekernel(spatial_features)
        c2_2 = self.largekernel_block1(c1)
        c2 = self.blocks[0](c1)
        c3 = self.blocks[1](c2)
        c3_2 = self.largekernel_block2(c2)
        c3_new = c3 + c2_2
        c4 = self.blocks[2](c3)
        c4_new = c4 + c3_2

        # up
        p4 = self.deblocks_4(c4_new)
        p3 = self.deblocks_3(c3_new)
        p2 = self.deblocks_2(c2)
        x = torch.cat([p2, p3, p4], dim=1)
        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackbone1119(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.largekernel_block = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            large_layer = []
            if idx < 2:
                large_layer = [
                    # nn.ZeroPad2d(1),
                    nn.Conv2d(
                        c_in_list[idx], num_filters[idx+1], kernel_size=7,
                        stride=4, padding=3, bias=False
                    ),
                    nn.BatchNorm2d(num_filters[idx+1], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
            # elif idx == 1:
            #     large_layer = [
            #         # nn.ZeroPad2d(1),
            #         nn.Conv2d(
            #             c_in_list[idx], num_filters[idx+1], kernel_size=7,
            #             stride=4, padding=3, bias=False
            #         ),
            #         nn.BatchNorm2d(num_filters[idx+1], eps=1e-3, momentum=0.01),
            #         nn.ReLU()
            #     ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
                if idx < 2:
                    large_layer.extend(([
                        nn.Conv2d(num_filters[idx+1], num_filters[idx+1], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx+1], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ]))
            self.blocks.append(nn.Sequential(*cur_layers))
            self.largekernel_block.append(nn.Sequential(*large_layer))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        # self.largekernel_block1 = nn.Sequential(
        #     LargeKernel(inplanes=64, planes=256, stride=4, downsample=True),
        #     BasicBlock(256, 256, 1, False),
        # )
        # self.largekernel_block2 = nn.Sequential(
        #     LargeKernel(inplanes=128, planes=256, stride=4, downsample=True),
        #     BasicBlock(256, 256, 1, False),
        # )
        self.fuse_conv1 = SeparableConvBlock(128)
        self.fuse_conv2 = SeparableConvBlock(256)

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        spatial_features = data_dict['spatial_features']        # 496, 432
        # down
        c1 = self.blocks[0](spatial_features)              # 248, 216
        c1_ = self.largekernel_block[0](spatial_features)    # 124, 108
        c2 = self.blocks[1](c1)                             # 124, 108
        c2_new = self.fuse_conv1(c2 * 0.5 + c1_ * 0.5)
        c2_ = self.largekernel_block[1](c1)               # 62, 54
        c3 = self.blocks[2](c2_new)                         # 62, 54
        c3_new = self.fuse_conv2(c3 * 0.5 + c2_ * 0.5)

        # up
        p1 = self.deblocks[0](c1)
        p2 = self.deblocks[1](c2_new)
        p3 = self.deblocks[2](c3_new)
        x = torch.cat([p1, p2, p3], dim=1)

        # spatial_features = data_dict['spatial_features']
        # ups = []
        # ret_dict = {}
        # x = spatial_features
        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #
        #     stride = int(spatial_features.shape[2] / x.shape[2])
        #     ret_dict['spatial_features_%dx' % stride] = x
        #     if len(self.deblocks) > 0:
        #         ups.append(self.deblocks[i](x))
        #     else:
        #         ups.append(x)
        #
        # if len(ups) > 1:
        #     x = torch.cat(ups, dim=1)
        # elif len(ups) == 1:
        #     x = ups[0]
        #
        # if len(self.deblocks) > len(self.blocks):
        #     x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackbone1120(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        # revise 1120
        # self.separbleconv = SeparableConvBlock(in_channels=c_in)
        self.bottleneck = BottleNeck(input_channel=c_in, output_channel=c_in)
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # revise 1120
        # x = self.separbleconv(x)
        x = self.bottleneck(x)
        data_dict['spatial_features_2d'] = x

        return data_dict