import math

import torch
import torch.nn as nn


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


class MiniBiFPN(nn.Module):  # input size: batch, C, width, length [1,64,400,600]
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        num_blocks = model_cfg.num_blocks
        mid_planes = model_cfg.mid_planes
        strides = model_cfg.layer_strides
        upsample_strides = model_cfg.upsample_strides
        num_upsample_filters = model_cfg.num_upsample_filters
        num_input_filters = model_cfg.num_input_filters

        self.act = Swish()
        self.layer1 = [
            nn.Conv2d(num_input_filters, mid_planes[0], 3, stride=strides[0], padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_planes[0], eps=1e-3, momentum=0.01),
            self.act
        ]
        for i in range(num_blocks[0]):
            # conv = nn.Conv2d if i < num_blocks[0] -1 else DeformableConv2d
            conv = nn.Conv2d
            self.layer1 += [conv(mid_planes[0], mid_planes[0], 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_planes[0], eps=1e-3, momentum=0.01),
                            self.act]
        self.layer1 = nn.Sequential(*self.layer1)

        upsample_strides = list(map(float, upsample_strides))
        print('upsample_strides',upsample_strides)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[0],
                               int(upsample_strides[0]),
                               stride=int(upsample_strides[0]), bias=False
                               ) if upsample_strides[0] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[0],
                      kernel_size=int(1/upsample_strides[0]),
                      stride=int(1/upsample_strides[0]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            self.act
        )

        self.layer2 = [
            nn.Conv2d(mid_planes[0], mid_planes[1], 3, stride=strides[1], padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1], eps=1e-3, momentum=0.01),
            self.act
        ]
        for i in range(num_blocks[1]):
            conv = nn.Conv2d
            self.layer2 += [conv(mid_planes[1], mid_planes[1], 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_planes[1], eps=1e-3, momentum=0.01),
                            self.act]
        self.layer2 = nn.Sequential(*self.layer2)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[1],
                               int(upsample_strides[1]),
                               stride=int(upsample_strides[1]), bias=False
                               ) if upsample_strides[1] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[1],
                      kernel_size=int(1/upsample_strides[1]),
                      stride=int(1/upsample_strides[1]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], eps=1e-3, momentum=0.01),
            self.act
        )

        self.layer3 = [
            nn.Conv2d(mid_planes[1], mid_planes[2], 3, stride=strides[2], padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[2], eps=1e-3, momentum=0.01),
            self.act
        ]
        for i in range(num_blocks[2]):
            conv = nn.Conv2d
            self.layer3 += [conv(mid_planes[2], mid_planes[2], 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_planes[2], eps=1e-3, momentum=0.01),
                            self.act]

        self.layer3 = nn.Sequential(*self.layer3)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[2],
                               int(upsample_strides[2]),
                               stride=int(upsample_strides[2]), bias=False
                               ) if upsample_strides[2] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[2],
                      kernel_size=int(1/upsample_strides[2]),
                      stride=int(1/upsample_strides[2]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], eps=1e-3, momentum=0.01),
            self.act
        )

        self.p1_down_channel = nn.Sequential(nn.Conv2d(mid_planes[0], mid_planes[1], 1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.01, eps=1e-3))
        self.p2_down_channel = nn.Sequential(nn.Conv2d(mid_planes[1], mid_planes[1], 1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.01, eps=1e-3))
        self.p3_down_channel = nn.Sequential(nn.Conv2d(mid_planes[2], mid_planes[1], 1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.01, eps=1e-3))
        self.conv2_up = SeparableConvBlock(mid_planes[1], norm=True, activation=False)
        self.conv1_up = SeparableConvBlock(mid_planes[1], norm=True, activation=True)  # use act

        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = nn.MaxPool2d((2, 2))
        self.p3_downsample = nn.MaxPool2d((2, 2))

        self.conv2_down = SeparableConvBlock(mid_planes[1], norm=True, activation=True)  # use act
        self.conv3_down = SeparableConvBlock(mid_planes[1], norm=True, activation=True)  # use act
        self.relu = nn.ReLU()
        self.attention = model_cfg.attention
        if self.attention:
            self.epsilon = 1e-5
            self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.num_bev_features = sum(num_upsample_filters)
    def _forward(self, p1, p2, p3):
        p2_up = self.conv2_up(self.act(p2 + self.p2_upsample(p3)))
        p1_out = self.conv1_up(self.act(p1 + self.p1_upsample(p2_up)))
        p2_out = self.conv2_down(self.act(p2 + p2_up + self.p2_downsample(p1_out)))
        p3_out = self.conv3_down(self.act(p3 + self.p3_downsample(p2_out)))

        return p1_out, p2_out, p3_out

    def _forward_fast_attention(self, p1, p2, p3):
        p2_w1 = self.relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_up = self.conv2_up(self.act(weight[0] * p2 + weight[1] * self.p2_upsample(p3)))
        p1_w1 = self.relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)

        p1_out = self.conv1_up(self.act(weight[0] * p1 + weight[1] * self.p1_upsample(p2_up)))

        p2_w2 = self.relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.conv2_down(self.act(weight[0] * p2 + weight[1] * p2_up + weight[2] * self.p2_downsample(p1_out)))

        p3_w2 = self.relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(self.act(weight[0] * p3 + weight[1] * self.p3_downsample(p2_out)))

        return p1_out, p2_out, p3_out

    def forward(self, batch_dict):
        x = batch_dict['spatial_features']
        x = self.layer1(x)
        p1 = self.p1_down_channel(x)
        x = self.layer2(x)
        p2 = self.p2_down_channel(x)
        x = self.layer3(x)
        p3 = self.p3_down_channel(x)

        if self.attention:
            p1_out, p2_out, p3_out = self._forward_fast_attention(p1, p2, p3)
        else:
            p1_out, p2_out, p3_out = self._forward(p1, p2, p3)

        up1 = self.deconv1(p1_out)
        up2 = self.deconv2(p2_out)
        up3 = self.deconv3(p3_out)

        x = torch.cat([up1, up2, up3], dim=1)

        batch_dict['spatial_features_2d'] = x
        return batch_dict