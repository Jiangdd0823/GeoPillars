from functools import partial

import torch.nn as nn
import torch
from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils_pvtssd


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    
    
class PillarBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
        )
        
        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }


    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict


class PillarBackBone8x_(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense
        # revise 32维改成64维
        self.conv1 = spconv.SparseSequential(
            block(32,32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,          # revise
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

        self.offset_range = self.model_cfg.QUERY_RANGE
        self.query_layer = self.model_cfg.QUERY_LAYER
        self.cat_coord = self.model_cfg.CAT_COORDS
        self.with_range = self.model_cfg.WITH_RANGE
        mlp = [[128], [256], [256], [256]]
        if self.model_cfg.ALONE_QUERY:
            assert isinstance(self.query_layer, (int, list))
            if isinstance(self.query_layer, int):
                self.offset_conv = self.make_fc_layers(mlp[int(self.query_layer)-1], self.backbone_channels[f'x_conv{self.query_layer}'], 2, linear=True)
            if isinstance(self.query_layer, list):
                self.offset_conv = nn.ModuleList()
                for i in self.query_layer:
                    offset_conv = self.make_fc_layers(mlp[self.query_layer[i-1]-1], self.backbone_channels[f'x_conv{self.query_layer[i-1]}'], 2, linear=True)
                    self.offset_conv.append(offset_conv)
        a =0

    def make_fc_layers(self, fc_cfg, input_channels, output_channels=None, linear=True, norm_fn=None):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False) if linear else nn.Conv1d(c_in, fc_cfg[k], kernel_size=1,
                                                                                bias=False),
                nn.BatchNorm1d(fc_cfg[k]) if norm_fn is None else norm_fn(fc_cfg[k]),
                nn.ReLU()
            ])
            c_in = fc_cfg[k]
        if output_channels is not None:
            fc_layers.append(
                nn.Linear(c_in, output_channels) if linear else nn.Conv1d(c_in, output_channels, kernel_size=1),
            )
        return nn.Sequential(*fc_layers)

    def smooth_l1_loss(self, input, target, beta: float = 1.0 / 9.0):
        diff = target - input
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def vote_layer(self, offset, indices, i):
        # 偏移量取整
        offset_ceil = torch.ceil(offset)        # fixme 要加入取整正则化
        # todo 取整正则化
        loss = 0
        # loss = self.smooth_l1_loss(input=offset, target=offset_ceil).sum(dim=-1).mean()

        if self.with_range:
            limited_offset = []
            for axis in range(len(self.offset_range[i])):
                limited_offset.append(offset_ceil[..., axis].clamp(
                    min=-self.offset_range[i][axis],
                    max=self.offset_range[i][axis]))       # clamp(min,max)限制范围
            limited_offset = torch.stack(limited_offset, dim=-1)
            votes = indices[:, 1:] + limited_offset          # 特征图体素中心（原始坐标系下）＋偏差
        else:
            votes = indices[:, 1:] + offset_ceil
        votes = torch.cat([indices[:, 0:1], votes], dim=-1)

        return votes, loss

    def cat_indices(self, vote_indices, origin_indices, features):
        all_indices = torch.cat([origin_indices, vote_indices], dim=0)
        unique_indices, inverse_indices, counts = torch.unique(all_indices, return_inverse=True, return_counts=True, dim=0)
        # 初始化新的 features
        num_unique_indices = unique_indices.shape[0]
        new_features = torch.zeros(num_unique_indices, features.shape[1], dtype=features.dtype,
                                   device=features.device)

        # 构建新的 features
        new_features.scatter_add_(0, inverse_indices[:origin_indices.shape[0]].unsqueeze(-1).expand(-1, features.shape[1]),
                                  features)
        new_features.scatter_add_(0, inverse_indices[origin_indices.shape[0]:].unsqueeze(-1).expand(-1, features.shape[1]),
                                  features)

        return unique_indices, new_features

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        voxel_indices = []
        offset_ceil_loss = 0
        x_conv1 = self.conv1(input_sp_tensor)
        if self.query_layer == 1:
            offset1 = self.offset_conv(x_conv1.features) * 16
            vote_indices1, loss = self.vote_layer(offset1, x_conv1.indices, 2)
            offset_ceil_loss += loss
            if self.cat_coord:
                indices1, features1 = self.cat_indices(vote_indices1, x_conv1.indices, x_conv1.features)
                x_conv1.indices = indices1.to(torch.int32)
                ori_feature = x_conv1.features
                x_conv1 = replace_feature(x_conv1, features1)
                # voxel_indices = vote_indices1
                voxel_indices.append(vote_indices1)
            else:
                x_conv1.indices = vote_indices1.to(torch.int32)
                # voxel_indices = vote_indices1
                voxel_indices.append(vote_indices1)
                ori_feature = x_conv1.features

        if isinstance(self.query_layer, list):
            if 1 in self.query_layer:
                offset1 = self.offset_conv[0](x_conv1.features) * 16
                vote_indices1, loss = self.vote_layer(offset1, x_conv1.indices, 2)
                offset_ceil_loss += loss
                if self.cat_coord:
                    indices1, features1 = self.cat_indices(vote_indices1, x_conv1.indices, x_conv1.features)
                    x_conv1.indices = indices1.to(torch.int32)
                    x_conv1 = replace_feature(x_conv1, features1)
                    voxel_indices.append(vote_indices1)
                else:
                    x_conv1.indices = vote_indices1.to(torch.int32)
                    voxel_indices.append(vote_indices1)

        x_conv2 = self.conv2(x_conv1)
        if self.query_layer == 2:
            offset2 = self.offset_conv(x_conv2.features) * 8
            vote_indices2, loss = self.vote_layer(offset2, x_conv2.indices, 2)
            offset_ceil_loss += loss
            if self.cat_coord:
                indices2, features2 = self.cat_indices(vote_indices2, x_conv2.indices, x_conv2.features)
                x_conv2.indices = indices2.to(torch.int32)
                ori_feature = x_conv2.features
                x_conv2 = replace_feature(x_conv2, features2)
                # voxel_indices = vote_indices2
                voxel_indices.append(vote_indices2)
            else:
                x_conv2.indices = vote_indices2.to(torch.int32)
                # voxel_indices = vote_indices2
                voxel_indices.append(vote_indices2)
                ori_feature = x_conv2.features

        if isinstance(self.query_layer, list):
            if 2 in self.query_layer:
                offset2 = self.offset_conv[1](x_conv2.features) * 8
                vote_indices2, loss = self.vote_layer(offset2, x_conv2.indices, 0)
                offset_ceil_loss += loss
                if self.cat_coord:
                    indices2, features2 = self.cat_indices(vote_indices2, x_conv2.indices, x_conv2.features)
                    x_conv2.indices = indices2.to(torch.int32)
                    x_conv2 = replace_feature(x_conv2, features2)
                    voxel_indices.append(vote_indices2)
                else:
                    x_conv2.indices = vote_indices2.to(torch.int32)
                    voxel_indices.append(vote_indices2)

        x_conv3 = self.conv3(x_conv2)
        if self.query_layer == 3:
            offset3 = self.offset_conv(x_conv3.features) * 4
            vote_indices3, loss = self.vote_layer(offset3, x_conv3.indices, 2)
            offset_ceil_loss += loss
            if self.cat_coord:
                indices3, features3 = self.cat_indices(vote_indices3, x_conv3.indices, x_conv3.features)
                x_conv3.indices = indices3.to(torch.int32)
                ori_feature = x_conv3.features
                x_conv3 = replace_feature(x_conv3, features3)
                # voxel_indices = vote_indices3
                voxel_indices.append(vote_indices3)

            else:
                x_conv3.indices = vote_indices3.to(torch.int32)
                # voxel_indices = vote_indices3
                voxel_indices.append(vote_indices3)
                ori_feature = x_conv3.features

        if isinstance(self.query_layer, list):
            if 3 in self.query_layer:
                offset3 = self.offset_conv[2](x_conv3.features) * 8
                vote_indices3, loss = self.vote_layer(offset3, x_conv3.indices, 0)
                offset_ceil_loss += loss
                if self.cat_coord:
                    indices3, features3 = self.cat_indices(vote_indices3, x_conv3.indices, x_conv3.features)
                    x_conv3.indices = indices3.to(torch.int32)
                    x_conv3 = replace_feature(x_conv3, features3)
                    voxel_indices.append(vote_indices3)
                else:
                    x_conv3.indices = vote_indices3.to(torch.int32)
                    voxel_indices.append(vote_indices3)

        x_conv4 = self.conv4(x_conv3)
        # if self.query_layer == 4:
        #     offset4 = self.offset_conv(x_conv4.features) * 2
        #     vote_indices4, loss = self.vote_layer(offset4, x_conv4.indices, 2)
        #     offset_ceil_loss += loss
        #     if self.cat_coord:
        #         indices4, features4 = self.cat_indices(vote_indices4, x_conv4.indices, x_conv4.features)
        #         x_conv4.indices = indices4.to(torch.int32)
        #         ori_feature = x_conv4.features
        #         x_conv4 = replace_feature(x_conv4, features4)
        #         voxel_indices = vote_indices4
        #     else:
        #         x_conv4.indices = vote_indices4.to(torch.int32)
        #         voxel_indices = vote_indices4
        #         ori_feature = x_conv4.features
        #
        # if isinstance(self.query_layer, list):
        #     if 4 in self.query_layer:
        #         offset4 = self.offset_conv[3](x_conv4.features) * 2
        #         vote_indices4, loss = self.vote_layer(offset4, x_conv4.indices, 0)
        #         offset_ceil_loss += loss
        #         if self.cat_coord:
        #             indices4, features4 = self.cat_indices(vote_indices4, x_conv4.indices, x_conv4.features)
        #             x_conv4.indices = indices4.to(torch.int32)
        #             x_conv4 = replace_feature(x_conv4, features4)
        #         else:
        #             x_conv4.indices = vote_indices4.to(torch.int32)

        x_conv4_sparse = x_conv4
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv4_sparse': x_conv4_sparse,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        # fixme 传参到AnchorHeadSingle_Aux类，但加上正则化后效果不好，已舍弃
        # fixme 在cat坐标后，不能反向传播
        # if isinstance(self.query_layer, int):
        batch_dict.update({
            'vote': voxel_indices,
            'query_layer': self.query_layer,
            # 'features': ori_feature,
            'offset_ceil_loss': offset_ceil_loss,
        })

        return batch_dict


class PillarRes18BackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        # batch_dict.update({
        #     'encoded_spconv_tensor': out,
        #     'encoded_spconv_tensor_stride': 8
        # })
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return batch_dict


class PillarRes18BackBone8x_(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv4_dense = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4_dense)

        # batch_dict.update({
        #     'encoded_spconv_tensor': out,
        #     'encoded_spconv_tensor_stride': 8
        # })
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,     # revise 相比上述源码，这里传进去是稀疏张量
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict