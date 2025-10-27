import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from ...model_utils.transformer import Transformer
from ...model_utils.attention_utils import SE, CoordAtt, SKAttention_voxel, SE_plus1120, SplitAttention, \
    SplitAttention1123, SE_plus


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class MLPlayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False,
                 use_relu=True):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.use_relu = use_relu
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True

        if self.use_relu:
            x = F.relu(x)

        return x


class AttentionVFE_1103(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_geometry = self.model_cfg.WITH_GEPMETRY
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.mlp_pfn = nn.Sequential(
            MLPlayer(num_filters[0], num_filters[1], self.use_norm, last_layer=True, use_relu=True),
        )

        if self.with_geometry:
            self.mlp_geometry = nn.Sequential(
                MLPlayer(22, 64, self.use_norm, last_layer=True)
            )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self. conv1d_point_wise= nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)

        # self.senet_global = SE(64, ratio=8)
        # revise 1109
        # revise 1110 调整weight
        # self.senet_global = SE_plus1110(64, ratio=8, weight=[1., 1.])
        # self.senet_global = SE_plus1109(64, ratio=8, weight=[0.5, 1.])
        # self.senet_global = SE(64, ratio=8)
        # revise 1121
        # self.senet_global = SplitAttention(128, 64, split_size=16)
        self.senet_global = SplitAttention1123(64, ratio=8)
        # self.senet_global2 = SE(64, ratio=8)
        # revise 1111 改动transformer层数
        self.num_head = 8
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)

        # 初始化query
        self.init_query = nn.Parameter(torch.randn(1, 1, num_filters[-1]))

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def geometry_prior(self, features, voxel_features, points_mean, f_cluster, f_center):
        # 质心编码 (3)
        f_point_mean = points_mean.repeat(1, voxel_features.shape[1], 1)  # 对每个点都进行重复来拼接 (N, 32, 1)
        features.extend([f_point_mean])

        # 几何信息高阶编码 (3)
        f_xyz_second = voxel_features[:, :, :3] ** 2  # 二阶位置特征 (N, 32, 3)
        features.extend([f_xyz_second])

        # 几何偏移范数 + 质心偏移范数 (2)
        f_cluster_norm = torch.norm(f_cluster, 1, 2, keepdim=True)  # 质心偏移范数
        f_center_norm = torch.norm(f_center, 1, 2, keepdim=True)  # 几何偏移范数
        features.extend([f_cluster_norm, f_center_norm])

        # 水平方向 + 垂直方向 (2)
        f_depth = torch.norm(voxel_features[:, :, :2], 2, 2, keepdim=True)  # xy的二范数就是深度信息
        # 对于角度的计算需要避免分母为0的情况，会得到nan与inf的结果
        f_horizontal_dir = \
            torch.arctan(voxel_features[:, :, 1:2] / (voxel_features[:, :, :1] + 1e-5)).clamp(min=-1.57,
                                                                                              max=1.57)  # argtan(y/x)
        f_vertical_dir = \
            torch.arctan(voxel_features[:, :, 2:3] / (f_depth + 1e-5)).clamp(min=-1.57, max=1.57)  # argtan(z/d)
        features.extend([f_horizontal_dir, f_vertical_dir])

        # 距离信息 + 深度信息 (2)
        f_distance = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)  # xyz的二范数就是距离信息
        f_depth = torch.norm(voxel_features[:, :, :2], 2, 2, keepdim=True)  # xy的二范数就是深度信息
        features.extend([f_distance, f_depth])
        return features

    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        if self.with_geometry:
            init_features = torch.cat(features, dim=-1)
            features_extra = self.geometry_prior(features, voxel_features, points_mean, f_cluster, f_center)
            geo_features = torch.cat(features_extra, dim=-1)
            geo_feat = self.mlp_geometry(geo_features)
            geo_max_feat = torch.max(geo_feat, dim=1)[0].unsqueeze(1).permute(1, 0, 2)
        else:
            init_features = torch.cat(features, dim=-1)
        # 原始vfe是10维 原始坐标+ 质心偏移+网格偏移特征 (6)
        # init_features = torch.cat(features, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask
        # revise 1107 将mask也加到geo_features上  1108前两个也加上了  后两个重做1107的
        # if self.with_geometry:
        #     geo_features *= mask
        #     geo_feat = self.mlp_geometry(geo_features)
        #     geo_max_feat = torch.max(geo_feat, dim=1)[0].unsqueeze(1).permute(1, 0, 2)

        N, M, C = init_features.shape
        kv_features = self.mlp_pfn(init_features)

        # revise point-wise channel-wise
        # revise 1101 用conv1d代替maxpool
        channel_wise_feat = self.conv1d_channel_wise(kv_features.permute(0, 2, 1)).squeeze()
        # revise 1107 conv后加入senet
        # channel_wise_feat = self.senet_channel(channel_wise_feat)
        # revise 1106 conv1d后加入mlp和relu，不用norm
        # channel_wise_feat = self.mlp_channel_wise(channel_wise_feat).unsqueeze(1)
        channel_wise_feat = self.norm_channel(channel_wise_feat).unsqueeze(1)
        # kv_features = kv_features * channel_wise_feat
        point_wise_feat = self.conv1d_point_wise(kv_features).squeeze()
        # point_wise_feat = self.mlp_point_wise(point_wise_feat)
        # point_wise_feat = self.mlp_point_wise(point_wise_feat).unsqueeze(-1)
        point_wise_feat = self.norm_point(point_wise_feat).unsqueeze(-1)
        feat_weight = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features)

        kv_features = kv_features * feat_weight
        # revise 1108  kv_features后senet
        # kv_features = self.senet_global(kv_features)

        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)
        # init_query = torch.max(kv_features, dim=1)[0].unsqueeze(0)
        query = self.transformer_l(init_query, kv_features.permute(1, 0, 2), point_pos)
        # query = self.transformer_l(init_query, feat_weight.permute(1, 0, 2), point_pos)
        # revise 1105 cross-attn
        # query = self.cross_attn(init_query, kv_features.permute(1, 0, 2), kv_features.permute(1, 0, 2))[0]
        # revise 1109 先senet后加
        # query = self.senet_global(query)
        # revise 1121 不想加，cat在一起
        # if self.with_geometry:
        #     query += geo_max_feat
        # else:
        #     # revise 1108 不用geo
        #     query += torch.max(kv_features, dim=1)[0].unsqueeze(0)

        # query = self.senet_global(query)
        # revise 1121 将两个cat在一起的新注意力机制
        query = self.senet_global(query, geo_max_feat)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        batch_dict['pillar_coords'] = batch_dict['voxel_coords'][:, [0, 2, 3]]
        return batch_dict


class AttentionVFE_final(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_geometry = self.model_cfg.WITH_GEPMETRY
        self.num_point_per_voxel = self.model_cfg.MAX_POINTS_PER_VOXEL
        self.attn_cfg = self.model_cfg.ATTENTION

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.mlp_pfn = nn.Sequential(
            MLPlayer(num_filters[0], num_filters[1], self.use_norm, last_layer=True, use_relu=True),
        )

        if self.with_geometry:
            self.mlp_geometry = nn.Sequential(
                MLPlayer(num_filters[0]+12, num_filters[1], self.use_norm, last_layer=True)
            )
        self.conv1d_channel_wise = nn.Conv1d(num_filters[1], num_filters[1], self.num_point_per_voxel, 1)
        self. conv1d_point_wise = nn.Conv1d(self.num_point_per_voxel, self.num_point_per_voxel, num_filters[1], 1)
        self.norm_channel = nn.BatchNorm1d(num_filters[1])
        self.norm_point = nn.BatchNorm1d(self.num_point_per_voxel)

        if self.model_cfg.WITH_SENET:
            # self.senet_global = SE_plus1110(64, ratio=8, weight=[1., 1.])
            self.senet_global = SE_plus(num_filters[1], ratio=self.attn_cfg.SE_RATIO, weight=[0.5, 1.5])
            # self.senet_global = SE(num_filters[1], ratio=self.attn_cfg.SE_RATIO)
            # self.senet_global = SE_plus1120(num_filters[1], ratio=self.attn_cfg.SE_RATIO)

        self.transformer_l = Transformer(d_model=num_filters[-1],
                                         nhead=self.attn_cfg.HEAD,
                                         num_decoder_layers=self.attn_cfg.NUM_LAYER,
                                         dim_feedforward=self.attn_cfg.DIM_FEEDFORWARD
                                         )
        self.init_query = nn.Parameter(torch.randn(1, 1, num_filters[-1]))

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def geometry_prior(self, features, voxel_features, points_mean, f_cluster, f_center):
        # 质心编码 (3)
        f_point_mean = points_mean.repeat(1, voxel_features.shape[1], 1)  # 对每个点都进行重复来拼接 (N, 32, 1)
        features.extend([f_point_mean])

        # 几何信息高阶编码 (3)
        f_xyz_second = voxel_features[:, :, :3] ** 2  # 二阶位置特征 (N, 32, 3)
        features.extend([f_xyz_second])

        # 几何偏移范数 + 质心偏移范数 (2)
        f_cluster_norm = torch.norm(f_cluster, 1, 2, keepdim=True)  # 质心偏移范数
        f_center_norm = torch.norm(f_center, 1, 2, keepdim=True)  # 几何偏移范数
        features.extend([f_cluster_norm, f_center_norm])

        # 水平方向 + 垂直方向 (2)
        f_depth = torch.norm(voxel_features[:, :, :2], 2, 2, keepdim=True)  # xy的二范数就是深度信息
        # 对于角度的计算需要避免分母为0的情况，会得到nan与inf的结果
        f_horizontal_dir = \
            torch.arctan(voxel_features[:, :, 1:2] / (voxel_features[:, :, :1] + 1e-5)).clamp(min=-1.57,
                                                                                              max=1.57)  # argtan(y/x)
        f_vertical_dir = \
            torch.arctan(voxel_features[:, :, 2:3] / (f_depth + 1e-5)).clamp(min=-1.57, max=1.57)  # argtan(z/d)
        features.extend([f_horizontal_dir, f_vertical_dir])

        # 距离信息 + 深度信息 (2)
        f_distance = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)  # xyz的二范数就是距离信息
        f_depth = torch.norm(voxel_features[:, :, :2], 2, 2, keepdim=True)  # xy的二范数就是深度信息
        features.extend([f_distance, f_depth])
        return features

    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        assert self.num_point_per_voxel == voxel_features.shape[1]

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        if self.with_geometry:
            init_features = torch.cat(features, dim=-1)
            features_extra = self.geometry_prior(features, voxel_features, points_mean, f_cluster, f_center)
            geo_features = torch.cat(features_extra, dim=-1)
            geo_feat = self.mlp_geometry(geo_features)
            geo_max_feat = torch.max(geo_feat, dim=1)[0].unsqueeze(1).permute(1, 0, 2)
        else:
            init_features = torch.cat(features, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask

        N, M, C = init_features.shape
        kv_features = self.mlp_pfn(init_features)

        channel_wise_feat = self.conv1d_channel_wise(kv_features.permute(0, 2, 1)).squeeze()
        channel_wise_feat = self.norm_channel(channel_wise_feat).unsqueeze(1)
        point_wise_feat = self.conv1d_point_wise(kv_features).squeeze()
        point_wise_feat = self.norm_point(point_wise_feat).unsqueeze(-1)
        feat_weight = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features)

        kv_features = kv_features * feat_weight

        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)
        query = self.transformer_l(init_query, kv_features.permute(1, 0, 2), point_pos)

        if self.with_geometry:
            # geo_max_feat = self.senet_global2(geo_max_feat)
            query += geo_max_feat
        else:
            query += torch.max(kv_features, dim=1)[0].unsqueeze(0)
            # query += self.senet_global2(torch.max(kv_features, dim=1)[0].unsqueeze(0))

        if self.model_cfg.output_variance:
            var_dict = {}
            # variances_per_voxel = torch.var(query, dim=2, unbiased=True)
            variances_per_voxel = torch.mean(query, dim=2)
            variances_list = variances_per_voxel.tolist()
            # var_dict[f'transformer_variances'] = variances_list[0]
            var_dict[f'transformer_means'] = variances_list[0]

        if self.model_cfg.WITH_SENET:
            query = self.senet_global(query)

        if self.model_cfg.get('output_variance', True):
            from pathlib import Path
            frame_id = batch_dict['frame_id'][0]
            # variances_per_voxel_se = torch.var(query, dim=2, unbiased=True)
            # variances_per_voxel_se = torch.mean(query, dim=2)
            # variances_se_list = variances_per_voxel_se.tolist()
            # # var_dict[f'senet_variances'] = variances_se_list[0]
            # var_dict[f'senet_means'] = variances_se_list[0]
            # # file_path = f'/home/jiangshaocong/mmlab/OpenPCDet/output_image/variance/{frame_id}.json'
            # file_path = f'/home/jiangshaocong/mmlab/OpenPCDet/output_image/mean/{frame_id}.json'
            # path = Path(file_path)
            # path.parent.mkdir(parents=True, exist_ok=True)
            # with open(file_path, 'w', encoding='utf-8') as fp:
            #     json.dump(var_dict, fp, ensure_ascii=False, indent=4)

            gt_file_path = f'/home/jiangshaocong/mmlab/OpenPCDet/output_image/pvtransformer/gt_box/{frame_id}.json'
            gt_path = Path(gt_file_path)
            gt_dict = {}
            gt_dict['gt_boxes'] = batch_dict['gt_boxes'].tolist()
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gt_path, 'w', encoding='utf-8') as fp:
                json.dump(gt_dict, fp, ensure_ascii=False, indent=4)

        query = query.squeeze()
        batch_dict['pillar_features'] = query
        batch_dict['pillar_coords'] = batch_dict['voxel_coords'][:, [0, 2, 3]]
        return batch_dict


class AttentionVFE_backup(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg)
        self.custom_vfe = AttentionVFE_final(model_cfg,
                                           num_point_features,
                                           voxel_size,
                                           point_cloud_range)
        self.num_filters = self.model_cfg.NUM_FILTERS

    def get_output_feature_dim(self):
        return self.num_filters

    def forward(self, batch_dict):
        batch_dict = self.custom_vfe(batch_dict)
        return batch_dict
