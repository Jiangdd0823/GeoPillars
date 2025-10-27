import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from ...model_utils.transformer import Transformer
from ...model_utils.attention_utils import SE, CoordAtt, SKAttention_voxel, SE_plus1120, SE_plus


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


class AttentionVFE_1030(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
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

        # revise 两层mlp，内存偏高
        # self.mlp_pfn = nn.Sequential(
        #     MLPlayer(num_filters[0], 256, self.use_norm, last_layer=True),
        #     MLPlayer(256, num_filters[1], self.use_norm, last_layer=True)
        # )
        self.mlp_pfn = nn.Sequential(
            MLPlayer(num_filters[0], num_filters[1], self.use_norm, last_layer=True),
        )
        # self.mlp_channel_wise = nn.Sequential(
        #     MLPlayer(num_filters[1], 1, self.use_norm, last_layer=True)
        # )
        # self.mlp_query = nn.Sequential(
        #     MLPlayer(320, 64, self.use_norm, last_layer=True)
        # )
        # self.senet = SE(64)
        self.num_head = 8
        # self.self_attn = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)
        # self.cross_attn = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)
        # self.transformer_r = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)

        # 初始化query
        self.init_query = nn.Parameter(torch.randn(1, num_filters[-1]))

    def point_dist_pos(self, point_coords):
        coords_square = point_coords.pow(2).sum(dim=-1, keepdim=True)
        dist_squared = \
            coords_square + coords_square.permute(0, 2, 1) - 2 * point_coords.float() @ point_coords.permute(0, 2, 1).float()
        dist_matrix = dist_squared.sqrt()
        return dist_matrix

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

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
        # 原始vfe是10维 原始坐标+ 质心偏移+网格偏移特征 (6)
        init_features = torch.cat(features, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask
        N, M, C = init_features.shape

        # revise 1030的，因 mlp_pfn transformer_l 所占内存太高修改
        # revise 所占内存偏高 mlp_pfn 换成单层，代码跑的是双层中间层256
        # point_aware_query = self.mlp_query(init_features.reshape(N, M*C).unsqueeze(1)).permute(1, 0, 2)
        init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)

        features = self.mlp_pfn(init_features)
        features = features.permute(1, 0, 2)
        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        query = self.transformer_l(init_query, features, point_pos)
        # query = self.cross_attn(point_aware_query, features, features)[0]
        # query = self.senet(query)

        # features = self.mlp_pfn(init_features)
        # N, M, C= features.shape
        # features = features.permute(1, 0, 2)
        # point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        # # fixme 使用point-wise的分区计算，减少内存  (N, M, C) -> (N, M/4, C) x 4  transformer块分别对四个分区计算 共享参数shared
        # fixme 分区计算不降反升，原本2g 增到 6g
        # part = M // 4
        # part_transformer_out = [self.transformer_l(
        #     point_aware_query,
        #     features[:part*(i+1)],
        #     point_pos[:, :part*(i+1)]
        # ) for i in range(4)]
        # query = torch.cat(part_transformer_out, dim=0)
        # # fixme 先用max试试四个分区的
        # query = torch.max(query, dim=0)[0]
        query = query.squeeze(0)
        batch_dict['pillar_features'] = query
        return batch_dict


class AttentionVFE_1031(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
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
        # self.mlp_channel_wise = nn.Sequential(
        #     MLPlayer(num_filters[1], num_filters[1], self.use_norm, last_layer=True)
        # )
        # self.mlp_point_wise = nn.Sequential(
        #     MLPlayer(64, 64, self.use_norm, last_layer=True)
        # )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self. conv1d_point_wise= nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)
        # self.mlp_query = nn.Sequential(
        #     MLPlayer(320, 64, self.use_norm, last_layer=True)
        # )
        # self.senet_channel = SE(64, ratio=16)
        # self.senet_point = SE(32, ratio=8)
        self.senet_global = SE(64, ratio=8)
        self.channel_max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.point_max_pool = nn.AdaptiveMaxPool2d((None, 1))

        self.num_head = 8
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)
        # self.cross_attm = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)

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

    def point_dist_pos(self, point_coords, mask):
        # 确保 mask 为布尔类型
        mask = mask.bool()
        mask = mask.squeeze()
        # 过滤点坐标
        filtered_coords = []
        for i in range(point_coords.size(0)):
            masked_coords = point_coords[i][mask[i]]
            filtered_coords.append(masked_coords)

        # 计算距离矩阵
        dist_matrices = []
        for coords in filtered_coords:
            if coords.numel() == 0:
                dist_matrices.append(torch.zeros((0, 0), device=point_coords.device))
            else:
                coords_square = coords.pow(2).sum(dim=-1, keepdim=True)
                dist_squared = \
                    coords_square + coords_square.t() - 2 * coords.float() @ coords.t().float()
                # dist_matrix = dist_squared.sqrt()
                dist_matrices.append(dist_squared)

        # 恢复原始形状
        final_dist_matrices = []
        for i in range(point_coords.size(0)):
            dist_matrix = dist_matrices[i]
            if dist_matrix.numel() == 0:
                final_dist_matrix = torch.zeros((point_coords.size(1), point_coords.size(1)),
                                                device=point_coords.device)
            else:
                final_dist_matrix = torch.full((point_coords.size(1), point_coords.size(1)), float('inf'),
                                               device=point_coords.device)
                mask_indices = mask[i].nonzero(as_tuple=True)[0]
                final_dist_matrix[mask_indices[:, None], mask_indices] = dist_matrix
            final_dist_matrices.append(final_dist_matrix)

        # 将所有距离矩阵堆叠成一个张量
        final_dist_matrices = torch.stack(final_dist_matrices, dim=0)

        return final_dist_matrices

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
        # 原始vfe是10维 原始坐标+ 质心偏移+网格偏移特征 (6)
        init_features = torch.cat(features, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask
        N, M, C = init_features.shape
        kv_features = self.mlp_pfn(init_features)
        # points_dist_matrix = self.point_dist_pos(voxel_features[..., :3], mask)
        # kv_features = kv_features.transpose(2, 1) @ points_dist_matrix
        # revise point-wise channel-wise
        # # todo 同一个变量不能经过两个不同位置maxpool，反向传播时变量会被替换，报错；所以在每次maxpool前加一层mlp
        # channel_wise_feat = self.mlp_channel_wise(kv_features)
        # channel_wise_feat = self.channel_max_pool(channel_wise_feat)
        # # channel_wise_feat = self.senet_channel(channel_wise_feat)
        #
        # point_wise_feat = self.mlp_point_wise(kv_features)
        # point_wise_feat = self.point_max_pool(point_wise_feat)
        # # point_wise_feat1 = self.senet_point(point_wise_feat).permute(0, 2, 1)
        # feature_weight = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features)
        # kv_features *= feature_weight

        # revise 1101 用conv1d代替maxpool
        channel_wise_feat = self.conv1d_channel_wise(kv_features.permute(0, 2, 1)).squeeze()
        channel_wise_feat = self.norm_channel(channel_wise_feat).unsqueeze(1)
        # kv_features = kv_features * channel_wise_feat
        point_wise_feat = self.conv1d_point_wise(kv_features).squeeze()
        point_wise_feat = self.norm_point(point_wise_feat).unsqueeze(-1)
        feat_weight = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features)
        kv_features = kv_features * feat_weight
        # kv_features = kv_features * point_wise_feat

        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)
        query = self.transformer_l(init_query, kv_features.permute(1, 0, 2), point_pos)
        query = self.senet_global(query)

        # revise 1101 conv1d 双层transformer

        # # revise 1030的，因 mlp_pfn transformer_l 所占内存太高修改
        # # revise 所占内存偏高 mlp_pfn 换成单层，代码跑的是双层中间层256
        # point_aware_query = self.mlp_query(init_features.reshape(N, M*C).unsqueeze(1)).permute(1, 0, 2)
        # features = self.mlp_pfn(init_features)
        # features = features.permute(1, 0, 2)
        # point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        # query = self.transformer_l(point_aware_query, features, point_pos)
        # query = self.senet(query)

        # features = self.mlp_pfn(init_features)
        # N, M, C= features.shape
        # features = features.permute(1, 0, 2)
        # point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        # # fixme 使用point-wise的分区计算，减少内存  (N, M, C) -> (N, M/4, C) x 4  transformer块分别对四个分区计算 共享参数shared
        # fixme 分区计算不降反升，原本2g 增到 6g
        # part = M // 4
        # part_transformer_out = [self.transformer_l(
        #     point_aware_query,
        #     features[:part*(i+1)],
        #     point_pos[:, :part*(i+1)]
        # ) for i in range(4)]
        # query = torch.cat(part_transformer_out, dim=0)
        # # fixme 先用max试试四个分区的
        # query = torch.max(query, dim=0)[0]
        query = query.squeeze(0)
        batch_dict['pillar_features'] = query
        return batch_dict


class AttentionVFE_1101(VFETemplate):
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
        self.mlp_channel_wise = nn.Sequential(
            MLPlayer(num_filters[1], num_filters[1], self.use_norm, last_layer=True)
        )
        self.mlp_point_wise = nn.Sequential(
            MLPlayer(64, 64, self.use_norm, last_layer=True)
        )
        if self.with_geometry:
            self.mlp_geometry = nn.Sequential(
                MLPlayer(22, 64, self.use_norm, last_layer=True)
            )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self. conv1d_point_wise= nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)
        # self.mlp_query = nn.Sequential(
        #     MLPlayer(320, 64, self.use_norm, last_layer=True)
        # )
        self.senet_channel = SE(64, ratio=16)
        self.senet_point = SE(32, ratio=8)
        self.senet_global = SE(64, ratio=8)
        self.channel_max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.point_max_pool = nn.AdaptiveMaxPool2d((None, 1))

        self.num_head = 8
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)
        self.cross_attm = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)

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

    def point_dist_pos(self, point_coords, mask):
        # 确保 mask 为布尔类型
        mask = mask.bool()
        mask = mask.squeeze()
        # 过滤点坐标
        filtered_coords = []
        for i in range(point_coords.size(0)):
            masked_coords = point_coords[i][mask[i]]
            filtered_coords.append(masked_coords)

        # 计算距离矩阵
        dist_matrices = []
        for coords in filtered_coords:
            if coords.numel() == 0:
                dist_matrices.append(torch.zeros((0, 0), device=point_coords.device))
            else:
                coords_square = coords.pow(2).sum(dim=-1, keepdim=True)
                dist_squared = \
                    coords_square + coords_square.t() - 2 * coords.float() @ coords.t().float()
                # dist_matrix = dist_squared.sqrt()
                dist_matrices.append(dist_squared)

        # 恢复原始形状
        final_dist_matrices = []
        for i in range(point_coords.size(0)):
            dist_matrix = dist_matrices[i]
            if dist_matrix.numel() == 0:
                final_dist_matrix = torch.zeros((point_coords.size(1), point_coords.size(1)),
                                                device=point_coords.device)
            else:
                final_dist_matrix = torch.full((point_coords.size(1), point_coords.size(1)), float('inf'),
                                               device=point_coords.device)
                mask_indices = mask[i].nonzero(as_tuple=True)[0]
                final_dist_matrix[mask_indices[:, None], mask_indices] = dist_matrix
            final_dist_matrices.append(final_dist_matrix)

        # 将所有距离矩阵堆叠成一个张量
        final_dist_matrices = torch.stack(final_dist_matrices, dim=0)

        return final_dist_matrices

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

        init_features = torch.cat(features, dim=-1)
        # 原始vfe是10维 原始坐标+ 质心偏移+网格偏移特征 (6)
        # init_features = torch.cat(features, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask
        N, M, C = init_features.shape
        kv_features = self.mlp_pfn(init_features)

        # revise point-wise channel-wise
        # revise 1101 用conv1d代替maxpool
        channel_wise_feat = self.conv1d_channel_wise(kv_features.permute(0, 2, 1)).squeeze()
        channel_wise_feat = self.norm_channel(channel_wise_feat).unsqueeze(1)
        # kv_features = kv_features * channel_wise_feat
        point_wise_feat = self.conv1d_point_wise(kv_features).squeeze()
        point_wise_feat = self.norm_point(point_wise_feat).unsqueeze(-1)
        feat_weight = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features)
        kv_features = kv_features * feat_weight
        # kv_features = kv_features * point_wise_feat

        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)
        query = self.transformer_l(init_query, kv_features.permute(1, 0, 2), point_pos)

        query = self.senet_global(query)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        return batch_dict


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
        # revise 1121 先max再mlp
        if self.with_geometry:
            self.mlp_geometry = nn.Sequential(
                MLPlayer(22, 64, self.use_norm, last_layer=True)
            )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self.conv1d_point_wise = nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)

        # self.senet_global = SE(64, ratio=8)
        # revise 1109
        # revise 1110 调整weight
        # self.senet_global = SE_plus1110(64, ratio=8, weight=[1., 1.])
        # revise 1125
        self.senet_global = SE_plus(64, ratio=8, weight=[0.5, 1.5])
        # revise 1125
        # self.senet_global = SE_plus(64, ratio=8, weight=[0.5, 0.5])
        # self.senet_global = SE(64, ratio=8)
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
            # revise 1121 先max再mlp
            # geo_max_feat = torch.max(geo_features, dim=1)[0].unsqueeze(1).permute(1, 0, 2)
            # geo_max_feat = self.mlp_geometry(geo_max_feat)
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
        if self.with_geometry:
            # geo_max_feat = self.senet_global2(geo_max_feat)
            query += geo_max_feat
        else:
            # revise 1108 不用geo
            query += torch.max(kv_features, dim=1)[0].unsqueeze(0)
            # query += self.senet_global2(torch.max(kv_features, dim=1)[0].unsqueeze(0))

        # revise 1126 不加senet
        query = self.senet_global(query)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        coord = batch_dict['voxel_coords'][:, [0, 2, 3]]
        batch_dict['pillar_coords'] = coord
        return batch_dict


class AttentionVFE_1104_geo(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_geometry = self.model_cfg.WITH_GEPMETRY
        assert self.with_geometry is True

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
            MLPlayer(num_filters[0], num_filters[1], self.use_norm, last_layer=True),
        )
        self.mlp_query = nn.Sequential(
            MLPlayer(32*10, 64, self.use_norm, last_layer=True)
        )
        self.mlp_geometry = nn.Sequential(
            MLPlayer(22, 64, self.use_norm, last_layer=True)
        )
        self.senet = SE(64)
        self.num_head = 8
        # self.self_attn = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)
        # self.cross_attn = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)
        # self.transformer_r = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)

        # 初始化query
        # self.init_query = nn.Parameter(torch.randn(1, num_filters[-1]))

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

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

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

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask
        N, M, C = init_features.shape

        point_aware_query = self.mlp_query(init_features.reshape(N, M*C).unsqueeze(1)).permute(1, 0, 2)
        # init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)

        features = self.mlp_pfn(init_features)
        features = features.permute(1, 0, 2)
        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        # query = self.transformer_l(init_query, features, point_pos)
        query = self.transformer_l(point_aware_query, features, point_pos)
        # query = self.cross_attn(point_aware_query, features, features)[0]
        if self.with_geometry:
            query += geo_max_feat
        query = self.senet(query)

        query = query.squeeze(0)
        batch_dict['pillar_features'] = query
        return batch_dict


class AttentionVFE_1105_withoutattn(VFETemplate):
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

        assert self.with_geometry is True
        self.mlp_geometry = nn.Sequential(
            MLPlayer(22, 64, self.use_norm, last_layer=True)
        )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self.conv1d_point_wise = nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)

        # self.conv2d_1 = nn.Conv2d(1, 1, (32, 32), 1, 0)
        # self.norm1 = nn.BatchNorm1d(32)
        self.conv2d_2 = nn.Conv2d(1, 1, (32, 64), 1, 0)
        self.sigmoid = nn.Sigmoid()
        # self.norm2 = nn.BatchNorm1d(32)
        # self.mlp_query = nn.Sequential(
        #     MLPlayer(320, 64, self.use_norm, last_layer=True)
        # )
        self.senet_global = SE(64, ratio=8)

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

        init_features = torch.cat(features, dim=-1)
        features_extra = self.geometry_prior(features, voxel_features, points_mean, f_cluster, f_center)
        geo_features = torch.cat(features_extra, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask

        kv_features = self.mlp_pfn(init_features)
        geo_feat = self.mlp_geometry(geo_features)
        geo_feat = self.conv2d_2(geo_feat.unsqueeze(1))
        global_score = self.sigmoid(geo_feat)
        # revise point-wise channel-wise
        # revise 1101 用conv1d代替maxpool
        channel_wise_feat = self.conv1d_channel_wise(kv_features.permute(0, 2, 1)).squeeze()
        channel_wise_feat = self.norm_channel(channel_wise_feat).unsqueeze(1)
        # kv_features = kv_features * channel_wise_feat
        point_wise_feat = self.conv1d_point_wise(kv_features).squeeze()
        point_wise_feat = self.norm_point(point_wise_feat).unsqueeze(-1)
        feat = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features) * global_score.squeeze(1).expand_as(kv_features)
        feat = torch.max(feat, dim=1)[0]
        # # revise 1105 conv2d global
        # feat = self.conv2d_1(feat)
        # feat = self.norm1(feat)
        # features = torch.cat([feat, geo_feat], dim=0)

        query = self.senet_global(feat)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        return batch_dict


class AttentionVFE_1107(VFETemplate):
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
        # self.mlp_channel_wise = nn.Sequential(
        #     MLPlayer(num_filters[1], num_filters[1], use_norm=False, last_layer=True)
        # )
        # self.mlp_point_wise = nn.Sequential(
        #     MLPlayer(32, 32, use_norm=False, last_layer=True)
        # )
        if self.with_geometry:
            self.mlp_geometry = nn.Sequential(
                MLPlayer(22, 64, self.use_norm, last_layer=True)
            )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self. conv1d_point_wise= nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)
        # self.mlp_query = nn.Sequential(
        #     MLPlayer(320, 64, self.use_norm, last_layer=True)
        # )
        # self.senet_channel = SE(64, ratio=16)
        # self.senet_point = SE(32, ratio=8)
        self.senet_global = SE(64, ratio=8)
        self.num_head = 8
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)
        # self.cross_attn = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=self.num_head, dropout=0.1)

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
        N, M, C = init_features.shape
        kv_features = self.mlp_pfn(init_features)

        # revise point-wise channel-wise
        # revise 1101 用conv1d代替maxpool
        channel_wise_feat = self.conv1d_channel_wise(kv_features.permute(0, 2, 1))
        channel_wise_feat = channel_wise_feat.permute(0, 2, 1)
        # revise 1106 conv1d后加入mlp和relu，不用norm
        # revise 1107 改1104_attentionvfe_initquer_conv1d_pointchannel_geometry_senet
        # channel_wise_feat = channel_wise_feat.squeeze()
        # channel_wise_feat = self.mlp_channel_wise(channel_wise_feat).unsqueeze(1)
        # channel_wise_feat = self.norm_channel(channel_wise_feat).unsqueeze(1)
        # kv_features = kv_features * channel_wise_feat
        point_wise_feat = self.conv1d_point_wise(kv_features)
        # point_wise_feat = point_wise_feat.squeeze()
        # point_wise_feat = self.mlp_point_wise(point_wise_feat).unsqueeze(-1)
        # point_wise_feat = self.norm_point(point_wise_feat).unsqueeze(-1)
        feat_weight = channel_wise_feat.expand_as(kv_features) * point_wise_feat.expand_as(kv_features)

        kv_features = kv_features * feat_weight


        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        init_query = self.init_query.expand(N, 1, self.init_query.shape[-1]).permute(1, 0, 2)
        # init_query = torch.max(kv_features, dim=1)[0].unsqueeze(0)

        query = self.transformer_l(init_query, kv_features.permute(1, 0, 2), point_pos)
        # query = self.transformer_l(init_query, feat_weight.permute(1, 0, 2), point_pos)
        # revise 1105 cross-attn
        # query = self.cross_attn(init_query, kv_features.permute(1, 0, 2), kv_features.permute(1, 0, 2))[0]
        if self.with_geometry:
            query += geo_max_feat

        # query = self.senet_global(query)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        return batch_dict


# revise 1122
class AttentionVFE_1103plus(VFETemplate):
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
        # revise 1121 先max再mlp
        if self.with_geometry:
            self.mlp_geometry = nn.Sequential(
                MLPlayer(22, 64, self.use_norm, last_layer=True)
            )
        self.conv1d_channel_wise = nn.Conv1d(64, 64, 32, 1)
        self.conv1d_point_wise = nn.Conv1d(32, 32, 64, 1)
        self.norm_channel = nn.BatchNorm1d(64)
        self.norm_point = nn.BatchNorm1d(32)

        # self.senet_global = SE(64, ratio=8)
        # revise 1109
        # revise 1110 调整weight
        # self.senet_global = SE_plus1110(64, ratio=8, weight=[1., 1.])
        # self.senet_global = SE_plus1109(64, ratio=8, weight=[0.5, 1.])
        self.senet_global = SE(64, ratio=8)
        # self.senet_global2 = SE(64, ratio=8)
        # revise 1111 改动transformer层数
        self.num_head = 8
        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=self.num_head, num_decoder_layers=1, dim_feedforward=512)

        # 初始化query
        self.init_query = nn.Parameter(torch.randn(1, 1, num_filters[-1]))
        # revise 1122
        self.init_position = nn.Parameter(torch.randn(1, 1, 3))

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
            # revise 1121 先max再mlp
            # geo_max_feat = torch.max(geo_features, dim=1)[0].unsqueeze(1).permute(1, 0, 2)
            # geo_max_feat = self.mlp_geometry(geo_max_feat)
        else:
            init_features = torch.cat(features, dim=-1)
        # 原始vfe是10维 原始坐标+ 质心偏移+网格偏移特征 (6)
        # init_features = torch.cat(features, dim=-1)

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
        # revise 1122
        qkv_features = torch.cat([init_query, kv_features.permute(1, 0, 2)], dim=0)
        init_position = self.init_position.expand(N, 1, 3).permute(1, 0, 2)
        point_pos = torch.cat([init_position.unsqueeze(0), point_pos], dim=1)
        query = self.transformer_l(qkv_features, qkv_features, point_pos)
        query = query[0].unsqueeze(0)

        # query = self.transformer_l(init_query, kv_features.permute(1, 0, 2), point_pos)
        if self.with_geometry:
            query += geo_max_feat
        else:
            # revise 1108 不用geo
            query += torch.max(kv_features, dim=1)[0].unsqueeze(0)

        query = self.senet_global(query)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        coord = batch_dict['voxel_coords'][:, [0, 2, 3]]
        batch_dict['pillar_coords'] = coord
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

        # self.senet_global = SE_plus1110(64, ratio=8, weight=[1., 1.])
        # self.senet_global = SE_plus(num_filters[1], ratio=self.attn_cfg.SE_RATIO, weight=[0.5, 1.5])
        self.senet_global = SE(num_filters[1], ratio=self.attn_cfg.SE_RATIO)
        # revise 1120
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

        query = self.senet_global(query)
        query = query.squeeze()
        batch_dict['pillar_features'] = query
        coord = batch_dict['voxel_coords'][:, [0, 2, 3]]
        batch_dict['pillar_coords'] = coord
        return batch_dict


class AttentionVFE(VFETemplate):
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


class PVTransformerVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.mlp = nn.Sequential(
            nn.Linear(num_filters[0], num_filters[1]),
            nn.ReLU(),
            nn.LayerNorm(num_filters[1])
        )

        self.self_attn = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=8, dropout=0.1)

        # self.cross_attn_l = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=8, dropout=0.1)
        # self.cross_attn_r = nn.MultiheadAttention(embed_dim=num_filters[-1], num_heads=8, dropout=0.1)

        self.transformer_l = Transformer(d_model=num_filters[-1], nhead=8, num_decoder_layers=1, dim_feedforward=512)
        self.transformer_r = Transformer(d_model=num_filters[-1], nhead=8, num_decoder_layers=1, dim_feedforward=512)

        # 初始化query
        self.init_query = nn.Parameter(torch.randn(1, num_filters[-1]))

        self.part = 10000

    def part_transformer(self, tgt, src, pos, func):
        # transformer performs randomly when batch size is too large
        num_parts = src.shape[1] // self.part
        part_transformer_out = [func(tgt[:, num_part * self.part:(num_part + 1) * self.part, :],
                                     src[:, num_part * self.part:(num_part + 1) * self.part, :],
                                     pos[:, :, num_part * self.part:(num_part + 1) * self.part, :])
                                for num_part in range(num_parts+1)]
        out = torch.cat(part_transformer_out, dim=1)
        return out

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

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
        # 原始vfe是10维 原始坐标+ 质心偏移+网格偏移特征 (6)
        init_features = torch.cat(features, dim=-1)

        voxel_count = init_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        init_features *= mask

        features = self.mlp(init_features)

        max_feat = torch.max(features, dim=1)[0].unsqueeze(1).permute(1, 0, 2)
        features = features.permute(1, 0, 2)
        # fixme 这里只是用了多头注意力，论文是transformer块
        latent_features = self.self_attn(features, features, features)[0]
        init_query = self.init_query.unsqueeze(0).expand(init_features.shape[0], 1, self.init_query.shape[-1]).permute(1, 0, 2)
        # latent_query = self.cross_attn_l(init_query, latent_features, latent_features)[0]
        # todo
        point_pos = voxel_features[..., :3].permute(1, 0, 2).unsqueeze(0)
        if latent_features.shape[1] > self.part:
            latent_query = self.part_transformer(init_query, latent_features, point_pos, self.transformer_l)
        else:
            latent_query = self.transformer_l(init_query, latent_features, point_pos)
        latent_query = latent_query + max_feat

        # residual_query = self.cross_attn_r(latent_query, latent_features, latent_features)[0]
        if latent_features.shape[1] > self.part:
            residual_query = self.part_transformer(latent_query, latent_features, point_pos, self.transformer_r)
        else:
            residual_query = self.transformer_r(latent_query, latent_features, point_pos)

        # residual_query = self.transformer_r(latent_query, latent_features, point_pos)
        features = residual_query.squeeze(0)
        batch_dict['pillar_features'] = features
        return batch_dict
