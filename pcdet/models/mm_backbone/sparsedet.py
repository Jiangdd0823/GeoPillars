import torch
from torch import nn
from torch.autograd import Function
from ...utils import common_utils_pvtssd as common_utils
from ...ops.pointnet2.pvtssd.pointnet2_stack.voxel_query_utils import voxel_knn_query
from torch.nn import functional as F
from ...utils.spconv_utils import spconv


def make_fc_layers(fc_cfg, input_channels, output_channels=None, linear=True, norm_fn=None):
    fc_layers = []
    c_in = input_channels
    for k in range(0, fc_cfg.__len__()):
        fc_layers.extend([
            nn.Linear(c_in, fc_cfg[k], bias=False) if linear else nn.Conv1d(c_in, fc_cfg[k], kernel_size=1, bias=False),
            nn.BatchNorm1d(fc_cfg[k]) if norm_fn is None else norm_fn(fc_cfg[k]),
            nn.ReLU()
        ])
        c_in = fc_cfg[k]
    if output_channels is not None:
        fc_layers.append(
            nn.Linear(c_in, output_channels) if linear else nn.Conv1d(c_in, output_channels, kernel_size=1),
        )
    return nn.Sequential(*fc_layers)


class LEMA(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_class):
        super().__init__()

        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range  = point_cloud_range
        self.num_class = num_class

        self.heat_conv = make_fc_layers(
        self.model_cfg.HEAT_CONV.MLP,
        self.model_cfg.HEAT_CONV.INPUT_DIM, num_class, linear=True)   # fixme

        # self.heat_conv = make_fc_layers(
        # self.model_cfg.HEAT_CONV.MLP,
        # self.model_cfg.HEAT_CONV.INPUT_DIM, num_class, linear=False)

        self.knn_feat_mlp = nn.ModuleList()
        for src in self.model_cfg.KNN_QUERY:
            k_feat_mlp = nn.Sequential(nn.Linear(self.model_cfg.KNN_QUERY[src].NSAMPLE, 1))
            self.knn_feat_mlp.append(k_feat_mlp)

        self.k_weight_mlp = make_fc_layers(
            self.model_cfg.WEIGHT_MLP.MLP,
            self.model_cfg.WEIGHT_MLP.INPUT_DIM, 3, linear=True
        )

        self.key_fuse_conv = make_fc_layers(
            self.model_cfg.FUSE_CONV.MLP,
            self.model_cfg.FUSE_CONV.INPUT_DIM,
            self.model_cfg.FUSE_CONV.OUT_DIM,
            linear=True
        )

    def voxel_knn_query_wrapper(self, key_world_indices, key_1440_indices,
                                src_feat, src_coords, spatial_shape, strides, bs_idx, model_cfg):
        """
        src_tensor: sparse tensor 没有z轴，坐标形式(x, y)
        key_1440_indices: (x, y)   original voxel 1440 coords
        key_world_indices： (x, y)  world coords
        """
        query_range = model_cfg.QUERY_RANGE
        radius = model_cfg.RADIUS
        nsample = model_cfg.NSAMPLE
        voxel_size = torch.tensor(self.voxel_size, device=src_coords.device).float()
        pc_range = torch.tensor(self.point_cloud_range, device=src_coords.device).float()
        # 特征图体素坐标转换为世界坐标
        src_world_indices_2d = (src_coords[:, 1:3] + 0.5) * strides * voxel_size[:2] + pc_range[:2]
        src_world_xyz = torch.cat([
            src_world_indices_2d,
            src_world_indices_2d.new_zeros((src_world_indices_2d.shape[0], 1))
        ], dim=-1)  # xyz, z=0
        # generate_voxels2pinds这里如果加bs_idx，就必须bs_idx=0
        src_coords_bs0 = torch.cat([src_coords.new_zeros(src_coords.shape[0]).unsqueeze(1), src_coords[:, [1,2]]], dim=-1)
        v2p_ind_tensor = common_utils.generate_voxels2pinds(src_coords_bs0.long(), spatial_shape, 1) # (1, 1, H, W)全是-1
        v2p_ind_tensor = v2p_ind_tensor.view(v2p_ind_tensor.shape[0], -1, v2p_ind_tensor.shape[-2],
                                             v2p_ind_tensor.shape[-1])      # (1, 1, H, W)全是-1
        strides = key_1440_indices.new_tensor(strides)
        # fixme 进来的变量没有bs_idx,要手动添加 (b,z,y,x), 而bs_idx必须=0
        key_src_coords = torch.cat([
            key_1440_indices.new_zeros((key_1440_indices.shape[0], 1)),
            key_1440_indices.new_zeros((key_1440_indices.shape[0], 1)),
            torch.div(key_1440_indices, strides, rounding_mode='floor'),
        ], dim=-1)
        key_world_indices = torch.cat([
            # key_world_indices.new_zeros((key_world_indices.shape[0], 1)) + int(bs_idx),
            key_world_indices[:, 1:],
            key_world_indices.new_zeros((key_world_indices.shape[0], 1))
        ], dim=-1)     # (k*b, (y,x,z))
        # todo knn
        dist, idx, empty = voxel_knn_query(
            query_range,
            radius,
            nsample,
            src_world_xyz,
            key_world_indices.contiguous(),
            key_src_coords.int(),
            v2p_ind_tensor,
        )
        # dist, idx, empty = voxel_knn_query(
        #     max_range=query_range,
        #     radius=radius,
        #     nsample=nsample,
        #     xyz=src_world_xyz,
        #     new_xyz=key_world_indices.contiguous(),
        #     new_coords=key_src_coords.int(),
        #     point_indices=v2p_ind_tensor
        # )
        # fixme
        assert idx.max().item() < src_feat.size(
            0), f"索引超界: idx.max() = {idx.max().item()}, src_feat.size(0) = {src_feat.size(0)}"

        voxel_k_feats = src_feat[idx.long()] * (empty == 0).unsqueeze(-1)
        voxel_k_pos = (src_world_xyz[idx.long()] - key_world_indices.unsqueeze(1)) \
            * (empty == 0).unsqueeze(-1)

        return voxel_k_feats, voxel_k_pos

    def height_compression(self, sparse_tensor):

        # indices = sparse_tensor.indices
        dense_tensor = sparse_tensor.dense()
        # B, C, D, H, W = dense_tensor.shape
        dense_tensor = torch.mean(dense_tensor, dim=2).contiguous()
        # dense_tensor = dense_tensor.view(B, -1, H, W).contiguous()
        dense_tensor = dense_tensor.permute(0, 2, 3, 1)
        sparse_tensor = spconv.SparseConvTensor.from_dense(dense_tensor)
        spatial_features = sparse_tensor.features
        indices = sparse_tensor.indices
        return  spatial_features, indices

    def trans_coords(self, fusion_indices, fusion_stride, src_stride):
        trans_fusion_indices = fusion_indices[:, 1:] // int(fusion_stride // src_stride)
        trans_fusion_indices = torch.cat([
            fusion_indices[:, 0].unsqueeze(1),
            trans_fusion_indices
        ], dim=-1)
        return trans_fusion_indices

    def forward(self, batch_dict):
        """
        sparse tensor indices: (bs_idx, z, y, x)
        """
        batch_size = batch_dict['batch_size']

        # revise 这里出来的encoded_spconv_tensor是已经过卷积的
        fusion_sparse_tensor = batch_dict['encoded_spconv_tensor']        # 对齐后的sparse tensor
        fusion_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
        fusion_indices = fusion_sparse_tensor.indices
        fusion_feat = fusion_sparse_tensor.features
        # fusion_dense_tensor = fusion_sparse_tensor.dense()

        voxel_size = torch.tensor(self.voxel_size, device=fusion_indices.device).float()
        pc_range = torch.tensor(self.point_cloud_range, device=voxel_size.device).float()

        # revise heat map 方法1  sparse tensor先经过fc  (N, 128) >> (N, 1)
        # revise heat map 方法2 (N, 128) >> (N, 10)
        heat_voxel_scores_cls10 = self.heat_conv(fusion_feat)     # (N, 128) >> (N, 10)
        heat_voxel_scores, proposal_class = torch.max(heat_voxel_scores_cls10, dim=-1)
        topk_nvoxels = self.model_cfg.MAX_KEY_VOXEL    # fixme
        # if topk_nvoxels < self.model_cfg.MAX_KEY_VOXEL:
        #     raise ValueError('the voxel num of feature map is so fucking less, less than 500')
        # _, topk_indices = torch.topk(heat_voxel_scores.sigmoid(), topk_nvoxels, dim=0)  # 选出得分前500个
        key_feat, key_indices, key_class = [], [], []
        for bs_idx in range(batch_size):
            cur_class = proposal_class[fusion_indices[:, 0] == bs_idx]
            cur_scores = heat_voxel_scores[fusion_indices[:, 0] == bs_idx]
            cur_fusion_feat = fusion_feat[fusion_indices[:, 0] == bs_idx]
            cur_fusion_indices = fusion_indices[fusion_indices[:, 0] == bs_idx]
            _, topk_indices = torch.topk(cur_scores.sigmoid(), topk_nvoxels, dim=0)  # 选出得分前500个
            # topk_indices = topk_indices.squeeze(1)
            cur_key_feat = cur_fusion_feat[topk_indices]
            cur_key_indices = cur_fusion_indices[topk_indices]
            cur_class = cur_class[topk_indices].unsqueeze(0)
            # cur_key_indices = torch.cat([
            #     cur_key_indices.new_zeros(cur_fusion_indices.shape[0], 1)+int(bs_idx),
            #     cur_key_indices
            # ], dim=-1)
            key_indices.append(cur_key_indices)
            key_feat.append(cur_key_feat)
            key_class.append(cur_class)

        key_voxel_feat = torch.cat(key_feat, dim=0)
        key_voxel_indices = torch.cat(key_indices, dim=0)
        key_class = torch.cat(key_class, dim=0)
        # key_voxel_feat = fusion_feat[topk_indices]
        # key_voxel_indices = fusion_indices[topk_indices]   # 在x_conv4下的坐标
        # key voxel 世界坐标 (N,(b,y,x)
        key_voxel_world_indices = (key_voxel_indices[:, 1:3] + 0.5) * fusion_tensor_stride * voxel_size[:2] + pc_range[:2]
        key_voxel_world_indices = torch.cat([
            key_voxel_indices[:, 0].unsqueeze(1),
            key_voxel_world_indices
        ], dim=-1)
        # knn 操作
        multi_k_dict = {}
        for i, src in enumerate(self.model_cfg.DATA_SOURSE):
            src_tensor = batch_dict['multi_scale_3d_features'][src]
            src_stride = batch_dict['multi_scale_3d_strides'][src]
            # src_feat = src_tensor.features
            spatial_shape = src_tensor.spatial_shape
            # 得到 (B, C*D, H, W)  (N, (bs,y,x))
            src_feat_bev, src_indices = self.height_compression(src_tensor)     # 特征图压缩z轴，坐标也
            # 特征图坐标转换成世界坐标
            # world_indices_2d = (src_indices[:, 1:3] + 0.5) * src_stride * voxel_size[:2] + pc_range[:2]
            # world_indices_2d = torch.cat([
            #     src_indices[:, 0].unsqueeze(1),
            #     world_indices_2d
            # ], dim=-1)
            # key_voxel x_conv4坐标转换其他多尺度特征图坐标
            # key_voxel_src_indices = self.trans_coords(key_voxel_indices, fusion_tensor_stride, src_stride)

            batch_k_feat = []
            batch_k_pos = []
            key_voxel = []
            # topk_indices_list = []
            for bs_idx in range(batch_size):
                # cur_voxel_feat, cur_world_indices_2d, cur_voxel_indices, cur_key_voxel_src_indices = \
                #     src_feat[world_indices_2d[:, 0] == bs_idx], \
                #     world_indices_2d[world_indices_2d[:, 0] == bs_idx], \
                #     src_indices[world_indices_2d[:, 0] == bs_idx], \
                #     key_voxel_src_indices[world_indices_2d[:, 0] == bs_idx]

                # cur_key_voxel_indices=key_voxel_indices[key_voxel_indices[:, 0] == bs_idx]
                # cur_key_voxel_feat=key_voxel_feat[key_voxel_indices[:, 0] == bs_idx]
                # cur_key_voxel_src_indices=key_voxel_src_indices[key_voxel_src_indices[:, 0] == bs_idx]

                cur_key_voxel_world_indices = key_voxel_world_indices[key_voxel_world_indices[:, 0] == bs_idx]
                # cur_world_indices_2d=world_indices_2d[world_indices_2d[:, 0] == bs_idx]

                # cur_src_feat_bev = src_feat_bev[src_indices[:, 0] == bs_idx]
                cur_src_feat = src_feat_bev[src_indices[:, 0] == bs_idx]
                cur_src_indices = src_indices[src_indices[:, 0] == bs_idx]
                # 无bs
                key_voxel_1440_indices = ((cur_key_voxel_world_indices[:, 1:] - pc_range[:2]) / voxel_size[:2]).to(torch.int64)  # key voxel 1440范围下的坐标

                knn_cfg = self.model_cfg.KNN_QUERY[src]
                knn_feat, knn_pos = self.voxel_knn_query_wrapper(
                    key_world_indices=cur_key_voxel_world_indices,
                    key_1440_indices=key_voxel_1440_indices,
                    src_feat=cur_src_feat,
                    src_coords=cur_src_indices,
                    spatial_shape=spatial_shape[1:],
                    strides=src_stride,
                    bs_idx=bs_idx,
                    model_cfg=knn_cfg
                )
                batch_k_feat.append(knn_feat)
                batch_k_pos.append(knn_pos)
                key_voxel.append(key_voxel_world_indices)
                # topk_indices_list.append(topk_indices)
            k_feat = torch.cat(batch_k_feat, dim=0)
            k_feat = k_feat.permute(0, 2, 1)
            k_feat_mlp = self.knn_feat_mlp[i](k_feat).squeeze(-1)
            k_pos = torch.cat(batch_k_pos, dim=0)
            key_voxel = torch.cat(key_voxel, dim=0)
            k = {'features': k_feat,
                 'indices': k_pos,
                 'mlp_feat': k_feat_mlp,
                 'key_voxel': key_voxel,
                 # 'topk_indices': topk_indices_,
                 }
            multi_k_dict[src] = k

        # adaptive fusion
        w = F.softmax(self.k_weight_mlp(key_voxel_feat))
        fuse_feat_list = []
        for i ,src in enumerate(self.model_cfg.DATA_SOURSE):
            key_voxel_feat = w[:, i].unsqueeze(1) * key_voxel_feat          # todo 逐点相乘，理解为对每个key voxel配置权重
            fuse_feat = key_voxel_feat * multi_k_dict[src]['mlp_feat']  # fixme
            fuse_feat_list.append(fuse_feat)
        fuse_feat = torch.stack(fuse_feat_list)
        fuse_feat = torch.sum(fuse_feat, dim=0)
        fuse_feat = torch.cat([fuse_feat, key_voxel_feat], dim=-1)

        key_feat = self.key_fuse_conv(fuse_feat)

        return key_feat, key_voxel_indices, heat_voxel_scores, heat_voxel_scores_cls10, key_class


# class LEMA_new(nn.Module):
#     def __init__(self, model_cfg, voxel_size, point_cloud_range):
#         super().__init__()
#         self.model_cfg = model_cfg
#         self.voxel_size = torch.tensor(voxel_size).float()
#         self.point_cloud_range = torch.tensor(point_cloud_range).float()
#
#         self.heat_conv = self._make_fc_layers(model_cfg.HEAT_CONV.MLP, model_cfg.HEAT_CONV.INPUT_DIM, 1)
#         self.knn_feat_mlp = nn.ModuleList([
#             nn.Linear(model_cfg.KNN_QUERY[src].NSAMPLE, 1) for src in model_cfg.KNN_QUERY
#         ])
#         self.k_weight_mlp = self._make_fc_layers(model_cfg.WEIGHT_MLP.MLP, model_cfg.WEIGHT_MLP.INPUT_DIM, 3)
#         self.key_fuse_conv = self._make_fc_layers(model_cfg.FUSE_CONV.MLP, model_cfg.FUSE_CONV.INPUT_DIM,
#                                                   model_cfg.FUSE_CONV.OUT_DIM)
#
#     @staticmethod
#     def _make_fc_layers(fc_cfg, input_channels, output_channels):
#         fc_layers = []
#         c_in = input_channels
#         for k in range(0, fc_cfg.__len__()):
#             fc_layers.extend([
#                 nn.Linear(c_in, fc_cfg[k], bias=False),
#                 nn.BatchNorm1d(fc_cfg[k]),
#                 nn.ReLU(),
#             ])
#             c_in = fc_cfg[k]
#         fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
#         return nn.Sequential(*fc_layers)
#
#     @torch.no_grad()
#     def voxel_knn_query(self, key_world_indices, key_1440_indices, src_feat, src_coords, spatial_shape, strides,
#                         model_cfg):
#         query_range, radius, nsample = model_cfg.QUERY_RANGE, model_cfg.RADIUS, model_cfg.NSAMPLE
#
#         src_world_indices_2d = (src_coords[:, 1:3] + 0.5) * strides * self.voxel_size[:2] + self.point_cloud_range[:2]
#         src_world_xyz = torch.cat(
#             [src_world_indices_2d, src_world_indices_2d.new_zeros((src_world_indices_2d.shape[0], 1))], dim=-1)
#
#         key_world_indices = torch.cat(
#             [key_world_indices[:, 1:], key_world_indices.new_zeros((key_world_indices.shape[0], 1))], dim=-1)
#
#         dist, idx, empty = voxel_knn_query(query_range, radius, nsample, src_world_xyz, key_world_indices.contiguous(),
#                                            key_1440_indices.int(), spatial_shape)
#
#         voxel_k_feats = src_feat[idx.long()] * (empty == 0).unsqueeze(-1)
#         voxel_k_pos = (src_world_xyz[idx.long()] - key_world_indices.unsqueeze(1)) * (empty == 0).unsqueeze(-1)
#
#         return voxel_k_feats, voxel_k_pos
#
#     @staticmethod
#     def height_compression(sparse_tensor):
#         dense_tensor = sparse_tensor.dense()
#         dense_tensor = torch.mean(dense_tensor, dim=2).permute(0, 2, 3, 1).contiguous()
#         sparse_tensor = spconv.SparseConvTensor.from_dense(dense_tensor)
#         return sparse_tensor.features, sparse_tensor.indices
#
#     def forward(self, batch_dict):
#         batch_size = batch_dict['batch_size']
#         fusion_sparse_tensor = batch_dict['encoded_spconv_tensor']
#         fusion_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
#         fusion_indices, fusion_feat = fusion_sparse_tensor.indices, fusion_sparse_tensor.features
#
#         heat_voxel_scores = self.heat_conv(fusion_feat)
#         topk_nvoxels = self.model_cfg.MAX_KEY_VOXEL
#
#         key_feat, key_indices = [], []
#         for bs_idx in range(batch_size):
#             bs_mask = fusion_indices[:, 0] == bs_idx
#             cur_scores = heat_voxel_scores[bs_mask]
#             cur_fusion_feat = fusion_feat[bs_mask]
#             cur_fusion_indices = fusion_indices[bs_mask]
#             _, topk_indices = torch.topk(cur_scores.sigmoid(), topk_nvoxels, dim=0)
#             topk_indices = topk_indices.squeeze(1)
#             key_feat.append(cur_fusion_feat[topk_indices])
#             key_indices.append(cur_fusion_indices[topk_indices])
#
#         key_voxel_feat = torch.cat(key_feat, dim=0)
#         key_voxel_indices = torch.cat(key_indices, dim=0)
#         key_voxel_world_indices = (key_voxel_indices[:, 1:3] + 0.5) * fusion_tensor_stride * self.voxel_size[:2] + self.point_cloud_range[:2]
#         key_voxel_world_indices = torch.cat([key_voxel_indices[:, 0].unsqueeze(1), key_voxel_world_indices], dim=-1)
#
#         multi_k_dict = {}
#         for i, src in enumerate(self.model_cfg.DATA_SOURSE):
#             src_tensor = batch_dict['multi_scale_3d_features'][src]
#             src_stride = batch_dict['multi_scale_3d_strides'][src]
#             src_feat_bev, src_indices = self.height_compression(src_tensor)
#
#             k_feat, k_pos = [], []
#             for bs_idx in range(batch_size):
#                 bs_mask = key_voxel_world_indices[:, 0] == bs_idx
#                 cur_key_voxel_world_indices = key_voxel_world_indices[bs_mask]
#                 cur_src_feat = src_feat_bev[src_indices[:, 0] == bs_idx]
#                 cur_src_indices = src_indices[src_indices[:, 0] == bs_idx]
#                 key_voxel_1440_indices = (
#                             (cur_key_voxel_world_indices[:, 1:] - self.point_cloud_range[:2]) / self.voxel_size[:2]).to(
#                     torch.int64)
#
#                 knn_feat, knn_pos = self.voxel_knn_query(
#                     cur_key_voxel_world_indices, key_voxel_1440_indices,
#                     cur_src_feat, cur_src_indices, src_tensor.spatial_shape[1:],
#                     src_stride, self.model_cfg.KNN_QUERY[src]
#                 )
#                 k_feat.append(knn_feat)
#                 k_pos.append(knn_pos)
#
#             k_feat = torch.cat(k_feat, dim=0).permute(0, 2, 1)
#             k_feat_mlp = self.knn_feat_mlp[i](k_feat).squeeze(-1)
#             k_pos = torch.cat(k_pos, dim=0)
#
#             multi_k_dict[src] = {
#                 'features': k_feat,
#                 'indices': k_pos,
#                 'mlp_feat': k_feat_mlp,
#                 'key_voxel': key_voxel_world_indices,
#             }
#
#         w = F.softmax(self.k_weight_mlp(key_voxel_feat), dim=1)
#         fuse_feat = sum(w[:, i].unsqueeze(1) * key_voxel_feat * multi_k_dict[src]['mlp_feat'] for i, src in
#                         enumerate(self.model_cfg.DATA_SOURSE))
#         fuse_feat = torch.cat([fuse_feat, key_voxel_feat], dim=-1)
#
#         key_feat = self.key_fuse_conv(fuse_feat)
#
#         return key_feat, key_voxel_indices, heat_voxel_scores


class SparseDet(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range,num_class, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        # self.num_class = num_class
        self.max_query = model_cfg.LMEA.MAX_KEY_VOXEL
        self.max_kv = model_cfg.MAX_KV_VOXEL
        self.lema = LEMA(model_cfg.LMEA, voxel_size, point_cloud_range, num_class)
        self.mlp = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.sasa = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.top_kv = 1000

    def forward(self, batch_dict):
        key_feat, key_voxel_indices, heat_voxel_scores, heat_voxel_scores_cls10, query_labels = self.lema(batch_dict)
        batch_size = batch_dict['batch_size']

        fusion_sparse_tensor = batch_dict['encoded_spconv_tensor']        # 对齐后的sparse tensor
        # bev_stride = batch_dict['encoded_spconv_tensor_stride']
        bev_pos = fusion_sparse_tensor.indices
        bev_feat = fusion_sparse_tensor.features
        spatial_shape = fusion_sparse_tensor.spatial_shape

        heatmap_sp = spconv.SparseConvTensor(
            features=heat_voxel_scores_cls10,
            indices=bev_pos,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        heat_map = heatmap_sp.dense()

        # GEA模块
        # 计算key voxel之间的空间距离
        batch_size = batch_dict['batch_size']
        kv_indices = []
        kv_feat = []
        q_feat = []
        q_indices = []
        query_heatmap_score = []
        for bs_idx in range(batch_size):

            cur_key_voxel_indices = key_voxel_indices[key_voxel_indices[:, 0] == bs_idx]
            cur_key_feat = key_feat[key_voxel_indices[:, 0] == bs_idx]
            coords_square = cur_key_voxel_indices.pow(2).sum(dim=1, keepdim=True)
            dist_squared = coords_square + coords_square.t() - 2 * cur_key_voxel_indices.float() @ cur_key_voxel_indices.t().float()
            dist_matrix = dist_squared.sqrt()
            eta = self.mlp(cur_key_feat)
            attn_mask = eta * dist_matrix
            cur_key_feat = cur_key_feat.unsqueeze(0).permute(1, 0, 2)
            Q, mask = self.sasa(cur_key_feat, cur_key_feat, cur_key_feat, attn_mask=attn_mask)
            q_feat.append(Q)

            cur_scores = heat_voxel_scores[bev_pos[:, 0] == bs_idx]
            cur_fusion_feat = bev_feat[bev_pos[:, 0] == bs_idx]
            cur_fusion_indices = bev_pos[bev_pos[:, 0] == bs_idx]
            _, topk_indices = torch.topk(cur_scores.sigmoid(), self.max_kv, dim=0)  # 选出得分前1000个
            # 随街挑选1000个体素
            num_voxels, C = cur_fusion_feat.shape
            rand_indices = torch.randperm(num_voxels)[:self.max_kv]

            cur_key_feat = cur_fusion_feat[rand_indices]
            cur_key_indices = cur_fusion_indices[rand_indices]
            kv_indices.append(cur_key_indices.unsqueeze(1))
            kv_feat.append(cur_key_feat.unsqueeze(1))
            q_indices.append(cur_key_voxel_indices.unsqueeze(1))
            cur_scores_cls10 = heat_voxel_scores_cls10[bev_pos[:, 0] == bs_idx]
            query_score = cur_scores_cls10[topk_indices]
            query_heatmap_score.append(query_score[:self.max_query].unsqueeze(-1))
        kv_feat = torch.cat(kv_feat, dim=1)
        kv_indices = torch.cat(kv_indices, dim=1)
        q_feat = torch.cat(q_feat, dim=1)
        q_indices = torch.cat(q_indices, dim=1)
        query_heatmap_score = torch.cat(query_heatmap_score, dim=-1)

        batch_dict['kv_feat'] = kv_feat
        batch_dict['kv_indices'] = kv_indices
        batch_dict['q_feat'] = q_feat
        batch_dict['q_indices'] = q_indices
        batch_dict['heatmap_dense'] = heat_map
        batch_dict['query_heatmap_score'] = query_heatmap_score
        batch_dict['query_labels'] = query_labels

        return batch_dict

