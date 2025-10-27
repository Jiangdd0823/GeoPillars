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

class Attention(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.fusion_mode = self.model_cfg.FUSION        # fixme 这里在配置文件添加，用来定义特征融合方式


class VoteAttentionNeck(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_class, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range  = point_cloud_range
        self.num_class = num_class

        self.data_source = self.model_cfg.DATA_SOURCE
        self.offset_range = self.model_cfg.QUERY_RANGE
        # 每层特征层的体素分类得分
        self.cls_conv = nn.ModuleList()
        for i in range(len(self.data_source)):
            cls_conv = make_fc_layers(
                self.model_cfg.MLP[i],
                self.model_cfg.NUM_BEV_FEATURES[i], num_class, linear=True
            )
            self.cls_conv.append(cls_conv)

        # 每层特征层的体素偏移量
        self.offset_conv = nn.ModuleList()
        for i in range(len(self.data_source)):
            offset_conv = make_fc_layers(
                self.model_cfg.MLP[i],
                self.model_cfg.NUM_BEV_FEATURES[i], 2, linear=True
            )
            self.offset_conv.append(offset_conv)


    def vote_layer(self, features, indices, strides, i):
        score = self.cls_conv[i](features)
        offset = self.offset_conv[i](features)  * 16 / strides
        # 偏移量取整
        offset_ceil = torch.ceil(offset)        # fixme 要加入取整正则化
        # ANCHOR_HEIGHT = 0.0
        # voxel_centers = common_utils.get_voxel_centers(indices[:, 1:], strides, self.voxel_size, self.point_cloud_range, dim=2)   # 以原坐标系为基础
        # voxel_centers = torch.cat([
        #     indices[:, 0:1],
        #     voxel_centers,
        #     voxel_centers.new_full((voxel_centers.shape[0], 1), ANCHOR_HEIGHT)
        # ], dim=-1)  # (N, 4), [bs_idx, x, y, z]     z轴为0，xy平面上的特征图体素中心
        limited_offset = []
        for axis in range(len(self.offset_range[i])):
            limited_offset.append(offset_ceil[..., axis].clamp(
                min=-self.offset_range[i][axis],
                max=self.offset_range[i][axis]))       # clamp(min,max)限制范围
        limited_offset = torch.stack(limited_offset, dim=-1)
        votes = indices[:, 1:] + limited_offset          # 特征图体素中心（原始坐标系下）＋偏差
        votes = torch.cat([indices[:, 0:1], votes], dim=-1)

        return votes, score, offset

    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']

        multi_scale_2d_features= batch_dict['multi_scale_2d_features']
        multi_scale_2d_strides = batch_dict['multi_scale_2d_strides']
        topk_npillars = self.model_cfg.MAX_SAMPLE

        for i, cur in enumerate(self.data_source):
            sparse_tensors = multi_scale_2d_features[cur]
            features = sparse_tensors.features
            indices = sparse_tensors.indices
            strides = multi_scale_2d_strides[cur]

            votes, score, offset = [], [], []
            for bs_idx in range(batch_size):
                bs_indices = indices[indices[:, 0] == bs_idx]
                bs_features = features[indices[:, 0] == bs_idx]
                bs_votes, bs_score, bs_offset = self.vote_layer(bs_features, bs_indices, strides, i)
                _, topk_indices = torch.topk(bs_score.sigmoid(), topk_npillars, dim=0)
                tk_bs_votes, tk_bs_features, tk_bs_scores = \
                    bs_votes[topk_indices], bs_features[topk_indices], bs_score[topk_indices]

