import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from ...ops.pointnet2.pvtssd.pointnet2_batch.pointnet2_utils import point_sampler
from ...ops.pointnet2.pvtssd.pointnet2_stack.voxel_query_utils import voxel_knn_query
from ...ops.pointnet2.pvtssd.pointnet2_stack.pointnet2_utils import k_interpolate
from ..model_utils.transformer import Transformer
from ...utils.spconv_utils import  spconv

from ...utils import common_utils_pvtssd as common_utils


def fps_pool_layer(points, point_feats, point_scores, batch_size, model_cfg, mode):
    fps_indices = []
    pre_sum = 0
    for bs_idx in range(batch_size):
        cur_points, cur_point_feats, cur_point_scores = \
            points[points[:, 0] == bs_idx][:, 1:4], point_feats[points[:, 0] == bs_idx], \
            point_scores[points[:, 0] == bs_idx]
        assert len(cur_points) == len(cur_point_feats) == len(cur_point_scores)
        topk_nponits = min(len(cur_point_scores), model_cfg.MAX_NPOINTS[mode])
        _, topk_indices = torch.topk(cur_point_scores.sigmoid(), topk_nponits, dim=0)  # todo 选出分数最高的体素位置

        cur_points, cur_point_feats, cur_point_scores = \
            cur_points[topk_indices], cur_point_feats[topk_indices], \
            cur_point_scores[topk_indices]

        cur_fps_indices = []
        for fps_npoints, fps_type in zip(model_cfg.NPOINTS[mode], model_cfg.TYPE[mode]):       # train阶段会fps两次
            if fps_npoints > 0:
                cur_fps_indices_ = point_sampler(  # 通过得分选出4096个体素后，用FPS再筛选64个体素
                    fps_type=fps_type,
                    xyz=cur_points.unsqueeze(0).contiguous(),
                    npoints=fps_npoints,
                    features=cur_point_feats.unsqueeze(0).transpose(1, 2).contiguous(),
                    scores=cur_point_scores.unsqueeze(0).contiguous()
                ).squeeze(0)
            else:
                cur_fps_indices_ = torch.arange(len(cur_points)).to(cur_points.device)
            cur_fps_indices.append(cur_fps_indices_)
        cur_fps_indices = torch.cat(cur_fps_indices, dim=0)

        cur_fps_indices = topk_indices[cur_fps_indices.long()]
        fps_indices.append(cur_fps_indices + pre_sum)
        pre_sum += torch.sum(points[:, 0] == bs_idx)
    fps_indices = torch.cat(fps_indices, dim=0).long()
    return fps_indices


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


class VoteLayer(nn.Module):
    def __init__(self, offset_range, input_channels, mlps, num_class=1):
        """
        offset_range: 体素中心的偏差
        """
        super().__init__()
        self.offset_range = offset_range
        # self.offset_conv = make_fc_layers(mlps, input_channels, 2, linear=True)
        self.offset_conv = make_fc_layers(mlps, input_channels, 3, linear=True)   # revise 改成xyz
        self.cls_conv = make_fc_layers(mlps, input_channels, num_class, linear=True)
        # self.cls_conv = make_fc_layers(mlps, input_channels, 1, linear=True)
        self.cls_conv[-1].bias.data.fill_(-2.19)

    def forward(self, seeds, seed_feats):
        """
        Args:
            seeds: (N, 4), [bs_idx, x, y, z]  世界坐标
            features: (N, C)
        Return:
            new_xyz: (N, 3)
        """
        seed_offset = self.offset_conv(seed_feats)  # offset (N, 3)
        seed_cls = self.cls_conv(seed_feats)  # (N, num_class)
        limited_offset = []
        for axis in range(len(self.offset_range)):
            limited_offset.append(seed_offset[..., axis].clamp(
                min=-self.offset_range[axis],
                max=self.offset_range[axis]))       # clamp(min,max)限制范围
        limited_offset = torch.stack(limited_offset, dim=-1)
        votes = seeds[:, 1:4] + limited_offset          # 特征图pillar中心（原始坐标系下）＋偏差  revise 改成 xyz+
        votes = torch.cat([seeds[:, 0:1], votes], dim=-1)
        return votes, seed_cls, seed_offset


class QueryInitial(nn.Module):
    """
    pvt-ssd 的query initialization branch 1 and branch 2
    """
    def __init__(self, num_class, total_cfg, knn_cfg, voxel_size, point_cloud_range,):
        super().__init__()

        self.fps_config = total_cfg.FPS_CONFIG
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.input_channels = knn_cfg.DIM

        self.vote_layer = VoteLayer(            # 生成reference point偏差？
            knn_cfg.OFFSET_RANGE,
            self.input_channels,
            knn_cfg.MLPS,
            num_class
        )

    def get_bev_features(self, points, bev_features, bev_stride):
        """
        Args:
            points: (B, K, 3)
        """
        point_cloud_range = torch.tensor(self.point_cloud_range, device=points.device, dtype=torch.float32)
        voxel_size = torch.tensor(self.voxel_size, device=points.device, dtype=torch.float32)
        xy = (points[..., 0:2] - point_cloud_range[0:2]) / voxel_size[0:2] / bev_stride  # (B, K, 2)  转换成bev特征图的坐标
        h, w = bev_features.shape[-2:]
        norm_xy = torch.cat([
            xy[..., 0:1] / (w - 1),
            xy[..., 1:2] / (h - 1)
        ], dim=-1)      #
        bev_feats = torch.nn.functional.grid_sample(        # 从 bev_features 张量中采样出由 norm_xy 定义的新坐标处的特征值。grid_sample 使用双线性插值来确定这些新坐标处的值，即使这些坐标可能不在原始特征图的整数位置上
            bev_features,  # (B, C, H, W)
            norm_xy.unsqueeze(2) * 2 - 1,  # (B, K, 1, 2)
            align_corners=True
        ).squeeze(-1).permute(0, 2, 1)  # (B, C, K) -> (B, K, C)

        return bev_feats

    def forward(self, spatial_coords, spatial_features, spatial_stride, batch_size):
        """
        spatial_coords: 特征图坐标
        spatial_stride：特征图步长
        """
        voxel_coords = spatial_coords
        voxel_features = spatial_features
        voxel_stride = spatial_stride
        batch_size = batch_size
        point_cloud_range = torch.tensor(self.point_cloud_range)
        voxel_centers = common_utils.get_voxel_centers(voxel_coords[:, 1:], voxel_stride, self.voxel_size, point_cloud_range, dim=2)   # 以原坐标系为基础
        voxel_centers = torch.cat([
            voxel_coords[:, 0:1],
            voxel_centers,
            voxel_centers.new_full((voxel_centers.shape[0], 1), 0)
        ], dim=-1)  # (N, 4), [bs_idx, x, y, z]     z轴为0，3d世界坐标体素中心

        # mode = 'train' if self.training else 'test'
        mode = 'train'
        seeds = voxel_centers  # (N, 4) 体素中心（用世界坐标）
        seed_features = voxel_features  # (N, C)

        # branch 1
        votes, seed_cls, seed_reg = self.vote_layer(seeds, seed_features)       # votes为体素中心＋偏差  # revise votes (N,4)
        # todo 这里面用到FPS，里面用到S-FPS，尽可能采样前景点
        fps_indices = fps_pool_layer(       # fixme seeds是(N, 3),fps操作时要有xyz坐标
            seeds, seed_features, torch.max(seed_cls, dim=-1)[0], batch_size,
            self.fps_config, mode
        )               # question 去看该函数fps出来的坐标是啥坐标：特征图坐标
        vote_candidates = votes[fps_indices].detach()  # 现在(B*K, 3)   源码(B*K, 4)     todo reference point revise (B*K, 4)
        vote_features = seed_features[fps_indices]
        vote = {
            'vote_candidates': vote_candidates,
            'vote_features': vote_features,
            'seeds': seeds,
            'seeds_cls': seed_cls,
            'votes': votes,
        }

        return vote, fps_indices


class Custom(nn.Module):
    """
    follow pvrcnn, 这个是voxel set abstraction
    """
    def __init__(self, model_cfg, num_class, voxel_size, point_cloud_range,  **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.queryinitial_cfg =self.model_cfg.QueryInitial
        self.voxel_knn_cfg = model_cfg.QueryInitial.VOXEL_KNN_QUERY

        # FIXME 这里在父类中有初始化，pvtssd里的box_coder和源码不一样
        a=0
        # target_cfg = self.model_cfg.TARGET_CONFIG
        # self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
        #     **target_cfg.BOX_CODER_CONFIG
        # )

        self.queryinitial = nn.ModuleList()
        self.voxel_feat_trans = nn.ModuleList()

        for src_name in self.model_cfg.FEATURES_SOURCE:
            knn_cfg = self.voxel_knn_cfg[src_name]
            queryinitial_layer = QueryInitial(
                num_class,
                self.queryinitial_cfg,
                knn_cfg,
                self.voxel_size,
                self.point_cloud_range,
            )
            self.queryinitial.append(queryinitial_layer)

            voxel_feat_trans = make_fc_layers(
                [256],
                input_channels=knn_cfg.DIM,
                linear=True,
            )
            self.voxel_feat_trans.append(voxel_feat_trans)

        self.voxel_feat_reduce = make_fc_layers(
            [256],
            input_channels=256*3,
            linear=True
        )

        trans_cfg = model_cfg.TRANS_CONFIG
        self.transformer = Transformer(
            d_model=256, nhead=trans_cfg.NHEAD, num_decoder_layers=trans_cfg.NUM_DEC, dim_feedforward=trans_cfg.FNN_DIM,
            dropout=trans_cfg.DP_RATIO
        )
        # fixme 体素特征出来，to sparse 再 to dense，过于稀疏
        self.bev_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # self.shared_conv = make_fc_layers(
        #     fc_cfg=self.model_cfg.SHARED_FC,
        #     input_channels=256,
        #     linear=True
        # # )
        # channel_out = self.model_cfg.SHARED_FC[-1]
        # channel_out = 256
        # self.num_anchors_per_location = sum(self.num_anchors_per_location)
        # fixme 这里是mlp，我转成bev图后不适用
        a = 0
        # self.cls_conv = make_fc_layers(
        #     fc_cfg=self.model_cfg.CLS_FC,
        #     input_channels=channel_out,
        #     output_channels=num_class,
        #     linear=True
        # )
        # self.box_conv = make_fc_layers(
        #     fc_cfg=self.model_cfg.REG_FC,
        #     input_channels=channel_out,
        #     output_channels=self.box_coder.code_size,
        #     linear=True
        # )
        # self.conv_dir_cls = make_fc_layers(
        #     fc_cfg=self.model_cfg.REG_FC,
        #     input_channels=channel_out,
        #     output_channels=self.box_coder.code_size,
        #     linear=True)
        # self.cls_conv = nn.Conv2d(
        #     channel_out, self.num_anchors_per_location * self.num_class,
        #     kernel_size=1
        # )
        # self.box_conv = nn.Conv2d(
        #     channel_out, self.num_anchors_per_location * self.box_coder.code_size,
        #     kernel_size=1
        # )
        # if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
        #     self.conv_dir_cls = nn.Conv2d(
        #         channel_out,
        #         self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
        #         kernel_size=1
        #     )
        # else:
        #     self.conv_dir_cls = None
        self._reset_parameters(weight_init='xavier')

    def _reset_parameters(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # revise 这是属于pvtssd的get loss
    a = 0
    # def get_loss(self, tb_dict=None):
    #     tb_dict = {} if tb_dict is None else tb_dict
    #     seed_reg_loss, tb_dict = self.get_seed_reg_loss(tb_dict)
    #     seed_cls_loss, tb_dict = self.get_seed_cls_loss(tb_dict)
    #     vote_cls_loss, tb_dict = self.get_vote_cls_loss(tb_dict)
    #     vote_reg_loss, tb_dict = self.get_vote_reg_loss(tb_dict)
    #     vote_corner_loss, tb_dict = self.get_vote_corner_loss(tb_dict)
    #     point_loss = seed_reg_loss + seed_cls_loss + vote_cls_loss + vote_reg_loss + vote_corner_loss
    #     return point_loss, tb_dict
    #
    # def get_seed_single_reg_loss(self, votes, seed_cls_labels, gt_box_of_fg_seeds, index, tb_dict=None):
    #     pos_mask = seed_cls_labels > 0
    #     seed_center_labels = gt_box_of_fg_seeds[:, 0:3]
    #     seed_center_loss = self.seed_reg_loss_func(
    #         votes[pos_mask][:, 1:], seed_center_labels
    #     ).sum(dim=-1).mean()
    #     seed_center_loss = seed_center_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seed_reg_weight_list'][index]
    #
    #     tb_dict.update({
    #         f'seed_reg_loss_{index}': seed_center_loss.item(),
    #         f'seed_pos_num_{index}': int(pos_mask.sum().item() / self.forward_ret_dict['batch_size'])
    #     })
    #     return seed_center_loss, tb_dict
    #
    # def get_seed_reg_loss(self, tb_dict=None):
    #     seed_cls_labels_list = self.forward_ret_dict['seed_cls_labels_list']
    #     gt_box_of_fg_seeds_list = self.forward_ret_dict['gt_box_of_fg_seeds_list']
    #     votes_list = self.forward_ret_dict['votes_list']
    #     seed_center_loss_list = []
    #     for i in range(len(votes_list)):
    #         seed_center_loss, tb_dict = self.get_seed_single_reg_loss(
    #             votes_list[i],
    #             seed_cls_labels_list[i],
    #             gt_box_of_fg_seeds_list[i],
    #             i,
    #             tb_dict
    #         )
    #         seed_center_loss_list.append(seed_center_loss)
    #     return sum(seed_center_loss_list), tb_dict
    #
    # def get_seed_single_cls_loss(self, point_cls_preds, point_cls_labels, index, tb_dict=None):
    #
    #     if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center':
    #         assert len(point_cls_preds) == len(point_cls_labels)
    #         point_loss_cls = self.seed_cls_loss_func(self.sigmoid(point_cls_preds), point_cls_labels)
    #     else:
    #         positives = point_cls_labels > 0
    #         negatives = point_cls_labels == 0
    #         cls_weights = negatives * 1.0 + positives * 1.0
    #         pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.AUX_CLS_POS_NORM else cls_weights.sum()
    #         cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    #
    #         num_class = 1
    #         one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), num_class + 1)
    #         one_hot_targets.scatter_(-1, (point_cls_labels > 0).unsqueeze(-1).long(), 1.0)
    #         one_hot_targets = one_hot_targets[..., 1:]
    #         cls_loss_src = self.aux_cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
    #         point_loss_cls = cls_loss_src.sum()
    #
    #     point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seed_cls_weight_list'][index]
    #     tb_dict.update({
    #         f'seed_cls_loss_{index}': point_loss_cls.item(),
    #     })
    #     return point_loss_cls, tb_dict
    #
    # def get_seed_cls_loss(self, tb_dict=None):
    #     seed_cls_labels_list = self.forward_ret_dict[
    #         'seed_cls_targets_list'] if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center' else self.forward_ret_dict[
    #         'seed_cls_labels_list']
    #     seeds_cls_list = self.forward_ret_dict['seeds_cls_list']
    #     seed_cls_loss_list = []
    #     for i in range(len(seeds_cls_list)):
    #         seed_cls_loss, tb_dict = self.get_seed_single_cls_loss(
    #             seeds_cls_list[i],
    #             seed_cls_labels_list[i],
    #             i,
    #             tb_dict
    #         )
    #         seed_cls_loss_list.append(seed_cls_loss)
    #     return sum(seed_cls_loss_list), tb_dict
    #
    # def get_vote_cls_loss(self, tb_dict=None):
    #     point_cls_labels = self.forward_ret_dict['vote_cls_labels']
    #     point_cls_preds = self.forward_ret_dict['vote_cls_preds']
    #
    #     positives = point_cls_labels > 0
    #     negatives = point_cls_labels == 0
    #     cls_weights = negatives * 1.0 + positives * 1.0
    #     pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.CLS_POS_NORM else cls_weights.sum()
    #     cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    #
    #     one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
    #     one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
    #     one_hot_targets = one_hot_targets[..., 1:]
    #
    #     if 'WithCenterness' in self.model_cfg.LOSS_CONFIG.CLS_LOSS:
    #         votes = self.forward_ret_dict['votes'].detach()
    #         gt_box_of_fg_votes = self.forward_ret_dict['gt_box_of_fg_votes']
    #         pos_centerness = box_utils.generate_centerness_mask(votes[positives][:, 1:], gt_box_of_fg_votes)
    #         centerness_mask = positives.new_zeros(positives.shape).float()
    #         centerness_mask[positives] = pos_centerness
    #         one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1)
    #
    #     cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
    #     point_loss_cls = cls_loss_src.sum()
    #
    #     point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_cls_weight']
    #     tb_dict.update({
    #         'vote_cls_loss': point_loss_cls.item(),
    #         'vote_pos_num': int(positives.sum().item() / self.forward_ret_dict['batch_size'])
    #     })
    #     return point_loss_cls, tb_dict
    #
    # def get_vote_reg_loss(self, tb_dict=None):
    #     pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
    #     point_box_labels = self.forward_ret_dict['vote_box_labels']
    #     point_box_preds = self.forward_ret_dict['vote_box_preds']
    #
    #     reg_weights = pos_mask.float()
    #     pos_normalizer = pos_mask.sum().float()
    #     reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    #
    #     xyzlwh_preds = point_box_preds[:, :6]
    #     xyzlwh_labels = point_box_labels[:, :6]
    #     point_loss_xyzlwh = self.reg_loss_func(xyzlwh_preds, xyzlwh_labels, reg_weights).sum() \
    #                         * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][0]
    #
    #     angle_bin_num = self.box_coder.angle_bin_num
    #     dir_cls_preds = point_box_preds[:, 6:6 + angle_bin_num]
    #     dir_cls_labels = point_box_labels[:, 6:6 + angle_bin_num]
    #     point_loss_dir_cls = F.cross_entropy(dir_cls_preds, dir_cls_labels.argmax(dim=-1), reduction='none')
    #     point_loss_dir_cls = (point_loss_dir_cls * reg_weights).sum() \
    #                          * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][1]
    #
    #     dir_res_preds = point_box_preds[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
    #     dir_res_labels = point_box_labels[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
    #
    #     dir_res_preds = torch.sum(dir_res_preds * dir_cls_labels, dim=-1)
    #     dir_res_labels = torch.sum(dir_res_labels * dir_cls_labels, dim=-1)
    #     point_loss_dir_res = self.reg_loss_func(dir_res_preds, dir_res_labels, weights=reg_weights)
    #     point_loss_dir_res = point_loss_dir_res.sum() \
    #                          * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][2]
    #
    #     point_loss_velo = 0
    #     if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
    #         point_loss_velo = self.reg_loss_func(
    #             point_box_preds[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
    #             point_box_labels[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
    #             reg_weights
    #         ).sum()
    #         tb_dict.update({
    #             'vote_reg_velo_loss': point_loss_velo.item()
    #         })
    #
    #     point_loss_box = point_loss_xyzlwh + point_loss_dir_cls + point_loss_dir_res + point_loss_velo
    #     point_loss_box = point_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_reg_weight']
    #     tb_dict.update({
    #         'vote_reg_loss': point_loss_box.item(),
    #         'vote_reg_xyzlwh_loss': point_loss_xyzlwh.item(),
    #         'vote_reg_dir_cls_loss': point_loss_dir_cls.item(),
    #         'vote_reg_dir_res_loss': point_loss_dir_res.item(),
    #     })
    #     return point_loss_box, tb_dict
    #
    # def get_vote_corner_loss(self, tb_dict=None):
    #     pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
    #     gt_boxes = self.forward_ret_dict['gt_box_of_fg_votes']
    #     pred_boxes = self.forward_ret_dict['point_box_preds']
    #     pred_boxes = pred_boxes[pos_mask]
    #     loss_corner = loss_utils.get_corner_loss_lidar(
    #         pred_boxes[:, 0:7],
    #         gt_boxes[:, 0:7],
    #         p=self.model_cfg.LOSS_CONFIG.CORNER_LOSS_TYPE
    #     ).mean()
    #     loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_corner_weight']
    #     tb_dict.update({'vote_corner_loss': loss_corner.item()})
    #     return loss_corner, tb_dict

    # def assign_targets(self, input_dict):
    #     """
    #     Args:
    #         input_dict:
    #             point_features: (N1 + N2 + N3 + ..., C)
    #             batch_size:
    #             point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
    #             gt_boxes (optional): (B, M, 8)
    #     Returns:
    #         point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
    #         point_part_labels: (N1 + N2 + N3 + ..., 3)
    #     """
    #     gt_boxes = input_dict['gt_boxes']
    #     batch_size = gt_boxes.shape[0]
    #
    #     """ Aux loss """
    #     spatial_features = input_dict['spatial_features']
    #     spatial_features_stride = input_dict['spatial_features_stride']
    #     feature_map_size = spatial_features.spatial_shape
    #     feature_map_stride = spatial_features_stride
    #
    #     gt_corners = box_utils.boxes_to_corners_3d(gt_boxes.view(-1, gt_boxes.shape[-1]))
    #     gt_corners = gt_corners[:, :4, :2].contiguous().view(batch_size, -1, 4, 2)
    #     center_map = torch.zeros((batch_size, self.num_class, feature_map_size[0], feature_map_size[1]),
    #                              dtype=torch.float32).to(gt_boxes.device)
    #     corner_map = torch.zeros((batch_size, self.num_class, 4, feature_map_size[0], feature_map_size[1]),
    #                              dtype=torch.float32).to(gt_boxes.device)
    #     center_ops_cuda.draw_bev_all_gpu(gt_boxes, gt_corners, center_map, corner_map,
    #                                      self.model_cfg.TARGET_CONFIG.MIN_RADIUS,
    #                                      self.voxel_size[0], self.voxel_size[1],
    #                                      self.point_cloud_range[0], self.point_cloud_range[1],
    #                                      feature_map_stride, self.model_cfg.TARGET_CONFIG.GAUSSIAN_OVERLAP)
    #
    #     extend_gt_boxes = box_utils.enlarge_box3d(
    #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
    #     ).view(batch_size, -1, gt_boxes.shape[-1])
    #
    #     central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
    #     vote_targets_dict = self.assign_stack_targets(
    #         points=input_dict['votes'], gt_boxes=gt_boxes,
    #         set_ignore_flag=False, use_ball_constraint=True,
    #         ret_part_labels=False, ret_box_labels=True, central_radius=central_radius
    #     )
    #
    #     seed_targets_dict = {
    #         'seed_cls_labels_list': [],
    #         'seed_cls_targets_list': [],
    #         'gt_box_of_fg_seeds_list': []
    #     }
    #     assert len(input_dict['seeds_list']) == 1
    #     for i, seeds in enumerate(input_dict['seeds_list']):
    #         cur_seed_targets_dict = self.assign_stack_targets(
    #             points=seeds, gt_boxes=extend_gt_boxes,
    #             # points=seeds, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
    #             set_ignore_flag=False, use_ball_constraint=False,
    #             # set_ignore_flag=True, use_ball_constraint=False,
    #             ret_part_labels=False, ret_box_labels=False,
    #             # use_topk=True, topk=1, dim=2
    #         )
    #         seed_targets_dict['seed_cls_labels_list'].append(cur_seed_targets_dict['point_cls_labels'])
    #         seed_targets_dict['gt_box_of_fg_seeds_list'].append(cur_seed_targets_dict['gt_box_of_fg_points'])
    #
    #     x_bev_coords = spatial_features.indices.long()
    #     seed_targets_dict['seed_cls_targets_list'].append(
    #         center_map[x_bev_coords[:, 0], :, x_bev_coords[:, 1], x_bev_coords[:, 2]]
    #     )
    #
    #     targets_dict = {
    #         'vote_cls_labels': vote_targets_dict['point_cls_labels'],
    #         'vote_box_labels': vote_targets_dict['point_box_labels'],
    #         'gt_box_of_fg_votes': vote_targets_dict['gt_box_of_fg_points'],
    #         'seed_cls_labels_list': seed_targets_dict['seed_cls_labels_list'],
    #         'seed_cls_targets_list': seed_targets_dict['seed_cls_targets_list'],
    #         'gt_box_of_fg_seeds_list': seed_targets_dict['gt_box_of_fg_seeds_list'],
    #     }
    #     return targets_dict

    def voxel_knn_query_wrapper(self, points, point_coords, sp_tensor, stride, dim, query_range, radius, nsample,
                                return_xyz=False):
        """
        源码：
        points: 挑选后体素的世界坐标
        point_coords: 挑选后体素的原始坐标
        sp_tensor: 特征层特征
        """
        coords = sp_tensor.indices
        # 获得特征层的3d世界坐标下的体素中心
        voxel_xyz = common_utils.get_voxel_centers(
            coords[:, -dim:],
            downsample_times=stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            dim=dim
        )
        voxel_xyz = torch.cat([
            voxel_xyz,
            voxel_xyz.new_zeros((voxel_xyz.shape[0], 3 - dim))
        ], dim=-1)
        v2p_ind_tensor = common_utils.generate_voxels2pinds(coords.long(), sp_tensor.spatial_shape,
                                                            sp_tensor.batch_size)  # (B, H, W)全是-1
        v2p_ind_tensor = v2p_ind_tensor.view(v2p_ind_tensor.shape[0], -1, v2p_ind_tensor.shape[-2],
                                             v2p_ind_tensor.shape[-1])  # (B, 1, H, W)全是-1
        stride = point_coords.new_tensor(stride)
        point_coords = torch.cat([    # 原始体素[1440,1440]范围的转换为特征层范围
            point_coords[:, 0:1],
            point_coords.new_zeros((point_coords.shape[0], 3 - dim)),
            torch.div(point_coords[:, -dim:], stride, rounding_mode='floor')
        ], dim=-1)
        points = torch.cat([
            points[:, :dim],
            points.new_zeros((points.shape[0], 3 - dim))
        ], dim=-1)
        dist, idx, empty = voxel_knn_query(  # (1024,128) 每个point 采样 nsample个体素
            query_range,
            radius,
            nsample,
            voxel_xyz,  # 世界坐标特征层体素中心
            points.contiguous(),        # reference 体素世界坐标
            point_coords.int().contiguous(),        # # reference 体素特征层坐标
            v2p_ind_tensor
        )
        if return_xyz:
            return dist, idx, empty, voxel_xyz, sp_tensor.features, points
        else:
            return dist, idx, empty

    def get_voxel_features(self, batch_dict, src_name, reference_voxel, fps_indices, dim=2):
        cur_stride = batch_dict['multi_scale_2d_strides'][src_name]
        cur_sp_tensors = batch_dict['multi_scale_2d_features'][src_name]

        vote_candidates = reference_voxel['vote_candidates'] # 这是世界坐标
        pc_range = vote_candidates.new_tensor(self.point_cloud_range)
        voxel_size = vote_candidates.new_tensor(self.voxel_size)
        # 这是原始体素坐标[1440, 1440]范围的
        vote_candidate_coords = ((vote_candidates[:, 1:3] - pc_range[:2]) / voxel_size[:2]).to(torch.int64)  # bev坐标转换为pillar坐标
        vote_candidate_coords = torch.cat(
            [vote_candidates[:, 0:1].long(), torch.flip(vote_candidate_coords, dims=[-1])], dim=-1)  # [bs_idx, Z, Y, X]

        knn_cfg = self.voxel_knn_cfg[src_name]
        _, voxel_idx, voxel_empty, voxel_k_pos, voxel_k_feats, voxel_q_pos = self.voxel_knn_query_wrapper(
            vote_candidates[:, 1:3], vote_candidate_coords, cur_sp_tensors, cur_stride, dim,
            knn_cfg.QUERY_RANGE, knn_cfg.RADIUS, knn_cfg.NSAMPLE, return_xyz=True,
        )
        voxel_key_features = voxel_k_feats[voxel_idx.long()] \
            * (voxel_empty == 0).unsqueeze(-1)  # (K1+K2+..., T, C)
        voxel_key_pos_emb = (voxel_k_pos[voxel_idx.long()] - voxel_q_pos.unsqueeze(1)) \
            * (voxel_empty == 0).unsqueeze(-1)

        return voxel_key_features, voxel_key_pos_emb

    def forward(self, batch_dict):
        batch_dict['spatial_features'] = batch_dict['multi_scale_2d_features']['x_conv4']
        batch_dict['spatial_features_stride'] = batch_dict['multi_scale_2d_strides']['x_conv4']
        batch_size = batch_dict['batch_size']
        voxel_key_features_list = []
        voxel_key_pos_emb_list = []
        seeds_cls_list = []
        seeds_list = []
        vote_candidates_list = []
        votes_list = []
        vote_feature_list = []
        reference = None

        for k, src_name in enumerate(self.model_cfg.FEATURES_SOURCE):
            pillar_coords = batch_dict['multi_scale_2d_features'][src_name].indices
            pillar_features = batch_dict['multi_scale_2d_features'][src_name].features.contiguous()
            pillar_strides = batch_dict['multi_scale_2d_strides'][src_name]

            reference, fps_indices = self.queryinitial[k](pillar_coords, pillar_features, pillar_strides, batch_size)
            voxel_key_features, voxel_key_pos_emb = self.get_voxel_features(batch_dict, src_name, reference, fps_indices)

            voxel_key_pos_emb_list.append(voxel_key_pos_emb)
            seeds_list.append(reference['seeds'])
            seeds_cls_list.append(reference['seeds_cls'])
            vote_candidates_list.append(reference['vote_candidates'])  # revise (B*K, 4)
            votes_list.append(reference['votes'])       # revise (N, 4)
            vote_feature_list.append(reference['vote_features'])

            # 方法1 全部转换成256
            N, K, C = voxel_key_features.shape
            voxel_key_features = voxel_key_features.contiguous().view(-1, C)
            voxel_key_features = self.voxel_feat_trans[k](voxel_key_features)
            voxel_key_features = voxel_key_features.view(N, K, -1)
            voxel_key_features_list.append(voxel_key_features)

        # 只用最后特征图生成的reference计算anchor，但是计算loss时需要多尺度的分割得分

        key_feat = torch.cat(voxel_key_features_list, dim=-1).contiguous()
        N, K, C = key_feat.shape
        key_feat = key_feat.view(-1, C)
        key_feat = self.voxel_feat_reduce(key_feat)
        key_feat = key_feat.view(N, K, -1)

        vote_feat = vote_feature_list[-1]
        B, N, C = key_feat.shape
        features = self.transformer(
            src=key_feat.permute(1, 0, 2),     # [128, 128, 256]
            tgt=vote_feat.unsqueeze(1).permute(1, 0, 2),          # [1, 512, 56]
            # tgt=key_feat.permute(1, 0, 2),          # revise 改成knn的所有体素
            pos_res=voxel_key_pos_emb_list[-1].permute(1, 0, 2).unsqueeze(0)     # [1, 128, 128, 3]
        ).squeeze(0)
        # features = features.view()
        vote_candidates_last = vote_candidates_list[-1][:, :3]  # revise 不要z
        input_sp_tensor = spconv.SparseConvTensor(
            features=features,
            indices=vote_candidates_last.int(),
            spatial_shape=[200, 176],
            batch_size=batch_size
        )
        # fixme feature.shape [512, 256] 所以一个batch里只有128个点，要放进[200, 176]里，就变得非常稀疏
        dense_feature_map = input_sp_tensor.dense()
        features = self.bev_conv(dense_feature_map)

        batch_dict['votes'] = vote_candidates_list
        batch_dict['votes_list'] = votes_list
        batch_dict['seeds_list'] = seeds_list
        batch_dict['seeds_cls_list'] = seeds_cls_list
        batch_dict['seeds_cls_list'] = seeds_cls_list

        batch_dict['spatial_features_2d'] = features

        return batch_dict
        # vote_features = self.shared_conv(features)
        # vote_cls_preds = self.cls_conv(vote_features)
        # vote_box_preds = self.box_conv(vote_features)
        #
        # cls_preds = self.cls_conv(features)     # [B, 6, 180, 180]
        # box_preds = self.box_conv(features)     # [B, 42, 180, 180]
        #
        # self.forward_ret_dict['cls_preds'] = cls_preds
        # self.forward_ret_dict['box_preds'] = box_preds
        #
        # if self.conv_dir_cls is not None:
        #     dir_cls_preds = self.conv_dir_cls(features)
        #     dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        #     self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        # else:
        #     dir_cls_preds = None
        #
        # if self.training:
        #     targets_dict = self.assign_targets(
        #         gt_boxes=batch_dict['gt_boxes']
        #     )
        #     self.forward_ret_dict.update(targets_dict)
        #
        # if not self.training or self.predict_boxes_when_training:
        #     batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #         batch_size=batch_dict['batch_size'],
        #         cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        #     )
        #     batch_dict['batch_cls_preds'] = batch_cls_preds
        #     batch_dict['batch_box_preds'] = batch_box_preds
        #     batch_dict['cls_preds_normalized'] = False
        #
        # return batch_dict
