import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import common_utils_pvtssd as common_utils
from ...utils import box_utils_pvtssd as box_utils
from ...ops.center_ops import center_ops_cuda
from ...utils import loss_utils_pvtssd as loss_utils
from ..dense_heads.point_head_template_pvtssd import PointHeadTemplate
from ...utils import box_coder_utils_pvtssd as box_coder_utils


class AnchorHeadAux(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, voxel_size, point_cloud_range, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        self.input_channels = input_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG # fixme 配置文件dense_head重新写
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.box_conv = nn.Conv2d(
            input_channels, self.model_cfg.CLS_FC,
            kernel_size=1
        )
        self.cls_conv = nn.Conv2d(
            input_channels, self.model_cfg.REG_FC,
            kernel_size=1
        )
        self.share_conv = nn.Conv2d(
            input_channels, self.model_cfg.CLS_FC,
            kernel_size=1
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        gt_boxes = input_dict['gt_boxes']
        batch_size = gt_boxes.shape[0]

        """ Aux loss """  # todo 这是对 vote 生成的128个reference点作损失，整个场景就取决于这128个reference点的相关性
        spatial_features = input_dict['spatial_features']           # fixme 这里是backbone3d出来的最后一层特征图
        spatial_features_stride = input_dict['spatial_features_stride']
        feature_map_size = spatial_features.spatial_shape
        feature_map_stride = spatial_features_stride
        # todo 这是用类似centerhead的
        gt_corners = box_utils.boxes_to_corners_3d(gt_boxes.view(-1, gt_boxes.shape[-1]))
        gt_corners = gt_corners[:, :4, :2].contiguous().view(batch_size, -1, 4, 2)
        center_map = torch.zeros((batch_size, self.num_class, feature_map_size[0], feature_map_size[1]),
                                 dtype=torch.float32).to(gt_boxes.device)
        corner_map = torch.zeros((batch_size, self.num_class, 4, feature_map_size[0], feature_map_size[1]),
                                 dtype=torch.float32).to(gt_boxes.device)
        center_ops_cuda.draw_bev_all_gpu(gt_boxes, gt_corners, center_map, corner_map,          # todo 生成center_map corner_map
                                         self.model_cfg.TARGET_CONFIG.MIN_RADIUS,
                                         self.voxel_size[0], self.voxel_size[1],
                                         self.point_cloud_range[0], self.point_cloud_range[1],
                                         feature_map_stride, self.model_cfg.TARGET_CONFIG.GAUSSIAN_OVERLAP)

        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)

        # fixme 现在是只用最后一层特征图生成的vote
        vote_targets_dict = self.assign_stack_targets(
            points=input_dict['votes'][-1], gt_boxes=gt_boxes,      # fixme 第一，input_dict['votes']是列表，要改；第二，votes只有xy，我的想法是在vote时直接预测一个高度z
            set_ignore_flag=False, use_ball_constraint=True,
            ret_part_labels=False, ret_box_labels=True, central_radius=central_radius
        )

        seed_targets_dict = {
            'seed_cls_labels_list': [],
            'seed_cls_targets_list': [],
            'gt_box_of_fg_seeds_list': []
        }
        # assert len(input_dict['seeds_list']) == 1   # fixme 只用了最后一层
        # for i, seeds in enumerate(input_dict['seeds_list']):
        for i, seeds in enumerate([input_dict['seeds_list'][-1]]):
            cur_seed_targets_dict = self.assign_stack_targets(
                points=seeds, gt_boxes=extend_gt_boxes,
                # points=seeds, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=False, use_ball_constraint=False,
                # set_ignore_flag=True, use_ball_constraint=False,
                ret_part_labels=False, ret_box_labels=False,
                # use_topk=True, topk=1, dim=2
            )
            seed_targets_dict['seed_cls_labels_list'].append(cur_seed_targets_dict['point_cls_labels'])
            seed_targets_dict['gt_box_of_fg_seeds_list'].append(cur_seed_targets_dict['gt_box_of_fg_points'])

        x_bev_coords = spatial_features.indices.long()
        seed_targets_dict['seed_cls_targets_list'].append(
            center_map[x_bev_coords[:, 0], :, x_bev_coords[:, 1], x_bev_coords[:, 2]]
        )

        targets_dict = {
            'vote_cls_labels': vote_targets_dict['point_cls_labels'],
            'vote_box_labels': vote_targets_dict['point_box_labels'],
            'gt_box_of_fg_votes': vote_targets_dict['gt_box_of_fg_points'],
            'seed_cls_labels_list': seed_targets_dict['seed_cls_labels_list'],
            'seed_cls_targets_list': seed_targets_dict['seed_cls_targets_list'],
            'gt_box_of_fg_seeds_list': seed_targets_dict['gt_box_of_fg_seeds_list'],
        }
        return targets_dict

    def get_seed_reg_loss(self, tb_dict=None):
        seed_cls_labels_list = self.forward_ret_dict['seed_cls_labels_list']
        gt_box_of_fg_seeds_list = self.forward_ret_dict['gt_box_of_fg_seeds_list']
        votes_list = self.forward_ret_dict['votes_list']
        seed_center_loss_list = []
        for i in range(len(votes_list)):
            seed_center_loss, tb_dict = self.get_seed_single_reg_loss(
                votes_list[i],
                seed_cls_labels_list[i],
                gt_box_of_fg_seeds_list[i],
                i,
                tb_dict
            )
            seed_center_loss_list.append(seed_center_loss)
        return sum(seed_center_loss_list), tb_dict

    def get_seed_single_cls_loss(self, point_cls_preds, point_cls_labels, index, tb_dict=None):

        if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center':
            assert len(point_cls_preds) == len(point_cls_labels)
            point_loss_cls = self.seed_cls_loss_func(self.sigmoid(point_cls_preds), point_cls_labels)
        else:
            positives = point_cls_labels > 0
            negatives = point_cls_labels == 0
            cls_weights = negatives * 1.0 + positives * 1.0
            pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.AUX_CLS_POS_NORM else cls_weights.sum()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            num_class = 1
            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels > 0).unsqueeze(-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.aux_cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
            point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seed_cls_weight_list'][index]
        tb_dict.update({
            f'seed_cls_loss_{index}': point_loss_cls.item(),
        })
        return point_loss_cls, tb_dict

    def get_seed_cls_loss(self, tb_dict=None):
        seed_cls_labels_list = self.forward_ret_dict[
            'seed_cls_targets_list'] if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center' else self.forward_ret_dict[
            'seed_cls_labels_list']
        seeds_cls_list = self.forward_ret_dict['seeds_cls_list']
        seed_cls_loss_list = []
        for i in range(len(seeds_cls_list)):
            seed_cls_loss, tb_dict = self.get_seed_single_cls_loss(
                seeds_cls_list[i],
                seed_cls_labels_list[i],
                i,
                tb_dict
            )
            seed_cls_loss_list.append(seed_cls_loss)
        return sum(seed_cls_loss_list), tb_dict

    def get_vote_cls_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['vote_cls_labels']
        point_cls_preds = self.forward_ret_dict['vote_cls_preds']

        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        cls_weights = negatives * 1.0 + positives * 1.0
        pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.CLS_POS_NORM else cls_weights.sum()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        if 'WithCenterness' in self.model_cfg.LOSS_CONFIG.CLS_LOSS:
            votes = self.forward_ret_dict['votes'].detach()
            gt_box_of_fg_votes = self.forward_ret_dict['gt_box_of_fg_votes']
            pos_centerness = box_utils.generate_centerness_mask(votes[positives][:, 1:], gt_box_of_fg_votes)
            centerness_mask = positives.new_zeros(positives.shape).float()
            centerness_mask[positives] = pos_centerness
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1)

        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_cls_weight']
        tb_dict.update({
            'vote_cls_loss': point_loss_cls.item(),
            'vote_pos_num': int(positives.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return point_loss_cls, tb_dict

    def get_vote_reg_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['vote_box_labels']
        point_box_preds = self.forward_ret_dict['vote_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        xyzlwh_preds = point_box_preds[:, :6]
        xyzlwh_labels = point_box_labels[:, :6]
        point_loss_xyzlwh = self.reg_loss_func(xyzlwh_preds, xyzlwh_labels, reg_weights).sum() \
                            * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][0]

        angle_bin_num = self.box_coder.angle_bin_num
        dir_cls_preds = point_box_preds[:, 6:6 + angle_bin_num]
        dir_cls_labels = point_box_labels[:, 6:6 + angle_bin_num]
        point_loss_dir_cls = F.cross_entropy(dir_cls_preds, dir_cls_labels.argmax(dim=-1), reduction='none')
        point_loss_dir_cls = (point_loss_dir_cls * reg_weights).sum() \
                             * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][1]

        dir_res_preds = point_box_preds[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
        dir_res_labels = point_box_labels[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]

        dir_res_preds = torch.sum(dir_res_preds * dir_cls_labels, dim=-1)
        dir_res_labels = torch.sum(dir_res_labels * dir_cls_labels, dim=-1)
        point_loss_dir_res = self.reg_loss_func(dir_res_preds, dir_res_labels, weights=reg_weights)
        point_loss_dir_res = point_loss_dir_res.sum() \
                             * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][2]

        point_loss_velo = 0
        if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
            point_loss_velo = self.reg_loss_func(
                point_box_preds[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
                point_box_labels[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
                reg_weights
            ).sum()
            tb_dict.update({
                'vote_reg_velo_loss': point_loss_velo.item()
            })

        point_loss_box = point_loss_xyzlwh + point_loss_dir_cls + point_loss_dir_res + point_loss_velo
        point_loss_box = point_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_reg_weight']
        tb_dict.update({
            'vote_reg_loss': point_loss_box.item(),
            'vote_reg_xyzlwh_loss': point_loss_xyzlwh.item(),
            'vote_reg_dir_cls_loss': point_loss_dir_cls.item(),
            'vote_reg_dir_res_loss': point_loss_dir_res.item(),
        })
        return point_loss_box, tb_dict

    def get_vote_corner_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['gt_box_of_fg_votes']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7],
            p=self.model_cfg.LOSS_CONFIG.CORNER_LOSS_TYPE
        ).mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_corner_weight']
        tb_dict.update({'vote_corner_loss': loss_corner.item()})
        return loss_corner, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        seed_reg_loss, tb_dict = self.get_seed_reg_loss(tb_dict)
        seed_cls_loss, tb_dict = self.get_seed_cls_loss(tb_dict)
        vote_cls_loss, tb_dict = self.get_vote_cls_loss(tb_dict)
        vote_reg_loss, tb_dict = self.get_vote_reg_loss(tb_dict)
        vote_corner_loss, tb_dict = self.get_vote_corner_loss(tb_dict)
        point_loss = seed_reg_loss + seed_cls_loss + vote_cls_loss + vote_reg_loss + vote_corner_loss
        return point_loss, tb_dict

    def forward(self, batch_dict):
        features = batch_dict['spatial_features_2d'] # fixme 暂时用'features'代替

        features = self.share_conv(features)
        cls_preds = self.cls_conv(features)     # fixme self.cls_conv要修改  cls output_channels=num_class
        box_preds = self.box_conv(features)     # fixme self.box_conv要修改  box output_channels=self.box_coder.code_size

        ret_dict = {
            'vote_cls_preds': cls_preds,
            'vote_box_preds': box_preds,
            'votes': batch_dict['votes'],
            'votes_list': batch_dict['votes_list'],
            'seeds_list': batch_dict['seeds_list'],
            'seeds_cls_list': batch_dict['seeds_cls_list'],
            'batch_size': batch_dict['batch_size']
        }
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.PREDICT_BOXES:
            # fixme 这是reference vote 128个点 预测box，所以cls_preds，box_preds是bev的
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(      # fixme batch_dict 的keys
                points=ret_dict['votes'][-1][:, 1:4],               # fixme
                point_cls_preds=cls_preds, point_box_preds=box_preds
            )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = ret_dict['votes'][:, 0].contiguous()
            batch_dict['cls_preds_normalized'] = False

            ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict
