import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate
from ...utils import loss_utils
from ...ops.center_ops import center_ops_cuda
# from .point_head_template_pvtssd import PointHeadTemplate
from ...utils import box_coder_utils_pvtssd
from ...utils import common_utils_pvtssd
from ...utils import loss_utils_pvtssd
from ...utils import box_utils_pvtssd
from ...ops.pvtssd.roiaware_pool3d import roiaware_pool3d_utils


class AnchorHeadSingle_Aux(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        # fixme 以下follow pvtssd 的token headLF
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder_aux = getattr(box_coder_utils_pvtssd, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

        # fixme 以下follow pvtssd 的token head
        aux_cls_loss_type = losses_cfg.get('AUX_CLS_LOSS', None)
        self.aux_cls_loss_func = getattr(loss_utils_pvtssd, aux_cls_loss_type)(
            **losses_cfg.get('AUX_CLS_LOSS_CONFIG', {})
        )
        self.seed_cls_loss_func = loss_utils_pvtssd.FocalLossCenterNet()
        self.seed_reg_loss_func = loss_utils_pvtssd.WeightedSmoothL1Loss()

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
        offset_ceil_loss = +self.forward_ret_dict['offset_ceil_loss'] * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS_AUX['offset_ceil_weight']
        return sum(seed_center_loss_list)+offset_ceil_loss, tb_dict

    def get_seed_single_reg_loss(self, votes, seed_cls_labels, gt_box_of_fg_seeds, index, tb_dict=None):

        # vote = common_utils_pvtssd.get_voxel_centers(votes[:, 1:], 4, self.voxel_size, self.point_cloud_range, dim=2)
        # votes = torch.cat([votes[:, 0].unsqueeze(-1), vote], dim=-1)
        pos_mask = seed_cls_labels > 0
        seed_center_labels = gt_box_of_fg_seeds[:, 0:3]
        # todo 这个votes是votes_list的，经过vote_layer后的voxel_center那个
        seed_center_loss = self.seed_reg_loss_func(
            votes[pos_mask][:, 1:], seed_center_labels[:, :2]       # revise 改为只和xy
        ).sum(dim=-1).mean()
        seed_center_loss = seed_center_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS_AUX['seed_reg_weight_list'][index]

        tb_dict.update({
            f'seed_reg_loss_{index}': seed_center_loss.item(),
            f'seed_pos_num_{index}': int(pos_mask.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return seed_center_loss, tb_dict

    def get_seed_cls_loss(self, tb_dict=None):
        seed_cls_labels_list = self.forward_ret_dict['seed_cls_targets_list'] if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center' else self.forward_ret_dict['seed_cls_labels_list']
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

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS_AUX['seed_cls_weight_list'][index]
        tb_dict.update({
            f'seed_cls_loss_{index}': point_loss_cls.item(),
        })
        return point_loss_cls, tb_dict

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0, use_topk=False,
                             topk=1, dim=3):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        # assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(points.shape) == 2 and points.shape[1] == 3, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], self.box_coder_aux.code_size)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        gt_box_of_fg_points_list = []
        gt_box_idx_of_fg_points_list = []
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            # points_single = points[bs_mask][:, 1:4]
            points_single = points[bs_mask][:, 1:]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())

            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt >= 0 and cur_gt[cnt, -1] == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            if cur_gt.__len__() == 0:
                continue

            # if use_topk_2d:

            # fixme 只计算bev视角的xy坐标下的(N, 1, 2) - (1, M, 2) -> (N, M, 2) -> (N, M)
            dist = (points_single.unsqueeze(1) - cur_gt[None, :, :2])[..., :dim].norm(dim=-1)
            _, indices = torch.topk(dist, topk, dim=0, largest=False)  # (K, M)
            box_idxs_of_pts = points_single.new_full((points_single.shape[0],), -1, dtype=torch.long)
            box_idxs_of_pts[indices] = torch.arange(indices.shape[-1], device=indices.device,
                                                    dtype=torch.long).unsqueeze(0).repeat(topk, 1)

            # if use_topk:
                # (N, 1, 3) - (1, M, 3) -> (N, M, 3) -> (N, M)
                # dist = (points_single.unsqueeze(1) - cur_gt[None, :, :3])[..., :dim].norm(dim=-1)
                # _, indices = torch.topk(dist, topk, dim=0, largest=False)  # (K, M)
                # box_idxs_of_pts = points_single.new_full((points_single.shape[0],), -1, dtype=torch.long)
                # box_idxs_of_pts[indices] = torch.arange(indices.shape[-1], device=indices.device,
                #                                         dtype=torch.long).unsqueeze(0).repeat(topk, 1)
            # else:
            #     # fixme roiaware_pool3d_utils.points_in_boxes_gpu 需要3d坐标输入
            #     box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            #         points_single.unsqueeze(dim=0), cur_gt[None, :, 0:7].contiguous()
            #     ).long().squeeze(dim=0)

            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
                fg_flag &= point_cls_labels_single >= 0
            elif use_ball_constraint:
                box_centers = cur_gt[box_idxs_of_pts][:, 0:3].clone()
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
                ignore_flag = fg_flag ^ box_fg_flag
                point_cls_labels_single[ignore_flag] = -1
                fg_flag &= point_cls_labels_single >= 0
            else:
                fg_flag = box_fg_flag

            gt_box_of_fg_points = cur_gt[box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            gt_box_of_fg_points_list.append(gt_box_of_fg_points)
            gt_box_idx_of_fg_points_list.append(box_idxs_of_pts[fg_flag] + k * cur_gt.shape[1])

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), self.box_coder_aux.code_size))
                fg_point_box_labels = self.box_coder_aux.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils_pvtssd.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels,
            'gt_box_of_fg_points': torch.cat(gt_box_of_fg_points_list, dim=0),
            'gt_box_idx_of_fg_points': torch.cat(gt_box_idx_of_fg_points_list, dim=0),
        }
        return targets_dict

    def assign_targets_aux(self, input_dict):
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

        """ Aux loss """
        spatial_features = input_dict['multi_scale_2d_features']['x_conv4_sparse']
        spatial_features_stride = input_dict['multi_scale_2d_strides']['x_conv4']
        feature_map_size = spatial_features.spatial_shape
        feature_map_stride = spatial_features_stride

        gt_corners = box_utils_pvtssd.boxes_to_corners_3d(gt_boxes.view(-1, gt_boxes.shape[-1]))
        gt_corners = gt_corners[:, :4, :2].contiguous().view(batch_size, -1, 4, 2)
        center_map = torch.zeros((batch_size, self.num_class, feature_map_size[0], feature_map_size[1]),
                                 dtype=torch.float32).to(gt_boxes.device)
        corner_map = torch.zeros((batch_size, self.num_class, 4, feature_map_size[0], feature_map_size[1]),
                                 dtype=torch.float32).to(gt_boxes.device)
        center_ops_cuda.draw_bev_all_gpu(gt_boxes, gt_corners, center_map, corner_map,
                                         self.model_cfg.TARGET_CONFIG.MIN_RADIUS,
                                         self.voxel_size[0], self.voxel_size[1],
                                         self.point_cloud_range[0], self.point_cloud_range[1],
                                         feature_map_stride, self.model_cfg.TARGET_CONFIG.GAUSSIAN_OVERLAP)

        extend_gt_boxes = box_utils_pvtssd.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)

        # fixme input_dict['votes']        votes 是那512个关键点， 可取消
        # vote_targets_dict = self.assign_stack_targets(
        #     points=input_dict['votes'], gt_boxes=gt_boxes,
        #     set_ignore_flag=False, use_ball_constraint=True,
        #     ret_part_labels=False, ret_box_labels=True, central_radius=central_radius
        # )

        seed_targets_dict = {
            'seed_cls_labels_list': [],
            'seed_cls_targets_list': [],
            'gt_box_of_fg_seeds_list': [],
            'votes_list': []
        }

        # fixme input_dict['seeds_list']        seeds 是特征图体素坐标转换成voxel center
        query_layer = input_dict['query_layer']
        if isinstance(query_layer, list):
            coords_list = input_dict['vote']
            for i, coords in enumerate(coords_list):
                strides = input_dict['multi_scale_2d_strides'][f'x_conv{query_layer[i]}']
                voxel_center = common_utils_pvtssd.get_voxel_centers(
                    coords[:, 1:], strides, self.voxel_size, self.point_cloud_range, dim=2)
                seeds = torch.cat([coords[:, 0].unsqueeze(-1), voxel_center], dim=-1)

                cur_seed_targets_dict = self.assign_stack_targets(
                    points=seeds, gt_boxes=extend_gt_boxes,
                    set_ignore_flag=False, use_ball_constraint=False,
                    ret_part_labels=False, ret_box_labels=False,
                )
                seed_targets_dict['seed_cls_labels_list'].append(cur_seed_targets_dict['point_cls_labels'])
                seed_targets_dict['gt_box_of_fg_seeds_list'].append(cur_seed_targets_dict['gt_box_of_fg_points'])
                seed_targets_dict['votes_list'].append(seeds)

            x_bev_coords = spatial_features.indices.long()
            seed_targets_dict['seed_cls_targets_list'].append(
                center_map[x_bev_coords[:, 0], :, x_bev_coords[:, 1], x_bev_coords[:, 2]]
            )
        else:
            # coords = input_dict['multi_scale_2d_features'][f'x_conv{query_layer}'].indices
            coords = input_dict['vote']
            strides = input_dict['multi_scale_2d_strides'][f'x_conv{query_layer}']
            voxel_center = common_utils_pvtssd.get_voxel_centers(
                coords[0][:, 1:], strides, self.voxel_size, self.point_cloud_range, dim=2)
            seeds = torch.cat([coords[:, 0].unsqueeze(-1), voxel_center], dim=-1)
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
            seed_targets_dict['votes_list'].append(seeds)

            x_bev_coords = spatial_features.indices.long()
            seed_targets_dict['seed_cls_targets_list'].append(
                center_map[x_bev_coords[:, 0], :, x_bev_coords[:, 1], x_bev_coords[:, 2]]
            )

        targets_dict = {
            # 'vote_cls_labels': vote_targets_dict['point_cls_labels'],
            # 'vote_box_labels': vote_targets_dict['point_box_labels'],
            # 'gt_box_of_fg_votes': vote_targets_dict['gt_box_of_fg_points'],
            'seed_cls_labels_list': seed_targets_dict['seed_cls_labels_list'],
            'seed_cls_targets_list': seed_targets_dict['seed_cls_targets_list'],
            'gt_box_of_fg_seeds_list': seed_targets_dict['gt_box_of_fg_seeds_list'],
            'votes_list': seed_targets_dict['votes_list']
        }
        return targets_dict

    def get_loss(self):
        # 普通rpn loss
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)

        # fixme aux loss
        seed_reg_loss, tb_dict = self.get_seed_reg_loss(tb_dict)        # revise 删除seed的正则化
        # seed_cls_loss, tb_dict = self.get_seed_cls_loss(tb_dict)

        rpn_loss = cls_loss + box_loss
        total_loss = cls_loss + box_loss + seed_reg_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        tb_dict['total_loss'] = total_loss.item()
        return total_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

            # revise 添加了aux
            targets_aux_dict = self.assign_targets_aux(data_dict)
            self.forward_ret_dict.update(targets_aux_dict)
            self.forward_ret_dict['batch_size'] = data_dict['batch_size']
            self.forward_ret_dict['offset_ceil_loss'] = data_dict['offset_ceil_loss']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

