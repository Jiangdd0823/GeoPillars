from .detector3d_template import Detector3DTemplate
from .. import mm_backbone

class test_custom(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'mm_backbone',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def build_mm_backbone(self, model_info_dict):
        if self.model_cfg.get('MM_BACKBONE', None) is None:
            return None, model_info_dict
        mm_backbone_model = mm_backbone.__all__[self.model_cfg.MM_BACKBONE.NAME](
            model_cfg=self.model_cfg.MM_BACKBONE,
            num_class=self.num_class,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )
        model_info_dict['module_list'].append(mm_backbone_model)
        model_info_dict['num_bev_features'] = self.model_cfg.MM_BACKBONE.NUM_BEV_FEATURES
        return mm_backbone_model, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
