import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    # from visual_utils import open3d_vis_utils_custom as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
import open3d
from visual_utils import open3d_vis_utils as V
# from visual_utils import open3d_vis_utils_custom as V
OPEN3D_FLAG = True
import numpy as np
import torch
import json

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,

        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def tensor_to_list(tensor):
    """将 PyTorch 张量转换为 Python 列表"""
    return tensor.cpu().detach().numpy().tolist()


def save_predictions_to_json(pred_dicts, filename):
    """将预测结果保存到 JSON 文件中"""
    # 创建一个新的字典来存储可序列化的内容
    serializable_pred_dicts = []

    for pred_dict in pred_dicts:
        serializable_pred_dict = {}
        for key, value in pred_dict.items():
            if isinstance(value, torch.Tensor):
                # 如果值是张量，则转换为列表
                serializable_pred_dict[key] = tensor_to_list(value)
            elif isinstance(value, dict):
                # 如果值是字典，则递归处理
                serializable_pred_dict[key] = {k: tensor_to_list(v) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
            else:
                # 其他类型直接复制
                serializable_pred_dict[key] = value
        serializable_pred_dicts.append(serializable_pred_dict)

    # 将可序列化的预测结果写入 JSON 文件
    with open(filename, 'w') as f:
        json.dump(serializable_pred_dicts, f, indent=4)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    # parser.add_argument('--pred_path', type=str, default='../ouput_image/predictions.json')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_dicts[0]['points'] = data_dict['points'][:, 1:]
            if 'gt_box' in data_dict:
                pred_dicts[0]['gt_boxs'] = data_dict['']
            file_id = args.data_path[-10:-4]
            # file_id = idx
            # save_predictions_to_json(pred_dicts, f'../output_image/predictions_pvt_{file_id}.json')

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                # save_path='../output_image/output.png',
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


def visual():

    def list_to_tensor(data):
        """将 Python 列表转换为 PyTorch 张量"""
        return torch.tensor(data)

    def load_predictions_from_json(filename):
        """从 JSON 文件中加载预测结果，并将其转换为 PyTorch 张量"""
        with open(filename, 'r') as f:
            predictions = json.load(f)

        # 创建一个新的列表来存储转换后的张量
        tensor_predictions = []

        for pred_dict in predictions:
            tensor_pred_dict = {}
            for key, value in pred_dict.items():
                if isinstance(value, list):
                    # 如果值是列表，则转换为张量
                    tensor_pred_dict[key] = list_to_tensor(value)
                elif isinstance(value, dict):
                    # 如果值是字典，则递归处理
                    tensor_pred_dict[key] = {k: list_to_tensor(v) if isinstance(v, list) else v for k, v in
                                             value.items()}
                else:
                    # 其他类型直接复制
                    tensor_pred_dict[key] = value
            tensor_predictions.append(tensor_pred_dict)

        return tensor_predictions

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick visual of pred-------------------------')
    # 示例：从 JSON 文件中加载预测结果
    pred_dicts = load_predictions_from_json(args.pred_path)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
        )


if __name__ == '__main__':
    main()
    # visual()
