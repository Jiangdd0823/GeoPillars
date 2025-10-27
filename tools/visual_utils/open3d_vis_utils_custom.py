import open3d as o3d
import numpy as np
from PIL import Image
import torch

# 定义颜色映射表
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

    return vis

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, save_path=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # 创建离屏渲染器
    w, h = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)

    # 设置相机参数
    camera_params = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.HD1080p)
    renderer.setup_camera(camera_params, lookat=[0, 0, 0], eye=[-5, -5, -5], up=[0, 0, 1])

    material = o3d.visualization.rendering.MaterialRecord()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.shader = 'defaultLit'

    # 添加几何体
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])
    if point_colors is None:
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = o3d.utility.Vector3dVector(point_colors)
    renderer.scene.add_geometry("points", pts, material)

    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        renderer.scene.add_geometry("origin", axis_pcd, material)

    if gt_boxes is not None:
        vis = draw_box(renderer, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(renderer, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    # 渲染并保存图像
    img = renderer.render_to_image()
    if save_path is not None:
        o3d.io.write_image(save_path, img)
