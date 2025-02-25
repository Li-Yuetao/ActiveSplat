from enum import Enum
from typing import Dict, Union, Tuple

import cv2
import torch
import numpy as np
import open3d as o3d
import quaternion
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Pose
from utils import OPENCV_TO_OPENGL

class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum
    
class GaussianPacket:
    def __init__(
        self,
        params=None,
        current_frame=None,
        finish=False
    ):
        self.has_gaussians = False
        if params is not None:
            self.has_gaussians = True
            self.params = params
            
            self.current_frame = current_frame

def vfov_to_hfov(vfov_deg, height, width):
    # http://paulbourke.net/miscellaneous/lens/
    return np.rad2deg(
        2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
    )

def rgbd_to_pointcloud(
    rgb_data:np.ndarray,
    depth_data:np.ndarray,
    pose_data:np.ndarray,
    camera_intrinsics_tensor:o3d.core.Tensor,
    depth_scale:float,
    depth_max:float,
    device:o3d.core.Device) -> o3d.t.geometry.PointCloud:
    if rgb_data.dtype == np.float32:
        rgb_data_uint8 = (rgb_data * 255).astype(np.uint8)
    elif rgb_data.dtype == np.uint8:
        rgb_data_uint8 = rgb_data
    else:
        raise ValueError(f"Invalid rgb_data dtype: {rgb_data.dtype}")
    if depth_data.dtype == np.float32:
        depth_data_uint16 = (depth_data * 1000).astype(np.uint16)
    elif depth_data.dtype == np.uint16:
        depth_data_uint16 = depth_data
    else:
        raise ValueError(f"Invalid depth_data dtype: {depth_data.dtype}")
    rgb_image = o3d.t.geometry.Image(o3d.core.Tensor(rgb_data_uint8, device=device))
    depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth_data_uint16, device=device))
    rgbd_image = o3d.t.geometry.RGBDImage(rgb_image, depth_image)
    current_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics_tensor,
        o3d.core.Tensor(np.linalg.inv(OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL), device=device),
        depth_scale=depth_scale,
        depth_max=depth_max)
    return current_pcd.cpu()

def pose_to_matrix(pose:Pose) -> np.ndarray:
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    transform_matrix[:3, :3] = quaternion.as_rotation_matrix(
        quaternion.from_float_array([
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z]))
    return transform_matrix

def matrix_to_pose(matrix: np.ndarray) -> Pose:
    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 transformation matrix")

    pose = Pose()
    pose.position.x = matrix[0, 3]
    pose.position.y = matrix[1, 3]
    pose.position.z = matrix[2, 3]
    rot_matrix = matrix[:3, :3]
    q = quaternion.from_rotation_matrix(rot_matrix)
    pose.orientation.w = q.w
    pose.orientation.x = q.x
    pose.orientation.y = q.y
    pose.orientation.z = q.z
    return pose

def rotation_matrix_from_vectors(vec_start:np.ndarray, vec_end:np.ndarray) -> np.ndarray:
    """ Find the rotation matrix that aligns vec_start to vec_end
    :param vec_start: A 3d "source" vector
    :param vec_end: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec_start, aligns it with vec_end.
    """
    a, b = (vec_start / np.linalg.norm(vec_start)), (vec_end / np.linalg.norm(vec_end))
    v = np.cross(a, b)
    if np.linalg.norm(v) == 0:
        return np.eye(3)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def translations_world_to_topdown(
    translations_world:np.ndarray,
    topdown_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray]]],
    translation_topdown_dtype:np.dtype) -> np.ndarray:
    translations_world_ = translations_world.reshape(-1, 3)
    translations_topdown = np.vstack([
        translations_world_[:, topdown_info['world_dim_index'][0]] - topdown_info['world_2d_bbox'][0][0],
        translations_world_[:, topdown_info['world_dim_index'][1]] + topdown_info['world_2d_bbox'][1][1]]).T
    if np.issubdtype(translation_topdown_dtype, np.integer):
        translations_topdown = (translations_topdown // topdown_info['meter_per_pixel']).astype(translation_topdown_dtype)
        assert np.all(0 <= translations_topdown) and np.all(translations_topdown < topdown_info['grid_map_shape']), f"Invalid translations: {translations_topdown}"
    elif np.issubdtype(translation_topdown_dtype, np.floating):
        translations_topdown = (translations_topdown / topdown_info['meter_per_pixel']).astype(translation_topdown_dtype)
    else:
        raise ValueError(f"Invalid dtype: {translation_topdown_dtype}")
    
    return translations_topdown

def c2w_world_to_topdown(
    c2w_world:np.ndarray,
    topdown_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray]]],
    height_direction:Tuple[np.ndarray, np.ndarray],
    translation_topdown_dtype:np.dtype,
    camera_to_center_front_back:float=0,
    need_pitch=False) -> Tuple[np.ndarray, np.ndarray]:
    
    c2w_world_new = c2w_world @ OPENCV_TO_OPENGL
    z_axis = c2w_world_new[:3, 2]
    pitch_angle = -np.degrees(np.arcsin(z_axis[1] / np.linalg.norm(z_axis)))
    
    coordinates_names = ['x', 'y', 'z']
    euler_rule = coordinates_names[height_direction[0]] +\
        coordinates_names[(height_direction[0] - 1) % 3] +\
            coordinates_names[height_direction[0]]
    
    rotation_vector_world:Rotation = Rotation.from_matrix(c2w_world[:3, :3])
    rotation_vector_world = rotation_vector_world.as_euler(euler_rule, degrees=False)
    rotation_theta_topdown = rotation_vector_world[0] + rotation_vector_world[-1] + np.pi / 2
    
    rotation_vector_topdown = np.array([
        np.cos(-rotation_theta_topdown),
        np.sin(-rotation_theta_topdown)])
    translation_topdown = translations_world_to_topdown(
        c2w_world[:3, 3],
        topdown_info,
        translation_topdown_dtype).reshape(-1)
    
    if need_pitch:
        return rotation_vector_topdown, translation_topdown, pitch_angle
    else:
        return rotation_vector_topdown, translation_topdown

def c2w_topdown_to_world(
    translation_topdown:np.ndarray,
    topdown_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray]]],
    height_value:float) -> np.ndarray:
    translation_world = np.ones(3) * height_value
    translation_world[topdown_info['world_dim_index'][0]] = translation_topdown[0] * topdown_info['meter_per_pixel'] + topdown_info['world_2d_bbox'][0][0]
    translation_world[topdown_info['world_dim_index'][1]] = translation_topdown[1] * topdown_info['meter_per_pixel'] - topdown_info['world_2d_bbox'][1][1]
    return translation_world.astype(np.float64)

def config_topdown_info(
    height_direction:Tuple[int, int],
    world_dim_index:Tuple[int, int],
    world_shape:Tuple[float, float],
    world_center:Tuple[float, float],
    meter_per_pixel:float,
    agent_foot:np.ndarray,
    agent_sensor:np.ndarray,
    agent_head:np.ndarray,
    body_sample_num:int,
    head_sample_num:int,
) -> Dict[str, Union[Tuple[int, int], int, np.ndarray, Tuple[float, float], Tuple[np.ndarray, np.ndarray], torch.Tensor]]:
    grid_map_shape = (
        int(np.ceil(world_shape[0] / meter_per_pixel)),
        int(np.ceil(world_shape[1] / meter_per_pixel)))
    topdown_height_array = np.hstack((
        np.linspace(
            agent_foot + 0.1 * (agent_sensor - agent_foot),
            agent_sensor,
            body_sample_num),
        np.linspace(
            agent_sensor,
            agent_head,
            head_sample_num)))
    topdown_panel_array = (
        np.arange(grid_map_shape[0]) * meter_per_pixel,
        np.arange(grid_map_shape[1]) * meter_per_pixel)
    topdown_panel_array = (
        topdown_panel_array[0] - np.average(topdown_panel_array[0]) + world_center[0],
        topdown_panel_array[1] - np.average(topdown_panel_array[1]) + world_center[1])
    topdown_world_array = {
        height_direction[0]: torch.from_numpy(topdown_height_array),
        world_dim_index[0]: torch.from_numpy(topdown_panel_array[0]),
        world_dim_index[1]: torch.from_numpy(topdown_panel_array[1])}
    return {
        'grid_map_shape': grid_map_shape,
        'body_sample_num': body_sample_num,
        'height_array': topdown_height_array,
        'panel_array': topdown_panel_array,
        'world_2d_bbox': (
            (topdown_panel_array[0][0] - meter_per_pixel / 2,
             topdown_panel_array[0][-1] + meter_per_pixel / 2),
            (topdown_panel_array[1][0] - meter_per_pixel / 2,
             topdown_panel_array[1][-1] + meter_per_pixel / 2)),
        'world_voxel_grid': torch.stack(
            torch.meshgrid(
                topdown_world_array[0],
                topdown_world_array[1],
                topdown_world_array[2],
                indexing='ij'),
            dim=-1).float()}

def visualize_agent(
    topdown_map:cv2.Mat,
    meter_per_pixel:float,
    agent_translation:np.ndarray,
    agent_rotation_vector:np.ndarray,
    agent_color:Tuple[int, int, int],
    agent_radius:float,
    rotation_vector_color:Tuple[int, int, int],
    rotation_vector_thickness:int,
    rotation_vector_length:float,
    resize_scale:float=1.0) -> cv2.Mat:
    topdown_map_cv2 = topdown_map.copy()
    cv2.arrowedLine(
        topdown_map_cv2,
        np.int32(agent_translation * resize_scale),
        np.int32((agent_translation + rotation_vector_length * agent_rotation_vector) * resize_scale),
        rotation_vector_color,
        int(rotation_vector_thickness * resize_scale))
    cv2.circle(
        topdown_map_cv2,
        np.int32(agent_translation * resize_scale),
        int(agent_radius / meter_per_pixel * resize_scale),
        agent_color,
        -1)
    return topdown_map_cv2

class PoseChangeType(Enum):
    NONE = 0
    TRANSLATION = 1
    ROTATION = 2
    BOTH = 3

def is_pose_changed(
    frame_c2w_old:np.ndarray,
    frame_c2w_new:np.ndarray,
    translation_threshold:float,
    rotation_threshold:float) -> PoseChangeType:
    assert frame_c2w_old is not None, "frame_c2w_old is None"
    assert frame_c2w_new is not None, "frame_c2w_new is None"
    frame_c2w_diff_translation = np.linalg.norm(frame_c2w_new[:3, 3] - frame_c2w_old[:3, 3])
    frame_c2w_diff_rotation = np.dot(frame_c2w_new[:3, :3], np.linalg.inv(frame_c2w_old[:3, :3]))
    frame_c2w_diff_rotation = np.arccos((np.trace(frame_c2w_diff_rotation) - 1) / 2)
    frame_c2w_diff_rotation = np.degrees(frame_c2w_diff_rotation)
    if frame_c2w_diff_translation > translation_threshold and frame_c2w_diff_rotation > rotation_threshold:
        rospy.logdebug(f'Get new c2w\nc2w_diff_translation: {frame_c2w_diff_translation}\nc2w_diff_rotation: {frame_c2w_diff_rotation}')
        return PoseChangeType.BOTH
    elif frame_c2w_diff_translation > translation_threshold:
        rospy.logdebug(f'Get new c2w\nc2w_diff_translation: {frame_c2w_diff_translation}\nc2w_diff_rotation: {frame_c2w_diff_rotation}')
        return PoseChangeType.TRANSLATION
    elif frame_c2w_diff_rotation > rotation_threshold:
        rospy.logdebug(f'Get new c2w\nc2w_diff_translation: {frame_c2w_diff_translation}\nc2w_diff_rotation: {frame_c2w_diff_rotation}')
        return PoseChangeType.ROTATION
    else:
        return PoseChangeType.NONE
    
def get_horizon_bound_topdown(
    horizon_bound_min:np.ndarray,
    horizon_bound_max:np.ndarray,
    topdown_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray]]],
    height_direction:Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    horizon_bound_min_c2w_world = np.eye(4)
    horizon_bound_min_c2w_world[:3, 3] = horizon_bound_max
    _, horizon_bound_min_translation = c2w_world_to_topdown(
        horizon_bound_min_c2w_world,
        topdown_info,
        height_direction,
        np.float64)
    horizon_bound_max_c2w_world = np.eye(4)
    horizon_bound_max_c2w_world[:3, 3] = horizon_bound_min
    _, horizon_bound_max_translation = c2w_world_to_topdown(
        horizon_bound_max_c2w_world,
        topdown_info,
        height_direction,
        np.float64)
    horizon_bound_translation = np.vstack([horizon_bound_min_translation, horizon_bound_max_translation])
    horizon_bbox = np.vstack([
        np.min(horizon_bound_translation, axis=0),
        np.max(horizon_bound_translation, axis=0)])
    return horizon_bbox

def update_traj(trajectory:list, color_name:str='cool')->o3d.geometry.LineSet:
    points = []
    lines = []
    colors = []
    line_colormap = plt.get_cmap(color_name)

    for i in range(len(trajectory)):
        points.append(trajectory[i])
        if i < len(trajectory)-1:
            lines.append([i, i+1])
            # Normalize the step index to the range [0, 1]
            color_value = i / (len(trajectory) - 1)
            # Get the color from the colormap
            color = line_colormap(color_value)[:3]  # Take only RGB values
            colors.append(color)
    
    points = np.array(points).astype(np.float32)
    points = points.reshape(-1, 3)
    lines = np.array(lines).reshape(-1, 2)
    
    cam_traj = o3d.geometry.LineSet()
    cam_traj.points = o3d.utility.Vector3dVector(points)
    cam_traj.lines = o3d.utility.Vector2iVector(lines)
    cam_traj.colors = o3d.utility.Vector3dVector(colors)
    
    return cam_traj