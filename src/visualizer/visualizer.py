import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
import json
import threading
import time
from enum import Enum
from copy import deepcopy
from queue import Queue, Empty
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import quaternion
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm
from imgviz import depth2rgb
import open3d as o3d
from open3d.visualization import rendering, gui

import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import Twist, PoseStamped, Pose

from dataloader import RGBDSensor, PoseDataType,  load_scene_mesh, dataset_config_to_ros, convert_to_c2w_opencv
from mapper import get_mapper, MapperState, GaussianColorType, MapperType
from mapper.splatam.utils.graphics_utils import fov2focal
from utils.gui_utils import OPENCV_TO_OPENGL, GaussianPacket, PoseChangeType, vfov_to_hfov, create_frustum, rgbd_to_pointcloud, pose_to_matrix, matrix_to_pose, rotation_matrix_from_vectors, c2w_topdown_to_world, c2w_world_to_topdown, is_pose_changed, get_horizon_bound_topdown, update_traj, config_topdown_info, visualize_agent, translations_world_to_topdown
from utils.logging_utils import Log
from utils import PROJECT_NAME, OPENCV_TO_OPENGL, CURRENT_FRUSTUM, CURRENT_AGENT, CURRENT_HORIZON, start_timing, end_timing, GlobalState
from utils.camera_utils import Camera
from dataloader.dataloader import HabitatDataset

from scripts.nodes import frame, TURN, SPEED,\
    GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
        ResetEnv, ResetEnvResponse, ResetEnvRequest,\
            GetTopdown, GetTopdownRequest, GetTopdownResponse,\
                GetTopdownConfig, GetTopdownConfigRequest, GetTopdownConfigResponse,\
                    SetPlannerState, SetPlannerStateRequest, SetPlannerStateResponse,\
                        SetMapper, SetMapperRequest, SetMapperResponse,\
                            GetOpacity, GetOpacityRequest, GetOpacityResponse,\
                                GetVoronoiGraph, GetVoronoiGraphRequest, GetVoronoiGraphResponse,\
                                    GetNavPath, GetNavPathRequest, GetNavPathResponse

KEYFRAME_FRUSTUM = {
    'color': [0.0, 0.0, 1.0],
    'scale': 0.1,
    'material': 'unlit_line_mat_slim',
}
ROTATION_FRUSTUM = {
    'color': [58. / 255., 142. / 255., 94. / 255.],
    'scale': 0.2,
    'material': 'unlit_line_mat',
}
CURRENT_AGENT = {
    'color': [0.961, 0.475, 0.000],
    'material': 'lit_mat_transparency',
}
CURRENT_HORIZON = {
    'color': [0.0, 1.0, 0.0],
}
VORONOI_GRAPH = {
    'nodes_color': [1.0,0.5, 0.0],
    'ridges_color': [1.0,0.7, 0.71],
    'nodes_radius': 0.06,
    'high_connectivity_nodes_radius': 0.1,
    'ridges_material': 'unlit_line_mat',
    'nodes_material': 'unlit_mat',
}
class Visualizer:
        
    class LocalDatasetState(Enum):
        INITIALIZING = 0
        INITIALIZED = 1
        RUNNING = 2
        
    class QueryTopdownFlag(Enum):
        NONE = 0
        ARRIVED = 1
        RUNNING = 2
        MANUAL = 3
        
    class QueryVisibilityFlag(Enum):
        NONE = 0
        GLOBAL = 1
        LOCAL = 2
        RUNNING = 3
        MANUAL = 4
        
    def __init__(self,
                 mapper_type:MapperType,
                 config_url:str,
                 init_state:GlobalState,
                 font_id:int,
                 device:torch.device,
                 actions_url:str,
                 local_dataset:Union[HabitatDataset],
                 parallelized:bool,
                 hide_windows:bool,
                 save_runtime_data:bool):
        self.__device = device
        self.__hide_windows = hide_windows
        self.__global_states_selectable = [GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL, GlobalState.PAUSE]
        self.__local_dataset = local_dataset
        self.__local_dataset_parallelized = parallelized
        
            
        os.chdir(PACKAGE_PATH)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(config_url) as f:
            config = json.load(f)

        # NOTE: Save runtime data
        self.__save_runtime_data = save_runtime_data
        self.__runtime_data_info = None
        if self.__save_runtime_data:
            self.__runtime_data_info = {"render_use_time": []}
        
        step_num = config['dataset']['step_num']
        self.scene_id = 'None'
        if actions_url == 'None':
            self.__actions = None
            self.__global_state = init_state
        else:
            self.__actions = np.loadtxt(actions_url, dtype=np.int32)
            self.__global_state = GlobalState.REPLAY
            self.__global_states_selectable.append(GlobalState.REPLAY)
        if self.__local_dataset is not None:
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZING
            self.__local_dataset_condition = threading.Condition()
            self.__local_dataset_pose_pub = rospy.Publisher('orb_slam3/camera_pose', PoseStamped, queue_size=1)
            self.__local_dataset_pose_ros = None
            self.__local_dataset_thread = threading.Thread(
                target=self.__update_dataset,
                name='UpdateDataset',
                daemon=True)
            self.__local_dataset_thread.start()
            self.__local_dataset_label = gui.Label('')
            self.__local_dataset_label.font_id = font_id
            _, step_num = self.__local_dataset.get_step_info()
            self.scene_id = self.__local_dataset.get_scene_id()
            
        self.__bbox_padding = config['mapper']['bbox_padding_ratio']
        self.__depth_limit = [config['dataset']['near'], config['dataset']['far']]
        self.__frame_update_translation_threshold = config['mapper']['pose']['update_threshold']['translation']
        self.__frame_update_rotation_threshold = config['mapper']['pose']['update_threshold']['rotation']
        self.__traj_info = dict()
        self.__traj_info['cam_centers'] = []
        self.__traj_info['line_colormap'] = plt.get_cmap('cool')
        self.__traj_info['norm_factor'] = 0.5
        self.__agent_foot_adjust = config['planner']['agent_foot_adjust']
        if self.scene_id == 'YmJkqBEsHnH':
            self.__agent_foot_adjust = 0.15 # NOTE: Special case for YmJkqBEsHnH
            
        self.__high_loss_samples_pose_pub = rospy.Publisher('high_loss_samples_pose', Pose, queue_size=1)
        
        self.__update_main_thread = threading.Thread(
            target=self.__update_main,
            name='UpdateMain',
            daemon=True)
        
        scene_mesh = self.__init_dataset()
        
        self.__init_o3d_elements()
        
        bbox = self.__bbox_visualize.copy()
        frame_first = self.__frames_cache.get()
        c2w = frame_first['c2w'].detach().cpu().numpy()
        agent_sensor = c2w[self.__height_direction[0], 3]
        agent_height_start = agent_sensor - self.__rgbd_sensor.position[self.__height_direction[0]]
        agent_height_end = agent_height_start + self.__dataset_config.agent_height
        
        if self.__height_direction[0] in [1, 2]:
            self.__agent_foot = -agent_height_start
            agent_sensor = -agent_sensor
            self.__agent_head = -agent_height_end
            agent_height_start, agent_height_end = self.__agent_head, self.__agent_foot
        elif self.__height_direction[0] == 0:
            self.__agent_foot = agent_height_start
            self.__agent_head = agent_height_end
        else:
            raise ValueError(f'Invalid height direction: {self.__height_direction}')
        
        if config['mapper']['single_floor']['enable']:
            current_pcd:o3d.geometry.PointCloud = rgbd_to_pointcloud(
                frame_first['rgb'].detach().cpu().numpy(),
                np.ones_like(frame_first['depth'].detach().cpu().numpy(), dtype=np.float32) * self.__rgbd_sensor.depth_max,
                c2w,
                self.__o3d_const_camera_intrinsics_o3c,
                1000,
                self.__rgbd_sensor.depth_max * 2,
                self.__device_o3c).to_legacy()
            current_pcd_bbox:o3d.geometry.AxisAlignedBoundingBox = current_pcd.get_axis_aligned_bounding_box()
            if self.__height_direction[0] in [1, 2]:
                single_floor_height_start = agent_height_start - config['mapper']['single_floor']['expansion']['head']
                single_floor_height_end = agent_height_end + config['mapper']['single_floor']['expansion']['foot']
            elif self.__height_direction[0] == 0:
                single_floor_height_start = agent_height_start - config['mapper']['single_floor']['expansion']['foot']
                single_floor_height_end = agent_height_end + config['mapper']['single_floor']['expansion']['head']
            else:
                raise ValueError(f'Invalid height direction: {self.__height_direction}')
            bbox[self.__height_direction[0]][0] = max(
                single_floor_height_start,
                bbox[self.__height_direction[0]][0],
                current_pcd_bbox.get_min_bound()[self.__height_direction[0]])
            bbox[self.__height_direction[0]][1] = min(
                single_floor_height_end,
                bbox[self.__height_direction[0]][1],
                current_pcd_bbox.get_max_bound()[self.__height_direction[0]])
            assert bbox[self.__height_direction[0]][0] < bbox[self.__height_direction[0]][1], 'Invalid height dimension'
        if self.__frames_cache.empty(): self.__frames_cache.put(frame_first)
        
        bbox:np.ndarray = bbox + self.__bbox_padding *\
            np.reshape(np.ptp(bbox, axis=1), (3, 1)) *\
                np.array([-1, 1])
                
        # NOTE: Get visibility information
        self.__visibility_info:Dict = {'local_visibility_cv2': None}
        
        # NOTE: Get basic information of topdown view
        self.__topdown_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray], torch.Tensor, np.ndarray, cv2.Mat]] = {
            'world_dim_index': (
                (self.__height_direction[0] + self.__height_direction[1]) % 3,  # x+ of the topdown view
                (self.__height_direction[0] - self.__height_direction[1]) % 3), # y- of the topdown view
        }
        
        # NOTE: Dynamically adjust height
        self.__dynamic_height_info:Dict[str, Union[int, float]] = {
            'step_height': 0.1,
        }
        
        topdown_world_2d_bbox = (
            (bbox[self.__topdown_info['world_dim_index'][0]][0], bbox[self.__topdown_info['world_dim_index'][0]][1]),
            (bbox[self.__topdown_info['world_dim_index'][1]][0], bbox[self.__topdown_info['world_dim_index'][1]][1]))
        
        self.__topdown_info['topdown_world_shape'] = (
            topdown_world_2d_bbox[0][1] - topdown_world_2d_bbox[0][0],
            topdown_world_2d_bbox[1][1] - topdown_world_2d_bbox[1][0])
        
        self.__topdown_info['world_center'] = (
            (topdown_world_2d_bbox[0][0] + topdown_world_2d_bbox[0][1]) / 2,
            (topdown_world_2d_bbox[1][0] + topdown_world_2d_bbox[1][1]) / 2)
        
        self.__topdown_info['pixel_per_meter'] =\
            config['painter']['grid_map']['pixel_max'] / max(self.__topdown_info['topdown_world_shape'])
        self.__topdown_info['meter_per_pixel'] = 1 / self.__topdown_info['pixel_per_meter']
        
        topdown_info = config_topdown_info(
            self.__height_direction,
            self.__topdown_info['world_dim_index'],
            self.__topdown_info['topdown_world_shape'],
            self.__topdown_info['world_center'],
            self.__topdown_info['meter_per_pixel'],
            self.__agent_foot,
            agent_sensor,
            self.__agent_head,
            20,
            2)
        
        self.__topdown_info.update(topdown_info)
        
        self.__topdown_info['world_topdown_origin'] = c2w_topdown_to_world(np.zeros(2), self.__topdown_info, 0)
        self.__topdown_info['free_map_binary'] = None
        self.__topdown_info['free_map_cv2'] = None
        self.__topdown_info['free_map_binary_cv2'] = None
        self.__topdown_info['visible_map_binary'] = None
        self.__topdown_info['visible_map_cv2'] = None
        self.__topdown_info['visible_map_binary_cv2'] = None
        self.__topdown_info['translation'] = None
        self.__topdown_info['rotation_vector'] = None
        self.__topdown_info['horizon_bbox'] = None
        self.__current_horizon = None
        
        rospy.Service('get_topdown_config', GetTopdownConfig, self.__get_topdown_config)
            
        bbox_o3d = bbox.copy()
        bbox[1, 0], bbox[1, 1] = -bbox[1, 1], -bbox[1, 0]
        bbox[2, 0], bbox[2, 1] = -bbox[2, 1], -bbox[2, 0]
        
        self.q_main2vis = Queue(maxsize=1)
        self.frustum_dict = {}
        self.__use_gaussian_condition = threading.Condition()
        
        Mapper = get_mapper(mapper_type)
        self.__mapper = Mapper(
            config,
            self.__rgbd_sensor,
            self.__device,
            self.q_main2vis,
            self.__results_dir,
            step_num)
        
        # Runtime data
        if self.__save_runtime_data:
            self.render_rgbd_dir = os.path.join(self.__results_dir, 'render_rgbd')
            if not os.path.exists(self.render_rgbd_dir): os.makedirs(self.render_rgbd_dir)
            self.__init_runtime_data_info()
        
        rospy.Service('set_mapper', SetMapper, self.__set_mapper)
        
        self.__init_window(
            config['mapper']['interval_max_ratio'],
            bbox_o3d[self.__height_direction[0]][(1 - self.__height_direction[1]) // 2],
            (agent_height_start * (1 - self.__height_direction[1]) + agent_height_end * (1 + self.__height_direction[1])) / 2,
            font_id,
            scene_mesh)
        
        self.__update_main_thread.start()
    
    # NOTE: initialization functions
    
    def __init_dataset(self) -> o3d.geometry.TriangleMesh:
        self.__frames_cache:Queue[Dict[str, Union[int, torch.Tensor]]] = Queue(maxsize=1)
        self.__frame_c2w_last = None
        
        self.__get_topdown_flag = self.QueryTopdownFlag.NONE
        self.__get_visibility_flag = self.QueryVisibilityFlag.NONE
        self.__get_voronoi_graph_service = rospy.ServiceProxy('get_voronoi_graph', GetVoronoiGraph)
        rospy.wait_for_service('get_voronoi_graph')
        self.__get_navigation_path_service = rospy.ServiceProxy('get_navigation_path', GetNavPath)
        rospy.wait_for_service('get_navigation_path')
        self.__get_topdown_condition = threading.Condition()
        self.__get_topdown_service = rospy.Service('get_topdown', GetTopdown, self.__get_topdown)
        self.__get_opacity_service = rospy.Service('get_opacity', GetOpacity, self.__get_opacity)
        self.__get_opacity_condition = threading.Condition()
        
        if self.__local_dataset is None:
            reset_env_service = rospy.ServiceProxy('reset_env', ResetEnv)
            rospy.wait_for_service('reset_env')
            
            reset_env_success:ResetEnvResponse = reset_env_service(ResetEnvRequest())
            
            self.__cmd_vel_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
            get_dataset_config_service = rospy.ServiceProxy('get_dataset_config', GetDatasetConfig)
            rospy.wait_for_service('get_dataset_config')
            
            self.__dataset_config:GetDatasetConfigResponse = get_dataset_config_service(GetDatasetConfigRequest())
        else:
            self.__local_dataset_condition.acquire()
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZING:
                self.__local_dataset_condition.wait()
                
        Trc = np.eye(4)
        Trc[:3, 3] = np.array([
            self.__dataset_config.rgbd_position.x,
            self.__dataset_config.rgbd_position.y,
            self.__dataset_config.rgbd_position.z])
        self.__Tcr = np.linalg.inv(Trc)
            
        self.__results_dir = self.__dataset_config.results_dir
        os.makedirs(self.__results_dir, exist_ok=True)
        self.__pose_data_type = PoseDataType(self.__dataset_config.pose_data_type)
        self.__height_direction = (self.__dataset_config.height_direction // 2, (self.__dataset_config.height_direction % 2) * 2 - 1)
        
        self.__rgbd_sensor = RGBDSensor(
            height=self.__dataset_config.rgbd_height,
            width=self.__dataset_config.rgbd_width,
            fx=self.__dataset_config.rgbd_fx,
            fy=self.__dataset_config.rgbd_fy,
            cx=self.__dataset_config.rgbd_cx,
            cy=self.__dataset_config.rgbd_cy,
            depth_min=self.__dataset_config.rgbd_depth_min,
            depth_max=self.__dataset_config.rgbd_depth_max,
            depth_scale=self.__dataset_config.rgbd_depth_scale,
            position=np.array([
                self.__dataset_config.rgbd_position.x,
                self.__dataset_config.rgbd_position.y,
                self.__dataset_config.rgbd_position.z]),
            downsample_factor=self.__dataset_config.rgbd_downsample_factor,
            need_downsample=False)

        if self.__local_dataset is None:
            rospy.Subscriber('frames', frame, self.__frame_callback)
            rospy.wait_for_message('frames', frame)
        else:
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                self.__local_dataset_condition.wait()
            self.__local_dataset_condition.notify_all()
            self.__local_dataset_condition.release()
        
        if os.path.exists(self.__dataset_config.scene_mesh_url):
            self.__scene_mesh_transform = pose_to_matrix(self.__dataset_config.scene_mesh_transform)
            scene_mesh, self.__bbox_visualize = load_scene_mesh(
                self.__dataset_config.scene_mesh_url,
                self.__scene_mesh_transform)
        else:
            scene_mesh = None
            self.__bbox_visualize = np.array([
                [self.__dataset_config.scene_bound_min.x, self.__dataset_config.scene_bound_max.x],
                [self.__dataset_config.scene_bound_min.y, self.__dataset_config.scene_bound_max.y],
                [self.__dataset_config.scene_bound_min.z, self.__dataset_config.scene_bound_max.z]])
        
        self.__agent_cylinder_mesh:o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.__dataset_config.agent_radius, height=self.__dataset_config.agent_height)
        self.__agent_cylinder_mesh.compute_vertex_normals()
        vector_end = np.zeros(3)
        vector_end[self.__height_direction[0]] = self.__height_direction[1]
        self.__agent_cylinder_mesh.rotate(
            rotation_matrix_from_vectors(np.array([0, 0, 1]), vector_end),
            np.zeros(3))
        self.__agent_cylinder_mesh.translate(
            (self.__dataset_config.agent_height / 2) * vector_end)
        self.__agent_cylinder_mesh.paint_uniform_color(CURRENT_AGENT['color'])
        return scene_mesh
    
    def __init_o3d_elements(self):
        self.__device_o3c = o3d.core.Device(self.__device.type, self.__device.index)
        
        # NOTE: Independent Open3D elements
        self.__o3d_meshes:Dict[str, o3d.geometry.TriangleMesh] = {
            'scene_mesh': None
        }
        self.__o3d_pcd:Dict[str, o3d.t.geometry.PointCloud] = {
            'current_pcd': None,
        }
        
        self.__o3d_const_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.__rgbd_sensor.width,
            self.__rgbd_sensor.height,
            self.__rgbd_sensor.fx,
            self.__rgbd_sensor.fy,
            self.__rgbd_sensor.cx,
            self.__rgbd_sensor.cy)
        self.__o3d_const_camera_intrinsics_o3c = o3d.core.Tensor(self.__o3d_const_camera_intrinsics.intrinsic_matrix, device=self.__device_o3c)
        
        self.__o3d_materials:Dict[str, rendering.MaterialRecord] = {
            'lit_mat': None,
            'lit_mat_transparency': None,
            'unlit_mat': None,
            'unlit_line_mat': None,
            'unlit_line_mat_slim': None,
        }
            
        if not self.__hide_windows:
            self.__o3d_materials['lit_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['lit_mat'].shader = 'defaultLit'
            self.__o3d_materials['lit_mat_transparency'] = rendering.MaterialRecord()
            self.__o3d_materials['lit_mat_transparency'].shader = 'defaultLitTransparency'
            self.__o3d_materials['lit_mat_transparency'].has_alpha = True
            self.__o3d_materials['lit_mat_transparency'].base_color = [1.0, 1.0, 1.0, 0.9]
            self.__o3d_materials['unlit_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_mat'].shader = 'defaultUnlit'
            self.__o3d_materials['unlit_mat'].sRGB_color = True
            self.__o3d_materials['unlit_line_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat'].line_width = 6.0
            self.__o3d_materials['unlit_line_mat_slim'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat_slim'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat_slim'].line_width = 3.0
            self.__o3d_materials['unlit_line_mat_thick'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat_thick'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat_thick'].line_width = 15.0
            self.__o3d_materials['gaussian_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['gaussian_mat'].shader = 'defaultUnlit'
        
    def __init_window(self, interval_max_ratio:float, foot_value:float, head_value:float, font_id:int, scene_mesh:o3d.geometry.TriangleMesh=None):
        kf_every = self.__mapper.get_kf_every()
        assert 0 < kf_every, f'Invalid keyframe every: {kf_every}'
        map_every = self.__mapper.get_map_every()
        mapping_iters = self.__mapper.get_mapping_iters()
        assert 0 < map_every, f'Invalid map every: {map_every}'
        update_interval_max = int(max(interval_max_ratio * max(kf_every, map_every), mapping_iters))
        assert interval_max_ratio >= 1.0, f'Invalid interval ratio: {interval_max_ratio}'
        
        self.__open3d_gui_widget_last_state = dict()

        self.__open3d_gui_widget_last_state['height_direction_bound_slider'] = [
            (foot_value * (1 + self.__height_direction[1]) + head_value * (1 - self.__height_direction[1])) / 2,
            (foot_value * (1 - self.__height_direction[1]) + head_value * (1 + self.__height_direction[1])) / 2]
        
        self.__set_planner_state_service = rospy.ServiceProxy('set_planner_state', SetPlannerState)
        rospy.wait_for_service('set_planner_state')
        rospy.Subscriber('update_voronoi_graph_vis', Bool, self.__update_voronoi_graph_trigger_callback, queue_size=1)
        rospy.Subscriber('update_high_connectivity_nodes_vis', Bool, self.__update_high_connectivity_nodes_trigger_callback, queue_size=1)
        rospy.Subscriber('update_global_visibility_map_vis', Int32, self.__update_global_visibility_map_callback, queue_size=1)
        
        if self.__hide_windows:
            self.__global_state_callback(self.__global_state.value, None)
            self.__o3d_meshes['scene_mesh'] = o3d.geometry.TriangleMesh(scene_mesh)
            return
        
        else:
            # NOTE: Initialize GUI
            self.__window:gui.Window = gui.Application.instance.create_window(PROJECT_NAME, 1920, 1080)
            self.__window.show(False)
            
            em = self.__window.theme.font_size
            margin = 0.5 * em
            spacing = int(np.round(0.25 * em))
            vspacing = int(np.round(0.5 * em))
            
            margins = gui.Margins(vspacing)
            self.__panel_control = gui.Vert(spacing, margins)
            self.__panel_visualize = gui.Vert(spacing, margins)
            
            self.__widget_3d = gui.SceneWidget()
            self.__widget_3d.scene = rendering.Open3DScene(self.__window.renderer)
            self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
            self.__widget_3d.scene.scene.set_sun_light([-0.2, 1.0 ,0.2], [1.0, 1.0, 1.0], 70000)
            self.__widget_3d.scene.scene.enable_sun_light(True)
            self.__widget_3d.set_on_key(self.__widget_3d_on_key)
            
            # NOTE: Widgets for control panel
            global_state_vgrid = gui.VGrid(2, spacing, gui.Margins(0, 0, em, 0))

            global_state_vgrid.add_child(gui.Label('Global State'))
            self.__global_state_combobox = gui.Combobox()
            for global_state in self.__global_states_selectable:
                self.__global_state_combobox.add_item(global_state.value)
            self.__global_state_combobox.selected_text = self.__global_state.value
            self.__global_state_combobox.set_on_selection_changed(self.__global_state_callback)
            global_state_vgrid.add_child(self.__global_state_combobox)
            self.__panel_control.add_child(global_state_vgrid)
            self.__global_state_callback(self.__global_state_combobox.selected_text, None)
            
            self.__panel_control.add_fixed(vspacing)
            if self.__local_dataset is not None:
                self.__panel_control.add_child(gui.Label('Local Dataset Info'))
                self.__panel_control.add_child(self.__local_dataset_label)
            
            self.__panel_control.add_child(gui.Label('3D Visualization Settings'))
            
            panel_control_vgrid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))
            
            panel_control_vgrid.add_child(gui.Label('    Mapper Configurations'))
            panel_control_vgrid.add_child(gui.Label(''))

            panel_control_vgrid.add_child(gui.Label('        Map Every'))
            self.__map_every_slider = gui.Slider(gui.Slider.INT)
            self.__map_every_slider.set_limits(1, update_interval_max)
            self.__map_every_slider.int_value = map_every
            self.__map_every_slider.set_on_value_changed(lambda value: self.__mapper.set_map_every(value))
            panel_control_vgrid.add_child(self.__map_every_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframe Every'))
            self.__kf_every_slider = gui.Slider(gui.Slider.INT)
            self.__kf_every_slider.set_limits(1, update_interval_max)
            self.__kf_every_slider.int_value = kf_every
            self.__kf_every_slider.set_on_value_changed(lambda value: self.__mapper.set_kf_every(value))
            panel_control_vgrid.add_child(self.__kf_every_slider)
            

            panel_control_vgrid.add_child(gui.Label('    Global Status'))
            panel_control_vgrid.add_child(gui.Label(''))
            
            view_gs_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            view_gs_grid.add_child(gui.Label('        View Gaussians'))
            self.__view_gaussians_box = gui.Checkbox('')
            self.__view_gaussians_box.checked = True
            def view_gaussians_callback(checked:bool):
                self.followcam_chbox.enabled = checked
                self.staybehind_chbox.enabled = checked
                if checked is False:
                    self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
                return gui.Checkbox.HANDLED
            self.__view_gaussians_box.set_on_checked(view_gaussians_callback)
            view_gs_grid.add_child(self.__view_gaussians_box)
            
            panel_control_vgrid.add_child(view_gs_grid)
            panel_control_vgrid.add_child(gui.Label(''))
            
            # Note: 3DGS Visualization, as referenced in MonoGS.
            panel_control_vgrid.add_child(gui.Label("        Viewing options"))
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            self.followcam_chbox = gui.Checkbox("Follow Camera")
            self.followcam_chbox.checked = False
            chbox_tile.add_child(self.followcam_chbox)
            self.__rerender_voronoi_3d_flag = False
            def followcam_chbox_callback(checked:bool):
                if checked:
                    self.staybehind_chbox.checked = True
                    self.__current_agent_box.checked = False
                    self.__voronoi_3d_box.checked = False
                    self.__cam_traj_box.checked = False
                    self.__bak_start_height = self.__height_direction_lower_bound_slider.double_value
                    self.__height_direction_lower_bound_slider.double_value = self.__height_direction_lower_bound_slider.get_minimum_value
                    self.__rerender_voronoi_3d_flag = False
                else:
                    self.staybehind_chbox.checked = False
                    self.__current_agent_box.checked = True
                    self.__voronoi_3d_box.checked = True
                    self.__cam_traj_box.checked = True
                    self.__height_direction_lower_bound_slider.double_value = self.__bak_start_height
                    self.__widget_3d.look_at(self.__init_look_at['center'], self.__init_look_at['eye'], self.__init_look_at['up'])
                    self.__rerender_voronoi_3d_flag = True
                    if self.__scene_mesh_box is not None:
                        self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
                return gui.Checkbox.HANDLED
            self.followcam_chbox.set_on_checked(followcam_chbox_callback)
            panel_control_vgrid.add_child(chbox_tile)
            
            panel_control_vgrid.add_child(gui.Label(''))
            
            self.staybehind_chbox = gui.Checkbox("From Behind")
            self.staybehind_chbox.checked = False
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            chbox_tile.add_child(self.staybehind_chbox)
            panel_control_vgrid.add_child(chbox_tile)
                
            panel_control_vgrid.add_child(gui.Label('        H Lower Bound'))
            self.__height_direction_lower_bound_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__height_direction_lower_bound_slider.set_limits(
                self.__bbox_visualize[self.__height_direction[0]][0] - 0.2,
                self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1])
            self.__height_direction_lower_bound_slider.double_value = self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0]
            self.__height_direction_lower_bound_slider.set_on_value_changed(
                lambda value: self.__height_direction_bound_slider_callback(value, 0))
            panel_control_vgrid.add_child(self.__height_direction_lower_bound_slider)
            
            panel_control_vgrid.add_child(gui.Label('        H Upper Bound'))
            self.__height_direction_upper_bound_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__height_direction_upper_bound_slider.set_limits(
                self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0],
                self.__bbox_visualize[self.__height_direction[0]][1] + 0.1)
            self.__height_direction_upper_bound_slider.double_value = self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1]
            self.__height_direction_upper_bound_slider.set_on_value_changed(
                lambda value: self.__height_direction_bound_slider_callback(value, 1))
            panel_control_vgrid.add_child(self.__height_direction_upper_bound_slider)
            
            panel_control_vgrid.add_child(gui.Label("        Rendering options"))
            self.__gaussian_color_combobox = gui.Combobox()
            self.__gaussian_color_type = None
            for gaussian_color_type in GaussianColorType:
                self.__gaussian_color_combobox.add_item(gaussian_color_type.value)
            self.__gaussian_color_combobox.selected_text = GaussianColorType.Color.value
            def gaussian_color_combobox_callback(color_type_name:str, color_type_index:int):
                self.__gaussian_color_type = GaussianColorType(color_type_name)
                return gui.Combobox.HANDLED
            self.__gaussian_color_combobox.set_on_selection_changed(gaussian_color_combobox_callback)
            gaussian_color_combobox_callback(self.__gaussian_color_combobox.selected_text, None)
            panel_control_vgrid.add_child(self.__gaussian_color_combobox)
            
            panel_control_vgrid.add_child(gui.Label('        Gaussian Scale (0-1)'))
            self.__gaussian_scale_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__gaussian_scale_slider.set_limits(0.001, 1.0)
            self.__gaussian_scale_slider.double_value = 1.0
            panel_control_vgrid.add_child(self.__gaussian_scale_slider)
            
            panel_control_vgrid.add_child(gui.Label(''))
            panel_control_vgrid.add_child(gui.Label(''))
            
            origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.__widget_3d.scene.add_geometry('origin_mesh', origin_mesh, self.__o3d_materials['lit_mat'])
            panel_control_vgrid.add_child(gui.Label('        Origin Mesh'))
            origin_mesh_box = gui.Checkbox('')
            origin_mesh_box.checked = True
            origin_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('origin_mesh', checked))
            panel_control_vgrid.add_child(origin_mesh_box)
            
            if scene_mesh is not None:
                self.__o3d_meshes['scene_mesh'] = o3d.geometry.TriangleMesh(scene_mesh)
                panel_control_vgrid.add_child(gui.Label('        Ground Truth Mesh'))
                self.__scene_mesh_box = gui.Checkbox('')
                self.__scene_mesh_box.checked = False
                self.__scene_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('scene_mesh', checked))
                panel_control_vgrid.add_child(self.__scene_mesh_box)
            else:
                self.__scene_mesh_box = None
            
            if scene_mesh is not None:
                self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
            
            panel_control_vgrid.add_child(gui.Label('    Local Status'))
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label('        Current Frustum'))
            self.__current_frustum_box = gui.Checkbox('')
            self.__current_frustum_box.checked = True
            self.__current_frustum_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_frustum', checked))
            panel_control_vgrid.add_child(self.__current_frustum_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current Agent'))
            self.__current_agent_box = gui.Checkbox('')
            self.__current_agent_box.checked = True
            self.__current_agent_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_agent', checked))
            panel_control_vgrid.add_child(self.__current_agent_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current Horizon'))
            self.__current_horizon_box = gui.Checkbox('')
            self.__current_horizon_box.checked = False
            self.__current_horizon_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_horizon', checked))
            panel_control_vgrid.add_child(self.__current_horizon_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current PCD'))
            self.__current_pcd_box = gui.Checkbox('')
            self.__current_pcd_box.checked = False
            self.__current_pcd_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_pcd', checked))
            panel_control_vgrid.add_child(self.__current_pcd_box)
            
            panel_control_vgrid.add_child(gui.Label('        Camera Trajectory'))
            self.__cam_traj_box = gui.Checkbox('')
            self.__cam_traj_box.checked = True
            self.__cam_traj_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('cam_traj', checked))
            panel_control_vgrid.add_child(self.__cam_traj_box)
            
            panel_control_vgrid.add_child(gui.Label('        Voronoi 3D'))
            self.__voronoi_3d_box = gui.Checkbox('')
            self.__voronoi_3d_box.checked = True
            self.__voronoi_3d_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('voronoi_nodes', checked))
            panel_control_vgrid.add_child(self.__voronoi_3d_box)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframes'))
            self.__keyframe_box = gui.Checkbox('')
            self.__keyframe_box.checked = False
            panel_control_vgrid.add_child(self.__keyframe_box)
            
            panel_control_vgrid.add_child(gui.Label('        Rotation Frustum'))
            self.__rotation_frustum_box = gui.Checkbox('')
            self.__rotation_frustum_box.checked = False
            panel_control_vgrid.add_child(self.__rotation_frustum_box)
            
            panel_control_vgrid.add_child(gui.Label('        Navigation Path'))
            self.__navigation_path_box = gui.Checkbox('')
            self.__navigation_path_box.checked = True
            panel_control_vgrid.add_child(self.__navigation_path_box)
            
            panel_control_vgrid.add_child(gui.Label('        Dynamic Height'))
            self.__dynamic_height_box = gui.Checkbox('')
            self.__dynamic_height_box.checked = False
            panel_control_vgrid.add_child(self.__dynamic_height_box)
            self.is_dynamic_height_adjusting = False
            self.__dynamic_height_info['continuous_step'] = np.abs(self.__height_direction_upper_bound_slider.double_value - self.__bbox_visualize[self.__height_direction[0]][0]) // self.__dynamic_height_info['step_height']
                
            self.__panel_control.add_child(panel_control_vgrid)
            
            panel_visualize_tabs = gui.TabControl()
            panel_visualize_tab_margin = gui.Margins(0, int(np.round(0.5 * em)), em, em)
            
            tab_live_view = gui.ScrollableVert(0, panel_visualize_tab_margin)
            
            image_placeholder_numpy = np.zeros((self.__rgbd_sensor.height, self.__rgbd_sensor.width * 2, 3), dtype=np.uint8)
            image_placeholder = o3d.geometry.Image(image_placeholder_numpy)
            
            if self.__save_runtime_data:
                save_current_data_button = gui.Button('Save Current Data')
                self.__save_current_data = lambda: self.__save_current_data_callback()
                save_current_data_button.set_on_clicked(self.__save_current_data)
                tab_live_view.add_child(save_current_data_button)
                tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('  RGBD Live Image'))
            self.__rgbd_live_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_live_image)
            tab_live_view.add_fixed(vspacing)
            
            render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            render_grid.add_child(gui.Label('  Rendered RGBD Image'))
            self.__render_box = gui.Checkbox('')
            self.__render_box.checked = True
            def render_box_callback(checked:bool):
                self.__render_every_slider.enabled = checked
                return gui.Checkbox.HANDLED
            self.__render_box.set_on_checked(render_box_callback)
            render_grid.add_child(self.__render_box)
            
            render_grid.add_child(gui.Label('    Render Every'))
            self.__render_every_slider = gui.Slider(gui.Slider.INT)
            self.__render_every_slider.set_limits(1, update_interval_max)
            self.__render_every_slider.int_value = 1
            self.__render_every_slider.enabled = self.__render_box.checked
            render_grid.add_child(self.__render_every_slider)
            
            tab_live_view.add_child(render_grid)
            
            self.__rgbd_render_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_render_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Rotation View Selection'))
            self.__local_visibility_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__local_visibility_map_image)
            tab_live_view.add_fixed(vspacing)
            self.__local_visibility_map_image.update_image(image_placeholder)
            
            tab_live_view.add_child(gui.Label('Position View Selection'))
            self.__global_visibility_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__global_visibility_map_image)
            tab_live_view.add_fixed(vspacing)
            self.__global_visibility_map_image.update_image(image_placeholder)
            
            tab_live_view.add_child(gui.Label('Occupied Areas (RGB)'))
            self.__topdown_visible_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_visible_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('  Obstacles Map'))
            self.__topdown_free_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_free_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Occupied Areas (binary)'))
            self.__topdown_visible_map_image_binary = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_visible_map_image_binary)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Obstacles Map (binary)'))
            self.__topdown_free_map_image_binary = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_free_map_image_binary)
            tab_live_view.add_fixed(vspacing)
            
            # NOTE: Information
            tab_live_info = gui.ScrollableVert(0, panel_visualize_tab_margin)
            self.num_gaussians_info = gui.Label("Number of Gaussians: ")
            tab_live_info.add_child(self.num_gaussians_info)
            self.cam_pose_info = gui.Label("Camera Pose: ")
            tab_live_info.add_child(self.cam_pose_info)
            self.render_use_time_info = gui.Label("Render Use Time: ")
            tab_live_info.add_child(self.render_use_time_info)
            
            panel_visualize_tabs.add_tab('Live View', tab_live_view)
            panel_visualize_tabs.add_tab('Live Info', tab_live_info)
            
            self.__panel_visualize.add_child(panel_visualize_tabs)
            
            self.__window.add_child(self.__panel_control)
            self.__window.add_child(self.__widget_3d)
            self.__window.add_child(self.__panel_visualize)
            
            self.__window.set_on_layout(self.__window_on_layout)
            self.__window.set_on_close(self.__window_on_close)
            
            # NOTE: Setup the UI Camera
            center = np.average(self.__bbox_visualize, axis=1)
            center[self.__height_direction[0]] = 0
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.__bbox_visualize[:, 0], self.__bbox_visualize[:, 1])
            self.__widget_3d.setup_camera(60.0, bbox, center)
            height_location = np.max(np.ptp(self.__bbox_visualize, axis=1))
            center_bias = np.zeros(3)
            use_topdown_view = False
            if use_topdown_view:
                center_bias[self.__height_direction[0]] = self.__height_direction[1] * (height_location - 1) - 1.5
            else:
                center_bias[self.__height_direction[0]] = self.__height_direction[1] * (height_location - 1)
                center_bias[(self.__height_direction[0] + 1) % 3] = 5 * self.__height_direction[1]
            up_vector = np.zeros(3)
            up_vector[(self.__height_direction[0] + 1) % 3] = -self.__height_direction[1]
            self.__init_look_at = {'center': center, 'eye': center + center_bias, 'up': up_vector}
            self.__widget_3d.look_at(center, center + center_bias, up_vector)
            self.__window.show(True)
    
    def __init_runtime_data_info(self):
        directories = {
            'runtime_data_dir': os.path.join(self.__results_dir, 'runtime_data'),
            'opacity_dir': os.path.join(self.__results_dir, 'runtime_data', 'opacity'),
            'current_vis_data_dir': os.path.join(self.__results_dir, 'runtime_data', 'current_vis_data'),
            'topdown_visible_map_dir': os.path.join(self.__results_dir, 'runtime_data', 'topdown_visible_map')
        }

        for key, path in directories.items():
            if not os.path.exists(path):
                os.makedirs(path)
            self.__runtime_data_info[key] = path

        self.__runtime_data_info['current_vis_data'] = dict()
    
    # NOTE: main function
    
    def __update_main(self):
        frame_id = None
        frame_last_received = None
        rendering_rgbd_last_frame_id = -np.inf
        self.__gaussian_for_render = None
        self.__gaussian_packet = None
        self.__render_topdown_free_map = None
        self.__render_topdown_visible_map = None
        self.voronoi_3d_ridges = None
        self.gaussians_num = 0
        self.nodes_score = np.zeros(100)
        self.__trigger_update_voronoi_graph_flag = False
        self.__trigger_high_connectivity_nodes_flag = False
        self.__trigger_global_visibility_map_flag = False
        height = 0
        self.voronoi_3d_vertices = []
        self.voronoi_3d_high_connectivity_nodes = []
        self.voronoi_3d_vertices_color = []
        self.whole_navigation_path = None
        self.__rgb_opacity_images = {}
        
        while self.__global_state != GlobalState.QUIT:
            if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
                # NOTE: Get observation
                if self.__frames_cache.empty() and (frame_id is not None):
                    frame_current = None
                else:
                    frame_current = self.__frames_cache.get()
                    if frame_id is None:
                        frame_id = 0
                        self.__update_ui_frame(frame_current)
                        self.__get_topdown_flag = self.QueryTopdownFlag.MANUAL
                        self.__get_visibility_flag = self.QueryVisibilityFlag.MANUAL
                    else:
                        frame_id += 1
                    frame_current['frame_id'] = frame_id
                    frame_last_received = frame_current.copy()
                    self.__traj_info["step_times"] = frame_id
                assert frame_id is not None, 'Initialize failed'
            else:
                frame_current = None
            timing_mapper_run = start_timing()
            if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
                mapper_state = self.__mapper.run(frame_current)
                if self.__high_loss_samples_pose_pub is not None and self.__mapper.high_loss_samples_pose_c2w is not None:
                    pose = Pose()
                    pose.position.x = self.__mapper.high_loss_samples_pose_c2w[0, 3]
                    pose.position.y = self.__mapper.high_loss_samples_pose_c2w[1, 3]
                    pose.position.z = self.__mapper.high_loss_samples_pose_c2w[2, 3]
                    pose_quaternion = quaternion.from_rotation_matrix(self.__mapper.high_loss_samples_pose_c2w[:3, :3])
                    pose_quaternion = quaternion.as_float_array(pose_quaternion)
                    pose.orientation.x = pose_quaternion[1]
                    pose.orientation.y = pose_quaternion[2]
                    pose.orientation.z = pose_quaternion[3]
                    pose.orientation.w = pose_quaternion[0]
                    self.__high_loss_samples_pose_pub.publish(pose)
            else:
                mapper_state = MapperState.IDLE
            Log(f'Mapper run used {end_timing(*timing_mapper_run):.2f} ms')
                
            if self.__save_runtime_data and ((frame_id % 100 == 0 and frame_id > 0)):
                self.__save_current_data_callback()
            
            if not self.__hide_windows and frame_id == (self.__mapper.get_step_num() - 2 * self.__dynamic_height_info['continuous_step']):
                self.__dynamic_height_box.checked = True
                
            timing_topdown_render = start_timing()
            # NOTE: Render topdown map
            rerender_topdown_flag = False
            if (self.__gaussian_packet is not None and\
                        self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]):
                topdown_cam = self.get_topdown_cam()
                with self.__use_gaussian_condition:
                    self.__use_gaussian_condition.acquire()
                    gaussian_free_map = {k: v.clone() for k, v in self.__gaussian_packet.params.items()}
                    gaussian_free_map = self.__cut_gaussian_by_height(
                                                    gaussian_free_map, 
                                                    self.__agent_head, 
                                                    self.__agent_foot - self.__agent_foot_adjust)
                    self.__render_topdown_free_map = self.__mapper.render_o3d_image(gaussian_free_map, topdown_cam, scale_modifier=0.01, gaussian_color_type=GaussianColorType.Opacity, is_original=True)
                    self.__render_topdown_visible_map = self.__mapper.render_o3d_image(self.__gaussian_for_render, topdown_cam, scale_modifier=0.01)
                    self.__use_gaussian_condition.release()
                rerender_topdown_flag = True
            Log(f'Topdown render used {end_timing(*timing_topdown_render):.2f} ms')
            # NOTE: Update topdown map
            if (self.__get_topdown_flag in [self.QueryTopdownFlag.ARRIVED, self.QueryTopdownFlag.RUNNING]) or \
                rerender_topdown_flag == True:
                max_opacity = np.max(self.__render_topdown_free_map)
                self.__render_topdown_free_map_cv2 = depth2rgb(
                    self.__render_topdown_free_map, min_value=0.0, max_value=max_opacity, colormap="jet"
                )
                self.__render_topdown_free_map_cv2 = torch.from_numpy(self.__render_topdown_free_map_cv2)
                self.__render_topdown_free_map_cv2 = torch.permute(self.__render_topdown_free_map_cv2, (2, 0, 1)).float()
                self.__render_topdown_free_map_cv2 = (self.__render_topdown_free_map_cv2).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                topdown_gray_visible_map = cv2.cvtColor(
                    self.__render_topdown_visible_map,
                    cv2.COLOR_RGB2GRAY)
                self.__topdown_info['free_map_binary'] = (self.__render_topdown_free_map <= 0.4).astype(np.uint8)
                self.__topdown_info['visible_map_binary'] = (topdown_gray_visible_map == 255).astype(np.uint8)
                
                self.__topdown_info['free_map_cv2'] = self.__render_topdown_free_map_cv2
                self.__topdown_info['free_map_binary_cv2'] = cv2.cvtColor(
                    self.__topdown_info['free_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                self.__topdown_info['visible_map_cv2'] = self.__render_topdown_visible_map
                self.__topdown_info['visible_map_binary_cv2'] = cv2.cvtColor(
                    self.__topdown_info['visible_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                self.__update_ui_topdown()
                
                if (self.__get_topdown_flag == self.QueryTopdownFlag.RUNNING) or\
                    (self.__get_topdown_flag == self.QueryTopdownFlag.ARRIVED):
                    self.__get_topdown_flag = self.QueryTopdownFlag.NONE
                    with self.__get_topdown_condition:
                        self.__get_topdown_condition.notify_all()
                        self.__get_topdown_condition.wait()
                elif self.__get_topdown_flag == self.QueryTopdownFlag.MANUAL:
                    self.__get_topdown_flag = self.QueryTopdownFlag.NONE
                    with self.__get_topdown_condition:
                        self.__get_topdown_condition.notify_all()
                        
            # NOTE: Update visibility map
            rerender_voronoi_3d_flag = False
            rerender_rotation_frustum_flag = False
            if self.__get_visibility_flag in [self.QueryVisibilityFlag.GLOBAL, self.QueryVisibilityFlag.LOCAL] and frame_last_received is not None:
                if self.__get_visibility_flag == self.QueryVisibilityFlag.GLOBAL:
                    # NOTE: Global visibility
                    frame_last_received_c2w = frame_last_received['c2w'].detach().cpu().numpy()
                    height = frame_last_received_c2w[self.__height_direction[0], 3]
                    if self.__runtime_data_info is not None:
                        current_vis_data_dir = self.__runtime_data_info['opacity_dir'] + f'/step_{self.__traj_info["step_times"]}'
                        os.makedirs(current_vis_data_dir, exist_ok=True)
                    self.__rgb_opacity_images = {} # node_id: rgb_opacity_vis
                    timing_get_visibility = start_timing()
                    for idx, node in enumerate(self.__mapper.voronoi_nodes):
                        cur_id = node['id']
                        node_position = node['position']
                        rgb_opacity_vis, node['invisibility'], node['volume'] = self.__mapper.get_global_invisibility(
                            frame_last_received, cur_id, node_position, scale_modifier=1.0, show_image=(not self.__hide_windows or self.__save_runtime_data))
                        if rgb_opacity_vis is not None:
                            if self.__save_runtime_data:
                                cv2.imwrite(os.path.join(current_vis_data_dir, f'{cur_id}.png'), rgb_opacity_vis)
                            self.__rgb_opacity_images[cur_id] = cv2.cvtColor(rgb_opacity_vis,cv2.COLOR_RGB2BGR)
                            
                    Log(f'Get global visibility used {end_timing(*timing_get_visibility):.2f} ms', tag='ActiveSplat')
                        
                elif self.__get_visibility_flag == self.QueryVisibilityFlag.LOCAL:
                    # NOTE: Local visibility
                    timing_get_visibility = start_timing()
                    self.local_node = dict()
                    rgb_opacity_vis, self.local_node['invisibility'], self.local_node['best_pose'] = self.__mapper.get_local_invisibility(
                        frame_last_received, scale_modifier=1.0, show_image=(not self.__hide_windows))
                    if rgb_opacity_vis is not None:
                        self.__visibility_info['local_visibility_cv2'] = cv2.cvtColor(
                            rgb_opacity_vis,
                            cv2.COLOR_RGB2BGR)
                        self.__update_ui_visibility()
                        
                    if self.local_node['best_pose'] is not None:
                        pose_data_o3d = OPENCV_TO_OPENGL @ self.local_node['best_pose']
                        self.rotation_frustum = self.update_target_frustums([pose_data_o3d], ROTATION_FRUSTUM['color'])
                        rerender_rotation_frustum_flag = True
                    Log(f'Get local visibility used {end_timing(*timing_get_visibility):.2f} ms', tag='ActiveSplat')
                    
                if (self.__get_visibility_flag == self.QueryVisibilityFlag.RUNNING) or\
                    (self.__get_visibility_flag == self.QueryVisibilityFlag.GLOBAL) or\
                        (self.__get_visibility_flag == self.QueryVisibilityFlag.LOCAL):
                    self.__get_visibility_flag = self.QueryVisibilityFlag.NONE
                    with self.__get_opacity_condition:
                        self.__get_opacity_condition.notify_all()
                        self.__get_opacity_condition.wait()
                elif self.__get_visibility_flag == self.QueryVisibilityFlag.MANUAL:
                    self.__get_visibility_flag = self.QueryVisibilityFlag.NONE
                    with self.__get_opacity_condition:
                        self.__get_opacity_condition.notify_all()
            
            rerender_global_visibility_map_flag = False
            if not self.__hide_windows and self.__rgb_opacity_images and self.__trigger_global_visibility_map_flag:
                if self.__show_node_id in self.__rgb_opacity_images.keys():
                    global_visibility_map = self.__rgb_opacity_images[self.__show_node_id]
                    self.__global_visibility_map_o3d = o3d.geometry.Image(global_visibility_map)
                    rerender_global_visibility_map_flag = True
                self.__trigger_global_visibility_map_flag = False
            
            # NOTE: Show Voronoi 3D
            cmap = plt.get_cmap('YlOrBr')
            if not self.__hide_windows and self.__trigger_update_voronoi_graph_flag and height != 0:
                voronoi_graph_response:GetVoronoiGraphResponse = self.__get_voronoi_graph_service(GetVoronoiGraphRequest())
                nodes_position_3d = np.array(voronoi_graph_response.nodes_position_3d).reshape(-1, 3)
                voronoi_graph_3d_lines = np.array(voronoi_graph_response.voronoi_graph_3d_lines).reshape(-1, 2)
                voronoi_graph_3d_points = np.array(voronoi_graph_response.voronoi_graph_3d_points).reshape(-1, 3)
                high_connectivity_nodes_3d = np.array(voronoi_graph_response.high_connectivity_nodes_3d).reshape(-1, 3)
                nodes_score = np.array(voronoi_graph_response.nodes_score)
                if len(nodes_score) > 0 and len(nodes_position_3d) > 0:
                    self.nodes_score = (nodes_score / np.max(nodes_score)).tolist() # Normalized to 0-10 score
                    if len(nodes_position_3d) == len(self.nodes_score):
                        self.voronoi_3d_vertices = []
                        self.voronoi_3d_vertices_color = []
                        for idx, node_position in enumerate(nodes_position_3d):
                            pose = np.eye(4)
                            pose[:3, 3] = node_position
                            pose[:3, 3][self.__height_direction[0]] = height - 0.05
                            pose = OPENCV_TO_OPENGL @ pose @ OPENCV_TO_OPENGL
                            self.voronoi_3d_vertices.append(pose[:3, 3].tolist())
                            # NOTE: The color is determined by the size of nodes_score nodes_score[idx]
                            color = cmap(self.nodes_score[idx])
                            self.voronoi_3d_vertices_color.append(color)
                    
                        voronoi_graph_3d_points[:, self.__height_direction[0]] = height - 0.05
                        points_homogeneous = np.hstack((voronoi_graph_3d_points, np.ones((voronoi_graph_3d_points.shape[0], 1))))
                        points_transformed = (OPENCV_TO_OPENGL @ points_homogeneous.T)[:3, :].T
                        
                        # Create and color LineSet
                        self.voronoi_3d_ridges = o3d.geometry.LineSet()
                        self.voronoi_3d_ridges.points = o3d.utility.Vector3dVector(points_transformed)
                        self.voronoi_3d_ridges.lines = o3d.utility.Vector2iVector(voronoi_graph_3d_lines)
                        self.voronoi_3d_ridges.paint_uniform_color(VORONOI_GRAPH['ridges_color'])
                        rerender_voronoi_3d_flag = True

                        if self.__trigger_high_connectivity_nodes_flag:
                            if len(high_connectivity_nodes_3d) > 0:
                                self.voronoi_3d_high_connectivity_nodes = []
                                for idx, high_connectivity_node_position in enumerate(high_connectivity_nodes_3d):
                                    pose = np.eye(4)
                                    pose[:3, 3] = high_connectivity_node_position
                                    pose[:3, 3][self.__height_direction[0]] = height - 0.05
                                    pose = OPENCV_TO_OPENGL @ pose @ OPENCV_TO_OPENGL
                                    self.voronoi_3d_high_connectivity_nodes.append(pose[:3, 3].tolist())
                            self.__trigger_high_connectivity_nodes_flag = False
                            rerender_voronoi_3d_flag = True
                self.__trigger_update_voronoi_graph_flag = False

            # NOTE: Dynamic height adjusting
            if not self.__hide_windows and self.__dynamic_height_box.checked:
                if not self.is_dynamic_height_adjusting:
                    start_frame_id = frame_id
                    middle_frame_id = start_frame_id + self.__dynamic_height_info['continuous_step'] // 2
                    stop_frame_id = start_frame_id + self.__dynamic_height_info['continuous_step']
                    self.is_dynamic_height_adjusting = True
                if frame_id == start_frame_id:
                    bak_start_height = self.__height_direction_lower_bound_slider.double_value
                    self.__current_agent_box.checked = False
                    self.__current_frustum_box.checked = False
                    self.__cam_traj_box.checked = False
                    self.__voronoi_3d_box.checked = False
                    self.__navigation_path_box.checked = False
                if frame_id >= start_frame_id and frame_id <= middle_frame_id:
                    current_value = self.__height_direction_lower_bound_slider.double_value
                    target_value = current_value - self.__dynamic_height_info['step_height'] 
                    if target_value > self.__height_direction_lower_bound_slider.get_minimum_value:
                        self.__height_direction_lower_bound_slider.double_value = target_value
                elif frame_id > middle_frame_id and frame_id < stop_frame_id:
                    current_value = self.__height_direction_lower_bound_slider.double_value
                    target_value = current_value + self.__dynamic_height_info['step_height'] 
                    if target_value < bak_start_height:
                        self.__height_direction_lower_bound_slider.double_value = target_value
                elif frame_id == stop_frame_id:
                    self.__height_direction_lower_bound_slider.double_value = bak_start_height
                    self.__current_agent_box.checked = True
                    self.__current_frustum_box.checked = True
                    self.__cam_traj_box.checked = True
                    self.__voronoi_3d_box.checked = True
                    self.__dynamic_height_box.checked = False
                    self.is_dynamic_height_adjusting = False
                    self.__navigation_path_box.checked = True
                    rerender_voronoi_3d_flag = True
            
            # NOTE: Render whole navigation path
            rerender_whole_navigation_path_flag = False
            self.whole_navigation_traj = None
            if not self.__hide_windows and self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
                navigation_path_response:GetNavPathResponse = self.__get_navigation_path_service(GetNavPathRequest())
                self.whole_navigation_path = np.array(navigation_path_response.whole_navigation_path).reshape(-1,3)
                if len(self.whole_navigation_path) > 1 and frame_id % 1 == 0:
                    if len(self.whole_navigation_path) > 0:
                        # To OpenGL coordinate
                        self.whole_navigation_path[:, self.__height_direction[0]] = height
                        whole_navigation_path_homogeneous = np.hstack((self.whole_navigation_path, np.ones((self.whole_navigation_path.shape[0], 1))))
                        whole_navigation_path_transformed = (OPENCV_TO_OPENGL @ whole_navigation_path_homogeneous.T)[:3, :].T
                        whole_navigation_path_diff = np.linalg.norm(
                            np.diff(whole_navigation_path_transformed, axis=0),
                            axis=1)
                        whole_navigation_path_len = np.sum(whole_navigation_path_diff)
                        
                        if whole_navigation_path_len > 0.01: # 0.01m
                            self.whole_navigation_traj = update_traj(whole_navigation_path_transformed, color_name='YlOrBr')
                            rerender_whole_navigation_path_flag = True
            
            # NOTE: Render RGBD image
            rerender_rgbd_flag = False
            if not self.__hide_windows or self.__save_runtime_data:
                if (frame_last_received is not None and\
                    frame_id > rendering_rgbd_last_frame_id):
                    color_vis, depth_vis = self.__mapper.render_rgbd(frame_last_received, scale_modifier=1.0)
                    rgbd = np.hstack((color_vis, depth_vis))
                    if self.__save_runtime_data:
                        cv2.imwrite(str(self.render_rgbd_dir) + f'/{frame_id}.png', cv2.cvtColor(rgbd, cv2.COLOR_BGR2RGB))
                        self.__runtime_data_info['current_vis_data']['rgb_render'] = color_vis
                        self.__runtime_data_info['current_vis_data']['depth_render'] = depth_vis
                    if (not self.__hide_windows and self.__render_box.checked and\
                            frame_id % self.__render_every_slider.int_value == 0):
                        self.__o3d_cache_render_rgbd = o3d.geometry.Image(rgbd)
                    rendering_rgbd_last_frame_id = frame_id
                    rerender_rgbd_flag = True
                
            # NOTE: Update GUI
            self.__update_ui_mapper(
                frame_current,
                rerender_rgbd_flag,
                rerender_topdown_flag,
                rerender_voronoi_3d_flag,
                rerender_rotation_frustum_flag,
                rerender_whole_navigation_path_flag,
                rerender_global_visibility_map_flag)
            
            if self.__local_dataset is not None:
                if self.__local_dataset_pose_ros is not None:
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
            time.sleep(0.05)
            
        # NOTE: Save constructed scene data
        manifest_json = json.dumps(self.__mapper.manifest, indent=4)
        with open(self.__mapper.save_path.joinpath("transforms.json"), "w") as f:
            f.write(manifest_json)
        self.traing_finished = True
        self.__mapper.post_processing()
        Log('Saving scene data finished')
        
        if os.path.exists(self.__dataset_config.scene_mesh_url):
            with open(os.path.join(self.__results_dir, 'gt_mesh.json'), 'w') as f:
                gt_mesh_config = {
                    'mesh_url': self.__dataset_config.scene_mesh_url,
                    'mesh_transform': self.__scene_mesh_transform.tolist()}
                json.dump(gt_mesh_config, f, indent=4)
        set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(GlobalState.QUIT.value))
        self.__close_all()

    def __update_ui_mapper(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]],
                        rerender_rgbd_flag:bool,
                        rerender_topdown_flag:bool,
                        rerender_voronoi_3d_flag:bool,
                        rerender_rotation_frustum_flag:bool,
                        rerender_whole_navigation_path_flag:bool,
                        rerender_global_visibility_map_flag:bool
                        ):
        if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
            self.receive_data(self.q_main2vis)
            if not self.__hide_windows:
                if self.__view_gaussians_box.checked == True:
                    self.render_gaussian()
        
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_mapper(
                    rerender_rgbd_flag,
                    rerender_topdown_flag,
                    rerender_voronoi_3d_flag,
                    rerender_rotation_frustum_flag,
                    rerender_whole_navigation_path_flag,
                    rerender_global_visibility_map_flag))
            Log(f'Update ui of mapper used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
        
        return
    
    def __update_main_thread_ui_mapper(self,
                        rerender_rgbd_flag:bool,
                        rerender_topdown_flag:bool,
                        rerender_voronoi_3d_flag:bool,
                        rerender_rotation_frustum_flag:bool,
                        rerender_whole_navigation_path_flag:bool,
                        rerender_global_visibility_map_flag:bool):
        
        # NOTE: Delete all keyframe frustum
        if not self.__keyframe_box.checked:
            i = 0
            while True:
                if (not self.__widget_3d.scene.has_geometry(f"kf_frustum_{i}")):
                    break
                else:
                    if self.__widget_3d.scene.has_geometry(f"kf_frustum_{i}"):
                        self.__widget_3d.scene.remove_geometry(f"kf_frustum_{i}")
                i += 1
                
        if self.__rotation_frustum_box.checked and rerender_rotation_frustum_flag:
            i = 0
            while True:
                if (not self.__widget_3d.scene.has_geometry(f"rotation_frustum_{i}")):
                    break
                else:
                    if self.__widget_3d.scene.has_geometry(f"rotation_frustum_{i}"):
                        self.__widget_3d.scene.remove_geometry(f"rotation_frustum_{i}")
            for i, target_frustum in enumerate(self.rotation_frustum):
                if (target_frustum is not None) and self.__rotation_frustum_box.checked:
                    self.__widget_3d.scene.add_geometry(f"rotation_frustum_{i}", target_frustum, self.__o3d_materials[ROTATION_FRUSTUM['material']])
                
        if not self.__voronoi_3d_box.checked or rerender_voronoi_3d_flag:
            if self.__widget_3d.scene.has_geometry("voronoi_ridge"):
                self.__widget_3d.scene.remove_geometry("voronoi_ridge")
            i = 0
            while True:
                if (not self.__widget_3d.scene.has_geometry(f"voronoi_vertices_{i}")):
                    break
                else:
                    if self.__widget_3d.scene.has_geometry(f"voronoi_vertices_{i}"):
                        self.__widget_3d.scene.remove_geometry(f"voronoi_vertices_{i}")
                i += 1
            i = 0
            while True:
                if not self.__widget_3d.scene.has_geometry(f"voronoi_vertices_high_connectivity_{i}"):
                    break
                else:
                    if self.__widget_3d.scene.has_geometry(f"voronoi_vertices_high_connectivity_{i}"):
                        self.__widget_3d.scene.remove_geometry(f"voronoi_vertices_high_connectivity_{i}")
                    i += 1
                
        if self.__voronoi_3d_box.checked and (rerender_voronoi_3d_flag or self.__rerender_voronoi_3d_flag):
            if self.__rerender_voronoi_3d_flag:
                self.__rerender_voronoi_3d_flag = False
            if not self.voronoi_3d_ridges is None:
                self.__widget_3d.scene.add_geometry("voronoi_ridge", self.voronoi_3d_ridges, self.__o3d_materials[VORONOI_GRAPH['ridges_material']])
            voronoi_3d_vertices = self.voronoi_3d_vertices
            
            for vertex_id, vertex in enumerate(voronoi_3d_vertices):
                vertex_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=VORONOI_GRAPH['nodes_radius'])
                vertex_sphere.translate(np.array(vertex))
                color = np.array(self.voronoi_3d_vertices_color[vertex_id][:3])
                vertex_sphere.paint_uniform_color(color)
                self.__widget_3d.scene.add_geometry(f"voronoi_vertices_{vertex_id}", vertex_sphere, self.__o3d_materials[VORONOI_GRAPH['nodes_material']])
        # if self.__voronoi_3d_box.checked and len(self.voronoi_3d_high_connectivity_nodes) > 0:
            voronoi_3d_high_connectivity_nodes = self.voronoi_3d_high_connectivity_nodes
            for node_id, high_connectivity_node in enumerate(voronoi_3d_high_connectivity_nodes):
                vertex_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=VORONOI_GRAPH['high_connectivity_nodes_radius'])
                vertex_sphere.translate(np.array(high_connectivity_node))
                color = np.array([0.2,1.0,0.2]) # green
                vertex_sphere.paint_uniform_color(color)
                self.__widget_3d.scene.add_geometry(f"voronoi_vertices_high_connectivity_{node_id}", vertex_sphere, self.__o3d_materials[VORONOI_GRAPH['nodes_material']])
                
        self.num_gaussians_info.text = "Number of Gaussians: {}".format(self.gaussians_num)
        
        if self.whole_navigation_traj is not None:
            self.__widget_3d.scene.remove_geometry("whole_navigation_traj")
            if self.__navigation_path_box.checked and rerender_whole_navigation_path_flag:
                self.__widget_3d.scene.add_geometry("whole_navigation_traj", self.whole_navigation_traj, self.__o3d_materials['unlit_line_mat_thick'])
            
        if rerender_rgbd_flag:
            self.__rgbd_render_image.update_image(self.__o3d_cache_render_rgbd)
        if rerender_global_visibility_map_flag:
            self.__global_visibility_map_image.update_image(self.__global_visibility_map_o3d)
        return

    def __update_ui_frame(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]]):
        if not self.__update_main_thread.is_alive():
            return
        
        current_frustum = None
        current_agent = None
        rgbd_image = None
        current_pcd = None
        current_horizon = None
        cam_traj = None
        
        if frame_current is not None:
            rgb_data:np.ndarray = frame_current['rgb'].detach().cpu().numpy()
            depth_data:np.ndarray = frame_current['depth'].detach().cpu().numpy()
        
            pose_data:Union[np.ndarray, torch.Tensor] = frame_current['c2w']
            if isinstance(pose_data, torch.Tensor):
                pose_data = pose_data.detach().cpu().numpy()
            
            # Twr = Twc * Tcr
            pose_quat = quaternion.from_rotation_matrix(pose_data[:3, :3])
            pose_rotation_vector_ = quaternion.as_rotation_vector(pose_quat)
            pose_rotation_vector = np.zeros(3)
            pose_rotation_vector[self.__height_direction[0]] = pose_rotation_vector_[self.__height_direction[0]]
            pose_quat_only_yaw = quaternion.from_rotation_vector(pose_rotation_vector)
            pose_data_only_yaw = deepcopy(pose_data)
            pose_data_only_yaw[:3, :3] = quaternion.as_rotation_matrix(pose_quat_only_yaw)
            pose_data_agent:np.ndarray = pose_data_only_yaw @ self.__Tcr
            pose_data_agent[:3, :3] = pose_data[:3, :3]
            
            self.__topdown_info['rotation_vector'], self.__topdown_info['translation'] = c2w_world_to_topdown(
                pose_data_agent,
                self.__topdown_info,
                self.__height_direction,
                np.float64)
            self.__topdown_info['translation_pixel'] = translations_world_to_topdown(
                pose_data_agent[:3, 3],
                self.__topdown_info,
                np.int32).reshape(-1)
            self.__update_ui_topdown()
            self.__update_ui_visibility()
        
            pose_data_o3d = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
            current_frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                CURRENT_FRUSTUM['scale'])
            current_frustum.paint_uniform_color(CURRENT_FRUSTUM['color'])
            pose_data_agent_o3d = OPENCV_TO_OPENGL @ pose_data_agent @ OPENCV_TO_OPENGL
            current_agent = o3d.geometry.TriangleMesh(self.__agent_cylinder_mesh)
            current_agent.translate(pose_data_agent_o3d[:3, 3])
        
            rgb_vis = np.uint8(rgb_data * 255)
            depth_vis = depth2rgb(depth_data, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max)
            rgbd_vis = np.hstack((rgb_vis, depth_vis))
            
            rgbd_image = o3d.geometry.Image(rgbd_vis)
            
            if self.__save_runtime_data:
                self.__runtime_data_info['current_vis_data']['rgb'] = rgb_vis
                self.__runtime_data_info['current_vis_data']['depth'] = depth_vis
            
            if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)) or np.all(depth_data == 0):
                rospy.logwarn('Depth contains NaN, Inf or all 0')
                self.__valid_depth_flag = False
            else:
                self.__o3d_pcd['current_pcd'] = rgbd_to_pointcloud(
                    rgb_vis,
                    depth_data,
                    pose_data,
                    self.__o3d_const_camera_intrinsics_o3c,
                    1000,
                    self.__rgbd_sensor.depth_max,
                    self.__device_o3c
                )
                current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                    'current_pcd',
                    False if self.__hide_windows else self.__current_pcd_box.checked,
                    self.__o3d_materials['unlit_mat'],
                    False)
                current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
                current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
                current_horizon.color = CURRENT_HORIZON['color']
                self.__topdown_info['horizon_bbox'] = (
                    OPENCV_TO_OPENGL[:3, :3] @ current_horizon.get_min_bound(),
                    OPENCV_TO_OPENGL[:3, :3] @ current_horizon.get_max_bound())
                self.__current_horizon = get_horizon_bound_topdown(
                    self.__topdown_info['horizon_bbox'][0],
                    self.__topdown_info['horizon_bbox'][1],
                    self.__topdown_info,
                    self.__height_direction)
            
            if not self.__hide_windows:
                # NOTE: show information
                if frame_current is not None:
                    c2w = pose_data.copy()
                    c2w = c2w @ OPENCV_TO_OPENGL # z-axis facing forward
                    log_info = 'Current cam pose(in opencv): \n{}'.format(c2w.round(3))
                    self.cam_pose_info.text = log_info
                    
                # NOTE: show the trajectory
                latest_location = pose_data_o3d[:3, 3].copy()
                if len(self.__traj_info['cam_centers']) == 0:
                    self.__traj_info['cam_centers'].append(latest_location)
                    if len(self.__traj_info['cam_centers']) > 1:
                        cam_traj = update_traj(self.__traj_info['cam_centers'], color_name='cool')
                else:
                    if np.linalg.norm(latest_location - self.__traj_info['cam_centers'][-1]) > 0.01:
                        self.__traj_info['cam_centers'].append(latest_location)
                        if len(self.__traj_info['cam_centers']) > 1:
                            cam_traj = update_traj(self.__traj_info['cam_centers'], color_name='cool')

                # NOTE: show keyframe frustums
                kf_frustums = None
                if (self.__keyframe_box.checked and len(self.__mapper.keyframe_list) > 0):
                    kf_frustums = self.__update_kf_frustums(self.__mapper.keyframe_list)
        
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_frame(
                    current_frustum,
                    current_agent,
                    rgbd_image,
                    current_pcd,
                    current_horizon,
                    cam_traj,
                    kf_frustums))
            Log(f'Update ui of frame used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
        
        return
    
    def __update_main_thread_ui_frame(self,
                        current_frustum:o3d.geometry.LineSet,
                        current_agent:o3d.geometry.TriangleMesh,
                        rgbd_image:o3d.geometry.Image,
                        current_pcd:o3d.geometry.PointCloud,
                        current_horizon:o3d.geometry.AxisAlignedBoundingBox,
                        cam_traj:o3d.geometry.LineSet,
                        kf_frustums:Union[None, List[o3d.geometry.LineSet]]):
        if current_frustum is not None:
            self.__widget_3d.scene.remove_geometry('current_frustum')
            self.__widget_3d.scene.add_geometry('current_frustum', current_frustum, self.__o3d_materials[CURRENT_FRUSTUM['material']])
            self.__widget_3d.scene.show_geometry('current_frustum', self.__current_frustum_box.checked)
            
        if current_agent is not None:
            self.__widget_3d.scene.remove_geometry('current_agent')
            self.__widget_3d.scene.add_geometry('current_agent', current_agent, self.__o3d_materials[CURRENT_AGENT['material']])
            self.__widget_3d.scene.show_geometry('current_agent', self.__current_agent_box.checked)
            
        if rgbd_image is not None:
            self.__rgbd_live_image.update_image(rgbd_image)
            
        if current_pcd is not None:
            self.__widget_3d.scene.remove_geometry('current_pcd')
            self.__widget_3d.scene.add_geometry(
                'current_pcd',
                current_pcd,
                self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('current_pcd', self.__current_pcd_box.checked)
            
        if current_horizon is not None:
            self.__widget_3d.scene.remove_geometry('current_horizon')
            self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
            self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
        
        if self.__cam_traj_box.checked:
            if cam_traj is not None:
                self.__widget_3d.scene.remove_geometry("cam_traj")
                self.__widget_3d.scene.add_geometry("cam_traj", cam_traj, self.__o3d_materials['unlit_line_mat'])
        else:
            self.__widget_3d.scene.remove_geometry("cam_traj")
        
        if kf_frustums is not None:
            for i, frustum in enumerate(kf_frustums):
                if self.__widget_3d.scene.has_geometry(f'kf_frustum_{i}'):
                    self.__widget_3d.scene.remove_geometry(f"kf_frustum_{i}")
                if frustum is not None:
                    self.__widget_3d.scene.add_geometry(f"kf_frustum_{i}", frustum, self.__o3d_materials[KEYFRAME_FRUSTUM['material']])
            
        return
    
    def __update_ui_visibility(self):
        local_visibility_map_o3d = None
        if self.__visibility_info['local_visibility_cv2'] is not None:
            local_visibility_map = self.__visibility_info['local_visibility_cv2'].copy()
            local_visibility_map_o3d = o3d.geometry.Image(local_visibility_map)
            if self.__save_runtime_data:
                self.__runtime_data_info['current_vis_data']['local_visibility_map'] = local_visibility_map
            
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_visibility(
                    local_visibility_map_o3d))
            Log(f'Update ui of visibility used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
            
    def __update_main_thread_ui_visibility(self,
            local_visibility_map_o3d:o3d.geometry.Image):
        if local_visibility_map_o3d is not None:
            self.__local_visibility_map_image.update_image(local_visibility_map_o3d)
                
    def render_gaussian(self):
        if self.__gaussian_for_render is None:
            return
        
        current_cam = self.__get_current_cam()
        
        timing_render_gaussian = start_timing()
        if self.__gaussian_color_type == GaussianColorType.RGBD:
            
            o3d_window_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                current_cam.image_width,
                current_cam.image_height,
                current_cam.fx,
                current_cam.fy,
                current_cam.cx,
                current_cam.cy)
            o3d_window_camera_intrinsics_o3c = o3d.core.Tensor(o3d_window_camera_intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32, device=self.__device_o3c)
            
            rgb_np, depth_np, view_w2c_gl = self.__mapper.render_o3d_image(
                self.__gaussian_for_render,
                current_cam,
                self.__gaussian_scale_slider.double_value,
                self.__gaussian_color_type)
                
            guassian_pcd = rgbd_to_pointcloud(
                rgb_np,
                depth_np,
                np.linalg.inv(OPENCV_TO_OPENGL @ view_w2c_gl @ OPENCV_TO_OPENGL),
                o3d_window_camera_intrinsics_o3c,
                1000,
                np.inf,
                self.__device_o3c)
            
            
            if not self.__widget_3d.scene.scene.has_geometry('gaussian'):
                guassian_pcd_max_num = 1920 * 1080
                guassian_pcd_init_positions = np.zeros((guassian_pcd_max_num, 3), dtype=np.float32)
                guassian_pcd_init_positions_o3c = o3d.core.Tensor(guassian_pcd_init_positions, dtype=o3d.core.Dtype.Float32, device=self.__device_o3c)
                guassian_pcd_init = o3d.t.geometry.PointCloud(guassian_pcd_init_positions_o3c)
                guassian_pcd_init.paint_uniform_color([0, 0, 0])
                self.__widget_3d.scene.scene.add_geometry(
                    'gaussian',
                    guassian_pcd_init,
                    self.__o3d_materials['gaussian_mat'])
            self.__widget_3d.scene.scene.update_geometry(
                'gaussian',
                guassian_pcd,
                self.__widget_3d.scene.scene.UPDATE_COLORS_FLAG |\
                    self.__widget_3d.scene.scene.UPDATE_POINTS_FLAG |\
                        self.__widget_3d.scene.scene.UPDATE_NORMALS_FLAG |\
                            self.__widget_3d.scene.scene.UPDATE_UV0_FLAG)
            self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
            self.__widget_3d.scene.scene.show_geometry('gaussian', True)
        else:
            if self.__widget_3d.scene.scene.has_geometry('gaussian'):
                self.__widget_3d.scene.scene.show_geometry('gaussian', False)
            render_img = self.__mapper.render_o3d_image(self.__gaussian_for_render, current_cam, self.__gaussian_scale_slider.double_value, self.__gaussian_color_type)
            
            self.__widget_3d.scene.set_background([0, 0, 0, 1], o3d.geometry.Image(render_img))
        self.render_use_time_info.text = f'Render gaussians used {end_timing(*timing_render_gaussian):.2f} ms'
    
    def get_topdown_cam(self):
        camera_pose_h = 1000.0
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ]
        )
        c2w[:3, 3] = [self.__topdown_info['world_center'][0], -camera_pose_h, self.__topdown_info['world_center'][1]]
        w2c = np.linalg.inv(c2w)
        scene_width = self.__topdown_info['topdown_world_shape'][0]
        scene_height = self.__topdown_info['topdown_world_shape'][1]
        height = self.__topdown_info['grid_map_shape'][1]
        width = self.__topdown_info['grid_map_shape'][0]
        image_topdown = torch.zeros(
            (1, int(height), int(width))
        )
        
        vfov_deg = np.rad2deg(2 * np.arctan(scene_height / (2 * camera_pose_h)))
        hfov_deg = np.rad2deg(2 * np.arctan(scene_width / (2 * camera_pose_h)))
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_topdown.shape[2])
        fy = fov2focal(FoVy, image_topdown.shape[1])
        cx = image_topdown.shape[2] // 2
        cy = image_topdown.shape[1] // 2
        T = torch.from_numpy(w2c)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_topdown.shape[1],
            W=image_topdown.shape[2],
        )
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])
        return current_cam
                
    def add_camera(self, camera:torch.Tensor, name, color=[0, 1, 0], gt=False, size=0.01):
        pose_data = camera.detach().cpu().numpy()
        C2W = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color)
            self.frustum_dict[name] = frustum
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        return frustum
                
    def receive_data(self, q:Queue):
        if q is None or q.empty():
            pass
        else:
            # update gaussian scene
            self.__gaussian_packet:GaussianPacket = q.get()
        
        if self.__gaussian_packet is None:
            return None

        if self.__gaussian_packet.has_gaussians:
            with self.__use_gaussian_condition:
                self.__use_gaussian_condition.acquire()
                self.__gaussian_for_render = {k: v.clone() for k, v in self.__gaussian_packet.params.items()}
                self.__gaussian_for_render = self.__cut_gaussian_by_height(
                                                    self.__gaussian_for_render, 
                                                    self.__height_direction_lower_bound_slider.double_value if not self.__hide_windows else self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0], 
                                                    self.__height_direction_upper_bound_slider.double_value if not self.__hide_windows else self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1])
                
                self.gaussians_num = self.__gaussian_packet.params['means3D'].shape[0]
                self.__use_gaussian_condition.release()

        if not self.__hide_windows and self.__gaussian_packet.current_frame is not None:
            frustum = self.add_camera(
                self.__gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            if self.__view_gaussians_box.checked and self.followcam_chbox.checked:
                if self.staybehind_chbox.checked:
                    self.__current_frustum_box.checked = True
                    viewpoint = frustum.view_dir_behind
                else:
                    self.__current_frustum_box.checked = False
                    viewpoint = frustum.view_dir
                self.__widget_3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
    
    def __close_all(self):
        while not self.q_main2vis.empty():
            self.q_main2vis.get()
        self.q_main2vis = None
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        self.__get_topdown_service.shutdown()
        with self.__get_topdown_condition:
            self.__get_topdown_condition.notify_all()
        if self.__get_opacity_service is not None:
            self.__get_opacity_service.shutdown()
        with self.__get_opacity_condition:
            self.__get_opacity_condition.notify_all()
        if self.__hide_windows:
            rospy.signal_shutdown('Quit')
        else:
            gui.Application.instance.quit()
        Log(f'Exit main update thread', tag='ActiveSplat')   
        
    def __get_current_cam(self):
        w2c = OPENCV_TO_OPENGL @ self.__widget_3d.scene.camera.get_view_matrix()

        image_gui = torch.zeros(
            (1, int(self.__window.size.height), int(self.__widget_3d_width))
        )
        vfov_deg = self.__widget_3d.scene.camera.get_field_of_view()
        hfov_deg = vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        T = torch.from_numpy(w2c)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_gui.shape[1],
            W=image_gui.shape[2],
        )
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])
        return current_cam
    
    def __update_dataset(self):
        with self.__local_dataset_condition:
            dataset_config = self.__local_dataset.setup()
            self.__dataset_config:GetDatasetConfigResponse = dataset_config_to_ros(dataset_config)
            rospy.Service('get_dataset_config', GetDatasetConfig, self.__get_dataset_config)
            self.__local_dataset_twist:Twist = None
            rospy.Subscriber('cmd_vel', Twist, self.__cmd_vel_callback, queue_size=1)
            movement_fail_times = 0
            movement_fail_times_pub = rospy.Publisher('movement_fail_times', Int32, queue_size=1)
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZED
            self.__local_dataset_condition.notify_all()
            while self.__global_state != GlobalState.QUIT:
                if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                    self.__local_dataset_state = self.LocalDatasetState.RUNNING
                    self.__local_dataset_condition.notify_all()
                if self.__local_dataset.is_finished():
                    self.__global_state = GlobalState.QUIT
                else:
                    self.__local_dataset_condition.wait()
                if self.__global_state == GlobalState.QUIT:
                    break
                apply_movement_flag = self.__local_dataset_twist is not None
                apply_movement_result = False
                if apply_movement_flag:
                    apply_movement_result = self.__local_dataset.apply_movement(self.__local_dataset_twist)
                    self.__local_dataset_twist = None
                step_times, step_num = self.__local_dataset.get_step_info()
                self.__local_dataset_label.text = f'Scene: {self.scene_id}\tStep: {step_times}/{step_num}'
                Log(f"Scene: {self.scene_id} Step:{step_times}/{step_num}")
                frame_numpy = self.__local_dataset.get_frame()
                frame_c2w = frame_numpy['c2w']
                pose_change_type = self.__is_pose_changed(frame_c2w)
                if pose_change_type != PoseChangeType.NONE:
                    if pose_change_type in [PoseChangeType.TRANSLATION, PoseChangeType.BOTH]:
                        movement_fail_times = 0
                    frame_quaternion = quaternion.from_rotation_matrix(frame_c2w[:3, :3])
                    frame_quaternion = quaternion.as_float_array(frame_quaternion)
                    pose_ros = PoseStamped()
                    pose_ros.header.stamp = rospy.Time.now()
                    pose_ros.header.frame_id = 'world'
                    pose_ros.pose.position.x = frame_c2w[0, 3]
                    pose_ros.pose.position.y = frame_c2w[1, 3]
                    pose_ros.pose.position.z = frame_c2w[2, 3]
                    pose_ros.pose.orientation.w = frame_quaternion[0]
                    pose_ros.pose.orientation.x = frame_quaternion[1]
                    pose_ros.pose.orientation.y = frame_quaternion[2]
                    pose_ros.pose.orientation.z = frame_quaternion[3]
                    self.__frame_c2w_last = frame_c2w
                    frame_torch = {
                        'rgb': torch.from_numpy(frame_numpy['rgb']),
                        'depth': torch.from_numpy(frame_numpy['depth']),
                        'c2w': torch.from_numpy(frame_c2w)}
                    self.__valid_depth_flag = True
                    self.__update_ui_frame(frame_torch)
                    if self.__frames_cache.empty() and self.__valid_depth_flag:
                        self.__frames_cache.put(frame_torch)
                    self.__local_dataset_pose_ros = pose_ros
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
                    movement_fail_times_pub.publish(Int32(movement_fail_times))
                elif apply_movement_flag:
                    if apply_movement_result:
                        movement_fail_times += 1
                    movement_fail_times_pub.publish(Int32(movement_fail_times))
                self.__local_dataset_condition.notify_all()
            self.__local_dataset.close()
        
    def __update_ui_topdown(self):
        topdown_free_map_o3d = None
        if self.__topdown_info['free_map_cv2'] is not None:
            topdown_free_map = self.__topdown_info['free_map_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_free_map = visualize_agent(
                    topdown_map=topdown_free_map,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_free_map_o3d = o3d.geometry.Image(topdown_free_map)
            if self.__save_runtime_data:
                self.__runtime_data_info['current_vis_data']['topdown_free_map'] = topdown_free_map
            
        topdown_free_map_binary_o3d = None
        if self.__topdown_info['free_map_binary_cv2'] is not None:
            topdown_free_map_binary = self.__topdown_info['free_map_binary_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_free_map_binary = visualize_agent(
                    topdown_map=topdown_free_map_binary,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_free_map_binary_o3d = o3d.geometry.Image(topdown_free_map_binary)
            if self.__save_runtime_data:
                self.__runtime_data_info['current_vis_data']['topdown_free_map_binary'] = topdown_free_map_binary
            
        topdown_visible_map_o3d = None
        if self.__topdown_info['visible_map_cv2'] is not None:
            topdown_visible_map = self.__topdown_info['visible_map_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_visible_map = visualize_agent(
                    topdown_map=topdown_visible_map,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            if self.__current_horizon is not None:
                cv2.rectangle(
                    topdown_visible_map,
                    np.int32(self.__current_horizon[0]),
                    np.int32(self.__current_horizon[1]),
                    (255, 0, 0),
                    1)
            topdown_visible_map_o3d = o3d.geometry.Image(topdown_visible_map)
            if self.__save_runtime_data:
                self.__runtime_data_info['current_vis_data']['topdown_visible_map'] = topdown_visible_map
                cv2.imwrite(os.path.join(self.__runtime_data_info['topdown_visible_map_dir'], f'{self.__traj_info["step_times"]}.png'), cv2.cvtColor(topdown_visible_map, cv2.COLOR_RGB2BGR))
            
        topdown_visible_map_binary_o3d = None
        if self.__topdown_info['visible_map_binary_cv2'] is not None:
            topdown_visible_map_binary = self.__topdown_info['visible_map_binary_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_visible_map_binary = visualize_agent(
                    topdown_map=topdown_visible_map_binary,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_visible_map_binary_o3d = o3d.geometry.Image(topdown_visible_map_binary)
            
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_topdown(
                    topdown_free_map_o3d,
                    topdown_free_map_binary_o3d,
                    topdown_visible_map_o3d,
                    topdown_visible_map_binary_o3d))
            Log(f'Update ui of topdown used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
        
    def __update_main_thread_ui_topdown(
        self,
        topdown_free_map_o3d:o3d.geometry.Image,
        topdown_free_map_binary_o3d:o3d.geometry.Image,
        topdown_visible_map_o3d:o3d.geometry.Image,
        topdown_visible_map_binary_o3d:o3d.geometry.Image):
        if topdown_free_map_o3d is not None:
            self.__topdown_free_map_image.update_image(topdown_free_map_o3d)
            
        if topdown_free_map_binary_o3d is not None:
            self.__topdown_free_map_image_binary.update_image(topdown_free_map_binary_o3d)
            
        if topdown_visible_map_o3d is not None:
            self.__topdown_visible_map_image.update_image(topdown_visible_map_o3d)
            
        if topdown_visible_map_binary_o3d is not None:
            self.__topdown_visible_map_image_binary.update_image(topdown_visible_map_binary_o3d)
            
    def __update_kf_frustums(self, keyframe_list:List[Dict[str, Union[int, torch.Tensor]]]):
        kf_frustums = []
        for keyframe in keyframe_list:
            # curr_keyframe = {'id': cur_frame_id, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
            pose_data = np.linalg.inv(keyframe['est_w2c'].detach().cpu().numpy()) @ OPENCV_TO_OPENGL # c2w
            pose_data_o3d = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
            keyframe_frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                KEYFRAME_FRUSTUM['scale'])
            keyframe_frustum.paint_uniform_color(KEYFRAME_FRUSTUM['color'])
            kf_frustums.append(keyframe_frustum)
        return kf_frustums
        
    # NOTE: callback functions for GUI

    def __window_on_layout(self, ctx:gui.LayoutContext):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect:gui.Rect = self.__window.content_rect

        self.__panel_control.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        x = self.__panel_control.frame.get_right()
        
        # 3D widget width
        self.__widget_3d_width = rect.width - 2*panel_width

        self.__widget_3d.frame = gui.Rect(x, rect.y, rect.get_right() - 2*panel_width, rect.height)
        self.__panel_visualize.frame = gui.Rect(self.__widget_3d.frame.get_right(), rect.y, panel_width, rect.height)

        return
        
    def __window_on_close(self) -> bool:
        self.__global_state = GlobalState.QUIT
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        self.__update_main_thread.join()
        gui.Application.instance.quit()
        return True

    def __widget_3d_on_key(self, event:gui.KeyEvent):
        if self.__global_state in [GlobalState.MANUAL_CONTROL, GlobalState.MANUAL_PLANNING] and event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.UP:
                twist_current = {
                    'linear': np.array([SPEED, 0.0, 0.0]),
                    'angular': np.zeros(3)
                }
            elif event.key == gui.KeyName.LEFT:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, 0.0, TURN])
                }
            elif event.key == gui.KeyName.RIGHT:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, 0.0, -TURN])
                }
            elif event.key == gui.KeyName.PAGE_UP:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, -TURN, 0.0])
                }
            elif event.key == gui.KeyName.PAGE_DOWN:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, TURN, 0.0])
                }
            else:
                return gui.Widget.IGNORED
            self.__apply_movement(twist_current)
            return gui.Widget.HANDLED
        return gui.Widget.IGNORED
    
    def __update_mesh(self, mesh_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.geometry.TriangleMesh:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['lit_mat']
        mesh = self.__cut_mesh_by_height(o3d.geometry.TriangleMesh(self.__o3d_meshes[mesh_name]))
        if update:
            self.__widget_3d.scene.remove_geometry(mesh_name)
            self.__widget_3d.scene.add_geometry(mesh_name, mesh, material)
            self.__widget_3d.scene.show_geometry(mesh_name, show)
        return mesh
    
    def __update_pcd(self, pointcloud_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.t.geometry.PointCloud:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['unlit_mat']
        pcd = o3d.t.geometry.PointCloud(self.__o3d_pcd[pointcloud_name])
        points = pcd.point.positions.numpy()
        points_condition = np.logical_or(
            points[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
            points[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        pcd = pcd.select_by_index(np.where(~points_condition)[0])
        if update:
            if self.__widget_3d.scene.has_geometry(pointcloud_name):
                self.__widget_3d.scene.scene.update_geometry(pointcloud_name,
                                                            pcd,
                                                            rendering.Scene.UPDATE_POINTS_FLAG +\
                                                                rendering.Scene.UPDATE_COLORS_FLAG +\
                                                                    rendering.Scene.UPDATE_NORMALS_FLAG +\
                                                                        rendering.Scene.UPDATE_UV0_FLAG)
            else:
                self.__widget_3d.scene.add_geometry(pointcloud_name, pcd, material)
            self.__widget_3d.scene.show_geometry(pointcloud_name, show)
        return pcd
    
    def __height_direction_bound_slider_callback(self, value:float, is_upper:int):
        if value == self.__open3d_gui_widget_last_state['height_direction_bound_slider'][is_upper]:
            return gui.Slider.HANDLED
        else:
            if self.__scene_mesh_box is not None:
                self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
            if self.__o3d_pcd['current_pcd'] is not None:
                current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                    'current_pcd',
                    self.__current_pcd_box.checked,
                    self.__o3d_materials['unlit_mat'])
                current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
                current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
                current_horizon.color = CURRENT_HORIZON['color']
                self.__widget_3d.scene.remove_geometry('current_horizon')
                self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
                self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
            self.__open3d_gui_widget_last_state['height_direction_bound_slider'][is_upper] = value
        if is_upper:
            self.__height_direction_lower_bound_slider.set_limits(self.__bbox_visualize[self.__height_direction[0]][0] - 0.2, value)
        else:
            self.__height_direction_upper_bound_slider.set_limits(value, self.__bbox_visualize[self.__height_direction[0]][1] + 0.1)
        return gui.Slider.HANDLED

    def __global_state_callback(self, global_state_str:str, global_state_index:int):
        global_state = GlobalState(global_state_str)
        set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(global_state_str))
        if global_state == self.__global_state:
            return gui.Combobox.HANDLED
        elif self.__global_state == GlobalState.REPLAY and not self.__hide_windows:
            self.__global_states_selectable.remove(GlobalState.REPLAY)
            self.__global_state_combobox.remove_item(GlobalState.REPLAY.value)
        self.__global_state = global_state
        return gui.Combobox.HANDLED
    
    # NOTE: ros functions
    
    def __frame_callback(self, msg:frame):
        frame_quaternion = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        frame_quaternion = quaternion.from_float_array(frame_quaternion)
        frame_translation = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        frame_c2w = np.eye(4)
        frame_c2w[:3, :3] = quaternion.as_rotation_matrix(frame_quaternion)
        frame_c2w[:3, 3] = frame_translation
        frame_c2w = convert_to_c2w_opencv(frame_c2w, self.__pose_data_type)
        if self.__is_pose_changed(frame_c2w) == PoseChangeType.NONE:
            return
        
        frame_rotation_vector = np.degrees(quaternion.as_rotation_vector(frame_quaternion))
        rospy.loginfo(f'Agent:\n\tX: {frame_translation[0]:.2f}, Y: {frame_translation[1]:.2f}, Z: {frame_translation[2]:.2f}\n\tX_angle: {frame_rotation_vector[0]:.2f}, Y_angle: {frame_rotation_vector[1]:.2f}, Z_angle: {frame_rotation_vector[2]:.2f}')
        
        if msg.rgb.encoding in ['rgb8', 'bgr8', 'rgba8', 'bgra8']:
            if msg.rgb.encoding in ['rgb8', 'bgr8']:
                channel_number = 3
            elif msg.rgb.encoding in ['rgba8', 'bgra8']:
                channel_number = 4
            else:
                raise NotImplementedError(f'Unsupported RGB encoding: {msg.rgb.encoding}')
            frame_rgb = np.frombuffer(msg.rgb.data, dtype=np.uint8).reshape(msg.rgb.height, msg.rgb.width, channel_number)
            if msg.rgb.encoding == 'bgr8':
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            elif msg.rgb.encoding == 'rgba8':
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
            elif msg.rgb.encoding == 'bgra8':
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGRA2RGB)
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
        else:
            raise NotImplementedError(f'Unsupported RGB encoding: {msg.rgb.encoding}')
        if msg.depth.encoding == '32FC1':
            frame_depth = np.frombuffer(msg.depth.data, dtype=np.float32).reshape(msg.depth.height, msg.depth.width)
            assert self.__rgbd_sensor.depth_scale == 1, 'Depth scale is not 1'
        elif msg.depth.encoding == '16UC1':
            frame_depth = np.frombuffer(msg.depth.data, dtype=np.uint16).reshape(msg.depth.height, msg.depth.width).astype(np.float32)
            assert self.__rgbd_sensor.depth_scale == 1000, 'Depth scale is not 1000'
        else:
            raise NotImplementedError(f'Unsupported Depth encoding: {msg.depth.encoding}')
        frame_depth = frame_depth / self.__rgbd_sensor.depth_scale
        
        if frame_rgb.shape[:2] != (self.__rgbd_sensor.height, self.__rgbd_sensor.width):
            frame_rgb = cv2.resize(frame_rgb, (self.__rgbd_sensor.width, self.__rgbd_sensor.height), interpolation=cv2.INTER_LINEAR)
        if frame_depth.shape[:2] != (self.__rgbd_sensor.height, self.__rgbd_sensor.width):
            frame_depth = cv2.resize(frame_depth, (self.__rgbd_sensor.width, self.__rgbd_sensor.height), interpolation=cv2.INTER_NEAREST)
        
        frame_depth = self.__preprocess_frame(frame_rgb, frame_depth, frame_c2w)
        if np.any(np.isnan(frame_depth)) or np.any(np.isinf(frame_depth)) or np.all(frame_depth == 0):
            rospy.logwarn('Depth contains NaN, Inf or all 0')
            return
        
        if self.__local_dataset_parallelized:
            while not self.__frames_cache.empty():
                try:
                    self.__frames_cache.get_nowait()
                except Empty:
                    break
            update_frames_cache = True
        elif self.__frames_cache.empty():
            update_frames_cache = True
        else:
            update_frames_cache = False
            
        if update_frames_cache:
            self.__frame_c2w_last = frame_c2w
            frame_current = {
                'rgb': torch.from_numpy(frame_rgb),
                'depth': torch.from_numpy(frame_depth.copy()),
                'c2w': torch.from_numpy(frame_c2w).float()}
            self.__update_ui_frame(frame_current)
            self.__frames_cache.put(frame_current)
        return
    
    def __preprocess_frame(self, frame_rgb:np.ndarray, frame_depth:np.ndarray, frame_c2w:np.ndarray):
        frame_depth = np.where((frame_depth > self.__depth_limit[1]) | (frame_depth < self.__depth_limit[0]), 0, frame_depth)
        return frame_depth
    
    def __apply_movement(self, twist:Dict[str, np.ndarray]):
        if self.__local_dataset is None:
            twist_msg = Twist()
            twist_msg.linear.x = twist['linear'][0]
            twist_msg.linear.y = twist['linear'][1]
            twist_msg.linear.z = twist['linear'][2]
            twist_msg.angular.x = twist['angular'][0]
            twist_msg.angular.y = twist['angular'][1]
            twist_msg.angular.z = twist['angular'][2]
            self.__cmd_vel_publisher.publish(twist_msg)
        else:
            if not self.__local_dataset_parallelized and not self.__frames_cache.empty():
                return
            with self.__local_dataset_condition:
                self.__local_dataset_twist = twist
                self.__local_dataset_condition.notify_all()
                self.__local_dataset_condition.wait()
        return
        
    def __cmd_vel_callback(self, twist:Twist):
        twist_current = {
            'linear': np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z]),
            'angular': np.array([
                twist.angular.x,
                twist.angular.y,
                twist.angular.z])}
        self.__apply_movement(twist_current)
        
    def __get_dataset_config(self, req:GetDatasetConfigRequest) -> GetDatasetConfigResponse:
        return self.__dataset_config
    
    def __get_topdown(self, req:GetTopdownRequest) -> GetTopdownResponse:
        with self.__get_topdown_condition:
            if req.arrived_flag:
                self.__get_topdown_flag = self.QueryTopdownFlag.ARRIVED
            elif self.__get_topdown_flag == self.QueryTopdownFlag.NONE:
                self.__get_topdown_flag = self.QueryTopdownFlag.RUNNING
            self.__get_topdown_condition.wait()
            if self.__global_state == GlobalState.QUIT:
                self.__get_topdown_condition.notify_all()
                return None
            free_map_binary:np.ndarray = self.__topdown_info['free_map_binary'].copy()
            visible_map_binary:np.ndarray = self.__topdown_info['visible_map_binary'].copy()
            self.__get_topdown_condition.notify_all()
        topdown_response = GetTopdownResponse()
        topdown_response.free_map = free_map_binary.flatten().tolist()
        topdown_response.visible_map = visible_map_binary.flatten().tolist()
        if req.arrived_flag:
            topdown_response.horizon_bound_min.x = self.__topdown_info['horizon_bbox'][0][0]
            topdown_response.horizon_bound_min.y = self.__topdown_info['horizon_bbox'][0][1]
            topdown_response.horizon_bound_min.z = self.__topdown_info['horizon_bbox'][0][2]
            topdown_response.horizon_bound_max.x = self.__topdown_info['horizon_bbox'][1][0]
            topdown_response.horizon_bound_max.y = self.__topdown_info['horizon_bbox'][1][1]
            topdown_response.horizon_bound_max.z = self.__topdown_info['horizon_bbox'][1][2]
        return topdown_response
    
    def __get_opacity(self, req:GetOpacityRequest) -> GetOpacityResponse:
        with self.__get_opacity_condition:
            if req.arrived_flag:
                # Global
                node_points = [{'id':req.nodes_id[idx],'position': np.array([node_point.x, node_point.y, node_point.z])} for idx, node_point in enumerate(req.nodes)]
                self.__mapper.set_voronoi_nodes(node_points)
                self.__get_visibility_flag = self.QueryVisibilityFlag.GLOBAL
                self.__get_opacity_condition.wait()
                if self.__global_state == GlobalState.QUIT:
                    self.__get_opacity_condition.notify_all()
                    return None
                invisibilities = [node['invisibility'] for node in self.__mapper.voronoi_nodes]
                volumes = [node['volume'] for node in self.__mapper.voronoi_nodes]
                opacity_response = GetOpacityResponse()
            else:
                # Local
                self.__get_visibility_flag = self.QueryVisibilityFlag.LOCAL
                self.__get_opacity_condition.wait()
                if self.__global_state == GlobalState.QUIT:
                    self.__get_opacity_condition.notify_all()
                    return None
                invisibilities = [self.local_node['invisibility']] # just one node
                volumes = [0,]
                opacity_response = GetOpacityResponse()
                if self.local_node['best_pose'] is None:
                    opacity_response.targets_frustums.append(Pose())
                else:
                    self.local_node['best_pose'] = self.local_node['best_pose'] @ OPENCV_TO_OPENGL # z-axis facing backward
                    pose = matrix_to_pose(self.local_node['best_pose'])
                    opacity_response.targets_frustums.append(pose)
                if self.__mapper.high_loss_samples_pose_c2w is not None:
                    self.__mapper.high_loss_samples_pose_c2w = self.__mapper.high_loss_samples_pose_c2w @ OPENCV_TO_OPENGL
                    pose = matrix_to_pose(self.__mapper.high_loss_samples_pose_c2w)
                    opacity_response.targets_frustums.append(pose)
                    Log(f'Add high loss samples pose.', tag='ActiveSplat')
                
            
            self.__get_opacity_condition.notify_all()
        
        opacity_response.targets_frustums_invisibility = invisibilities
        opacity_response.targets_frustums_volume = volumes
        return opacity_response
    
    def __get_topdown_config(self, req:GetTopdownConfigRequest) -> GetTopdownConfigResponse:
        topdown_config_response = GetTopdownConfigResponse()
        topdown_config_response.topdown_x_world_dim_index = self.__topdown_info['world_dim_index'][0]
        topdown_config_response.topdown_y_world_dim_index = self.__topdown_info['world_dim_index'][1]
        topdown_config_response.topdown_x_world_lower_bound = self.__topdown_info['world_2d_bbox'][0][0]
        topdown_config_response.topdown_x_world_upper_bound = self.__topdown_info['world_2d_bbox'][0][1]
        topdown_config_response.topdown_y_world_lower_bound = self.__topdown_info['world_2d_bbox'][1][0]
        topdown_config_response.topdown_y_world_upper_bound = self.__topdown_info['world_2d_bbox'][1][1]
        topdown_config_response.topdown_x_length = self.__topdown_info['grid_map_shape'][0]
        topdown_config_response.topdown_y_length = self.__topdown_info['grid_map_shape'][1]
        topdown_config_response.meter_per_pixel = self.__topdown_info['meter_per_pixel']
        return topdown_config_response
    
    def __set_mapper(self, req:SetMapperRequest) -> SetMapperResponse:
        
        kf_every_old = self.__mapper.get_kf_every()
        map_every_old = self.__mapper.get_map_every()
        if req.map_every != 0:
            map_every = req.map_every
            self.__mapper.set_map_every(map_every)
            if not self.__hide_windows:
                self.__map_every_slider.int_value = map_every
        if req.kf_every != 0:
            kf_every = req.kf_every
            self.__mapper.set_kf_every(kf_every)
            if not self.__hide_windows:
                self.__kf_every_slider.int_value = kf_every
        
        response = SetMapperResponse()
        response.kf_every_old = kf_every_old
        response.map_every_old = map_every_old
        return response
        
    # NOTE: Common Funtions
    
    def __is_pose_changed(self, frame_c2w:np.ndarray) -> PoseChangeType:
        if self.__frame_c2w_last is None:
            self.__frame_c2w_last = frame_c2w
            return PoseChangeType.BOTH
        else:
            return is_pose_changed(
                self.__frame_c2w_last,
                frame_c2w,
                self.__frame_update_translation_threshold,
                self.__frame_update_rotation_threshold)
            
    def __cut_mesh_by_height(self, mesh:o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        vertices = np.array(mesh.vertices)
        vertices_condition = np.logical_or(
            vertices[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
            vertices[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        mesh.remove_vertices_by_mask(vertices_condition)
        return mesh
    
    def __cut_gaussian_by_height(self, gaussian_params:Dict[str, torch.Tensor], upper_limit, lower_limit) -> Dict[str, torch.Tensor]:
        gauss_condition = torch.logical_or(
            -gaussian_params['means3D'][:,1] < upper_limit,
            -gaussian_params['means3D'][:,1] > lower_limit)
        gaussian_params['means3D'] = gaussian_params['means3D'][~gauss_condition]
        gaussian_params['rgb_colors'] = gaussian_params['rgb_colors'][~gauss_condition]
        gaussian_params['unnorm_rotations'] = gaussian_params['unnorm_rotations'][~gauss_condition]
        gaussian_params['logit_opacities'] = gaussian_params['logit_opacities'][~gauss_condition]
        gaussian_params['log_scales'] = gaussian_params['log_scales'][~gauss_condition]
        return gaussian_params
    
    def __save_current_data_callback(self):
        current_vis_data_dir = self.__runtime_data_info['current_vis_data_dir'] + f'/step_{self.__traj_info["step_times"]}'
        os.makedirs(current_vis_data_dir, exist_ok=True)
        if self.__runtime_data_info['current_vis_data'] is not None:
            for key, value in self.__runtime_data_info['current_vis_data'].items():
                if value.shape[2] == 3:
                    value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{str(current_vis_data_dir)}/{key}.png', value)
        Log(f'Save current data done', tag='GUI')
        return
    
    def update_target_frustums(self, targets_poses, color:List[float]=None):
        target_frustums = []
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.tab20)
        for i, pose_data_o3d in enumerate(targets_poses):
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__rgbd_sensor.width,
                self.__rgbd_sensor.height,
                self.__rgbd_sensor.intrinsics,
                np.linalg.inv(pose_data_o3d),
                scale=ROTATION_FRUSTUM['scale'],
            )
            
            if not color is None:
                frustum.paint_uniform_color(color)
            elif self.vis_cluster_box.checked:
                frustum.paint_uniform_color(mapper.to_rgba(i)[0:3])
            else:
                frustum.paint_uniform_color([88./255, 214./255, 141./255])
            target_frustums.append(frustum)
        return target_frustums
    
    def __update_voronoi_graph_trigger_callback(self, value:bool):
        if not self.__trigger_update_voronoi_graph_flag:
            self.__trigger_update_voronoi_graph_flag = True
            
    def __update_high_connectivity_nodes_trigger_callback(self, value:bool):
        if not self.__trigger_high_connectivity_nodes_flag:
            self.__trigger_high_connectivity_nodes_flag = True
            
    def __update_global_visibility_map_callback(self, value:Int32):
        if not self.__trigger_global_visibility_map_flag:
            self.__trigger_global_visibility_map_flag = True
            self.__show_node_id = value.data # node id
        