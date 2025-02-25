import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
import time
from typing import Tuple, Union, Dict
import json

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import quaternion
from omegaconf import OmegaConf
import habitat
from habitat import sims
from habitat import Simulator as HabSimulator
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from habitat_sim import Simulator as HabSimSimulator
from habitat_sim.agent import AgentState

import rospy

from src.dataloader import HABITAT_TRANSFORM_MATRIX,RGBDSensor, DatasetFormats, PoseDataType, HeightDirection, get_scene_mesh_url, compute_intrinsics
from dataloader.image_transforms import DepthFilter

class HabitatDataset(Dataset):
    
    _transform_matrix = HABITAT_TRANSFORM_MATRIX
    
    _pose_data_type = PoseDataType.C2W_OPENCV
    
    _height_direction = HeightDirection.Y_NEGATIVE
    
    def __init__(self, config:dict, user_config:dict, scene_id:str, remark:str) -> None:
        depth_scale = config['dataset']['depth_scale']
        self._env_config_url = config['env']['config']
        self._sc_factor = config['dataset']['sc_factor']
        self._step_num = config['dataset']['step_num']
        self._remark = remark
        
        with open(self._env_config_url) as f:
            env_config = yaml.safe_load(f)
        
        # NOTE: Load RGBD sensor
        rgb_sensor:dict = env_config['habitat']['simulator']['agents']['main_agent']['sim_sensors']['rgb_sensor']
        depth_sensor:dict = env_config['habitat']['simulator']['agents']['main_agent']['sim_sensors']['depth_sensor']
        
        rgb_sensor_position = rgb_sensor['position']
        depth_sensor_position = depth_sensor['position']
        assert np.allclose(rgb_sensor_position, depth_sensor_position), f'RGB ({rgb_sensor_position}) and Depth ({depth_sensor_position}) sensors positions are not the same'
        
        rgb_sensor_width = rgb_sensor['width']
        depth_sensor_width = depth_sensor['width']
        assert np.isclose(rgb_sensor_width, depth_sensor_width), f'RGB ({rgb_sensor_width}) and Depth ({depth_sensor_width}) sensors widths are not the same'
        
        rgb_sensor_height = rgb_sensor['height']
        depth_sensor_height = depth_sensor['height']
        assert np.isclose(rgb_sensor_height, depth_sensor_height), f'RGB ({rgb_sensor_height}) and Depth ({depth_sensor_height}) sensors heights are not the same'
        
        rgb_sensor_hfov = rgb_sensor['hfov']
        depth_sensor_hfov = depth_sensor['hfov']
        assert np.isclose(rgb_sensor_hfov, depth_sensor_hfov), f'RGB ({rgb_sensor_hfov}) and Depth ({depth_sensor_hfov}) sensors hfov are not the same'
        rgb_sensor_hfov = np.deg2rad(rgb_sensor_hfov)
        
        fx, fy, cx, cy = compute_intrinsics(rgb_sensor_width, rgb_sensor_height, rgb_sensor_hfov)
        
        depth_max = depth_sensor['max_depth']
        depth_min = depth_sensor['min_depth']
        
        self._rgbd_sensor = RGBDSensor(
            height=rgb_sensor_height,
            width=rgb_sensor_width,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_scale=depth_scale,
            position=np.array(rgb_sensor_position),
            downsample_factor=config['dataset']['downsample'])
        
        # NOTE: Load scene mesh
        config['dataset']['scene_id'] = config['dataset']['scene_id'] if scene_id in ['None', 'Eval'] else scene_id
        self._scene_id = config['dataset']['scene_id']
        dataset_format = DatasetFormats(config['dataset']['format'])
        dataset_root_url = user_config['datasets'][dataset_format.value]['root']
        
        self.__habitat_mesh_url, self._scene_mesh_url = get_scene_mesh_url(
            dataset_format,
            dataset_root_url,
            self._scene_id)
        self._scene_bbox = np.array(config['dataset']['bbox'])
            
        self._frame_id = 0
        self._step_times = 0
        self._finished_flag = False
        
        self._ros_to_camera_matrix = np.array(
            [
                [0, -1, 0],
                [0,  0, -1],
                [1, 0, 0]
            ]
        )
        self._camera_to_ros_matrix = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ]
        )
        results_dir_name = time.strftime('%Y-%m-%d_%H-%M-%S') + f'_{dataset_format.value}_{self._scene_id}'
        if self._remark != 'NONE':
            results_dir_name += f'_{self._remark}'
        self.__results_dir = os.path.join(PACKAGE_PATH, 'results', results_dir_name)
        if scene_id != 'Eval':
            os.makedirs(self.__results_dir, exist_ok=True)
            with open(os.path.join(self.__results_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
        self._action_file = os.path.join(self.__results_dir, 'actions.txt')
    
    def setup(self) -> Dict[str, Union[float, int, str, np.ndarray]]:
        config = habitat.get_config(self._env_config_url)
        OmegaConf.set_readonly(config, False)
        config.habitat.simulator.scene = self.__habitat_mesh_url
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        OmegaConf.set_readonly(config, True)
        assert config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1] == config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position[1]

        self._sim:Union[HabSimulator, HabSimSimulator] = sims.make_sim(config.habitat.simulator.type, config=config.habitat.simulator)
        
        self._sim.seed(0)
        
        self.reset()
        
        self._get_frame = lambda: self._sim.sensor_suite.get_observations(self._sim.get_sensor_observations())
        
        dataset_config = {
            'agent_forward_step_size': config.habitat.simulator.forward_step_size,
            'agent_turn_angle':        config.habitat.simulator.turn_angle,
            'agent_tilt_angle':        config.habitat.simulator.tilt_angle,
            'agent_height':            config.habitat.simulator.agents.main_agent.height,
            'agent_radius':            config.habitat.simulator.agents.main_agent.radius,
            'rgbd_height':             self._rgbd_sensor.height,
            'rgbd_width':              self._rgbd_sensor.width,
            'rgbd_fx':                 self._rgbd_sensor.fx,
            'rgbd_fy':                 self._rgbd_sensor.fy,
            'rgbd_cx':                 self._rgbd_sensor.cx,
            'rgbd_cy':                 self._rgbd_sensor.cy,
            'rgbd_depth_max':          self._rgbd_sensor.depth_max,
            'rgbd_depth_min':          self._rgbd_sensor.depth_min,
            'rgbd_depth_scale':        self._rgbd_sensor.depth_scale,
            'rgbd_position':           self._rgbd_sensor.position,
            'rgbd_downsample_factor':  self._rgbd_sensor.downsample_factor,
            'scene_mesh_url':          self._scene_mesh_url,
            'scene_mesh_transform':    self._transform_matrix,
            'scene_bound_min':         self._scene_bbox[0],
            'scene_bound_max':         self._scene_bbox[1],
            'pose_data_type':          self._pose_data_type.value,
            'height_direction':        self._height_direction.value,
            'results_dir':             self.__results_dir
        }
        
        return dataset_config
        
    # NOTE: we find that in habitat, the y-axis is the height axis, but the rotation matrix with identity point to negative z-axis
    def get_frame(self) -> Dict[str, Union[int, np.ndarray]]:
        '''
        depth image: unit in meters
        '''
        rgbd_data = self._get_frame()
        
        color_data_torch:torch.Tensor = rgbd_data['rgb']
        depth_data_torch:torch.Tensor = rgbd_data['depth']
        
        color_data_numpy:np.ndarray = color_data_torch.detach().cpu().numpy()
        depth_data_numpy:np.ndarray = depth_data_torch.detach().cpu().numpy()
        
        color_data_numpy = color_data_numpy.astype(np.float32) / 255.0
        depth_data_numpy = np.squeeze(depth_data_numpy)
        depth_data_numpy = depth_data_numpy.astype(np.float32) / self._rgbd_sensor.depth_scale
        depth_data_numpy = DepthFilter(
            min_depth=self._rgbd_sensor.depth_min,
            max_depth=self._rgbd_sensor.depth_max)(depth_data_numpy)
        depth_data_numpy *= self._sc_factor
        
        image_height, image_width = depth_data_numpy.shape
        color_data_numpy = cv2.resize(color_data_numpy, (image_width, image_height))
        if image_height / self._rgbd_sensor.height == image_width / self._rgbd_sensor.width > 1.0:
            color_data_numpy = cv2.resize(color_data_numpy,
                                          (self._rgbd_sensor.width, self._rgbd_sensor.height),
                                          interpolation=cv2.INTER_AREA)
            depth_data_numpy = cv2.resize(depth_data_numpy,
                                            (self._rgbd_sensor.width, self._rgbd_sensor.height),
                                            interpolation=cv2.INTER_NEAREST)
        elif image_height / self._rgbd_sensor.height == image_width / self._rgbd_sensor.width < 1.0:
            raise NotImplementedError('Downsample factor < 1.0 not implemented')
        elif image_height / self._rgbd_sensor.height != image_width / self._rgbd_sensor.width:
            raise ValueError(f'Invalid image shape: {depth_data_numpy.shape}, rgbd sensor shape: {self._rgbd_sensor.height}x{self._rgbd_sensor.width}')
        
        agent_data:AgentState = self._sim.get_agent_state()
        
        agent_position = agent_data.position.copy()
        agent_rotation = agent_data.rotation.copy()
        
        rospy.logdebug(f'frame_id: {self._frame_id}')
        agent_rotation_vector = np.degrees(quaternion.as_rotation_vector(agent_rotation))
        rospy.logdebug(f'\tAgent\n\t\tX: {agent_position[0]:.2e}, Y: {agent_position[1]:.2e}, Z: {agent_position[2]:.2e}, x_angle: {agent_rotation_vector[0]:.2f}, y_angle: {agent_rotation_vector[1]:.2f}, z_angle: {agent_rotation_vector[2]:.2f}, quat: {quaternion.as_float_array(agent_rotation).tolist()}')
        
        rgb_sensor_data = agent_data.sensor_states['rgb']
        rgb_sensor_position = rgb_sensor_data.position
        rgb_sensor_rotation = rgb_sensor_data.rotation
        depth_sensor_data = agent_data.sensor_states['depth']
        depth_sensor_position = depth_sensor_data.position
        depth_sensor_rotation = depth_sensor_data.rotation
        assert np.allclose(rgb_sensor_position, depth_sensor_position), f'RGB ({rgb_sensor_position}) and Depth ({depth_sensor_position}) sensors positions are not the same'
        assert quaternion.allclose(rgb_sensor_rotation, depth_sensor_rotation), f'RGB ({rgb_sensor_rotation}) and Depth ({depth_sensor_rotation}) sensors rotations are not the same'
        rgb_sensor_rotation_vector = np.degrees(quaternion.as_rotation_vector(rgb_sensor_rotation))
        rospy.logdebug(f'\tRGB Sensor\n\t\tX: {rgb_sensor_position[0]:.2e}, Y: {rgb_sensor_position[1]:.2e}, Z: {rgb_sensor_position[2]:.2e}, x_angle: {rgb_sensor_rotation_vector[0]:.2f}, y_angle: {rgb_sensor_rotation_vector[1]:.2f}, z_angle: {rgb_sensor_rotation_vector[2]:.2f}, quat: {quaternion.as_float_array(rgb_sensor_rotation).tolist()}')
        
        c2w = np.eye(4)
        c2w[:3, :3] = quaternion.as_rotation_matrix(rgb_sensor_rotation)
        c2w[:3, 3] = rgb_sensor_position
        
        ret = {
            'frame_id': self._frame_id,
            'c2w': c2w.astype(np.float32),
            'rgb': color_data_numpy,
            'depth': depth_data_numpy
        }
        
        self._frame_id += 1
        
        return ret
    
    def apply_movement(self, twist:Dict[str, np.ndarray]) -> bool:
        if self._step_times >= self._step_num:
            self._finished_flag = True
            return False
        linear_speed = twist['linear']
        angular_speed = twist['angular']
        apply_movement_flag = False
        if angular_speed[2] > 0:
            step_value = _DefaultHabitatSimActions.turn_left.value
            apply_movement_flag = True
        elif angular_speed[2] < 0:
            step_value = _DefaultHabitatSimActions.turn_right.value
            apply_movement_flag = True
        elif angular_speed[1] > 0:
            step_value = _DefaultHabitatSimActions.look_down.value
            apply_movement_flag = True
        elif angular_speed[1] < 0:
            step_value = _DefaultHabitatSimActions.look_up.value
            apply_movement_flag = True
        else:
            if linear_speed[0] > 0:
                step_value = _DefaultHabitatSimActions.move_forward.value
                apply_movement_flag = True
        if apply_movement_flag:
            self._sim.step(step_value)
            self._step_times += 1
            rospy.logdebug(f'step_times: {self._step_times}')
            with open(self._action_file, 'a') as f:
                f.write(f'{step_value}\n')
        return apply_movement_flag
    
    def apply_action(self, act_value) -> bool:
        self._sim.action_space
        self._sim.step(act_value)
        self._step_times += 1
        return True
    
    def reset(self) -> None:
        self._sim.reset()
        self._frame_id = 0
        self._step_times = 0
        self._finished_flag = False
    
    def close(self):
        self._sim.close()
        
    def is_finished(self) -> bool:
        return self._finished_flag
    
    def get_step_info(self) -> Tuple[int, int]:
        return self._step_times, self._step_num
    
    def get_scene_id(self) -> str:
        return self._scene_id

def get_dataset(config:dict, user_config:dict, scene_id:str, remark:str='NONE') -> Union[HabitatDataset]:
    dataset_format = DatasetFormats(config['dataset']['format'])
    if scene_id != 'Eval':
        step_num = rospy.get_param('step_num', -1)
        config['dataset']['step_num'] = config['dataset']['step_num'] if step_num == -1 else step_num
    if dataset_format in [DatasetFormats.GIBSON, DatasetFormats.MP3D, DatasetFormats.REPLICA]:
        return HabitatDataset(config, user_config, scene_id, remark)
    else:
        raise NotImplementedError(f'Dataset format {dataset_format.name} not support.')