import os
from queue import Queue
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
import numpy as np
from pathlib import Path
import shutil
import copy
import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from importlib.machinery import SourceFileLoader
from imgviz import depth2rgb
from sklearn.cluster import DBSCAN

from mapper import get_convexhull_volume, get_invisibility_clusters, MapperState, GaussianColorType, MapperType
from mapper.splatam.utils.geometryutils import relative_transformation
from mapper.splatam.utils.common_utils import seed_everything, save_params_ckpt, save_params, save_keyframes
from mapper.splatam.utils.slam_external import build_rotation, prune_gaussians, densify
from mapper.splatam.utils.recon_helpers import setup_camera
from mapper.splatam.utils.keyframe_selection import keyframe_selection_overlap
from mapper.splatam.utils.eval_helpers import report_progress
from mapper.splatam.splatam import get_loss, initialize_optimizer, initialize_params, get_pointcloud, add_new_gaussians, render, get_rendervars

from dataloader import RGBDSensor, compute_intrinsics
from utils.logging_utils import Log
from utils.gui_utils import GaussianPacket
from utils.camera_utils import Camera
from utils.pose_utils import mat_to_q_pos, rot_axis
from utils import start_timing, end_timing, OPENCV_TO_OPENGL

@dataclass
class sample():
    image:torch.Tensor
    depth_image:torch.Tensor
    transform_matrix:torch.Tensor
    
    id:int = 1
    timestamp:float  = 0.0
    has_depth:bool = True
    
    width:int = 0
    height:int = 0
    cx:float = 0.0
    cy:float = 0.0
    fl_x:float = 0.0
    fl_y:float = 0.0
    
    depth_scale:float = 0.0
    depth_width: int = 0
    depth_height: int = 0

class SplaTAM:
    def __init__(self, config:dict, rgbd_sensor:RGBDSensor, device:torch.device, q_main2vis:Queue, results_dir:str, step_num:int) -> None:
        self.__device = device
        self.__rgbd_sensor = rgbd_sensor
        self.q_main2vis = q_main2vis
        self.flag_mapping = False
        
        self.__downsample_render = config['painter']['render_rgbd_downsample']
        if self.__downsample_render > 1:
            self.__rgbd_sensor_render = RGBDSensor(
                height=rgbd_sensor.height,
                width=rgbd_sensor.width,
                fx=rgbd_sensor.fx,
                fy=rgbd_sensor.fy,
                cx=rgbd_sensor.cx,
                cy=rgbd_sensor.cy,
                depth_max=rgbd_sensor.depth_max,
                depth_min=rgbd_sensor.depth_min,
                depth_scale=rgbd_sensor.depth_scale,
                position=rgbd_sensor.position,
                downsample_factor=self.__downsample_render)
        else:
            self.__rgbd_sensor_render = rgbd_sensor
        
        self.save_path = Path(os.path.join(results_dir, 'gaussians_data'))
        
        # Load SplaTAM config parameters "config/splatam/online_habitat_sim.py"
        self.splatam_cfg_url = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, config['mapper']['splatam_cfg_path']))
        experiment = SourceFileLoader(os.path.basename(self.splatam_cfg_url), self.splatam_cfg_url).load_module()
        self.config = experiment.config
        seed_everything(seed=self.config['seed'])
        
        self.__step_num = step_num
        self.__kf_every = config['mapper']['keyframe_every']
        self.__map_every = config['mapper']['map_every']
        self.__densify_downscale_factor = config['mapper']['densify_downscale_factor']
        self.__mapping_window_size = config['mapper']['mapping_window_size']
        self.__mapping_iters = config['mapper']['mapping_iters']
        self.__cluster_invisibility_threshold = config['mapper']['cluster_invisibility_threshold']
        
        self.__mapping_idx = None
        self.__tracking_idx = 0
        self.__est_c2w_data = torch.zeros((self.__step_num, 4, 4)).to(self.__device)
        self.__flag_mapper_finished = False
        self.high_loss_samples_pose_c2w = None
        self.non_presence_depth_mask_cv2 = None

        self.splatam_init()
    
    def splatam_init(self):
        rgb_path = self.save_path.joinpath("rgb")
        if rgb_path.exists():
            shutil.rmtree(self.save_path)
        self.images_dir = rgb_path
        self.manifest = {
            "fl_x":  0.0,
            "fl_y":  0.0,
            "cx": 0.0,
            "cy": 0.0,
            "w": 0.0,
            "h": 0.0,
            "frames": []
        }

        self.keyframe_list = []
        self.keyframe_time_indices = []

        self.gt_w2c_all_frames = []
        self.tracking_iter_time_sum = 0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0
        self.mapping_iter_time_count = 0
        self.tracking_frame_time_sum = 0
        self.tracking_frame_time_count = 0
        self.mapping_frame_time_sum = 0
        self.mapping_frame_time_count = 0
        self.params = None
        self.config['data']['desired_image_width'] = int(self.__rgbd_sensor.width)
        self.config['data']['desired_image_height'] = int(self.__rgbd_sensor.height)
        self.config['data']['densification_image_width'] = int(self.__rgbd_sensor_render.width / self.__densify_downscale_factor)
        self.config['data']['densification_image_height'] = int(self.__rgbd_sensor_render.height / self.__densify_downscale_factor)
        Log("splatam init success!")
        
    def run(self, batch:Union[Dict[str, torch.Tensor], None]) -> MapperState:
        if self.__mapping_idx is not None:
            mapping_idx_cur = self.__mapping_idx + 1
        
        if batch is None or self.__flag_mapper_finished == True:
            batch_copy = None
            mapper_state = MapperState.MAPPING
            return mapper_state
        else:
            assert batch['frame_id'] == self.__tracking_idx, f'Frame id must be the same, but got {batch["frame_id"]} and {self.__tracking_idx}'
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.__device)
            if self.__tracking_idx < self.__step_num:
                self.__est_c2w_data[self.__tracking_idx] = batch['c2w'].to(self.__device)
            batch_copy = batch.copy()
            self.__tracking_idx += 1
            if self.__mapping_idx is None:
                mapper_state = MapperState.BOOTSTRAP
                self.__mapping_idx = 0
            elif self.__tracking_idx > mapping_idx_cur and self.__tracking_idx <= self.__step_num:
                self.__mapping_idx = mapping_idx_cur
                mapper_state = MapperState.MAPPING
            else:
                mapper_state = MapperState.IDLE
                
        if mapper_state == MapperState.BOOTSTRAP:
            self.__mapping(batch_copy, self.__mapping_idx)
        elif mapper_state == MapperState.MAPPING:
            self.__mapping(batch_copy, self.__mapping_idx)
        elif mapper_state == MapperState.IDLE:
            pass
        else:
            raise NotImplementedError(f'Unsupported mapper state: {mapper_state}')
        
        return mapper_state
        
    def __get_np_data(self, batch:Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_np:np.ndarray = batch['rgb'].detach().cpu().numpy()
        depth_np:np.ndarray = batch['depth'].detach().cpu().numpy()
        c2w:np.ndarray = batch['c2w'].detach().cpu().numpy()
        c2w = c2w @ OPENCV_TO_OPENGL # z-axis facing forward
        
        return rgb_np, depth_np, c2w

    @torch.no_grad()
    def get_high_loss_samples(self, batch:Dict[str, torch.Tensor], hfov=90, vfov=90):
        high_loss_samples_pose_c2w = None
        _, _, view_c2w = self.__get_np_data(batch)
        view_w2c =  np.linalg.inv(view_c2w)
        
        scene_data, scene_depth_data = get_rendervars(self.params, view_w2c)
        
        k = copy.deepcopy(self.__rgbd_sensor.intrinsics)
        viz_cfg = copy.deepcopy(self.config['viz'])
        viz_cfg['viz_w'] = self.__rgbd_sensor.width
        viz_cfg['viz_h'] = self.__rgbd_sensor.height

        im, depth, opacity, _, = render(view_w2c, k, scene_data, scene_depth_data, viz_cfg, 1.0)
        
        # get mask
        if im.shape[0] != batch['rgb'].shape[0]:
            im = im.permute(1, 2, 0).detach().cpu()
        depth = depth.detach().cpu()
        if batch['depth'].dim() == 2:
            batch['depth'] = batch['depth'].unsqueeze(0)
        
        if im.device != batch['rgb'].device:
            im = im.to(batch['rgb'].device)
        if depth.device != batch['depth'].device:
            depth = depth.to(batch['depth'].device)
            opacity = opacity.to(batch['depth'].device)
            
        depth_diff = (depth - batch['depth']).abs().squeeze() # (H, W)
        
        depth_error = depth_diff * (batch['depth'] > 0)
        non_presence_depth_mask = (depth > batch['depth']) * (depth_error > 0.3) * (opacity[0, :, :] > 0.8) # NOTE: mask criterion
        non_presence_depth_mask_np = non_presence_depth_mask.cpu().numpy().astype(np.uint8)
        non_presence_depth_mask_np = np.squeeze(non_presence_depth_mask_np, axis=0)
        non_presence_depth_mask_np = cv2.resize(non_presence_depth_mask_np, (hfov, vfov), interpolation=cv2.INTER_LINEAR)
        non_presence_depth_points = np.column_stack(np.where(non_presence_depth_mask_np > 0))
        if len(non_presence_depth_points) == 0:
            return high_loss_samples_pose_c2w # None
        
        if np.sum(non_presence_depth_mask_np) > 20:
            cluster_centers = []
            cluster_invisibilities = []
            valid_clusters = []
            # DBSCAN clustering
            dbscan = DBSCAN(eps=5, min_samples=10)
            clusters = dbscan.fit_predict(non_presence_depth_points)
            
            for cluster in set(clusters):
                if cluster != -1:
                    points = non_presence_depth_points[clusters == cluster]
                    center = points.mean(axis=0)
                    invisibility_sum = np.sum(non_presence_depth_mask_np[points[:, 0], points[:, 1]])
                    if invisibility_sum > self.__cluster_invisibility_threshold:
                        cluster_centers.append(center)
                        cluster_invisibilities.append(invisibility_sum)
                        valid_clusters.append(cluster)
            if len(cluster_invisibilities) > 0:
                max_area_index = np.argmax(cluster_invisibilities)  
                max_area_center = cluster_centers[max_area_index]  
                center_vec = np.array([max_area_center[1] / non_presence_depth_mask_np.shape[1] * hfov - hfov / 2, max_area_center[0] / non_presence_depth_mask_np.shape[0] * vfov- vfov / 2])
                # cv2.imwrite('non_presence_depth_mask_np.png',non_presence_depth_mask_np * 255) # save mask
                self.non_presence_depth_mask_cv2 = non_presence_depth_mask_np * 255
                horizontal_angle = np.deg2rad(center_vec[0])
                vertical_angle = np.deg2rad(center_vec[1])
                if np.abs(horizontal_angle) > np.deg2rad(5) or np.abs(vertical_angle) > np.deg2rad(5):
                    high_loss_samples_pose_c2w = rot_axis(view_c2w, 'y', horizontal_angle)
                    high_loss_samples_pose_c2w = rot_axis(high_loss_samples_pose_c2w, 'x', vertical_angle)
            
        return high_loss_samples_pose_c2w
    
    def __mapping(self, batch:Union[Dict[str, torch.Tensor], None], cur_frame_id:int=None):
        rgb_np, depth_np, c2w = self.__get_np_data(batch)
        if self.params is not None:
            batch_copy = batch.copy()
            self.high_loss_samples_pose_c2w= self.get_high_loss_samples(batch_copy)
    
        rgb_np = (rgb_np * 255).astype(np.uint8) # (H, W, 3)
        rgb_np = rgb_np.reshape(self.__rgbd_sensor.height, self.__rgbd_sensor.width, 3)
        depth_np = depth_np.astype(np.float32)
        depth_np = depth_np.reshape(self.__rgbd_sensor.height, self.__rgbd_sensor.width)
        w2c = np.linalg.inv(c2w)
        quat, position = mat_to_q_pos(w2c)
        
        if cur_frame_id == 0:
            Log("Initialize sample using first frame...")
            assert batch['frame_id'] == self.__mapping_idx == 0, f'Everything must be 0, but got batch frame id {batch["frame_id"]} and mapping idx {self.__mapping_idx}'
            self.sample = sample(image=batch['rgb'],depth_image=batch['depth'],transform_matrix=batch['c2w'])
            self.sample.width = self.__rgbd_sensor.width
            self.sample.height = self.__rgbd_sensor.height
            self.sample.depth_width = self.__rgbd_sensor.width
            self.sample.depth_height = self.__rgbd_sensor.height
            self.sample.cx = self.__rgbd_sensor.cx
            self.sample.cy = self.__rgbd_sensor.cy
            self.sample.fl_x = self.__rgbd_sensor.fx
            self.sample.fl_y = self.__rgbd_sensor.fy
            self.sample.depth_scale = self.__rgbd_sensor.depth_scale
            
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(exist_ok=True)
            self.manifest["w"] = self.sample.width
            self.manifest["h"] = self.sample.height
            self.manifest["cx"] = self.sample.cx
            self.manifest["cy"] = self.sample.cy
            self.manifest["fl_x"] = self.sample.fl_x
            self.manifest["fl_y"] = self.sample.fl_y
            self.manifest["integer_depth_scale"] = float(self.sample.depth_scale)/65535.0
            if self.sample.has_depth:
                self.depth_dir = self.save_path.joinpath("depth")
                self.depth_dir.mkdir(exist_ok=True)
        else:
            self.sample.image = batch['rgb']
            self.sample.depth_image = batch['depth']
            self.sample.transform_matrix = batch['c2w']
        # RGB
        image = np.asarray(rgb_np, dtype=np.uint8).reshape((self.sample.height, self.sample.width, 3))
        image_save_path = str(self.images_dir.joinpath(f"{str(cur_frame_id).zfill(4)}.png"))
        cv2.imwrite(image_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Depth
        save_depth = None
        if self.sample.has_depth:
            # Save Depth Image(16-bit PNG)
            save_depth = (depth_np * 1000).astype(np.uint16)
            save_depth = cv2.resize(save_depth, dsize=(
                self.sample.width, self.sample.height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(self.depth_dir.joinpath(f"{str(cur_frame_id).zfill(4)}.png")), save_depth)
            curr_depth = depth_np.astype(np.float32)
        else:
            print("No Depth Image Received. Skipping Frame...")
            return
        
        # Poses for saving dataset, as referenced in SplaTAM
        X_WV = np.asarray(w2c,
                            dtype=np.float32).reshape((4, 4)).T
        frame = {
            "transform_matrix": X_WV.tolist(),
            "file_path": f"rgb/{str(cur_frame_id).zfill(4)}.png",
            "fl_x": self.sample.fl_x,
            "fl_y": self.sample.fl_y,
            "cx": self.sample.cx,
            "cy": self.sample.cy,
            "w": self.sample.width,
            "h": self.sample.height
        }
        if save_depth is not None:
            frame["depth_path"] = f"depth/{str(cur_frame_id).zfill(4)}.png"
        self.manifest["frames"].append(frame)
        
        # Convert Pose to GradSLAM format
        gt_pose = torch.from_numpy(OPENCV_TO_OPENGL @ X_WV @ OPENCV_TO_OPENGL).float()
        if cur_frame_id == 0:
            self.first_abs_gt_pose = gt_pose
        gt_pose = relative_transformation(self.first_abs_gt_pose.unsqueeze(0), gt_pose.unsqueeze(0), orthogonal_rotations=False)
        gt_w2c = torch.linalg.inv(gt_pose[0])
        self.gt_w2c_all_frames.append(gt_w2c)
        
        # Initialize Tracking & Mapping Resolution Data (Downscaled)
        color = cv2.resize(image, dsize=(
            self.config['data']['desired_image_width'], self.config['data']['desired_image_height']), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(curr_depth, dsize=(
            self.config['data']['desired_image_width'], self.config['data']['desired_image_height']), interpolation=cv2.INTER_NEAREST)
        depth = np.expand_dims(depth, -1)
        color = torch.from_numpy(color).cuda().float()
        color = color.permute(2, 0, 1) / 255
        depth = torch.from_numpy(depth).cuda().float()
        depth = depth.permute(2, 0, 1)
        if cur_frame_id == 0:
            self.intrinsics = torch.tensor([[self.sample.fl_x, 0, self.sample.cx], [0, self.sample.fl_y, self.sample.cy], [0, 0, 1]]).cuda().float()
            self.intrinsics[2, 2] = 1.0
            
            self.first_frame_w2c = torch.eye(4).cuda().float()
            self.init_frame_w2c = torch.eye(4).cuda().float()
            self.init_frame_w2c[:3, :3] = build_rotation(torch.tensor([[quat.w, quat.x, quat.y, quat.z]]).to(self.__device))
            self.init_frame_w2c[:3, 3] = torch.tensor([[position[0], position[1], position[2]]]).to(self.__device)
            self.cam = setup_camera(color.shape[2], color.shape[1], self.intrinsics.cpu().numpy(), self.first_frame_w2c.cpu().numpy())
            Log("Initialize Camera intrinsics success.")
            
        # Initialize Densification Resolution Data
        densify_color = cv2.resize(image, dsize=(
            self.config['data']['densification_image_width'], self.config['data']['densification_image_height']), interpolation=cv2.INTER_LINEAR)
        densify_depth = cv2.resize(curr_depth, dsize=(
            self.config['data']['densification_image_width'], self.config['data']['densification_image_height']), interpolation=cv2.INTER_NEAREST)
        densify_depth = np.expand_dims(densify_depth, -1)
        densify_color = torch.from_numpy(densify_color).cuda().float()
        densify_color = densify_color.permute(2, 0, 1) / 255
        densify_depth = torch.from_numpy(densify_depth).cuda().float()
        densify_depth = densify_depth.permute(2, 0, 1)
        if cur_frame_id == 0:
            self.densify_intrinsics = torch.tensor([[self.sample.fl_x, 0, self.sample.cx], [0, self.sample.fl_y, self.sample.cy], [0, 0, 1]]).cuda().float()
            self.densify_intrinsics = self.densify_intrinsics / self.__densify_downscale_factor
            self.densify_intrinsics[2, 2] = 1.0
            self.densify_cam = setup_camera(densify_color.shape[2], densify_color.shape[1], self.densify_intrinsics.cpu().numpy(), self.first_frame_w2c.cpu().numpy())
        
        # Initialize Params for first time step
        if cur_frame_id == 0:
            # Get Initial Point Cloud
            mask = (densify_depth > 0) # Mask out invalid depth values
            mask = mask.reshape(-1)
            init_pt_cld, mean3_sq_dist = get_pointcloud(densify_color, densify_depth, self.densify_intrinsics, self.init_frame_w2c, 
                                                        mask=mask, compute_mean_sq_dist=True, 
                                                        mean_sq_dist_method=self.config['mean_sq_dist_method'])
            self.params, self.variables = initialize_params(init_pt_cld, self.__step_num, mean3_sq_dist, self.config['gaussian_distribution'])
            self.variables['scene_radius'] = torch.max(densify_depth)/self.config['scene_radius_depth_ratio']
        
        # Initialize Mapping & Tracking for current frame
        iter_time_idx = cur_frame_id
        curr_gt_w2c = self.gt_w2c_all_frames
        curr_data = {'cam': self.cam, 'im': color, 'depth':depth, 'id': iter_time_idx, 
                        'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Optimization Iterations
        iter_per_frame = int(self.__mapping_iters // self.__map_every)
        if iter_per_frame == 0 and cur_frame_id % self.__map_every == 0:
            iter_per_frame = self.__mapping_iters
            
        # NOTE: Tracking Skip
        with torch.no_grad():
            quat_tensor = torch.tensor([quat.w, quat.x, quat.y, quat.z]).to('cuda:0')
            position_tensor = torch.tensor([[position[0], position[1], position[2]]]).to('cuda:0')
            
            self.params['cam_unnorm_rots'][..., cur_frame_id] = quat_tensor.detach().clone()
            self.params['cam_trans'][..., cur_frame_id] = position_tensor.detach().clone()         
                
        # Densification & KeyFrame-based Mapping
        if cur_frame_id == 0 or (cur_frame_id+1) % self.__map_every == 0:
            # Densification
            if self.config['mapping']['add_new_gaussians'] and cur_frame_id > 0:
                densify_curr_data = {'cam': self.densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': cur_frame_id, 
                            'intrinsics': self.densify_intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}

                # Add new Gaussians to the scene based on the Silhouette
                self.params, self.variables = add_new_gaussians(self.params, self.variables, densify_curr_data, 
                                                    self.config['mapping']['sil_thres'], cur_frame_id,
                                                    self.config['mean_sq_dist_method'], self.config['gaussian_distribution'])
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., cur_frame_id].detach())
                curr_cam_tran = self.params['cam_trans'][..., cur_frame_id].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = self.__mapping_window_size-2
                self.selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1], num_keyframes)
                selected_time_idx = [self.keyframe_list[frame_idx]['id'] for frame_idx in self.selected_keyframes]
                if len(self.keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(self.keyframe_list[-1]['id'])
                    self.selected_keyframes.append(len(self.keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(cur_frame_id)
                self.selected_keyframes.append(-1)
                Log(f"\nSelected Keyframes at Frame {cur_frame_id}: {selected_time_idx}", tag='ActiveSplat')

            # Reset Optimizer & Learning Rates for Full Map Optimization
            self.optimizer = initialize_optimizer(self.params, self.config['mapping']['lrs'], tracking=False) 

        # NOTE: Mapping
        timing_mapping = start_timing()
        mapping_start_time = time.time()
        if iter_per_frame > 0:
            progress_bar = tqdm(range(iter_per_frame), desc=f"Mapping Time Step: {cur_frame_id}")
        for iter in range(iter_per_frame):
            iter_start_time = time.time()
            # Randomly select a frame until current time step amongst keyframes
            rand_idx = np.random.randint(0, len(self.selected_keyframes))
            selected_rand_keyframe_idx = self.selected_keyframes[rand_idx]
            if selected_rand_keyframe_idx == -1:
                # Use Current Frame Data
                iter_time_idx = cur_frame_id
                iter_color = color
                iter_depth = depth
            else:
                # Use Keyframe Data
                iter_time_idx = self.keyframe_list[selected_rand_keyframe_idx]['id']
                iter_color = self.keyframe_list[selected_rand_keyframe_idx]['color']
                iter_depth = self.keyframe_list[selected_rand_keyframe_idx]['depth']
            iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx + 1]
            iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                        'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
            # Loss for current frame
            loss, self.variables, losses = get_loss(self.params, iter_data, self.variables, iter_time_idx, self.config['mapping']['loss_weights'],
                                                    self.config['mapping']['use_sil_for_loss'], self.config['mapping']['sil_thres'],
                                                    self.config['mapping']['use_l1'], self.config['mapping']['ignore_outlier_depth_loss'], mapping=True)
            # Backprop
            loss.backward()
            with torch.no_grad():
                # Prune Gaussians
                if self.config['mapping']['prune_gaussians']:
                    self.params, self.variables = prune_gaussians(self.params, self.variables, self.optimizer, iter, self.config['mapping']['pruning_dict'])
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['mapping']['use_gaussian_splatting_densification']:
                    self.params, self.variables = densify(self.params, self.variables, self.optimizer, iter, self.config['mapping']['densify_dict'])
                # Optimizer Update
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Report Progress
                if self.config['report_iter_progress']:
                    report_progress(self.params, iter_data, iter + 1, progress_bar, iter_time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                    mapping=True, online_time_idx=cur_frame_id)
                else:
                    progress_bar.update(1)
            # Update the runtime numbers
            iter_end_time = time.time()
            self.mapping_iter_time_sum += iter_end_time - iter_start_time
            self.mapping_iter_time_count += 1
        if iter_per_frame > 0:
            Log(f'Mapping Iteration Time: {end_timing(*timing_mapping):.2f} ms')
            progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
            self.mapping_frame_time_count += 1

            if cur_frame_id == 0 or (cur_frame_id+1) % self.config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {cur_frame_id}")
                    with torch.no_grad():
                        report_progress(self.params, curr_data, 1, progress_bar, cur_frame_id, sil_thres= self.config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=cur_frame_id)
                    progress_bar.close()
                except:
                    ckpt_output_dir = self.save_path.joinpath("checkpoints")
                    os.makedirs(ckpt_output_dir, exist_ok=True)
                    save_params_ckpt(self.params, ckpt_output_dir, cur_frame_id)
                    print('Failed to evaluate trajectory.')

        # Add frame to keyframe list
        if ((cur_frame_id == 0) or ((cur_frame_id+1) % self.__kf_every == 0) or \
                    (cur_frame_id == self.__step_num-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., cur_frame_id].detach())
                curr_cam_tran = self.params['cam_trans'][..., cur_frame_id].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                curr_keyframe = {'id': cur_frame_id, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                self.keyframe_list.append(curr_keyframe)
                self.keyframe_time_indices.append(cur_frame_id)
        
        # Checkpoint every iteration
        if cur_frame_id % self.config["checkpoint_interval"] == 0 and self.config['save_checkpoints']:
            ckpt_output_dir = self.save_path.joinpath("checkpoints")
            save_params_ckpt(self.params, ckpt_output_dir, cur_frame_id)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{cur_frame_id}.npy"), np.array(self.keyframe_time_indices))

        torch.cuda.empty_cache()
        cur_frame_id += 1
        cur_frame_id = cur_frame_id
        
        with torch.no_grad():
            self.q_main2vis.put(
                    (GaussianPacket(
                        self.params,
                        batch['c2w'])),
                    block=False,
                )
        
    def post_processing(self):
        # Compute Average Runtimes
        if self.mapping_iter_time_count == 0:
            self.mapping_iter_time_count = 1
            self.mapping_frame_time_count = 1
        mapping_iter_time_avg = self.mapping_iter_time_sum / self.mapping_iter_time_count
        mapping_frame_time_avg = self.mapping_frame_time_sum / self.mapping_frame_time_count
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")

        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        self.params['org_width'] = self.config['data']["desired_image_width"]
        self.params['org_height'] = self.config['data']["desired_image_height"]
        self.params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in self.gt_w2c_all_frames:
            self.params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        self.params['gt_w2c_all_frames'] = np.stack(self.params['gt_w2c_all_frames'], axis=0)
        self.params['keyframe_time_indices'] = np.array(self.keyframe_time_indices)

        # Trim excess content based on actual frame count for 3DGS offline optimization
        actual_frame_count = self.params['gt_w2c_all_frames'].shape[0]
        for key in ['cam_trans', 'cam_unnorm_rots']:
            if self.params[key].shape[-1] > actual_frame_count:
                self.params[key] = self.params[key][..., :actual_frame_count]

        # Save Parameters
        save_params(self.params, self.save_path)
        if self.config['mapping']['save_keyframes']:
            self.keyframes_dir = self.save_path.joinpath("keyframes")
            save_keyframes(self.keyframe_list, self.keyframes_dir, self.__rgbd_sensor)
        shutil.copy(self.splatam_cfg_url, os.path.join(self.save_path, "config.py"))
        print("Saved SplaTAM results to: ", self.save_path)
        
    def get_step_num(self) -> int:
        return int(self.__step_num)
    
    def get_kf_every(self) -> int:
        return int(self.__kf_every)

    def set_kf_every(self, kf_every:int) -> None:
        self.__kf_every = kf_every
    
    def get_map_every(self) -> int:
        return int(self.__map_every)
    
    def set_map_every(self, map_every:int) -> None:
        self.__map_every = map_every
    
    def get_mapping_iters(self) -> int:
        return int(self.__mapping_iters)
    
    def get_mapper_type(self) -> str:
        return MapperType.SplaTAM
    
    def set_voronoi_nodes(self, voronoi_nodes:List[Dict[str, None]]) -> None:
        self.voronoi_nodes = voronoi_nodes
    
    @torch.no_grad()
    def render_rgbd(self, batch:Dict[str, torch.Tensor], scale_modifier=1.0):
        
        _, _, view_c2w = self.__get_np_data(batch)
        view_w2c =  np.linalg.inv(view_c2w)
        
        scene_data, scene_depth_data = get_rendervars(self.params, view_w2c)
        
        k = copy.deepcopy(self.__rgbd_sensor.intrinsics)
        viz_cfg = copy.deepcopy(self.config['viz'])
        viz_cfg['viz_w'] = self.__rgbd_sensor_render.width
        viz_cfg['viz_h'] = self.__rgbd_sensor_render.height
        k[0, 0] = self.__rgbd_sensor_render.fx
        k[1, 1] = self.__rgbd_sensor_render.fy
        k[0, 2] = self.__rgbd_sensor_render.cx
        k[1, 2] = self.__rgbd_sensor_render.cy

        im, depth, _, _, = render(view_w2c, k, scene_data, scene_depth_data, viz_cfg, scale_modifier)
        
        color_vis:np.ndarray = torch.permute(im, (1, 2, 0)).detach().cpu().numpy()
        color_vis = color_vis.reshape(self.__rgbd_sensor_render.height, self.__rgbd_sensor_render.width, 3)
        color_vis = np.clip(color_vis, 0, 1)
        color_vis = np.uint8(color_vis * 255)
        
        depth_vis:np.ndarray = depth.float().detach().cpu().numpy()
        depth_vis = depth_vis.reshape(self.__rgbd_sensor_render.height, self.__rgbd_sensor_render.width)
        depth_vis = depth2rgb(depth_vis, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max)
        
        return color_vis, depth_vis
    
    @torch.no_grad()
    def render_o3d_image(self, params:dict, current_cam:Camera, scale_modifier=1.0, gaussian_color_type:GaussianColorType=GaussianColorType.Color, is_original=False):
        view_w2c_gl = torch.eye(4).cuda().float()
        view_w2c_gl[:3, :3] = current_cam.R
        view_w2c_gl[:3, 3] = current_cam.T
        
        view_w2c_gl = view_w2c_gl.detach().cpu().numpy()
        view_w2c = view_w2c_gl @ OPENCV_TO_OPENGL
        
        scene_data, scene_depth_data = get_rendervars(params, view_w2c)
        
        k = copy.deepcopy(self.__rgbd_sensor.intrinsics)
        k[0, 0] = current_cam.fx
        k[1, 1] = current_cam.fy
        k[0, 2] = current_cam.cx
        k[1, 2] = current_cam.cy
        
        ui_cam_cfg = copy.deepcopy(self.config['viz'])
        ui_cam_cfg['viz_w'] = current_cam.image_width
        ui_cam_cfg['viz_h'] = current_cam.image_height
        
        im, depth, opacity, _, = render(view_w2c, k, scene_data, scene_depth_data, ui_cam_cfg, scale_modifier)
        
        # Choose the type of Gaussian to render
        if gaussian_color_type == GaussianColorType.Color:
            rgb_np = (
                (torch.clamp(im, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            return rgb_np
        elif gaussian_color_type == GaussianColorType.Depth:
            depth_vis:np.ndarray = depth[0, :, :].float().detach().cpu().numpy()
            depth_vis = depth2rgb(depth_vis, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max)
            return depth_vis
        elif gaussian_color_type == GaussianColorType.Opacity:
            opacity = opacity[0, :, :].detach().cpu().numpy()
            if is_original:
                return opacity
            max_opacity = np.max(opacity)
            opacity = depth2rgb(
                opacity, min_value=0.0, max_value=max_opacity, colormap="jet"
            )
            opacity = torch.from_numpy(opacity)
            opacity = torch.permute(opacity, (2, 0, 1)).float()
            opacity = (opacity).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            return opacity
        
        elif gaussian_color_type == GaussianColorType.RGBD:
            rgb_np = (
                (torch.clamp(im, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            depth_np = depth[0, :, :].float().detach().cpu().numpy()
            return rgb_np, depth_np, view_w2c_gl
    
    @torch.no_grad()
    def get_global_invisibility(self, batch:Dict[str, torch.Tensor], node_id, position:np.ndarray, scale_modifier=1.0, show_image=False):
        _, _, view_c2w = self.__get_np_data(batch)
        assert position.shape == (3,), f'Position must be a numpy array with shape (3,), but got {position.shape}'
        if (position == np.zeros(3)).all():
            return None, 0, 0
        view_c2w[0,3] = position[0]
        view_c2w[2,3] = position[2] # use agent camera height
        
        # NOTE: Use one camera vertically (large hfov) and multiple cameras horizontally
        hfov = 120
        vfov = 150
        width = 120
        height = 150
        fx, fy, cx, cy = compute_intrinsics(width, height, np.deg2rad(hfov), np.deg2rad(vfov)) # 1 pixel represents 1 degree in rotation
        look_around_k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        look_around_cfg = copy.deepcopy(self.config['viz'])
        look_around_cfg['viz_w'] = width
        look_around_cfg['viz_h'] = height
        for i in range(int(360 / hfov)):
            view_c2w_tmp = rot_axis(view_c2w, 'y', np.deg2rad(hfov*i))
            view_w2c_tmp = np.linalg.inv(view_c2w_tmp)
            scene_data, scene_depth_data = get_rendervars(self.params, view_w2c_tmp)
            im, depth, opacity, _, = render(view_w2c_tmp, look_around_k, scene_data, scene_depth_data, look_around_cfg, scale_modifier)
            im_np = (
                (torch.clamp(im, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            if i == 0:
                opacity_np = opacity[0, :, :].detach().cpu().numpy()
                rgb_np = im_np.copy()
                depth_np = depth.float().detach().permute(1, 2, 0).cpu().numpy()
            else:
                opacity_np = np.hstack((opacity_np, opacity[0, :, :].detach().cpu().numpy()))
                rgb_np = np.hstack((rgb_np, im_np))
                depth_np = np.hstack((depth_np, depth.float().detach().permute(1, 2, 0).cpu().numpy()))

        # NOTE: Compute invisibility
        invisibility_np = 1 - opacity_np
        sum_invisibility,sum_volume = get_convexhull_volume(depth_np, invisibility_np, look_around_k)
        
        if show_image:
            max_opacity = np.max(opacity_np)
            opacity_np = depth2rgb(
                opacity_np, min_value=0.0, max_value=max_opacity, colormap="jet"
            )
            opacity_np = torch.from_numpy(opacity_np)
            opacity_np = torch.permute(opacity_np, (2, 0, 1)).float()
            opacity_np = (opacity_np).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        
        rgb_opacity_np = None
        if show_image:
            # cv2.putText(opacity_np, f"Sum Invisibility: {sum_invisibility:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.putText(opacity_np, f"Sum Volume: {sum_volume:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.putText(opacity_np, f"Node ID: {node_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            rgb_opacity_np = np.vstack((rgb_np, opacity_np))
        
        return rgb_opacity_np, sum_invisibility, sum_volume
    
    @torch.no_grad()
    def get_local_invisibility(self, batch:Dict[str, torch.Tensor], scale_modifier=1.0, show_image=False):
        _, _, view_c2w = self.__get_np_data(batch)
        
        hfov = 120
        vfov = 150
        width = 120
        height = 150
        fx, fy, cx, cy = compute_intrinsics(width, height, np.deg2rad(hfov), np.deg2rad(vfov)) # 1 pixel represents 1 degree in rotation
        look_around_k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        look_around_cfg = copy.deepcopy(self.config['viz'])
        look_around_cfg['viz_w'] = width
        look_around_cfg['viz_h'] = height
        for i in range(int(360 / hfov)):
            view_c2w_tmp = rot_axis(view_c2w, 'y', np.deg2rad(hfov*i))
            view_w2c_tmp = np.linalg.inv(view_c2w_tmp)
            scene_data, scene_depth_data = get_rendervars(self.params, view_w2c_tmp)
            im, depth, opacity, _, = render(view_w2c_tmp, look_around_k, scene_data, scene_depth_data, look_around_cfg, scale_modifier)
            im_np = (
                (torch.clamp(im, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            if i == 0:
                opacity_np = opacity[0, :, :].detach().cpu().numpy()
                rgb_np = im_np.copy()
            else:
                opacity_np = np.hstack((opacity_np, opacity[0, :, :].detach().cpu().numpy()))
                rgb_np = np.hstack((rgb_np, im_np))

        # NOTE: Compute invisibility
        invisibility_np = 1 - opacity_np
        sum_invisibility = np.sum(invisibility_np)
        
        if show_image:
            max_opacity = np.max(opacity_np)
            opacity_np = depth2rgb(
                opacity_np, min_value=0.0, max_value=max_opacity, colormap="jet"
            )
            opacity_np = torch.from_numpy(opacity_np)
            opacity_np = torch.permute(opacity_np, (2, 0, 1)).float()
            opacity_np = (opacity_np).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        
        best_pose_c2w = None
        timing_cluster = start_timing()
        if sum_invisibility > 100:
            factor_width = factor_height = 0.5 # use downsampled image to accelerate clustering
            target_width = int(factor_width * invisibility_np.shape[1])
            target_height = int(factor_height * invisibility_np.shape[0])
            invisibility_np = cv2.resize(invisibility_np, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cluster_centers, cluster_invisibilities = get_invisibility_clusters(invisibility_np, self.__cluster_invisibility_threshold)
            
            # NOTE: Find the cluster with the maximum area
            if len(cluster_invisibilities) > 0:
                max_area_index = np.argmax(cluster_invisibilities)  
                max_area_center = cluster_centers[max_area_index]  
                # Vector from max_area_center to camera center
                center_vec = np.array([max_area_center[1] / factor_width - width / 2, max_area_center[0] / factor_height - height / 2])
                horizontal_angle = np.deg2rad(center_vec[0])
                vertical_angle = np.deg2rad(center_vec[1])
                # Skip the points in the middle of the field of view because I have already seen them
                if np.abs(horizontal_angle) > np.deg2rad(15) or np.abs(vertical_angle) > np.deg2rad(15):
                    cv2.circle(opacity_np, (int(max_area_center[1] / factor_width), int(max_area_center[0] / factor_height)), 5, (0, 0, 255), -1)
                    cv2.rectangle(opacity_np, (int(max_area_center[1] / factor_width) - round(np.rad2deg(self.__rgbd_sensor.hfov))//2, int(max_area_center[0] / factor_height) - round(np.rad2deg(self.__rgbd_sensor.vfov))//2), 
                                  (int(max_area_center[1] / factor_width) + round(np.rad2deg(self.__rgbd_sensor.hfov))//2, int(max_area_center[0] / factor_height) + round(np.rad2deg(self.__rgbd_sensor.vfov))//2), (0, 0, 255), 2)
                    best_pose_c2w = rot_axis(view_c2w, 'y', horizontal_angle)
                    best_pose_c2w = rot_axis(best_pose_c2w, 'x', vertical_angle)
                
        Log(f'Get Cluster used {end_timing(*timing_cluster):.2f} ms', tag='ActiveSplat')
        
        rgb_opacity_np = None
        if show_image:
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            rgb_opacity_np = np.vstack((rgb_np, opacity_np))
        
        return rgb_opacity_np, sum_invisibility, best_pose_c2w