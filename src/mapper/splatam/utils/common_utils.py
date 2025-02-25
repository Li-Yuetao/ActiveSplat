import os

import numpy as np
import random
import torch
import cv2
from imgviz import depth2rgb


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)
    
def save_keyframes(keyframe_list, keyframe_dir, rgbd_sensor):
    os.makedirs(keyframe_dir, exist_ok=True)
    for keyframe in keyframe_list:
        # save color and depth
        color_vis:np.ndarray = torch.permute(keyframe['color'], (1, 2, 0)).detach().cpu().numpy()
        color_vis = np.clip(color_vis, 0, 1)
        color_vis = np.uint8(color_vis * 255)
        depth_vis:np.ndarray = keyframe['depth'].float().detach().cpu().numpy()
        depth_vis = depth_vis.reshape(rgbd_sensor.height, rgbd_sensor.width)
        depth_vis = depth2rgb(depth_vis, min_value=rgbd_sensor.depth_min, max_value=rgbd_sensor.depth_max)
        cur_frame_id = keyframe['id']
        
        rgbd = np.hstack((color_vis, depth_vis)) # (H, W*2, 3)
        cv2.imwrite(os.path.join(keyframe_dir, f"{str(cur_frame_id).zfill(4)}.png"), cv2.cvtColor(rgbd, cv2.COLOR_RGB2BGR))
        
def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)