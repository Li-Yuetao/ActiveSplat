import numpy as np
import quaternion

def mat_to_q_pos(pose_np):
    if pose_np.shape != (4, 4):
        raise ValueError("The pose matrix should be a 4x4 matrix.")

    position = pose_np[:3, 3]
    q = quaternion.from_rotation_matrix(pose_np[:3, :3])
    
    return q, position

def rot_axis(view_c2w, axis, angle_rad):
    """
    Rotates the camera pose along its own coordinate axis.
    """
    
    if axis == 'x':
        R = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    new_view_c2w = view_c2w @ R
    return new_view_c2w