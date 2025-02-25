import cv2
import numpy as np
import scipy
from enum import Enum
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

def get_convexhull_volume(depth_np: np.ndarray, invisibility_np: np.ndarray, hfov=120, vfov=150):
    if depth_np.ndim == 3:
        depth_np = np.squeeze(depth_np, axis=-1)
    # Considering low visibility points
    low_visibility_points = np.column_stack(np.where(invisibility_np > 0.8))
    
    if len(low_visibility_points) == 0:
        return 0, 0

    # DBSCAN clustering
    dbscan = DBSCAN(eps=5, min_samples=25)
    clusters = dbscan.fit_predict(low_visibility_points)
    cluster_points_3d = []
    points_3d = np.empty((0, 3), dtype=np.float32)

    # Generate a single mask image for each cluster block
    unique_clusters = set(clusters)
    masks = []
    cluster_invisibility_sums = np.array([])
    cluster_volume_sums = np.array([])

    for cluster in unique_clusters:
        if cluster == -1:
            continue
        mask = np.zeros_like(invisibility_np, dtype=np.uint8)
        cluster_points = low_visibility_points[clusters == cluster]
        mask[cluster_points[:, 0], cluster_points[:, 1]] = 255
        cluster_invisibility = invisibility_np[cluster_points[:, 0], cluster_points[:, 1]]
        
        # NOTE: Mask dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # NOTE: Find contours
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        
        if len(contours) != 1:
            print(f'Warning: {len(contours)} contours found for cluster {cluster}')
        
        cluster_points_3d_current = []
        cluster_points = []
        for point in max_contour:
            x, y = point[0]
            if 0 <= y < depth_np.shape[0] and 0 <= x < depth_np.shape[1]:
                z = depth_np[y, x]
                if z == 15:
                    continue
                point_3d = [x, y, z]
                cluster_points.append(point_3d)
                points_3d = np.vstack((points_3d, point_3d))
        cluster_points_np = np.array(cluster_points)
        
        if cluster_points_np.shape[0] == 0:
            continue
        
        h_angle_per_pixel = np.deg2rad(360 / depth_np.shape[1]) # use radian
        v_angle_per_pixel = np.deg2rad(vfov / depth_np.shape[0])
        
        cluster_points_np[:, 0] = cluster_points_np[:, 0] * h_angle_per_pixel
        cluster_points_np[:, 1] = cluster_points_np[:, 1] * v_angle_per_pixel
        
        if len(cluster_points_np) >= 4:  # ConvexHull requires at least 4 points
            if np.linalg.matrix_rank(cluster_points_np) < 3:
                cluster_points_np += np.random.normal(scale=1e-10, size=cluster_points_np.shape)
            try:
                hull = ConvexHull(cluster_points_np)
                cluster_volume = hull.volume
            except scipy.spatial.qhull.QhullError:
                cluster_volume = 0
        else:
            cluster_volume = 0
        
        cluster_points_3d_current.append(cluster_points)
        cluster_invisibility_sum = np.sum(cluster_invisibility * cluster_volume)
        cluster_invisibility_sums = np.append(cluster_invisibility_sums, cluster_invisibility_sum)
        cluster_volume_sums = np.append(cluster_volume_sums, cluster_volume)
        cluster_points_3d.append(np.array(cluster_points_3d_current))
        masks.append(dilated_mask)
    
    last_invisibility = np.sum(cluster_invisibility_sums)
    last_volume = np.sum(cluster_volume_sums)
    return last_invisibility, last_volume

def get_invisibility_clusters(invisibility_np: np.ndarray, cluster_invisibility_threshold=30):
    low_visibility_points = np.column_stack(np.where(invisibility_np > 0.3))
    
    if len(low_visibility_points) == 0:
        return [], []

    # NOTE: DBSCAN clustering
    dbscan = DBSCAN(eps=5, min_samples=10)
    clusters = dbscan.fit_predict(low_visibility_points)

    cluster_centers = []
    cluster_invisibilities = []
    valid_clusters = []

    for cluster in set(clusters):
        if cluster != -1:
            points = low_visibility_points[clusters == cluster]
            center = points.mean(axis=0)
            # cluster invisibility
            invisibility_sum = np.sum(invisibility_np[points[:, 0], points[:, 1]])
            if invisibility_sum > cluster_invisibility_threshold:
                cluster_centers.append(center)
                cluster_invisibilities.append(invisibility_sum)
                valid_clusters.append(cluster)

    return cluster_centers, cluster_invisibilities

class MapperState(Enum):
    BOOTSTRAP = 0
    INITIALIZING = 1
    MAPPING = 2
    IDLE = 3
    
class GaussianColorType(Enum):
    Color = 'Color'
    Depth = 'Depth'
    Opacity = 'Opacity'
    RGBD = 'RGBD'

class MapperType(Enum):
    SplaTAM = 'SplaTAM'
    
def get_mapper(model_type:MapperType):
    if model_type == MapperType.SplaTAM:
        from mapper.splatam import SplaTAM
        return SplaTAM
    else:
        raise ValueError(f"Model type {model_type} not supported")