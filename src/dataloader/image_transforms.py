import cv2
import numpy as np

class BGRtoRGB(object):
    """bgr format to rgb"""

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale


class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        return depth

class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, min_depth, max_depth):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        near_mask = depth < self.min_depth
        depth[far_mask] = 0.
        depth[near_mask] = 0.
        return depth
