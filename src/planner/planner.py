from enum import Enum
from typing import List, Dict, Union, Tuple
import os
import time
import cv2
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
import networkx as nx
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

import rospy

from utils import start_timing, end_timing

class Frustum(object):
    
    def __init__(self):
        self.c2w = None
        self.translation_2d = None
        self.rotation_vector_2d = None
        self.pitch_angle = None
        self.invisibility_score = 0.0 
        self.hole_valume = 0.0
        self.is_looked = False

def is_line_segment_out_of_circle(
    line_segment_start:np.ndarray,
    line_segment_end:np.ndarray,
    circle_center:np.ndarray,
    circle_radius:float) -> np.ndarray:
    line_segment_start_to_circle_center = circle_center - line_segment_start
    line_segment_end_to_circle_center = circle_center - line_segment_end
    line_segment_start_to_end = line_segment_end - line_segment_start
    line_segment_end_to_start = -line_segment_start_to_end
    line_segment_start_dot:np.ndarray = np.einsum('ij,ij->i', line_segment_start_to_circle_center, line_segment_start_to_end)
    line_segment_end_dot:np.ndarray = np.einsum('ij,ij->i', line_segment_end_to_circle_center, line_segment_end_to_start)
    vertical_foot_on_line_segment = np.logical_and(
        line_segment_start_dot >= 0,
        line_segment_end_dot >= 0)
    line_segment_length = np.linalg.norm(line_segment_start_to_end, axis=1)
    assert np.all(line_segment_length > 0), 'Line segment length should be greater than zero.'
    line_segment_vertical_foot_distance:np.ndarray = np.abs(np.cross(line_segment_start_to_end, line_segment_start_to_circle_center)) / line_segment_length
    line_segment_vertical_foot_in_circle_condition = np.logical_and(
        vertical_foot_on_line_segment,
        line_segment_vertical_foot_distance <= circle_radius)
    line_segment_start_in_circle_condition = np.linalg.norm(line_segment_start_to_circle_center, axis=1) <= circle_radius
    line_segment_end_in_circle_condition = np.linalg.norm(line_segment_end_to_circle_center, axis=1) <= circle_radius
    line_segment_out_of_circle_condition = np.logical_and(
        np.logical_not(line_segment_vertical_foot_in_circle_condition),
        np.logical_and(
            np.logical_not(line_segment_start_in_circle_condition),
            np.logical_not(line_segment_end_in_circle_condition)))
    return line_segment_out_of_circle_condition

def splat_inaccessible_database(
    agent_position:np.ndarray,
    global_obstacle_map:np.ndarray,
    inaccessible_database:Dict[Tuple[float, float], np.ndarray],
    splat_size_pixel:float) -> np.ndarray:
    global_obstacle_map_ = global_obstacle_map.copy()
    global_obstacle_map_vis = cv2.cvtColor(global_obstacle_map_, cv2.COLOR_GRAY2BGR)
    splat_radius = max(np.int32(np.round(splat_size_pixel / 2)), 1)
    splat_flag = False
    for translation, rotation_vectors in inaccessible_database.items():
        translation_np = np.array(translation)
        splat_centers:np.ndarray = translation_np + rotation_vectors / np.linalg.norm(rotation_vectors, axis=1)[:, np.newaxis] * splat_size_pixel
        splat_centers = np.int32(np.round(splat_centers))
        splat_centers_condition = np.logical_and(
            np.logical_and(
                0 <= splat_centers[:, 0],
                splat_centers[:, 0] < global_obstacle_map_.shape[1]),
            np.logical_and(
                0 <= splat_centers[:, 1],
                splat_centers[:, 1] < global_obstacle_map_.shape[0]))
        splat_centers = splat_centers[splat_centers_condition]
        for splat_center in splat_centers.tolist():
            global_obstacle_map_ = cv2.circle(
                global_obstacle_map_,
                splat_center,
                splat_radius,
                0,
                -1)
            global_obstacle_map_vis = cv2.circle(
                global_obstacle_map_vis,
                splat_center,
                splat_radius,
                (0, 0, 255),
                -1)
            splat_flag = True
    if splat_flag:
        test_splat_dir = os.path.join(os.getcwd(), 'test', 'test_splat')
        os.makedirs(test_splat_dir, exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        global_obstacle_map_vis = cv2.circle(
            global_obstacle_map_vis,
            np.int32(agent_position),
            np.int32(np.ceil(splat_size_pixel / 2)),
            (0, 255, 0),
            -1)
        cv2.imwrite(os.path.join(test_splat_dir, current_time + '_raw.png'), global_obstacle_map)
        cv2.imwrite(os.path.join(test_splat_dir, current_time + '_splat.png'), global_obstacle_map_vis)
    return global_obstacle_map_

def update_topdown_free_map(global_obstacle_map:np.ndarray, topdown_visible_map, kernel) -> None:
    origin_topdown_visible_map = topdown_visible_map.copy()
    topdown_visible_map = cv2.bitwise_not(topdown_visible_map)
    topdown_visible_map_contours, _ = cv2.findContours(topdown_visible_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(topdown_visible_map_contours, key=cv2.contourArea)
    
    topdown_visible_map = np.zeros_like(topdown_visible_map)
    # get max_contour's area
    max_contour_area = cv2.contourArea(max_contour)
    cv2.drawContours(topdown_visible_map, [max_contour], -1, 255, -1)
    
    topdown_tmp = cv2.bitwise_not(global_obstacle_map)
    topdown_tmp = cv2.bitwise_and(topdown_visible_map, topdown_tmp)
    topdown_tmp = cv2.bitwise_not(topdown_tmp)
    missing_visible_map = cv2.bitwise_and(topdown_visible_map, origin_topdown_visible_map)
    
    topdown_visible_map[topdown_tmp == 0] = 0
    topdown_visible_map[missing_visible_map == 255] = 0
    topdown_tmp = cv2.morphologyEx(topdown_visible_map, cv2.MORPH_OPEN, kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topdown_tmp = cv2.dilate(topdown_tmp, dilate_kernel)
    return topdown_tmp

def get_obstacle_map(
    global_obstacle_map:np.ndarray,
    topdown_visible_map:np.ndarray,
    agent_position:np.ndarray,
    kernel,
    approx_precision:float) -> Tuple[cv2.Mat, np.ndarray, List[np.ndarray]]:
    get_obstacle_map_timing = start_timing()
    global_obstacle_map_in_scene = update_topdown_free_map(global_obstacle_map, topdown_visible_map, kernel)
    global_obstacle_map_contours, _ = cv2.findContours(
        global_obstacle_map_in_scene,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    agent_to_contours = np.array([cv2.pointPolygonTest(contour, agent_position, False) for contour in global_obstacle_map_contours])
    agent_to_contours_condition = agent_to_contours >= 0
    if np.any(agent_to_contours_condition):
        contours_index_selected = np.where(agent_to_contours_condition)[0]
        agent_to_contours_selected = agent_to_contours[contours_index_selected]
        contour_index_selected = contours_index_selected[np.argmin(agent_to_contours_selected)]
        local_obstacle_map_contour = global_obstacle_map_contours[contour_index_selected]
    else:
        rospy.logwarn('Robot position is not in the obstacle map.')
        global_obstacle_map_contours, _ = cv2.findContours(
            global_obstacle_map,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        agent_to_contours = np.array([cv2.pointPolygonTest(contour, agent_position, False) for contour in global_obstacle_map_contours])
        agent_to_contours_condition = agent_to_contours >= 0
        contours_index_selected = np.where(agent_to_contours_condition)[0]
        agent_to_contours_selected = agent_to_contours[contours_index_selected]
        contour_index_selected = contours_index_selected[np.argmin(agent_to_contours_selected)]
        local_obstacle_map_contour = global_obstacle_map_contours[contour_index_selected]
    if approx_precision is None:
        local_obstacle_map_approx_contour = local_obstacle_map_contour
    else:
        local_obstacle_map_approx_contour = cv2.approxPolyDP(
            local_obstacle_map_contour,
            approx_precision,
            True)
    white = np.ones_like(global_obstacle_map) * 255
    black = np.zeros_like(global_obstacle_map)
    
    local_obstacle_map_approx_inverse = cv2.drawContours(white.copy(), [local_obstacle_map_approx_contour], -1, 0, -1)
    local_obstacle_map_inverse = cv2.drawContours(white.copy(), [local_obstacle_map_contour], -1, 0, -1)
    local_obstacle_map_approx:cv2.Mat = cv2.drawContours(black.copy(), [local_obstacle_map_approx_contour], -1, 255, -1)
    obstacle_map_of_children = cv2.bitwise_or(
        cv2.bitwise_or(
            local_obstacle_map_inverse,
            local_obstacle_map_approx_inverse),
        global_obstacle_map)
    obstacle_map_of_children_inverse = cv2.bitwise_not(obstacle_map_of_children)
    obstacle_map_children_contours, _ = cv2.findContours(
        obstacle_map_of_children_inverse,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    obstacle_map_children_approx_contours = []
    for contour in obstacle_map_children_contours:
        if cv2.contourArea(contour) > 0:
            if approx_precision is None:
                approx_contour = contour
            else:
                approx_contour = cv2.approxPolyDP(contour, approx_precision, True)
            if cv2.contourArea(approx_contour) > 0:
                obstacle_map_children_approx_contours.append(approx_contour)
    obstacle_map = cv2.drawContours(local_obstacle_map_approx.copy(), obstacle_map_children_approx_contours, -1, 0, -1)
    rospy.logdebug(f'Get obstacle map timing: {end_timing(*get_obstacle_map_timing)} ms')
    return obstacle_map, local_obstacle_map_approx_contour, obstacle_map_children_approx_contours

def get_voronoi_graph(
    obstacle_map:cv2.Mat,
    local_obstacle_map_approx_contour:np.ndarray,
    obstacle_map_children_approx_contours:List[np.ndarray],
    edge_sample_num:int,
    agent_radius_pixel:float,
    inaccessible_points:np.ndarray) -> Dict[str, Union[nx.Graph, np.ndarray, cv2.Mat, List[List[np.ndarray]]]]:
    '''
    Given the obstacle map and the robot position, return the voronoi graph.
    '''
    edge_length_min = np.inf
    obstacle_map_approx_contours_vertices = []
    obstacle_map_approx_contours_edges_length = []
    for contour in ([local_obstacle_map_approx_contour] + obstacle_map_children_approx_contours):
        contour_vertices = contour.reshape(-1, 2)
        contour_edges_length = np.linalg.norm(contour_vertices - np.roll(contour_vertices, 1, axis=0), axis=1)
        obstacle_map_approx_contours_vertices.append(contour_vertices)
        obstacle_map_approx_contours_edges_length.append(contour_edges_length)
        contour_edges_length_min = np.min(contour_edges_length)
        if contour_edges_length_min > 0:
            edge_length_min = min(edge_length_min, contour_edges_length_min)
    assert edge_length_min != np.inf, 'Edge length min should be less than infinity.'
    edge_sample_resolution = edge_length_min / edge_sample_num
    
    obstacle_points = np.array([]).reshape(0, 2)
    for contour_vertices, contour_edges_length in zip(
        obstacle_map_approx_contours_vertices,
        obstacle_map_approx_contours_edges_length):
        for vertex_start, vertex_end, edge_length in zip(
            contour_vertices,
            np.roll(contour_vertices, 1, axis=0),
            contour_edges_length):
            edge_sample_num = int(edge_length / edge_sample_resolution)
            edge_sample = np.linspace(vertex_start, vertex_end, edge_sample_num, endpoint=False)
            obstacle_points = np.vstack((obstacle_points, edge_sample))
            
    # Adding a small perturbation to avoid coplanar issue
    perturbation = np.random.normal(scale=1e-10, size=obstacle_points.shape)
    obstacle_points += perturbation
            
    voronoi_graph = Voronoi(obstacle_points)
    
    voronoi_graph_ridge_vertices = np.array(voronoi_graph.ridge_vertices)
    voronoi_graph_ridge_vertices = voronoi_graph_ridge_vertices[np.all(voronoi_graph_ridge_vertices >= 0, axis=1)]
    voronoi_graph_ridge_matrix = np.zeros((len(voronoi_graph.vertices), len(voronoi_graph.vertices)))
    voronoi_graph_vertices:np.ndarray = voronoi_graph.vertices
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 0], voronoi_graph_ridge_vertices[:, 1]] = 1
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 1], voronoi_graph_ridge_vertices[:, 0]] = 1
        
    vertices_indexes = []
    for vertex_index, vertex in enumerate(voronoi_graph_vertices):
        if cv2.pointPolygonTest(local_obstacle_map_approx_contour, vertex, True) > agent_radius_pixel:
            in_children_obstacle_flag = False
            for obstacle_map_children_approx_contour in obstacle_map_children_approx_contours:
                if cv2.pointPolygonTest(obstacle_map_children_approx_contour, vertex, True) > -agent_radius_pixel:
                    in_children_obstacle_flag = True
                    break
            if not in_children_obstacle_flag:
                vertices_indexes.append(vertex_index)
    voronoi_graph_vertices = voronoi_graph_vertices[vertices_indexes]
    voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[vertices_indexes][:, vertices_indexes]
    
    voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
    voronoi_graph_vertices_condition = voronoi_graph_vertices_connectivity > 0
    voronoi_graph_vertices = voronoi_graph_vertices[voronoi_graph_vertices_condition]
    voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[voronoi_graph_vertices_condition][:, voronoi_graph_vertices_condition]
    voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
    
    voronoi_graph_vertices_fix_condition = voronoi_graph_vertices_connectivity >= 3
    
    if len(inaccessible_points) > 0:
        inaccessible_points_distance_to_voronoi_graph_vertices = cdist(inaccessible_points, voronoi_graph_vertices)
        close_vertices_sorted_index = np.argsort(inaccessible_points_distance_to_voronoi_graph_vertices, axis=1)
        close_vertices_start_index = close_vertices_sorted_index[:, 0]
        close_vertices_end_index = close_vertices_sorted_index[:, 1]
        close_vertices_connectivity = np.bool8(voronoi_graph_ridge_matrix[close_vertices_start_index, close_vertices_end_index])
        inaccessible_points_selected = inaccessible_points[close_vertices_connectivity]
        close_vertices_start_index_selected = close_vertices_start_index[close_vertices_connectivity]
        close_vertices_end_index_selected = close_vertices_end_index[close_vertices_connectivity]
        close_vertices_start_selected = voronoi_graph_vertices[close_vertices_start_index_selected]
        close_vertices_end_selected = voronoi_graph_vertices[close_vertices_end_index_selected]
        line_segement_selected_out_of_circle_condition = is_line_segment_out_of_circle(
            close_vertices_start_selected,
            close_vertices_end_selected,
            inaccessible_points_selected,
            agent_radius_pixel)
        line_segement_selected_not_out_of_circle_condition = np.logical_not(line_segement_selected_out_of_circle_condition)
        pruned_vertex_start_index = close_vertices_start_index_selected[line_segement_selected_not_out_of_circle_condition]
        pruned_vertex_end_index = close_vertices_end_index_selected[line_segement_selected_not_out_of_circle_condition]
        pruned_vertex_index = np.unique(np.hstack((pruned_vertex_start_index, pruned_vertex_end_index)))
                
        voronoi_graph_vertices_inaccessible_condition = np.zeros(len(voronoi_graph_vertices), dtype=bool)
        voronoi_graph_vertices_inaccessible_condition[pruned_vertex_index] = True
        voronoi_graph_vertices_inaccessible_condition = np.logical_and(
            voronoi_graph_vertices_inaccessible_condition,
            np.logical_not(voronoi_graph_vertices_fix_condition))
        if np.any(voronoi_graph_vertices_inaccessible_condition):
            rospy.logwarn('Inaccessible points are detected in the voronoi graph.')
        voronoi_graph_vertices_accessibile_condition = np.logical_not(voronoi_graph_vertices_inaccessible_condition)
        voronoi_graph_vertices = voronoi_graph_vertices[voronoi_graph_vertices_accessibile_condition]
        voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[voronoi_graph_vertices_accessibile_condition][:, voronoi_graph_vertices_accessibile_condition]
        voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
        voronoi_graph_vertices_fix_condition = voronoi_graph_vertices_fix_condition[voronoi_graph_vertices_accessibile_condition]
        voronoi_graph_vertices_nodes_index = np.where(voronoi_graph_vertices_fix_condition)[0]
    
    pruned_chains:List[List[np.ndarray]] = []
    while True:
        voronoi_graph_vertices_nodes_index = np.where(voronoi_graph_vertices_fix_condition)[0]
        voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
        pruned_vertices_index = np.where(voronoi_graph_vertices_connectivity <= 1)[0]
        pruned_vertices_index = np.setdiff1d(pruned_vertices_index, voronoi_graph_vertices_nodes_index)
        remain_vertices_index = np.setdiff1d(np.arange(len(voronoi_graph_vertices)), pruned_vertices_index)
        if len(pruned_vertices_index) == 0:
            break
        else:
            if len(pruned_chains) == 0:
                for pruned_vertex_index in pruned_vertices_index:
                    if np.sum(voronoi_graph_ridge_matrix[pruned_vertex_index]) == 0:
                        continue
                    else:
                        pruned_vertex_index_next = np.where(voronoi_graph_ridge_matrix[pruned_vertex_index])[0][0]
                        pruned_chain = [
                            voronoi_graph_vertices[pruned_vertex_index],
                            voronoi_graph_vertices[pruned_vertex_index_next]]
                        pruned_chains.append(pruned_chain)
            else:
                isolated_chains_index = []
                for pruned_vertex_index in pruned_vertices_index:
                    if np.sum(voronoi_graph_ridge_matrix[pruned_vertex_index]) == 0:
                        for pruned_chain_index, pruned_chain in enumerate(pruned_chains):
                            if np.allclose(pruned_chain[-1], voronoi_graph_vertices[pruned_vertex_index]):
                                isolated_chains_index.append(pruned_chain_index)
                                break
                    else:
                        pruned_vertex_index_next = np.where(voronoi_graph_ridge_matrix[pruned_vertex_index])[0][0]
                        for pruned_chain_index, pruned_chain in enumerate(pruned_chains):
                            if np.allclose(pruned_chain[-1], voronoi_graph_vertices[pruned_vertex_index]):
                                pruned_chains[pruned_chain_index].append(voronoi_graph_vertices[pruned_vertex_index_next])
                                break
                
                isolated_chains_index = np.unique(isolated_chains_index)
                pruned_chains = [pruned_chain for pruned_chain_index, pruned_chain in enumerate(pruned_chains) if pruned_chain_index not in isolated_chains_index]
                
            voronoi_graph_vertices = voronoi_graph_vertices[remain_vertices_index]
            voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[remain_vertices_index][:, remain_vertices_index]
            voronoi_graph_vertices_fix_condition = voronoi_graph_vertices_fix_condition[remain_vertices_index]
    
    assert np.sum(voronoi_graph_ridge_matrix[np.diag_indices(len(voronoi_graph_vertices))]) == 0, 'Diagonal elements of the voronoi graph ridge matrix should be zero.'
    voronoi_graph_ridge_vertices = np.argwhere(np.triu(voronoi_graph_ridge_matrix))
    voronoi_graph_ridge_edges_length = np.linalg.norm(
        voronoi_graph_vertices[voronoi_graph_ridge_vertices[:, 0]] -\
            voronoi_graph_vertices[voronoi_graph_ridge_vertices[:, 1]],
            axis=1)
    voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 0], voronoi_graph_ridge_vertices[:, 1]] =\
        voronoi_graph_ridge_edges_length
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 1], voronoi_graph_ridge_vertices[:, 0]] =\
        voronoi_graph_ridge_edges_length
        
    # find the > 2 connectivity nodes
    high_connectivity_nodes_index = np.where(voronoi_graph_vertices_connectivity > 2)[0]
    high_connectivity_nodes_index = np.intersect1d(high_connectivity_nodes_index, voronoi_graph_vertices_nodes_index)
    
    return {
        'graph': nx.from_numpy_array(voronoi_graph_ridge_matrix),
        'vertices': voronoi_graph_vertices,
        'obstacle_map': obstacle_map,
        'pruned_chains': pruned_chains,
        'nodes_index': voronoi_graph_vertices_nodes_index,
        'high_connectivity_nodes_index': high_connectivity_nodes_index}
    
def draw_voronoi_graph(
    background:cv2.Mat,
    voronoi_graph_vertices:np.ndarray,
    voronoi_graph_ridge_matrix:np.ndarray,
    voronoi_graph_nodes_index:np.ndarray,
    voronoi_graph_nodes_score:np.ndarray,
    voronoi_graph_nodes_score_max:int,
    voronoi_graph_nodes_score_min:int,
    global_invisibility_info:Dict[int, Frustum],
    pruned_chains:List[List[np.ndarray]],
    voronoi_graph_ridge_color:List[int],
    voronoi_graph_ridge_thickness:int,
    voronoi_graph_nodes_colormap:colors.Colormap,
    voronoi_graph_nodes_radius:int,
    pruned_chains_color:List[int],
    pruned_chains_thickness:int) -> cv2.Mat:
    
    voronoi_graph_image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for pruned_chain in pruned_chains:
        cv2.polylines(voronoi_graph_image, [np.array(pruned_chain).astype(np.int32)], False, pruned_chains_color, pruned_chains_thickness)
        
    voronoi_graph_ridge_vertices = np.argwhere(np.triu(voronoi_graph_ridge_matrix))
    for voronoi_graph_ridge_vertex in voronoi_graph_ridge_vertices:
        vertex_start:np.ndarray = voronoi_graph_vertices[voronoi_graph_ridge_vertex[0]]
        vertex_end:np.ndarray = voronoi_graph_vertices[voronoi_graph_ridge_vertex[1]]
        cv2.line(
            voronoi_graph_image,
            vertex_start.astype(np.int32),
            vertex_end.astype(np.int32),
            voronoi_graph_ridge_color,
            voronoi_graph_ridge_thickness)
        
    for voronoi_graph_node_index, voronoi_graph_node_score in zip(voronoi_graph_nodes_index, voronoi_graph_nodes_score):
        voronoi_graph_node = voronoi_graph_vertices[voronoi_graph_node_index]
        if voronoi_graph_node_index in global_invisibility_info:
            if global_invisibility_info[voronoi_graph_node_index].translation_2d is not None:
                cv2.arrowedLine(
                    voronoi_graph_image,
                    global_invisibility_info[voronoi_graph_node_index].translation_2d.astype(np.int32).tolist(),
                    (global_invisibility_info[voronoi_graph_node_index].translation_2d + global_invisibility_info[voronoi_graph_node_index].rotation_vector_2d * 10).astype(np.int32).tolist(),
                    (0, 255, 0),
                    2)
        cv2.circle(
            voronoi_graph_image,
            np.int32(voronoi_graph_node),
            voronoi_graph_nodes_radius,
            np.uint8(
                np.array(voronoi_graph_nodes_colormap(
                    (voronoi_graph_node_score - voronoi_graph_nodes_score_min) / (voronoi_graph_nodes_score_max - voronoi_graph_nodes_score_min))[:3]) * 255).tolist()[::-1],
            -1)
        
    return voronoi_graph_image

def get_closest_vertex_index(
    voronoi_graph_vertices:np.ndarray,
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_radius_pixel:float) -> int:
    voronoi_graph_vertices_to_agent_distance = np.linalg.norm(voronoi_graph_vertices - agent_position, axis=1)
    voronoi_graph_vertices_index_sorted = np.argsort(voronoi_graph_vertices_to_agent_distance)
    voronoi_graph_vertices_sorted = voronoi_graph_vertices[voronoi_graph_vertices_index_sorted]
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    agent_mask = cv2.circle(
        np.zeros_like(obstacle_map),
        np.int32(agent_position),
        int(np.ceil(agent_radius_pixel)),
        255,
        -1)
    for vertex_index, vertex in enumerate(voronoi_graph_vertices_sorted):
        line_test_result = cv2.line(
            obstacle_map.copy(),
            np.int32(agent_position),
            np.int32(vertex),
            255,
            int(np.ceil(agent_radius_pixel * 3)))
        line_test_result[agent_mask > 0] = obstacle_map[agent_mask > 0]
        if cv2.countNonZero(line_test_result) == free_space_pixels_num:
            return voronoi_graph_vertices_index_sorted[vertex_index]
    line_test_results = []
    for vertex_index, vertex in enumerate(voronoi_graph_vertices_sorted):
        line_test_result = cv2.line(
            obstacle_map.copy(),
            np.int32(agent_position),
            np.int32(vertex),
            255,
            1)
        line_test_results.append(cv2.countNonZero(line_test_result))
        if line_test_results[-1] == free_space_pixels_num:
            return voronoi_graph_vertices_index_sorted[vertex_index]
    vertex_index = np.argmin(line_test_results)
    return voronoi_graph_vertices_index_sorted[vertex_index]

def get_closest_node_index(
    voronoi_graph_vertices: np.ndarray,
    voronoi_graph_vertices_nodes_index: np.ndarray,
    agent_position: np.ndarray
) -> int:
    distances = np.linalg.norm(voronoi_graph_vertices[voronoi_graph_vertices_nodes_index] - agent_position, axis=1)
    closest_node_index = voronoi_graph_vertices_nodes_index[np.argmin(distances)]
    return closest_node_index

def optimize_navigation_path_using_fast_forward(
    navigation_path:np.ndarray,
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_radius_pixel:float) -> np.ndarray:
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    
    navigation_point_last_distance = np.inf
    for navigation_point_index, navigation_point in enumerate(navigation_path[::-1]):
        line_test_result = cv2.line(
            obstacle_map.copy(),
            np.int32(agent_position),
            np.int32(navigation_point),
            255,
            int(np.ceil(agent_radius_pixel * 3)))
        if cv2.countNonZero(line_test_result) != free_space_pixels_num:
            continue
        navigation_point_distance = np.linalg.norm(agent_position - navigation_point)
        if navigation_point_distance > navigation_point_last_distance:
            break
        navigation_point_last_distance = navigation_point_distance
        
    return navigation_path[-(navigation_point_index + 1):]

def get_safe_dijkstra_path(
    voronoi_graph:nx.Graph,
    vertex_start_index:int,
    vertex_end_index:int,
    voronoi_graph_vertices:np.ndarray,
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_radius_pixel:float,
    fast_forward_radius_ratio:float=1.0) -> Tuple[np.ndarray, np.ndarray, bool]:
    try:
        navigation_path_index = nx.dijkstra_path(voronoi_graph, vertex_start_index, vertex_end_index)
    except nx.NetworkXNoPath:
        return None, None, False
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    navigation_path = voronoi_graph_vertices[navigation_path_index]
        
    navigation_path = optimize_navigation_path_using_fast_forward(
        navigation_path=navigation_path,
        obstacle_map=obstacle_map,
        agent_position=agent_position,
        agent_radius_pixel=agent_radius_pixel * fast_forward_radius_ratio)
    
    line_test_result = cv2.polylines(
        obstacle_map.copy(),
        [np.int32(navigation_path)],
        False,
        255,
        int(np.ceil(agent_radius_pixel * 2)))
    if cv2.countNonZero(line_test_result) == free_space_pixels_num:
        return navigation_path_index, navigation_path, True
    else:
        return None, None, True
    
def get_subregions(
    voronoi_graph: nx.Graph,
    nodes_index: np.ndarray,
    voronoi_graph_vertices: np.ndarray,
    meter_per_pixel: float,
    path_weight=0.5,
    coord_weight=0.5,
    image_shape=(500, 400),
    save_runtime_data=False
):
    n = len(nodes_index)
    path_distance_matrix = np.full((n, n), np.inf)
    lengths = dict(nx.all_pairs_dijkstra_path_length(voronoi_graph))

    for i, node_i in enumerate(nodes_index):
        for j, node_j in enumerate(nodes_index):
            if node_i in lengths and node_j in lengths[node_i]:
                path_distance_matrix[i][j] = lengths[node_i][node_j]
    
    coord_distance_matrix = sp.distance.cdist(voronoi_graph_vertices[nodes_index], voronoi_graph_vertices[nodes_index], 'euclidean')
    combined_distance_matrix = path_weight * path_distance_matrix + coord_weight * coord_distance_matrix
    combined_distance_matrix = (combined_distance_matrix + combined_distance_matrix.T) / 2

    if np.isinf(combined_distance_matrix).any():
        max_finite = np.max(combined_distance_matrix[np.isfinite(combined_distance_matrix)])
        combined_distance_matrix[np.isinf(combined_distance_matrix)] = max_finite + 1

    # NOTE: Heirarchical Clustering
    Z = hc.linkage(sp.distance.squareform(combined_distance_matrix), method='average')
    max_distance = 0
    for i in range(len(voronoi_graph_vertices)):
        for j in range(i + 1, len(voronoi_graph_vertices)):
            distance = np.linalg.norm(voronoi_graph_vertices[i] - voronoi_graph_vertices[j])
            if distance > max_distance:
                max_distance = distance
    max_distance_in_meters = max_distance * meter_per_pixel
    distance_threshold = 2 / meter_per_pixel # 2m scope
    clusters = hc.fcluster(Z, t=distance_threshold, criterion='distance')

    clusters_cv2 = None
    if save_runtime_data:
        clusters_cv2 = plot_voronoi_subregions(voronoi_graph_vertices, nodes_index, clusters, voronoi_graph, image_shape)

    voronoi_subregions = {node: cluster for node, cluster in zip(nodes_index, clusters)}
    return voronoi_subregions, clusters_cv2

def plot_voronoi_subregions(voronoi_graph_vertices, nodes_index, clusters, voronoi_graph, image_shape=(400, 500)):
    unique_clusters = np.unique(clusters)
    colors = plt.cm.get_cmap('tab20', len(unique_clusters))

    dpi = 80
    scale = 2
    fig, ax = plt.subplots(figsize=(scale * image_shape[1] / dpi, scale * image_shape[0] / dpi), dpi=dpi)

    for edge in voronoi_graph.edges(data=True):
        p1, p2 = edge[0], edge[1]
        ax.plot([voronoi_graph_vertices[p1][0], voronoi_graph_vertices[p2][0]],
                 [-voronoi_graph_vertices[p1][1], -voronoi_graph_vertices[p2][1]],
                 color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    for idx, node_idx in enumerate(nodes_index):
        cluster_id = clusters[idx] - 1
        ax.scatter(
            voronoi_graph_vertices[node_idx][0], 
            -voronoi_graph_vertices[node_idx][1], 
            color=colors(cluster_id), 
            s=100,
            alpha=0.6,
            edgecolors='k'
        )
    for cluster_id in unique_clusters:
        ax.scatter([], [], color=colors(cluster_id - 1), label=f'Subregion {cluster_id}')
    ax.legend()

    ax.set_title("Voronoi Subregions Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("-Y Coordinate")
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def update_with_subregion(subregion: int, voronoi_graph: Dict) -> Tuple[np.ndarray, np.ndarray]:
    keys_with_current_subregion = [key for key, value in voronoi_graph['subregions'].items() if value == subregion]
    current_subregion_nodes_index = np.array([key for key in keys_with_current_subregion if key in voronoi_graph['nodes_index']])
    current_subregion_nodes_score = voronoi_graph['nodes_score'][np.isin(voronoi_graph['nodes_index'], current_subregion_nodes_index)]
    current_subregion_nodes_invisibility_score = voronoi_graph['nodes_invisibility_score'][np.isin(voronoi_graph['nodes_index'], current_subregion_nodes_index)]
    return current_subregion_nodes_index, current_subregion_nodes_score, current_subregion_nodes_invisibility_score
    
class TurnLineTestResult(Enum):
    BOTH_FREE_SPACE = 0
    LEFT_FREE_SPACE = 1
    RIGHT_FREE_SPACE = -1
    LEFT_MORE_FREE_SPACE = 2
    RIGHT_MORE_FREE_SPACE = -2
    RIGHT_TRY_FAILED = 3
    LEFT_TRY_FAILED = -3
    BOTH_FREE_SPACE_WITH_OBSTACLE = 4
    BOTH_TRY_FAILED = 5
    
def get_escape_plan(
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_rotation_vector:np.ndarray,
    agent_turn_angle:float,
    agent_step_size_pixel:float,
    inaccessible_database:np.ndarray) -> Tuple[int, np.ndarray]:
    # FIXME: Fix the bug in escape plan and visualize it.
    agent_turn_angle_rad = np.radians(agent_turn_angle)
    turn_times_half = int(np.ceil(180 / agent_turn_angle))
    turn_left_theta = (np.arange(turn_times_half) + 1) * agent_turn_angle_rad
    turn_right_theta = -turn_left_theta
    assert np.allclose(np.linalg.norm(agent_rotation_vector), 1), 'Agent rotation vector should be normalized.'
    turn_left_rotation_vectors = np.vstack((
        agent_rotation_vector[0] * np.cos(turn_left_theta) -\
            agent_rotation_vector[1] * np.sin(turn_left_theta),
        agent_rotation_vector[0] * np.sin(turn_left_theta) +\
            agent_rotation_vector[1] * np.cos(turn_left_theta))).T
    turn_right_rotation_vectors = np.vstack((
        agent_rotation_vector[0] * np.cos(turn_right_theta) -\
            agent_rotation_vector[1] * np.sin(turn_right_theta),
        agent_rotation_vector[0] * np.sin(turn_right_theta) +\
            agent_rotation_vector[1] * np.cos(turn_right_theta))).T
    assert np.allclose(np.linalg.norm(turn_left_rotation_vectors, axis=1), 1), 'Turn left rotation vectors should be normalized.'
    assert np.allclose(np.linalg.norm(turn_right_rotation_vectors, axis=1), 1), 'Turn right rotation vectors should be normalized.'
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    if len(inaccessible_database) > 0:
        turn_left_rotation_vectors_inaccessible = cdist(turn_left_rotation_vectors, inaccessible_database)
        turn_right_rotation_vectors_inaccessible = cdist(turn_right_rotation_vectors, inaccessible_database)
        turn_left_rotation_vectors_inaccessible = np.any(turn_left_rotation_vectors_inaccessible < agent_turn_angle_rad * 0.1, axis=1)
        turn_right_rotation_vectors_inaccessible = np.any(turn_right_rotation_vectors_inaccessible < agent_turn_angle_rad * 0.1, axis=1)
    else:
        turn_left_rotation_vectors_inaccessible = np.zeros(turn_times_half, dtype=bool)
        turn_right_rotation_vectors_inaccessible = np.zeros(turn_times_half, dtype=bool)
    line_test_results = []
    for turn_left_rotation_vector, turn_left_rotation_vector_inaccessible, turn_right_rotation_vector, turn_right_rotation_vector_inaccessible in zip(turn_left_rotation_vectors, turn_left_rotation_vectors_inaccessible, turn_right_rotation_vectors, turn_right_rotation_vectors_inaccessible):
        if turn_left_rotation_vector_inaccessible:
            turn_left_free_space_pixels_num = np.inf
        else:
            turn_left_line_test_result = cv2.line(
                obstacle_map.copy(),
                np.int32(agent_position),
                np.int32(agent_position + turn_left_rotation_vector * agent_step_size_pixel),
                255,
                1)
            turn_left_free_space_pixels_num = cv2.countNonZero(turn_left_line_test_result)
        if turn_right_rotation_vector_inaccessible:
            turn_right_free_space_pixels_num = np.inf
        else:
            turn_right_line_test_result = cv2.line(
                obstacle_map.copy(),
                np.int32(agent_position),
                np.int32(agent_position + turn_right_rotation_vector * agent_step_size_pixel),
                255,
                1)
            turn_right_free_space_pixels_num = cv2.countNonZero(turn_right_line_test_result)
        assert turn_left_free_space_pixels_num >= free_space_pixels_num, 'Turn left line test result should have more free space pixels than the obstacle map.'
        assert turn_right_free_space_pixels_num >= free_space_pixels_num, 'Turn right line test result should have more free space pixels than the obstacle map.'
        if turn_left_free_space_pixels_num == free_space_pixels_num == turn_right_free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.BOTH_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num == free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.LEFT_FREE_SPACE.value)
        elif turn_right_free_space_pixels_num == free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.RIGHT_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num == turn_right_free_space_pixels_num == np.inf:
            line_test_results.append(TurnLineTestResult.BOTH_TRY_FAILED.value)
        elif turn_right_free_space_pixels_num == np.inf:
            line_test_results.append(TurnLineTestResult.RIGHT_TRY_FAILED.value)
        elif turn_left_free_space_pixels_num == np.inf:
            line_test_results.append(TurnLineTestResult.LEFT_TRY_FAILED.value)
        elif turn_left_free_space_pixels_num - free_space_pixels_num < turn_right_free_space_pixels_num - free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.LEFT_MORE_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num - free_space_pixels_num > turn_right_free_space_pixels_num - free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.RIGHT_MORE_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num == turn_right_free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.BOTH_FREE_SPACE_WITH_OBSTACLE.value)
        else:
            raise ValueError('Invalid turn line test result.')
    line_test_results = np.array(line_test_results)
    line_test_results_abs = np.abs(line_test_results)
    if 1 in line_test_results_abs:
        indices = np.argwhere(line_test_results_abs == 1)
        first_index = indices[0, 0]
        rotation_direction = line_test_results[first_index]
    else:
        line_test_results_condition = np.logical_or(
            line_test_results_abs == TurnLineTestResult.BOTH_TRY_FAILED.value,
            line_test_results_abs == TurnLineTestResult.BOTH_FREE_SPACE_WITH_OBSTACLE.value)
        line_test_results[line_test_results_condition] = 0
        rotation_direction = np.sign(np.sum(line_test_results))
        if rotation_direction == 0:
            rotation_direction = np.random.choice([-1, 1])
        
    turn_times = int(np.ceil(360 / agent_turn_angle))
    turn_test_condition = np.zeros(turn_times, dtype=bool)
    if rotation_direction == TurnLineTestResult.LEFT_FREE_SPACE.value:
        turn_test_condition[:turn_times_half] = line_test_results != TurnLineTestResult.LEFT_TRY_FAILED.value
    elif rotation_direction == TurnLineTestResult.RIGHT_FREE_SPACE.value:
        turn_test_condition[:turn_times_half] = line_test_results != TurnLineTestResult.RIGHT_TRY_FAILED.value
    else:
        raise ValueError('Invalid rotation direction.')
    
    turn_test_condition_index_remain = np.arange(turn_times_half, turn_times)
    turn_theta_remain = (turn_test_condition_index_remain + 1) * agent_turn_angle_rad * rotation_direction
    turn_rotation_vectors_remain = np.vstack((
        agent_rotation_vector[0] * np.cos(turn_theta_remain) -\
            agent_rotation_vector[1] * np.sin(turn_theta_remain),
        agent_rotation_vector[0] * np.sin(turn_theta_remain) +\
            agent_rotation_vector[1] * np.cos(turn_theta_remain))).T
    if len(inaccessible_database) > 0:
        turn_rotation_vectors_remain_inaccessible = cdist(turn_rotation_vectors_remain, inaccessible_database)
        turn_rotation_vectors_remain_inaccessible = np.any(turn_rotation_vectors_remain_inaccessible < agent_turn_angle_rad * 0.1, axis=1)
    else:
        turn_rotation_vectors_remain_inaccessible = np.zeros(turn_times_half, dtype=bool)
    for turn_test_condition_index, turn_rotation_vector_inaccessible in zip(turn_test_condition_index_remain, turn_rotation_vectors_remain_inaccessible):
        if turn_rotation_vector_inaccessible:
            continue
        else:
            turn_test_condition[turn_test_condition_index] = True
    assert np.any(turn_test_condition), 'No valid turn test condition.'
    return rotation_direction, turn_test_condition

def interpolate_path(
    navigation_path:np.ndarray,
    interpolate_number:int=50) -> np.ndarray:
    tck, u = splprep(navigation_path.T, s=0)
    u = np.linspace(0, 1, interpolate_number)
    navigation_path_interpolated = splev(u, tck)
    return np.vstack(navigation_path_interpolated).T
    