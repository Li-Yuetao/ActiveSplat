#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import argparse
import threading
import json
from typing import Dict, Tuple, Union, List
from copy import deepcopy
from enum import Enum

import faulthandler

import torch
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import quaternion
from matplotlib import cm, colors
import cv2
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point

from utils import PROJECT_NAME, GlobalState
from utils.logging_utils import Log
from utils.gui_utils import PoseChangeType, c2w_world_to_topdown, c2w_topdown_to_world, is_pose_changed, get_horizon_bound_topdown
from dataloader import PoseDataType, convert_to_c2w_opencv
from planner.planner import Frustum, get_voronoi_graph, draw_voronoi_graph, get_closest_vertex_index, get_safe_dijkstra_path, get_escape_plan, get_obstacle_map, interpolate_path, get_closest_node_index, get_subregions, update_with_subregion
from scripts.nodes import TURN, SPEED, USE_ROTATION_SELECTION, USE_HIGH_CONNECTIVITY, USE_RANDOM_SELECTION, USE_HIERARCHICAL_PLAN,\
    GetTopdownConfig, GetTopdownConfigResponse, GetTopdownConfigRequest,\
        GetTopdown, GetTopdownResponse, GetTopdownRequest,\
            SetPlannerState, SetPlannerStateResponse, SetPlannerStateRequest,\
                GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
                    SetMapper, SetMapperResponse, SetMapperRequest,\
                        GetOpacity, GetOpacityRequest, GetOpacityResponse,\
                            GetVoronoiGraph, GetVoronoiGraphRequest, GetVoronoiGraphResponse,\
                                GetNavPath, GetNavPathRequest, GetNavPathResponse
class NodesFlagsType(Enum):
    UNARRIVED = 'UNARRIVED'
    IN_HORIZON = 'IN_HORIZON'
    OPACITY_INVISIBILITY = 'OPACITY_INVISIBILITY'
    HOLE_INVISIBILITY = 'HOLE_INVISIBILITY'
    REAL_OPACITY_INVISIBILITY = 'REAL_OPACITY_INVISIBILITY'
    FAIL = 'FAIL'

NODES_FLAGS_WEIGHT_INIT = {
    NodesFlagsType.UNARRIVED: 20,
    NodesFlagsType.IN_HORIZON: 10,
    NodesFlagsType.OPACITY_INVISIBILITY: 2,
    NodesFlagsType.HOLE_INVISIBILITY: 1,
    NodesFlagsType.REAL_OPACITY_INVISIBILITY: 1,
    NodesFlagsType.FAIL: -60
}

class PlannerNode:
    
    __ENABLE_STATES = (GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING)

    __NODES_FLAGS_WEIGHT = NODES_FLAGS_WEIGHT_INIT.copy()
    class EscapeFlag(Enum):
        NONE = 'NONE'
        ESCAPE_ROTATION = 'ESCAPE_ROTATION'
        ESCAPE_TRANSLATION = 'ESCAPE_TRANSLATION'
    
    def __init__(
        self,
        config_url:str,
        hide_windows:bool,
        save_runtime_data:bool) -> None:
        self.__hide_windows = hide_windows
        self.__voronoi_graph_nodes_score_max = 0
        self.__voronoi_graph_nodes_score_min = 0
        for key, value in self.__NODES_FLAGS_WEIGHT.items():
            if value > 0 and (key == NodesFlagsType.OPACITY_INVISIBILITY or key == NodesFlagsType.HOLE_INVISIBILITY):
                self.__voronoi_graph_nodes_score_max += value * 10
            elif value > 0:
                self.__voronoi_graph_nodes_score_max += value
            elif value < 0:
                self.__voronoi_graph_nodes_score_min += value
        
        voronoi_graph_nodes_colormap = cm.get_cmap('Reds')
        voronoi_graph_nodes_colormap_colors = voronoi_graph_nodes_colormap(np.linspace(0.25, 1, 256))
        self.__voronoi_graph_nodes_colormap = colors.LinearSegmentedColormap.from_list('voronoi_graph_nodes_colormap', voronoi_graph_nodes_colormap_colors)
        
        os.chdir(PACKAGE_PATH)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(config_url) as f:
            config = json.load(f)
            
        self.__pose_update_translation_threshold = config['mapper']['pose']['update_threshold']['translation']
        self.__pose_update_rotation_threshold = config['mapper']['pose']['update_threshold']['rotation']
        self.__step_num_as_visited = config['planner']['step_num_as_visited']
        self.__step_num_as_arrived = config['planner']['step_num_as_arrived']
        self.__step_num_as_too_far = 200
        self.__max_pitch_angle = config['planner']['max_pitch_angle']
        self.__local_view_count_limit = config['planner']['local_view_limit']
        self.__radius_num_as_rotated = config['planner']['radius_num_as_rotated']
        self.__save_runtime_data = save_runtime_data
        
        self.__global_state = None
        self.__global_state_condition = threading.Condition()
        rospy.Service('set_planner_state', SetPlannerState, self.__set_planner_state)
        rospy.Service('get_voronoi_graph', GetVoronoiGraph, self.__get_voronoi_graph_callback)
        rospy.Service('get_navigation_path', GetNavPath, self.__get_navigation_path_callback)
        with self.__global_state_condition:
            self.__global_state_condition.wait()
        
        self.__get_dataset_config_service = rospy.ServiceProxy('get_dataset_config', GetDatasetConfig)
        rospy.wait_for_service('get_dataset_config')
        
        self.__get_topdown_config_service = rospy.ServiceProxy('get_topdown_config', GetTopdownConfig)
        rospy.wait_for_service('get_topdown_config')

        self.__get_topdown_service = rospy.ServiceProxy('get_topdown', GetTopdown)
        rospy.wait_for_service('get_topdown')
        
        self.__update_map_cv2_condition = threading.Condition()
        
        self.__setup_for_episode(init=True)
        
        set_mapper = rospy.ServiceProxy('set_mapper', SetMapper)
        rospy.wait_for_service('set_mapper')
        
        rospy.Subscriber('high_loss_samples_pose', Pose, self.__get_high_loss_samples_pose)
        self.__get_opacity_service = rospy.ServiceProxy('get_opacity', GetOpacity)
        rospy.wait_for_service('get_opacity')

        rospy.Subscriber('orb_slam3/camera_pose', PoseStamped, self.__camera_pose_callback)
        rospy.wait_for_message('orb_slam3/camera_pose', PoseStamped)
        
        rospy.Subscriber('movement_fail_times', Int32, self.__movement_fail_times_callback)
        
        self.__cv2_windows_with_callback_opened = {
            'topdown_free_map': False}
        
        threading.Thread(
            name='update_map_cv2',
            target=self.__update_map_cv2,
            daemon=True).start()
        
        self.__cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.__trigger_update_voronoi_graph_pub = rospy.Publisher('update_voronoi_graph_vis', Bool, queue_size=1)
        self.__trigger_update_high_connectivity_nodes_pub = rospy.Publisher('update_high_connectivity_nodes_vis', Bool, queue_size=1)
        self.__trigger_update_global_visibility_map_pub = rospy.Publisher('update_global_visibility_map_vis', Int32, queue_size=1)
        self.__fail_vertices_nodes_index = []
        
        while not rospy.is_shutdown() and self.__global_state != GlobalState.QUIT:
            if self.__global_state not in self.__ENABLE_STATES:
                if self.__global_state == GlobalState.REPLAY:
                    self.__setup_for_episode()
                with self.__global_state_condition:
                    self.__global_state_condition.wait()
                    self.__rotation_arrived_flag = True
                continue
            else:
                if self.__bootstrap_flag:
                    set_mapper_request:SetMapperRequest = SetMapperRequest()
                    set_mapper_request.kf_every = 1
                    set_mapper_request.map_every = 2
                    try:
                        set_mapper_response:SetMapperResponse = set_mapper(set_mapper_request)
                    except rospy.ServiceException as e:
                        rospy.logerr(f'Set mapper service call failed: {e}')
                        self.__global_state = GlobalState.QUIT
                        continue
                    kf_every_old = set_mapper_response.kf_every_old
                    map_every_old = set_mapper_response.map_every_old
                    twist_bootstrap = Twist()
                    twist_bootstrap.angular.z = 1
                    twist_bootstrap_up_down = Twist()
                    self.__rotation_arrived_flag = False
                    for booststrap_turn_index in range(int(np.ceil(360 / self.__dataset_config.agent_turn_angle))):
                        
                        pose_c2w_world = self.__pose_last['c2w_world'].copy()
                        self.__publish_cmd_vel(twist_bootstrap)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        while is_pose_changed(
                                pose_c2w_world,
                                self.__pose_last['c2w_world'],
                                self.__pose_update_translation_threshold,
                                self.__pose_update_rotation_threshold) == PoseChangeType.NONE and self.__global_state != GlobalState.QUIT:
                            self.__publish_cmd_vel(twist_bootstrap)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                                
                        pose_c2w_world = self.__pose_last['c2w_world'].copy()
                        updown_times = 3
                        twist_bootstrap_up_down.angular.y = -1 if (((2*updown_times-1-booststrap_turn_index % (2*updown_times) * 2)) < 0) else 1
                        self.__publish_cmd_vel(twist_bootstrap_up_down)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        while is_pose_changed(
                                pose_c2w_world,
                                self.__pose_last['c2w_world'],
                                self.__pose_update_translation_threshold,
                                self.__pose_update_rotation_threshold) == PoseChangeType.NONE and self.__global_state != GlobalState.QUIT:
                            self.__publish_cmd_vel(twist_bootstrap_up_down)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                    booststrap_turn_index += 1
                    if booststrap_turn_index % 2 == 1:
                        pose_c2w_world = self.__pose_last['c2w_world'].copy()
                        twist_bootstrap_up_down.angular.y = -1
                        self.__publish_cmd_vel(twist_bootstrap_up_down)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        while is_pose_changed(
                                pose_c2w_world,
                                self.__pose_last['c2w_world'],
                                self.__pose_update_translation_threshold,
                                self.__pose_update_rotation_threshold) == PoseChangeType.NONE and self.__global_state != GlobalState.QUIT:
                            self.__publish_cmd_vel(twist_bootstrap_up_down)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                    set_mapper_request.kf_every = kf_every_old
                    set_mapper_request.map_every = map_every_old
                    try:
                        set_mapper_response:SetMapperResponse = set_mapper(set_mapper_request)
                    except rospy.ServiceException as e:
                        rospy.logerr(f'Set mapper service call failed: {e}')
                        self.__global_state = GlobalState.QUIT
                        continue
                    self.__bootstrap_flag = False
                    self.__rotation_arrived_flag = True
                    self.__high_connectivity_node_view_count = 0
                    if self.__global_state == GlobalState.MANUAL_PLANNING:
                        Log("Manual planning arrived. Please click the target node in the Topdown Free Map and press 'Enter' to continue", tag='ActiveSplat')
                elif self.__rotation_arrived_flag:
                    self.__publish_cmd_vel(Twist())
                    self.__get_topdown()
                    if not (self.__global_state in self.__ENABLE_STATES):
                        continue
                    if self.__global_state == GlobalState.AUTO_PLANNING:
                        pose_last = self.__pose_last['topdown_translation'].copy()
                        closest_vertex_index = get_closest_vertex_index(
                            self.__voronoi_graph['vertices'],
                            self.__voronoi_graph['obstacle_map'],
                            pose_last,
                            self.__agent_radius_pixel)
                        self.closest_node_index = get_closest_node_index(
                            self.__voronoi_graph['vertices'],
                            self.__voronoi_graph['nodes_index'],
                            self.__pose_last['topdown_translation'])
                        self.__navigation_path = None
                        self.__destination_orientations = None
                        
                        # NOTE: Get agent's current subregion
                        current_subregion = None
                        if self.closest_node_index in self.__voronoi_graph['subregions'].keys():
                            current_subregion = self.__voronoi_graph['subregions'][self.closest_node_index]
                            
                        if USE_HIERARCHICAL_PLAN and current_subregion is not None:
                            current_subregion_nodes_index, current_subregion_nodes_score, current_subregion_nodes_invisibility_score = update_with_subregion(current_subregion, self.__voronoi_graph)
                            arrived_nodes_count = 0
                            for node_index in current_subregion_nodes_index:
                                if self.__is_close_to_position_selected(self.__voronoi_graph['vertices'][node_index]):
                                    current_subregion_nodes_score[current_subregion_nodes_index.tolist().index(node_index)] = 0 # delete the node which is arrived
                                    arrived_nodes_count += 1
                                if current_subregion_nodes_score[current_subregion_nodes_index.tolist().index(node_index)] <= 0:
                                    current_subregion_nodes_invisibility_score[current_subregion_nodes_index.tolist().index(node_index)] = 0
                            # max score in current subregion
                            subregion_max_score_threshold = 250
                            is_max_score_below_threshold = np.nanmax(current_subregion_nodes_invisibility_score) < subregion_max_score_threshold
                            all_nodes_are_visited = arrived_nodes_count == len(current_subregion_nodes_index)
                            if not all_nodes_are_visited:
                                if is_max_score_below_threshold:
                                    Log("The max score in the current subregion is below the threshold, use 'Global Plan'.", tag='ActiveSplat')
                                    global_plan_flag = True
                                else:
                                    Log("There are still nodes higher than the threshold in the current subregion, use 'Local Plan'.", tag='ActiveSplat')
                                    global_plan_flag = False
                            else:
                                Log("All nodes in the current subregion are visited, use 'Global Plan'.", tag='ActiveSplat')
                                global_plan_flag = True
                            if self.use_global_plan_flag or \
                                global_plan_flag:
                                # NOTE: Global Plan
                                if self.use_global_plan_flag:
                                    self.use_global_plan_flag = False
                                current_subregion_nodes_score = self.__voronoi_graph['nodes_score']
                                current_subregion_nodes_index = self.__voronoi_graph['nodes_index']
                                subregion_nodes_path_length = {}
                                subregion_nodes_score = {}
                                for node_index, subregion in self.__voronoi_graph['subregions'].items():
                                    subregion_nodes_path_length.setdefault(subregion, [])
                                    subregion_nodes_score.setdefault(subregion, [])
                                    
                                    node_vertice = self.__voronoi_graph['vertices'][node_index]
                                    if subregion == current_subregion:
                                        rospy.logdebug(f'The node {node_index} is in the current subregion, skip.')
                                        subregion_nodes_path_length[subregion].append(np.nan)
                                        continue
                                    if self.__is_close_to_arrived(node_vertice):
                                        rospy.logdebug(f'The node {node_index} is close to the arrived position, skip.')
                                        subregion_nodes_path_length[subregion].append(np.nan)
                                        continue
                                    
                                    navigation_path_index, navigation_path, graph_search_success = get_safe_dijkstra_path(
                                        self.__voronoi_graph['graph'],
                                        closest_vertex_index,
                                        node_index,
                                        self.__voronoi_graph['vertices'],
                                        self.__voronoi_graph['obstacle_map'],
                                        pose_last,
                                        self.__agent_radius_pixel
                                    )
                                    
                                    if navigation_path_index is None or navigation_path is None:
                                        subregion_nodes_path_length[subregion].append(np.nan)
                                    else:
                                        whole_path = np.vstack([pose_last, navigation_path])
                                        whole_path_length = np.sum(np.linalg.norm(whole_path[1:] - whole_path[:-1], axis=1))
                                        subregion_nodes_path_length[subregion].append(whole_path_length)
                                    subregion_nodes_score[subregion].append(self.__voronoi_graph['nodes_score'][self.__voronoi_graph['nodes_index'].tolist().index(node_index)])

                                # The max score per subregion
                                subregions_max_score = {
                                    subregion: np.max([max(score, 0) for score in scores]) if scores else 0
                                    for subregion, scores in subregion_nodes_score.items()
                                }
                                max_score_subregion = max(subregions_max_score, key=subregions_max_score.get)
                                current_subregion_nodes_index, current_subregion_nodes_score, _ = update_with_subregion(max_score_subregion, self.__voronoi_graph)
                            else:
                                # Local Plan
                                pass
                        else:
                            # Global plan
                            current_subregion_nodes_score = self.__voronoi_graph['nodes_score']
                            current_subregion_nodes_index = self.__voronoi_graph['nodes_index']
                        target_too_far_but_prioritize = {
                            "node_index": None,
                            "navigation_path": None,
                            "navigation_path_length": None}
                        for map_used, nodes_scores_used, nodes_index_used, bootstrap_used, max_step_used, agent_radius_pixel_used in zip(
                            [self.__voronoi_graph['obstacle_map'], ],
                            [current_subregion_nodes_score, ],
                            [current_subregion_nodes_index, ],
                            [True, ],
                            [self.__step_num_as_too_far, ],
                            [self.__agent_radius_pixel, ]):
                            
                            if self.__navigation_path is not None:
                                break
                            target_too_far_but_prioritize = {
                                "node_index": None,
                                "navigation_path": None,
                                "navigation_path_length": None}
                            for voronoi_graph_score in range(max(nodes_scores_used), min(nodes_scores_used) - 1, -1):
                                # NOTE: Select the node with the highest score. If the scores are the same, select the closest node.
                                nodes_condition = nodes_scores_used == voronoi_graph_score
                                nodes_index = nodes_index_used[nodes_condition]
                                nodes_path = []
                                nodes_path_length = []
                                nodes_path_index = []
                                for node_index in nodes_index:
                                    node_vertice = self.__voronoi_graph['vertices'][node_index]
                                    if np.linalg.norm(pose_last - node_vertice) < self.__pixel_as_arrived:
                                        nodes_path_length.append(np.nan)
                                        nodes_path.append(None)
                                        nodes_path_index.append(None)
                                        continue
                                    navigation_path_index, navigation_path, graph_search_success = get_safe_dijkstra_path(
                                        self.__voronoi_graph['graph'],
                                        closest_vertex_index,
                                        node_index,
                                        self.__voronoi_graph['vertices'],
                                        map_used,
                                        pose_last,
                                        agent_radius_pixel_used)
                                    if not graph_search_success:
                                        self.__fail_vertices_nodes_index.append(node_index)
                                        self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__voronoi_graph['vertices'][node_index]])
                                    if navigation_path_index is None or navigation_path is None:
                                        nodes_path_length.append(np.nan)
                                    else:
                                        whole_path = np.vstack([pose_last, navigation_path])
                                        whole_path_length = np.sum(np.linalg.norm(whole_path[1:] - whole_path[:-1], axis=1))
                                        nodes_path_length.append(whole_path_length)
                                    nodes_path.append(navigation_path)
                                    if navigation_path is None:
                                        nodes_path_index.append(None)
                                    else:
                                        nodes_path_index.append(navigation_path_index)
                                nodes_path_length = np.array(nodes_path_length)
                                if np.all(np.isnan(nodes_path_length)):
                                    continue
                                else:
                                    if self.__NODES_FLAGS_WEIGHT == NODES_FLAGS_WEIGHT_INIT:
                                        if (target_too_far_but_prioritize["node_index"] is not None) and\
                                        (target_too_far_but_prioritize["navigation_path"] is not None) and\
                                        (target_too_far_but_prioritize["navigation_path_length"] is not None):
                                            nodes_path_condition = nodes_path_length < max_step_used * self.__agent_step_size_pixel
                                            if np.any(nodes_path_condition):
                                                nodes_index = nodes_index[nodes_path_condition]
                                                nodes_path = [nodes_path[i] for i in np.where(nodes_path_condition)[0]]
                                                nodes_path_length = nodes_path_length[nodes_path_condition]
                                                nodes_to_target_node = []
                                                nodes_to_target_node_length = []
                                                for node_index in nodes_index:
                                                    node_navigation_path_index, node_navigation_path, node_graph_search_success = get_safe_dijkstra_path(
                                                        self.__voronoi_graph['graph'],
                                                        node_index,
                                                        target_too_far_but_prioritize["node_index"],
                                                        self.__voronoi_graph['vertices'],
                                                        map_used,
                                                        pose_last,
                                                        agent_radius_pixel_used)
                                                    if node_navigation_path_index is None or node_navigation_path is None:
                                                        nodes_to_target_node.append(None)
                                                        nodes_to_target_node_length.append(np.nan)
                                                    else:
                                                        node_navigation_path_length = np.sum(np.linalg.norm(node_navigation_path[1:] - node_navigation_path[:-1], axis=1))
                                                        nodes_to_target_node.append(node_navigation_path)
                                                        nodes_to_target_node_length.append(node_navigation_path_length)
                                                nodes_to_target_node_length = np.array(nodes_to_target_node_length)
                                                nodes_to_target_node_length_condition = nodes_to_target_node_length < target_too_far_but_prioritize["navigation_path_length"]
                                                if np.any(nodes_to_target_node_length_condition):
                                                    node_count = np.nanargmin(nodes_to_target_node_length)
                                                    node_index = nodes_index[node_count]
                                                    navigation_path = nodes_path[node_count]
                                                    navigation_path_length = nodes_path_length[node_count]
                                                else:
                                                    continue
                                            else:
                                                continue
                                        else:
                                            node_count = np.nanargmin(nodes_path_length)
                                            node_index = nodes_index[node_count]
                                            navigation_path = nodes_path[node_count]
                                            navigation_path_length = nodes_path_length[node_count]
                                            if navigation_path_length > max_step_used * self.__agent_step_size_pixel:
                                                if bootstrap_used:  
                                                    target_too_far_but_prioritize["node_index"] = node_index
                                                    target_too_far_but_prioritize["navigation_path"] = navigation_path
                                                    target_too_far_but_prioritize["navigation_path_length"] = navigation_path_length
                                                continue
                                    else:
                                        # NOTE: Randomly choose a node
                                        non_nan_node_counts = np.where(np.invert(np.isnan(nodes_path_length)))[0]
                                        node_count = np.random.choice(non_nan_node_counts)
                                        node_index = nodes_index[node_count]
                                        navigation_path = nodes_path[node_count]
                                    self.__navigation_path = navigation_path
                                    self.__interpolate_path()
                                    self.__navigation_path_index = nodes_path_index[node_count]
                                    # NOTE: update global invisibility map
                                    self.__trigger_update_global_visibility_map_pub.publish(Int32(node_index))
                                    break
                            if self.__navigation_path is None:
                                if (target_too_far_but_prioritize["node_index"] is not None) and\
                                    (target_too_far_but_prioritize["navigation_path"] is not None) and\
                                        (target_too_far_but_prioritize["navigation_path_length"] is not None):
                                    self.__navigation_path = target_too_far_but_prioritize["navigation_path"]
                                    self.__interpolate_path()
                                elif bootstrap_used:
                                    rospy.logwarn('No node is reachable.')
                                    self.__bootstrap_flag = True
                                    self.use_global_plan_flag = True
                    with self.__update_map_cv2_condition:
                        self.__update_map_cv2_condition.notify_all()
                        if self.__global_state == GlobalState.MANUAL_PLANNING:
                            self.__navigation_path = None
                            self.__destination_orientations = None
                            # NOTE: Wait for the manual click for the destination.
                            self.__update_map_cv2_condition.wait()
                    if self.__navigation_path is not None:
                        self.__rotation_arrived_flag = False
                elif self.__position_arrived_flag:
                    # NOTE: position arrived
                    self.__get_topdown()
                    
                    if self.__check_agent_close_to_obstacle_flag:
                        # check agent is close to the obstacle
                        obstacle_distance_threshold = self.__agent_radius_pixel * 1.0
                        if self.__is_close_to_obstacle(self.__pose_last['topdown_translation'], obstacle_distance_threshold):
                            self.__rotation_arrived_flag = True
                            self.__position_arrived_flag = False
                            self.__local_path_executing = False
                            self.__local_view_count = 1
                            rospy.logwarn('Agent is close to the obstacle. Skip the local view.')
                            continue
                        self.__check_agent_close_to_obstacle_flag = False
                    
                    if not self.__local_path_executing:
                        try:
                            get_opacity_response:GetOpacityResponse = self.__get_opacity_service(GetOpacityRequest(self.__rotation_arrived_flag, [], []))
                        except rospy.ServiceException as e:
                            rospy.logerr(f'Get local opacity service call failed: {e}')
                            self.__global_state = GlobalState.QUIT
                            with self.__global_state_condition:
                                self.__global_state_condition.notify()
                            return
                        
                        self.__local_invisibility_info = dict()
                        for idx, target_frustum in enumerate(get_opacity_response.targets_frustums):
                            target_frustum:Pose
                            self.__local_invisibility_info[idx] = Frustum()
                            if target_frustum.position.x == 0 and target_frustum.position.y == 0 and target_frustum.position.z == 0:
                                continue
                            target_frustum_quaternion = np.array([
                                target_frustum.orientation.w,
                                target_frustum.orientation.x,
                                target_frustum.orientation.y,
                                target_frustum.orientation.z])
                            target_frustum_c2w_world = np.eye(4)
                            target_frustum_c2w_world[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(target_frustum_quaternion))
                            target_frustum_c2w_world[:3, 3] = np.array([
                                target_frustum.position.x,
                                target_frustum.position.y,
                                target_frustum.position.z])
                            target_frustum_rotation_vector, target_frustum_translation, pitch_angle = c2w_world_to_topdown(
                                target_frustum_c2w_world,
                                self.__topdown_config,
                                self.__height_direction,
                                np.float64,
                                need_pitch=True)
                            self.__local_invisibility_info[idx].rotation_vector_2d = target_frustum_rotation_vector
                            self.__local_invisibility_info[idx].translation_2d = target_frustum_translation
                            self.__local_invisibility_info[idx].pitch_angle = pitch_angle

                        self.__destination_orientations = None
                        if self.__local_invisibility_info[0].rotation_vector_2d is not None:
                            rotation_vector_2d = self.__local_invisibility_info[0].rotation_vector_2d.copy()
                            target_horizon_orientation = np.arctan2(rotation_vector_2d[1], rotation_vector_2d[0])
                            self.__destination_orientations = np.expand_dims(target_horizon_orientation, axis=0)
                            if self.__local_invisibility_info[0].pitch_angle is not None:
                                self.__destination_orientations = np.expand_dims(np.append(self.__destination_orientations[0], self.__local_invisibility_info[0].pitch_angle), axis=0)
                        else:
                            Log('No local viewpoint is selected.', tag='ActiveSplat')
                            # NOTE: Check if there are still high loss samples
                            if len(self.__local_invisibility_info) == 2:
                                if self.__local_invisibility_info[1] is not None:
                                    self.__destination_orientations = self.__local_invisibility_info[1].rotation_vector_2d.copy()
                                    if self.__local_invisibility_info[1].pitch_angle is not None:
                                        self.__destination_orientations = np.expand_dims(np.append(self.__destination_orientations, self.__local_invisibility_info[1].pitch_angle), axis=0)
                                        
                    start_horizon_orientation = np.arctan2(self.__pose_last['topdown_rotation_vector'][1], self.__pose_last['topdown_rotation_vector'][0])
                    start_vertical_orientation = self.__pose_last['pitch_angle']
                    if len(self.__local_invisibility_info) == 2:
                        if self.__local_invisibility_info[1] is not None:
                            target_horizon_orientation = np.arctan2(self.__local_invisibility_info[1].rotation_vector_2d[1], self.__local_invisibility_info[1].rotation_vector_2d[0])
                            diff_horizon_orientation = (np.degrees(target_horizon_orientation - start_horizon_orientation) + 180) % 360 - 180
                            if np.abs(diff_horizon_orientation) < 5:
                                Log('High loss samples have been observed, skipped', tag='ActiveSplat')
                                self.__local_invisibility_info[1] = None
                            else:
                                Log(f'High loss samples still exist, diff_horizon_orientation = {diff_horizon_orientation}', tag='ActiveSplat')
                    if self.__destination_orientations is not None and self.__destination_orientations.size > 0:
                        if not self.__local_path_executing:
                            self.__local_path_executing = True
                        if self.__local_set_mapper_flag:
                            set_mapper_request:SetMapperRequest = SetMapperRequest()
                            set_mapper_request.kf_every = 2
                            set_mapper_request.map_every = 2
                            if self.kf_every_old == set_mapper_request.kf_every and self.map_every_old == set_mapper_request.map_every:
                                self.__local_path_executing = True
                                pass
                            try:
                                set_mapper_response:SetMapperResponse = set_mapper(set_mapper_request)
                            except rospy.ServiceException as e:
                                rospy.logerr(f'Set mapper service call failed: {e}')
                                self.__global_state = GlobalState.QUIT
                                continue
                            self.kf_every_old = set_mapper_response.kf_every_old
                            self.map_every_old = set_mapper_response.map_every_old
                            self.__local_path_executing = True
                            self.__local_set_mapper_flag = False
                            self.__local_view_count = 1
                        # NOTE: Adjust heading, yaw and pitch
                        diff_vertical_orientation = self.__destination_orientations[0][1] - start_vertical_orientation
                        diff_horizon_orientation = (np.degrees(self.__destination_orientations[0][0] - start_horizon_orientation) + 180) % 360 - 180
                        if np.abs(diff_vertical_orientation) - self.__dataset_config.agent_tilt_angle > 0: 
                            cmd_vel_msg = Twist()
                            if diff_vertical_orientation < 0:
                                # looking up
                                cmd_vel_msg.angular.y = 1
                            elif diff_vertical_orientation > 0:
                                # looking down
                                cmd_vel_msg.angular.y = -1
                            else:
                                raise ValueError('Unknown vertical condition.')
                            self.__publish_cmd_vel(cmd_vel_msg)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                            continue
                        elif np.abs(diff_horizon_orientation) > self.__dataset_config.agent_turn_angle > 0:
                            cmd_vel_msg = Twist()
                            if diff_horizon_orientation > self.__dataset_config.agent_turn_angle:
                                cmd_vel_msg.angular.z = -TURN
                            elif diff_horizon_orientation < -self.__dataset_config.agent_turn_angle:
                                cmd_vel_msg.angular.z = TURN
                            else:
                                raise ValueError('Unknown horizon condition.')
                            self.__publish_cmd_vel(cmd_vel_msg)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                            continue
                        # NOTE: turn back to horizon
                        if self.__destination_orientations[0][1] is not None:
                            self.__destination_orientations[0][1] = 0

                    if self.__destination_orientations is not None and np.abs(start_vertical_orientation) < self.__max_pitch_angle\
                        and self.__local_view_count <= (self.__local_view_count_limit if not self.__continue_global_navigation else 4):
                        self.__local_view_count += 1
                        self.__local_path_executing = False
                        continue
                    
                    if np.abs(start_vertical_orientation) >= (self.__dataset_config.agent_tilt_angle - self.__local_adjust_pitch_epsilon):
                        cmd_vel_msg = Twist()
                        if start_vertical_orientation < 0:
                            cmd_vel_msg.angular.y = -1
                        elif start_vertical_orientation > 0:
                            cmd_vel_msg.angular.y = 1
                        else:
                            raise ValueError('Unknown vertical condition.')
                        self.__publish_cmd_vel(cmd_vel_msg)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        continue 
                    if self.__escape_flag != self.EscapeFlag.NONE:
                        rospy.logwarn('Cancel the escape plan because arrived.')
                        self.__escape_flag = self.EscapeFlag.NONE

                    if self.__local_set_mapper_flag == False:
                        set_mapper_request:SetMapperRequest = SetMapperRequest()
                        set_mapper_request.kf_every = self.kf_every_old
                        set_mapper_request.map_every = self.map_every_old
                        try:
                            set_mapper_response:SetMapperResponse = set_mapper(set_mapper_request)
                        except rospy.ServiceException as e:
                            rospy.logerr(f'Set mapper service call failed: {e}')
                            self.__global_state = GlobalState.QUIT
                            continue
                        self.__local_set_mapper_flag = True
                    
                    # NOTE: rotation view arrived
                    self.__position_arrived_flag = False
                    self.__local_path_executing = False
                    self.__local_view_count = 1
                    self.__check_agent_close_to_obstacle_flag = True
                    self.__rotation_observed_vertices = np.vstack([self.__rotation_observed_vertices, self.__pose_last['topdown_translation']])
                    if self.__continue_global_navigation:
                        self.__continue_global_navigation = False
                        self.__rotation_arrived_flag = False
                        self.__high_connectivity_node_view_count += 1
                        self.__position_selected_vertices = np.vstack([self.__position_selected_vertices, self.__pose_last['topdown_translation']])
                        Log("Continue global navigation.", tag='ActiveSplat')
                    else:
                        self.__rotation_arrived_flag = True
                        self.__high_connectivity_node_view_count = 0
                        self.__position_selected_vertices = np.vstack([self.__position_selected_vertices, self.__pose_last['topdown_translation']])
                        if self.__global_state == GlobalState.MANUAL_PLANNING:
                            Log("Manual planning arrived. Please click the target node in the Topdown Free Map and press 'Enter' to continue", tag='ActiveSplat')
                        else:
                            Log('Auto planning arrived.', tag='ActiveSplat')
                else:
                    self.__get_topdown()
                    # update whole navigation path
                    whole_navigation_path_2d = np.vstack([self.__pose_last['topdown_translation'], self.__navigation_path])
                    self.whole_navigation_path_3d = np.array([
                        c2w_topdown_to_world(vertex, self.__topdown_config, height_value=0).tolist()
                        for vertex in whole_navigation_path_2d])
                    
                    pixel_success = self.__pixel_as_arrived
                    
                    if (np.linalg.norm(self.__pose_last['topdown_translation'] - self.__navigation_path[-1]) < pixel_success):
                        if USE_ROTATION_SELECTION:
                            if self.__is_close_to_rotation_observed_region(self.__pose_last['topdown_translation']):
                                self.__rotation_arrived_flag = True
                                continue
                            Log("Arrived at the target position, use 'position_arrived'.", tag='ActiveSplat')
                            self.__position_arrived_flag = True
                            continue
                        else:
                            self.__rotation_arrived_flag = True
                            continue
                    navigation_point_index_start = 0
                    for navigation_point_index, navigation_point in enumerate(self.__navigation_path):
                        if np.linalg.norm(self.__pose_last['topdown_translation'] - navigation_point) <= self.__agent_step_size_pixel:
                            navigation_point_index_start = navigation_point_index + 1
                    self.__navigation_path = self.__navigation_path[navigation_point_index_start:]
                    
                    if self.__navigation_path_index is not None and len(self.__navigation_path_index) > 0:
                        if USE_HIGH_CONNECTIVITY:
                            # NOTE: Check if the agent is close to the high connectivity node
                            if 'high_connectivity_nodes_index' in self.__voronoi_graph.keys():
                                if self.__is_close_to_high_connectivity_nodes(self.__pose_last['topdown_translation']) and not self.__is_close_to_rotation_observed_region(self.__pose_last['topdown_translation']) and self.__high_connectivity_node_view_count < 3:
                                    Log("Passing through the high connectivity node, use 'position_arrived'.", tag='ActiveSplat')
                                    self.__position_arrived_flag = True
                                    self.__continue_global_navigation = True
                                    continue
                                
                    navigation_path = self.__navigation_path.copy()
                    pose_last = self.__pose_last['topdown_translation'].copy()
                    whole_path = np.vstack([pose_last, navigation_path]).reshape(-1, 2)
                    if len(whole_path) >= 2:
                        if whole_path.shape[0] < 20:
                            # NOTE: Check if the target is close to the obstacle
                            obstacle_distance_threshold = self.__agent_radius_pixel * 1.0
                            if self.__is_close_to_obstacle(self.__navigation_path[-1], obstacle_distance_threshold):
                                if USE_ROTATION_SELECTION:
                                    self.__rotation_arrived_flag = False
                                    self.__position_arrived_flag = True
                                    continue
                                else:
                                    self.__rotation_arrived_flag = True
                                    continue
                        whole_path_length = np.linalg.norm(np.diff(whole_path, axis=0), axis=1)
                        whole_path_accumulated_length = np.cumsum(whole_path_length)
                        whole_path_accumulated_length_condition = whole_path_accumulated_length <= self.__pixel_as_visited
                        if not np.any(whole_path_accumulated_length_condition):
                            whole_path = whole_path[:2]
                        elif np.all(whole_path_accumulated_length_condition):
                            pass
                        else:
                            whole_path = whole_path[:np.argmin(whole_path_accumulated_length_condition)]
                        free_space_pixels_num = cv2.countNonZero(self.__topdown_free_map)
                        line_test_result = cv2.polylines(
                            self.__topdown_free_map.copy(),
                            [np.int32(whole_path)],
                            False,
                            255,
                            1)
                        agent_mask = cv2.circle(
                            np.zeros_like(self.__topdown_free_map),
                            np.int32(pose_last),
                            int(np.ceil(self.__agent_radius_pixel)),
                            255,
                            -1)
                        line_test_result[agent_mask > 0] = self.__topdown_free_map[agent_mask > 0]
                        assert cv2.countNonZero(self.__topdown_free_map) == free_space_pixels_num, 'self.__topdown_free_map changed.'
                        if cv2.countNonZero(line_test_result) != free_space_pixels_num:
                            rospy.logwarn('Line test failed, crash if follow the routine.')
                            self.__rotation_arrived_flag = True
                            if self.__escape_flag != self.EscapeFlag.NONE:
                                rospy.logwarn('Cancel the escape plan because line test.')
                                self.__escape_flag = self.EscapeFlag.NONE
                            continue
                    if self.__escape_flag == self.EscapeFlag.NONE:
                        diff_vector = self.__navigation_path[0] - self.__pose_last['topdown_translation']
                        start_orientation = np.arctan2(self.__pose_last['topdown_rotation_vector'][1], self.__pose_last['topdown_rotation_vector'][0])
                        end_orientation = np.arctan2(diff_vector[1], diff_vector[0])
                        diff_orientation = (np.degrees(end_orientation - start_orientation) + 180) % 360 - 180
                        diff_translation = np.linalg.norm(diff_vector)
                        cmd_vel_msg = Twist()
                        if diff_orientation > self.__dataset_config.agent_turn_angle:
                            cmd_vel_msg.angular.z = -TURN
                        elif diff_orientation < -self.__dataset_config.agent_turn_angle:
                            cmd_vel_msg.angular.z = TURN
                        elif diff_translation > self.__agent_step_size_pixel:
                            cmd_vel_msg.linear.x = SPEED
                        else:
                            raise ValueError('Unknown condition.')
                        self.__publish_cmd_vel(cmd_vel_msg)
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                    elif self.__escape_flag == self.EscapeFlag.ESCAPE_ROTATION:
                        topdown_translation_np = self.__pose_last['topdown_translation'].copy()
                        if len(self.__inaccessible_database) == 0:
                            topdown_translation = tuple(topdown_translation_np.tolist())
                            self.__inaccessible_database.setdefault(topdown_translation, np.array([]).reshape(-1, 2))
                        else:
                            inaccessible_database_topdown_translation_array =\
                                np.array(list(self.__inaccessible_database.keys())).reshape(-1, 2)
                            assert np.issubdtype(inaccessible_database_topdown_translation_array.dtype, np.floating) or np.issubdtype(inaccessible_database_topdown_translation_array.dtype, np.integer), f"Invalid dtype: {inaccessible_database_topdown_translation_array.dtype}"
                            topdown_translation_to_inaccessible_database = np.linalg.norm(
                                topdown_translation - inaccessible_database_topdown_translation_array,
                                axis=1)
                            inaccessible_database_topdown_translation_array_condition =\
                                topdown_translation_to_inaccessible_database < self.__agent_step_size_pixel * 0.1
                            if np.any(inaccessible_database_topdown_translation_array_condition):
                                topdown_translation_np:np.ndarray = inaccessible_database_topdown_translation_array[
                                    np.argmin(topdown_translation_to_inaccessible_database)]
                                topdown_translation = tuple(topdown_translation_np.tolist())
                            else:
                                topdown_translation = tuple(topdown_translation_np.tolist())
                                self.__inaccessible_database.setdefault(topdown_translation, np.array([]).reshape(-1, 2))
                        assert topdown_translation in self.__inaccessible_database, f"Invalid topdown_translation: {topdown_translation}"
                        rotation_direction, translation_test_condition = get_escape_plan(
                            self.__topdown_free_map,
                            topdown_translation_np,
                            self.__pose_last['topdown_rotation_vector'],
                            self.__dataset_config.agent_turn_angle,
                            self.__agent_step_size_pixel,
                            self.__inaccessible_database[topdown_translation])
                        twist_rotation = Twist()
                        twist_rotation.angular.z = -rotation_direction
                        twist_translation = Twist()
                        twist_translation.linear.x = SPEED
                        for translation_success in translation_test_condition:
                            rospy.logwarn('Start escape rotation.')
                            pose_c2w_world = self.__pose_last['c2w_world'].copy()
                            self.__publish_cmd_vel(twist_rotation)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                            while (not is_pose_changed(
                                    pose_c2w_world,
                                    self.__pose_last['c2w_world'],
                                    self.__pose_update_translation_threshold,
                                    self.__pose_update_rotation_threshold) in [PoseChangeType.ROTATION, PoseChangeType.BOTH]) and self.__global_state != GlobalState.QUIT:
                                pose_c2w_world = self.__pose_last['c2w_world'].copy()
                                self.__publish_cmd_vel(twist_rotation)
                                self.__get_topdown()
                                with self.__update_map_cv2_condition:
                                    self.__update_map_cv2_condition.notify_all()
                            if not (self.__global_state in self.__ENABLE_STATES):
                                break
                            if translation_success:
                                rospy.logwarn('Start escape translation.')
                                self.__escape_flag = self.EscapeFlag.ESCAPE_TRANSLATION
                                while self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION and self.__global_state != GlobalState.QUIT:
                                    self.__publish_cmd_vel(twist_translation)
                                    self.__get_topdown()
                                    with self.__update_map_cv2_condition:
                                        self.__update_map_cv2_condition.notify_all()
                                if not (self.__global_state in self.__ENABLE_STATES):
                                    break
                                if self.__escape_flag == self.EscapeFlag.NONE:
                                    rospy.logwarn('Escape finished.')
                                    break
                                elif self.__escape_flag == self.EscapeFlag.ESCAPE_ROTATION:
                                    rospy.logwarn('Cancel the escape translation plan.')
                                    if len(self.__inaccessible_database[topdown_translation]) > 0:
                                        assert np.linalg.norm(self.__inaccessible_database[topdown_translation] - self.__pose_last['topdown_translation']) >= self.__agent_step_size_pixel * 0.1, f"Invalid inaccessible_database: {self.__inaccessible_database[topdown_translation]}"
                                    self.__inaccessible_database[topdown_translation] = np.vstack([
                                        self.__inaccessible_database[topdown_translation],
                                        self.__pose_last['topdown_rotation_vector']])
                                else:
                                    raise ValueError('Uprecise_voronoi_graphnknown escape flag.')
                        if not (self.__global_state in self.__ENABLE_STATES):
                            continue
                        if self.__escape_flag == self.EscapeFlag.NONE:
                            rospy.logwarn('Escape finished, now replan.')
                            if USE_ROTATION_SELECTION:
                                if not self.__is_close_to_rotation_observed_region(self.__pose_last['topdown_translation']):
                                    self.__position_arrived_flag = True
                                    self.__continue_global_navigation = True
                                else:
                                    self.__rotation_arrived_flag = True
                            else:
                                self.__rotation_arrived_flag = True
                        else:
                            # FIXME: Escape failed, it should not happen.
                            rospy.logerr('Escape failed, it should not happen.')
                    elif self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION:
                        # FIXME: Escape failed, it should not happen.
                        rospy.logerr('Invalid escape flag, it should not happen.')
                        self.__escape_flag = self.EscapeFlag.NONE
        self.__save_results()
        
    def __is_close_to_obstacle(self, topdown_position:np.ndarray, obstacle_distance_threshold:float) -> bool:
        proximity_mask = cv2.circle(
            np.zeros_like(self.__topdown_free_map),
            np.int32(topdown_position).tolist(),
            int(np.ceil(obstacle_distance_threshold)),
            255,
            -1
        )
        if cv2.countNonZero(proximity_mask & (self.__topdown_free_map == 0)) > 0:
            return True
        return False
    
    def __is_close_to_high_connectivity_nodes(self, pose) -> bool:
        agent_mask = cv2.circle(
            np.zeros_like(self.__voronoi_graph['obstacle_map']),
            np.int32(pose),
            1, # 1 pixel
            255,
            -1)
        high_connectivity_nodes_index = self.__voronoi_graph['high_connectivity_nodes_index']
        high_connectivity_nodes_mask = np.zeros_like(self.__topdown_free_map)
        for high_connectivity_node_index in high_connectivity_nodes_index:
            high_connectivity_node = self.__voronoi_graph['vertices'][high_connectivity_node_index]
            high_connectivity_nodes_mask = cv2.circle(
                high_connectivity_nodes_mask,
                np.int32(high_connectivity_node),
                1, # 1 pixel
                255,
                -1)
        # check if agent is close to the high connectivity nodes
        if cv2.countNonZero(agent_mask & high_connectivity_nodes_mask) > 0:
            return True
        return False
    
    def __is_close_to_rotation_observed_region(self, pose, radius_num = None) -> bool:
        # check if agent is close to the rotation observed area
        distance = self.__rotation_observed_vertices - pose
        if radius_num is None:
            radius_num = self.__radius_num_as_rotated
        if np.any(np.linalg.norm(distance, axis=1) < self.__agent_radius_pixel * radius_num):
            return True
        return False
    
    def __is_close_to_position_selected(self, pose) -> bool:
        distance = self.__position_selected_vertices - pose
        if np.any(np.linalg.norm(distance, axis=1) < self.__pixel_as_visited):
            return True
        return False
    
    def __is_close_to_arrived(self, pose) -> bool:
        distance = self.__position_selected_vertices - pose
        if np.any(np.linalg.norm(distance, axis=1) < self.__pixel_as_arrived):
            return True
        return False
                        
    def __setup_for_episode(self, init:bool=False) -> None:
        if not init:
            self.__save_results()
        if USE_RANDOM_SELECTION:
            self.__NODES_FLAGS_WEIGHT = None
        else:
            self.__NODES_FLAGS_WEIGHT = NODES_FLAGS_WEIGHT_INIT.copy()
        self.__visited_map = None
        self.__topdown_free_map_imshow = None
        self.__cluster_nodes_map = None
        self.__update_cluster_nodes_map_flag = False

        self.__dataset_config:GetDatasetConfigResponse = self.__get_dataset_config_service(GetDatasetConfigRequest())
        self.__results_dir = self.__dataset_config.results_dir
        os.makedirs(self.__results_dir, exist_ok=True)
        self.__save_topdown_map_count = 0
        if self.__save_runtime_data:
            self.__topdown_map_dir = os.path.join(self.__results_dir, 'topdown_map')
            os.makedirs(self.__topdown_map_dir, exist_ok=True)
            self.__subregion_map_dir = os.path.join(self.__results_dir, 'subregion_map')
            os.makedirs(self.__subregion_map_dir, exist_ok=True)
        
        self.__pose_data_type = PoseDataType(self.__dataset_config.pose_data_type)
        self.__height_direction = (self.__dataset_config.height_direction // 2, (self.__dataset_config.height_direction % 2) * 2 - 1)
        
        self.__interpolate_path_flag = False
        self.__controller_destination_flag = False
        
        topdown_config_response:GetTopdownConfigResponse = self.__get_topdown_config_service(GetTopdownConfigRequest())
        self.__topdown_config = {
            'world_dim_index': (
                topdown_config_response.topdown_x_world_dim_index,
                topdown_config_response.topdown_y_world_dim_index),
            'world_2d_bbox': (
                (topdown_config_response.topdown_x_world_lower_bound, topdown_config_response.topdown_x_world_upper_bound),
                (topdown_config_response.topdown_y_world_lower_bound, topdown_config_response.topdown_y_world_upper_bound)),
            'meter_per_pixel': topdown_config_response.meter_per_pixel,
            'grid_map_shape': (
                topdown_config_response.topdown_x_length,
                topdown_config_response.topdown_y_length)}
        self.__topdown_image_shape = (topdown_config_response.topdown_y_length, topdown_config_response.topdown_x_length)
        self.__agent_radius_pixel = self.__dataset_config.agent_radius / self.__topdown_config['meter_per_pixel']
        self.__agent_step_size_pixel = self.__dataset_config.agent_forward_step_size / self.__topdown_config['meter_per_pixel']
        self.__pixel_as_visited = self.__agent_step_size_pixel * self.__step_num_as_visited
        self.__pixel_as_arrived = self.__agent_step_size_pixel * self.__step_num_as_arrived

        self.__inaccessible_database:Dict[Tuple[float, float], np.ndarray] = dict()
        
        self.__pose_last:Dict[str, np.ndarray] = {
            'c2w_world': None,
            'topdown_rotation_vector': None,
            'topdown_translation': None}
        self.__topdown_translation_array = np.array([]).reshape(-1, 2)
        self.__fail_vertices_nodes = np.array([]).reshape(-1, 2)
        
        self.__movement_fail_times = 0
        self.__rotation_arrived_flag = True
        self.__position_arrived_flag = False
        self.__escape_flag = self.EscapeFlag.NONE
        self.__check_agent_close_to_obstacle_flag = True
        self.__rotation_observed_vertices = np.array([]).reshape(-1, 2)
        self.__continue_global_navigation = False
        self.__high_connectivity_node_view_count = 0
        
        self.__navigation_path:np.ndarray = None
        self.__navigation_path_index:np.ndarray = None
        self.__destination_orientations:np.ndarray = None
        self.__voronoi_graph:Dict[str, Union[nx.Graph, np.ndarray, cv2.Mat, List[List[np.ndarray]], Dict[int, List[int]]]] = None
        self.__topdown_free_map:np.ndarray = None
        self.__topdown_visible_map:np.ndarray = None
        self.__voronoi_graph_cv2:cv2.Mat = None
        self.__topdown_free_map_cv2:cv2.Mat = None
        self.__topdown_visible_map_cv2:cv2.Mat = None
        self.__horizon_bbox:np.ndarray = None
        self.__horizon_bbox_last_translation:np.ndarray = None
        self.__global_invisibility_info:Dict[int, Frustum] = dict()
        self.__local_invisibility_info:Dict[int, Frustum] = dict()
        self.__local_path_executing = False
        self.__local_set_mapper_flag = True
        self.__local_adjust_pitch_epsilon = 1e-5
        self.closest_node_index = -1
        self.kf_every_old = 0
        self.map_every_old = 0
        self.whole_navigation_path_3d = None
        self.voronoi_graph_3d_points = None
        self.voronoi_graph_3d_lines = None
        self.nodes_position_3d = None
        self.high_connectivity_nodes_3d = None
        self.is_voronoi_graph_ready = False
        self.__position_selected_vertices = np.array([]).reshape(-1, 2)
        self.use_global_plan_flag = False
        
        self.__last_twist = Twist()
        
        self.__bootstrap_flag = self.__global_state in self.__ENABLE_STATES if init else True
        with self.__update_map_cv2_condition:
            self.__update_map_cv2_condition.notify_all()
                        
    def __get_topdown(self) -> None:
        try:
            get_topdown_response:GetTopdownResponse = self.__get_topdown_service(GetTopdownRequest(self.__rotation_arrived_flag))
        except rospy.ServiceException as e:
            rospy.logerr(f'Get topdown service call failed: {e}')
            self.__global_state = GlobalState.QUIT
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
            return
        pose_last = self.__pose_last['topdown_translation'].copy()
        topdown_free_map_raw = np.array(get_topdown_response.free_map).reshape(self.__topdown_image_shape).astype(np.uint8) * 255
        topdown_visible_map = np.array(get_topdown_response.visible_map).reshape(self.__topdown_image_shape).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        self.__topdown_free_map, local_obstacle_map_approx_contour, obstacle_map_children_approx_contours = get_obstacle_map(
            topdown_free_map_raw,
            topdown_visible_map,
            pose_last,
            kernel,
            0.225 / self.__topdown_config['meter_per_pixel'])
        self.__topdown_visible_map = topdown_visible_map.copy()
        self.__topdown_free_map_cv2 = cv2.cvtColor(self.__topdown_free_map, cv2.COLOR_GRAY2BGR)
        self.__topdown_visible_map_cv2 = cv2.cvtColor(self.__topdown_visible_map, cv2.COLOR_GRAY2BGR)
        if self.__rotation_arrived_flag:
            self.__horizon_bbox = get_horizon_bound_topdown(
                np.array([
                    get_topdown_response.horizon_bound_min.x,
                    get_topdown_response.horizon_bound_min.y,
                    get_topdown_response.horizon_bound_min.z]),
                np.array([
                    get_topdown_response.horizon_bound_max.x,
                    get_topdown_response.horizon_bound_max.y,
                    get_topdown_response.horizon_bound_max.z]),
                self.__topdown_config,
                self.__height_direction)
            
        if self.__last_twist.linear.x > 0 and self.__last_twist.angular.z == 0:
            self.__horizon_bbox_last_translation = deepcopy(self.__horizon_bbox)
            
        if self.__rotation_arrived_flag or self.__voronoi_graph is None or self.__voronoi_graph_cv2 is None:
            inaccessible_points = np.array([]).reshape(-1, 2)
            self.__voronoi_graph = get_voronoi_graph(
                obstacle_map=self.__topdown_free_map,
                local_obstacle_map_approx_contour=local_obstacle_map_approx_contour,
                obstacle_map_children_approx_contours=obstacle_map_children_approx_contours,
                edge_sample_num=5,
                agent_radius_pixel=self.__agent_radius_pixel,
                inaccessible_points=inaccessible_points)
            if self.__rotation_arrived_flag == True:
                # get subregions
                self.__voronoi_graph['subregions'], self.__voronoi_graph['clusters_cv2'] = get_subregions(self.__voronoi_graph['graph'], 
                                                        self.__voronoi_graph['nodes_index'], 
                                                        self.__voronoi_graph['vertices'], 
                                                        self.__topdown_config['meter_per_pixel'], 
                                                        image_shape=self.__topdown_free_map.shape,
                                                        save_runtime_data=self.__save_runtime_data)
                self.__update_cluster_nodes_map_flag = self.__voronoi_graph['clusters_cv2'] is not None
            self.__voronoi_graph['nodes_score'] = np.ones_like(
                self.__voronoi_graph['nodes_index']) * self.__voronoi_graph_nodes_score_max
            
            # NOTE: Opacity invisibility
            if self.__rotation_arrived_flag:
                node_vertex_c2w_worlds = []
                nodes_id = self.__voronoi_graph['nodes_index'].copy()
                for node_index in self.__voronoi_graph['nodes_index']:
                    obstacle_distance_threshold = self.__agent_radius_pixel * 2.0
                    if node_index in self.__fail_vertices_nodes_index:
                        node_vertex_c2w_worlds.append(Point())
                        rospy.logdebug(f'Fail vertices: {node_index}, skip.')
                        continue
                    elif self.__is_close_to_obstacle(self.__voronoi_graph['vertices'][node_index], obstacle_distance_threshold):
                        self.__fail_vertices_nodes_index.append(node_index)
                        node_vertex_c2w_worlds.append(Point())
                        rospy.logdebug(f'The {node_index} node is close to the obstacle, skip.')
                        continue
                    elif self.__is_close_to_rotation_observed_region(self.__voronoi_graph['vertices'][node_index], radius_num=1.0):
                        node_vertex_c2w_worlds.append(Point())
                        rospy.logdebug(f'The {node_index} node is close to the rotation observed region, skip.')
                        continue
                    node_vertex = self.__voronoi_graph['vertices'][node_index]
                    node_vertex_c2w_world = c2w_topdown_to_world(
                        node_vertex,
                        self.__topdown_config,
                        height_value=0)
                    p = Point()
                    p.x = node_vertex_c2w_world[0]
                    p.y = node_vertex_c2w_world[1]
                    p.z = node_vertex_c2w_world[2]
                    node_vertex_c2w_worlds.append(p)
                try:
                    get_opacity_response:GetOpacityResponse = self.__get_opacity_service(GetOpacityRequest(self.__rotation_arrived_flag, node_vertex_c2w_worlds, nodes_id))
                except rospy.ServiceException as e:
                    rospy.logerr(f'Get global opacity service call failed: {e}')
                    self.__global_state = GlobalState.QUIT
                    with self.__global_state_condition:
                        self.__global_state_condition.notify()
                    return
                
                self.__global_invisibility_info = dict()
                cur_max_invisibility = np.nanmax(get_opacity_response.targets_frustums_invisibility)
                cur_max_volume = np.nanmax(get_opacity_response.targets_frustums_volume)
                for idx, target_frustum_invisibility in enumerate(get_opacity_response.targets_frustums_invisibility):
                    self.__global_invisibility_info[self.__voronoi_graph['nodes_index'][idx]] = Frustum()
                    self.__global_invisibility_info[self.__voronoi_graph['nodes_index'][idx]].invisibility_score = target_frustum_invisibility
                    self.__global_invisibility_info[self.__voronoi_graph['nodes_index'][idx]].hole_valume = get_opacity_response.targets_frustums_volume[idx]
                
            if self.__rotation_arrived_flag:
                nodes_vertices = self.__voronoi_graph['vertices'][self.__voronoi_graph['nodes_index']]
                nodes_flags:Dict[NodesFlagsType, np.ndarray] = dict()
                
                if len(self.__topdown_translation_array) > 0:
                    nodes_to_translation_array = cdist(
                        nodes_vertices,
                        self.__topdown_translation_array)
                    nodes_to_translation_array = np.min(nodes_to_translation_array, axis=1)
                else:
                    nodes_to_translation_array = np.ones_like(self.__voronoi_graph['nodes_index']) * np.inf
                nodes_flags[NodesFlagsType.UNARRIVED] = np.int32(nodes_to_translation_array > self.__pixel_as_visited)
                
                if len(self.__fail_vertices_nodes) > 0:
                    nodes_to_fail_vertices = cdist(
                        nodes_vertices,
                        self.__fail_vertices_nodes)
                    nodes_to_fail_vertices = np.min(nodes_to_fail_vertices, axis=1)
                else:
                    nodes_to_fail_vertices = np.ones_like(self.__voronoi_graph['nodes_index']) * np.inf
                nodes_flags[NodesFlagsType.FAIL] = np.int32(nodes_to_fail_vertices <= self.__agent_radius_pixel)
                
                if self.__NODES_FLAGS_WEIGHT is not None and np.all(np.logical_or(
                    np.logical_not(nodes_flags[NodesFlagsType.UNARRIVED]),
                    nodes_flags[NodesFlagsType.FAIL])):
                    self.__fail_vertices_nodes = np.array([]).reshape(-1, 2)
                    nodes_flags[NodesFlagsType.FAIL] = np.zeros_like(self.__voronoi_graph['nodes_index'])
                    self.__NODES_FLAGS_WEIGHT[NodesFlagsType.OPACITY_INVISIBILITY] = 10
                    self.__NODES_FLAGS_WEIGHT[NodesFlagsType.HOLE_INVISIBILITY] = 10
                    self.__NODES_FLAGS_WEIGHT[NodesFlagsType.IN_HORIZON] = -1
                    self.__voronoi_graph_nodes_score_max = 0
                    self.__voronoi_graph_nodes_score_min = 0
                    for value in self.__NODES_FLAGS_WEIGHT.values():
                        if value > 0:
                            self.__voronoi_graph_nodes_score_max += value
                        elif value < 0:
                            self.__voronoi_graph_nodes_score_min += value
                
                free_space_pixels_num = cv2.countNonZero(self.__topdown_free_map)
                agent_mask = cv2.circle(
                    np.zeros_like(self.__topdown_free_map),
                    np.int32(pose_last),
                    int(np.ceil(self.__agent_radius_pixel)),
                    255,
                    -1)
                line_test_results = []
                for node_vertice in nodes_vertices:
                    line_test_result = cv2.line(
                        self.__topdown_free_map.copy(),
                        np.int32(pose_last),
                        np.int32(node_vertice),
                        255,
                        1)
                    line_test_result[agent_mask > 0] = self.__topdown_free_map[agent_mask > 0]
                    line_test_results.append(cv2.countNonZero(line_test_result) == free_space_pixels_num)
                
                if self.__horizon_bbox_last_translation is not None:
                    in_horizon_bbox_condition = np.int32(
                        np.logical_and(
                            np.logical_and(
                                nodes_vertices[:, 0] >= self.__horizon_bbox_last_translation[0, 0],
                                nodes_vertices[:, 0] <= self.__horizon_bbox_last_translation[1, 0]),
                            np.logical_and(
                                nodes_vertices[:, 1] >= self.__horizon_bbox_last_translation[0, 1],
                                nodes_vertices[:, 1] <= self.__horizon_bbox_last_translation[1, 1])))
                    
                    line_test_results_new = np.logical_and(
                        in_horizon_bbox_condition,
                        line_test_results)
                    if np.any(line_test_results_new):
                        line_test_results = line_test_results_new
                    
                nodes_flags[NodesFlagsType.IN_HORIZON] = np.int32(line_test_results)
                real_invisibility = np.array([self.__global_invisibility_info[node_index].invisibility_score for node_index in self.__voronoi_graph['nodes_index']])
                norm_invisibility = np.ceil(real_invisibility / cur_max_invisibility * 10).astype(np.int32)
                norm_volume = np.ceil(np.array([self.__global_invisibility_info[node_index].hole_valume for node_index in self.__voronoi_graph['nodes_index']]) / cur_max_volume * 10).astype(np.int32)
                nodes_flags[NodesFlagsType.OPACITY_INVISIBILITY] = norm_invisibility
                nodes_flags[NodesFlagsType.HOLE_INVISIBILITY] = norm_volume
                nodes_flags[NodesFlagsType.REAL_OPACITY_INVISIBILITY] = np.ceil(real_invisibility).astype(np.int32)
                
                self.__voronoi_graph['nodes_score'] = np.zeros_like(self.__voronoi_graph['nodes_index'])
                self.__voronoi_graph['nodes_invisibility_score'] = np.zeros_like(self.__voronoi_graph['nodes_index'])
                if self.__NODES_FLAGS_WEIGHT is not None:
                    for key, value in nodes_flags.items():
                        if key == NodesFlagsType.REAL_OPACITY_INVISIBILITY:
                            self.__voronoi_graph['nodes_invisibility_score'] += self.__NODES_FLAGS_WEIGHT[key] * value
                        else:
                            self.__voronoi_graph['nodes_score'] += self.__NODES_FLAGS_WEIGHT[key] * value
                
                # NOTE: Voronoi graph 3D for visualization
                self.voronoi_graph_3d_points, self.voronoi_graph_3d_lines = self.__get_voronoi_graph_3d(
                    nx.to_numpy_array(self.__voronoi_graph['graph']),
                    self.__voronoi_graph['vertices'],
                    local_obstacle_map_approx_contour)
                self.nodes_position_3d = np.array([
                    c2w_topdown_to_world(
                        self.__voronoi_graph['vertices'][node_index],
                        self.__topdown_config,
                        height_value=0)
                    for node_index in self.__voronoi_graph['nodes_index']])
    
                self.high_connectivity_nodes_3d = np.array([
                    c2w_topdown_to_world(vertex, self.__topdown_config, height_value=0).tolist()
                    for vertex in self.__voronoi_graph['vertices'][self.__voronoi_graph['high_connectivity_nodes_index']]
                ])
                self.is_voronoi_graph_ready = True
                
                self.__trigger_update_high_connectivity_nodes_pub.publish(Bool(True))
                self.__trigger_update_voronoi_graph_pub.publish(Bool(True))
                    
            self.__voronoi_graph_cv2 = draw_voronoi_graph(
                background=np.zeros_like(self.__topdown_free_map),
                voronoi_graph_vertices=self.__voronoi_graph['vertices'],
                voronoi_graph_ridge_matrix=nx.to_numpy_array(self.__voronoi_graph['graph']),
                voronoi_graph_nodes_index=self.__voronoi_graph['nodes_index'],
                pruned_chains=self.__voronoi_graph['pruned_chains'],
                voronoi_graph_nodes_score=self.__voronoi_graph['nodes_score'],
                voronoi_graph_nodes_score_max=self.__voronoi_graph_nodes_score_max,
                voronoi_graph_nodes_score_min=self.__voronoi_graph_nodes_score_min,
                global_invisibility_info=self.__global_invisibility_info,
                voronoi_graph_ridge_color=[255, 0, 0],
                voronoi_graph_ridge_thickness=3,
                voronoi_graph_nodes_colormap=self.__voronoi_graph_nodes_colormap,
                voronoi_graph_nodes_radius=5,
                pruned_chains_color=[0, 255, 0],
                pruned_chains_thickness=2)
        if self.__voronoi_graph_cv2 is not None:
            voronoi_graph_cv2_mask = cv2.cvtColor(self.__voronoi_graph_cv2, cv2.COLOR_BGR2GRAY)
            self.__topdown_free_map_cv2[voronoi_graph_cv2_mask > 0] = self.__voronoi_graph_cv2[voronoi_graph_cv2_mask > 0]
            
    def __get_voronoi_graph_3d(
        self,
        voronoi_graph_ridge_matrix,
        voronoi_graph_vertices,
        local_obstacle_map_approx_contour: np.ndarray
    ) -> np.ndarray:
        points = []
        lines = []

        # Convert 2D vertices to 3D
        voronoi_graph_vertices_3d = [
            c2w_topdown_to_world(vertex, self.__topdown_config, height_value=0).tolist()
            for vertex in voronoi_graph_vertices
        ]

        for start_id, row in enumerate(voronoi_graph_ridge_matrix):
            start = voronoi_graph_vertices[start_id]
            start_x, start_y = int(start[0]), int(start[1])
            start_point = (start_x, start_y)
            points.append(voronoi_graph_vertices_3d[start_id])

            for step_id, value in enumerate(row[start_id + 1:]):
                if value > 0:
                    end_id = start_id + step_id + 1
                    end = voronoi_graph_vertices[end_id]
                    end_x, end_y = int(end[0]), int(end[1])
                    end_point = (end_x, end_y)
                    # Check if both start and end points are inside the local obstacle map contour
                    if cv2.pointPolygonTest(local_obstacle_map_approx_contour, start_point, False) > 0 and cv2.pointPolygonTest(local_obstacle_map_approx_contour, end_point, False) > 0:
                        lines.append([start_id, end_id])

        points = np.array(points, dtype=np.float32).reshape(-1, 3)
        lines = np.array(lines, dtype=np.int32).reshape(-1, 2)
        return points, lines

        
    def __update_map_cv2(self) -> None:
        
        def mouse_callback(event:int, x:int, y:int, flags:int, param:int) -> None:
            if event == cv2.EVENT_LBUTTONDBLCLK:
                rospy.logdebug(f'Left button double clicked at: ({x}, {y})')
                if self.__global_state == GlobalState.MANUAL_PLANNING and\
                    self.__rotation_arrived_flag and\
                        self.__voronoi_graph is not None:
                    vertices_nodes = self.__voronoi_graph['vertices'][self.__voronoi_graph['nodes_index']]
                    vertices_to_click_distance = np.linalg.norm(
                        vertices_nodes - np.array([x, y]),
                        axis=1)
                    if np.min(vertices_to_click_distance) > 20:
                        return
                    vertex_destination_index = self.__voronoi_graph['nodes_index'][np.argmin(vertices_to_click_distance)]
                    closest_vertex_index = get_closest_vertex_index(
                        self.__voronoi_graph['vertices'],
                        self.__voronoi_graph['obstacle_map'],
                        self.__pose_last['topdown_translation'],
                        self.__agent_radius_pixel)
                    navigation_path_index, navigation_path, graph_search_success = get_safe_dijkstra_path(
                        self.__voronoi_graph['graph'],
                        closest_vertex_index,
                        vertex_destination_index,
                        self.__voronoi_graph['vertices'],
                        self.__voronoi_graph['obstacle_map'],
                        self.__pose_last['topdown_translation'],
                        self.__agent_radius_pixel)
                    if not graph_search_success:
                        self.__fail_vertices_nodes_index.append(vertex_destination_index)
                        self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__voronoi_graph['vertices'][vertex_destination_index]])
                    if navigation_path_index is None or navigation_path is None:
                        rospy.logwarn('No path found.')
                        self.__destination_orientations = None
                        return
                    else:
                        self.__navigation_path = navigation_path
                        self.__interpolate_path()
                        self.__navigation_path_index = navigation_path_index
                        return
        
        while not rospy.is_shutdown() and self.__global_state != GlobalState.QUIT:
            if self.__global_state not in self.__ENABLE_STATES:
                cv2.destroyAllWindows()
                for window_name in self.__cv2_windows_with_callback_opened.keys():
                    self.__cv2_windows_with_callback_opened[window_name] = False
                with self.__global_state_condition:
                    self.__global_state_condition.wait()
                continue
            wait_for_manual_planning_flag = self.__rotation_arrived_flag and self.__global_state == GlobalState.MANUAL_PLANNING
            if not wait_for_manual_planning_flag:
                with self.__update_map_cv2_condition:
                    self.__update_map_cv2_condition.wait()
            cv2_imshow_flag = False
            if self.__topdown_free_map_cv2 is not None:
                if (not self.__cv2_windows_with_callback_opened['topdown_free_map']) and (not self.__hide_windows):
                    cv2.namedWindow('Topdown Free Map', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Topdown Free Map', width=self.__topdown_config['grid_map_shape'][0], height=self.__topdown_config['grid_map_shape'][1])
                    cv2.moveWindow('Topdown Free Map', 0, 1200)
                    cv2.namedWindow('Topdown Free Map Visited', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Topdown Free Map Visited', width=self.__topdown_config['grid_map_shape'][0], height=self.__topdown_config['grid_map_shape'][1])
                    cv2.moveWindow('Topdown Free Map Visited', 450, 1200)
                    cv2.namedWindow('Topdown Occupied Areas', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Topdown Occupied Areas', width=self.__topdown_config['grid_map_shape'][0], height=self.__topdown_config['grid_map_shape'][1])
                    cv2.moveWindow('Topdown Occupied Areas', 800, 1200)
                    self.__cv2_windows_with_callback_opened['topdown_free_map'] = True
                    cv2.setMouseCallback('Topdown Free Map', mouse_callback)
                topdown_free_map = self.__topdown_free_map_cv2.copy()
                navigation_path = deepcopy(self.__navigation_path)
                if navigation_path is not None and len(navigation_path) > 0:
                    navigation_path_cv2 = np.zeros_like(topdown_free_map)
                    cv2.polylines(
                        navigation_path_cv2,
                        [np.int32(np.vstack([self.__pose_last['topdown_translation'], navigation_path]))],
                        False,
                        (0, 64, 255),
                        6)
                    cv2.circle(
                        navigation_path_cv2,
                        np.int32(navigation_path[-1]),
                        8,
                        (0, 128, 255),
                        -1)
                    navigation_path_cv2_mask = cv2.cvtColor(navigation_path_cv2, cv2.COLOR_BGR2GRAY)
                    voronoi_graph_cv2_mask = cv2.cvtColor(self.__voronoi_graph_cv2, cv2.COLOR_BGR2GRAY)
                    navigation_path_cv2_condition = np.logical_and(
                        navigation_path_cv2_mask > 0,
                        voronoi_graph_cv2_mask == 0)
                    topdown_free_map[navigation_path_cv2_condition] = navigation_path_cv2[navigation_path_cv2_condition]
                if self.closest_node_index != -1:
                    if self.closest_node_index < len(self.__voronoi_graph['vertices']):
                        cv2.circle(
                            topdown_free_map,
                            np.int32(self.__voronoi_graph['vertices'][self.closest_node_index]),
                            int(np.ceil(self.__agent_radius_pixel)),
                            (0, 255, 255), # yellow
                            -1)
                if self.__pose_last['topdown_translation'] is not None:
                    if self.__pose_last['topdown_rotation_vector'] is not None:
                        cv2.arrowedLine(
                            topdown_free_map,
                            self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                            (self.__pose_last['topdown_translation'] + self.__pose_last['topdown_rotation_vector'] * 10).astype(np.int32).tolist(),
                            (0, 255, 0),
                            2)
                    cv2.circle(
                        topdown_free_map,
                        self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                        int(np.ceil(self.__agent_radius_pixel)),
                        (128, 255, 128),
                        -1)
                self.__topdown_free_map_imshow = topdown_free_map.copy()
                if not self.__hide_windows:
                    cv2.imshow('Topdown Free Map', self.__topdown_free_map_imshow)
                    cv2_imshow_flag = True
                if len(self.__topdown_translation_array) > 1:
                    visited_map = np.ones_like(self.__topdown_free_map_cv2) * 255
                    cv2.polylines(
                        visited_map,
                        [np.int32(self.__topdown_translation_array)],
                        False,
                        (0, 255, 0),
                        int(np.ceil(2 * self.__pixel_as_visited)))
                    visited_map[self.__topdown_free_map == 0] = [0, 0, 0]
                    if self.__horizon_bbox is not None:
                        cv2.rectangle(
                            visited_map,
                            np.int32(self.__horizon_bbox[0]),
                            np.int32(self.__horizon_bbox[1]),
                            (255, 0, 0),
                            1)
                    if self.__horizon_bbox_last_translation is not None:
                        cv2.rectangle(
                            visited_map,
                            np.int32(self.__horizon_bbox_last_translation[0]),
                            np.int32(self.__horizon_bbox_last_translation[1]),
                            (255, 0, 255),
                            1)
                    for fail_vertex in self.__fail_vertices_nodes:
                        cv2.circle(
                            visited_map,
                            np.int32(fail_vertex),
                            int(np.ceil(self.__agent_radius_pixel)),
                            (0, 0, 255),
                            -1)
                    if 'high_connectivity_nodes_index' in self.__voronoi_graph.keys():
                        for high_connectivity_node_index in self.__voronoi_graph['high_connectivity_nodes_index']:
                            overlay = visited_map.copy()
                            cv2.circle(
                                overlay,
                                np.int32(self.__voronoi_graph['vertices'][high_connectivity_node_index]),
                                int(np.ceil(self.__agent_radius_pixel)),
                                (255, 0, 255),
                                -1)
                            cv2.addWeighted(overlay, 0.5, visited_map, 0.5, 0, visited_map)
                    if len(self.__rotation_observed_vertices) > 0:
                        for rotation_observed_vertex in self.__rotation_observed_vertices:
                            overlay = visited_map.copy()
                            cv2.circle(
                                overlay,
                                np.int32(rotation_observed_vertex),
                                int(np.ceil(self.__agent_radius_pixel * self.__radius_num_as_rotated)),
                                (0, 255, 255),
                                -1)
                            cv2.addWeighted(overlay, 0.5, visited_map, 0.5, 0, visited_map)
                    self.__visited_map = visited_map.copy()
                    if self.__save_runtime_data and not self.__rotation_arrived_flag:
                        topdown_map = np.hstack([self.__topdown_free_map_imshow, self.__visited_map])
                        cv2.imwrite(os.path.join(self.__topdown_map_dir, f'{self.__save_topdown_map_count}.png'), topdown_map)
                        self.__save_topdown_map_count += 1
                    if not self.__hide_windows:
                        cv2.imshow('Topdown Free Map Visited', self.__visited_map)
            if self.__topdown_visible_map_cv2 is not None:
                topdown_visible_map = self.__topdown_visible_map_cv2.copy()
                if self.__pose_last['topdown_translation'] is not None:
                    if self.__pose_last['topdown_rotation_vector'] is not None:
                        cv2.arrowedLine(
                            topdown_visible_map,
                            self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                            (self.__pose_last['topdown_translation'] + self.__pose_last['topdown_rotation_vector'] * 10).astype(np.int32).tolist(),
                            (0, 255, 0),
                            2)
                    cv2.circle(
                        topdown_visible_map,
                        self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                        int(np.ceil(self.__agent_radius_pixel)),
                        (128, 255, 128),
                        -1)
                if not self.__hide_windows:
                    cv2.imshow('Topdown Occupied Areas', topdown_visible_map)
                    cv2_imshow_flag = True
            if self.__save_runtime_data and self.__update_cluster_nodes_map_flag:
                self.__update_cluster_nodes_map_flag = False
                self.__cluster_nodes_map = self.__voronoi_graph['clusters_cv2'].copy()
                if self.__save_topdown_map_count > 1:
                    cv2.imwrite(os.path.join(self.__subregion_map_dir, f'{self.__save_topdown_map_count - 1}.png'), self.__cluster_nodes_map)
            if cv2_imshow_flag:
                key = cv2.waitKey(1)
                if wait_for_manual_planning_flag:
                    if key == 13:
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                            
    def __set_planner_state(self, request:SetPlannerStateRequest) -> SetPlannerStateResponse:
        rospy.loginfo(f'Set planner state: {request.global_state}')
        if self.__global_state is None:
            self.__global_state = GlobalState(request.global_state)
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
        else:
            global_state_old = GlobalState(self.__global_state)
            self.__global_state = GlobalState(request.global_state)
            if (self.__global_state in self.__ENABLE_STATES) and (global_state_old not in self.__ENABLE_STATES):
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
                if global_state_old in [GlobalState.MANUAL_PLANNING]:
                    with self.__update_map_cv2_condition:
                        self.__update_map_cv2_condition.notify_all()
            if self.__global_state == GlobalState.QUIT:
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
        return SetPlannerStateResponse()
    
    def __get_voronoi_graph_callback(self, request:GetVoronoiGraphRequest) -> GetVoronoiGraphResponse:
        voronoi_graph = GetVoronoiGraphResponse()
        if self.__voronoi_graph is not None and self.is_voronoi_graph_ready:
            voronoi_graph.voronoi_graph_3d_points = self.voronoi_graph_3d_points.flatten().tolist()
            voronoi_graph.voronoi_graph_3d_lines = self.voronoi_graph_3d_lines.flatten().tolist()
            voronoi_graph.nodes_position_3d = self.nodes_position_3d.flatten().tolist()
            voronoi_graph.high_connectivity_nodes_3d = self.high_connectivity_nodes_3d.flatten().tolist()
            voronoi_graph.nodes_score = self.__voronoi_graph['nodes_score'].tolist()
        return voronoi_graph
    
    def __get_navigation_path_callback(self, request:GetNavPathRequest) -> GetNavPathResponse:
        navigation_path = GetNavPathResponse()
        if self.whole_navigation_path_3d is not None:
            navigation_path.whole_navigation_path = self.whole_navigation_path_3d.flatten().tolist()
        else:
            navigation_path.whole_navigation_path = []
        return navigation_path
    
    def __get_high_loss_samples_pose(self, target_frustum:Pose) -> None:
        target_frustum_quaternion = np.array([
            target_frustum.orientation.w,
            target_frustum.orientation.x,
            target_frustum.orientation.y,
            target_frustum.orientation.z])
        target_frustum_c2w_world = np.eye(4)
        target_frustum_c2w_world[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(target_frustum_quaternion))
        target_frustum_c2w_world[:3, 3] = np.array([
            target_frustum.position.x,
            target_frustum.position.y,
            target_frustum.position.z])
        target_frustum_rotation_vector, target_frustum_translation, pitch_angle = c2w_world_to_topdown(
            target_frustum_c2w_world,
            self.__topdown_config,
            self.__height_direction,
            np.float64,
            need_pitch=True)
        frustum = Frustum()
        frustum.rotation_vector_2d = target_frustum_rotation_vector
        frustum.translation_2d = target_frustum_translation
        frustum.pitch_angle = pitch_angle

        if self.__local_invisibility_info is not None:
            self.__local_invisibility_info[1] = frustum
    
    def __camera_pose_callback(self, pose:PoseStamped) -> None:
        pose_translation = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        pose_quaternion = np.array([
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z])
        pose_quaternion = quaternion.from_float_array(pose_quaternion)
        pose_rotation = quaternion.as_rotation_matrix(pose_quaternion)
        pose_c2w_world = np.eye(4)
        pose_c2w_world[:3, :3] = pose_rotation
        pose_c2w_world[:3, 3] = pose_translation
        
        pose_c2w_world = convert_to_c2w_opencv(pose_c2w_world, self.__pose_data_type)
        
        if self.__pose_last['c2w_world'] is None:
            pass
        elif is_pose_changed(
                self.__pose_last['c2w_world'],
                pose_c2w_world,
                self.__pose_update_translation_threshold,
                self.__pose_update_rotation_threshold) == PoseChangeType.NONE:
            return
        
        pose_rotation_vector = np.degrees(quaternion.as_rotation_vector(pose_quaternion))
        rospy.loginfo(f'Agent:\n\tX: {pose_translation[0]:.2f}, Y: {pose_translation[1]:.2f}, Z: {pose_translation[2]:.2f}\n\tX_angle: {pose_rotation_vector[0]:.2f}, Y_angle: {pose_rotation_vector[1]:.2f}, Z_angle: {pose_rotation_vector[2]:.2f}')
        
        pose_topdown_rotation_vector, pose_topdown_translation, pitch_angle = c2w_world_to_topdown(
            pose_c2w_world,
            self.__topdown_config,
            self.__height_direction,
            np.float64,
            self.__dataset_config.rgbd_position.z,
            need_pitch=True)
        
        # NOTE: compute the closest vertex index and node index
        closest_vertex_index = -1
        self.closest_node_index = -1
        if self.__voronoi_graph is not None:
            closest_vertex_index = get_closest_vertex_index(
                self.__voronoi_graph['vertices'],
                self.__voronoi_graph['obstacle_map'],
                pose_topdown_translation,
                self.__agent_radius_pixel)
            self.closest_node_index = get_closest_node_index(
                self.__voronoi_graph['vertices'],
                self.__voronoi_graph['nodes_index'],
                self.__pose_last['topdown_translation'])
        
        pose_current = {
            'c2w_world': pose_c2w_world,
            'topdown_rotation_vector': pose_topdown_rotation_vector,
            'topdown_translation': pose_topdown_translation,
            'pitch_angle': -pitch_angle,
            'closest_vertex_index': closest_vertex_index,
            'closest_node_index': self.closest_node_index}
        
        self.__pose_last = pose_current.copy()
        self.__topdown_translation_array = np.vstack([self.__topdown_translation_array, pose_topdown_translation])
        
        if self.__global_state in self.__ENABLE_STATES:
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
        return
    
    def __movement_fail_times_callback(self, movement_fail_times:Int32) -> None:
        if movement_fail_times.data > self.__movement_fail_times and not self.__rotation_arrived_flag:
            rospy.logwarn(f'Movement fail times: {self.__movement_fail_times}')
            self.__movement_fail_times = movement_fail_times.data
            if self.__escape_flag == self.EscapeFlag.NONE:
                self.__escape_flag = self.EscapeFlag.ESCAPE_ROTATION
                rospy.logwarn('Start escaping.')
                if self.__navigation_path is not None:
                    if len(self.__navigation_path) > 0 and len(self.__navigation_path) < 100:
                        self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__navigation_path[-1]])
            elif self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION:
                self.__escape_flag = self.EscapeFlag.ESCAPE_ROTATION
                rospy.logwarn('Escape failed.')
        elif movement_fail_times.data == 0 and self.__movement_fail_times > 0:
            self.__movement_fail_times = 0
            rospy.loginfo('Movement fail times reset.')
            if self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION:
                self.__escape_flag = self.EscapeFlag.NONE
                rospy.loginfo('Escape success.')
        return
    
    def __publish_cmd_vel(self, twist:Twist) -> None:
        self.__last_twist = twist
        self.__cmd_vel_pub.publish(twist)
        return
    
    def __save_results(self) -> None:
        if self.__visited_map is not None:
            cv2.imwrite(os.path.join(self.__results_dir, 'visited_map.png'), self.__visited_map)
        if self.__topdown_free_map_imshow is not None:
            cv2.imwrite(os.path.join(self.__results_dir, 'topdown_free_map.png'), self.__topdown_free_map_imshow)
            
    def __interpolate_path(self) -> None:
        self.__controller_destination_flag = False
        if self.__interpolate_path_flag:
            self.__navigation_path = interpolate_path(self.__navigation_path)

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} planner node.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--hide_windows',
                        type=int,
                        required=True,
                        help='Disable windows.')
    parser.add_argument('--save_runtime_data',
                        type=int,
                        required=True,
                        help='Save runtime data.')
    parser.add_argument('--debug',
                        type=int,
                        required=False,
                        help='Debug mode, output more logs.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.DEBUG if bool(args.debug) else rospy.INFO)
    
    PlannerNode(args.config, bool(args.hide_windows), bool(args.save_runtime_data))
    
    rospy.loginfo(f'{PROJECT_NAME} planner node finished.')