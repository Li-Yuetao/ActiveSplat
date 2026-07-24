"""Launch the native ROS2 ActiveSplat Habitat pipeline."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('activesplat'))
    source_root = share.parents[3] / 'src' / 'activesplat'
    config = LaunchConfiguration('config')
    common = [
        DeclareLaunchArgument('mapper', default_value='SplaTAM'),
        DeclareLaunchArgument('config', default_value=str(share / 'config/datasets/gibson.json')),
        DeclareLaunchArgument('scene_id', default_value='Denmark'),
        DeclareLaunchArgument('remark', default_value='NONE'),
        DeclareLaunchArgument('gpu_id', default_value='0'),
        DeclareLaunchArgument('user_config', default_value=str(share / 'config/user_config.json')),
        DeclareLaunchArgument('mode', default_value='AUTO_PLANNING'),
        DeclareLaunchArgument('actions', default_value='None'),
        DeclareLaunchArgument('parallelized', default_value='0'),
        DeclareLaunchArgument('debug', default_value='0'),
        DeclareLaunchArgument('hide_mapper_windows', default_value='0'),
        DeclareLaunchArgument('hide_planner_windows', default_value='0'),
        DeclareLaunchArgument('step_num', default_value='-1'),
        DeclareLaunchArgument('save_runtime_data', default_value='0'),
        SetEnvironmentVariable(
            'ACTIVESPLAT_SOURCE_DIR',
            EnvironmentVariable('ACTIVESPLAT_SOURCE_DIR', default_value=str(source_root))),
    ]
    mapper = Node(
        package='activesplat', executable='mapper_node.py', name='mapper_node', output='screen',
        parameters=[{'step_num': LaunchConfiguration('step_num')}],
        arguments=[
            '--mapper', LaunchConfiguration('mapper'),
            '--config', config,
            '--scene_id', LaunchConfiguration('scene_id'),
            '--user_config', LaunchConfiguration('user_config'),
            '--gpu_id', LaunchConfiguration('gpu_id'),
            '--mode', LaunchConfiguration('mode'),
            '--actions', LaunchConfiguration('actions'),
            '--parallelized', LaunchConfiguration('parallelized'),
            '--hide_windows', LaunchConfiguration('hide_mapper_windows'),
            '--debug', LaunchConfiguration('debug'),
            '--save_runtime_data', LaunchConfiguration('save_runtime_data'),
            '--remark', LaunchConfiguration('remark'),
        ])
    planner = Node(
        package='activesplat', executable='planner_node.py', name='planner_node', output='screen',
        arguments=[
            '--config', config,
            '--hide_windows', LaunchConfiguration('hide_planner_windows'),
            '--debug', LaunchConfiguration('debug'),
            '--save_runtime_data', LaunchConfiguration('save_runtime_data'),
        ])
    return LaunchDescription(common + [mapper, planner])