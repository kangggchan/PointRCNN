import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_repo_root = os.environ.get('POINT_RCNN_ROOT', str(Path(__file__).resolve().parents[3]))
    default_params_file = PathJoinSubstitution([FindPackageShare('point_rcnn_ros'), 'config', 'point_rcnn_node.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params_file),
        DeclareLaunchArgument('repo_root', default_value=default_repo_root),
        DeclareLaunchArgument('input_topic', default_value='/points'),
        DeclareLaunchArgument('marker_topic', default_value='/point_rcnn/markers'),
        DeclareLaunchArgument('input_frame_mode', default_value='ros_lidar'),
        Node(
            package='point_rcnn_ros',
            executable='point_rcnn_node',
            name='point_rcnn_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'repo_root': LaunchConfiguration('repo_root'),
                    'input_topic': LaunchConfiguration('input_topic'),
                    'marker_topic': LaunchConfiguration('marker_topic'),
                    'input_frame_mode': LaunchConfiguration('input_frame_mode'),
                },
            ],
        ),
    ])