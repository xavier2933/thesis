from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # MoveIt demo launch
    moveit_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("moveit_resources_panda_moveit_config"),
                "launch",
                "demo.launch.py",
            ])
        )
    )

    # ROS–Unity TCP endpoint
    tcp_endpoint = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        output="screen"
    )

    # Your Unity bridge node
    unity_bridge = Node(
        package="unity_moveit_bridge",
        executable="unity_target_pubsub",
        output="screen"
    )

    return LaunchDescription([
        moveit_demo,
        tcp_endpoint,
        unity_bridge
    ])
