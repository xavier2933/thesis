# my_scaled_panda.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open('/home/xavie/thesis_ws/src/arm_description/config/panda_scaled.urdf').read()}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/opt/ros/humble/share/moveit_resources_panda_moveit_config/config/demo.rviz']
        )
    ])
