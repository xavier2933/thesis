from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='driving_package',
            executable='driving_package',
            name='driving_package',
            output='screen'
        )
    ])
