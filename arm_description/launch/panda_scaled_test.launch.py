# File: panda_scaled_test.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os
import xacro

def generate_launch_description():
    # Path to your xacro file (update if your folder differs)
    xacro_file = os.path.join(
        os.getcwd(), 'src', 'arm_description', 'config', 'panda.urdf.xacro'
    )

    # Expand xacro into URDF XML string
    urdf_xml = xacro.process_file(xacro_file).toxml()

    # Node to publish robot_state
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml}]
    )

    # Node to launch RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(
            os.getcwd(), 'src', 'arm_description', 'config', 'panda.rviz'
        )]  # optional: remove if you don't have an RViz config
    )

    return LaunchDescription([
        robot_state_publisher_node,
        rviz_node
    ])
