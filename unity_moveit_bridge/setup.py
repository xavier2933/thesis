from setuptools import find_packages, setup

package_name = 'unity_moveit_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/unity_full_demo.launch.py']),
        ('share/' + package_name + '/launch', ['launch/arm_launch.launch.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xavie',
    maintainer_email='xaok7569@colorado.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'unity_goal_to_moveit = unity_moveit_bridge.unity_goal_to_moveit:main',
            'unity_target_pubsub = unity_moveit_bridge.unity_target_pubsub:main',
            'control_arbitrator = unity_moveit_bridge.control_arbitrator:main',
            'lerobot_recorder = unity_moveit_bridge.lerobot_recorder:main'
        ],
    },
)
