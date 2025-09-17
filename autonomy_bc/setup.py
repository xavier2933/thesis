from setuptools import find_packages, setup

package_name = 'autonomy_bc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/robot_bc_model.pth']),

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
            'extract_trajectories = autonomy_bc.extract_and_train:main',
            'bc_controller = autonomy_bc.bc_controller:main',
        ],
    },
)
