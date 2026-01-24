from setuptools import find_packages, setup

package_name = 'driving_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/driving_package.launch.py']),
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
            'driving_package = driving_package.bt_node:main',
        ],
    },
)
