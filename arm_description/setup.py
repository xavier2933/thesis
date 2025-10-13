from setuptools import setup
from glob import glob
import os

package_name = 'arm_description'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],  # <-- nothing here
    data_files=[
        ('share/' + package_name + '/launch', ['launch/panda_scaled.launch.py']),
        ('share/' + package_name + '/launch', ['launch/panda_scaled_test.launch.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Xavier O\'Keefe',
    maintainer_email='xavie@example.com',
    description='Scaled Panda robot description package',
    license='Apache License 2.0',
)
