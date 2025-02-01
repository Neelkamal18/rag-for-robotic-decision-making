from setuptools import setup
import os
from glob import glob

package_name = 'rag_robot_interface'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=[
        'setuptools', 
        'launch',  # Required for ROS2 launch files
        'launch_ros',  # Ensures ROS2 launch dependencies are met
    ],
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),  # Ensures package.xml is installed
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),  # Installs all launch files
    ],
    entry_points={
        'console_scripts': [
            'robot_api = rag_robot_interface.robot_api:main',  # ROS2 Humble Node Entry
        ],
    },
    zip_safe=True,
)
