from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rag_robot_interface',
            executable='robot_api',
            name='robot_nlp_interface',
            namespace='rag',
            output='log',
            parameters=[
                {"command_topic": "/robot_commands"},
                {"response_topic": "/robot_response"}
            ],
        )
    ])
