import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory("listener")

    language = "en"
    model_path = os.path.join(pkg_share, "model")
    grammar_path = os.path.join(pkg_share, "grammar")
    sample_rate = 44100
    listen_service = "/listen"

    return LaunchDescription([
        Node(
            package="listener",
            executable="listener",
            name="listener",
            output="screen",
            parameters=[
                {
                    "language": language,
                    "model_path": model_path,
                    "grammar_path": grammar_path,
                    "sample_rate": sample_rate,
                    "listen_service": listen_service,
                }
            ],
        )
    ])

