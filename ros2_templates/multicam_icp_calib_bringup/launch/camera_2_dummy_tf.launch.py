"""Publish only the temporary identity base->camera-2-link transform."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("base_frame", default_value="base"),
            DeclareLaunchArgument(
                "camera2_link_frame", default_value="camera_2_link"
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_2_temporary_identity_tf",
                output="log",
                arguments=[
                    "--frame-id",
                    LaunchConfiguration("base_frame"),
                    "--child-frame-id",
                    LaunchConfiguration("camera2_link_frame"),
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--qx",
                    "0",
                    "--qy",
                    "0",
                    "--qz",
                    "0",
                    "--qw",
                    "1",
                ],
            ),
        ]
    )
