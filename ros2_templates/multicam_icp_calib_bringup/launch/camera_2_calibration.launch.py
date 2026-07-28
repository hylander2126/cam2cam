"""Launch camera 2 with PointCloud2, its internal TF tree, and a dummy base TF.

Use this when the robot's normal bringup already launches camera 1. The dummy
base->camera_2_link transform must be stopped and replaced after calibration.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    realsense_launch = PythonLaunchDescriptionSource(
        str(
            Path(get_package_share_directory("realsense2_camera"))
            / "launch"
            / "rs_launch.py"
        )
    )

    camera2 = IncludeLaunchDescription(
        realsense_launch,
        launch_arguments={
            "serial_no": LaunchConfiguration("camera2_serial"),
            "camera_name": LaunchConfiguration("camera2_name"),
            "camera_namespace": LaunchConfiguration("camera2_namespace"),
            "enable_depth": "true",
            "enable_color": "true",
            "pointcloud.enable": "true",
            "publish_tf": "true",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
        }.items(),
    )

    dummy_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_2_temporary_identity_tf",
        output="log",
        condition=IfCondition(LaunchConfiguration("publish_dummy_tf")),
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
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera2_serial",
                description=(
                    "Camera 2 serial. Prefix numeric serials with an underscore."
                ),
            ),
            DeclareLaunchArgument("camera2_name", default_value="camera_2"),
            DeclareLaunchArgument(
                "camera2_namespace", default_value="camera_2"
            ),
            DeclareLaunchArgument("base_frame", default_value="base"),
            DeclareLaunchArgument(
                "camera2_link_frame", default_value="camera_2_link"
            ),
            DeclareLaunchArgument("publish_dummy_tf", default_value="true"),
            camera2,
            dummy_tf,
        ]
    )
