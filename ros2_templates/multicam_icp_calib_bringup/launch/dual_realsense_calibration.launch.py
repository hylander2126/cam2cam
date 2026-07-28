"""Launch two RealSense cameras and a temporary base->camera-2-link TF."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera_include(
    source: PythonLaunchDescriptionSource,
    *,
    serial: str,
    name: str,
    namespace: str,
) -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        source,
        launch_arguments={
            "serial_no": LaunchConfiguration(serial),
            "camera_name": LaunchConfiguration(name),
            "camera_namespace": LaunchConfiguration(namespace),
            "enable_depth": "true",
            "enable_color": "true",
            "pointcloud.enable": "true",
            "publish_tf": "true",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
        }.items(),
    )


def generate_launch_description() -> LaunchDescription:
    realsense_launch = PythonLaunchDescriptionSource(
        str(
            Path(get_package_share_directory("realsense2_camera"))
            / "launch"
            / "rs_launch.py"
        )
    )
    camera1 = _camera_include(
        realsense_launch,
        serial="camera1_serial",
        name="camera1_name",
        namespace="camera1_namespace",
    )
    camera2 = _camera_include(
        realsense_launch,
        serial="camera2_serial",
        name="camera2_name",
        namespace="camera2_namespace",
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
                "camera1_serial",
                description=(
                    "Camera 1 serial. Prefix numeric serials with an underscore."
                ),
            ),
            DeclareLaunchArgument(
                "camera2_serial",
                description=(
                    "Camera 2 serial. Prefix numeric serials with an underscore."
                ),
            ),
            DeclareLaunchArgument("camera1_name", default_value="camera_1"),
            DeclareLaunchArgument("camera2_name", default_value="camera_2"),
            DeclareLaunchArgument(
                "camera1_namespace", default_value="camera_1"
            ),
            DeclareLaunchArgument(
                "camera2_namespace", default_value="camera_2"
            ),
            DeclareLaunchArgument("base_frame", default_value="base"),
            DeclareLaunchArgument(
                "camera2_link_frame", default_value="camera_2_link"
            ),
            DeclareLaunchArgument("publish_dummy_tf", default_value="true"),
            camera1,
            camera2,
            dummy_tf,
        ]
    )
