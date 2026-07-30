"""Publish the two camera attachments used during calibration.

Edit the constants below. Camera drivers publish each camera's internal
link-to-optical TF tree; this file only attaches those trees to BASE_FRAME.

If the robot already publishes camera 1's trusted attachment, set
PUBLISH_CAMERA_1 = False. Leave camera 2 at identity until cam2cam produces
its calibrated transform, then replace CAMERA_2_XYZ and CAMERA_2_XYZW.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


BASE_FRAME = "base"

PUBLISH_CAMERA_1 = True
CAMERA_1_FRAME = "camera_1_link"
CAMERA_1_XYZ = (0.0, 0.0, 0.0)  # Replace with the known pose.
CAMERA_1_XYZW = (0.0, 0.0, 0.0, 1.0)

CAMERA_2_FRAME = "camera_2_link"
CAMERA_2_XYZ = (0.0, 0.0, 0.0)  # Temporary identity during calibration.
CAMERA_2_XYZW = (0.0, 0.0, 0.0, 1.0)


def _publisher(name, child_frame, xyz, xyzw):
    x, y, z = xyz
    qx, qy, qz, qw = xyzw
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="log",
        arguments=[
            "--frame-id",
            BASE_FRAME,
            "--child-frame-id",
            child_frame,
            "--x",
            str(x),
            "--y",
            str(y),
            "--z",
            str(z),
            "--qx",
            str(qx),
            "--qy",
            str(qy),
            "--qz",
            str(qz),
            "--qw",
            str(qw),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    nodes = []
    if PUBLISH_CAMERA_1:
        nodes.append(
            _publisher(
                "camera_1_static_tf",
                CAMERA_1_FRAME,
                CAMERA_1_XYZ,
                CAMERA_1_XYZW,
            )
        )
    nodes.append(
        _publisher(
            "camera_2_static_tf",
            CAMERA_2_FRAME,
            CAMERA_2_XYZ,
            CAMERA_2_XYZW,
        )
    )
    return LaunchDescription(nodes)
