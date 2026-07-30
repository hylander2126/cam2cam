"""Optional live ROS 2 TF adapter.

This module is imported only when the user explicitly connects the standalone
GUI to ROS. The core calibration package has no ROS runtime dependency.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any
import struct

import numpy as np


class RosTfUnavailableError(RuntimeError):
    """Raised when ROS 2 Python or tf2 is unavailable."""


def _transform_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.translation
    quaternion = transform.rotation
    q = np.array(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("TF contains an invalid quaternion")
    x, y, z, w = q / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def _pointcloud2_to_open3d(message: Any, topic: str) -> Any:
    """Convert an XYZ/RGB PointCloud2 message without ROS helper packages."""
    from sensor_msgs.msg import PointField
    import open3d as o3d

    fields = {field.name: field for field in message.fields}
    for name in ("x", "y", "z"):
        if name not in fields or fields[name].datatype != PointField.FLOAT32:
            raise ValueError(
                f"PointCloud2 '{topic}' requires FLOAT32 x/y/z fields"
            )
    color_field = fields.get("rgb") or fields.get("rgba")
    endian = ">" if message.is_bigendian else "<"
    unpack_float = struct.Struct(endian + "f").unpack_from
    unpack_uint = struct.Struct(endian + "I").unpack_from
    count = message.width * message.height
    points = np.empty((count, 3), dtype=np.float64)
    colors = np.empty((count, 3), dtype=np.float64)
    index = 0
    for row in range(message.height):
        row_offset = row * message.row_step
        for column in range(message.width):
            offset = row_offset + column * message.point_step
            points[index] = [
                unpack_float(message.data, offset + fields[axis].offset)[0]
                for axis in ("x", "y", "z")
            ]
            if color_field is not None:
                packed = unpack_uint(
                    message.data, offset + color_field.offset
                )[0]
                colors[index] = (
                    (packed >> 16) & 255,
                    (packed >> 8) & 255,
                    packed & 255,
                )
            index += 1
    valid = np.all(np.isfinite(points), axis=1)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[valid]))
    if color_field is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors[valid] / 255.0)
    return cloud


class LiveRosTf:
    """Own a background ROS context and a continuously updated tf2 buffer."""

    def __init__(self) -> None:
        try:
            import rclpy
            from rclpy.context import Context
            from tf2_ros import Buffer, TransformListener
        except ImportError as error:
            raise RosTfUnavailableError(
                "ROS 2 TF support is unavailable. Source your ROS installation "
                "before starting the GUI, for example: "
                "'source /opt/ros/jazzy/setup.bash'."
            ) from error

        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            "icp_calib_tf_listener", context=self._context
        )
        self._buffer = Buffer(node=self._node)
        self._listener = TransformListener(
            self._buffer, self._node, spin_thread=False
        )
        self._executor = rclpy.executors.SingleThreadedExecutor(
            context=self._context
        )
        self._executor.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(
            target=self._spin, name="icp-calib-tf-listener", daemon=True
        )
        self._thread.start()

    def _spin(self) -> None:
        while self._running and self._context.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def wait_for_frames(
        self,
        callback: Callable[[list[str] | Exception], None],
        *,
        wait_seconds: float = 2.0,
    ) -> None:
        """Collect TF traffic briefly, then return known frame names."""
        def worker() -> None:
            time.sleep(wait_seconds)
            try:
                import yaml

                graph = yaml.safe_load(self._buffer.all_frames_as_yaml()) or {}
                frames = set(graph)
                for details in graph.values():
                    parent = details.get("parent")
                    if parent:
                        frames.add(parent)
                callback(sorted(frames))
            except Exception as error:
                callback(error)

        threading.Thread(target=worker, daemon=True).start()

    def lookup_matrix(
        self,
        target_frame: str,
        source_frame: str,
        *,
        timeout_seconds: float = 1.0,
    ) -> np.ndarray:
        """Return ``T_target_source`` using the latest live TF data."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        if not target_frame or not source_frame:
            raise ValueError("TF frame names must be non-empty")
        stamped = self._buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=timeout_seconds),
        )
        return _transform_to_matrix(stamped.transform)

    def pointcloud_topics(self) -> list[str]:
        """Return currently advertised ROS PointCloud2 topic names."""
        expected_type = "sensor_msgs/msg/PointCloud2"
        return sorted(
            name
            for name, topic_types in self._node.get_topic_names_and_types()
            if expected_type in topic_types
        )

    def capture_pointcloud(
        self, topic: str, *, timeout_seconds: float = 8.0
    ) -> tuple[Any, str]:
        """Block for one ROS PointCloud2 and convert it to an Open3D cloud."""
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data

        if not topic:
            raise ValueError("PointCloud2 topic must be non-empty")
        received: list[PointCloud2] = []
        ready = threading.Event()

        def callback(message: PointCloud2) -> None:
            if not received:
                received.append(message)
                ready.set()

        subscription = self._node.create_subscription(
            PointCloud2, topic, callback, qos_profile_sensor_data
        )
        try:
            if not ready.wait(timeout_seconds):
                raise TimeoutError(
                    f"No PointCloud2 received from '{topic}' within "
                    f"{timeout_seconds:.1f} seconds"
                )
            message = received[0]
        finally:
            self._node.destroy_subscription(subscription)

        return _pointcloud2_to_open3d(message, topic), message.header.frame_id

    def capture_pointcloud_pair(
        self,
        topic1: str,
        topic2: str,
        *,
        timeout_seconds: float = 8.0,
    ) -> tuple[Any, str, Any, str]:
        """Capture the next messages from two topics using concurrent subscriptions."""
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data

        if not topic1 or not topic2:
            raise ValueError("Both PointCloud2 topics are required")
        if topic1 == topic2:
            raise ValueError("Camera 1 and camera 2 topics must be different")
        received: list[PointCloud2 | None] = [None, None]
        ready = threading.Event()

        def make_callback(index: int) -> Callable[[PointCloud2], None]:
            def callback(message: PointCloud2) -> None:
                if received[index] is None:
                    received[index] = message
                    if all(item is not None for item in received):
                        ready.set()
            return callback

        subscriptions = [
            self._node.create_subscription(
                PointCloud2, topic, make_callback(index), qos_profile_sensor_data
            )
            for index, topic in enumerate((topic1, topic2))
        ]
        try:
            if not ready.wait(timeout_seconds):
                missing = [
                    topic
                    for topic, message in zip((topic1, topic2), received)
                    if message is None
                ]
                raise TimeoutError(
                    "No PointCloud2 received within "
                    f"{timeout_seconds:.1f} seconds from: {', '.join(missing)}"
                )
        finally:
            for subscription in subscriptions:
                self._node.destroy_subscription(subscription)

        message1, message2 = received
        assert message1 is not None and message2 is not None
        return (
            _pointcloud2_to_open3d(message1, topic1),
            message1.header.frame_id,
            _pointcloud2_to_open3d(message2, topic2),
            message2.header.frame_id,
        )

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=1.0)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        if self._context.ok():
            self._context.shutdown()
