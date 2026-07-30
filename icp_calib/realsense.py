"""Optional Intel RealSense acquisition backend."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import open3d as o3d

LOGGER = logging.getLogger(__name__)


class RealSenseUnavailableError(RuntimeError):
    """Raised when the optional RealSense backend is not installed."""


def realsense_link_to_depth_optical() -> np.ndarray:
    """Return the conventional colocated RealSense body-to-optical transform.

    The body frame is X-forward/Y-left/Z-up and the depth optical frame is
    X-right/Y-down/Z-forward. Direct capture has no ROS ``camera_link`` object,
    so the body origin is defined at the native depth optical origin.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    return transform


def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RealSenseUnavailableError(
            "RealSense support is not installed. Run "
            "'python -m pip install icp-calib[realsense]'."
        ) from error
    return rs


@dataclass(frozen=True)
class RealSenseDevice:
    serial: str
    name: str
    usb_type: str = "unknown"

    @property
    def label(self) -> str:
        return f"{self.name} — {self.serial} — USB {self.usb_type}"


def list_devices() -> list[RealSenseDevice]:
    """Return attached RealSense devices with stable serial identifiers."""
    rs = _rs()
    devices = []
    for device in rs.context().query_devices():
        usb_type = (
            device.get_info(rs.camera_info.usb_type_descriptor)
            if device.supports(rs.camera_info.usb_type_descriptor)
            else "unknown"
        )
        devices.append(
            RealSenseDevice(
                serial=device.get_info(rs.camera_info.serial_number),
                name=device.get_info(rs.camera_info.name),
                usb_type=usb_type,
            )
        )
    return devices


def factory_extrinsics(serial: str) -> dict[str, Any]:
    """Read factory depth-to-stream extrinsics directly from a device.

    Matrices follow this package's ``T_target_source`` convention. No streams
    need to be started; librealsense exposes rigid factory calibration through
    stream profiles.
    """
    rs = _rs()
    device = next(
        (
            candidate
            for candidate in rs.context().query_devices()
            if candidate.get_info(rs.camera_info.serial_number) == serial
        ),
        None,
    )
    if device is None:
        raise ValueError(f"RealSense device '{serial}' was not found")

    profiles: dict[tuple[Any, int], Any] = {}
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            key = (profile.stream_type(), profile.stream_index())
            profiles.setdefault(key, profile)
    depth = next(
        (
            profile
            for (stream_type, _index), profile in profiles.items()
            if stream_type == rs.stream.depth
        ),
        None,
    )
    if depth is None:
        raise RuntimeError(f"RealSense device '{serial}' has no depth profile")

    transforms: dict[str, list[list[float]]] = {}
    for (stream_type, index), target in profiles.items():
        try:
            extrinsics = depth.get_extrinsics_to(target)
        except RuntimeError:
            continue
        rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(
            3, 3, order="F"
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = np.asarray(extrinsics.translation, dtype=np.float64)
        stream_name = str(stream_type).removeprefix("stream.")
        transforms[f"depth_to_{stream_name}_{index}"] = matrix.tolist()

    return {
        "serial": serial,
        "model": device.get_info(rs.camera_info.name),
        "convention": "T_target_source; p_target = T_target_source @ p_source",
        "camera_link_note": (
            "RealSense ROS camera_link, depth, and left-IR origins coincide"
        ),
        "T_link_depth_optical": realsense_link_to_depth_optical().tolist(),
        "factory_depth_to_stream": transforms,
    }


def _texture_colors(points: Any, color_frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return colors and a validity mask for RealSense point vertices."""
    image = np.asanyarray(color_frame.get_data())
    height, width = image.shape[:2]
    texcoords = np.asanyarray(points.get_texture_coordinates()).view(
        np.float32
    ).reshape(-1, 2)
    u = np.rint(texcoords[:, 0] * (width - 1)).astype(np.int64)
    v = np.rint(texcoords[:, 1] * (height - 1)).astype(np.int64)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    colors = np.zeros((len(texcoords), 3), dtype=np.float64)
    colors[valid] = image[v[valid], u[valid], :3] / 255.0
    return colors, valid


def _frames_to_cloud(
    rs: Any,
    frames: Any,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> o3d.geometry.PointCloud:
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        raise RuntimeError("RealSense frameset is missing depth or color")
    point_calculator = rs.pointcloud()
    point_calculator.map_to(color)
    rs_points = point_calculator.calculate(depth)
    vertices = np.asanyarray(rs_points.get_vertices()).view(np.float32).reshape(-1, 3)
    colors, texture_valid = _texture_colors(rs_points, color)
    valid = (
        np.all(np.isfinite(vertices), axis=1)
        & (vertices[:, 2] >= min_depth_m)
        & (vertices[:, 2] <= max_depth_m)
        & texture_valid
    )
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(vertices[valid].astype(np.float64))
    )
    cloud.colors = o3d.utility.Vector3dVector(colors[valid])
    return cloud


def capture_pair(
    serial_cam1: str,
    serial_cam2: str,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup_frames: int = 30,
    accumulate_frames: int = 3,
    min_depth_cam1_m: float = 0.10,
    max_depth_cam1_m: float = 3.0,
    min_depth_cam2_m: float = 0.07,
    max_depth_cam2_m: float = 1.00,
    preview_callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Capture two colored clouds while both hardware pipelines are running.

    Point geometry stays in each camera's native depth optical frame. RGB is
    projected onto those points only as a texture; no depth-to-color resampling
    is applied. This avoids model-specific alignment artifacts, notably on D405.
    """
    if not serial_cam1 or not serial_cam2 or serial_cam1 == serial_cam2:
        raise ValueError("Select two different RealSense serial numbers")
    if min(width, height, fps, accumulate_frames) <= 0 or warmup_frames < 0:
        raise ValueError("Capture dimensions/counts must be positive")
    for name, minimum, maximum in (
        ("camera 1", min_depth_cam1_m, max_depth_cam1_m),
        ("camera 2", min_depth_cam2_m, max_depth_cam2_m),
    ):
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or minimum < 0
            or minimum >= maximum
        ):
            raise ValueError(f"{name} depth range must satisfy 0 <= min < max")
    rs = _rs()

    def start(serial: str):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        pipeline.start(config)
        return pipeline

    pipeline1 = start(serial_cam1)
    pipeline2 = None
    try:
        pipeline2 = start(serial_cam2)
        for index in range(warmup_frames):
            frames1 = pipeline1.wait_for_frames(5000)
            frames2 = pipeline2.wait_for_frames(5000)
            if preview_callback and (
                index % 5 == 0 or index == warmup_frames - 1
            ):
                preview_callback(
                    np.asanyarray(frames1.get_color_frame().get_data()).copy(),
                    np.asanyarray(frames2.get_color_frame().get_data()).copy(),
                )

        clouds1: list[o3d.geometry.PointCloud] = []
        clouds2: list[o3d.geometry.PointCloud] = []
        for _ in range(accumulate_frames):
            frames1 = pipeline1.wait_for_frames(5000)
            frames2 = pipeline2.wait_for_frames(5000)
            if preview_callback:
                preview_callback(
                    np.asanyarray(frames1.get_color_frame().get_data()).copy(),
                    np.asanyarray(frames2.get_color_frame().get_data()).copy(),
                )
            clouds1.append(
                _frames_to_cloud(
                    rs,
                    frames1,
                    min_depth_m=min_depth_cam1_m,
                    max_depth_m=max_depth_cam1_m,
                )
            )
            clouds2.append(
                _frames_to_cloud(
                    rs,
                    frames2,
                    min_depth_m=min_depth_cam2_m,
                    max_depth_m=max_depth_cam2_m,
                )
            )
        return (
            sum(clouds1[1:], clouds1[0]),
            sum(clouds2[1:], clouds2[0]),
        )
    finally:
        pipeline1.stop()
        if pipeline2 is not None:
            pipeline2.stop()


def capture_single(
    serial: str,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup_frames: int = 30,
    accumulate_frames: int = 3,
    min_depth_m: float = 0.07,
    max_depth_m: float = 1.00,
    preview_callback: Callable[[np.ndarray], None] | None = None,
) -> o3d.geometry.PointCloud:
    """Capture one native-depth cloud without opening any other device."""
    if not serial:
        raise ValueError("A RealSense serial number is required")
    if min(width, height, fps, accumulate_frames) <= 0 or warmup_frames < 0:
        raise ValueError("Capture dimensions/counts must be positive")
    if min_depth_m < 0 or min_depth_m >= max_depth_m:
        raise ValueError("Depth range must satisfy 0 <= min < max")
    rs = _rs()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    try:
        pipeline.start(config)
    except RuntimeError as error:
        if "busy" in str(error).lower() or "errno=16" in str(error).lower():
            raise RuntimeError(
                f"RealSense {serial} is busy. Stop its ROS camera driver or "
                "select ROS PointCloud2 acquisition for that camera."
            ) from error
        raise
    try:
        for index in range(warmup_frames):
            frames = pipeline.wait_for_frames(5000)
            if preview_callback and (
                index % 5 == 0 or index == warmup_frames - 1
            ):
                preview_callback(
                    np.asanyarray(frames.get_color_frame().get_data()).copy()
                )
        clouds = []
        for _ in range(accumulate_frames):
            frames = pipeline.wait_for_frames(5000)
            if preview_callback:
                preview_callback(
                    np.asanyarray(frames.get_color_frame().get_data()).copy()
                )
            clouds.append(
                _frames_to_cloud(
                    rs,
                    frames,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
            )
        return sum(clouds[1:], clouds[0])
    finally:
        pipeline.stop()
