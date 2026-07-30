"""Rigid-transform conversion and standalone calibration file I/O."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def validate_transform(matrix: np.ndarray, name: str = "transform") -> np.ndarray:
    """Validate and return a copied 4x4 rigid transform."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7):
        raise ValueError(f"{name} must have homogeneous final row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform.copy()


def matrix_from_translation_quaternion(
    translation: Any, quaternion_xyzw: Any
) -> np.ndarray:
    """Build ``T_parent_child`` from XYZ and an XYZW quaternion."""
    xyz = np.asarray(translation, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("translation must contain three finite values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite XYZW values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = xyz
    return matrix


def matrix_from_translation_euler(
    translation: Any,
    roll_pitch_yaw_degrees: Any,
) -> np.ndarray:
    """Build a transform from XYZ and fixed-frame roll/pitch/yaw in degrees.

    Rotation uses the common ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` convention.
    """
    xyz = np.asarray(translation, dtype=np.float64)
    rpy = np.asarray(roll_pitch_yaw_degrees, dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError("translation and roll/pitch/yaw must each have 3 values")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rpy)):
        raise ValueError("translation and roll/pitch/yaw must be finite")
    roll, pitch, yaw = np.deg2rad(rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a normalized XYZW quaternion from a rigid transform."""
    rotation = validate_transform(matrix)[:3, :3]
    # Stable branch-based conversion, including rotations close to 180 degrees.
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    # q and -q represent the same rotation; positive W gives stable file output.
    return -quaternion if quaternion[3] < 0 else quaternion


def _matrix_from_mapping(data: dict[str, Any]) -> np.ndarray:
    for key in ("matrix", "transformation", "T_base_cam1", "T_base_cam2"):
        if key in data:
            return validate_transform(np.asarray(data[key], dtype=np.float64))
    translation = data.get("translation")
    quaternion = data.get("quaternion_xyzw", data.get("quaternion"))
    if isinstance(translation, dict):
        translation = [translation[key] for key in ("x", "y", "z")]
    if isinstance(quaternion, dict):
        quaternion = [quaternion[key] for key in ("x", "y", "z", "w")]
    if translation is not None and quaternion is not None:
        return matrix_from_translation_quaternion(translation, quaternion)
    raise ValueError(
        "JSON must contain a 4x4 'matrix', or 'translation' and "
        "'quaternion_xyzw'"
    )


def _load_static_transform_launch(path: Path) -> np.ndarray:
    """Read constant static_transform_publisher flags without importing ROS."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    arguments: list[str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "arguments":
            continue
        try:
            candidate = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, list) and "--x" in candidate and "--qw" in candidate:
            arguments = [str(value) for value in candidate]
            break
    if arguments is None:
        # Comments or simple formatting should not break importing the common
        # static publisher format, but substitutions intentionally are rejected.
        text = path.read_text(encoding="utf-8")
        flags = dict(
            re.findall(
                r'["\']--(x|y|z|qx|qy|qz|qw)["\']\s*,\s*["\']([^"\']+)["\']',
                text,
            )
        )
    else:
        flags = {
            arguments[index][2:]: arguments[index + 1]
            for index in range(len(arguments) - 1)
            if arguments[index].startswith("--")
        }
    required = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if not all(key in flags for key in required):
        raise ValueError(
            "Launch import supports constant --x/--y/--z and --qx/--qy/--qz/--qw "
            "static_transform_publisher arguments"
        )
    return matrix_from_translation_quaternion(
        [float(flags[key]) for key in ("x", "y", "z")],
        [float(flags[key]) for key in ("qx", "qy", "qz", "qw")],
    )


def load_transform(path: str | Path) -> np.ndarray:
    """Load a transform from JSON, NPY, text/CSV, or a constant ROS launch file."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        with source.open(encoding="utf-8") as stream:
            return _matrix_from_mapping(json.load(stream))
    if suffix == ".npy":
        return validate_transform(np.load(source, allow_pickle=False))
    if source.name.endswith(".launch.py"):
        return _load_static_transform_launch(source)
    delimiter = "," if suffix == ".csv" else None
    return validate_transform(np.loadtxt(source, delimiter=delimiter))


def save_transform(
    path: str | Path,
    matrix: np.ndarray,
    *,
    parent_frame: str = "base",
    child_frame: str = "camera_2_depth_optical_frame",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a transform as JSON, NPY, CSV, text, or a ROS 2 launch file."""
    destination = Path(path)
    transform = validate_transform(matrix)
    suffix = destination.suffix.lower()
    if destination.name.endswith(".launch.py"):
        translation = transform[:3, 3]
        quaternion = quaternion_from_matrix(transform)
        values = {
            "x": translation[0],
            "y": translation[1],
            "z": translation[2],
            "qx": quaternion[0],
            "qy": quaternion[1],
            "qz": quaternion[2],
            "qw": quaternion[3],
        }
        argument_lines = "\n".join(
            f'                "--{name}",\n'
            f'                "{float(value):.12g}",'
            for name, value in values.items()
        )
        is_temporary = bool(
            metadata and metadata.get("temporary_identity_transform")
        )
        purpose = (
            "TEMPORARY IDENTITY ONLY; replace this after calibration"
            if is_temporary
            else "Calibration result"
        )
        launch_source = (
            '"""Static transform generated by icp_calib."""\n'
            f'"""{purpose}: {parent_frame} -> {child_frame}."""\n'
            "from launch import LaunchDescription\n"
            "from launch_ros.actions import Node\n\n\n"
            "def generate_launch_description() -> LaunchDescription:\n"
            "    nodes = [\n"
            "        Node(\n"
            '            package="tf2_ros",\n'
            '            executable="static_transform_publisher",\n'
            '            output="log",\n'
            "            arguments=[\n"
            '                "--frame-id",\n'
            f"                {parent_frame!r},\n"
            '                "--child-frame-id",\n'
            f"                {child_frame!r},\n"
            f"{argument_lines}\n"
            "            ],\n"
            "        ),\n"
            "    ]\n"
            "    return LaunchDescription(nodes)\n"
        )
        destination.write_text(launch_source, encoding="utf-8")
    elif suffix == ".json":
        payload: dict[str, Any] = {
            "convention": "p_parent = T_parent_child @ p_child",
            "parent_frame": parent_frame,
            "child_frame": child_frame,
            "matrix": transform.tolist(),
            "translation": transform[:3, 3].tolist(),
            "quaternion_xyzw": quaternion_from_matrix(transform).tolist(),
        }
        if metadata:
            payload["registration"] = metadata
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    elif suffix == ".npy":
        np.save(destination, transform, allow_pickle=False)
    else:
        np.savetxt(destination, transform, delimiter="," if suffix == ".csv" else " ")
