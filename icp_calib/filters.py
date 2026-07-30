"""Point-cloud filters for commodity RGB-D sensor data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import open3d as o3d


def _validate_cloud(pcd: o3d.geometry.PointCloud, name: str = "pcd") -> None:
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError(f"{name} must be an open3d.geometry.PointCloud")
    if pcd.is_empty():
        raise ValueError(f"{name} is empty")


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return result


def crop_point_cloud(
    pcd: o3d.geometry.PointCloud,
    min_bound: Sequence[float],
    max_bound: Sequence[float],
) -> o3d.geometry.PointCloud:
    """Return points inside an axis-aligned box in the cloud's frame.

    Point colors and normals are retained. The input cloud is not modified.
    """
    _validate_cloud(pcd)
    lower = _vector3(min_bound, "min_bound")
    upper = _vector3(max_bound, "max_bound")
    if np.any(lower >= upper):
        raise ValueError("Each min_bound component must be less than max_bound")
    box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    return pcd.crop(box)


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    *,
    remove_non_finite: bool = True,
) -> o3d.geometry.PointCloud:
    """Return a cleaned, voxel-downsampled copy of ``pcd``.

    ``voxel_size`` uses the cloud's units, which should normally be metres.
    """
    _validate_cloud(pcd)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be a finite positive number")

    cloud = o3d.geometry.PointCloud(pcd)
    if remove_non_finite:
        cloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    return cloud.voxel_down_sample(float(voxel_size))


def prepare_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    *,
    min_bound: Sequence[float] | None = None,
    max_bound: Sequence[float] | None = None,
    remove_statistical_outliers: bool = False,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Crop, clean, optionally reject outliers, and downsample a cloud.

    Cropping bounds are optional, but must be supplied as a pair. Statistical
    filtering is opt-in because aggressive filtering can erase thin robot or
    workpiece geometry.
    """
    if (min_bound is None) != (max_bound is None):
        raise ValueError("min_bound and max_bound must be supplied together")

    cloud = (
        crop_point_cloud(pcd, min_bound, max_bound)
        if min_bound is not None and max_bound is not None
        else o3d.geometry.PointCloud(pcd)
    )
    cloud = voxel_downsample(cloud, voxel_size)

    if remove_statistical_outliers:
        if nb_neighbors < 3:
            raise ValueError("nb_neighbors must be at least 3")
        if not np.isfinite(std_ratio) or std_ratio <= 0:
            raise ValueError("std_ratio must be finite and positive")
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
        )
    return cloud
