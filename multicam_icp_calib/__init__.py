"""Targetless calibration for overlapping RGB-D cameras.

Transform names use ``T_target_source``: points are mapped with
``p_target = T_target_source @ p_source``.
"""

from .filters import crop_point_cloud, prepare_point_cloud, voxel_downsample
from .registration import (
    RegistrationConfig,
    RegistrationError,
    RegistrationResult,
    calibrate,
    global_ransac_registration,
    refine_registration,
)
from .transforms import (
    load_transform,
    matrix_from_translation_euler,
    matrix_from_translation_quaternion,
    quaternion_from_matrix,
    save_transform,
)

__all__ = [
    "RegistrationConfig",
    "RegistrationError",
    "RegistrationResult",
    "calibrate",
    "crop_point_cloud",
    "global_ransac_registration",
    "load_transform",
    "matrix_from_translation_euler",
    "matrix_from_translation_quaternion",
    "prepare_point_cloud",
    "quaternion_from_matrix",
    "refine_registration",
    "save_transform",
    "voxel_downsample",
]

__version__ = "0.1.0"
