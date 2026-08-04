"""Targetless calibration for overlapping RGB-D cameras.

Transform names use ``T_target_source``: points are mapped with
``p_target = T_target_source @ p_source``.
"""

from .filters import crop_point_cloud, prepare_point_cloud, voxel_downsample
from .registration import (
    Candidate,
    RegistrationConfig,
    RegistrationError,
    RegistrationResult,
    calibrate,
    calibrate_consensus,
    cluster_candidates,
    generate_candidates,
    global_ransac_registration,
    refine_registration,
    score_candidate,
    select_consensus_pose,
)
from .transforms import (
    load_transform,
    matrix_from_translation_euler,
    matrix_from_translation_quaternion,
    quaternion_from_matrix,
    save_transform,
)

__all__ = [
    "Candidate",
    "RegistrationConfig",
    "RegistrationError",
    "RegistrationResult",
    "calibrate",
    "calibrate_consensus",
    "cluster_candidates",
    "crop_point_cloud",
    "global_ransac_registration",
    "generate_candidates",
    "load_transform",
    "matrix_from_translation_euler",
    "matrix_from_translation_quaternion",
    "prepare_point_cloud",
    "quaternion_from_matrix",
    "refine_registration",
    "score_candidate",
    "save_transform",
    "select_consensus_pose",
    "voxel_downsample",
]

__version__ = "0.1.0"
