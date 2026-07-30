"""Minimal targetless calibration example using two saved RGB-D clouds."""

import logging

import numpy as np
import open3d as o3d

from icp_calib import RegistrationConfig, calibrate

logging.basicConfig(level=logging.INFO)

# Known robot-base <- camera-1 transform from the trusted calibration.
T_base_cam1 = np.array(
    [
        [1.0, 0.0, 0.0, 0.50],
        [0.0, 1.0, 0.0, 0.00],
        [0.0, 0.0, 1.0, 0.80],
        [0.0, 0.0, 0.0, 1.00],
    ],
    dtype=np.float64,
)

# Replace these paths with approximately simultaneous captures. Coordinates
# must be in metres; colored ICP also requires per-point RGB on both clouds.
pcd_cam1 = o3d.io.read_point_cloud("camera_1_workspace.ply")
pcd_cam2 = o3d.io.read_point_cloud("camera_2_workspace.ply")

# Calibration itself takes fewer than five lines.
config = RegistrationConfig(voxel_size=0.008, refinement="colored")
T_cam1_cam2 = calibrate(pcd_cam1, pcd_cam2, config=config)
T_base_cam2 = T_base_cam1 @ T_cam1_cam2
print("T_base_cam2:\n", T_base_cam2)
