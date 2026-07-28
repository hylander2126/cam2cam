# cam2cam - Multicam ICP Calibration

Targetless extrinsic calibration for two overlapping RGB-D cameras, built as a
standalone Python tool with optional ROS 2 integration.

## Motivation
Ever look at your robotics setup and think *"I need another camera in the mix"*? A second view can expand the perception stack's field of view or improve visibility of partially occluded objects.

### The usual approach
To make depth/stereo cameras useful, one often has to accurately locate it in 3D space with respect to world coordinates. Handeye calibration has come a long way (see [MoveIt](https://moveit.picknik.ai/humble/doc/examples/hand_eye_calibration/hand_eye_calibration_tutorial.html)), but generally requires several viewpoints, ArUco/ChArUco boards, custom end effector fixturing, etc. **You get the point.** And that's just one camera. Adding another introduces a second view of the same workspace that must agree with the robot's existing coordinate system.

### Good news: there's a better way.

`cam2cam` estimates camera 2's pose relative to an already-calibrated camera 1 by registering their point clouds. It uses Open3D, FPFH descriptors, global RANSAC (with Fast Global Registration as a fallback), and multiscale
point-to-plane or colored ICP. It does not require a calibration board, GPU, Torch, or a learned model.

![cam2cam aligns two overlapping RGB-D views to estimate the new camera pose](img/cam2cam_pipeline.svg)

Under the hood, the tool:
1. Extracts distinctive local-shape "fingerprints" (FPFH) from both clouds;
2. Matches those features to estimate a rough alignment with RANSAC; and
3. Tightens the alignment with multiscale ICP.

The result is camera 2's pose relative to the trusted camera 1 OR relative to world coordinates. Your choice.

## Features

- Capture two RealSense point clouds directly or receive ROS 2
  `PointCloud2` topics.
- Estimate camera 2's pose relative to a trusted camera without a calibration
  target.
- Review individual clouds, their registered overlay, and quality metrics in a
  GUI.
- Use geometric or colored ICP and optionally provide an approximate initial
  pose.
- Export the result as transform data or a ROS 2 static-transform launch file.

`cam2cam` calibrates one stationary camera at a time. It does not estimate
camera intrinsics, synchronize camera clocks, track moving cameras, or perform
multi-camera bundle adjustment.

## Limitations

This is an early-stage engineering tool, not a certified calibration system.

- Tested primarily on Ubuntu 24.04, Python 3.12, ROS 2 Jazzy, and Intel
  RealSense D400-series cameras.
- Direct camera capture currently requires RealSense. Other RGB-D cameras may
  be used through Open3D or ROS 2, but remain untested.
- Both cameras and the scene must remain stationary during capture.
- Registration requires meaningful overlapping, non-planar geometry; blank
  walls, floors, repeated geometry, and small shared crops are unreliable.
- Always inspect and independently validate the transform before using it for
  robot motion.

## Installation

```bash
git clone https://github.com/hylander2126/cam2cam.git
cd cam2cam

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[all]'
```

On Ubuntu, the GUI may also require:

```bash
sudo apt install python3-venv python3-tk
```

To install only the reusable registration core without the GUI and direct
RealSense backend:

```bash
python3 -m pip install -e .
```

## Quick start

Launch the GUI:

```bash
python3 -m multicam_icp_calib.gui
```

Then:

1. Select camera 1, whose pose is already trusted, and camera 2, whose pose is
   unknown.
2. Enter or import camera 1's transform if the result should be expressed in a
   robot or world frame.
3. Capture both clouds and adjust their depth limits.
4. Run registration and inspect the aligned overlay and quality metrics.
5. Export the accepted transform.

Direct capture currently supports Intel RealSense cameras through
`pyrealsense2`. Stop ROS camera drivers or other programs that already own the
devices before capturing.

## ROS 2 usage

When both cameras already publish `PointCloud2` data in a robot TF tree, use the
supplied ROS 2 bringup package:

```bash
mkdir -p ~/ros2_ws/src
cp -r ros2_templates/multicam_icp_calib_bringup ~/ros2_ws/src/

cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select multicam_icp_calib_bringup
source install/setup.bash
```

See [ROS 2 setup and calibration](docs/ros2_setup.md) for camera launch, TF
configuration, registration, export, and validation.

## Python API

```python
import numpy as np
import open3d as o3d

from multicam_icp_calib import RegistrationConfig, calibrate

pcd_camera1 = o3d.io.read_point_cloud("camera_1_workspace.ply")
pcd_camera2 = o3d.io.read_point_cloud("camera_2_workspace.ply")
T_base_camera1 = np.eye(4)  # Replace with the trusted base <- camera 1 pose.

config = RegistrationConfig(
    voxel_size=0.008,
    global_voxel_size=0.025,
    refinement="point_to_plane",
)
result = calibrate(
    pcd_camera1,
    pcd_camera2,
    config=config,
    return_result=True,
)

T_camera1_camera2 = result.transformation
T_base_camera2 = T_base_camera1 @ T_camera1_camera2

print(result.fitness, result.inlier_rmse)
```

By default, `calibrate()` returns a NumPy transform matrix. Pass
`return_result=True` to also receive registration metrics and point counts.

## Documentation

- [ROS 2 setup and calibration](docs/ros2_setup.md)
- [Transform conventions](docs/transforms.md)
- [Parameter tuning](docs/tuning.md)
- [Troubleshooting](docs/troubleshooting.md)

Contributions, tested hardware configurations, bug reports, and documentation
improvements are welcome.

## License

[MIT](LICENSE).