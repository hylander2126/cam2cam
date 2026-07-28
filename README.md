# multicam_icp_calib

Targetless extrinsic calibration for two overlapping RGB-D cameras, built as a
standalone Python tool with optional ROS 2 integration.

`multicam_icp_calib` estimates camera 2's pose relative to an already-calibrated
camera 1 by registering their point clouds. It uses Open3D, FPFH descriptors,
global RANSAC (with Fast Global Registration as a fallback), and multiscale
point-to-plane or colored ICP. It does not require a calibration board, GPU,
Torch, or a learned model.

The project is intentionally split into two layers:

```text
multicam_icp_calib/                         Python core, acquisition, GUI, API
└── ros2_templates/multicam_icp_calib_bringup/
                                             Optional ROS 2 launch integration
```

Use the Python package on its own with Open3D point clouds or direct Intel
RealSense capture. Use the ROS 2 integration when the cameras already belong to
a robot TF tree and publish `sensor_msgs/msg/PointCloud2`.

## Project status and validation scope

This is an early-stage engineering tool released for inspection, adaptation,
and testing. It is not a metrology-certified calibration system and does not
provide a formal accuracy guarantee.

The implementation targets a Linux/ROS 2 workflow using Intel RealSense
D400-series RGB-D cameras. The capture code contains explicit handling for the
RealSense D405's short working range and avoids depth-to-color resampling that
can introduce model-specific artifacts. The exact camera models and firmware
used during development have not yet been recorded in this repository. The
repository also does not yet contain a controlled benchmark, ground-truth
accuracy study, or a complete camera/firmware compatibility matrix. Accordingly:

- direct hardware capture should currently be considered Intel
  RealSense-specific;
- other RGB-D cameras may be used through Open3D point clouds or ROS 2
  `PointCloud2`, but have not been validated here;
- the D405-oriented `0.07–1.00 m` depth default is a starting point, not a
  statement of guaranteed sensor performance;
- ROS frame names, topics, and RealSense launch arguments vary by wrapper
  release and robot stack—inspect your running system instead of copying names
  blindly;
- ICP fitness and inlier RMSE describe agreement between the captured clouds;
  they are not independent proof of extrinsic accuracy.

The current development environment is Ubuntu 24.04, Python 3.12, and ROS 2
Jazzy. The package metadata allows Python 3.10 and newer, but the repository
does not yet include continuous-integration coverage across every supported
Python or ROS release.

If you use a different camera model, ROS distribution, operating system, or
RealSense firmware version, treat that as a new validation configuration and
report what worked.

## What the tool does

- Captures colored point clouds from two directly connected RealSense cameras.
- Receives two live ROS 2 `PointCloud2` messages and reads the corresponding TF
  transforms.
- Accepts a known `base -> camera_1` transform manually or from ROS TF.
- Finds a coarse camera-to-camera alignment using FPFH and RANSAC.
- Falls back to Fast Global Registration when RANSAC produces no valid pose.
- Refines the pose at multiple voxel scales with geometric or colored ICP.
- Rejects results below configured fitness or above configured RMSE thresholds.
- Checks for nearly collinear geometry and warns about nearly planar scenes.
- Displays the individual clouds and registered overlay for visual review.
- Exports transform data and a ROS 2 static-transform launch file.

The tool calibrates one new camera at a time. It does not estimate camera
intrinsics, synchronize hardware clocks, continuously track a moving camera, or
perform full multi-camera bundle adjustment.

## Calibration model

Camera 1 is the trusted reference. Camera 2 is the camera being calibrated:

```text
T_camera1_camera2 = register(camera_2_cloud, camera_1_cloud)
T_base_camera2    = T_base_camera1 @ T_camera1_camera2
```

Transform names follow `T_target_source`:

```text
p_target = T_target_source @ p_source
```

For ROS 2, point clouds normally arrive in depth optical frames, while the
desired persistent transform usually ends at the physical camera link:

```text
base
└── camera_2_link                   <- generated calibration
    └── camera_2_depth_frame        <- camera driver
        └── camera_2_depth_optical_frame
```

The ROS workflow uses the camera driver's internal
`camera_2_link -> camera_2_depth_optical_frame` transform to convert the
registration result into `base -> camera_2_link`.

## Requirements

### Scene and mounting

- Two rigidly mounted depth cameras with meaningful overlapping views.
- A static scene with non-coplanar geometry such as corners, boxes, machinery,
  or objects at several depths.
- A trusted pose for camera 1 relative to the desired base frame.
- Depth data expressed in metres.
- Stationary cameras, robot, and scene during capture.

A floor or blank wall alone is weakly constrained. Moving people, robot motion,
reflective or transparent objects, sunlight affecting active-IR sensors, flying
depth pixels, and small shared crops can all produce plausible-looking but
incorrect registration.

### Standalone Python

- Linux
- Python 3.10 or newer
- Tkinter for the GUI
- NumPy and Open3D
- `pyrealsense2` for direct RealSense acquisition
- `ttkbootstrap` for the optional themed GUI

On Ubuntu:

```bash
sudo apt update
sudo apt install python3-venv python3-tk
```

### ROS 2 integration

- A sourced ROS 2 installation
- `rclpy`, `tf2_ros`, `sensor_msgs`, `launch`, and `launch_ros`
- Point-cloud topics for both cameras
- A connected TF chain from `base` to camera 1's cloud frame
- For the supplied bringup templates, Intel's `realsense2_camera` wrapper

For the current ROS 2 Jazzy development environment:

```bash
sudo apt install \
  ros-jazzy-realsense2-camera \
  ros-jazzy-tf2-ros
```

Package names and launch arguments may differ on other ROS distributions.

## Install

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/hylander2126/cam2cam.git
cd cam2cam
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install only the reusable registration core:

```bash
python3 -m pip install -e .
```

Install the GUI and direct RealSense backend:

```bash
python3 -m pip install -e '.[all]'
```

For ROS 2 use, system-installed ROS Python modules must remain visible. One
practical approach is to create the environment with:

```bash
python3 -m venv --system-site-packages .venv
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python3 -m pip install -e '.[all]'
```

Mixing system and virtual-environment packages can introduce version conflicts.
If your ROS installation uses a different Python version from the environment,
use the ROS-provided interpreter or install the package in a compatible
workspace.

Launch the GUI:

```bash
python3 -m multicam_icp_calib.gui
```

or:

```bash
multicam-icp-calib-gui
```

## Standalone workflow

Direct capture currently supports Intel RealSense devices through
`pyrealsense2`.

1. Stop ROS camera drivers or other processes that own either device.
2. Start the GUI and refresh the device list.
3. Select camera 1, which already has a trusted base transform.
4. Select camera 2, which will be calibrated.
5. Enter or import `T_base_camera1`.
6. Set depth limits appropriate for each sensor and scene.
7. Capture and inspect both clouds.
8. Run FPFH global registration plus multiscale ICP, or provide an approximate
   camera 2 XYZ/RPY pose when global matching is unreliable.
9. Inspect the registered overlay and quality metrics.
10. Export the accepted result.

Direct capture preserves each cloud in its native depth optical frame. The
standalone result therefore ends at the configured camera 2 optical frame:

```text
base -> camera_2_depth_optical_frame
```

Do not relabel that matrix as `base -> camera_2_link`. Converting it requires
the sensor's actual link-to-optical transform.

The direct backend requests depth and RGB at `640 × 360`, 30 FPS, warms up for
30 frames, and accumulates three frames by default. A device that does not
support that stream profile will fail to start; these values are currently
code defaults rather than GUI-configurable stream settings.

## Recommended ROS 2 workflow

The ROS path is the best fit when camera 1 is already calibrated in a robot
stack and both cameras publish point clouds and internal TF frames.

### 1. Add the bringup template to a workspace

```bash
mkdir -p ~/ros2_ws/src
cp -r \
  ros2_templates/multicam_icp_calib_bringup \
  ~/ros2_ws/src/

cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select multicam_icp_calib_bringup
source install/setup.bash
```

The template is deliberately separate from the Python package:

```text
multicam_icp_calib_bringup/
├── CMakeLists.txt
├── package.xml
└── launch/
    ├── camera_2_calibration.launch.py
    ├── camera_2_dummy_tf.launch.py
    └── dual_realsense_calibration.launch.py
```

### 2. Inspect devices, topics, and frames

For RealSense serial numbers:

```bash
rs-enumerate-devices | grep "Serial Number"
```

The supplied RealSense launch templates expect numeric serials with a leading
underscore, for example `_207322251310`. Confirm this convention against your
installed `realsense2_camera` wrapper.

Inspect your actual point-cloud topics:

```bash
ros2 topic list -t | grep PointCloud2
```

Read the frame ID carried by a cloud:

```bash
ros2 topic echo /YOUR/POINTS_TOPIC --field header --once
```

Verify camera 1's trusted chain:

```bash
ros2 run tf2_ros tf2_echo base CAMERA_1_CLOUD_FRAME
```

### 3A. Camera 1 is already launched by the robot

Launch only camera 2 and a temporary identity transform:

```bash
ros2 launch multicam_icp_calib_bringup \
  camera_2_calibration.launch.py \
  camera2_serial:=_CAMERA_2_SERIAL \
  base_frame:=base \
  camera2_name:=camera_2 \
  camera2_namespace:=camera_2 \
  camera2_link_frame:=camera_2_link
```

### 3B. Neither camera is running

```bash
ros2 launch multicam_icp_calib_bringup \
  dual_realsense_calibration.launch.py \
  camera1_serial:=_CAMERA_1_SERIAL \
  camera2_serial:=_CAMERA_2_SERIAL \
  base_frame:=base
```

This launch file does not create camera 1's real calibration. Another publisher
must still provide the trusted `base -> camera_1` chain.

The temporary identity transform only connects camera 2's factory TF subtree
so its link-to-optical transform can be queried. It is not calibration data and
must never be mistaken for the final result.

Depending on wrapper configuration, topics may resemble:

```text
/camera_1/camera_1/depth/color/points
/camera_2/camera_2/depth/color/points
```

Select topics from the GUI's discovered list rather than assuming these exact
names.

### 4. Capture and register

Start the GUI from a shell with ROS, the workspace, and Python environment
sourced:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source /path/to/cam2cam/.venv/bin/activate
python3 -m multicam_icp_calib.gui
```

In the GUI:

1. Open **ROS & TF**.
2. Set the base frame and click **Connect ROS TF**.
3. Select both `PointCloud2` topics.
4. Verify camera 1 and camera 2 cloud frame IDs after capture.
5. Select camera 2's physical link frame.
6. Capture while the robot, cameras, and scene remain stationary.
7. Inspect the individual clouds.
8. Run registration and inspect the overlay and metrics.
9. Export **ROS 2 launch.py** only after accepting the result.

The GUI subscribes concurrently and takes the next message from each topic, but
it does not perform hardware timestamp synchronization. For dynamic scenes or
moving platforms, use synchronized acquisition upstream.

### 5. Install the calibrated transform

Copy the exported file into the bringup package:

```bash
cp /path/to/camera_2_tf.launch.py \
  ~/ros2_ws/src/multicam_icp_calib_bringup/launch/

cd ~/ros2_ws
colcon build --packages-select multicam_icp_calib_bringup
source install/setup.bash
```

Stop the temporary identity publisher before publishing the calibrated edge.
Never run both publishers simultaneously: they claim the same TF relationship
and can make downstream behavior nondeterministic.

Relaunch camera 2 without the dummy publisher:

```bash
ros2 launch multicam_icp_calib_bringup \
  camera_2_calibration.launch.py \
  camera2_serial:=_CAMERA_2_SERIAL \
  publish_dummy_tf:=false
```

Then launch the generated transform:

```bash
ros2 launch multicam_icp_calib_bringup camera_2_tf.launch.py
```

### 6. Validate before robot operation

```bash
ros2 run tf2_ros tf2_echo base camera_2_link
```

In RViz:

1. Set the fixed frame to `base`.
2. Display both point clouds in contrasting colors.
3. Inspect corners, edges, and object surfaces throughout the overlap.
4. Repeat capture and calibration to assess repeatability.
5. If practical, compare against an independent physical measurement or
   target-based calibration.

Do not move a robot based solely on a visually plausible overlay.

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

By default, `calibrate()` returns the transform as a NumPy matrix. Pass
`return_result=True` to receive a `RegistrationResult` containing the matrix,
global and refined fitness/RMSE values, and point counts.

## Registration pipeline

1. Remove invalid values and apply capture-time depth limits.
2. Voxel-downsample the clouds.
3. Check whether scene geometry is degenerate.
4. Estimate surface normals and compute FPFH descriptors.
5. Find a global pose with RANSAC; try Fast Global Registration only if RANSAC
   yields no valid result.
6. Refine at coarse, medium, and fine scales.
7. Reject the pose when configured quality or seed-deviation checks fail.

With an approximate pose, the tool can skip global feature matching and use
multiscale ICP from that seed. Seed checks reduce—but cannot eliminate—the risk
of convergence to a wrong local minimum.

## Starting parameters

These are heuristics, not universal calibration settings:

| Setting | Initial value | Meaning |
|---|---:|---|
| ICP voxel | `0.008 m` | Fine registration scale |
| Global voxel | `0.025 m` | FPFH/RANSAC scale |
| Camera 1 depth | `0.10–3.00 m` | General D400-series starting crop |
| Camera 2 depth | `0.07–1.00 m` | D405-oriented starting crop |
| Refinement | `point_to_plane` | Geometry-only default |
| Minimum fitness | `0.20` | Implementation default |
| Maximum refined RMSE | `1 × ICP voxel` | Implementation default |

Tune the crop and voxel sizes to sensor noise, working distance, scene scale,
and overlap. Colored ICP requires valid RGB values on both clouds and can be
misled by exposure differences, lighting changes, or repeated textures.

## Troubleshooting

### `Device or resource busy`

Only one process can normally own a RealSense device. Stop its ROS camera
driver, or use the ROS `PointCloud2` workflow instead of direct capture.

### A `PointCloud2` topic is missing

For the RealSense wrapper, confirm that depth, color, and point-cloud output are
enabled. Exact parameter names can vary by wrapper release; commonly used
arguments include:

```text
pointcloud.enable:=true
enable_depth:=true
enable_color:=true
```

### TF lookup fails

- Wait for static and dynamic TF publishers to initialize.
- Read the cloud's actual `header.frame_id`.
- Confirm camera 1 connects to `base`.
- Confirm camera 2's selected link is an ancestor of its cloud frame.
- Check for namespace and `camera_name` differences.

### Registration has low fitness

- Increase shared non-coplanar geometry.
- Remove unmatched background with depth limits.
- Keep the scene static and capture both clouds close together in time.
- Confirm camera 1 and camera 2 topics were not reversed.
- Try an approximate pose if global feature matching is ambiguous.

### RMSE is low but fitness is low

A small shared patch may align closely while most of the source has no
correspondence. Improve the common crop and scene geometry rather than simply
lowering the acceptance threshold.

### The overlay looks convincing but the transform is wrong

Repeated geometry, planar scenes, symmetry, and limited overlap can produce a
wrong local minimum. Inspect the full overlap, repeat the calibration, and use
an independent validation method.

## Safety and responsible use

- Treat every generated transform as untrusted until independently reviewed.
- Keep the robot stationary during capture and initial validation.
- Validate in RViz and repeat the process to check repeatability.
- Recalibrate after moving a camera, changing its mount, or changing a relevant
  frame convention.
- Preserve transform metadata and quality metrics with deployed calibration
  files when traceability matters.
- Add workspace margins appropriate to the observed calibration uncertainty.
- This software is provided without warranty under the MIT License.

## Contributing

Useful contributions include tested camera/firmware combinations, reproducible
datasets, accuracy comparisons against target-based methods, ROS distribution
reports, improved synchronization, automated tests, and documentation fixes.
Please avoid reporting only “it worked”; include the sensor models, stream
settings, working distance, scene, software versions, and validation method.

## License

[MIT](LICENSE).
