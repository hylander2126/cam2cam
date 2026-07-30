# Troubleshooting

## `Device or resource busy`

Only one process can normally own a RealSense device. Stop its ROS camera
driver or use the ROS `PointCloud2` workflow instead of direct capture.

## A `PointCloud2` topic is missing

Confirm that depth, color, and point-cloud output are enabled. Exact parameter
names vary by RealSense wrapper release; commonly used arguments include:

```text
pointcloud.enable:=true
enable_depth:=true
enable_color:=true
```

Inspect the topics actually published by your system:

```bash
ros2 topic list -t | grep PointCloud2
```

## TF lookup fails

- Wait for static and dynamic TF publishers to initialize.
- Read the cloud's actual `header.frame_id`.
- Confirm camera 1 connects to the requested base frame.
- Confirm camera 2's selected link is an ancestor of its cloud frame.
- Check for namespace and `camera_name` differences.

See [Transform conventions](transforms.md) for the expected frame relationships.

## Registration has low fitness

- Increase shared non-planar geometry.
- Remove unmatched background with depth limits.
- Keep the scene static and capture both clouds close together in time.
- Confirm the camera 1 and camera 2 topics were not reversed.
- Try an approximate pose when global feature matching is ambiguous.

## RMSE is low but fitness is low

A small shared patch may align closely while most of the source cloud has no
correspondence. Improve the common crop and scene geometry instead of simply
lowering the acceptance threshold.

## The overlay looks convincing but the transform is wrong

Repeated geometry, planar scenes, symmetry, and limited overlap can create an
incorrect local minimum. Inspect the full overlap, repeat the calibration, and
compare against an independent validation method.

## A direct-capture camera fails to start

The direct backend currently requests depth and RGB at `640 × 360`, 30 FPS,
warms up for 30 frames, and accumulates three frames by default. A camera that
does not support this stream profile will fail to start.

## ROS Python imports fail inside the virtual environment

ROS Python modules are normally installed by the operating system. Create the
environment with access to system packages:

```bash
python3 -m venv --system-site-packages .venv
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If ROS uses a different Python version from the environment, use the
ROS-provided interpreter or install the package in a compatible workspace.
