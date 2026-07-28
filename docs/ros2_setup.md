# ROS 2 Setup and Calibration

Use this workflow when camera 1 is already calibrated in a robot stack and both
cameras publish `sensor_msgs/msg/PointCloud2` data and internal TF frames.

## Requirements

- A sourced ROS 2 installation
- `rclpy`, `tf2_ros`, `sensor_msgs`, `launch`, and `launch_ros`
- Point-cloud topics for both cameras
- A connected TF chain from the desired base frame to camera 1's cloud frame
- For the supplied templates, Intel's `realsense2_camera` wrapper

On ROS 2 Jazzy:

```bash
sudo apt install \
  ros-jazzy-realsense2-camera \
  ros-jazzy-tf2-ros
```

Package names and launch arguments may differ on other ROS distributions.

## 1. Add the bringup template

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

The package contains:

```text
multicam_icp_calib_bringup/
â”œâ”€â”€ CMakeLists.txt
â”œâ”€â”€ package.xml
â””â”€â”€ launch/
    â”œâ”€â”€ camera_2_calibration.launch.py
    â”œâ”€â”€ camera_2_dummy_tf.launch.py
    â””â”€â”€ dual_realsense_calibration.launch.py
```

## 2. Inspect devices, topics, and frames

List RealSense serial numbers:

```bash
rs-enumerate-devices | grep "Serial Number"
```

The supplied RealSense templates expect numeric serials with a leading
underscore, such as `_207322251310`. Confirm this convention against your
installed wrapper.

Find point-cloud topics:

```bash
ros2 topic list -t | grep PointCloud2
```

Read a cloud's frame ID:

```bash
ros2 topic echo /YOUR/POINTS_TOPIC --field header --once
```

Verify camera 1's trusted chain:

```bash
ros2 run tf2_ros tf2_echo base CAMERA_1_CLOUD_FRAME
```

## 3. Launch the cameras

If camera 1 is already launched by the robot, launch camera 2 and its temporary
identity transform:

```bash
ros2 launch multicam_icp_calib_bringup \
  camera_2_calibration.launch.py \
  camera2_serial:=_CAMERA_2_SERIAL \
  base_frame:=base \
  camera2_name:=camera_2 \
  camera2_namespace:=camera_2 \
  camera2_link_frame:=camera_2_link
```

If neither camera is running:

```bash
ros2 launch multicam_icp_calib_bringup \
  dual_realsense_calibration.launch.py \
  camera1_serial:=_CAMERA_1_SERIAL \
  camera2_serial:=_CAMERA_2_SERIAL \
  base_frame:=base
```

The dual-camera launch does not create camera 1's real calibration. Another
publisher must provide the trusted `base -> camera_1` chain.

The temporary identity publisher only connects camera 2's factory TF subtree
so the tool can query its link-to-optical transform. It is not calibration
data.

## 4. Capture and register

Start the GUI from a shell with ROS, the workspace, and the Python environment
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
4. Capture and verify both cloud frame IDs.
5. Select camera 2's physical link frame.
6. Run registration and inspect the overlay and metrics.
7. Export **ROS 2 launch.py** after accepting the result.

The GUI takes the next message from each topic concurrently but does not
perform hardware timestamp synchronization. Keep the cameras, robot, and scene
stationary.

## 5. Install the calibrated transform

```bash
cp /path/to/camera_2_tf.launch.py \
  ~/ros2_ws/src/multicam_icp_calib_bringup/launch/

cd ~/ros2_ws
colcon build --packages-select multicam_icp_calib_bringup
source install/setup.bash
```

Stop the temporary identity publisher before publishing the calibrated
transform. Never run both publishers simultaneously because they claim the
same TF relationship.

Relaunch camera 2 without the dummy transform:

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

## 6. Validate

```bash
ros2 run tf2_ros tf2_echo base camera_2_link
```

In RViz:

1. Set the fixed frame to `base`.
2. Display both clouds in contrasting colors.
3. Inspect corners, edges, and surfaces throughout the overlap.
4. Repeat the calibration to assess repeatability.
5. When practical, compare against a physical measurement or target-based
   calibration.

Do not move a robot based only on a visually plausible overlay.