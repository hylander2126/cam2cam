# ROS 2 Setup

## What you need to bring

An existing extrinsic (hand-eye) calibration for camera 1, provided as either:

- a known 6D pose relative to your robot or world frame; or
- an existing ROS 2 TF publisher launch file.

If camera 1 is not calibrated yet, the
[MoveIt hand-eye calibration tutorial](https://moveit.picknik.ai/humble/doc/examples/hand_eye_calibration/hand_eye_calibration_tutorial.html)
describes an approachable GUI-based method.

That trusted camera 1 pose is the only existing calibration `cam2cam` needs.
If it is already published in TF, keep using that publisher. Otherwise, enter
the pose in the supplied two-camera template.

Both camera drivers must publish:

- a `sensor_msgs/msg/PointCloud2` topic
- their normal internal TF, from the camera link to the cloud's optical frame

The template does not launch or configure camera drivers, so it works with an
existing robot stack and is not tied to RealSense.

## 1. Add the TF template

```bash
mkdir -p ~/ros2_ws/src
cp -r ros2_templates/icp_calib_bringup ~/ros2_ws/src/
```

Edit `icp_calib_bringup/launch/camera_transforms.launch.py`:

- If camera 1 already has a TF publisher, set `PUBLISH_CAMERA_1 = False`.
- Otherwise, set camera 1's frame, translation, and quaternion to its known
  `base -> camera_1_link` pose.
- Set the base and camera 2 link frame names.
- Leave camera 2's pose at identity for the calibration run.

Build and launch the template:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select icp_calib_bringup
source install/setup.bash
ros2 launch icp_calib_bringup camera_transforms.launch.py
```

The camera 2 identity is temporary. It attaches the driver's internal TF tree
so `cam2cam` can read the link-to-optical transform; it is not a calibration.

## 2. Calibrate

Start both camera drivers using your normal bringup, then launch the GUI from a
ROS-enabled shell:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source /path/to/cam2cam/.venv/bin/activate
python3 -m icp_calib.gui
```

In the GUI:

1. Under **ROS & TF**, connect to ROS and select both point-cloud topics.
2. Confirm camera 2's physical link frame.
3. Capture the stationary scene and run calibration.
4. Inspect the overlay, then export **ROS 2 launch.py**.

The GUI uses camera 1's trusted TF and camera 2's internal link-to-optical TF
to calculate the final `base -> camera_2_link` pose.

## 3. Make camera 2's result permanent

Stop `camera_transforms.launch.py`. Copy the exported camera 2 translation and
quaternion into `CAMERA_2_XYZ` and `CAMERA_2_XYZW` in that same template, then
rebuild and relaunch it:

```bash
cd ~/ros2_ws
colcon build --packages-select icp_calib_bringup
source install/setup.bash
ros2 launch icp_calib_bringup camera_transforms.launch.py
```

The one template now publishes camera 2's calibrated pose and, only when
needed, camera 1's known pose. Do not also run another publisher for either
same parent/child TF pair.

Check the result:

```bash
ros2 run tf2_ros tf2_echo base camera_2_link
```

In RViz, use `base` as the fixed frame and display both clouds in contrasting
colors. Check several surfaces and viewpoints; a plausible overlay alone is
not sufficient evidence for safety-critical robot motion.
