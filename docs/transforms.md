# Transform Conventions

Camera 1 is the trusted reference and camera 2 is the camera being calibrated:

```text
T_camera1_camera2 = register(camera_2_cloud, camera_1_cloud)
T_base_camera2    = T_base_camera1 @ T_camera1_camera2
```

Transform names follow `T_target_source`:

```text
p_target = T_target_source @ p_source
```

In other words, `T_camera1_camera2` maps points expressed in camera 2's frame
into camera 1's frame.

## Optical frames and physical links

ROS 2 point clouds normally arrive in depth optical frames, while the
persistent transform generally ends at the physical camera link:

```text
base
└── camera_2_link                   <- generated calibration
    └── camera_2_depth_frame        <- camera driver
        └── camera_2_depth_optical_frame
```

The ROS workflow uses the camera driver's internal
`camera_2_link -> camera_2_depth_optical_frame` transform to convert the
registration result into `base -> camera_2_link`.

Direct standalone RealSense capture preserves each cloud in its native depth
optical frame for registration. In the default manual workflow, imported poses
target the conventional camera body/link frame (X-forward, Y-left, Z-up).
`cam2cam` composes the colocated RealSense body-to-depth-optical rotation before
registration and converts the result back to the configured camera 2 link:

```text
base -> camera_2_link
```

If the supplied camera 1 pose already targets its depth optical frame, select
**Depth optical frame (Z-forward)** instead. The standalone body frame is
defined at the depth optical origin because ROS `camera_link` itself does not
exist when the ROS driver is not running.

RealSense factory sensor-to-sensor offsets are available directly from
librealsense and can be exported with:

```bash
icp-calib-realsense-extrinsics --serial CAMERA_SERIAL --output extrinsics.json
```

The RealSense ROS convention places `camera_link`, depth, and left IR at the
same origin. Consequently, factory `depth_to_color` translation is important
when converting geometry to the color imager, but it is not an additional
depth-to-`camera_link` offset.

## Before exporting

- Confirm which camera is the trusted reference.
- Verify the source and target frame names.
- Check that camera 1 connects to the intended base frame.
- Confirm camera 2's physical link is an ancestor of its cloud frame.
- Inspect the resulting transform with `tf2_echo` and in RViz.
