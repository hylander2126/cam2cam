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
â””â”€â”€ camera_2_link                   <- generated calibration
    â””â”€â”€ camera_2_depth_frame        <- camera driver
        â””â”€â”€ camera_2_depth_optical_frame
```

The ROS workflow uses the camera driver's internal
`camera_2_link -> camera_2_depth_optical_frame` transform to convert the
registration result into `base -> camera_2_link`.

Direct standalone capture preserves each cloud in its native optical frame.
Its result therefore ends at the configured camera 2 optical frame:

```text
base -> camera_2_depth_optical_frame
```

Do not relabel this matrix as `base -> camera_2_link`. Converting between them
requires the sensor's actual link-to-optical transform.

## Before exporting

- Confirm which camera is the trusted reference.
- Verify the source and target frame names.
- Check that camera 1 connects to the intended base frame.
- Confirm camera 2's physical link is an ancestor of its cloud frame.
- Inspect the resulting transform with `tf2_echo` and in RViz.