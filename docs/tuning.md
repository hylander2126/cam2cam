# Parameter Tuning

These values are starting points, not universal calibration settings:

| Setting | Initial value | Meaning |
|---|---:|---|
| ICP voxel | `0.008 m` | Fine registration scale |
| Global voxel | `0.025 m` | FPFH/RANSAC scale |
| Camera 1 depth | `0.10–3.00 m` | General D400-series crop |
| Camera 2 depth | `0.07–1.00 m` | D405-oriented crop |
| Refinement | `point_to_plane` | Geometry-only default |
| Minimum fitness | `0.20` | Implementation default |
| Maximum refined RMSE | `1 × ICP voxel` | Implementation default |

Tune depth crops and voxel sizes to the sensor noise, working distance, scene
scale, and amount of overlap.

## Scene selection


Good scenes contain shared geometry at several depths, including corners,
boxes, machinery, and other non-coplanar surfaces.

Avoid relying on:

- A floor or blank wall alone
- Repeated or symmetric geometry
- Reflective or transparent objects
- Moving people or robot links
- Small shared crops
- Flying depth pixels
- Direct sunlight that interferes with active-IR sensors

## Refinement choice

`point_to_plane` is the geometry-only default. Colored ICP may help when both
clouds contain reliable RGB values, but it can be misled by exposure
differences, lighting changes, or repeated textures.

## Approximate-pose initialization

When global feature matching is ambiguous, provide an approximate camera 2
XYZ/RPY pose and start multiscale ICP from that seed. Seed-deviation checks
reduce, but do not eliminate, the possibility of convergence to an incorrect
local minimum.

## Automatic pose-guided consensus

Use **Automatic pose-guided (consensus)** when a single FPFH/RANSAC run is not
repeatable or the scene contains symmetric or repeated geometry. This mode
generates four independent RANSAC hypotheses, an FGR hypothesis, and—when a
rough camera 2 pose is entered—a seeded multiscale-ICP hypothesis. It refines
all candidates identically, scores both cloud directions at one fixed physical
tolerance, and accepts only a uniquely supported pose cluster.

The pose guess is optional and does not need to be an accurate ICP seed. An
estimate within tens of centimetres and tens of degrees is useful because it
acts as a soft plausibility tiebreaker between recurring pose clusters. With no
guess, selection uses recurrence and fixed-tolerance registration quality
alone.

Consensus controls currently use these API defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `consensus_translation_tol_m` | `0.02 m` | Maximum translation separation within one pose cluster |
| `consensus_rotation_tol_deg` | `5°` | Maximum rotation separation within one pose cluster |
| `consensus_min_candidates` | `2` | Minimum independently generated candidates supporting the accepted cluster |

Do not compare raw fitness values obtained with different correspondence
radii. A looser radius can increase fitness without improving the pose; the
consensus scorer therefore uses the same finest-stage evaluation distance for
every candidate.

## Interpreting metrics

- **Fitness** describes how much of the source cloud finds corresponding
  geometry in the target cloud.
- **Inlier RMSE** describes the residual distance among accepted
  correspondences.

Low RMSE alone does not prove a correct calibration. A small shared patch can
align closely while most of the scene remains unmatched. Always inspect the
full overlay and assess repeatability.
