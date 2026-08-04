"""FPFH/RANSAC global registration and local RGB-D refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import open3d as o3d

from .filters import prepare_point_cloud

LOGGER = logging.getLogger(__name__)
RefinementMethod = Literal["point_to_plane", "colored"]
_CONSENSUS_RANSAC_ATTEMPTS = 4


class RegistrationError(RuntimeError):
    """Raised when the data cannot support a trustworthy calibration."""


@dataclass(frozen=True)
class RegistrationConfig:
    """Registration parameters; spatial values use point-cloud units (metres)."""

    voxel_size: float = 0.01
    global_voxel_size: float | None = None
    normal_radius_factor: float = 2.5
    feature_radius_factor: float = 5.0
    ransac_distance_factor: float = 1.5
    icp_distance_factor: float = 0.75
    ransac_iterations: int = 100_000
    ransac_confidence: float = 0.999
    icp_iterations: int = 80
    min_points: int = 100
    min_fitness: float = 0.20
    max_rmse_factor: float = 1.0
    min_geometry_ratio: float = 1.0e-4
    max_seed_translation_deviation_m: float = 0.25
    max_seed_rotation_deviation_deg: float = 45.0
    consensus_translation_tol_m: float = 0.02
    consensus_rotation_tol_deg: float = 5.0
    consensus_min_candidates: int = 2
    refinement: RefinementMethod = "point_to_plane"

    def __post_init__(self) -> None:
        for name in (
            "voxel_size",
            "normal_radius_factor",
            "feature_radius_factor",
            "ransac_distance_factor",
            "icp_distance_factor",
            "max_rmse_factor",
            "min_geometry_ratio",
            "max_seed_translation_deviation_m",
            "max_seed_rotation_deviation_deg",
            "consensus_translation_tol_m",
            "consensus_rotation_tol_deg",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.global_voxel_size is not None and (
            not np.isfinite(self.global_voxel_size)
            or self.global_voxel_size <= 0
        ):
            raise ValueError("global_voxel_size must be finite and positive")
        if self.min_points < 4:
            raise ValueError("min_points must be at least 4")
        if self.consensus_min_candidates < 1:
            raise ValueError("consensus_min_candidates must be at least 1")
        if self.ransac_iterations < 1 or self.icp_iterations < 1:
            raise ValueError("iteration counts must be positive")
        if not 0 < self.ransac_confidence <= 1:
            raise ValueError("ransac_confidence must be in (0, 1]")
        if not 0 <= self.min_fitness <= 1:
            raise ValueError("min_fitness must be in [0, 1]")
        if self.refinement not in ("point_to_plane", "colored"):
            raise ValueError("refinement must be 'point_to_plane' or 'colored'")


@dataclass(frozen=True)
class RegistrationResult:
    """The source-to-target transform and registration quality diagnostics."""

    transformation: np.ndarray
    global_fitness: float
    global_inlier_rmse: float
    fitness: float
    inlier_rmse: float
    source_points: int
    target_points: int


@dataclass(frozen=True)
class Candidate:
    """A refined pose hypothesis scored at a shared physical tolerance."""

    transformation: np.ndarray
    source_label: str
    bidirectional_fitness: float
    bidirectional_rmse: float


def _pose_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return translation and rotation separating two homogeneous poses."""
    a = np.asarray(T_a, dtype=np.float64)
    b = np.asarray(T_b, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("poses must be 4x4 matrices")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("poses must contain only finite values")
    delta = np.linalg.inv(a) @ b
    translation_delta = float(np.linalg.norm(delta[:3, 3]))
    rotation_cosine = np.clip(
        (np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0
    )
    rotation_delta_deg = float(np.rad2deg(np.arccos(rotation_cosine)))
    return translation_delta, rotation_delta_deg


def score_candidate(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    eval_distance: float,
) -> tuple[float, float]:
    """Score both registration directions at one caller-supplied tolerance."""
    if not np.isfinite(eval_distance) or eval_distance <= 0:
        raise ValueError("eval_distance must be finite and positive")
    transform = np.asarray(transformation, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transformation must be a finite 4x4 matrix")
    forward = o3d.pipelines.registration.evaluate_registration(
        source, target, eval_distance, transform
    )
    backward = o3d.pipelines.registration.evaluate_registration(
        target, source, eval_distance, np.linalg.inv(transform)
    )
    return (
        min(float(forward.fitness), float(backward.fitness)),
        max(float(forward.inlier_rmse), float(backward.inlier_rmse)),
    )


def _check_cloud(
    cloud: o3d.geometry.PointCloud,
    name: str,
    config: RegistrationConfig,
) -> None:
    if not isinstance(cloud, o3d.geometry.PointCloud):
        raise TypeError(f"{name} must be an open3d.geometry.PointCloud")
    points = np.asarray(cloud.points)
    if len(points) < config.min_points:
        raise RegistrationError(
            f"{name} has {len(points)} usable points after filtering; "
            f"at least {config.min_points} are required"
        )
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise RegistrationError(f"{name} contains invalid 3-D coordinates")

    # Reject line-like geometry and warn for planar geometry. A plane can still
    # register, but may leave some degrees of freedom weakly constrained.
    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
    scale = max(float(eigenvalues[-1]), np.finfo(np.float64).eps)
    if eigenvalues[1] / scale < config.min_geometry_ratio:
        raise RegistrationError(
            f"{name} is nearly collinear. Capture corners, edges, and objects "
            "at several depths."
        )
    if eigenvalues[0] / scale < config.min_geometry_ratio:
        LOGGER.warning(
            "%s is nearly planar; calibration may be weakly constrained. "
            "Include non-coplanar static geometry if possible.",
            name,
        )


def _estimate_normals(
    cloud: o3d.geometry.PointCloud, config: RegistrationConfig
) -> None:
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.voxel_size * config.normal_radius_factor,
            max_nn=40,
        )
    )
    cloud.normalize_normals()


def _preprocess_for_fpfh(
    cloud: o3d.geometry.PointCloud,
    name: str,
    config: RegistrationConfig,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    downsampled = prepare_point_cloud(cloud, config.voxel_size)
    _check_cloud(downsampled, name, config)
    _estimate_normals(downsampled, config)
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.voxel_size * config.feature_radius_factor,
            max_nn=100,
        ),
    )
    descriptors = np.asarray(feature.data)
    if (
        descriptors.shape != (33, len(downsampled.points))
        or not np.all(np.isfinite(descriptors))
    ):
        raise RegistrationError(f"FPFH extraction failed for {name}")
    if float(np.mean(np.std(descriptors, axis=1))) < 1.0e-8:
        raise RegistrationError(
            f"{name} lacks local geometric texture for reliable FPFH matching"
        )
    return downsampled, feature


def global_ransac_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: RegistrationConfig | None = None,
) -> o3d.pipelines.registration.RegistrationResult:
    """Globally align ``source`` into ``target`` using FPFH and RANSAC."""
    cfg = config or RegistrationConfig()
    global_voxel = cfg.global_voxel_size or cfg.voxel_size
    global_cfg = replace(
        cfg, voxel_size=global_voxel, global_voxel_size=None
    )
    source_down, source_fpfh = _preprocess_for_fpfh(
        source, "source", global_cfg
    )
    target_down, target_fpfh = _preprocess_for_fpfh(
        target, "target", global_cfg
    )
    threshold = global_voxel * cfg.ransac_distance_factor
    LOGGER.info(
        "Global clouds: source=%d, target=%d, voxel=%.4f m",
        len(source_down.points),
        len(target_down.points),
        global_voxel,
    )

    method = "RANSAC"
    result = (
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=threshold,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=False
                )
            ),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    threshold
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                cfg.ransac_iterations, cfg.ransac_confidence
            ),
        )
    )
    if not np.all(np.isfinite(result.transformation)) or result.fitness <= 0:
        LOGGER.warning(
            "RANSAC found no valid pose; trying Fast Global Registration "
            "with the same FPFH features"
        )
        method = "Fast Global Registration"
        result = (
            o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=threshold
                ),
            )
        )
        if (
            not np.all(np.isfinite(result.transformation))
            or result.fitness <= 0
        ):
            raise RegistrationError(
                "RANSAC and Fast Global Registration both found no valid "
                f"alignment using {len(source_down.points)} camera-2 and "
                f"{len(target_down.points)} camera-1 points at "
                f"{global_voxel:.3f} m global voxels. Inspect the captured "
                "clouds for shared geometry."
            )
    LOGGER.info(
        "Global %s: fitness=%.3f, inlier RMSE=%.6f",
        method,
        result.fitness,
        result.inlier_rmse,
    )
    return result


def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    config: RegistrationConfig | None = None,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine a source-to-target estimate with geometric or colored ICP."""
    cfg = config or RegistrationConfig()
    initial = np.asarray(initial_transform, dtype=np.float64)
    if initial.shape != (4, 4) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_transform must be a finite 4x4 matrix")
    if not np.allclose(initial[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7):
        raise ValueError("initial_transform must be a homogeneous transform")

    source_down = prepare_point_cloud(source, cfg.voxel_size)
    target_down = prepare_point_cloud(target, cfg.voxel_size)
    _check_cloud(source_down, "source", cfg)
    _check_cloud(target_down, "target", cfg)
    _estimate_normals(source_down, cfg)
    _estimate_normals(target_down, cfg)

    max_distance = cfg.voxel_size * cfg.icp_distance_factor
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=cfg.icp_iterations
    )
    if cfg.refinement == "colored":
        if not source_down.has_colors() or not target_down.has_colors():
            raise RegistrationError(
                "Colored ICP requires RGB colors on both PointCloud objects"
            )
        result = o3d.pipelines.registration.registration_colored_icp(
            source_down,
            target_down,
            max_distance,
            initial,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria,
        )
    else:
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_distance,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )

    LOGGER.info(
        "Local %s ICP: fitness=%.3f, inlier RMSE=%.6f",
        "colored" if cfg.refinement == "colored" else "point-to-plane",
        result.fitness,
        result.inlier_rmse,
    )
    return result


def multiscale_refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    config: RegistrationConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine without jumping directly from coarse RANSAC to millimetric ICP."""
    coarse_voxel = config.global_voxel_size or max(
        config.voxel_size * 3.0, 0.02
    )
    middle_voxel = float(np.sqrt(coarse_voxel * config.voxel_size))
    stages: list[tuple[str, RegistrationConfig]] = []

    if coarse_voxel > config.voxel_size * 1.05:
        stages.append(
            (
                "coarse",
                replace(
                    config,
                    voxel_size=coarse_voxel,
                    global_voxel_size=None,
                    icp_distance_factor=2.0,
                    icp_iterations=max(40, config.icp_iterations // 2),
                    refinement="point_to_plane",
                ),
            )
        )
    if (
        middle_voxel > config.voxel_size * 1.20
        and middle_voxel < coarse_voxel * 0.90
    ):
        stages.append(
            (
                "medium",
                replace(
                    config,
                    voxel_size=middle_voxel,
                    global_voxel_size=None,
                    icp_distance_factor=1.5,
                    icp_iterations=max(50, config.icp_iterations // 2),
                    refinement="point_to_plane",
                ),
            )
        )
    stages.append(("fine", config))

    transform = np.asarray(initial_transform, dtype=np.float64)
    result = None
    for name, stage_config in stages:
        max_distance = (
            stage_config.voxel_size * stage_config.icp_distance_factor
        )
        if progress_callback:
            progress_callback(
                f"Running {name} ICP: voxel {stage_config.voxel_size:.4f} m, "
                f"correspondence radius {max_distance:.4f} m…"
            )
        result = refine_registration(
            source, target, transform, stage_config
        )
        if result.fitness <= 0:
            raise RegistrationError(
                f"{name.capitalize()} ICP found no correspondences within "
                f"{max_distance:.4f} m"
            )
        transform = result.transformation

    assert result is not None
    return result


def generate_candidates(
    pcd_cam2: o3d.geometry.PointCloud,
    pcd_cam1: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    guess: np.ndarray | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Candidate]:
    """Generate independently initialized poses and score them uniformly."""
    global_voxel = config.global_voxel_size or config.voxel_size
    global_cfg = replace(
        config, voxel_size=global_voxel, global_voxel_size=None
    )
    source_down, source_fpfh = _preprocess_for_fpfh(
        pcd_cam2, "source", global_cfg
    )
    target_down, target_fpfh = _preprocess_for_fpfh(
        pcd_cam1, "target", global_cfg
    )
    threshold = global_voxel * config.ransac_distance_factor
    initial_poses: list[tuple[str, np.ndarray]] = []

    for index in range(_CONSENSUS_RANSAC_ATTEMPTS):
        if progress_callback:
            progress_callback(
                "Generating global RANSAC candidate "
                f"{index + 1} of {_CONSENSUS_RANSAC_ATTEMPTS}…"
            )
        # Open3D exposes no seed argument on this registration call. Separate
        # back-to-back calls advance its random state and can find different
        # local optima in an ambiguous scene.
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=threshold,
                estimation_method=(
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        with_scaling=False
                    )
                ),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        threshold
                    ),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    config.ransac_iterations, config.ransac_confidence
                ),
            )
        )
        if np.all(np.isfinite(result.transformation)) and result.fitness > 0:
            initial_poses.append(
                (f"ransac_seed{index}", np.asarray(result.transformation).copy())
            )

    if progress_callback:
        progress_callback("Generating Fast Global Registration candidate…")
    fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=threshold
        ),
    )
    if np.all(np.isfinite(fgr.transformation)) and fgr.fitness > 0:
        initial_poses.append(("fgr", np.asarray(fgr.transformation).copy()))
    if guess is not None:
        initial_poses.append(
            ("seeded_icp", np.asarray(guess, dtype=np.float64).copy())
        )

    eval_distance = config.voxel_size * config.icp_distance_factor
    candidates: list[Candidate] = []
    errors: list[str] = []
    for label, initial in initial_poses:
        try:
            if progress_callback:
                progress_callback(f"Refining consensus candidate '{label}'…")
            refined = multiscale_refine_registration(
                pcd_cam2,
                pcd_cam1,
                initial,
                config,
                progress_callback,
            )
            fitness, rmse = score_candidate(
                pcd_cam2,
                pcd_cam1,
                refined.transformation,
                eval_distance,
            )
            candidates.append(
                Candidate(
                    transformation=np.asarray(refined.transformation).copy(),
                    source_label=label,
                    bidirectional_fitness=fitness,
                    bidirectional_rmse=rmse,
                )
            )
        except Exception as error:
            LOGGER.warning("Consensus candidate %s failed: %s", label, error)
            errors.append(f"{label}: {error}")
    if not candidates:
        detail = "; ".join(errors) if errors else "no valid global hypotheses"
        raise RegistrationError(f"Consensus generated no usable candidates: {detail}")
    return candidates


def cluster_candidates(
    candidates: list[Candidate],
    translation_tol_m: float,
    rotation_tol_deg: float,
) -> list[list[Candidate]]:
    """Greedily group pose hypotheses around each cluster's first member."""
    if not np.isfinite(translation_tol_m) or translation_tol_m <= 0:
        raise ValueError("translation_tol_m must be finite and positive")
    if not np.isfinite(rotation_tol_deg) or rotation_tol_deg <= 0:
        raise ValueError("rotation_tol_deg must be finite and positive")
    clusters: list[list[Candidate]] = []
    for candidate in candidates:
        for cluster in clusters:
            translation, rotation = _pose_delta(
                cluster[0].transformation, candidate.transformation
            )
            if (
                translation <= translation_tol_m
                and rotation <= rotation_tol_deg
            ):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters


def select_consensus_pose(
    clusters: list[list[Candidate]],
    config: RegistrationConfig,
    guess: np.ndarray | None = None,
) -> Candidate:
    """Select a uniquely recurring, plausible pose cluster."""
    if not clusters or any(not cluster for cluster in clusters):
        raise RegistrationError("Consensus produced no non-empty pose clusters")

    ranked: list[tuple[bool, int, float, list[Candidate]]] = []
    for cluster in clusters:
        plausible = True
        if guess is not None:
            translation, rotation = _pose_delta(
                guess, cluster[0].transformation
            )
            plausible = (
                translation <= config.max_seed_translation_deviation_m
                and rotation <= config.max_seed_rotation_deviation_deg
            )
        ranked.append(
            (
                plausible,
                len(cluster),
                min(candidate.bidirectional_rmse for candidate in cluster),
                cluster,
            )
        )
    ranked.sort(key=lambda item: (-int(item[0]), -item[1], item[2]))
    if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
        raise RegistrationError(
            "Consensus registration is unstable: the two leading pose clusters "
            f"have equal plausibility and support ({ranked[0][1]} candidates "
            "each). Supply a rough camera 2 pose or improve scene geometry."
        )
    plausible, cluster_size, _cluster_rmse, winning_cluster = ranked[0]
    if cluster_size < config.consensus_min_candidates:
        raise RegistrationError(
            "Consensus registration is unstable: the winning pose cluster has "
            f"only {cluster_size} candidate(s), but "
            f"{config.consensus_min_candidates} are required."
        )
    if guess is not None and not plausible:
        raise RegistrationError(
            "Consensus registration found no pose cluster within the configured "
            "translation and rotation limits of the rough camera 2 pose."
        )
    return min(
        winning_cluster,
        key=lambda candidate: (
            -candidate.bidirectional_fitness,
            candidate.bidirectional_rmse,
        ),
    )


def calibrate_consensus(
    pcd_cam1: o3d.geometry.PointCloud,
    pcd_cam2: o3d.geometry.PointCloud,
    config: RegistrationConfig | None = None,
    guess: np.ndarray | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> RegistrationResult:
    """Calibrate from a uniquely recurring cluster of refined candidates."""
    cfg = config or RegistrationConfig()
    candidates = generate_candidates(
        pcd_cam2, pcd_cam1, cfg, guess, progress_callback
    )
    clusters = cluster_candidates(
        candidates,
        cfg.consensus_translation_tol_m,
        cfg.consensus_rotation_tol_deg,
    )
    winner = select_consensus_pose(clusters, cfg, guess)
    max_rmse = cfg.voxel_size * cfg.max_rmse_factor
    if (
        winner.bidirectional_fitness < cfg.min_fitness
        or winner.bidirectional_rmse > max_rmse
    ):
        raise RegistrationError(
            "Consensus pose failed quality checks at the fixed evaluation "
            f"distance: fitness={winner.bidirectional_fitness:.3f} "
            f"(minimum {cfg.min_fitness:.3f}), "
            f"RMSE={winner.bidirectional_rmse:.6f} "
            f"(maximum {max_rmse:.6f})."
        )
    if progress_callback:
        progress_callback(
            f"Consensus selected '{winner.source_label}' from "
            f"{len(candidates)} candidates across {len(clusters)} clusters."
        )
    # Candidate does not retain raw global-stage diagnostics; these fields use
    # its fixed-tolerance bidirectional score so RegistrationResult stays API-
    # compatible with existing GUI/export consumers.
    return RegistrationResult(
        transformation=np.asarray(winner.transformation).copy(),
        global_fitness=winner.bidirectional_fitness,
        global_inlier_rmse=winner.bidirectional_rmse,
        fitness=winner.bidirectional_fitness,
        inlier_rmse=winner.bidirectional_rmse,
        source_points=len(pcd_cam2.points),
        target_points=len(pcd_cam1.points),
    )


def calibrate(
    pcd_cam1: o3d.geometry.PointCloud,
    pcd_cam2: o3d.geometry.PointCloud,
    *,
    config: RegistrationConfig | None = None,
    return_result: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    initial_transform: np.ndarray | None = None,
    use_global_registration: bool = True,
) -> np.ndarray | RegistrationResult:
    """Estimate ``T_cam1_cam2`` from overlapping camera-frame point clouds.

    Camera 2 is the source and camera 1 is the target, so the returned transform
    obeys ``p_cam1 = T_cam1_cam2 @ p_cam2``.
    """
    cfg = config or RegistrationConfig()
    if initial_transform is not None and not use_global_registration:
        initial = np.asarray(initial_transform, dtype=np.float64)
        if initial.shape != (4, 4) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_transform must be a finite 4x4 matrix")
        coarse_voxel = cfg.global_voxel_size or max(
            cfg.voxel_size * 3.0, 0.02
        )
        if progress_callback:
            progress_callback("Starting multiscale ICP from the approximate pose…")
        refined = multiscale_refine_registration(
            pcd_cam2, pcd_cam1, initial, cfg, progress_callback
        )
        coarse_cfg = replace(
            cfg,
            voxel_size=coarse_voxel,
            global_voxel_size=None,
            icp_distance_factor=2.0,
        )
        seed_quality = o3d.pipelines.registration.evaluate_registration(
            prepare_point_cloud(pcd_cam2, coarse_voxel),
            prepare_point_cloud(pcd_cam1, coarse_voxel),
            coarse_voxel * coarse_cfg.icp_distance_factor,
            initial,
        )
        global_fitness = float(seed_quality.fitness)
        global_rmse = float(seed_quality.inlier_rmse)
    else:
        if progress_callback:
            progress_callback("Extracting FPFH features and running global RANSAC…")
        global_result = global_ransac_registration(pcd_cam2, pcd_cam1, cfg)
        if progress_callback:
            progress_callback(
                f"Global alignment found (fitness {global_result.fitness:.3f}); "
                f"running {cfg.refinement.replace('_', '-')} ICP…"
            )
        refined = multiscale_refine_registration(
            pcd_cam2,
            pcd_cam1,
            global_result.transformation,
            cfg,
            progress_callback,
        )
        global_fitness = float(global_result.fitness)
        global_rmse = float(global_result.inlier_rmse)

    max_rmse = cfg.voxel_size * cfg.max_rmse_factor
    if progress_callback:
        progress_callback("Checking registration quality…")
    if refined.fitness < cfg.min_fitness or refined.inlier_rmse > max_rmse:
        raise RegistrationError(
            "Registration failed quality checks: "
            f"fitness={refined.fitness:.3f} (minimum {cfg.min_fitness:.3f}), "
            f"RMSE={refined.inlier_rmse:.6f} (maximum {max_rmse:.6f}). "
            "Do not use this calibration; improve overlap, cropping, or scene "
            "geometry."
        )
    if initial_transform is not None and not use_global_registration:
        translation_delta, rotation_delta_deg = _pose_delta(
            initial, refined.transformation
        )
        if (
            translation_delta > cfg.max_seed_translation_deviation_m
            or rotation_delta_deg > cfg.max_seed_rotation_deviation_deg
        ):
            raise RegistrationError(
                "Seeded ICP moved implausibly far from the approximate pose: "
                f"{translation_delta:.3f} m and {rotation_delta_deg:.1f}°. "
                "The configured limits are "
                f"{cfg.max_seed_translation_deviation_m:.3f} m and "
                f"{cfg.max_seed_rotation_deviation_deg:.1f}°. Inspect overlap "
                "or correct the initial XYZ/RPY instead of exporting this pose."
            )

    transform = np.asarray(refined.transformation, dtype=np.float64).copy()
    details = RegistrationResult(
        transformation=transform,
        global_fitness=global_fitness,
        global_inlier_rmse=global_rmse,
        fitness=float(refined.fitness),
        inlier_rmse=float(refined.inlier_rmse),
        source_points=len(pcd_cam2.points),
        target_points=len(pcd_cam1.points),
    )
    return details if return_result else transform
