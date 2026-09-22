"""Self-calibrate camera intrinsics from multiple pure-rotation homographies."""

from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np


_SYMMETRIC_BASIS = (
    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64),
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64),
)


def estimate_rotation_homography(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """Estimate a scene homography without requiring camera intrinsics."""
    if before.ndim != 2 or after.ndim != 2 or before.shape != after.shape:
        return {"reliable": False, "reason": "invalid_or_mismatched_frames"}

    height, width = before.shape
    orb = cv2.ORB_create(nfeatures=1800, fastThreshold=10)
    before_keypoints, before_descriptors = orb.detectAndCompute(before, None)
    after_keypoints, after_descriptors = orb.detectAndCompute(after, None)
    if before_descriptors is None or after_descriptors is None:
        return {"reliable": False, "reason": "insufficient_features"}

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        before_descriptors,
        after_descriptors,
        k=2,
    )
    matches = [
        first for pair in pairs if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(matches) < 24:
        return {"reliable": False, "reason": "too_few_feature_matches", "matches": len(matches)}

    source = np.float64([before_keypoints[item.queryIdx].pt for item in matches])
    destination = np.float64([after_keypoints[item.trainIdx].pt for item in matches])
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 2.5)
    if homography is None or mask is None:
        return {"reliable": False, "reason": "homography_failed", "matches": len(matches)}

    inliers = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 24:
        return {
            "reliable": False,
            "reason": "too_few_global_inliers",
            "matches": len(matches),
            "inliers": inlier_count,
        }

    source_inliers = source[inliers]
    destination_inliers = destination[inliers]
    coverage_x = float(np.ptp(source_inliers[:, 0]) / width)
    coverage_y = float(np.ptp(source_inliers[:, 1]) / height)
    coverage_area = coverage_x * coverage_y
    inlier_ratio = inlier_count / len(matches)
    if coverage_x < 0.2 or coverage_y < 0.2 or coverage_area < 0.06 or inlier_ratio < 0.35:
        return {
            "reliable": False,
            "reason": "insufficient_global_coverage",
            "matches": len(matches),
            "inliers": inlier_count,
            "inlier_ratio": round(inlier_ratio, 3),
            "coverage_area": round(coverage_area, 3),
        }

    projected = cv2.perspectiveTransform(
        source_inliers.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    reprojection_error = np.linalg.norm(projected - destination_inliers, axis=1)
    return {
        "reliable": True,
        "homography": homography / homography[2, 2],
        "matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": round(inlier_ratio, 3),
        "coverage_area": round(coverage_area, 3),
        "median_reprojection_error_px": round(float(np.median(reprojection_error)), 3),
    }


def _constraint_matrix(homographies: Iterable[np.ndarray]) -> np.ndarray:
    rows = []
    upper_triangle = np.triu_indices(3)
    for raw_homography in homographies:
        homography = np.asarray(raw_homography, dtype=np.float64)
        determinant = float(np.linalg.det(homography))
        if not np.isfinite(determinant) or determinant <= 0.0:
            continue
        homography = homography / np.cbrt(determinant)
        transformed = [
            homography.T @ basis @ homography - basis
            for basis in _SYMMETRIC_BASIS
        ]
        for row, column in zip(*upper_triangle):
            rows.append([matrix[row, column] for matrix in transformed])
    return np.asarray(rows, dtype=np.float64)


def _orthogonality_residuals(
    parameters: np.ndarray,
    homographies: list[np.ndarray],
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    fx = math.exp(float(parameters[0]))
    fy = math.exp(float(parameters[1]))
    cx = float(parameters[2]) * width
    cy = float(parameters[3]) * height
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(intrinsic)
    residuals = []
    triangle = np.triu_indices(3)
    for homography in homographies:
        normalized = inverse @ homography @ intrinsic
        determinant = float(np.linalg.det(normalized))
        if determinant <= 0.0:
            residuals.extend([10.0] * 6)
            continue
        normalized /= np.cbrt(determinant)
        error = normalized.T @ normalized - np.eye(3)
        residuals.extend(error[triangle])
    return np.asarray(residuals, dtype=np.float64)


def _nonlinear_intrinsic_solution(
    homographies: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    width, height = image_size
    starts = []
    for focal_scale in (0.45, 0.7, 1.0, 1.5, 2.2):
        for center_x, center_y in ((0.5, 0.5), (0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6)):
            starts.append(np.array([
                math.log(width * focal_scale),
                math.log(width * focal_scale),
                center_x,
                center_y,
            ], dtype=np.float64))

    lower = np.array([math.log(width * 0.15), math.log(height * 0.15), 0.0, 0.0])
    upper = np.array([math.log(width * 8.0), math.log(height * 8.0), 1.0, 1.0])
    best_parameters = starts[0]
    best_cost = float("inf")
    finite_steps = np.array([1e-4, 1e-4, 1e-5, 1e-5])

    for start in starts:
        parameters = np.clip(start, lower, upper)
        damping = 1e-3
        residual = _orthogonality_residuals(parameters, homographies, image_size)
        cost = float(np.mean(residual * residual))
        for _ in range(80):
            jacobian_columns = []
            for index, step in enumerate(finite_steps):
                shifted = parameters.copy()
                shifted[index] += step
                shifted_residual = _orthogonality_residuals(shifted, homographies, image_size)
                jacobian_columns.append((shifted_residual - residual) / step)
            jacobian = np.column_stack(jacobian_columns)
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            try:
                delta = np.linalg.solve(normal + damping * np.eye(4), -gradient)
            except np.linalg.LinAlgError:
                break
            candidate = np.clip(parameters + delta, lower, upper)
            candidate_residual = _orthogonality_residuals(candidate, homographies, image_size)
            candidate_cost = float(np.mean(candidate_residual * candidate_residual))
            if candidate_cost < cost:
                parameters = candidate
                residual = candidate_residual
                if abs(cost - candidate_cost) < 1e-14:
                    cost = candidate_cost
                    break
                cost = candidate_cost
                damping = max(1e-9, damping * 0.3)
            else:
                damping = min(1e6, damping * 10.0)
        if cost < best_cost:
            best_cost = cost
            best_parameters = parameters

    fx = math.exp(float(best_parameters[0]))
    fy = math.exp(float(best_parameters[1]))
    intrinsic = np.array([
        [fx, 0.0, float(best_parameters[2]) * width],
        [0.0, fy, float(best_parameters[3]) * height],
        [0.0, 0.0, 1.0],
    ])
    return intrinsic, best_cost


def _solve_intrinsic_matrix(
    homographies: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    if len(homographies) < 3:
        return {"reliable": False, "reason": "too_few_homographies"}

    constraints = _constraint_matrix(homographies)
    if constraints.shape[0] < 18:
        return {"reliable": False, "reason": "too_few_valid_homographies"}

    _, singular_values, right_t = np.linalg.svd(constraints)
    intrinsic, nonlinear_cost = _nonlinear_intrinsic_solution(homographies, image_size)

    nullspace_gap = float(singular_values[-2] / max(singular_values[-1], 1e-12))
    return {
        "reliable": True,
        "intrinsic_matrix": intrinsic,
        "nullspace_gap": nullspace_gap,
        "constraint_residual": float(singular_values[-1] / singular_values[0]),
        "nonlinear_cost": nonlinear_cost,
    }


def _rotation_from_homography(homography: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, float]:
    normalized = np.linalg.inv(intrinsic) @ homography @ intrinsic
    determinant = float(np.linalg.det(normalized))
    if determinant <= 0.0:
        raise ValueError("invalid normalized homography")
    normalized /= np.cbrt(determinant)
    left, _, right_t = np.linalg.svd(normalized)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    error = float(np.linalg.norm(normalized - rotation, ord="fro"))
    return rotation, error


def calibrate_intrinsics(
    homographies: list[np.ndarray],
    image_size: tuple[int, int],
    bootstrap_samples: int = 100,
    random_seed: int = 41,
) -> dict[str, Any]:
    """Recover K and FOV from rotations, rejecting unstable solutions."""
    solution = _solve_intrinsic_matrix(homographies, image_size)
    if not solution.get("reliable"):
        return solution

    intrinsic = solution["intrinsic_matrix"]
    width, height = image_size
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0 or not np.all(np.isfinite(intrinsic)):
        return {"reliable": False, "reason": "invalid_intrinsic_solution"}

    orthogonality_errors = []
    rotations = []
    for homography in homographies:
        try:
            scene_rotation, error = _rotation_from_homography(homography, intrinsic)
        except (ValueError, np.linalg.LinAlgError):
            continue
        camera_rotation = scene_rotation.T
        rotation_vector, _ = cv2.Rodrigues(camera_rotation)
        rotations.append(np.degrees(rotation_vector.ravel()).tolist())
        orthogonality_errors.append(error)
    if len(rotations) < 3:
        return {"reliable": False, "reason": "too_few_valid_rotations"}

    random = np.random.default_rng(random_seed)
    bootstrap = []
    subset_size = max(3, int(math.ceil(len(homographies) * 0.75)))
    for _ in range(bootstrap_samples):
        indices = random.choice(len(homographies), size=subset_size, replace=False)
        candidate = _solve_intrinsic_matrix(
            [homographies[index] for index in indices],
            image_size,
        )
        if candidate.get("reliable"):
            matrix = candidate["intrinsic_matrix"]
            bootstrap.append([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]])
    if len(bootstrap) < max(20, bootstrap_samples // 2):
        return {
            "reliable": False,
            "reason": "unstable_bootstrap",
            "successful_bootstrap_samples": len(bootstrap),
        }

    bootstrap_array = np.asarray(bootstrap)
    bootstrap_median = np.median(bootstrap_array, axis=0)
    bootstrap_mad = 1.4826 * np.median(np.abs(bootstrap_array - bootstrap_median), axis=0)
    relative_focal_uncertainty = max(bootstrap_mad[0] / fx, bootstrap_mad[1] / fy)
    principal_uncertainty = max(bootstrap_mad[2] / width, bootstrap_mad[3] / height)
    principal_offset = math.hypot(
        (cx - (width - 1.0) / 2.0) / width,
        (cy - (height - 1.0) / 2.0) / height,
    )

    hfov_deg = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    reliable = (
        solution["nullspace_gap"] >= 10.0
        and relative_focal_uncertainty <= 0.05
        and principal_uncertainty <= 0.05
        and principal_offset <= 0.2
        and float(np.median(orthogonality_errors)) <= 0.02
    )
    return {
        "reliable": reliable,
        "reason": None if reliable else "intrinsic_solution_not_stable",
        "intrinsic_matrix": intrinsic.tolist(),
        "fx_px": round(fx, 3),
        "fy_px": round(fy, 3),
        "cx_px": round(cx, 3),
        "cy_px": round(cy, 3),
        "hfov_deg": round(hfov_deg, 3),
        "vfov_deg": round(vfov_deg, 3),
        "nullspace_gap": round(solution["nullspace_gap"], 3),
        "constraint_residual": solution["constraint_residual"],
        "nonlinear_cost": solution["nonlinear_cost"],
        "median_orthogonality_error": round(float(np.median(orthogonality_errors)), 6),
        "principal_offset_ratio": round(principal_offset, 6),
        "bootstrap_mad": {
            "fx_px": round(float(bootstrap_mad[0]), 3),
            "fy_px": round(float(bootstrap_mad[1]), 3),
            "cx_px": round(float(bootstrap_mad[2]), 3),
            "cy_px": round(float(bootstrap_mad[3]), 3),
        },
        "successful_bootstrap_samples": len(bootstrap),
        "rotation_vectors_deg": rotations,
    }
