"""Snapshot-based measurement for relative ONVIF PTZ rotations."""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from tools.onvif.ptz_tool import PtzTool


EXPECTED_ROTATION_SIGN = {
    "left": ("pan", -1.0),
    "right": ("pan", 1.0),
    "up": ("tilt", 1.0),
    "down": ("tilt", -1.0),
}
MIN_SNAPSHOT_SETTLE_S = 1.2


def decode_gray(image_bytes: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError("PTZ_SNAPSHOT_DECODE_FAILED")
    return frame


def estimate_relative_rotation(
    before: np.ndarray,
    after: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate camera Pan/Tilt rotation from matching static scene rays.

    Positive Pan means camera-right and positive Tilt means camera-up, matching
    the ONVIF direction names used by :class:`PtzTool`.
    """
    if before.ndim != 2 or after.ndim != 2 or before.shape != after.shape:
        return {"reliable": False, "reason": "invalid_or_mismatched_frames"}
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        return {"reliable": False, "reason": "invalid_intrinsic_matrix"}
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0 or abs(intrinsic[2, 2]) < 1e-12:
        return {"reliable": False, "reason": "invalid_intrinsic_matrix"}

    height, width = before.shape
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
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

    source = np.float32([before_keypoints[item.queryIdx].pt for item in matches])
    destination = np.float32([after_keypoints[item.trainIdx].pt for item in matches])
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
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

    source = source[inliers]
    destination = destination[inliers]
    coverage_x = float(np.ptp(source[:, 0]) / width)
    coverage_y = float(np.ptp(source[:, 1]) / height)
    coverage_area = coverage_x * coverage_y
    inlier_ratio = inlier_count / len(matches)
    grid_x = np.clip((source[:, 0] * 3 / width).astype(int), 0, 2)
    grid_y = np.clip((source[:, 1] * 2 / height).astype(int), 0, 1)
    occupied_grid_cells = len(set(zip(grid_x.tolist(), grid_y.tolist())))
    if (
        coverage_x < 0.2
        or coverage_y < 0.2
        or coverage_area < 0.06
        or inlier_ratio < 0.18
        or occupied_grid_cells < 4
    ):
        return {
            "reliable": False,
            "reason": "insufficient_global_coverage",
            "matches": len(matches),
            "inliers": inlier_count,
            "inlier_ratio": round(inlier_ratio, 3),
            "coverage": [round(coverage_x, 3), round(coverage_y, 3)],
            "coverage_area": round(coverage_area, 3),
            "occupied_grid_cells": occupied_grid_cells,
        }

    intrinsic = intrinsic / intrinsic[2, 2]
    distortion = (
        np.zeros(5, dtype=np.float64)
        if distortion_coefficients is None
        else np.asarray(distortion_coefficients, dtype=np.float64)
    )
    source_undistorted = cv2.undistortPoints(
        source.reshape(-1, 1, 2), intrinsic, distortion, P=intrinsic,
    ).reshape(-1, 2)
    destination_undistorted = cv2.undistortPoints(
        destination.reshape(-1, 1, 2), intrinsic, distortion, P=intrinsic,
    ).reshape(-1, 2)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    def camera_rays(points: np.ndarray) -> np.ndarray:
        x = (points[:, 0] - cx) / fx
        y = (points[:, 1] - cy) / fy
        rays = np.column_stack((x, y, np.ones_like(x)))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    source_rays = camera_rays(source_undistorted)
    destination_rays = camera_rays(destination_undistorted)
    covariance = source_rays.T @ destination_rays
    left, _, right_t = np.linalg.svd(covariance)
    scene_rotation = right_t.T @ left.T
    if np.linalg.det(scene_rotation) < 0:
        right_t[-1, :] *= -1
        scene_rotation = right_t.T @ left.T

    camera_rotation = scene_rotation.T
    rotation_vector, _ = cv2.Rodrigues(camera_rotation)
    rotation_vector_deg = np.degrees(rotation_vector.ravel())
    tilt_deg = float(rotation_vector_deg[0])
    pan_deg = float(rotation_vector_deg[1])

    predicted_rays = (scene_rotation @ source_rays.T).T
    angular_residual = np.degrees(np.arccos(np.clip(
        np.sum(predicted_rays * destination_rays, axis=1),
        -1.0,
        1.0,
    )))
    residual_median = float(np.median(angular_residual))
    residual_mad = float(np.median(np.abs(angular_residual - residual_median)))
    rotation_uncertainty = residual_median + 1.4826 * residual_mad

    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    reprojection_error = np.linalg.norm(projected - destination, axis=1)
    return {
        "reliable": True,
        "pan_deg": round(pan_deg, 3),
        "tilt_deg": round(tilt_deg, 3),
        "pan_uncertainty_deg": round(rotation_uncertainty, 3),
        "tilt_uncertainty_deg": round(rotation_uncertainty, 3),
        "uncertainty_scope": "feature_fit_only_excludes_fov_error",
        "matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": round(inlier_ratio, 3),
        "coverage": [round(coverage_x, 3), round(coverage_y, 3)],
        "coverage_area": round(coverage_area, 3),
        "occupied_grid_cells": occupied_grid_cells,
        "median_reprojection_error_px": round(float(np.median(reprojection_error)), 3),
        "intrinsic_source": "calibrated_matrix",
        "distortion_corrected": distortion_coefficients is not None,
    }


def observe_nudge(
    tool: PtzTool,
    direction: str,
    speed: float,
    duration_s: float,
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray | None = None,
    settle_s: float = MIN_SNAPSHOT_SETTLE_S,
) -> dict[str, Any]:
    """Execute one Layer-1 nudge and measure its rotation and stopped state."""
    if direction not in EXPECTED_ROTATION_SIGN:
        raise ValueError(f"unsupported observation direction: {direction}")
    settle_s = max(MIN_SNAPSHOT_SETTLE_S, settle_s)

    before = tool.take_snapshot(640, 360)
    tool.nudge(direction, speed=speed, duration_s=duration_s)
    time.sleep(settle_s)
    after = tool.take_snapshot(640, 360)
    time.sleep(settle_s)
    stopped = tool.take_snapshot(640, 360)

    rotation = estimate_relative_rotation(
        decode_gray(before),
        decode_gray(after),
        intrinsic_matrix,
        distortion_coefficients,
    )
    stop_rotation = estimate_relative_rotation(
        decode_gray(after),
        decode_gray(stopped),
        intrinsic_matrix,
        distortion_coefficients,
    )

    axis, expected_sign = EXPECTED_ROTATION_SIGN[direction]
    measured_angle = float(rotation.get(f"{axis}_deg", 0.0))
    cross_axis = "tilt" if axis == "pan" else "pan"
    cross_angle = abs(float(rotation.get(f"{cross_axis}_deg", 0.0)))
    stopped_angle = math.hypot(
        float(stop_rotation.get("pan_deg", 0.0)),
        float(stop_rotation.get("tilt_deg", 0.0)),
    )
    passed = (
        rotation.get("reliable", False)
        and stop_rotation.get("reliable", False)
        and measured_angle * expected_sign > 0.2
        and cross_angle <= max(abs(measured_angle) * 0.35, 0.5)
        and stopped_angle <= 0.2
    )
    return {
        "status": "MOVED" if passed else "INCONCLUSIVE",
        "direction": direction,
        "speed": speed,
        "duration_s": duration_s,
        "axis": axis,
        "rotation": rotation,
        "stop_rotation": stop_rotation,
        "stopped_angle_deg": round(stopped_angle, 3),
        "passed": passed,
        "snapshots": {"before": before, "after": after, "stopped": stopped},
    }
