"""Calibrate camera intrinsics from checkerboard snapshots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


INNER_CORNERS = (9, 6)


def generate_target(path: Path, square_px: int = 140, margin_px: int = 140) -> Path:
    columns = INNER_CORNERS[0] + 1
    rows = INNER_CORNERS[1] + 1
    board = np.full(
        (rows * square_px + margin_px * 2, columns * square_px + margin_px * 2),
        255,
        dtype=np.uint8,
    )
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                y0 = margin_px + row * square_px
                x0 = margin_px + column * square_px
                board[y0:y0 + square_px, x0:x0 + square_px] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), board):
        raise RuntimeError(f"could not write calibration target: {path}")
    return path


def generate_pdf_target(path: Path, square_mm: float = 18.0) -> Path:
    """Generate a vector A4 target; print at 100% without fit-to-page scaling."""
    columns = INNER_CORNERS[0] + 1
    rows = INNER_CORNERS[1] + 1
    points_per_mm = 72.0 / 25.4
    page_width = 210.0 * points_per_mm
    page_height = 297.0 * points_per_mm
    square = square_mm * points_per_mm
    origin_x = (page_width - columns * square) / 2.0
    origin_y = (page_height - rows * square) / 2.0
    commands = ["0 0 0 rg"]
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                x = origin_x + column * square
                y = origin_y + (rows - row - 1) * square
                commands.append(f"{x:.3f} {y:.3f} {square:.3f} {square:.3f} re f")
    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.3f} {page_height:.3f}] "
            "/Resources << >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(document)
    return path


def detect_corners(frame: np.ndarray) -> np.ndarray | None:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        frame,
        INNER_CORNERS,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    )
    return corners.reshape(-1, 2).astype(np.float32) if found else None


def checkerboard_object_points() -> np.ndarray:
    points = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float32)
    points[:, :2] = np.mgrid[
        0:INNER_CORNERS[0],
        0:INNER_CORNERS[1],
    ].T.reshape(-1, 2)
    return points


def estimate_checkerboard_rotation(
    before: np.ndarray,
    after: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> dict[str, Any]:
    """Measure camera rotation from the fixed checkerboard pose in two frames."""
    before_corners = detect_corners(before)
    after_corners = detect_corners(after)
    if before_corners is None or after_corners is None:
        return {"reliable": False, "reason": "checkerboard_not_detected"}

    object_points = checkerboard_object_points()
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    distortion = np.asarray(distortion_coefficients, dtype=np.float64)
    rotations = []
    reprojection_errors = []
    for corners in (before_corners, after_corners):
        solved, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            corners,
            intrinsic,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not solved:
            return {"reliable": False, "reason": "checkerboard_pose_failed"}
        rotation, _ = cv2.Rodrigues(rotation_vector)
        rotations.append(rotation)
        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            intrinsic,
            distortion,
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
        reprojection_errors.append(float(np.sqrt(np.mean(error * error))))

    scene_rotation = rotations[1] @ rotations[0].T
    camera_rotation = scene_rotation.T
    relative_vector, _ = cv2.Rodrigues(camera_rotation)
    relative_degrees = np.degrees(relative_vector.ravel())
    reliable = bool(max(reprojection_errors) <= 0.8)
    return {
        "reliable": reliable,
        "reason": None if reliable else "checkerboard_pose_reprojection_too_high",
        "tilt_deg": round(float(relative_degrees[0]), 4),
        "pan_deg": round(float(relative_degrees[1]), 4),
        "roll_deg": round(float(relative_degrees[2]), 4),
        "pose_reprojection_error_px": [round(value, 4) for value in reprojection_errors],
    }


def _calibrate(object_points, image_points, image_size):
    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9),
    )


def calibrate_checkerboard_images(image_paths: list[Path]) -> dict[str, Any]:
    object_template = checkerboard_object_points()
    object_points = []
    image_points = []
    accepted_paths = []
    rejected_paths = []
    image_size = None

    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            rejected_paths.append(str(path))
            continue
        current_size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected_paths.append(str(path))
            continue
        corners = detect_corners(frame)
        if corners is None:
            rejected_paths.append(str(path))
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted_paths.append(str(path))

    if image_size is None or len(image_points) < 8:
        return {
            "reliable": False,
            "reason": "need_at_least_8_detected_checkerboard_views",
            "accepted_images": accepted_paths,
            "rejected_images": rejected_paths,
        }

    rms, intrinsic, distortion, rotation_vectors, translation_vectors = _calibrate(
        object_points,
        image_points,
        image_size,
    )
    per_view_errors = []
    for object_set, image_set, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(
            object_set,
            rotation,
            translation,
            intrinsic,
            distortion,
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - image_set, axis=1)
        per_view_errors.append(float(np.sqrt(np.mean(error * error))))

    leave_one_out = []
    for excluded in range(len(image_points)):
        subset_objects = [item for index, item in enumerate(object_points) if index != excluded]
        subset_images = [item for index, item in enumerate(image_points) if index != excluded]
        _, candidate, _, _, _ = _calibrate(subset_objects, subset_images, image_size)
        leave_one_out.append([
            candidate[0, 0],
            candidate[1, 1],
            candidate[0, 2],
            candidate[1, 2],
        ])
    leave_one_out = np.asarray(leave_one_out)
    loo_median = np.median(leave_one_out, axis=0)
    loo_mad = 1.4826 * np.median(np.abs(leave_one_out - loo_median), axis=0)

    width, height = image_size
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    principal_offset = math.hypot(
        (cx - (width - 1.0) / 2.0) / width,
        (cy - (height - 1.0) / 2.0) / height,
    )
    focal_uncertainty = max(loo_mad[0] / fx, loo_mad[1] / fy)
    distortion_values = distortion.ravel()
    distortion_plausible = bool(
        len(distortion_values) >= 5
        and abs(float(distortion_values[0])) <= 1.0
        and abs(float(distortion_values[1])) <= 2.0
        and abs(float(distortion_values[2])) <= 0.1
        and abs(float(distortion_values[3])) <= 0.1
        and abs(float(distortion_values[4])) <= 5.0
    )
    reliable = bool(
        float(rms) <= 0.8
        and float(np.median(per_view_errors)) <= 0.6
        and focal_uncertainty <= 0.03
        and principal_offset <= 0.2
        and distortion_plausible
    )
    return {
        "reliable": reliable,
        "reason": None if reliable else "checkerboard_solution_not_stable",
        "image_size": [width, height],
        "intrinsic_matrix": intrinsic.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
        "distortion_plausible": distortion_plausible,
        "fx_px": round(fx, 3),
        "fy_px": round(fy, 3),
        "cx_px": round(cx, 3),
        "cy_px": round(cy, 3),
        "hfov_deg": round(math.degrees(2.0 * math.atan(width / (2.0 * fx))), 3),
        "vfov_deg": round(math.degrees(2.0 * math.atan(height / (2.0 * fy))), 3),
        "rms_reprojection_error_px": round(float(rms), 4),
        "median_view_error_px": round(float(np.median(per_view_errors)), 4),
        "max_view_error_px": round(float(np.max(per_view_errors)), 4),
        "leave_one_out_mad": {
            "fx_px": round(float(loo_mad[0]), 3),
            "fy_px": round(float(loo_mad[1]), 3),
            "cx_px": round(float(loo_mad[2]), 3),
            "cy_px": round(float(loo_mad[3]), 3),
        },
        "principal_offset_ratio": round(principal_offset, 6),
        "accepted_images": accepted_paths,
        "rejected_images": rejected_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-target", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.generate_target:
        generated = (
            generate_pdf_target(args.generate_target)
            if args.generate_target.suffix.lower() == ".pdf"
            else generate_target(args.generate_target)
        )
        print(generated)
        return 0
    if not args.images or not args.output:
        parser.error("--images and --output are required for calibration")
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    result = calibrate_checkerboard_images(paths)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("reliable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
