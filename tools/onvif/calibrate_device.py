"""Onboard a physical ONVIF PTZ camera and create its measured profile."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from core.auto_calibration import load_camera_profile, save_camera_profile
from core.device_identity import camera_identity
from tools.onvif.checkerboard_calibration import calibrate_checkerboard_images, detect_corners
from tools.onvif.motion_observer import decode_gray, estimate_relative_rotation
from tools.onvif.ptz_tool import PtzTool


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTLE_S = 1.2
DIRECTIONS = {
    "left": ("pan", -1.0),
    "right": ("pan", 1.0),
    "up": ("tilt", 1.0),
    "down": ("tilt", -1.0),
}
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}


class DeviceCalibrationError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _pulse(tool: PtzTool, direction: str, speed: float, duration_s: float) -> None:
    tool.start_move(direction, speed=speed, timeout_s=duration_s + 1.0)
    try:
        time.sleep(duration_s)
    finally:
        tool.stop()


def capture_checkerboard_views(
    tool: PtzTool,
    output_dir: Path,
    settle_s: float = DEFAULT_SETTLE_S,
) -> list[Path]:
    """Create diverse target poses while returning after every PTZ offset."""
    settle_s = max(DEFAULT_SETTLE_S, settle_s)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []

    def capture(name: str) -> None:
        image = tool.take_snapshot(1280, 720)
        if detect_corners(decode_gray(image)) is not None:
            path = output_dir / f"{name}.jpg"
            path.write_bytes(image)
            accepted.append(path)

    capture("view-00-base")
    sequence = [
        ("left", 0.10), ("right", 0.10), ("up", 0.10), ("down", 0.10),
        ("left", 0.18), ("right", 0.18), ("up", 0.18), ("down", 0.18),
    ]
    for index, (direction, duration_s) in enumerate(sequence, start=1):
        _pulse(tool, direction, 0.2, duration_s)
        time.sleep(settle_s)
        capture(f"view-{index:02d}-{direction}-{duration_s:.2f}")
        _pulse(tool, OPPOSITE[direction], 0.2, duration_s)
        time.sleep(settle_s)

    if len(accepted) < 8:
        raise DeviceCalibrationError(
            f"CHECKERBOARD_VIEWS_INSUFFICIENT: detected {len(accepted)}, need at least 8"
        )
    return accepted


def characterize_actuator(
    tool: PtzTool,
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    image_size: tuple[int, int],
    repeats: int = 3,
    settle_s: float = DEFAULT_SETTLE_S,
) -> dict[str, Any]:
    """Measure minimum pulse response, direction mapping, and post-Stop drift."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    settle_s = max(DEFAULT_SETTLE_S, settle_s)
    width, height = image_size
    samples: dict[str, list[dict[str, Any]]] = {direction: [] for direction in DIRECTIONS}

    maximum_cycles = repeats * 3
    for _ in range(maximum_cycles):
        for direction, (axis, expected_sign) in DIRECTIONS.items():
            if sum(item["passed"] for item in samples[direction]) >= repeats:
                continue
            before = tool.take_snapshot(width, height)
            _pulse(tool, direction, 0.15, 0.05)
            time.sleep(settle_s)
            after = tool.take_snapshot(width, height)
            time.sleep(settle_s)
            stopped = tool.take_snapshot(width, height)
            movement = estimate_relative_rotation(
                decode_gray(before),
                decode_gray(after),
                intrinsic_matrix,
                distortion_coefficients,
            )
            stop_motion = estimate_relative_rotation(
                decode_gray(after),
                decode_gray(stopped),
                intrinsic_matrix,
                distortion_coefficients,
            )
            measured = float(movement.get(f"{axis}_deg", 0.0))
            cross_axis = "tilt" if axis == "pan" else "pan"
            cross_motion = abs(float(movement.get(f"{cross_axis}_deg", 0.0)))
            stop_drift = math.hypot(
                float(stop_motion.get("pan_deg", 0.0)),
                float(stop_motion.get("tilt_deg", 0.0)),
            )
            passed = bool(
                movement.get("reliable")
                and stop_motion.get("reliable")
                and measured * expected_sign > 0.2
                and cross_motion <= max(abs(measured) * 0.4, 1.0)
                and stop_drift <= 1.0
            )
            samples[direction].append({
                "passed": passed,
                "measured_deg": round(measured, 4),
                "cross_axis_deg": round(cross_motion, 4),
                "stop_drift_deg": round(stop_drift, 4),
                "movement": movement,
                "stop_motion": stop_motion,
            })
        if all(sum(item["passed"] for item in values) >= repeats for values in samples.values()):
            break

    response = {}
    for direction, direction_samples in samples.items():
        valid = [abs(item["measured_deg"]) for item in direction_samples if item["passed"]]
        if len(valid) < repeats:
            raise DeviceCalibrationError(
                f"ACTUATOR_CHARACTERIZATION_FAILED: {direction} has {len(valid)}/{repeats} valid samples"
            )
        response[direction] = {
            "median_deg": round(statistics.median(valid), 4),
            "mean_deg": round(statistics.mean(valid), 4),
            "stdev_deg": round(statistics.stdev(valid), 4) if len(valid) > 1 else 0.0,
            "valid_samples": len(valid),
        }

    pan_resolution = max(response["left"]["median_deg"], response["right"]["median_deg"])
    tilt_resolution = max(response["up"]["median_deg"], response["down"]["median_deg"])
    return {
        "status": "CONFIRMED",
        "method": "visual_minimum_pulse_characterization",
        "command": {"speed": 0.15, "duration_s": 0.05, "snapshot_settle_s": settle_s},
        "minimum_pulse_deg": {
            direction: details["median_deg"] for direction, details in response.items()
        },
        "response": response,
        "recommended_tolerance_deg": {
            "pan": max(5, math.ceil(pan_resolution)),
            "tilt": max(7, math.ceil(tilt_resolution)),
        },
        "samples": samples,
        "updated_at": time.time(),
    }


def build_confirmed_profile(
    tool: PtzTool,
    intrinsic: dict[str, Any],
    actuator: dict[str, Any],
    settle_s: float,
) -> dict[str, Any]:
    client = tool.client
    existing = load_camera_profile(client.camera_key) or {}
    profile = dict(existing)
    profile.update({
        "schema_version": 2,
        "camera_key": client.camera_key,
        "camera_identity": camera_identity(client),
        "device_info": camera_identity(client),
        "stream_signature": {
            "width": intrinsic["image_size"][0],
            "height": intrinsic["image_size"][1],
            "rtsp_ready": bool(tool.rtsp_url),
        },
        "transport": {
            "status": "VALIDATED",
            "snapshot_settle_s": max(DEFAULT_SETTLE_S, settle_s),
        },
        "ptz_layer": {
            "status": "CONFIRMED",
            "continuous_move": True,
            "stop": True,
            "relative_angle": True,
            "absolute_angle": False,
        },
        "ptz_actuator": actuator,
    })
    optical = dict(profile.get("optical", {}))
    optical.update({
        "hfov_deg": intrinsic["hfov_deg"],
        "vfov_deg": intrinsic["vfov_deg"],
        "intrinsic_calibration": {
            "status": "CONFIRMED",
            "method": "checkerboard_9x6_opencv",
            **intrinsic,
            "updated_at": time.time(),
        },
    })
    profile["optical"] = optical
    profile["onboarding_calibration"] = {
        "status": "CONFIRMED",
        "schema_version": 1,
        "completed_at": time.time(),
    }
    return profile


def calibrate_device(
    tool: PtzTool,
    checkerboard_paths: list[Path],
    repeats: int = 3,
    settle_s: float = DEFAULT_SETTLE_S,
) -> dict[str, Any]:
    info = tool.connect()
    tool.stop()
    try:
        intrinsic = calibrate_checkerboard_images(checkerboard_paths)
        if not intrinsic.get("reliable"):
            raise DeviceCalibrationError(
                f"INTRINSIC_CALIBRATION_FAILED: {intrinsic.get('reason')}"
            )
        actuator = characterize_actuator(
            tool,
            np.asarray(intrinsic["intrinsic_matrix"], dtype=np.float64),
            np.asarray(intrinsic["distortion_coefficients"], dtype=np.float64),
            tuple(intrinsic["image_size"]),
            repeats=repeats,
            settle_s=settle_s,
        )
        profile = build_confirmed_profile(tool, intrinsic, actuator, settle_s)
        profile_path = save_camera_profile(tool.client.camera_key, _json_safe(profile))
        return {
            "ok": True,
            "camera_key": tool.client.camera_key,
            "device": info,
            "profile_path": str(profile_path),
            "intrinsic": intrinsic,
            "actuator": actuator,
        }
    finally:
        tool.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate one physical ONVIF PTZ camera")
    parser.add_argument("--checkerboard-images", type=Path)
    parser.add_argument("--capture-checkerboard", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "outputs" / "calibration_evidence")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S)
    args = parser.parse_args()
    if not args.checkerboard_images and not args.capture_checkerboard:
        parser.error("provide --checkerboard-images or --capture-checkerboard")

    load_dotenv()
    tool = PtzTool.from_environment()
    try:
        tool.connect()
        if args.capture_checkerboard:
            camera_dir = args.evidence_dir / tool.client.camera_key / f"session-{int(time.time())}"
            paths = capture_checkerboard_views(tool, camera_dir, args.settle)
        else:
            paths = sorted(
                path for path in args.checkerboard_images.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        report = calibrate_device(tool, paths, repeats=args.repeats, settle_s=args.settle)
        report_path = Path(report["profile_path"]).with_suffix(".calibration-report.json")
        report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "ok": True,
            "camera_key": report["camera_key"],
            "profile_path": report["profile_path"],
            "report_path": str(report_path),
        }, ensure_ascii=False, indent=2))
        return 0
    except (DeviceCalibrationError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    finally:
        try:
            tool.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
