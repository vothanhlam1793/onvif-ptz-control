"""Physically evaluate deterministic ONVIF PTZ movement using snapshot evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from core.virtual_ptz import VirtualPTZTracker
from tools.onvif.ptz_tool import PtzTool


EXPECTED_SCENE_FLOW = {
    "left": ("x", 1.0),
    "right": ("x", -1.0),
    "up": ("y", 1.0),
    "down": ("y", -1.0),
}


def decode_gray(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError("PTZ_SNAPSHOT_DECODE_FAILED")
    return frame


def motion_metrics(before: bytes, after: bytes) -> dict[str, Any]:
    return VirtualPTZTracker._global_motion_metrics(decode_gray(before), decode_gray(after))


def classify_motion(direction: str, metrics: dict[str, Any], threshold_px: float) -> dict[str, Any]:
    result = dict(metrics)
    if not metrics.get("reliable"):
        result.update({"passed": False, "reason": metrics.get("reason", "unreliable_motion")})
        return result

    dx, dy = metrics["global_vector_px"]
    axis, expected_sign = EXPECTED_SCENE_FLOW[direction]
    component = dx if axis == "x" else dy
    orthogonal = dy if axis == "x" else dx
    alignment = abs(component) / max(abs(component) + abs(orthogonal), 1e-6)
    result.update({
        "axis_alignment": round(alignment, 3),
        "expected_scene_axis": axis,
        "expected_scene_sign": expected_sign,
        "direction_correct": component * expected_sign > 0,
    })
    result["passed"] = (
        metrics["global_motion_px"] >= threshold_px
        and alignment >= 0.7
        and result["direction_correct"]
    )
    if not result["passed"]:
        result["reason"] = "motion_direction_or_magnitude_mismatch"
    return result


def evaluate(speed: float, duration_s: float, settle_s: float, output_dir: Path) -> dict[str, Any]:
    load_dotenv()
    settle_s = max(1.2, settle_s)
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = PtzTool.from_environment()
    report: dict[str, Any] = {
        "speed": speed,
        "duration_s": duration_s,
        "settle_s": settle_s,
        "directions": {},
    }

    try:
        report["device"] = tool.connect()
        tool.stop()
        baseline_a = tool.take_snapshot(640, 360)
        time.sleep(settle_s)
        baseline_b = tool.take_snapshot(640, 360)
        (output_dir / "baseline-a.jpg").write_bytes(baseline_a)
        (output_dir / "baseline-b.jpg").write_bytes(baseline_b)
        baseline = motion_metrics(baseline_a, baseline_b)
        report["baseline"] = baseline
        if not baseline.get("reliable"):
            report.update({"passed": False, "reason": "unreliable_static_baseline"})
            return report

        threshold_px = max(1.0, float(baseline["global_motion_px"]) * 2.5)
        report["motion_threshold_px"] = round(threshold_px, 3)
        pair_starts: dict[str, bytes] = {}
        pair_ends: dict[str, bytes] = {}

        for index, direction in enumerate(("left", "right", "up", "down"), start=1):
            before = tool.take_snapshot(640, 360)
            if direction in ("left", "up"):
                pair_starts[direction] = before

            tool.nudge(direction, speed=speed, duration_s=duration_s)
            time.sleep(settle_s)
            after = tool.take_snapshot(640, 360)
            time.sleep(settle_s)
            stopped = tool.take_snapshot(640, 360)

            prefix = f"{index}-{direction}"
            (output_dir / f"{prefix}-before.jpg").write_bytes(before)
            (output_dir / f"{prefix}-after.jpg").write_bytes(after)
            (output_dir / f"{prefix}-stopped.jpg").write_bytes(stopped)

            movement = classify_motion(direction, motion_metrics(before, after), threshold_px)
            stop_metrics = motion_metrics(after, stopped)
            stopped_ok = (
                stop_metrics.get("reliable", False)
                and float(stop_metrics["global_motion_px"]) < threshold_px
            )
            report["directions"][direction] = {
                "movement": movement,
                "stop": {**stop_metrics, "passed": stopped_ok},
                "passed": movement["passed"] and stopped_ok,
            }
            if direction in ("right", "down"):
                pair_ends[direction] = stopped

        pan_return = motion_metrics(pair_starts["left"], pair_ends["right"])
        tilt_return = motion_metrics(pair_starts["up"], pair_ends["down"])
        report["return_error"] = {"pan": pan_return, "tilt": tilt_return}
        report["passed"] = all(item["passed"] for item in report["directions"].values())
        return report
    finally:
        try:
            tool.stop()
        except Exception as exc:
            report["final_stop_error"] = str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--settle", type=float, default=1.2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/ptz-layer1"))
    args = parser.parse_args()
    report = evaluate(args.speed, args.duration, args.settle, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
