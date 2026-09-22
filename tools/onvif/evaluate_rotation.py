"""Run repeated physical PTZ nudges and report snapshot-derived angles."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from tools.onvif.motion_observer import observe_nudge
from tools.onvif.ptz_tool import PtzTool


def summarize(samples: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    axis = "pan" if direction in ("left", "right") else "tilt"
    angles = [abs(float(item["rotation"][f"{axis}_deg"])) for item in samples if item["passed"]]
    uncertainties = [float(item["rotation"][f"{axis}_uncertainty_deg"]) for item in samples if item["passed"]]
    return {
        "passed_samples": len(angles),
        "total_samples": len(samples),
        "mean_angle_deg": round(statistics.mean(angles), 3) if angles else None,
        "stdev_angle_deg": round(statistics.stdev(angles), 3) if len(angles) > 1 else 0.0 if angles else None,
        "mean_uncertainty_deg": round(statistics.mean(uncertainties), 3) if uncertainties else None,
    }


def evaluate(
    cycles: int,
    speed: float,
    duration_s: float,
    settle_s: float,
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = PtzTool.from_environment()
    report: dict[str, Any] = {
        "cycles": cycles,
        "speed": speed,
        "duration_s": duration_s,
        "intrinsic_matrix": intrinsic_matrix.tolist(),
        "distortion_coefficients": distortion_coefficients.tolist(),
        "samples": {direction: [] for direction in ("left", "right", "up", "down")},
    }

    try:
        report["device"] = tool.connect()
        tool.stop()
        for cycle in range(1, cycles + 1):
            for direction in ("left", "right", "up", "down"):
                result = observe_nudge(
                    tool,
                    direction,
                    speed,
                    duration_s,
                    intrinsic_matrix,
                    distortion_coefficients,
                    settle_s=settle_s,
                )
                snapshots = result.pop("snapshots")
                for stage, image in snapshots.items():
                    (output_dir / f"cycle-{cycle}-{direction}-{stage}.jpg").write_bytes(image)
                report["samples"][direction].append(result)

        report["summary"] = {
            direction: summarize(samples, direction)
            for direction, samples in report["samples"].items()
        }
        report["passed"] = all(
            item["passed"]
            for samples in report["samples"].values()
            for item in samples
        )
        return report
    finally:
        try:
            tool.stop()
        except Exception as exc:
            report["final_stop_error"] = str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--settle", type=float, default=1.2)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/ptz-rotation"))
    args = parser.parse_args()
    calibration_document = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = calibration_document.get("calibration", calibration_document)
    if not calibration.get("reliable") or not calibration.get("intrinsic_matrix"):
        parser.error("calibration file does not contain a reliable intrinsic matrix")
    intrinsic_matrix = np.asarray(calibration["intrinsic_matrix"], dtype=np.float64)
    distortion_coefficients = np.asarray(
        calibration.get("distortion_coefficients", []),
        dtype=np.float64,
    )
    report = evaluate(
        args.cycles,
        args.speed,
        args.duration,
        args.settle,
        intrinsic_matrix,
        distortion_coefficients,
        args.output,
    )
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report.get("passed"), "summary": report.get("summary")}, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
