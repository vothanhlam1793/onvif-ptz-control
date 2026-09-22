"""Collect physical PTZ rotations and attempt intrinsic self-calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from tools.onvif.intrinsic_calibration import calibrate_intrinsics, estimate_rotation_homography
from tools.onvif.motion_observer import decode_gray
from tools.onvif.ptz_tool import PtzTool


def collect(output_dir: Path, speed: float, durations: list[float], settle_s: float):
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)
    settle_s = max(1.2, settle_s)
    tool = PtzTool.from_environment()
    homographies = []
    samples = []
    try:
        device = tool.connect()
        tool.stop()
        sample_index = 0
        for duration_s in durations:
            for direction in ("left", "right", "up", "down"):
                sample_index += 1
                before = tool.take_snapshot(640, 360)
                tool.nudge(direction, speed=speed, duration_s=duration_s)
                time.sleep(settle_s)
                after = tool.take_snapshot(640, 360)
                time.sleep(settle_s)
                stopped = tool.take_snapshot(640, 360)

                prefix = f"{sample_index:02d}-{direction}-{duration_s:.2f}"
                (output_dir / f"{prefix}-before.jpg").write_bytes(before)
                (output_dir / f"{prefix}-after.jpg").write_bytes(after)
                (output_dir / f"{prefix}-stopped.jpg").write_bytes(stopped)

                evidence = estimate_rotation_homography(decode_gray(before), decode_gray(after))
                sample = {
                    "direction": direction,
                    "speed": speed,
                    "duration_s": duration_s,
                    **{key: value for key, value in evidence.items() if key != "homography"},
                }
                samples.append(sample)
                if evidence.get("reliable"):
                    homographies.append(evidence["homography"])

        calibration = calibrate_intrinsics(
            homographies,
            (640, 360),
            bootstrap_samples=60,
        )
        return {
            "device": device,
            "speed": speed,
            "durations": durations,
            "samples": samples,
            "accepted_homographies": len(homographies),
            "calibration": calibration,
        }
    finally:
        try:
            tool.stop()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/ptz-intrinsics"))
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--durations", type=float, nargs="+", default=[0.2, 0.25, 0.3])
    parser.add_argument("--settle", type=float, default=1.2)
    args = parser.parse_args()
    report = collect(args.output, args.speed, args.durations, args.settle)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "accepted_homographies": report["accepted_homographies"],
        "calibration": report["calibration"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["calibration"].get("reliable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
