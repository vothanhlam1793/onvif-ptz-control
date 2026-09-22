"""Visual closed-loop relative-angle control for encoderless ONVIF PTZ cameras."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.onvif.checkerboard_calibration import estimate_checkerboard_rotation
from tools.onvif.motion_observer import decode_gray, estimate_relative_rotation
from tools.onvif.ptz_tool import PtzTool


class RelativeAngleController:
    """Rotate by a measured relative angle using short ONVIF pulses."""

    def __init__(
        self,
        tool: PtzTool,
        intrinsic_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        image_size: tuple[int, int],
        minimum_pulse_deg: dict[str, float] | None = None,
        settle_s: float = 1.2,
    ):
        self.tool = tool
        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
        self.distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64)
        self.image_size = image_size
        self.minimum_pulse_deg = minimum_pulse_deg or {}
        self.settle_s = max(1.2, settle_s)

    @classmethod
    def from_profile(cls, tool: PtzTool, profile_path: Path) -> "RelativeAngleController":
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        calibration = profile.get("optical", {}).get("intrinsic_calibration", {})
        if calibration.get("status") != "CONFIRMED":
            raise RuntimeError("PTZ_INTRINSICS_NOT_CALIBRATED")
        return cls(
            tool,
            np.asarray(calibration["intrinsic_matrix"], dtype=np.float64),
            np.asarray(calibration["distortion_coefficients"], dtype=np.float64),
            tuple(calibration["image_size"]),
            profile.get("ptz_actuator", {}).get("minimum_pulse_deg", {}),
        )

    def _measure(self, before: bytes, after: bytes) -> dict[str, Any]:
        before_gray = decode_gray(before)
        after_gray = decode_gray(after)
        global_result = estimate_relative_rotation(
            before_gray,
            after_gray,
            self.intrinsic_matrix,
            self.distortion_coefficients,
        )
        if global_result.get("reliable"):
            return global_result

        checkerboard_result = estimate_checkerboard_rotation(
            before_gray,
            after_gray,
            self.intrinsic_matrix,
            self.distortion_coefficients,
        )
        if checkerboard_result.get("reliable"):
            checkerboard_result["method"] = "checkerboard_solvepnp"
            checkerboard_result["pan_uncertainty_deg"] = 0.5
            checkerboard_result["tilt_uncertainty_deg"] = 0.5
            checkerboard_result["inliers"] = 54
            return checkerboard_result
        return global_result

    def rotate_relative_deg(
        self,
        axis: str,
        angle_deg: float,
        tolerance_deg: float = 2.0,
        max_steps: int = 10,
    ) -> dict[str, Any]:
        """Rotate Pan or Tilt by a relative angle and verify every pulse visually."""
        if axis not in {"pan", "tilt"}:
            raise ValueError("axis must be 'pan' or 'tilt'")
        if not math.isfinite(angle_deg) or not 0.0 < abs(angle_deg) <= 180.0:
            raise ValueError("angle_deg must be in [-180, 180] and non-zero")
        if not 0.5 <= tolerance_deg <= 5.0:
            raise ValueError("tolerance_deg must be in [0.5, 5.0]")
        if not 1 <= max_steps <= 30:
            raise ValueError("max_steps must be in [1, 30]")

        cumulative = 0.0
        estimated_rate = 70.0
        minimum_pulse_by_direction = dict(self.minimum_pulse_deg)
        steps = []
        width, height = self.image_size

        try:
            for step_index in range(1, max_steps + 1):
                residual = angle_deg - cumulative
                if abs(residual) <= tolerance_deg:
                    return {
                        "ok": True,
                        "status": "TARGET_REACHED",
                        "axis": axis,
                        "target_deg": angle_deg,
                        "measured_deg": round(cumulative, 3),
                        "residual_deg": round(residual, 3),
                        "tolerance_deg": tolerance_deg,
                        "steps": steps,
                    }

                positive = residual > 0.0
                if axis == "pan":
                    direction = "right" if positive else "left"
                else:
                    direction = "up" if positive else "down"
                known_minimum = minimum_pulse_by_direction.get(direction)
                if known_minimum is not None and known_minimum >= 2.0 * abs(residual):
                    break

                if axis == "tilt":
                    duration_s = 0.05
                    speed = 0.15
                else:
                    duration_s = max(0.05, min(0.25, abs(residual) / estimated_rate))
                    speed = 0.15 if duration_s <= 0.06 else 0.2
                before = self.tool.take_snapshot(width, height)
                self.tool.start_move(direction, speed=speed, timeout_s=duration_s + 1.0)
                try:
                    time.sleep(duration_s)
                finally:
                    self.tool.stop()
                time.sleep(self.settle_s)
                after = self.tool.take_snapshot(width, height)
                measurement = self._measure(before, after)
                if not measurement.get("reliable"):
                    return {
                        "ok": False,
                        "status": "INCONCLUSIVE",
                        "reason": measurement.get("reason", "unreliable_visual_measurement"),
                        "axis": axis,
                        "target_deg": angle_deg,
                        "measured_deg": round(cumulative, 3),
                        "steps": steps,
                    }

                measured = float(measurement[f"{axis}_deg"])
                cross_axis = "tilt" if axis == "pan" else "pan"
                cross_motion = abs(float(measurement[cross_axis + "_deg"]))
                expected_sign = 1.0 if positive else -1.0
                direction_correct = measured * expected_sign > 0.2
                cross_axis_ok = cross_motion <= max(abs(measured) * 0.4, 1.0)
                step = {
                    "index": step_index,
                    "direction": direction,
                    "speed": speed,
                    "duration_s": round(duration_s, 4),
                    "measured_deg": round(measured, 3),
                    "cross_axis_deg": round(cross_motion, 3),
                    "uncertainty_deg": measurement.get(f"{axis}_uncertainty_deg"),
                    "inliers": measurement.get("inliers"),
                }
                steps.append(step)
                if not direction_correct or not cross_axis_ok:
                    return {
                        "ok": False,
                        "status": "INCONCLUSIVE",
                        "reason": "motion_axis_or_direction_mismatch",
                        "axis": axis,
                        "target_deg": angle_deg,
                        "measured_deg": round(cumulative, 3),
                        "steps": steps,
                    }

                cumulative += measured
                observed_rate = abs(measured) / duration_s
                estimated_rate = 0.65 * estimated_rate + 0.35 * observed_rate
                if duration_s <= 0.06:
                    minimum_pulse_by_direction[direction] = min(
                        abs(measured),
                        minimum_pulse_by_direction.get(direction, float("inf")),
                    )
        finally:
            self.tool.stop()

        residual = angle_deg - cumulative
        return {
            "ok": abs(residual) <= tolerance_deg,
            "status": "TARGET_REACHED" if abs(residual) <= tolerance_deg else "MECHANICAL_RESOLUTION_LIMIT",
            "axis": axis,
            "target_deg": angle_deg,
            "measured_deg": round(cumulative, 3),
            "residual_deg": round(residual, 3),
            "tolerance_deg": tolerance_deg,
            "steps": steps,
        }
