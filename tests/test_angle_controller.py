import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.onvif.angle_controller import RelativeAngleController


class FakeTool:
    def __init__(self):
        self.moves = []
        self.stop_count = 0
        frame = np.random.default_rng(4).integers(0, 256, size=(90, 160), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        self.snapshot = encoded.tobytes()

    def take_snapshot(self, width, height):
        return self.snapshot

    def start_move(self, direction, speed, timeout_s):
        self.moves.append((direction, speed, timeout_s))

    def stop(self):
        self.stop_count += 1


def measurement(pan=0.0, tilt=0.0, reliable=True, reason=None):
    return {
        "reliable": reliable,
        "reason": reason,
        "pan_deg": pan,
        "tilt_deg": tilt,
        "pan_uncertainty_deg": 0.2,
        "tilt_uncertainty_deg": 0.2,
        "inliers": 100,
    }


class RelativeAngleControllerTests(unittest.TestCase):
    def setUp(self):
        self.tool = FakeTool()
        self.controller = RelativeAngleController(
            self.tool,
            np.eye(3),
            np.zeros(5),
            (160, 90),
        )

    def test_converges_from_measured_steps(self):
        with patch.object(
            self.controller,
            "_measure",
            side_effect=[measurement(pan=6.0), measurement(pan=5.0)],
        ), patch("tools.onvif.angle_controller.time.sleep"):
            result = self.controller.rotate_relative_deg("pan", 12.0)

        self.assertTrue(result["ok"])
        self.assertEqual("TARGET_REACHED", result["status"])
        self.assertEqual(11.0, result["measured_deg"])
        self.assertEqual(["right", "right"], [item[0] for item in self.tool.moves])

    def test_corrects_overshoot_in_opposite_direction(self):
        with patch.object(
            self.controller,
            "_measure",
            side_effect=[measurement(pan=11.0), measurement(pan=-2.0)],
        ), patch("tools.onvif.angle_controller.time.sleep"):
            result = self.controller.rotate_relative_deg("pan", 8.0)

        self.assertTrue(result["ok"])
        self.assertEqual(9.0, result["measured_deg"])
        self.assertEqual(["right", "left"], [item[0] for item in self.tool.moves])

    def test_coarse_step_does_not_set_mechanical_resolution(self):
        with patch.object(
            self.controller,
            "_measure",
            side_effect=[measurement(pan=-12.7), measurement(pan=-3.2)],
        ), patch("tools.onvif.angle_controller.time.sleep"):
            result = self.controller.rotate_relative_deg("pan", -15.0)

        self.assertTrue(result["ok"])
        self.assertEqual(-15.9, result["measured_deg"])
        self.assertEqual(2, len(result["steps"]))

    def test_stops_on_unreliable_measurement(self):
        with patch.object(
            self.controller,
            "_measure",
            return_value=measurement(reliable=False, reason="too_few_feature_matches"),
        ), patch("tools.onvif.angle_controller.time.sleep"):
            result = self.controller.rotate_relative_deg("tilt", 10.0)

        self.assertFalse(result["ok"])
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertEqual("too_few_feature_matches", result["reason"])
        self.assertGreaterEqual(self.tool.stop_count, 2)

    def test_does_not_pulse_when_minimum_step_would_increase_error(self):
        self.controller.minimum_pulse_deg = {"up": 6.4}

        with patch.object(self.controller, "_measure") as measure:
            result = self.controller.rotate_relative_deg("tilt", 2.5, tolerance_deg=2.0)

        self.assertFalse(result["ok"])
        self.assertEqual("MECHANICAL_RESOLUTION_LIMIT", result["status"])
        self.assertEqual([], self.tool.moves)
        measure.assert_not_called()

    def test_tilt_always_uses_minimum_safe_pulse(self):
        with patch.object(
            self.controller,
            "_measure",
            side_effect=[measurement(tilt=6.5), measurement(tilt=6.5)],
        ), patch("tools.onvif.angle_controller.time.sleep"):
            result = self.controller.rotate_relative_deg("tilt", 14.0)

        self.assertTrue(result["ok"])
        self.assertEqual(13.0, result["measured_deg"])
        self.assertTrue(all(move[1] == 0.15 for move in self.tool.moves))
        self.assertTrue(all(step["duration_s"] == 0.05 for step in result["steps"]))

    def test_rejects_cross_axis_motion(self):
        with patch.object(
            self.controller,
            "_measure",
            return_value=measurement(pan=3.0, tilt=5.0),
        ), patch("tools.onvif.angle_controller.time.sleep"):
            result = self.controller.rotate_relative_deg("pan", 10.0)

        self.assertFalse(result["ok"])
        self.assertEqual("motion_axis_or_direction_mismatch", result["reason"])


if __name__ == "__main__":
    unittest.main()
