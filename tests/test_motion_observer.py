import math
import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.onvif.motion_observer import estimate_relative_rotation, observe_nudge


class FakeSnapshotTool:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.nudges = []

    def take_snapshot(self, width, height):
        return next(self.snapshots)

    def nudge(self, direction, speed, duration_s):
        self.nudges.append((direction, speed, duration_s))


class MotionObserverTests(unittest.TestCase):
    width = 640
    height = 360
    hfov = 78.0
    vfov = 44.0

    def setUp(self):
        random = np.random.default_rng(73)
        self.scene = random.integers(0, 256, size=(self.height, self.width), dtype=np.uint8)
        self.scene = cv2.GaussianBlur(self.scene, (3, 3), 0)

    def intrinsic_matrix(self):
        fx = self.width / (2.0 * math.tan(math.radians(self.hfov) / 2.0))
        fy = self.height / (2.0 * math.tan(math.radians(self.vfov) / 2.0))
        return np.float64([
            [fx, 0.0, (self.width - 1.0) / 2.0],
            [0.0, fy, (self.height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ])

    def warp_for_camera_rotation(self, pan_deg=0.0, tilt_deg=0.0):
        pan = math.radians(pan_deg)
        tilt = math.radians(tilt_deg)
        scene_pan = np.float64([
            [math.cos(pan), 0.0, -math.sin(pan)],
            [0.0, 1.0, 0.0],
            [math.sin(pan), 0.0, math.cos(pan)],
        ])
        scene_tilt = np.float64([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt), math.sin(tilt)],
            [0.0, -math.sin(tilt), math.cos(tilt)],
        ])
        intrinsic = self.intrinsic_matrix()
        homography = intrinsic @ scene_tilt @ scene_pan @ np.linalg.inv(intrinsic)
        return cv2.warpPerspective(self.scene, homography, (self.width, self.height))

    def encode(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        return encoded.tobytes()

    def test_measures_known_pan_rotation(self):
        result = estimate_relative_rotation(
            self.scene,
            self.warp_for_camera_rotation(pan_deg=6.0),
            self.intrinsic_matrix(),
        )

        self.assertTrue(result["reliable"])
        self.assertAlmostEqual(6.0, result["pan_deg"], delta=0.25)
        self.assertLess(abs(result["tilt_deg"]), 0.25)
        self.assertLess(result["pan_uncertainty_deg"], 0.25)

    def test_measures_known_tilt_rotation(self):
        result = estimate_relative_rotation(
            self.scene,
            self.warp_for_camera_rotation(tilt_deg=4.0),
            self.intrinsic_matrix(),
        )

        self.assertTrue(result["reliable"])
        self.assertAlmostEqual(4.0, result["tilt_deg"], delta=0.25)
        self.assertLess(abs(result["pan_deg"]), 0.25)
        self.assertLess(result["tilt_uncertainty_deg"], 0.25)

    def test_rejects_featureless_frames(self):
        blank = np.zeros((self.height, self.width), dtype=np.uint8)

        result = estimate_relative_rotation(blank, blank, self.intrinsic_matrix())

        self.assertFalse(result["reliable"])
        self.assertEqual("insufficient_features", result["reason"])

    def test_rejects_motion_confined_to_small_image_region(self):
        before = np.zeros((self.height, self.width), dtype=np.uint8)
        patch = self.scene[100:160, 230:360]
        before[100:160, 230:360] = patch
        after = np.zeros_like(before)
        after[110:170, 250:380] = patch

        result = estimate_relative_rotation(before, after, self.intrinsic_matrix())

        self.assertFalse(result["reliable"])

    def test_rejects_invalid_intrinsic_matrix(self):
        result = estimate_relative_rotation(self.scene, self.scene, np.zeros((3, 3)))

        self.assertFalse(result["reliable"])
        self.assertEqual("invalid_intrinsic_matrix", result["reason"])

    def test_observe_nudge_measures_command_and_confirmed_stop(self):
        moved = self.warp_for_camera_rotation(pan_deg=-6.0)
        tool = FakeSnapshotTool([self.encode(self.scene), self.encode(moved), self.encode(moved)])

        with patch("tools.onvif.motion_observer.time.sleep") as sleep:
            result = observe_nudge(tool, "left", 0.2, 0.3, self.intrinsic_matrix())

        self.assertTrue(result["passed"])
        self.assertEqual("MOVED", result["status"])
        self.assertAlmostEqual(-6.0, result["rotation"]["pan_deg"], delta=0.3)
        self.assertLess(result["stopped_angle_deg"], 0.2)
        self.assertEqual([("left", 0.2, 0.3)], tool.nudges)
        self.assertEqual([call(1.2), call(1.2)], sleep.call_args_list)

    def test_observe_nudge_rejects_motion_continuing_after_stop(self):
        moved = self.warp_for_camera_rotation(pan_deg=6.0)
        drifting = self.warp_for_camera_rotation(pan_deg=8.0)
        tool = FakeSnapshotTool([self.encode(self.scene), self.encode(moved), self.encode(drifting)])

        with patch("tools.onvif.motion_observer.time.sleep"):
            result = observe_nudge(tool, "right", 0.2, 0.3, self.intrinsic_matrix())

        self.assertFalse(result["passed"])
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertGreater(result["stopped_angle_deg"], 1.0)


if __name__ == "__main__":
    unittest.main()
