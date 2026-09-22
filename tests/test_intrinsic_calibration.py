import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.onvif.intrinsic_calibration import calibrate_intrinsics


def rotation_x(degrees):
    angle = math.radians(degrees)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])


def rotation_y(degrees):
    angle = math.radians(degrees)
    return np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])


class IntrinsicCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.height = 360
        self.intrinsic = np.array([
            [410.0, 0.0, 326.0],
            [0.0, 425.0, 176.0],
            [0.0, 0.0, 1.0],
        ])
        inverse = np.linalg.inv(self.intrinsic)
        rotations = [
            rotation_y(-6.0), rotation_y(8.0), rotation_y(-13.0), rotation_y(17.0),
            rotation_x(-5.0), rotation_x(7.0), rotation_x(-11.0), rotation_x(15.0),
            rotation_x(5.0) @ rotation_y(7.0),
            rotation_x(-8.0) @ rotation_y(-9.0),
        ]
        self.homographies = [self.intrinsic @ rotation @ inverse for rotation in rotations]

    def test_recovers_intrinsics_from_unknown_rotations(self):
        result = calibrate_intrinsics(
            self.homographies,
            (self.width, self.height),
            bootstrap_samples=40,
        )

        self.assertTrue(result["reliable"], result)
        self.assertAlmostEqual(410.0, result["fx_px"], delta=0.01)
        self.assertAlmostEqual(425.0, result["fy_px"], delta=0.01)
        self.assertAlmostEqual(326.0, result["cx_px"], delta=0.01)
        self.assertAlmostEqual(176.0, result["cy_px"], delta=0.01)

    def test_rejects_single_axis_degenerate_rotations(self):
        result = calibrate_intrinsics(
            self.homographies[:4],
            (self.width, self.height),
            bootstrap_samples=20,
        )

        self.assertFalse(result["reliable"])

    def test_rejects_too_few_homographies(self):
        result = calibrate_intrinsics(self.homographies[:2], (self.width, self.height))

        self.assertFalse(result["reliable"])
        self.assertEqual("too_few_homographies", result["reason"])


if __name__ == "__main__":
    unittest.main()
