import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.virtual_ptz import VirtualPTZTracker


class FakeClient:
    camera_key = "test_mechanical_endstops"
    profile_token = "Profile000"
    ptz_url = "http://camera/onvif/ptz_service"

    def __init__(self):
        self.moves = []
        self.stops = 0

    def continuous_move(self, pan, tilt, *args, **kwargs):
        self.moves.append((pan, tilt))

    def stop(self, *args, **kwargs):
        self.stops += 1


class SimulatedEndstopClient(FakeClient):
    """Camera model with bounded axes; each move advances one calibration nudge."""

    def __init__(self):
        super().__init__()
        self.pan = 3
        self.tilt = 2
        self.pan_bounds = (0, 6)
        self.tilt_bounds = (0, 4)

    def continuous_move(self, pan, tilt, *args, **kwargs):
        super().continuous_move(pan, tilt, *args, **kwargs)
        if pan:
            self.pan = min(self.pan_bounds[1], max(self.pan_bounds[0], self.pan + (1 if pan > 0 else -1)))
        if tilt:
            self.tilt = min(self.tilt_bounds[1], max(self.tilt_bounds[0], self.tilt + (1 if tilt > 0 else -1)))


class MechanicalEndstopTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.tracker = VirtualPTZTracker(self.client, "rtsp://test/stream")

    def test_global_motion_detects_coherent_camera_translation(self):
        before = np.zeros((160, 240), dtype=np.uint8)
        for x in range(20, 220, 30):
            for y in range(20, 140, 30):
                cv2.circle(before, (x, y), 4, 255, -1)
        after = cv2.warpAffine(before, np.float32([[1, 0, 8], [0, 1, 0]]), (240, 160))

        metrics = self.tracker._global_motion_metrics(before, after)

        self.assertTrue(metrics["reliable"])
        self.assertGreater(metrics["global_motion_px"], 6.0)
        self.assertGreater(metrics["coherence"], 0.9)
        self.assertGreater(metrics["inlier_ratio"], 0.9)

    def test_global_motion_tracks_large_camera_translation(self):
        before = np.random.default_rng(21).integers(0, 256, size=(360, 640), dtype=np.uint8)
        before = cv2.GaussianBlur(before, (3, 3), 0)
        after = cv2.warpAffine(before, np.float32([[1, 0, 96], [0, 1, 0]]), (640, 360))

        metrics = self.tracker._global_motion_metrics(before, after)

        self.assertTrue(metrics["reliable"])
        self.assertGreater(metrics["global_motion_px"], 90.0)
        self.assertGreater(metrics["coherence"], 0.9)

    def test_global_motion_ignores_local_object_motion(self):
        before = np.random.default_rng(7).integers(0, 256, size=(180, 260), dtype=np.uint8)
        before = cv2.GaussianBlur(before, (3, 3), 0)
        after = before.copy()
        cv2.rectangle(after, (90, 60), (145, 120), 255, -1)

        metrics = self.tracker._global_motion_metrics(before, after)

        self.assertTrue(metrics["reliable"])
        self.assertLess(metrics["global_motion_px"], 1.0)

    def test_endstop_requires_stillness_and_reverse_recovery(self):
        moving = {"reliable": True, "global_motion_px": 12.0, "coherence": 0.95, "moved": True}
        still = {"reliable": True, "global_motion_px": 0.1, "coherence": 0.0, "moved": False}
        baseline = {"reliable": True, "global_motion_px": 0.1, "coherence": 0.0}

        with patch.object(self.tracker, "_capture_gray_frame", return_value=np.zeros((20, 20), dtype=np.uint8)), \
             patch.object(self.tracker, "_global_motion_metrics", return_value=baseline), \
             patch.object(self.tracker, "_measure_axis_nudge", side_effect=[moving, still, still, still, moving]):
            result = self.tracker._seek_endstop("pan", -1.0, 0.6, 0.45, 12, 3)

        self.assertTrue(result["confirmed"])
        self.assertEqual("CONFIRMED", result["status"])
        self.assertEqual(0.45, result["moving_duration_s"])
        self.assertEqual((-0.6, 0.0), self.client.moves[-1])

    def test_endstop_is_inconclusive_without_reverse_motion(self):
        still = {"reliable": True, "global_motion_px": 0.1, "coherence": 0.0, "moved": False}
        baseline = {"reliable": True, "global_motion_px": 0.1, "coherence": 0.0}

        with patch.object(self.tracker, "_capture_gray_frame", return_value=np.zeros((20, 20), dtype=np.uint8)), \
             patch.object(self.tracker, "_global_motion_metrics", return_value=baseline), \
             patch.object(self.tracker, "_measure_axis_nudge", side_effect=[still, still, still, still]):
            result = self.tracker._seek_endstop("tilt", 1.0, 0.6, 0.45, 12, 3)

        self.assertFalse(result["confirmed"])
        self.assertEqual("no_reverse_recovery", result["reason"])

    def test_calibration_saves_profile_only_after_four_confirmed_bounds(self):
        evidence = {"confirmed": True, "moving_duration_s": 4.5}
        with patch.object(self.tracker, "_seek_endstop", side_effect=[evidence] * 4), \
             patch("core.virtual_ptz.save_camera_profile") as save_profile, \
             patch.object(self.tracker, "load_calibration"):
            result = self.tracker.calibrate_mechanical_endstops(speed=0.6)

        self.assertTrue(result["ok"])
        saved_profile = save_profile.call_args.args[1]
        self.assertEqual("bounded_stops", saved_profile["pan"]["pan_type"])
        self.assertFalse(saved_profile["pan"]["is_360_continuous"])
        self.assertEqual(4.5, saved_profile["pan"]["full_pan_time_sec"])
        self.assertEqual(4.5, saved_profile["tilt"]["full_tilt_time_sec"])
        self.assertEqual(0.6, saved_profile["pan"]["reference_speed"])
        self.assertEqual(0.6, saved_profile["tilt"]["reference_speed"])
        self.assertEqual({"pan": 1.0, "tilt": 1.0}, result["virtual_position"])
        self.assertTrue(self.tracker.is_homed)
        self.assertTrue(self.tracker.mechanical_calibration_confirmed)

    def test_calibration_seeks_all_four_bounds_in_physical_order(self):
        evidence = {"confirmed": True, "moving_duration_s": 1.0}
        with patch.object(self.tracker, "_seek_endstop", side_effect=[evidence] * 4) as seek, \
             patch("core.virtual_ptz.save_camera_profile"), \
             patch.object(self.tracker, "load_calibration"):
            self.tracker.calibrate_mechanical_endstops(speed=0.2, duration_s=0.2)

        calls = [(item.args[0], item.args[1]) for item in seek.call_args_list]
        self.assertEqual([
            ("pan", -1.0),
            ("pan", 1.0),
            ("tilt", -1.0),
            ("tilt", 1.0),
        ], calls)

    def test_calibration_does_not_save_if_any_bound_is_inconclusive(self):
        confirmed = {"confirmed": True, "moving_duration_s": 4.5}
        failed = {"confirmed": False, "status": "INCONCLUSIVE", "reason": "no_reverse_recovery"}

        for failed_index, expected_axis in enumerate(("pan_left", "pan_right", "tilt_bottom", "tilt_top")):
            with self.subTest(axis=expected_axis), \
                 patch.object(self.tracker, "_seek_endstop", side_effect=[confirmed] * failed_index + [failed]), \
                 patch("core.virtual_ptz.save_camera_profile") as save_profile:
                result = self.tracker.calibrate_mechanical_endstops(speed=0.6)

            self.assertFalse(result["ok"])
            self.assertEqual(expected_axis, result["axis"])
            save_profile.assert_not_called()

    def test_nudge_always_stops_motor_if_wait_fails(self):
        with patch.object(self.tracker, "_capture_gray_frame", return_value=np.zeros((20, 20), dtype=np.uint8)), \
             patch("core.virtual_ptz.time.sleep", side_effect=RuntimeError("wait interrupted")):
            with self.assertRaisesRegex(RuntimeError, "wait interrupted"):
                self.tracker._measure_axis_nudge("pan", 1.0, 0.6, 0.45, 1.0)

        self.assertEqual([(0.6, 0.0)], self.client.moves)
        self.assertEqual(1, self.client.stops)

    def test_nudge_rejects_motion_on_wrong_axis(self):
        wrong_axis_motion = {
            "reliable": True,
            "global_motion_px": 8.0,
            "global_vector_px": [0.2, 8.0],
            "coherence": 0.98,
            "inlier_ratio": 0.95,
        }
        with patch.object(self.tracker, "_capture_gray_frame", return_value=np.zeros((20, 20), dtype=np.uint8)), \
             patch.object(self.tracker, "_global_motion_metrics", return_value=wrong_axis_motion), \
             patch("core.virtual_ptz.time.sleep"):
            result = self.tracker._measure_axis_nudge("pan", 1.0, 0.6, 0.45, 1.0)

        self.assertFalse(result["moved"])
        self.assertLess(result["axis_alignment"], 0.75)

    def test_nudge_accepts_large_axis_motion_with_localized_features(self):
        axis_motion = {
            "reliable": True,
            "global_motion_px": 80.0,
            "global_vector_px": [80.0, 2.0],
            "coherence": 0.98,
            "inlier_ratio": 0.3,
            "occupied_grid_cells": 2,
        }
        with patch.object(self.tracker, "_capture_gray_frame", return_value=np.zeros((20, 20), dtype=np.uint8)), \
             patch.object(self.tracker, "_global_motion_metrics", return_value=axis_motion), \
             patch("core.virtual_ptz.time.sleep"):
            result = self.tracker._measure_axis_nudge("pan", 1.0, 0.2, 0.1, 1.0)

        self.assertTrue(result["moved"])

    def test_endstop_tolerates_transient_unreliable_frame(self):
        unreliable = {"reliable": False, "reason": "too_few_global_inliers"}
        moving = {"reliable": True, "moved": True}
        still = {"reliable": True, "moved": False}
        baseline = {"reliable": True, "global_motion_px": 0.1}
        with patch.object(self.tracker, "_capture_gray_frame", return_value=np.zeros((20, 20), dtype=np.uint8)), \
             patch.object(self.tracker, "_global_motion_metrics", return_value=baseline), \
             patch.object(self.tracker, "_measure_axis_nudge", side_effect=[unreliable, moving, still, still, still, moving]):
            result = self.tracker._seek_endstop("pan", 1.0, 0.2, 0.1, 10, 3)

        self.assertTrue(result["confirmed"])
        self.assertEqual(0.1, result["uncertain_duration_s"])
        self.assertEqual(0.2, result["traversal_duration_s"])

    def test_calibration_rejects_unsafe_parameters_before_moving(self):
        invalid_arguments = (
            {"speed": 0.0},
            {"speed": 1.1},
            {"duration_s": 0.0},
            {"max_pan_steps": 0},
            {"max_tilt_steps": 0},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.tracker.calibrate_mechanical_endstops(**arguments)

        self.assertEqual([], self.client.moves)

    def test_full_calibration_runs_real_image_measurement_against_bounded_camera(self):
        client = SimulatedEndstopClient()
        tracker = VirtualPTZTracker(client, "rtsp://simulated/stream")
        base = np.random.default_rng(42).integers(0, 256, size=(180, 260), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0)

        def simulated_snapshot(*_args, **_kwargs):
            matrix = np.float32([[1, 0, client.pan * 14], [0, 1, client.tilt * 14]])
            frame = cv2.warpAffine(base, matrix, (260, 180))
            ok, encoded = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            return encoded.tobytes()

        with patch("core.virtual_ptz.snapshot", side_effect=simulated_snapshot), \
             patch("core.virtual_ptz.time.sleep"), \
             patch("core.virtual_ptz.save_camera_profile") as save_profile, \
             patch.object(tracker, "load_calibration"):
            result = tracker.calibrate_mechanical_endstops(speed=0.6, duration_s=0.45)

        self.assertTrue(result["ok"])
        self.assertEqual(2.7, result["pan_full_time_sec"])
        self.assertEqual(1.8, result["tilt_full_time_sec"])
        self.assertTrue(all(item["confirmed"] for item in result["evidence"].values()))
        self.assertIn((-0.6, 0.0), client.moves)
        self.assertIn((0.6, 0.0), client.moves)
        self.assertIn((0.0, -0.6), client.moves)
        self.assertIn((0.0, 0.6), client.moves)
        saved_profile = save_profile.call_args.args[1]
        self.assertEqual("CONFIRMED", saved_profile["mechanical_calibration"]["status"])
        self.assertEqual(2.7, saved_profile["pan"]["full_pan_time_sec"])
        self.assertEqual(1.8, saved_profile["tilt"]["full_tilt_time_sec"])
        self.assertEqual((1.0, 1.0), (tracker.virtual_pan, tracker.virtual_tilt))

    def test_absolute_movement_requires_confirmed_calibration(self):
        with self.assertRaisesRegex(RuntimeError, "PTZ_NOT_CALIBRATED"):
            self.tracker.goto_virtual(0.2, 0.0)


if __name__ == "__main__":
    unittest.main()
