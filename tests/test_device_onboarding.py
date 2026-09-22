import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import auto_calibration
from core.device_identity import physical_camera_key
from tools.onvif.calibrate_device import build_confirmed_profile, characterize_actuator


class FakeClient:
    manufacturer = "LC"
    model = "IPC-A42-L"
    serial_number = "SERIAL 01"
    mac_address = "90:6A:94:DA:03:2D"
    firmware_version = "2.0"
    hardware_id = "1.0"
    legacy_camera_key = "lc_ipc_a42_l"
    camera_key = "lc_ipc_a42_l_serial_01"


class FakeTool:
    def __init__(self):
        self.client = FakeClient()
        self.rtsp_url = "rtsp://camera/stream"
        self.moves = []
        self.stops = 0
        frame = np.random.default_rng(9).integers(0, 256, size=(90, 160), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        self.frame = encoded.tobytes()

    def take_snapshot(self, width, height):
        return self.frame

    def start_move(self, direction, speed, timeout_s):
        self.moves.append((direction, speed, timeout_s))

    def stop(self):
        self.stops += 1


def metrics(pan=0.0, tilt=0.0):
    return {
        "reliable": True,
        "pan_deg": pan,
        "tilt_deg": tilt,
        "pan_uncertainty_deg": 0.1,
        "tilt_uncertainty_deg": 0.1,
        "inliers": 100,
    }


class DeviceOnboardingTests(unittest.TestCase):
    def test_physical_key_separates_same_model_cameras(self):
        first = physical_camera_key("LC", "IPC-A42-L", "SERIAL 01")
        second = physical_camera_key("LC", "IPC-A42-L", "SERIAL 02")

        self.assertEqual("lc_ipc_a42_l_serial_01", first)
        self.assertNotEqual(first, second)

    def test_legacy_profile_migrates_only_for_matching_device(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            auto_calibration, "PROFILES_DIR", Path(directory)
        ):
            auto_calibration.save_camera_profile(FakeClient.legacy_camera_key, {
                "camera_key": FakeClient.legacy_camera_key,
                "device_info": {
                    "serial_number": FakeClient.serial_number,
                    "mac_address": FakeClient.mac_address.replace(":", "").lower(),
                },
                "optical": {"hfov_deg": 60.0},
            })

            migrated_path = auto_calibration.migrate_legacy_camera_profile(FakeClient())
            migrated = json.loads(migrated_path.read_text(encoding="utf-8"))

        self.assertEqual(FakeClient.camera_key, migrated["camera_key"])
        self.assertEqual(FakeClient.legacy_camera_key, migrated["migrated_from_camera_key"])
        self.assertEqual(FakeClient.serial_number, migrated["camera_identity"]["serial_number"])

    def test_profile_save_is_atomic_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            auto_calibration, "PROFILES_DIR", Path(directory)
        ):
            path = auto_calibration.save_camera_profile("camera_1", {"status": "CONFIRMED"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            temporary_files = list(Path(directory).glob(".*.tmp"))

        self.assertEqual("CONFIRMED", payload["status"])
        self.assertEqual([], temporary_files)

    def test_actuator_characterization_confirms_all_four_directions(self):
        tool = FakeTool()
        side_effect = []
        movements = {
            "left": metrics(pan=-4.0),
            "right": metrics(pan=5.0),
            "up": metrics(tilt=6.0),
            "down": metrics(tilt=-6.5),
        }
        for _ in range(2):
            for direction in ("left", "right", "up", "down"):
                side_effect.extend([movements[direction], metrics()])

        with patch(
            "tools.onvif.calibrate_device.estimate_relative_rotation",
            side_effect=side_effect,
        ), patch("tools.onvif.calibrate_device.time.sleep"):
            result = characterize_actuator(
                tool,
                np.eye(3),
                np.zeros(5),
                (160, 90),
                repeats=2,
            )

        self.assertEqual("CONFIRMED", result["status"])
        self.assertEqual(5, result["recommended_tolerance_deg"]["pan"])
        self.assertEqual(7, result["recommended_tolerance_deg"]["tilt"])
        self.assertEqual({"left", "right", "up", "down"}, set(result["minimum_pulse_deg"]))

    def test_confirmed_profile_preserves_mechanical_section(self):
        tool = FakeTool()
        intrinsic = {
            "reliable": True,
            "image_size": [1280, 720],
            "intrinsic_matrix": np.eye(3).tolist(),
            "distortion_coefficients": [0.0] * 5,
            "hfov_deg": 60.0,
            "vfov_deg": 35.0,
        }
        actuator = {"status": "CONFIRMED", "minimum_pulse_deg": {}}
        with patch(
            "tools.onvif.calibrate_device.load_camera_profile",
            return_value={"mechanical_calibration": {"status": "CONFIRMED"}},
        ):
            profile = build_confirmed_profile(tool, intrinsic, actuator, 1.2)

        self.assertEqual(2, profile["schema_version"])
        self.assertEqual("CONFIRMED", profile["onboarding_calibration"]["status"])
        self.assertEqual("CONFIRMED", profile["mechanical_calibration"]["status"])


if __name__ == "__main__":
    unittest.main()
