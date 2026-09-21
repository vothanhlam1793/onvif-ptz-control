import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.onvif.ptz_tool import PtzRuntimeAdapter, PtzTool, PtzToolError
from core.onvif_client import OnvifClient


class FakeClient:
    profile_token = "Profile000"
    ptz_url = "http://camera/onvif/ptz_service"

    def __init__(self):
        self.moves = []
        self.stops = []

    def continuous_move(self, pan, tilt, zoom, timeout_s):
        self.moves.append((pan, tilt, zoom, timeout_s))

    def stop(self, pan_tilt, zoom):
        self.stops.append((pan_tilt, zoom))


class PtzToolTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.tool = PtzTool(client=self.client, rtsp_url="rtsp://camera/stream")

    def test_start_move_maps_direction_and_timeout(self):
        self.tool.start_move("right", speed=0.4, timeout_s=5.0)
        self.assertEqual([(0.4, 0.0, 0.0, 5.0)], self.client.moves)

    def test_stop_stops_pan_tilt_and_zoom(self):
        self.tool.stop()
        self.assertEqual([(True, True)], self.client.stops)

    def test_rejects_unsafe_nudge_duration(self):
        with self.assertRaisesRegex(PtzToolError, "PTZ_INVALID_DURATION"):
            self.tool.nudge("left", duration_s=2.1)

    def test_rejects_unknown_direction(self):
        with self.assertRaisesRegex(PtzToolError, "PTZ_INVALID_DIRECTION"):
            self.tool.start_move("north")

    def test_runtime_blocks_move_before_login(self):
        runtime = PtzRuntimeAdapter()
        with self.assertRaisesRegex(PtzToolError, "PTZ_NOT_LOGGED_IN"):
            runtime.move("right", 0.3, 0.5)

    def test_continuous_move_omits_zero_zoom_and_sets_timeout(self):
        client = OnvifClient("camera", 80, "user", "password")
        client.profile_token = "Profile000"
        captured = []
        client._post = lambda url, body: captured.append(body)
        client.continuous_move(0.3, 0.0, timeout_s=5.0)
        self.assertIn("<Timeout>PT5.0S</Timeout>", captured[0])
        self.assertNotIn("<Zoom", captured[0])


if __name__ == "__main__":
    unittest.main()
