import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.spatial_agent import tools


class FakeClient:
    def __init__(self):
        self.moves = []
        self.stopped = False
        self.profile_token = "Profile000"
        self.ptz_url = "http://camera/onvif/ptz_service"

    def continuous_move(self, pan, tilt, zoom, timeout_s):
        self.moves.append((pan, tilt, zoom, timeout_s))

    def stop(self, pan_tilt=True, zoom=True):
        self.stopped = True


class AgentPtzToolTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        tools.set_ptz_hardware(self.client, None, "rtsp://camera/stream")

    def test_move_tool_returns_direction_and_vector(self):
        result = json.loads(tools.camera_move_tool.invoke({"direction": "right", "speed": 0.4, "duration_s": 0.1}))
        self.assertTrue(result["ok"])
        self.assertEqual({"pan": 0.4, "tilt": 0.0, "zoom": 0.0}, result["vector"])
        self.assertEqual([(0.4, 0.0, 0.0, 1.1)], self.client.moves)

    def test_invalid_direction_does_not_move_camera(self):
        result = json.loads(tools.camera_move_tool.invoke({"direction": "north"}))
        self.assertFalse(result["ok"])
        self.assertEqual([], self.client.moves)

    def test_registered_tools_include_legacy_and_manual_motion(self):
        names = {tool.name for tool in tools.get_spatial_agent_tools()}
        self.assertTrue({"camera_status_tool", "camera_move_tool", "camera_stop_tool", "camera_snapshot_tool", "camera_motion_check_tool"}.issubset(names))
        self.assertTrue({"slew_and_verify_target_tool", "scan_and_index_space_tool", "calibrate_camera_hardware_tool"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
