"""
Unit tests cho Virtual PTZ Engine & Toạ độ không gian.
"""
import sys
from pathlib import Path

# Đảm bảo import đúng project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.virtual_ptz import VirtualPTZTracker
from core.auto_calibration import load_camera_profile

class DummyClient:
    def __init__(self):
        self.camera_key = "lc_ipc_k2e_3h3w"
    def continuous_move(self, pan_speed, tilt_speed):
        pass
    def stop(self):
        pass

def test_virtual_engine():
    print("=== TEST VIRTUAL PTZ ENGINE ===")
    client = DummyClient()
    tracker = VirtualPTZTracker(client, "dummy_rtsp")

    assert tracker.camera_key == "lc_ipc_k2e_3h3w"

    # 1. Test chuyển đổi biên trái dưới [-1.0, -1.0]
    p_deg, t_deg = tracker.virtual_to_physical_angles(-1.0, -1.0)
    assert p_deg == 0.0
    assert t_deg == -5.0

    # 2. Test chuyển đổi tâm [0.0, 0.0]
    p_deg, t_deg = tracker.virtual_to_physical_angles(0.0, 0.0)
    assert p_deg == 183.0
    assert t_deg == 37.5

    # 3. Test chuyển đổi biên phải trên [1.0, 1.0]
    p_deg, t_deg = tracker.virtual_to_physical_angles(1.0, 1.0)
    assert p_deg == 366.0
    assert t_deg == 80.0

    # 4. Test chuyển đổi ngược lại (Physical -> Virtual)
    v_pan, v_tilt = tracker.physical_angles_to_virtual(183.0, 37.5)
    assert abs(v_pan - 0.0) < 1e-3
    assert abs(v_tilt - 0.0) < 1e-3

    # 5. Test Status Telemetry
    tracker.virtual_pan = -0.5
    tracker.virtual_tilt = 0.2
    status = tracker.get_status()
    assert "pan" in status and "pan_deg" in status

    print("ALL VIRTUAL PTZ ENGINE TESTS PASSED!")

if __name__ == "__main__":
    test_virtual_engine()
