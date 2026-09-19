"""
Test toàn diện Virtual PTZ Mechanical & Spatial Positioning Engine:
1. Load Camera Profile & VirtualPTZTracker.
2. Kiểm tra công thức chuyển đổi 2 chiều: Virtual Coordinates [-1.0, 1.0] <-> Physical Angles (Degrees).
3. Kiểm tra tính năng Telemetry Status.
"""

import sys
from pathlib import Path

# Đảm bảo import đúng core
sys.path.insert(0, str(Path(__file__).parent))

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

    print(f"Camera Key: {tracker.camera_key}")
    print(f"Pan Range: {tracker.total_pan_range_deg}° (Full Pan Time: {tracker.full_pan_time}s)")
    print(f"Tilt Range: [{tracker.tilt_min_deg}°, {tracker.tilt_max_deg}°] (Full Tilt Time: {tracker.full_tilt_time}s)")

    # 1. Test chuyển đổi biên trái dưới [-1.0, -1.0]
    p_deg, t_deg = tracker.virtual_to_physical_angles(-1.0, -1.0)
    assert p_deg == 0.0, f"Expected 0.0, got {p_deg}"
    assert t_deg == -5.0, f"Expected -5.0, got {t_deg}"
    print(f"[-1.0, -1.0] -> Pan: {p_deg}°, Tilt: {t_deg}° (PASS)")

    # 2. Test chuyển đổi tâm [0.0, 0.0]
    p_deg, t_deg = tracker.virtual_to_physical_angles(0.0, 0.0)
    assert p_deg == 183.0, f"Expected 183.0, got {p_deg}"
    assert t_deg == 37.5, f"Expected 37.5, got {t_deg}"
    print(f"[ 0.0,  0.0] -> Pan: {p_deg}°, Tilt: {t_deg}° (PASS)")

    # 3. Test chuyển đổi biên phải trên [1.0, 1.0]
    p_deg, t_deg = tracker.virtual_to_physical_angles(1.0, 1.0)
    assert p_deg == 366.0, f"Expected 366.0, got {p_deg}"
    assert t_deg == 80.0, f"Expected 80.0, got {t_deg}"
    print(f"[ 1.0,  1.0] -> Pan: {p_deg}°, Tilt: {t_deg}° (PASS)")

    # 4. Test chuyển đổi ngược lại (Physical -> Virtual)
    v_pan, v_tilt = tracker.physical_angles_to_virtual(183.0, 37.5)
    assert abs(v_pan - 0.0) < 1e-3, f"Expected 0.0, got {v_pan}"
    assert abs(v_tilt - 0.0) < 1e-3, f"Expected 0.0, got {v_tilt}"
    print(f"Angles [183.0°, 37.5°] -> Virtual [{v_pan}, {v_tilt}] (PASS)")

    # 5. Test Status Telemetry
    tracker.virtual_pan = -0.5
    tracker.virtual_tilt = 0.2
    status = tracker.get_status()
    print("Telemetry Status Output:")
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\nALL VIRTUAL PTZ ENGINE UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_virtual_engine()
