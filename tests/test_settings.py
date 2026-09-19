"""
Unit tests cho cấu hình hệ thống & camera sync.
"""
import os
import sys
from pathlib import Path

# Đảm bảo import đúng project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from api.routes import get_system_settings, camera_test_sync, settings_test_llm, CameraTestSyncRequest, LLMTestRequest

def test_settings_logic():
    print("=== TEST SETTINGS & SYNC ===")
    res = get_system_settings()
    assert "camera_host" in res
    assert "ninerouter_base_url" in res

    # Test Camera ONVIF Sync với host hiện tại
    cam_req = CameraTestSyncRequest(
        host=os.getenv("CAMERA_HOST", "192.168.110.110"),
        port=int(os.getenv("CAMERA_PORT", 80)),
        username=os.getenv("CAMERA_USER", "admin"),
        password=os.getenv("CAMERA_PASS", "asrkpVg10!@#")
    )
    cam_res = camera_test_sync(cam_req)
    print("Camera Sync Result:", cam_res)
    assert cam_res.get("ok") is True

    print("ALL SETTINGS LOGIC TESTS PASSED!")

if __name__ == "__main__":
    test_settings_logic()
