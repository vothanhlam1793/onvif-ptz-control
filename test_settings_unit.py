"""
Unit test kiểm tra hàm xử lý backend cho settings
"""
import os
import json
from dotenv import load_dotenv
load_dotenv()

from api.routes import get_system_settings, camera_test_sync, settings_test_llm, CameraTestSyncRequest, LLMTestRequest

def test_unit_settings():
    print("=== 1. Test get_system_settings ===")
    res = get_system_settings()
    print("Current settings loaded:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    assert "camera_host" in res
    assert "ninerouter_base_url" in res

    print("\n=== 2. Test LLM Connection ===")
    llm_req = LLMTestRequest(
        base_url=os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1"),
        api_key=os.getenv("NINEROUTER_API_KEY", ""),
        model=os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")
    )
    llm_res = settings_test_llm(llm_req)
    print("LLM Test Result:", llm_res)
    assert llm_res.get("ok") is True

    print("\n=== 3. Test Camera ONVIF Sync ===")
    cam_req = CameraTestSyncRequest(
        host=os.getenv("CAMERA_HOST", "192.168.110.14"),
        port=int(os.getenv("CAMERA_PORT", 80)),
        username=os.getenv("CAMERA_USER", "admin"),
        password=os.getenv("CAMERA_PASS", "a12345678")
    )
    cam_res = camera_test_sync(cam_req)
    print("Camera Sync Result:", cam_res)
    assert cam_res.get("ok") is True

    print("\nALL SETTINGS LOGIC UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_unit_settings()
