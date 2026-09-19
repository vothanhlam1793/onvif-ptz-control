"""
Integration test: Kiểm thử toàn diện module tự động nhận diện thiết bị ONVIF,
nạp camera profile và chạy chu trình Panorama tự động không tốn thời gian đo lại.
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

with client:
    print("--- [1] Test lấy thông tin profile camera ---")
    res = client.get("/ptz/calibration/profile")
    assert res.status_code == 200
    data = res.json()
    print(f"Profile API Data: {data}")
    assert data.get("is_calibrated") is True
    assert data.get("camera_key") == "lc_ipc_k2e_3h3w"

    print("\n--- [2] Test khởi chạy Panorama với profile đã nạp ---")
    res = client.post("/panorama/start")
    assert res.status_code == 200
    print(f"Panorama Start response: {res.json()}")

print("\nALL CALIBRATION ENGINE & PROFILE TESTS PASSED!")
