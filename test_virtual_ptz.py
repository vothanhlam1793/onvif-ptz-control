"""
Script test và hiệu chuẩn VirtualPTZTracker.
Kiểm tra:
1. Homing (quét kịch biên trái/dưới rồi về tâm ảo 0,0)
2. Ghi nhận trạng thái và toạ độ
3. Thử nghiệm Goto Virtual (-0.5, 0.5, 0.0)
4. Thử nghiệm Click-to-center giả lập
5. Lưu và tải file cấu hình calibration
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from core.onvif_client import OnvifClient
from core.virtual_ptz import VirtualPTZTracker

host = os.getenv("CAMERA_HOST", "192.168.110.14")
port = int(os.getenv("CAMERA_PORT", "80"))
user = os.getenv("CAMERA_USER", "admin")
pwd = os.getenv("CAMERA_PASS", "a12345678")

client = OnvifClient(host, port, user, pwd)
client.discover()
rtsp_url = client.get_stream_uri()

tracker = VirtualPTZTracker(client, rtsp_url)

print("=== [1] Test Virtual PTZ Initial Status ===")
st = tracker.get_status()
print(f"Status: {st}")

print("\n=== [2] Test Homing Routine ===")
print("Running home()...")
home_res = tracker.home(speed=0.8)
print(f"Home result: {home_res}")
time.sleep(1)

print("\n=== [3] Test Goto Virtual Position ===")
print("Moving to virtual pan: 0.5, tilt: 0.0 ...")
tracker.goto_virtual(target_pan=0.5, target_tilt=0.0, speed=0.8)
print(f"Status after pan=0.5: {tracker.get_status()}")
time.sleep(1)

print("Moving to virtual pan: -0.5, tilt: 0.2 ...")
tracker.goto_virtual(target_pan=-0.5, target_tilt=0.2, speed=0.8)
print(f"Status after pan=-0.5, tilt=0.2: {tracker.get_status()}")
time.sleep(1)

print("Moving back to center (0.0, 0.0) ...")
tracker.goto_virtual(target_pan=0.0, target_tilt=0.0, speed=0.8)
print(f"Status after return center: {tracker.get_status()}")
time.sleep(1)

print("\n=== [4] Test Click Aim (simulated click at x=960, y=360 on 1280x720) ===")
# Điểm x=960 lệch phải so với tâm 640 -> camera sẽ nhích sang phải
print("Executing click_aim at (960, 360)...")
tracker.click_aim(click_x=960, click_y=360, frame_w=1280, frame_h=720)
print(f"Status after click aim: {tracker.get_status()}")

print("\n=== [5] Test Continuous Move tracking ===")
print("Starting move right for 1.0 second...")
tracker.start_move(pan_dir=1.0, tilt_dir=0.0, speed=0.4)
time.sleep(1.0)
tracker.stop_move()
print(f"Status after continuous move: {tracker.get_status()}")

print("\n=== [6] Test Save & Load Calibration ===")
tracker.save_calibration()
print("Saved calibration. Re-instantiating tracker...")
new_tracker = VirtualPTZTracker(client, rtsp_url)
print(f"Loaded config: full_pan_time={new_tracker.full_pan_time}, fov_h={new_tracker.fov_degrees_h}")

print("\n=== ALL VIRTUAL PTZ TESTS FINISHED ===")
