"""
Dùng RelativeMove ép xuống thêm một lần nữa và kiểm tra độ dịch chuyển pixel qua OpenCV
"""
import time
from pathlib import Path
import cv2
import numpy as np
from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

client = OnvifClient("192.168.110.14", 80, "admin", "a12345678")
client.discover()
rtsp_url = client.get_stream_uri()

# Chụp trước
f_before = snapshot(rtsp_url, 1280, 720)

# Ép RelativeMove xuống tối đa
print("Gửi lệnh RelativeMove xuống -1.0...")
try:
    client.relative_move(0.0, -1.0, tilt_speed=1.0)
    time.sleep(2.0)
except Exception as e:
    print(f"RelativeMove error: {e}")

# Chụp sau
f_after = snapshot(rtsp_url, 1280, 720)

img1 = cv2.imdecode(np.frombuffer(f_before, np.uint8), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imdecode(np.frombuffer(f_after, np.uint8), cv2.IMREAD_GRAYSCALE)

diff = cv2.absdiff(img1, img2)
mean_diff = np.mean(diff)
print(f"Mức độ thay đổi pixel (Mean Diff): {mean_diff:.3f}")
if mean_diff < 1.0:
    print("XÁC NHẬN: Camera đã chạm chặn cơ khí (physical hard stop) ở phía dưới, không thể chúc sâu hơn.")
else:
    print(f"Camera có dịch chuyển nhẹ: {mean_diff:.3f}")
