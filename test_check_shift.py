"""
Kiểm tra xem độ dịch chuyển 10.087 là do camera chúc xuống sâu hơn hay do rung lắc/nhiễu
"""
import time
import cv2
import numpy as np
from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

client = OnvifClient("192.168.110.14", 80, "admin", "a12345678")
client.discover()
rtsp_url = client.get_stream_uri()

# Chụp f0
f0 = snapshot(rtsp_url, 1280, 720)

# Liên tục nhồi 3 lệnh ContinuousMove xuống mạnh
for _ in range(3):
    client.continuous_move(0.0, -1.0)
    time.sleep(1.0)
    client.stop()
    time.sleep(0.3)

f1 = snapshot(rtsp_url, 1280, 720)

img0 = cv2.imdecode(np.frombuffer(f0, np.uint8), cv2.IMREAD_GRAYSCALE)
img1 = cv2.imdecode(np.frombuffer(f1, np.uint8), cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des0, des1, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good) >= 10:
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    dy = np.median(pts1[:, 1] - pts0[:, 1])
    print(f"Độ dịch chuyển trục dọc thật sự giữa 2 lần ép: Δy = {dy:.2f}px")
    if abs(dy) < 1.0:
        print("KẾT LUẬN: Đã kịch chặn cơ khí vật lý (0px shift).")
    else:
        print(f"Camera vẫn chúc thêm được {dy:.2f}px!")
