"""
Chạy toàn bộ quy trình quét thực tế 8 frames với góc chân trời tối ưu từ camera và ghép ảnh Panorama mới.
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from core.onvif_client import OnvifClient
from core.virtual_ptz import VirtualPTZTracker
from panorama.agent import run_panorama

host = os.getenv("CAMERA_HOST", "192.168.110.14")
port = int(os.getenv("CAMERA_PORT", "80"))
user = os.getenv("CAMERA_USER", "admin")
pwd = os.getenv("CAMERA_PASS", "a12345678")

print(f"Connecting to camera {host}:{port}...")
client = OnvifClient(host, port, user, pwd)
client.discover()
rtsp_url = client.get_stream_uri()

tracker = VirtualPTZTracker(client, rtsp_url)

def progress_cb(step, total, frame_bytes):
    print(f"Live Progress: Chụp frame {step}/{total} ({len(frame_bytes)} bytes)")

print("\n--- Bắt đầu chạy Panorama 8 Frames với Stitcher Mới ---")
res = run_panorama(client=client, rtsp_url=rtsp_url, tracker=tracker, on_progress=progress_cb)

print("\n--- KẾT QUẢ PANORAMA 8 FRAMES ---")
print(f"Error: {res.get('error')}")
print(f"Số frames đã chụp: {res.get('frame_count')}")
print(f"Ghi chú: {res.get('calibration_note')}")
print(f"Saved Paths: {res.get('saved_paths')}")
if res.get("panorama"):
    print(f"Kích thước file ảnh Panorama: {len(res['panorama'])} bytes")
