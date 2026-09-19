"""
Test chạy toàn bộ quy trình Panorama với VirtualPTZTracker được kích hoạt.
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
    print(f"Progress Callback: Step {step}/{total} captured ({len(frame_bytes)} bytes)")

print("\n--- Starting run_panorama with Virtual PTZ ---")
res = run_panorama(client=client, rtsp_url=rtsp_url, tracker=tracker, on_progress=progress_cb)

print("\n--- Panorama Execution Finished ---")
print(f"Error: {res.get('error')}")
print(f"Frame count: {res.get('frame_count')}")
print(f"Calibration Note: {res.get('calibration_note')}")
print(f"Saved Paths: {res.get('saved_paths')}")
if res.get("panorama"):
    print(f"Panorama Bytes: {len(res['panorama'])} bytes")
if res.get("grid"):
    print(f"Grid Bytes: {len(res['grid'])} bytes")
