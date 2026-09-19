"""
Test kết nối ONVIF và RTSP stream từ camera thực tế.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

host = os.getenv("CAMERA_HOST", "192.168.110.14")
port = int(os.getenv("CAMERA_PORT", "80"))
user = os.getenv("CAMERA_USER", "admin")
pwd = os.getenv("CAMERA_PASS", "a12345678")

print(f"Connecting ONVIF to {host}:{port}...")
client = OnvifClient(host, port, user, pwd)

try:
    info = client.discover()
    print("ONVIF Discovery Success:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    st = client.get_status()
    print(f"Current Status: {st}")
    
    rtsp_url = client.get_stream_uri()
    print(f"RTSP Stream URL: {rtsp_url}")
    
    print("Testing snapshot...")
    frame = snapshot(rtsp_url, width=1280, height=720)
    if frame:
        print(f"Snapshot OK: {len(frame)} bytes received.")
    else:
        print("Snapshot failed (no frame).")
        
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
