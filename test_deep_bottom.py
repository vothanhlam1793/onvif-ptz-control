"""
Thử ép camera quay xuống kịch sàn (absolute physical bottom limit).
Quay tilt xuống với tốc độ tối đa (-1.0) trong 5 giây.
Chụp ảnh tilt_bottom_deep.jpg và nhờ Gemini 3.7 so sánh với tilt_bottom.jpg cũ.
"""

import os
import sys
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

import cv2
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "outputs" / "tilt_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_HOST = os.getenv("CAMERA_HOST", "192.168.110.14")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "80"))
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS", "a12345678")

NINEROUTER_BASE_URL = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
NINEROUTER_API_KEY  = os.getenv("NINEROUTER_API_KEY", "sk-1aa6a2183c3f40e1-6zg43d-fcff8a05")
VLM_MODEL           = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")


def encode_image(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def main():
    print(f"Connecting to camera {CAMERA_HOST}:{CAMERA_PORT}...")
    client = OnvifClient(CAMERA_HOST, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
    client.discover()
    rtsp_url = client.get_stream_uri()

    print("\n[Action] Ép camera quay xuống hết cỡ (speed=-1.0, duration=5.0s)...")
    client.continuous_move(0.0, -1.0)
    time.sleep(5.0)
    client.stop()
    time.sleep(1.0)

    deep_path = OUTPUT_DIR / "tilt_bottom_deep.jpg"
    frame_bytes = snapshot(rtsp_url, width=1920, height=1080)
    if frame_bytes:
        deep_path.write_bytes(frame_bytes)
        print(f"Đã lưu ảnh đáy sâu nhất: {deep_path}")
    else:
        print("Không chụp được snapshot!")
        return

    old_bottom = OUTPUT_DIR / "tilt_bottom.jpg"
    if not old_bottom.exists():
        print("Chưa có ảnh tilt_bottom cũ để so sánh.")
        return

    print("\n[VLM] Gửi cả 2 ảnh đáy (cũ vs mới) cho Gemini 3.7 để so sánh độ hạ thấp...")
    b64_old = encode_image(old_bottom)
    b64_new = encode_image(deep_path)

    llm = ChatOpenAI(
        model=VLM_MODEL,
        base_url=NINEROUTER_BASE_URL,
        api_key=NINEROUTER_API_KEY,
        max_tokens=500,
        timeout=30,
    )

    prompt = """So sánh 2 ảnh chụp từ camera PTZ:
- Ảnh 1 (OLD_BOTTOM): Ảnh chụp vị trí đáy trước đó.
- Ảnh 2 (DEEP_BOTTOM): Ảnh chụp sau khi ép quay xuống kịch đáy tối đa (speed=-1.0, 5s).

Hãy phân tích:
1. Ảnh 2 có thực sự nhìn xuống sâu hơn ảnh 1 không, hay cả 2 đã chạm cùng một giới hạn cơ khí kịch đáy?
2. Có những vật thể / chi tiết sàn nào mới xuất hiện ở ảnh 2 không?
3. Ước tính góc nghiêng trục quang hiện tại ở ảnh 2 (so với phương ngang 0 độ, ví dụ chúc xuống -10°, -20°...)?
"""

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_old}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_new}"}},
    ])

    resp = llm.invoke([msg])
    print("\n=== KẾT QUẢ PHÂN TÍCH TỪ GEMINI 3.7 ===")
    print(resp.content.strip())
    print("=======================================")


if __name__ == "__main__":
    main()
