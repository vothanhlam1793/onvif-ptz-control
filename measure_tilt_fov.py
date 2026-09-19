"""
Đo đạc và hiệu chuẩn chiều dọc (Tilt FOV & Mechanical Tilt Range) của PTZ Camera.
Kết hợp OpenCV feature matching và VLM Gemini 3.7 qua 9Router.
"""

import os
import sys
import time
import json
import base64
import math
from pathlib import Path
from dotenv import load_dotenv

import cv2
import numpy as np
import requests

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

load_dotenv()

# Cấu hình
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
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def capture_tilt_sequence(client: OnvifClient, rtsp_url: str):
    """
    1. Homing & đưa Pan về tâm (0.0).
    2. Quay kịch dưới (tilt -1.0) -> Chụp tilt_bottom.jpg
    3. Quay kịch trên (tilt +1.0) -> Chụp tilt_top.jpg
    4. Chụp chuỗi 6 bước từ dưới lên trên.
    """
    print("\n--- [Bước 1] Chuẩn bị camera & chụp 2 biên dọc ---")
    
    # Đưa Pan về tâm
    client.continuous_move(-0.8, 0.0)
    time.sleep(5.5)
    client.stop()
    time.sleep(0.3)
    client.continuous_move(0.8, 0.0)
    time.sleep(2.6)
    client.stop()
    time.sleep(0.3)

    # 1. Kịch dưới
    print("Quay kịch dưới (Tilt = -1.0)...")
    client.continuous_move(0.0, -0.8)
    time.sleep(3.0)
    client.stop()
    time.sleep(0.8)

    bottom_bytes = snapshot(rtsp_url, width=1920, height=1080)
    bottom_path = OUTPUT_DIR / "tilt_bottom.jpg"
    if bottom_bytes:
        bottom_path.write_bytes(bottom_bytes)
        print(f"Đã lưu: {bottom_path}")

    # 2. Kịch trên
    print("Quay kịch trên (Tilt = +1.0)...")
    client.continuous_move(0.0, 0.8)
    time.sleep(3.0)
    client.stop()
    time.sleep(0.8)

    top_bytes = snapshot(rtsp_url, width=1920, height=1080)
    top_path = OUTPUT_DIR / "tilt_top.jpg"
    if top_bytes:
        top_path.write_bytes(top_bytes)
        print(f"Đã lưu: {top_path}")

    # 3. Chuỗi quét từng nấc nhỏ từ dưới lên (6 steps)
    print("\nQuay lại kịch dưới để bắt đầu quét chuỗi 6 bước từ dưới lên...")
    client.continuous_move(0.0, -0.8)
    time.sleep(3.0)
    client.stop()
    time.sleep(0.8)

    steps = 6
    step_time = 2.4 / (steps - 1)  # thời gian quay mỗi nấc
    step_paths = []

    for i in range(steps):
        print(f"Chụp frame nấc dọc {i+1}/{steps}...")
        frame_bytes = snapshot(rtsp_url, width=1920, height=1080)
        p = OUTPUT_DIR / f"tilt_step_{i:02d}.jpg"
        if frame_bytes:
            p.write_bytes(frame_bytes)
            step_paths.append(p)
        
        if i < steps - 1:
            client.continuous_move(0.0, 0.8)
            time.sleep(step_time)
            client.stop()
            time.sleep(0.8)

    # Trả camera về vị trí trung tâm
    print("Đưa camera về góc ngang trung tâm...")
    client.continuous_move(0.0, -0.8)
    time.sleep(1.2)
    client.stop()

    return bottom_path, top_path, step_paths


def calculate_opencv_shifts(step_paths: list[Path]):
    """
    Tính độ dịch chuyển điểm ảnh (pixel shift) giữa các frame liền kề qua SIFT/ORB.
    """
    print("\n--- [Bước 2] Tính toán quang học qua OpenCV ---")
    sift = cv2.SIFT_create()
    
    total_y_shift = 0.0
    shifts = []
    
    for i in range(len(step_paths) - 1):
        img1 = cv2.imread(str(step_paths[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(step_paths[i+1]), cv2.IMREAD_GRAYSCALE)
        
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            continue
            
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Tính Translation vector
            dy_list = dst_pts[:, 0, 1] - src_pts[:, 0, 1]
            median_dy = float(np.median(dy_list))
            shifts.append(median_dy)
            total_y_shift += abs(median_dy)
            
            overlap_pct = max(0.0, (1.0 - abs(median_dy) / img1.shape[0]) * 100)
            print(f"Frame {i} -> {i+1}: Dịch chuyển Δy = {median_dy:.1f}px, Overlap ước tính = {overlap_pct:.1f}%")
        else:
            print(f"Frame {i} -> {i+1}: Không đủ điểm đặc trưng (matches={len(good_matches)})")

    h = 1080
    avg_shift = np.mean([abs(s) for s in shifts]) if shifts else 0
    print(f"Tổng pixel shift tích lũy: {total_y_shift:.1f}px (trên độ cao frame {h}px)")
    
    return {
        "shifts": shifts,
        "total_y_shift": total_y_shift,
        "avg_shift_per_step": avg_shift,
    }


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def encode_image_resized(img_path: Path, max_dim: int = 1024) -> str:
    """Đọc ảnh, resize vừa phải để gửi VLM tối ưu tốc độ và dung lượng."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def call_vlm_tilt_analysis(bottom_path: Path, top_path: Path, step_paths: list[Path]):
    """
    Gửi ảnh cho VLM Gemini 3.7 qua 9Router phân tích góc dọc dùng ChatOpenAI.
    """
    print("\n--- [Bước 3] Hỏi VLM Gemini 3.7 phân tích góc quét dọc ---")
    
    b64_bottom = encode_image_resized(bottom_path)
    b64_top = encode_image_resized(top_path)
    
    prompt = """Bạn là chuyên gia phân tích quang học và hiệu chuẩn camera PTZ giám sát.
Tôi cung cấp 2 ảnh chụp từ cùng 1 camera PTZ đặt cố định (Pan = 0):
- Ảnh 1 (BOTTOM): Camera quay kịch biên dưới cùng (Tilt = -1.0).
- Ảnh 2 (TOP): Camera quay kịch biên trên cùng (Tilt = +1.0).

Nhiệm vụ của bạn:
1. Quan sát các vật thể, trần nhà, mặt sàn, góc nghiêng ống kính giữa 2 ảnh.
2. Ước tính Góc nhìn dọc của 1 frame tĩnh (Single Frame Vertical FOV - VFOV) theo độ (°).
3. Ước tính Tổng dải quay cơ khí chiều dọc của camera từ kịch dưới đến kịch trên (Total Mechanical Tilt Range) theo độ (°).
4. Tính toán Số khung hình dọc (Optimal Vertical Frames) cần chụp để quét phủ trọn từ dưới lên trên với độ gối đầu 25-30% overlap.

Trả về DUY NHẤT một chuỗi JSON hợp lệ theo đúng format:
{
  "single_frame_vfov_deg": <float>,
  "total_tilt_range_deg": <float>,
  "optimal_vertical_frames": <int>,
  "overlap_percentage_est": <float>,
  "analysis_summary": "<Nhận xét chi tiết về không gian quét dọc và vật thể nhìn thấy ở 2 biên>"
}
"""

    llm = ChatOpenAI(
        model=VLM_MODEL,
        base_url=NINEROUTER_BASE_URL,
        api_key=NINEROUTER_API_KEY,
        max_tokens=600,
        timeout=30,
    )
    
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_bottom}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_top}"}},
    ])
    
    response = llm.invoke([msg])
    content = response.content.strip()
    print(f"VLM Response raw:\n{content}\n")
    
    # Trích xuất JSON
    s = content.find("{")
    e = content.rfind("}") + 1
    parsed = json.loads(content[s:e])
    return parsed


def main():
    client = OnvifClient(CAMERA_HOST, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
    client.discover()
    rtsp_url = client.get_stream_uri()
    
    # 1. Chụp chuỗi ảnh
    bottom_path, top_path, step_paths = capture_tilt_sequence(client, rtsp_url)
    
    # 2. Tính toán OpenCV
    cv_res = calculate_opencv_shifts(step_paths)
    
    # 3. Phân tích VLM
    vlm_res = call_vlm_tilt_analysis(bottom_path, top_path, step_paths)
    
    print("\n=======================================================")
    print(" KẾT QUẢ ĐO ĐẠC VÀ PHÂN TÍCH GÓC QUÉT CHIỀU DỌC (TILT)")
    print("=======================================================")
    print(f"1. Sensor Vertical FOV (1 frame):  ~{vlm_res.get('single_frame_vfov_deg')}°")
    print(f"2. Total Mechanical Tilt Range:     ~{vlm_res.get('total_tilt_range_deg')}°")
    print(f"3. Số khung hình dọc tối ưu:        {vlm_res.get('optimal_vertical_frames')} frames")
    print(f"4. Tỷ lệ Overlap ước tính:          {vlm_res.get('overlap_percentage_est')}%")
    print(f"5. Nhận xét phân tích:              {vlm_res.get('analysis_summary')}")
    print("=======================================================")
    
    # Cập nhật ptz_calibration.json
    calib_file = Path(__file__).parent / "outputs" / "ptz_calibration.json"
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
    else:
        calib_data = {}
        
    calib_data["fov_degrees_v"] = float(vlm_res.get("single_frame_vfov_deg", 50.0))
    calib_data["total_tilt_range_deg"] = float(vlm_res.get("total_tilt_range_deg", 90.0))
    calib_data["optimal_vertical_frames"] = int(vlm_res.get("optimal_vertical_frames", 3))
    calib_data["tilt_analysis_summary"] = vlm_res.get("analysis_summary", "")
    calib_data["updated_at"] = time.time()
    
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu cấu hình hiệu chuẩn vào: {calib_file}")


if __name__ == "__main__":
    main()
