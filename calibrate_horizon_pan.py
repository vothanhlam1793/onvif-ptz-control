"""
Chương trình đo đạc và tìm công thức:
1. Tìm góc Tilt chân trời tối ưu (Optimal Horizon Tilt) thông qua việc quét Pan qua lại ở nhiều tầng Tilt.
2. Đo chính xác góc rộng cơ khí trục Pan (Total Pan Range) & kiểm tra Loop Closure (360 độ).
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "outputs" / "horizon_pan_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_HOST = os.getenv("CAMERA_HOST", "192.168.110.14")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "80"))
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS", "a12345678")

NINEROUTER_BASE_URL = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
NINEROUTER_API_KEY  = os.getenv("NINEROUTER_API_KEY", "sk-1aa6a2183c3f40e1-6zg43d-fcff8a05")
VLM_MODEL           = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")


def encode_image_resized(img_path: Path, max_dim: int = 1024) -> str:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def calculate_image_metrics(img_bgr: np.ndarray) -> dict:
    """
    Tính toán các chỉ số quang học của frame:
    - Edge Density (độ phong phú đường nét/cấu trúc vật thể)
    - Horizon Balance (phân bố thông tin nửa trên vs nửa dưới)
    - Gradient Variance (độ tương phản và đa dạng chi tiết)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w)

    # 2. Phân bố trên / dưới
    top_half = gray[:h//2, :]
    bot_half = gray[h//2:, :]

    std_top = float(np.std(top_half))
    std_bot = float(np.std(bot_half))

    # Cân bằng thông tin (tỷ lệ std giữa 2 nửa càng gần 1 càng cân đối)
    balance = min(std_top, std_bot) / (max(std_top, std_bot) + 1e-5)

    # 3. Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    return {
        "edge_density": edge_density,
        "std_top": std_top,
        "std_bot": std_bot,
        "balance_ratio": balance,
        "laplacian_var": lap_var,
        "composite_score": edge_density * 1000 + balance * 50 + min(100, lap_var / 10),
    }


def sweep_at_tilt(client: OnvifClient, rtsp_url: str, tilt_val: float, tag: str, num_samples: int = 5):
    """
    Quay đến mức tilt_val, sau đó quét Pan từ trái sang phải, chụp num_samples frame.
    tilt_val: [-1.0, 1.0] (trong đó -1.0 là kịch đáy, 0.0 là trung tâm, +1.0 là kịch trần)
    """
    tag_dir = OUTPUT_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Sweep Test] Mức Tilt: {tag} (tilt_val={tilt_val:.2f})")
    
    # 1. Đưa Tilt về kịch đáy trước rồi nhích lên mốc tilt_val
    client.continuous_move(0.0, -0.8)
    time.sleep(2.5)
    client.stop()
    time.sleep(0.3)

    if tilt_val > -0.95:
        # Nhích lên một lượng thời gian tương ứng
        t_up = ((tilt_val - (-1.0)) / 2.0) * 2.4
        if t_up > 0.1:
            client.continuous_move(0.0, 0.8)
            time.sleep(t_up)
            client.stop()
            time.sleep(0.4)

    # 2. Quay kịch trái
    print(f"Quay kịch trái...")
    client.continuous_move(-0.8, 0.0)
    time.sleep(5.5)
    client.stop()
    time.sleep(0.8)

    # 3. Quét dần sang phải num_samples bước
    step_pan_time = 5.2 / (num_samples - 1)
    frames = []
    paths = []
    metrics_list = []

    for i in range(num_samples):
        print(f"Chụp mẫu Pan {i+1}/{num_samples} tại {tag}...")
        fb = snapshot(rtsp_url, width=1920, height=1080)
        p = tag_dir / f"frame_{i:02d}.jpg"
        if fb:
            p.write_bytes(fb)
            paths.append(p)
            arr = np.frombuffer(fb, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            m = calculate_image_metrics(img)
            metrics_list.append(m)

        if i < num_samples - 1:
            client.continuous_move(0.8, 0.0)
            time.sleep(step_pan_time)
            client.stop()
            time.sleep(0.8)

    avg_score = np.mean([m["composite_score"] for m in metrics_list]) if metrics_list else 0
    avg_balance = np.mean([m["balance_ratio"] for m in metrics_list]) if metrics_list else 0
    print(f"-> Điểm đánh giá {tag}: Composite Score = {avg_score:.2f}, Balance Ratio = {avg_balance:.2f}")

    return {
        "tag": tag,
        "tilt_val": tilt_val,
        "paths": paths,
        "metrics": metrics_list,
        "avg_score": float(avg_score),
        "avg_balance": float(avg_balance),
    }


def measure_pan_rotation(client: OnvifClient, rtsp_url: str, best_tilt_val: float):
    """
    Quét liên tục trục Pan từ kịch trái sang kịch phải tại mức Tilt tối ưu.
    Sử dụng SIFT Feature Matching cộng dồn các góc dời và kiểm tra Loop Closure (Khép vòng 360°).
    """
    print("\n--- [Đo dải Pan] Quét liên tục và kiểm tra góc quay toàn phần ---")
    
    # 1. Đưa về Tilt tối ưu
    client.continuous_move(0.0, -0.8)
    time.sleep(2.5)
    client.stop()
    time.sleep(0.3)
    if best_tilt_val > -0.95:
        t_up = ((best_tilt_val - (-1.0)) / 2.0) * 2.4
        client.continuous_move(0.0, 0.8)
        time.sleep(t_up)
        client.stop()
        time.sleep(0.4)

    # 2. Quay kịch trái
    client.continuous_move(-0.8, 0.0)
    time.sleep(5.5)
    client.stop()
    time.sleep(1.0)

    pan_dir = OUTPUT_DIR / "pan_sweep_fine"
    pan_dir.mkdir(parents=True, exist_ok=True)

    # Quét 10 bước mịn sang phải
    num_steps = 10
    step_time = 5.2 / (num_steps - 1)
    frames_bgr = []
    pan_paths = []

    for i in range(num_steps):
        fb = snapshot(rtsp_url, width=1920, height=1080)
        p = pan_dir / f"pan_step_{i:02d}.jpg"
        p.write_bytes(fb)
        pan_paths.append(p)
        arr = np.frombuffer(fb, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frames_bgr.append(img)
        print(f"Đã chụp mốc Pan {i+1}/{num_steps}")

        if i < num_steps - 1:
            client.continuous_move(0.8, 0.0)
            time.sleep(step_time)
            client.stop()
            time.sleep(0.8)

    # 3. Tính toán độ dịch chuyển góc ngang bằng OpenCV SIFT
    sift = cv2.SIFT_create()
    hfov_deg = 85.0  # HFOV 1 frame 1080p
    frame_w = 1920
    deg_per_pixel = hfov_deg / frame_w

    accumulated_pan_deg = 0.0
    step_deg_list = []

    for i in range(len(frames_bgr) - 1):
        g1 = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames_bgr[i+1], cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None:
            continue

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) >= 15:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            dx_list = pts1[:, 0] - pts2[:, 0]  # độ dịch chuyển pixel ngang
            dx = float(np.median(dx_list))
            step_deg = dx * deg_per_pixel
            step_deg_list.append(step_deg)
            accumulated_pan_deg += step_deg
            print(f"Pan Step {i} -> {i+1}: Dịch chuyển Δx = {dx:.1f}px (~{step_deg:.1f}°)")

    # 4. Kiểm tra Loop Closure (khớp frame_0 và frame_last)
    g_start = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    g_end = cv2.cvtColor(frames_bgr[-1], cv2.COLOR_BGR2GRAY)

    kp_s, des_s = sift.detectAndCompute(g_start, None)
    kp_e, des_e = sift.detectAndCompute(g_end, None)

    is_loop_closed = False
    loop_matches_count = 0
    if des_s is not None and des_e is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_s, des_e, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        loop_matches_count = len(good)
        if loop_matches_count >= 25:
            is_loop_closed = True

    print(f"\n[Loop Closure Test] Số điểm khớp giữa biên trái và biên phải: {loop_matches_count}")
    print(f"-> Khép vòng 360°: {'CÓ (Trùng lặp 360°)' if is_loop_closed else 'KHÔNG (Giới hạn góc cơ khí < 360°)'}")
    print(f"-> Tổng góc quay ngang tích luỹ (quang học): ~{accumulated_pan_deg:.1f}°")

    return {
        "pan_paths": pan_paths,
        "accumulated_pan_deg": accumulated_pan_deg,
        "is_loop_closed": is_loop_closed,
        "loop_matches_count": loop_matches_count,
    }


def call_vlm_final_analysis(sweep_results: list[dict], pan_result: dict):
    """
    Gửi ảnh đại diện của 3 mức Tilt và 2 biên Pan cho Gemini 3.7 để đúc kết công thức.
    """
    print("\n--- [VLM Gemini 3.7] Phân tích tổng hợp & trích xuất công thức tối ưu ---")

    # Lấy ảnh giữa của 3 tầng Tilt
    img_low = encode_image_resized(sweep_results[0]["paths"][2])
    img_mid = encode_image_resized(sweep_results[1]["paths"][2])
    img_high = encode_image_resized(sweep_results[2]["paths"][2])

    # Lấy 2 biên trái/phải của dải Pan
    img_pan_left = encode_image_resized(pan_result["pan_paths"][0])
    img_pan_right = encode_image_resized(pan_result["pan_paths"][-1])

    prompt = f"""Bạn là kỹ sư thị giác máy tính và hiệu chuẩn robot PTZ camera.
Tôi gửi 5 ảnh thực nghiệm thu được từ camera:
- Ảnh 1: Tầng Tilt Thấp (tilt = {sweep_results[0]['tilt_val']:.2f})
- Ảnh 2: Tầng Tilt Vừa (tilt = {sweep_results[1]['tilt_val']:.2f})
- Ảnh 3: Tầng Tilt Cao (tilt = {sweep_results[2]['tilt_val']:.2f})
- Ảnh 4: Biên cực trái trục Pan (Pan = -1.0)
- Ảnh 5: Biên cực phải trục Pan (Pan = +1.0)

Dữ liệu đo đạc cảm biến & OpenCV SIFT:
- Điểm cân bằng thông tin tầng: Thấp={sweep_results[0]['avg_balance']:.2f}, Vừa={sweep_results[1]['avg_balance']:.2f}, Cao={sweep_results[2]['avg_balance']:.2f}
- Góc quay ngang tích lũy SIFT: {pan_result['accumulated_pan_deg']:.1f}°
- Trạng thái khép vòng 360°: {pan_result['is_loop_closed']} (Khớp {pan_result['loop_matches_count']} inliers)

Nhiệm vụ:
1. Đánh giá trong 3 tầng Tilt, tầng nào là 'Góc chân trời tối ưu' nhất cho Panorama toàn cảnh (chứa đầy đủ thông tin cảnh quan, không bị cắm mặt xuống sàn hay ngửa trần quá mức).
2. Xác định Tổng góc quay ngang cơ khí thực tế của trục Pan (Total Mechanical Pan Range) theo độ (°).
3. Đưa ra Công thức chuẩn xác để tính toán góc Tilt chân trời tối ưu và số frame ngang (N_pan) cần chụp để quét phủ toàn cảnh.

Trả về DUY NHẤT một chuỗi JSON chuẩn:
{{
  "optimal_horizon_tilt_val": <float: giá trị tilt [-1.0, 1.0]>,
  "optimal_horizon_tilt_deg": <float: góc độ so với phương ngang [-5, 80]>,
  "total_pan_range_deg": <float: góc quay ngang toàn dải>,
  "is_360_continuous": <bool>,
  "optimal_pan_steps": <int: số frame ngang cần chụp>,
  "formula_horizon_tilt": "<Công thức tính góc tilt chân trời>",
  "formula_pan_steps": "<Công thức tính số frame Pan>",
  "scientific_summary": "<Tóm tắt nhận định kỹ thuật>"
}}
"""

    llm = ChatOpenAI(
        model=VLM_MODEL,
        base_url=NINEROUTER_BASE_URL,
        api_key=NINEROUTER_API_KEY,
        max_tokens=800,
        timeout=35,
    )

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_low}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_mid}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_high}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_pan_left}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_pan_right}"}},
    ])

    response = llm.invoke([msg])
    content = response.content.strip()
    print(f"VLM Response:\n{content}\n")

    s = content.find("{")
    e = content.rfind("}") + 1
    return json.loads(content[s:e])


def main():
    client = OnvifClient(CAMERA_HOST, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
    client.discover()
    rtsp_url = client.get_stream_uri()

    # 1. Thử nghiệm 3 mức Tilt quét ngang qua lại
    # - tilt_low: -0.8 (sát đáy ~ +4°)
    # - tilt_mid: -0.55 (vừa ~ +15°)
    # - tilt_high: -0.2 (cao ~ +30°)
    tilt_levels = [
        {"val": -0.80, "tag": "tilt_low"},
        {"val": -0.55, "tag": "tilt_mid"},
        {"val": -0.20, "tag": "tilt_high"},
    ]

    sweep_results = []
    for lvl in tilt_levels:
        res = sweep_at_tilt(client, rtsp_url, lvl["val"], lvl["tag"], num_samples=5)
        sweep_results.append(res)

    # Tìm mức tilt có điểm balance cao nhất
    best_sweep = max(sweep_results, key=lambda x: x["avg_balance"])
    best_tilt_val = best_sweep["tilt_val"]
    print(f"\n=> Mức Tilt sơ bộ tốt nhất theo OpenCV: {best_sweep['tag']} (tilt={best_tilt_val:.2f})")

    # 2. Đo chi tiết dải quay Pan tại mức Tilt tối ưu
    pan_res = measure_pan_rotation(client, rtsp_url, best_tilt_val)

    # 3. Tổng hợp và nhờ Gemini 3.7 trích xuất công thức
    vlm_final = call_vlm_final_analysis(sweep_results, pan_res)

    print("\n=======================================================")
    print(" KẾT QUẢ ĐO ĐẠC & CÔNG THỨC HIỆU CHUẨN HORIZON & PAN")
    print("=======================================================")
    print(f"1. Góc Tilt chân trời tối ưu:     tilt_val = {vlm_final.get('optimal_horizon_tilt_val')} (tương đương ~{vlm_final.get('optimal_horizon_tilt_deg')}°)")
    print(f"2. Tổng góc quay ngang (Pan):     ~{vlm_final.get('total_pan_range_deg')}°")
    print(f"3. Khép vòng 360 độ:              {vlm_final.get('is_360_continuous')}")
    print(f"4. Số khung hình Pan tối ưu:      {vlm_final.get('optimal_pan_steps')} frames")
    print(f"5. Công thức Tilt chân trời:      {vlm_final.get('formula_horizon_tilt')}")
    print(f"6. Công thức số bước Pan:         {vlm_final.get('formula_pan_steps')}")
    print(f"7. Nhận định khoa học:            {vlm_final.get('scientific_summary')}")
    print("=======================================================")

    # Lưu lại cấu hình vào disk
    calib_file = Path(__file__).parent / "outputs" / "ptz_calibration.json"
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
    else:
        calib_data = {}

    calib_data.update({
        "optimal_horizon_tilt_val": vlm_final.get("optimal_horizon_tilt_val"),
        "optimal_horizon_tilt_deg": vlm_final.get("optimal_horizon_tilt_deg"),
        "total_pan_range_deg": vlm_final.get("total_pan_range_deg"),
        "is_360_continuous": vlm_final.get("is_360_continuous"),
        "optimal_pan_steps": vlm_final.get("optimal_pan_steps"),
        "formula_horizon_tilt": vlm_final.get("formula_horizon_tilt"),
        "formula_pan_steps": vlm_final.get("formula_pan_steps"),
        "scientific_summary": vlm_final.get("scientific_summary"),
        "updated_at": time.time(),
    })

    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu toàn bộ thông số hiệu chuẩn vào: {calib_file}")


if __name__ == "__main__":
    main()
