"""
Chạy xử lý phân tích OpenCV & VLM từ các frame đã chụp (không cần quay lại camera).
"""
import json
import base64
import time
from pathlib import Path
from dotenv import load_dotenv

import cv2
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "outputs" / "horizon_pan_calibration"

NINEROUTER_BASE_URL = "https://9router.camerangochoang.com/v1"
NINEROUTER_API_KEY  = "sk-1aa6a2183c3f40e1-6zg43d-fcff8a05"
VLM_MODEL           = "ag/gemini-3.7-flash-high"


def encode_image_resized(img_path: Path, max_dim: int = 1024) -> str:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def main():
    print("--- [1] Tính toán lại ma trận dịch chuyển Pan từ ảnh đã lưu ---")
    pan_dir = OUTPUT_DIR / "pan_sweep_fine"
    pan_paths = sorted(list(pan_dir.glob("pan_step_*.jpg")))
    print(f"Tìm thấy {len(pan_paths)} frames pan.")

    frames_bgr = [cv2.imread(str(p)) for p in pan_paths]

    sift = cv2.SIFT_create()
    hfov_deg = 85.0
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
            dx_list = pts1[:, 0] - pts2[:, 0]
            dx = float(np.median(dx_list))
            step_deg = dx * deg_per_pixel
            step_deg_list.append(step_deg)
            accumulated_pan_deg += step_deg
            print(f"Pan Step {i} -> {i+1}: Dịch chuyển Δx = {dx:.1f}px (~{step_deg:.1f}°)")

    # Loop Closure (khớp frame_0 và frame_last)
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

    print(f"\n[Loop Closure Test] Số inliers giữa biên trái và biên phải: {loop_matches_count}")
    print(f"-> Khép vòng 360°: {'CÓ (Trùng lặp 360°)' if is_loop_closed else 'KHÔNG (Giới hạn góc cơ khí < 360°)'}")
    print(f"-> Tổng góc quay ngang tích luỹ SIFT: ~{accumulated_pan_deg:.1f}°")

    print("\n--- [2] Gửi VLM Gemini 3.7 phân tích 3 tầng Tilt và 2 biên Pan ---")
    img_low = encode_image_resized(OUTPUT_DIR / "tilt_low" / "frame_02.jpg")
    img_mid = encode_image_resized(OUTPUT_DIR / "tilt_mid" / "frame_02.jpg")
    img_high = encode_image_resized(OUTPUT_DIR / "tilt_high" / "frame_02.jpg")
    img_pan_left = encode_image_resized(pan_paths[0])
    img_pan_right = encode_image_resized(pan_paths[-1])

    prompt = f"""Bạn là kỹ sư thị giác máy tính và hiệu chuẩn robot PTZ camera.
Tôi gửi 5 ảnh thực nghiệm thu được từ camera:
- Ảnh 1: Tầng Tilt Thấp (tilt_val = -0.80, tương đương ~ +4° so với đáy)
- Ảnh 2: Tầng Tilt Vừa (tilt_val = -0.55, tương đương ~ +15° so với đáy)
- Ảnh 3: Tầng Tilt Cao (tilt_val = -0.20, tương đương ~ +30° so với đáy)
- Ảnh 4: Biên cực trái trục Pan (Pan = -1.0)
- Ảnh 5: Biên cực phải trục Pan (Pan = +1.0)

Dữ liệu đo đạc cảm biến & OpenCV SIFT:
- Góc quay ngang tích lũy SIFT: ~{accumulated_pan_deg:.1f}°
- Trạng thái khép vòng 360°: {is_loop_closed} (Khớp {loop_matches_count} inliers giữa 2 biên)

Nhiệm vụ:
1. Đánh giá trong 3 tầng Tilt, tầng nào là 'Góc chân trời tối ưu' nhất cho Panorama toàn cảnh (chứa đầy đủ thông tin cảnh quan, không bị cắm mặt xuống sàn hay ngửa trần quá mức).
2. Xác định Tổng dải quay ngang cơ khí thực tế của trục Pan (Total Mechanical Pan Range) theo độ (°).
3. Đưa ra Công thức chuẩn xác để tính toán góc Tilt chân trời tối ưu và số frame ngang (N_pan) cần chụp để quét phủ toàn cảnh với độ gối đầu an toàn (25-30% overlap).

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
    vlm_final = json.loads(content[s:e])

    # Cập nhật ptz_calibration.json
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
    print(f"Đã lưu toàn bộ kết quả vào {calib_file}")


if __name__ == "__main__":
    main()
