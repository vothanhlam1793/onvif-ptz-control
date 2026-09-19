"""
Core Module: Auto-Homing & 3-Step Camera Calibration Engine.
Quy trình:
  - Nhận diện camera profile theo camera_key (Manufacturer + Model + Serial).
  - Nếu đã có profile trong outputs/camera_profiles/{camera_key}.json -> Tự động nạp (0s overhead).
  - Nếu chưa có -> Tự động kích hoạt quy trình 3 bước (Tilt VFOV -> Horizon Tilt -> Pan 360 Loop Closure).
"""

import os
import time
import json
import base64
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot

PROFILES_DIR = Path(__file__).parent.parent / "outputs" / "camera_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

NINEROUTER_BASE_URL = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
NINEROUTER_API_KEY  = os.getenv("NINEROUTER_API_KEY", "sk-1aa6a2183c3f40e1-6zg43d-fcff8a05")
VLM_MODEL           = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")


def get_profile_path(camera_key: str) -> Path:
    return PROFILES_DIR / f"{camera_key}.json"


def load_camera_profile(camera_key: str) -> Optional[Dict[str, Any]]:
    """Tải profile nếu đã tồn tại."""
    path = get_profile_path(camera_key)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[AutoCalibration] Error reading profile {path}: {e}")
    return None


def save_camera_profile(camera_key: str, profile_data: Dict[str, Any]) -> Path:
    """Lưu profile cấu hình."""
    path = get_profile_path(camera_key)
    profile_data["updated_at"] = time.time()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    print(f"[AutoCalibration] Đã lưu profile thành công: {path}")
    return path


def _encode_bgr(img: np.ndarray, max_dim: int = 1024) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


class AutoCalibrationEngine:
    def __init__(self, client: OnvifClient, rtsp_url: str):
        self.client = client
        self.rtsp_url = rtsp_url
        self.camera_key = client.camera_key or "generic_ptz"
        self.profile: Optional[Dict[str, Any]] = None

    def get_or_calibrate(self, force: bool = False) -> Dict[str, Any]:
        """
        Nạp profile có sẵn hoặc tự động chạy quy trình 3 bước nếu chưa có.
        """
        if not force:
            existing = load_camera_profile(self.camera_key)
            if existing:
                print(f"[AutoCalibration] Tìm thấy profile cho '{self.camera_key}'. Nạp cấu hình tức thì!")
                self.profile = existing
                return self.profile

        print(f"[AutoCalibration] Bắt đầu chu trình hiệu chuẩn 3 bước cho camera mới: '{self.camera_key}'...")
        self.profile = self.run_full_calibration()
        save_camera_profile(self.camera_key, self.profile)
        return self.profile

    # ──────────────────────────────────────────────
    # BƯỚC 1: Hiệu chuẩn trục dọc (Tilt & VFOV)
    # ──────────────────────────────────────────────
    def step1_calibrate_tilt(self) -> Dict[str, Any]:
        print("\n=== [Bước 1/3] Hiệu chuẩn trục dọc (Tilt VFOV & Range) ===")
        # Ép kịch đáy
        self.client.continuous_move(0.0, -0.8)
        time.sleep(3.0)
        self.client.stop()
        time.sleep(0.5)
        fb_bottom = snapshot(self.rtsp_url, 1920, 1080)

        # Ngửa kịch trần
        self.client.continuous_move(0.0, 0.8)
        time.sleep(3.0)
        self.client.stop()
        time.sleep(0.5)
        fb_top = snapshot(self.rtsp_url, 1920, 1080)

        img_b = cv2.imdecode(np.frombuffer(fb_bottom, np.uint8), cv2.IMREAD_COLOR)
        img_t = cv2.imdecode(np.frombuffer(fb_top, np.uint8), cv2.IMREAD_COLOR)

        prompt = """Phân tích 2 ảnh 1 camera PTZ đặt cố định (Pan=0):
- Ảnh 1 (BOTTOM): Kịch đáy (-1.0).
- Ảnh 2 (TOP): Kịch trần (+1.0).
Ước tính:
1. single_frame_vfov_deg: Góc nhìn dọc 1 frame tĩnh.
2. total_tilt_range_deg: Tổng dải quay dọc kịch đáy -> kịch trần.
3. tilt_min_deg: Góc trục quang kịch đáy (so với mặt ngang 0°).
4. tilt_max_deg: Góc trục quang kịch trần.
5. optimal_vertical_frames: Số frame dọc cần quét (overlap 25-30%).

Trả về JSON:
{"single_frame_vfov_deg": <float>, "total_tilt_range_deg": <float>, "tilt_min_deg": <float>, "tilt_max_deg": <float>, "optimal_vertical_frames": <int>}"""

        llm = ChatOpenAI(model=VLM_MODEL, base_url=NINEROUTER_BASE_URL, api_key=NINEROUTER_API_KEY, max_tokens=400)
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_bgr(img_b)}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_bgr(img_t)}"}},
        ])
        resp = llm.invoke([msg])
        content = resp.content.strip()
        s, e = content.find("{"), content.rfind("}") + 1
        res = json.loads(content[s:e])
        print(f"[Bước 1 Done] VFOV={res.get('single_frame_vfov_deg')}°, Dải Tilt={res.get('total_tilt_range_deg')}°")
        return res

    # ──────────────────────────────────────────────
    # BƯỚC 2: Dò góc chân trời tối ưu (Horizon Tilt)
    # ──────────────────────────────────────────────
    def step2_find_horizon_tilt(self, tilt_info: Dict[str, Any]) -> Dict[str, Any]:
        print("\n=== [Bước 2/3] Dò góc chân trời tối ưu (Horizon Tilt) ===")
        # Quét thử ở 3 mốc: -0.80 (thấp), -0.55 (vừa), -0.20 (cao)
        candidate_tilts = [-0.80, -0.55, -0.20]
        samples = []

        for val in candidate_tilts:
            # Di chuyển đến mức tilt
            self.client.continuous_move(0.0, -0.8)
            time.sleep(2.5)
            self.client.stop()
            time.sleep(0.3)
            if val > -0.95:
                t_up = ((val - (-1.0)) / 2.0) * 2.4
                self.client.continuous_move(0.0, 0.8)
                time.sleep(t_up)
                self.client.stop()
                time.sleep(0.3)

            fb = snapshot(self.rtsp_url, 1920, 1080)
            img = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
            samples.append((val, img))

        prompt = """So sánh 3 ảnh ở 3 mức nâng ống kính (Tilt):
- Ảnh 1: Tilt Thấp (val = -0.80)
- Ảnh 2: Tilt Vừa (val = -0.55)
- Ảnh 3: Tilt Cao (val = -0.20)
Chọn mức Tilt nào giữ đường chân trời / tầm mắt ở trung tâm nhất, cân bằng giữa sàn và trần cho ảnh toàn cảnh Panorama.

Trả về JSON:
{"optimal_horizon_tilt_val": <float: ví dụ -0.80>, "optimal_horizon_tilt_deg": <float: ví dụ 4.0>, "reason": "<lý do chọn>"}"""

        llm = ChatOpenAI(model=VLM_MODEL, base_url=NINEROUTER_BASE_URL, api_key=NINEROUTER_API_KEY, max_tokens=300)
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_bgr(samples[0][1])}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_bgr(samples[1][1])}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_bgr(samples[2][1])}"}},
        ])
        resp = llm.invoke([msg])
        content = resp.content.strip()
        s, e = content.find("{"), content.rfind("}") + 1
        res = json.loads(content[s:e])
        print(f"[Bước 2 Done] Góc chân trời tối ưu: tilt_val={res.get('optimal_horizon_tilt_val')} (~{res.get('optimal_horizon_tilt_deg')}°)")
        return res

    # ──────────────────────────────────────────────
    # BƯỚC 3: Dò dải Pan & Khép vòng 360° (Loop Closure)
    # ──────────────────────────────────────────────
    def step3_calibrate_pan(self, horizon_tilt_val: float) -> Dict[str, Any]:
        print("\n=== [Bước 3/3] Dò dải quay ngang (Pan Range & Loop Closure) ===")
        # Đưa về Tilt tối ưu
        self.client.continuous_move(0.0, -0.8)
        time.sleep(2.5)
        self.client.stop()
        time.sleep(0.3)
        if horizon_tilt_val > -0.95:
            t_up = ((horizon_tilt_val - (-1.0)) / 2.0) * 2.4
            self.client.continuous_move(0.0, 0.8)
            time.sleep(t_up)
            self.client.stop()
            time.sleep(0.3)

        # Quay kịch trái
        self.client.continuous_move(-0.8, 0.0)
        time.sleep(5.5)
        self.client.stop()
        time.sleep(0.8)

        # Quét 7 mốc ngang sang phải
        num_samples = 7
        step_time = 5.2 / (num_samples - 1)
        frames_bgr = []

        for i in range(num_samples):
            fb = snapshot(self.rtsp_url, 1920, 1080)
            img = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
            frames_bgr.append(img)
            if i < num_samples - 1:
                self.client.continuous_move(0.8, 0.0)
                time.sleep(step_time)
                self.client.stop()
                time.sleep(0.6)

        # SIFT matching giữa frame đầu và frame cuối
        sift = cv2.SIFT_create()
        kp0, des0 = sift.detectAndCompute(cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY), None)
        kp_last, des_last = sift.detectAndCompute(cv2.cvtColor(frames_bgr[-1], cv2.COLOR_BGR2GRAY), None)

        is_360 = False
        inliers = 0
        if des0 is not None and des_last is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des0, des_last, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            inliers = len(good)
            if inliers >= 25:
                is_360 = True

        total_pan_deg = 360.0 if is_360 else 300.0
        hfov = 85.0
        overlap = 0.40   # 40% overlap an toàn cho panorama 360
        optimal_steps = max(8, math.ceil(total_pan_deg / (hfov * (1.0 - overlap))))
        
        # Mốc tọa độ từng bước
        step_delta = 1.8 / (optimal_steps - 1)
        pan_coords = [round(-0.9 + i * step_delta, 2) for i in range(optimal_steps)]

        print(f"[Bước 3 Done] Loop Closure inliers: {inliers}. Khép vòng 360°: {is_360}. Số bước Pan: {optimal_steps}")
        return {
            "total_pan_range_deg": total_pan_deg,
            "is_360_continuous": is_360,
            "loop_closure_inliers": inliers,
            "optimal_pan_steps": optimal_steps,
            "pan_step_coords": pan_coords,
        }

    # ──────────────────────────────────────────────
    # Chạy quy trình tổng hợp 3 bước
    # ──────────────────────────────────────────────
    def run_full_calibration(self) -> Dict[str, Any]:
        t0 = time.time()
        
        # Bước 1
        t_info = self.step1_calibrate_tilt()
        # Bước 2
        h_info = self.step2_find_horizon_tilt(t_info)
        # Bước 3
        p_info = self.step3_calibrate_pan(h_info.get("optimal_horizon_tilt_val", -0.80))

        profile = {
            "camera_key": self.camera_key,
            "device_info": {
                "manufacturer": self.client.manufacturer,
                "model": self.client.model,
                "serial_number": self.client.serial_number,
                "mac_address": self.client.mac_address,
                "firmware_version": self.client.firmware_version,
                "hardware_id": self.client.hardware_id,
            },
            "optical": {
                "hfov_deg": 85.0,
                "vfov_deg": float(t_info.get("single_frame_vfov_deg", 46.0)),
            },
            "tilt": {
                "tilt_min_deg": float(t_info.get("tilt_min_deg", -5.0)),
                "tilt_max_deg": float(t_info.get("tilt_max_deg", 80.0)),
                "total_tilt_range_deg": float(t_info.get("total_tilt_range_deg", 85.0)),
                "full_tilt_time_sec": 2.4,
                "optimal_horizon_tilt_val": float(h_info.get("optimal_horizon_tilt_val", -0.80)),
                "optimal_horizon_tilt_deg": float(h_info.get("optimal_horizon_tilt_deg", 4.0)),
                "optimal_vertical_frames": int(t_info.get("optimal_vertical_frames", 4)),
            },
            "pan": {
                "total_pan_range_deg": float(p_info.get("total_pan_range_deg", 360.0)),
                "full_pan_time_sec": 5.2,
                "is_360_continuous": bool(p_info.get("is_360_continuous", True)),
                "optimal_pan_steps": int(p_info.get("optimal_pan_steps", 6)),
                "pan_step_coords": p_info.get("pan_step_coords", [-0.90, -0.54, -0.18, 0.18, 0.54, 0.90]),
            },
            "formulas": {
                "formula_horizon_tilt": "tilt_deg = tilt_min_deg + ((tilt_val - (-1.0)) / 2.0) * (tilt_max_deg - tilt_min_deg)",
                "formula_pan_steps": "N_pan = ceil(total_pan_range_deg / (HFOV * (1.0 - overlap_ratio)))",
            },
            "calibration_duration_sec": round(time.time() - t0, 1),
        }
        return profile
