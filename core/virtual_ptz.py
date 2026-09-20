"""
Virtual PTZ Coordinate Engine (VCC).
Giải pháp tạo hệ toạ độ ảo cho camera PTZ giá rẻ không có optical encoder / AbsoluteMove.

Hệ trục:
  - Pan ∈ [-1.0, 1.0] (tương ứng -1.0 là kịch biên trái, 0.0 là trung tâm, +1.0 là kịch biên phải)
  - Tilt ∈ [-1.0, 1.0] (tương ứng -1.0 là kịch biên dưới, 0.0 là ngang, +1.0 là kịch biên trên)
"""

import time
import json
import base64
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from core.onvif_client import OnvifClient
from core.auto_calibration import AutoCalibrationEngine, load_camera_profile, save_camera_profile
from streaming.stream_relay import snapshot

CALIBRATION_FILE = Path(__file__).parent.parent / "outputs" / "ptz_calibration.json"


class VirtualPTZTracker:
    """
    Theo dõi và điều khiển vị trí Pan/Tilt ảo trong không gian [-1.0, 1.0].
    Tự động liên kết theo camera_key và nạp profile tương ứng.
    """

    def __init__(self, client: OnvifClient, rtsp_url: str):
        self.client = client
        self.rtsp_url = rtsp_url
        self.camera_key = client.camera_key or "generic_ptz"
        self._lock = threading.Lock()

        # Toạ độ ảo hiện tại [-1.0, 1.0]
        self.virtual_pan: float = 0.0
        self.virtual_tilt: float = 0.0
        self.is_homed: bool = False

        # Thông số mặc định
        self.full_pan_time: float = 5.2
        self.full_tilt_time: float = 2.4
        self.fov_degrees_h: float = 85.0
        self.fov_degrees_v: float = 46.0
        self.pan_speed_factor: float = 355.0 / 5.2
        
        # Profile đầy đủ
        self.profile: Optional[Dict[str, Any]] = None
        self.total_pan_range_deg: float = 360.0
        self.tilt_min_deg: float = -5.0
        self.tilt_max_deg: float = 80.0

        # Trạng thái di chuyển liên tục
        self._moving: bool = False
        self._move_dir: Tuple[float, float] = (0.0, 0.0)
        self._move_speed: float = 0.4
        self._move_start_time: float = 0.0

        self.load_calibration()

    def load_calibration(self):
        """Ưu tiên tải theo camera profile (camera_key), fallback ptz_calibration.json."""
        prof = load_camera_profile(self.camera_key)
        if prof:
            self.profile = prof
            self.full_pan_time = prof.get("pan", {}).get("full_pan_time_sec", self.full_pan_time)
            self.full_tilt_time = prof.get("tilt", {}).get("full_tilt_time_sec", self.full_tilt_time)
            self.fov_degrees_h = prof.get("optical", {}).get("hfov_deg", self.fov_degrees_h)
            self.fov_degrees_v = prof.get("optical", {}).get("vfov_deg", self.fov_degrees_v)
            self.total_pan_range_deg = prof.get("pan", {}).get("total_pan_range_deg", 360.0)
            self.pan_type = prof.get("pan", {}).get("pan_type", "bounded_stops")
            self.allow_zero_wrap_around = prof.get("pan", {}).get("allow_zero_wrap_around", False)
            self.tilt_min_deg = prof.get("tilt", {}).get("tilt_min_deg", -5.0)
            self.tilt_max_deg = prof.get("tilt", {}).get("tilt_max_deg", 80.0)
            self.pan_speed_factor = self.total_pan_range_deg / self.full_pan_time
            print(f"[VirtualPTZ] Đã nạp profile thiết bị '{self.camera_key}' (Pan Type: {self.pan_type})")
            return

        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.full_pan_time = data.get("full_pan_time", self.full_pan_time)
                    self.full_tilt_time = data.get("full_tilt_time", self.full_tilt_time)
                    self.fov_degrees_h = data.get("fov_degrees_h", self.fov_degrees_h)
                    self.fov_degrees_v = data.get("fov_degrees_v", self.fov_degrees_v)
                    self.pan_speed_factor = data.get("pan_speed_factor", self.pan_speed_factor)
            except Exception as e:
                print(f"[VirtualPTZ] Load calibration error: {e}")

    def save_calibration(self):
        """Lưu thông số calibration vào disk."""
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_key": self.camera_key,
            "full_pan_time": self.full_pan_time,
            "full_tilt_time": self.full_tilt_time,
            "fov_degrees_h": self.fov_degrees_h,
            "fov_degrees_v": self.fov_degrees_v,
            "pan_speed_factor": self.pan_speed_factor,
            "updated_at": time.time(),
        }
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────
    # Homing & Zero-Point Calibration
    # ──────────────────────────────────────────────

    def home(self, speed: float = 0.8) -> Dict[str, Any]:
        """
        Quy trình chuẩn hoá góc (Homing):
        Quay kịch trái -> Gán Pan = -1.0
        Quay kịch dưới -> Gán Tilt = -1.0
        Quay về vị trí trung tâm (0.0, 0.0)
        """
        with self._lock:
            # 1. Quay kịch trái
            self.client.continuous_move(-speed, 0.0)
            time.sleep(self.full_pan_time + 0.8)
            self.client.stop()
            time.sleep(0.3)
            self.virtual_pan = -1.0

            # 2. Quay kịch dưới
            self.client.continuous_move(0.0, -speed)
            time.sleep(self.full_tilt_time + 0.5)
            self.client.stop()
            time.sleep(0.3)
            self.virtual_tilt = -1.0

            # 3. Quay về trung tâm (0.0, 0.0)
            center_pan_time = self.full_pan_time / 2.0
            center_tilt_time = self.full_tilt_time / 2.0

            # Quay phải về tâm
            self.client.continuous_move(speed, 0.0)
            time.sleep(center_pan_time)
            self.client.stop()
            time.sleep(0.2)

            # Quay lên về tâm
            self.client.continuous_move(0.0, speed)
            time.sleep(center_tilt_time)
            self.client.stop()
            time.sleep(0.2)

            self.virtual_pan = 0.0
            self.virtual_tilt = 0.0
            self.is_homed = True

        return {
            "is_homed": True,
            "virtual_pan": self.virtual_pan,
            "virtual_tilt": self.virtual_tilt,
        }

    # ──────────────────────────────────────────────
    # Continuous Movement Tracker
    # ──────────────────────────────────────────────

    def start_move(self, pan_dir: float, tilt_dir: float, speed: float = 0.4):
        """Bắt đầu quay và ghi nhận mốc thời gian để tính toạ độ ảo."""
        with self._lock:
            # Nếu đang quay thì chốt quãng đường đã di chuyển trước
            if self._moving:
                self._update_position_from_move()

            self._moving = True
            self._move_dir = (pan_dir, tilt_dir)
            self._move_speed = speed
            self._move_start_time = time.time()
            self.client.continuous_move(pan_dir * speed, tilt_dir * speed)

    def stop_move(self):
        """Dừng quay và cập nhật toạ độ ảo."""
        with self._lock:
            if self._moving:
                self._update_position_from_move()
                self._moving = False
            self.client.stop()

    def _update_position_from_move(self):
        """Tính toán độ dời toạ độ dựa trên thời gian bấm giữ."""
        elapsed = time.time() - self._move_start_time
        if elapsed <= 0:
            return

        pan_dir, tilt_dir = self._move_dir
        # delta_pan: 2.0 toàn dải chia cho full_pan_time (tỉ lệ theo speed)
        delta_pan = pan_dir * (2.0 / self.full_pan_time) * (self._move_speed / 0.8) * elapsed
        delta_tilt = tilt_dir * (2.0 / self.full_tilt_time) * (self._move_speed / 0.8) * elapsed

        self.virtual_pan = max(-1.0, min(1.0, self.virtual_pan + delta_pan))
        self.virtual_tilt = max(-1.0, min(1.0, self.virtual_tilt + delta_tilt))

    # ──────────────────────────────────────────────
    # Goto Virtual Position & Click-to-Center
    # ──────────────────────────────────────────────

    def goto_virtual(self, target_pan: float, target_tilt: float, speed: float = 0.8):
        """
        Quay camera tới toạ độ ảo xác định (target_pan, target_tilt) ∈ [-1.0, 1.0]
        bằng Timed ContinuousMove.
        """
        with self._lock:
            if not self.is_homed:
                # Tự động home nếu chưa khởi tạo
                pass

            target_pan = max(-1.0, min(1.0, target_pan))
            target_tilt = max(-1.0, min(1.0, target_tilt))

            delta_pan = target_pan - self.virtual_pan
            delta_tilt = target_tilt - self.virtual_tilt

            # 1. Quay Pan
            if abs(delta_pan) > 0.02:
                pan_dir = 1.0 if delta_pan > 0 else -1.0
                t_pan = (abs(delta_pan) / 2.0) * self.full_pan_time * (0.8 / speed)
                self.client.continuous_move(pan_dir * speed, 0.0)
                time.sleep(t_pan)
                self.client.stop()
                time.sleep(0.2)
                self.virtual_pan = target_pan

            # 2. Quay Tilt
            if abs(delta_tilt) > 0.02:
                tilt_dir = 1.0 if delta_tilt > 0 else -1.0
                t_tilt = (abs(delta_tilt) / 2.0) * self.full_tilt_time * (0.8 / speed)
                self.client.continuous_move(0.0, tilt_dir * speed)
                time.sleep(t_tilt)
                self.client.stop()
                time.sleep(0.2)
                self.virtual_tilt = target_tilt

    def click_aim(self, click_x: float, click_y: float, frame_w: float = 1280.0, frame_h: float = 720.0):
        """
        Click-to-Center đưa pixel (click_x, click_y) về trung tâm khung hình.
        Tính góc lệch dựa trên FOV camera -> Chuyển thành thời gian quay chính xác.
        """
        # Độ lệch pixel so với tâm
        dx_px = click_x - (frame_w / 2.0)
        dy_px = (frame_h / 2.0) - click_y   # Đảo trục Y (ảnh vs góc quay)

        # Đổi ra góc (độ)
        angle_pan = (dx_px / frame_w) * self.fov_degrees_h
        angle_tilt = (dy_px / frame_h) * self.fov_degrees_v

        # Thời gian quay cần thiết tại speed=0.6
        speed = 0.6
        deg_per_sec_pan = self.pan_speed_factor * (speed / 0.8)
        deg_per_sec_tilt = (180.0 / self.full_tilt_time) * (speed / 0.8)

        t_pan = abs(angle_pan) / deg_per_sec_pan if deg_per_sec_pan > 0 else 0
        t_tilt = abs(angle_tilt) / deg_per_sec_tilt if deg_per_sec_tilt > 0 else 0

        # Giới hạn an toàn [0.05s, 2.0s]
        if t_pan > 0.05:
            pan_dir = 1.0 if angle_pan > 0 else -1.0
            self.client.continuous_move(pan_dir * speed, 0.0)
            time.sleep(min(1.5, t_pan))
            self.client.stop()
            time.sleep(0.2)
            delta_pan = (pan_dir * min(1.5, t_pan) / self.full_pan_time) * 2.0 * (speed / 0.8)
            self.virtual_pan = max(-1.0, min(1.0, self.virtual_pan + delta_pan))

        if t_tilt > 0.05:
            tilt_dir = 1.0 if angle_tilt > 0 else -1.0
            self.client.continuous_move(0.0, tilt_dir * speed)
            time.sleep(min(1.2, t_tilt))
            self.client.stop()
            time.sleep(0.2)
            delta_tilt = (tilt_dir * min(1.2, t_tilt) / self.full_tilt_time) * 2.0 * (speed / 0.8)
            self.virtual_tilt = max(-1.0, min(1.0, self.virtual_tilt + delta_tilt))

    # ──────────────────────────────────────────────
    # Coordinate Conversion Formulas
    # ──────────────────────────────────────────────

    def virtual_to_physical_angles(self, pan_val: float, tilt_val: float) -> Tuple[float, float]:
        """
        Chuyển đổi toạ độ ảo [-1.0, 1.0] sang góc vật lý thực tế (độ):
          - pan_deg ∈ [0.0, total_pan_range_deg] (0° -> 366°)
          - tilt_deg ∈ [tilt_min_deg, tilt_max_deg] (-5° -> 80°)
        """
        pan_clamped = max(-1.0, min(1.0, pan_val))
        tilt_clamped = max(-1.0, min(1.0, tilt_val))

        pan_deg = ((pan_clamped + 1.0) / 2.0) * self.total_pan_range_deg
        tilt_deg = self.tilt_min_deg + ((tilt_clamped + 1.0) / 2.0) * (self.tilt_max_deg - self.tilt_min_deg)
        return round(pan_deg, 2), round(tilt_deg, 2)

    def physical_angles_to_virtual(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        """
        Chuyển đổi góc vật lý thực tế (độ) sang toạ độ ảo [-1.0, 1.0].
        """
        pan_val = (pan_deg / self.total_pan_range_deg) * 2.0 - 1.0
        tilt_range = self.tilt_max_deg - self.tilt_min_deg
        tilt_val = ((tilt_deg - self.tilt_min_deg) / (tilt_range if tilt_range > 0 else 1.0)) * 2.0 - 1.0
        return round(max(-1.0, min(1.0, pan_val)), 4), round(max(-1.0, min(1.0, tilt_val)), 4)

    def goto_angle(self, target_pan_deg: float, target_tilt_deg: float, speed: float = 0.8):
        """
        Quay camera tới góc vật lý thực tế (target_pan_deg, target_tilt_deg).
        Tự động xử lý cơ chế Bounded Clamping hoặc Shortest Path tuỳ theo pan_type.
        """
        # Nếu là bounded_stops: Clamp trong dải vật lý cho phép
        if getattr(self, "pan_type", "bounded_stops") == "bounded_stops" or not getattr(self, "allow_zero_wrap_around", False):
            clamped_pan = max(0.0, min(self.total_pan_range_deg, target_pan_deg))
        else:
            clamped_pan = target_pan_deg % self.total_pan_range_deg

        clamped_tilt = max(self.tilt_min_deg, min(self.tilt_max_deg, target_tilt_deg))
        v_pan, v_tilt = self.physical_angles_to_virtual(clamped_pan, clamped_tilt)
        self.goto_virtual(v_pan, v_tilt, speed=speed)

    def get_status(self) -> Dict[str, Any]:
        """Trả về toạ độ ảo thời gian thực kèm góc vật lý thực tế."""
        pan_deg, tilt_deg = self.virtual_to_physical_angles(self.virtual_pan, self.virtual_tilt)
        return {
            "pan": round(self.virtual_pan, 3),
            "tilt": round(self.virtual_tilt, 3),
            "zoom": 0.0,
            "pan_deg": pan_deg,
            "tilt_deg": tilt_deg,
            "pan_tilt_status": "MOVING" if self._moving else "IDLE",
            "is_homed": self.is_homed,
            "full_pan_time": self.full_pan_time,
            "full_tilt_time": self.full_tilt_time,
            "fov_degrees_h": self.fov_degrees_h,
            "fov_degrees_v": self.fov_degrees_v,
            "total_pan_range_deg": self.total_pan_range_deg,
            "tilt_min_deg": self.tilt_min_deg,
            "tilt_max_deg": self.tilt_max_deg,
        }
