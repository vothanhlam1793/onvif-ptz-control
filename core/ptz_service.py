"""
High-level PTZ Service.
Wraps OnvifClient với các hành động cấp cao: D-pad, Click-to-Center, AI Aim, Patrol.
Đây là interface mà API routes và AI Agent gọi vào.
"""

import threading
import time
from typing import Optional

from core.onvif_client import OnvifClient
from core.coordinate import pixel_to_relative_pantilt, bbox_center_to_relative_pantilt
from core.virtual_ptz import VirtualPTZTracker


# Tốc độ mặc định cho D-pad
DEFAULT_SPEED = 0.4
MAX_SPEED     = 1.0


class PTZService:
    def __init__(self, client: OnvifClient, rtsp_url: str = ""):
        self.cam = client
        self.rtsp_url = rtsp_url
        self._speed = DEFAULT_SPEED
        self.virtual_tracker = VirtualPTZTracker(client, rtsp_url) if rtsp_url else None
        self._continuous_timer: Optional[threading.Timer] = None

    # ──────────────────────────────────────────────
    # D-pad (Web UI keyboard / button)
    # ──────────────────────────────────────────────

    def move(self, direction: str, speed: Optional[float] = None):
        """
        Continuous move theo 8 hướng + zoom.
        direction: left | right | up | down | up-left | up-right | down-left | down-right | zoom-in | zoom-out
        """
        s = speed if speed is not None else self._speed
        mapping = {
            "left":       (-s,  0.0, 0.0),
            "right":      ( s,  0.0, 0.0),
            "up":         ( 0.0,  s, 0.0),
            "down":       ( 0.0, -s, 0.0),
            "up-left":    (-s,   s, 0.0),
            "up-right":   ( s,   s, 0.0),
            "down-left":  (-s,  -s, 0.0),
            "down-right": ( s,  -s, 0.0),
            "zoom-in":    ( 0.0, 0.0,  s),
            "zoom-out":   ( 0.0, 0.0, -s),
        }
        if direction not in mapping:
            raise ValueError(f"Unknown direction: {direction}")
        pan, tilt, zoom = mapping[direction]
        self.cam.continuous_move(pan, tilt, zoom)

    def stop(self):
        self.cam.stop()

    def set_speed(self, speed: float):
        self._speed = max(0.05, min(MAX_SPEED, speed))

    # ──────────────────────────────────────────────
    # Click-to-Center (Web UI click on video)
    # ──────────────────────────────────────────────

    def click_to_center(
        self,
        click_x: float,
        click_y: float,
        frame_width: float,
        frame_height: float,
        sensitivity: float = 0.8,
    ):
        """
        User click vào điểm (click_x, click_y) trên video frame.
        Camera RelativeMove để đưa điểm đó về tâm hình.
        """
        pan, tilt = pixel_to_relative_pantilt(
            click_x, click_y, frame_width, frame_height, sensitivity
        )
        move_speed = min(0.8, max(0.2, abs(pan) + abs(tilt)) / 2)
        self.cam.relative_move(pan, tilt, pan_speed=move_speed, tilt_speed=move_speed)

    # ──────────────────────────────────────────────
    # AI Aim (Phase 2: detector/VLM gọi vào)
    # ──────────────────────────────────────────────

    def aim_at_bbox(
        self,
        x1: float, y1: float, x2: float, y2: float,
        frame_width: float,
        frame_height: float,
        sensitivity: float = 1.0,
    ):
        """
        Nhận BBox từ AI Detector/VLM -> quay camera nhanh nhất có thể
        để đưa trung tâm BBox vào tâm hình.
        Dùng tốc độ tối đa (1.0) cho phản ứng nhanh nhất.
        """
        pan, tilt = bbox_center_to_relative_pantilt(
            x1, y1, x2, y2, frame_width, frame_height, sensitivity
        )
        self.cam.relative_move(pan, tilt, pan_speed=1.0, tilt_speed=1.0)

    def aim_at_point(
        self,
        x: float, y: float,
        frame_width: float,
        frame_height: float,
    ):
        """Aim vào 1 điểm pixel. Shortcut cho AI gọi nhanh."""
        pan, tilt = pixel_to_relative_pantilt(x, y, frame_width, frame_height, sensitivity=1.0)
        self.cam.relative_move(pan, tilt, pan_speed=1.0, tilt_speed=1.0)

    # ──────────────────────────────────────────────
    # Presets
    # ──────────────────────────────────────────────

    def list_presets(self) -> list[dict]:
        return self.cam.get_presets()

    def save_preset(self, name: str, token: Optional[str] = None) -> str:
        return self.cam.set_preset(name, token)

    def goto_preset(self, token: str, speed: float = 1.0):
        self.cam.goto_preset(token, speed)

    def remove_preset(self, token: str):
        self.cam.remove_preset(token)

    # ──────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────

    def status(self) -> dict:
        s = self.cam.get_status()
        s["speed"] = self._speed
        if self.virtual_tracker:
            v_stat = self.virtual_tracker.get_status()
            s["virtual_pan"] = v_stat["pan"]
            s["virtual_tilt"] = v_stat["tilt"]
            s["is_homed"] = v_stat["is_homed"]
            s["mechanical_calibration_confirmed"] = v_stat["mechanical_calibration_confirmed"]
        return s

    # ──────────────────────────────────────────────
    # Patrol Tour
    # ──────────────────────────────────────────────

    def patrol_presets(self, preset_tokens: list[str], dwell_sec: float = 3.0):
        """
        Tuần tra qua danh sách preset theo thứ tự, dừng lại dwell_sec giây mỗi góc.
        Chạy non-blocking trên thread riêng.
        Trả về stop_event để caller dừng patrol khi cần.
        """
        stop_event = threading.Event()

        def _run():
            while not stop_event.is_set():
                for token in preset_tokens:
                    if stop_event.is_set():
                        break
                    try:
                        self.cam.goto_preset(token, speed=1.0)
                    except Exception as e:
                        print(f"[patrol] Error goto_preset {token}: {e}")
                    stop_event.wait(timeout=dwell_sec)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return stop_event
