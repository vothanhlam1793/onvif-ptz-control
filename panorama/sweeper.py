"""
Panorama module - sweeper.py
Thực hiện sweep camera từ trái qua phải, chụp N frames.
Tích hợp VirtualPTZTracker để xác định toạ độ tuyệt đối ảo [-1.0, 1.0] chính xác theo thời gian & góc.
Fallback linh hoạt sang OnvifClient trực tiếp nếu chưa truyền tracker.
Phát progress qua callback để UI cập nhật realtime.
"""

import time
import math
from typing import Callable, Optional, Any

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot


def wait_idle(client: OnvifClient, timeout: float = 5.0):
    """Chờ camera dừng hẳn (MoveStatus == IDLE)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = client.get_status()
            if st.get("pan_tilt_status") == "IDLE":
                return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def move_to_far_left(client: OnvifClient, tracker: Optional[Any] = None, duration_sec: float = 3.5):
    """Quay camera về kịch biên trái."""
    if tracker:
        tracker.goto_virtual(-0.95, tracker.virtual_tilt, speed=0.8)
        return

    try:
        client.absolute_move(-0.95, 0.0, speed=0.9)
        wait_idle(client, timeout=4.0)
    except Exception:
        client.continuous_move(-0.7, 0.0)
        time.sleep(duration_sec)
        client.stop()
        time.sleep(0.5)


def move_to_far_right(client: OnvifClient, tracker: Optional[Any] = None, duration_sec: float = 3.5):
    """Quay camera về kịch biên phải."""
    if tracker:
        tracker.goto_virtual(0.95, tracker.virtual_tilt, speed=0.8)
        return

    try:
        client.absolute_move(0.95, 0.0, speed=0.9)
        wait_idle(client, timeout=4.0)
    except Exception:
        client.continuous_move(0.7, 0.0)
        time.sleep(duration_sec)
        client.stop()
        time.sleep(0.5)


def sweep_and_capture(
    client: OnvifClient,
    rtsp_url: str,
    tracker: Optional[Any] = None,
    num_steps: int = 8,
    tilt_pos: float = -0.80,
    pan_min: float = -0.9,
    pan_max: float = 0.9,
    dwell_sec: float = 0.8,
    snapshot_w: int = 1920,
    snapshot_h: int = 1080,
    on_progress: Optional[Callable[[int, int, bytes], None]] = None,
) -> list[bytes]:
    """
    Quét camera 1 dải ngang tại vị trí tilt_pos (Single-Row Horizon Panorama).
    """
    return sweep_2d_grid(
        client=client,
        rtsp_url=rtsp_url,
        tracker=tracker,
        tilt_rows=[tilt_pos],
        cols=num_steps,
        pan_min=pan_min,
        pan_max=pan_max,
        dwell_sec=dwell_sec,
        snapshot_w=snapshot_w,
        snapshot_h=snapshot_h,
        on_progress=on_progress,
    )


def sweep_2d_grid(
    client: OnvifClient,
    rtsp_url: str,
    tracker: Optional[Any] = None,
    tilt_rows: Optional[list[float]] = None,
    cols: int = 8,
    pan_min: float = -0.9,
    pan_max: float = 0.9,
    dwell_sec: float = 0.8,
    snapshot_w: int = 1920,
    snapshot_h: int = 1080,
    on_progress: Optional[Callable[[int, int, bytes], None]] = None,
) -> list[bytes]:
    """
    Quét không gian 2D theo lưới nhiều tầng (Multi-row Pan & Tilt Grid).
    tilt_rows: Danh sách mức Tilt cần quét (mặc định 2 tầng: Lower Horizon = -0.80, Upper Ceiling = -0.10).
    cols: Số bước quét ngang trên mỗi tầng (mặc định 8 frames).
    """
    if tilt_rows is None:
        tilt_rows = [-0.80, -0.10]

    total_frames = len(tilt_rows) * cols
    frames: list[bytes] = []
    current_frame_idx = 0

    step_pan_delta = (pan_max - pan_min) / max(1, cols - 1)
    pan_coords = [pan_min + i * step_pan_delta for i in range(cols)]

    for row_idx, tilt_val in enumerate(tilt_rows):
        print(f"\n[sweep_2d] === Bắt đầu quét tầng {row_idx + 1}/{len(tilt_rows)} (Tilt = {tilt_val:.2f}) ===")
        
        # Di chuyển tới mốc Pan đầu tiên của hàng và hạ/nâng Tilt
        if tracker:
            tracker.goto_virtual(pan_coords[0], tilt_val, speed=0.8)
        else:
            move_to_far_left(client)
        time.sleep(dwell_sec + 0.2)

        for col_idx in range(cols):
            pan_val = pan_coords[col_idx]
            current_frame_idx += 1
            print(f"[sweep_2d] Chụp frame {current_frame_idx}/{total_frames} (Row {row_idx+1}, Col {col_idx+1} -> Pan={pan_val:.2f}, Tilt={tilt_val:.2f})...")
            
            if col_idx > 0:
                if tracker:
                    tracker.goto_virtual(pan_val, tilt_val, speed=0.7)
                else:
                    client.continuous_move(0.5, 0.0)
                    time.sleep(0.5)
                    client.stop()
                    time.sleep(0.3)
                time.sleep(dwell_sec)

            frame = snapshot(rtsp_url, width=snapshot_w, height=snapshot_h)
            if frame:
                frames.append(frame)
                if on_progress:
                    on_progress(current_frame_idx, total_frames, frame)

    return frames


