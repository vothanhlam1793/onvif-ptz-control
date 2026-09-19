"""
Coordinate mapping: Pixel (u, v) -> ONVIF Pan/Tilt normalized vector [-1.0, 1.0].

Dùng cho:
  - Click-to-Center (Web UI click chuột vào khung hình)
  - AI Aim (BBox từ detector -> điểm trung tâm mục tiêu -> lệch so với tâm hình)
"""


def pixel_to_relative_pantilt(
    click_x: float,
    click_y: float,
    frame_width: float,
    frame_height: float,
    sensitivity: float = 1.0,
) -> tuple[float, float]:
    """
    Chuyển toạ độ pixel click trên khung hình -> vector RelativeMove (pan, tilt).

    Hệ trục:
      - click_x == frame_width/2  -> pan delta = 0 (tâm)
      - click_x > center          -> pan delta > 0 (quay phải)
      - click_y == frame_height/2 -> tilt delta = 0 (tâm)
      - click_y < center          -> tilt delta > 0 (quay lên)
        (trục Y ảnh ngược chiều trục tilt ONVIF)

    sensitivity: hệ số nhân (0.1 - 1.0). Nhỏ = dịch chuyển nhỏ hơn mỗi click.
    """
    dx = (2.0 * click_x - frame_width)  / frame_width   # [-1.0, 1.0]
    dy = (frame_height - 2.0 * click_y) / frame_height  # [-1.0, 1.0] (đảo chiều Y)

    pan  = max(-1.0, min(1.0, dx * sensitivity))
    tilt = max(-1.0, min(1.0, dy * sensitivity))
    return pan, tilt


def bbox_center_to_relative_pantilt(
    x1: float, y1: float, x2: float, y2: float,
    frame_width: float,
    frame_height: float,
    sensitivity: float = 1.0,
) -> tuple[float, float]:
    """
    Tính vector RelativeMove để đưa trung tâm BBox vào tâm hình.
    Dùng cho AI Aim: detector trả về [x1, y1, x2, y2] pixel.
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return pixel_to_relative_pantilt(cx, cy, frame_width, frame_height, sensitivity)


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
