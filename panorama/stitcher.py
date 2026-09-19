"""
Panorama module - stitcher.py
Ghép N JPEG frames thành 1 ảnh panorama toàn cảnh chất lượng cao:
- Primary: OpenCV Stitcher_PANORAMA với Cylindrical warping.
- Fallback nâng cao: Cylindrical Projection + SIFT/RANSAC Translation + Linear Ramp Alpha Blending
  (Tránh hoàn toàn lỗi lặp lại vật thể hoặc sai cấu trúc không gian).
- Lưu cả panorama và grid ảnh riêng lẻ.
"""

import os
import time
import math
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np


from panorama.spherical_projector import stitch_ptz_guided

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "panorama"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_camera_matrix_and_dist(h: int, w: int, hfov_deg: float = 85.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ước tính ma trận nội tại Camera K và hệ số méo ống kính k1, k2 (Barrel Distortion).
    Ống kính an ninh góc rộng 85° thường có hệ số méo viền ~ -0.15 đến -0.18.
    """
    f = (w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    K = np.array([
        [f, 0, w / 2.0],
        [0, f, h / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float32)
    # k1 = -0.14 (khử méo phình cạnh), k2 = 0.02
    dist_coeffs = np.array([-0.14, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)
    return K, dist_coeffs


def undistort_image(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Khử cong đường thẳng mép ảnh."""
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, K, dist, None, new_K)


def spherical_warp(img: np.ndarray, f: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiếu ảnh phẳng sang mặt cầu (Spherical Surface) để bảo toàn góc nhìn 3D đa chiều.
    """
    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w))
    x_c = x_i - w / 2.0
    y_c = y_i - h / 2.0

    theta = x_c / f
    phi = y_c / f

    x_p = np.sin(theta) * np.cos(phi)
    y_p = np.sin(phi)
    z_p = np.cos(theta) * np.cos(phi)

    x_map = f * (x_p / z_p) + w / 2.0
    y_map = f * (y_p / z_p) + h / 2.0

    warped = cv2.remap(img, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = (warped > 0).astype(np.uint8) * 255
    return warped, mask


def _align_and_blend_pair(img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Tìm độ dịch chuyển ngang Δx, Δy giữa 2 ảnh bằng SIFT/ORB và ghép hòa trộn (Alpha Blending).
    """
    h, w = img_left.shape[:2]
    sift = cv2.SIFT_create()

    g_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(g_l, None)
    kp2, des2 = sift.detectAndCompute(g_r, None)

    dx = int(w * 0.65)  # fallback 35% overlap
    dy = 0

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            dx_vals = pts1[:, 0] - pts2[:, 0]
            dy_vals = pts1[:, 1] - pts2[:, 1]

            valid_idx = dx_vals > 50
            if np.count_nonzero(valid_idx) >= 5:
                dx = int(np.median(dx_vals[valid_idx]))
                dy = int(np.median(dy_vals[valid_idx]))

    out_w = dx + w
    out_h = h + abs(dy)
    y_offset_l = max(0, dy)
    y_offset_r = max(0, -dy)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[y_offset_l:y_offset_l + h, :w] = img_left

    overlap_w = w - dx
    if overlap_w > 0:
        for col in range(overlap_w):
            alpha = col / float(overlap_w)
            x_pos = dx + col
            l_px = img_left[:, x_pos].astype(np.float32)
            r_px = img_right[:, col].astype(np.float32)
            blended = (1.0 - alpha) * l_px + alpha * r_px
            canvas[y_offset_r:y_offset_r + h, x_pos] = np.clip(blended, 0, 255).astype(np.uint8)

        canvas[y_offset_r:y_offset_r + h, w:] = img_right[:, overlap_w:]
    else:
        canvas[y_offset_r:y_offset_r + h, dx:dx + w] = img_right

    return canvas, float(dx)


def _align_and_blend_vertical(img_top: np.ndarray, img_bottom: np.ndarray) -> np.ndarray:
    """
    Ghép 2 tầng Panorama theo trục dọc Y bằng SIFT và Linear Vertical Ramp Blending.
    """
    # Đồng bộ chiều rộng 2 tầng trước khi ghép
    h_top, w_top = img_top.shape[:2]
    h_bot, w_bot = img_bottom.shape[:2]
    w_target = max(w_top, w_bot)

    if w_top < w_target:
        pad = np.zeros((h_top, w_target - w_top, 3), dtype=np.uint8)
        img_top = np.hstack([img_top, pad])
    if w_bot < w_target:
        pad = np.zeros((h_bot, w_target - w_bot, 3), dtype=np.uint8)
        img_bottom = np.hstack([img_bottom, pad])

    sift = cv2.SIFT_create()
    g_t = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
    g_b = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(g_t, None)
    kp2, des2 = sift.detectAndCompute(g_b, None)

    dy = int(h_top * 0.70)  # fallback 30% vertical overlap
    dx = 0

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            dy_vals = pts1[:, 1] - pts2[:, 1]
            dx_vals = pts1[:, 0] - pts2[:, 0]

            valid_idx = dy_vals > 50
            if np.count_nonzero(valid_idx) >= 5:
                dy = int(np.median(dy_vals[valid_idx]))
                dx = int(np.median(dx_vals[valid_idx]))

    out_h = dy + h_bot
    out_w = w_target + abs(dx)
    x_offset_t = max(0, dx)
    x_offset_b = max(0, -dx)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:h_top, x_offset_t:x_offset_t + w_target] = img_top

    overlap_h = h_top - dy
    if overlap_h > 0:
        for row in range(overlap_h):
            alpha = row / float(overlap_h)
            y_pos = dy + row
            t_px = img_top[y_pos, :].astype(np.float32)
            b_px = img_bottom[row, :].astype(np.float32)
            blended = (1.0 - alpha) * t_px + alpha * b_px
            canvas[y_pos, x_offset_b:x_offset_b + w_target] = np.clip(blended, 0, 255).astype(np.uint8)

        canvas[h_top:, x_offset_b:x_offset_b + w_target] = img_bottom[overlap_h:, :]
    else:
        canvas[dy:dy + h_bot, x_offset_b:x_offset_b + w_target] = img_bottom

    return canvas


def advanced_cylindrical_stitch(imgs: List[np.ndarray], hfov_deg: float = 85.0, grid_rows: int = 1) -> bytes:
    """
    Thuật toán ghép toàn cảnh:
    - 1 hàng: Khử méo -> Chiếu mặt cầu Spherical -> Ghép ngang lũy tiến SIFT.
    - 2 hàng: Tách hàng -> Ghép ngang từng hàng -> Ghép dọc 2 hàng hoàn chỉnh.
    """
    h, w = imgs[0].shape[:2]
    K, dist = estimate_camera_matrix_and_dist(h, w, hfov_deg)
    f = K[0, 0]

    # 1. Khử méo ống kính (Undistort) + Chiếu mặt cầu (Spherical Warp)
    processed_imgs = []
    for img in imgs:
        undist = undistort_image(img, K, dist)
        sph, _ = spherical_warp(undist, f)
        processed_imgs.append(sph)

    # 2. Phân loại theo số hàng grid
    n = len(processed_imgs)
    if grid_rows > 1 and n % grid_rows == 0:
        cols_per_row = n // grid_rows
        row_panos = []
        for r in range(grid_rows):
            row_slice = processed_imgs[r * cols_per_row:(r + 1) * cols_per_row]
            row_pano = row_slice[0]
            for i in range(1, len(row_slice)):
                row_pano, _ = _align_and_blend_pair(row_pano, row_slice[i])
            row_panos.append(row_pano)

        # Ghép tầng trên xuống tầng dưới (r=1 là Upper/Ceiling, r=0 là Lower/Horizon)
        # Thứ tự sweep 2D: row 0 = Lower, row 1 = Upper
        # Ta đặt row 1 (Upper) ở trên, row 0 (Lower) ở dưới
        pano = _align_and_blend_vertical(img_top=row_panos[1], img_bottom=row_panos[0])
    else:
        # Ghép lũy tiến chuỗi 1 hàng
        pano = processed_imgs[0]
        for i in range(1, len(processed_imgs)):
            pano, _ = _align_and_blend_pair(pano, processed_imgs[i])

    # 3. Crop viền đen thừa
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, cw, ch = cv2.boundingRect(contours[0])
        if cw > 100 and ch > 100:
            pano = pano[y:y + ch, x:x + cw]

    _, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return bytes(buf)


def stitch(
    frames_bytes: list[bytes],
    grid_rows: int = 1,
    pan_angles: Optional[List[float]] = None,
    tilt_angles: Optional[List[float]] = None,
    hfov_deg: float = 85.0,
    dist_coeffs: Optional[np.ndarray] = None,
) -> tuple[Optional[bytes], Optional[bytes]]:
    """
    Nhận list JPEG bytes -> trả về (panorama_bytes, grid_bytes).
    Nếu cung cấp pan_angles và tilt_angles: Dùng thuật toán Spherical Projection chuẩn xác tuyệt đối.
    """
    if not frames_bytes:
        return None, None

    imgs = []
    for b in frames_bytes:
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)

    if not imgs:
        return None, None

    if len(imgs) == 16 and grid_rows == 1:
        grid_rows = 2

    # --- 1. Grid ---
    grid_bytes = _make_grid(imgs, grid_rows=grid_rows)

    if len(imgs) == 1:
        _, pano_bytes = cv2.imencode(".jpg", imgs[0], [cv2.IMWRITE_JPEG_QUALITY, 90])
        return bytes(pano_bytes), grid_bytes

    # --- 2. PTZ-Guided Spherical Stitch (Ưu tiên số 1 nếu có toạ độ góc) ---
    if pan_angles is not None and tilt_angles is not None and len(pan_angles) == len(imgs) and len(tilt_angles) == len(imgs):
        try:
            print("[stitcher] Áp dụng PTZ-Guided Spherical Projection...")
            pano = stitch_ptz_guided(
                frames=imgs,
                pan_angles=pan_angles,
                tilt_angles=tilt_angles,
                hfov_deg=hfov_deg,
                dist_coeffs=dist_coeffs,
                out_w=3840,
                out_h=1440
            )
            _, pano_bytes = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return bytes(pano_bytes), grid_bytes
        except Exception as e:
            print(f"[stitcher] Lỗi PTZ-Guided: {e}. Chuyển sang fallback...")

    # --- 3. OpenCV Stitcher (Chế độ PANORAMA chuẩn) ---
    try:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(imgs)
        if status == cv2.Stitcher_OK:
            print("[stitcher] OpenCV Stitcher_PANORAMA thành công mỹ mãn.")
            _, pano_bytes = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return bytes(pano_bytes), grid_bytes
        else:
            print(f"[stitcher] Stitcher_PANORAMA code={status}. Chuyển sang 2D Multi-Row Cylindrical & Seam Blending...")
    except Exception as e:
        print(f"[stitcher] Stitcher exception: {e}. Chuyển sang 2D Multi-Row Cylindrical & Seam Blending...")

    # --- 3. Fallback: Multi-Row Cylindrical Warping + SIFT 2D Alignment + Blending ---
    try:
        pano_bytes = advanced_cylindrical_stitch(imgs, hfov_deg=85.0, grid_rows=grid_rows)
        return pano_bytes, grid_bytes
    except Exception as e:
        print(f"[stitcher] Advanced stitch error: {e}, fallback simple concat.")
        pano_bytes = _simple_concat(imgs)
        return pano_bytes, grid_bytes


def _make_grid(imgs: list[np.ndarray], grid_rows: int = 1, grid_cols: Optional[int] = None) -> bytes:
    """
    Ghép ảnh thành lưới trực quan (grid) đúng cấu trúc quét:
    - Nếu 2 tầng (16 ảnh): 2 hàng x 8 cột (Hàng 1: Upper/Ceiling, Hàng 2: Lower/Horizon).
    - Mặc định: cols x rows theo tham số hoặc tự động tính.
    """
    n = len(imgs)
    if grid_cols is None:
        if grid_rows > 1 and n % grid_rows == 0:
            grid_cols = n // grid_rows
        elif n == 8:
            grid_rows = 1
            grid_cols = 8
        elif n == 16:
            grid_rows = 2
            grid_cols = 8
        else:
            grid_cols = math.ceil(math.sqrt(n))
            grid_rows = math.ceil(n / grid_cols)

    thumb_w, thumb_h = 480, 270
    thumbs = [cv2.resize(img, (thumb_w, thumb_h)) for img in imgs]

    while len(thumbs) < grid_rows * grid_cols:
        thumbs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

    # Nếu đúng cấu trúc 2 tầng (tầng 0: lower, tầng 1: upper), đảo tầng 1 lên trên để hiển thị tự nhiên
    ordered_thumbs = thumbs
    if grid_rows == 2 and n == 16:
        row_lower = thumbs[0:8]
        row_upper = thumbs[8:16]
        ordered_thumbs = row_upper + row_lower

    grid_row_list = []
    for r in range(grid_rows):
        row_imgs = ordered_thumbs[r * grid_cols:(r + 1) * grid_cols]
        grid_row_list.append(np.hstack(row_imgs))
    grid = np.vstack(grid_row_list)

    _, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf)


def _simple_concat(imgs: list[np.ndarray]) -> bytes:
    """Ghép ngang đơn giản."""
    h_target = min(img.shape[0] for img in imgs)
    resized = []
    for img in imgs:
        scale = h_target / img.shape[0]
        w = int(img.shape[1] * scale)
        resized.append(cv2.resize(img, (w, h_target)))
    pano = np.hstack(resized)
    _, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf)


def save_results(
    frames_bytes: list[bytes],
    panorama_bytes: Optional[bytes],
    grid_bytes: Optional[bytes],
) -> dict:
    """Lưu tất cả file ra disk."""
    ts = int(time.time())
    run_dir = OUTPUT_DIR / str(ts)
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {"frames": [], "panorama": None, "grid": None, "timestamp": ts}

    for i, fb in enumerate(frames_bytes):
        p = run_dir / f"frame_{i:02d}.jpg"
        p.write_bytes(fb)
        paths["frames"].append(str(p))

    if panorama_bytes:
        p = run_dir / "panorama.jpg"
        p.write_bytes(panorama_bytes)
        paths["panorama"] = str(p)

    if grid_bytes:
        p = run_dir / "grid.jpg"
        p.write_bytes(grid_bytes)
        paths["grid"] = str(p)

    return paths
