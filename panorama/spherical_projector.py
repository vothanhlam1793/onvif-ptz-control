"""
Spherical Projector Module - panorama/spherical_projector.py
Ghép toàn cảnh Equirectangular Panorama theo toạ độ góc quay Ground-Truth PTZ:
- Khử méo ống kính (Undistortion).
- Chiếu cầu ngược (Backward Equirectangular Warping) với ma trận quay 3D R(pan, tilt).
- Hòa trộn mượt mà đa tầng bằng Feathered Distance Transform Blending.
- Hoàn toàn độc lập với SIFT/Feature matching (Xử lý hoàn hảo tường trơn & trần trắng).
"""

import math
from typing import List, Tuple, Optional
import cv2
import numpy as np


def euler_to_rotation_matrix(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """
    Tạo ma trận quay 3D R từ góc Pan (quay quanh trục Y) và Tilt (quay quanh trục X).
    pan_deg: Góc phương vị ngang (0 -> 360 độ).
    tilt_deg: Góc tà đứng (-90 -> +90 độ, dương là ngửa lên).
    """
    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)

    # R_pan: Quay quanh trục Y (trục đứng thẳng)
    R_pan = np.array([
        [math.cos(pan_rad), 0, math.sin(pan_rad)],
        [0, 1, 0],
        [-math.sin(pan_rad), 0, math.cos(pan_rad)]
    ], dtype=np.float64)

    # R_tilt: Quay quanh trục X (trục ngang)
    R_tilt = np.array([
        [1, 0, 0],
        [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
        [0, math.sin(tilt_rad), math.cos(tilt_rad)]
    ], dtype=np.float64)

    # R = R_pan * R_tilt (hướng nhìn camera)
    return R_pan @ R_tilt


def warp_image_to_equirectangular(
    img: np.ndarray,
    pan_deg: float,
    tilt_deg: float,
    hfov_deg: float = 85.0,
    dist_coeffs: Optional[np.ndarray] = None,
    out_w: int = 4096,
    out_h: int = 2048,
    lat_min_deg: float = -20.0,
    lat_max_deg: float = 85.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiếu 1 frame camera lên Canvas Equirectangular tại toạ độ (pan_deg, tilt_deg).
    Trả về (warped_image, weight_mask).
    """
    h_in, w_in = img.shape[:2]
    
    # 1. Ma trận camera K
    f = (w_in / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    K = np.array([
        [f, 0, w_in / 2.0],
        [0, f, h_in / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.array([-0.14, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)

    # Khử méo trước khi chiếu
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w_in, h_in), 0, (w_in, h_in))
    undist = cv2.undistort(img, K, dist_coeffs, None, new_K)

    # 2. Xây dựng ma trận góc quay R
    R = euler_to_rotation_matrix(pan_deg, tilt_deg)
    R_inv = R.T  # Ma trận quay ngược từ toạ độ cầu thế giới về toạ độ camera

    # 3. Tạo lưới toạ độ trên canvas Equirectangular
    # Giới hạn vùng tính toán theo Bounding Box để tăng tốc độ xử lý
    vfov_deg = hfov_deg * (h_in / float(w_in))
    pad_angle = 15.0

    # Phạm vi góc của frame này
    p_min = (pan_deg - hfov_deg / 2.0 - pad_angle) % 360.0
    p_max = (pan_deg + hfov_deg / 2.0 + pad_angle) % 360.0
    t_min = max(lat_min_deg, tilt_deg - vfov_deg / 2.0 - pad_angle)
    t_max = min(lat_max_deg, tilt_deg + vfov_deg / 2.0 + pad_angle)

    # Map sang toạ độ pixel trên canvas
    y_min_px = int(np.clip(out_h * (1.0 - (t_max - lat_min_deg) / (lat_max_deg - lat_min_deg)), 0, out_h - 1))
    y_max_px = int(np.clip(out_h * (1.0 - (t_min - lat_min_deg) / (lat_max_deg - lat_min_deg)), 0, out_h))

    # Xử lý toạ độ kinh độ X
    sub_x = np.arange(0, out_w, dtype=np.float32)
    sub_y = np.arange(y_min_px, y_max_px, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(sub_x, sub_y)

    # Đổi sang kinh độ (lon) và vĩ độ (lat)
    lon = (grid_x / float(out_w)) * 2.0 * math.pi  # 0 -> 2pi
    lat = (1.0 - grid_y / float(out_h)) * math.radians(lat_max_deg - lat_min_deg) + math.radians(lat_min_deg)

    # Toạ độ 3D trên mặt cầu đơn vị thế giới
    X_w = np.cos(lat) * np.sin(lon)
    Y_w = np.sin(lat)
    Z_w = np.cos(lat) * np.cos(lon)

    pts_w = np.stack([X_w, Y_w, Z_w], axis=-1)  # (sub_h, out_w, 3)

    # Chiếu ngược vào camera: P_c = R_inv * P_w
    pts_c = np.einsum("ij,abj->abi", R_inv, pts_w)

    # Lọc các điểm nằm phía trước camera (Z_c > 0)
    valid_z = pts_c[:, :, 2] > 0.05
    z_safe = np.where(valid_z, pts_c[:, :, 2], 1.0)

    # Chiếu vào cảm biến ảnh (u, v)
    u = (new_K[0, 0] * pts_c[:, :, 0] / z_safe + new_K[0, 2]).astype(np.float32)
    v = (new_K[1, 1] * pts_c[:, :, 1] / z_safe + new_K[1, 2]).astype(np.float32)

    # Remap pixel
    sub_warped = cv2.remap(undist, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Mặt nạ hợp lệ (nằm trong kích thước ảnh gốc)
    mask = (valid_z & (u >= 0) & (u < w_in) & (v >= 0) & (v < h_in)).astype(np.uint8) * 255

    # Đưa vào canvas toàn phần
    warped_full = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    warped_full[y_min_px:y_max_px, :] = sub_warped

    mask_full = np.zeros((out_h, out_w), dtype=np.uint8)
    mask_full[y_min_px:y_max_px, :] = mask

    # Tính trọng số hình nón (Distance transform) để hòa trộn mượt
    dist_map = cv2.distanceTransform(mask_full, cv2.DIST_L2, 5)
    max_d = dist_map.max()
    if max_d > 0:
        weight_map = (dist_map / max_d).astype(np.float32)
    else:
        weight_map = np.zeros((out_h, out_w), dtype=np.float32)

    return warped_full, weight_map


def stitch_ptz_guided(
    frames: List[np.ndarray],
    pan_angles: List[float],
    tilt_angles: List[float],
    hfov_deg: float = 85.0,
    dist_coeffs: Optional[np.ndarray] = None,
    out_w: int = 3840,
    out_h: int = 1440,
) -> np.ndarray:
    """
    Ghép đa tầng toàn không gian bằng mô hình chiếu cầu PTZ Ground Truth.
    """
    accum_img = np.zeros((out_h, out_w, 3), dtype=np.float32)
    accum_weight = np.zeros((out_h, out_w), dtype=np.float32)

    total = len(frames)
    print(f"[SphericalProjector] Bắt đầu chiếu cầu {total} frames lên canvas {out_w}x{out_h}...")

    for i in range(total):
        img = frames[i]
        p_deg = pan_angles[i]
        t_deg = tilt_angles[i]

        warped, weight = warp_image_to_equirectangular(
            img=img,
            pan_deg=p_deg,
            tilt_deg=t_deg,
            hfov_deg=hfov_deg,
            dist_coeffs=dist_coeffs,
            out_w=out_w,
            out_h=out_h,
            lat_min_deg=-20.0,
            lat_max_deg=85.0
        )

        w_3d = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
        accum_img += warped.astype(np.float32) * w_3d
        accum_weight += weight

    # Chuẩn hóa trọng số hòa trộn (Weighted Average Blending)
    valid_mask = accum_weight > 1e-4
    final_pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    for c in range(3):
        final_pano[:, :, c] = np.where(
            valid_mask,
            np.clip(accum_img[:, :, c] / np.maximum(accum_weight, 1e-4), 0, 255),
            0
        ).astype(np.uint8)

    return final_pano
