"""
Lưu profile ban đầu cho camera LC_IPC-K2E-3H3W từ kết quả đo đạc thực nghiệm.
"""
from core.auto_calibration import save_camera_profile

profile_data = {
    "camera_key": "lc_ipc_k2e_3h3w",
    "device_info": {
        "manufacturer": "LC",
        "model": "IPC-K2E-3H3W",
        "serial_number": "83DBEBEPSFCA093",
        "mac_address": "a83162edd9de",
        "firmware_version": "3.11.0000000.3.R 2026-07-09",
        "hardware_id": "1.30"
    },
    "optical": {
        "hfov_deg": 85.0,
        "vfov_deg": 46.0
    },
    "tilt": {
        "tilt_min_deg": -5.0,
        "tilt_max_deg": 80.0,
        "total_tilt_range_deg": 85.0,
        "full_tilt_time_sec": 2.4,
        "optimal_horizon_tilt_val": -0.80,
        "optimal_horizon_tilt_deg": 4.0,
        "optimal_vertical_frames": 4
    },
    "pan": {
        "total_pan_range_deg": 366.0,
        "full_pan_time_sec": 5.2,
        "is_360_continuous": True,
        "optimal_pan_steps": 6,
        "pan_step_coords": [-0.90, -0.54, -0.18, 0.18, 0.54, 0.90]
    },
    "formulas": {
        "formula_horizon_tilt": "tilt_deg = tilt_min_deg + ((tilt_val - (-1.0)) / 2.0) * (tilt_max_deg - tilt_min_deg)",
        "formula_pan_steps": "N_pan = ceil(total_pan_range_deg / (HFOV * (1.0 - overlap_ratio)))"
    }
}

p = save_camera_profile("lc_ipc_k2e_3h3w", profile_data)
print(f"Đã lưu profile chuẩn cho lc_ipc_k2e_3h3w tại: {p}")
