"""
Script kiểm thử thuật toán PTZ-Guided Spherical Projection trên tập dữ liệu 1789831690
"""
import json
from pathlib import Path
import cv2
import numpy as np
from panorama.spherical_projector import stitch_ptz_guided

DATA_DIR = Path("outputs/panorama/1789831690")
PROFILE_PATH = Path("outputs/camera_profiles/lc_ipc_k2e_3h3w.json")

with open(PROFILE_PATH, "r") as f:
    profile = json.load(f)

dist_coeffs = np.array(profile["optical"]["dist_coeffs"], dtype=np.float64)
hfov_deg = profile["optical"]["hfov_deg"]

# Tầng sàn (Y=2): frames 00..07, Tilt ~ 4.0 deg
# Tầng giữa (Y=1): frames 08..15, Tilt ~ 35.0 deg
# Tầng trần (Y=0): 14 frames sau dời pha 9 bước [25..29, 16..24], Tilt ~ 70.0 deg

frames = []
pan_angles = []
tilt_angles = []

# Tier 2: Floor (8 frames)
pan_step_8 = np.linspace(0.0, 360.0, 8, endpoint=False)
for idx, pan_deg in enumerate(pan_step_8):
    img_path = DATA_DIR / f"frame_{idx:02d}.jpg"
    img = cv2.imread(str(img_path))
    if img is not None:
        frames.append(img)
        pan_angles.append(float(pan_deg))
        tilt_angles.append(4.0)

# Tier 1: Middle (8 frames)
for idx, pan_deg in enumerate(pan_step_8):
    img_path = DATA_DIR / f"frame_{idx + 8:02d}.jpg"
    img = cv2.imread(str(img_path))
    if img is not None:
        frames.append(img)
        pan_angles.append(float(pan_deg))
        tilt_angles.append(35.0)

# Tier 0: Ceiling (14 frames)
tier0_indices = [25, 26, 27, 28, 29, 16, 17, 18, 19, 20, 21, 22, 23, 24]
pan_step_14 = np.linspace(0.0, 360.0, len(tier0_indices), endpoint=False)
for idx_in_list, frame_num in enumerate(tier0_indices):
    img_path = DATA_DIR / f"frame_{frame_num:02d}.jpg"
    img = cv2.imread(str(img_path))
    if img is not None:
        frames.append(img)
        pan_angles.append(float(pan_step_14[idx_in_list]))
        tilt_angles.append(70.0)

print(f"Loaded {len(frames)} frames for spherical stitching.")

pano = stitch_ptz_guided(
    frames=frames,
    pan_angles=pan_angles,
    tilt_angles=tilt_angles,
    hfov_deg=hfov_deg,
    dist_coeffs=dist_coeffs,
    out_w=3840,
    out_h=1440
)

out_path = DATA_DIR / "panorama_spherical.jpg"
cv2.imwrite(str(out_path), pano)
print(f"Đã xuất ảnh Spherical Panorama tại: {out_path} ({pano.shape[1]}x{pano.shape[0]})")
