"""
Kiểm thử trực tiếp thuật toán Stitcher mới trên tập frame 1789825866 và tập nấc dọc/ngang.
"""
from pathlib import Path
import cv2
from panorama.stitcher import stitch

sample_dir = Path("outputs/panorama/1789825866")
frame_files = sorted(list(sample_dir.glob("frame_*.jpg")))
print(f"Kiểm thử ghép tập ảnh: {sample_dir.name} ({len(frame_files)} frames)")

frames_bytes = [f.read_bytes() for f in frame_files]

pano_bytes, grid_bytes = stitch(frames_bytes)

if pano_bytes:
    out_p = sample_dir / "panorama_fixed.jpg"
    out_p.write_bytes(pano_bytes)
    print(f"Ghép thành công! File lưu tại: {out_p} ({len(pano_bytes)} bytes)")
else:
    print("Ghép thất bại!")
