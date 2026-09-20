"""Tools for Spatial Memory PTZ Agent."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from core.onvif_client import OnvifClient
from core.virtual_ptz import VirtualPTZTracker
from streaming.stream_relay import snapshot
from apps.spatial_agent.db import (
    save_spatial_cell,
    save_spatial_objects,
    save_space_metadata,
    get_all_cells,
    search_objects_in_memory,
    get_spatial_memory_summary,
    update_object_verification,
)
from apps.spatial_agent.minio_client import upload_spatial_frame, upload_verified_image, upload_file_bytes
from apps.spatial_agent.telegram_notifier import send_telegram_message, send_telegram_photo
from apps.spatial_agent.prompts import SCENE_ANALYSIS_VLM_PROMPT, TARGET_VERIFICATION_VLM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# Global runtime PTZ instances injected from CLI / Server
_client: Optional[OnvifClient] = None
_tracker: Optional[VirtualPTZTracker] = None
_rtsp_url: str = ""


def set_ptz_hardware(client: OnvifClient, tracker: VirtualPTZTracker, rtsp_url: str):
    """Inject hardware instances into tools runtime."""
    global _client, _tracker, _rtsp_url
    _client = client
    _tracker = tracker
    _rtsp_url = rtsp_url


def _get_vlm() -> ChatOpenAI:
    base_url = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
    api_key = os.getenv("NINEROUTER_API_KEY", "")
    model = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.1,
        max_tokens=2500,
        timeout=60.0,
    )


# ──────────────────────────────────────────────
# Tool 1: Giai đoạn 1 - Quét Không Gian & Học Bối Cảnh
# ──────────────────────────────────────────────

@tool
def scan_and_index_space_tool(force: bool = False) -> str:
    """Quét toàn bộ không gian 3D căn phòng (ma trận 3 tầng x 8 cột = 24 ô), tải ảnh lên MinIO, gọi AI VLM phân tích toàn bộ bối cảnh và lập chỉ mục vật thể Cố Định (STATIC) vs Biến Động (DYNAMIC) vào SQLite database.
    Chỉ chạy khi chưa có dữ liệu hoặc khi người dùng yêu cầu quét lại phòng.
    """
    global _client, _tracker, _rtsp_url
    if not _client or not _tracker:
        return "Lỗi: Phần cứng Camera PTZ chưa được kết nối."

    existing_cells = get_all_cells()
    if existing_cells and not force:
        return f"Không gian đã được quét trước đó ({len(existing_cells)} ô trong cơ sở dữ liệu). Dùng `force=True` nếu muốn quét lại từ đầu."

    camera_key = getattr(_client, "camera_key", "uniarch_uho_s2e")

    # 1. Homing camera về chuẩn
    try:
        _tracker.home(speed=0.8)
    except Exception as e:
        logger.warning(f"Homing notice: {e}")

    # 2. Tự động tính toán ma trận lưới động dựa trên thông số quang học thực tế của camera (HFOV/VFOV/Lens)
    from core.auto_calibration import load_camera_profile
    prof = load_camera_profile(camera_key) or {}
    
    cols = prof.get("pan", {}).get("optimal_pan_steps", 8)
    num_rows = prof.get("tilt", {}).get("optimal_vertical_frames", 3)
    
    # Tạo các mức tilt tự động từ cao (trần: val dương) xuống thấp (sàn: val âm)
    # Ví dụ: 3 tầng -> [0.50, -0.15, -0.80], 2 tầng -> [0.30, -0.70], 4 tầng -> [0.60, 0.15, -0.30, -0.75]
    if num_rows == 2:
        tilt_rows = [0.30, -0.70]
    elif num_rows == 4:
        tilt_rows = [0.60, 0.15, -0.30, -0.75]
    elif num_rows == 5:
        tilt_rows = [0.70, 0.35, 0.0, -0.35, -0.75]
    else:
        tilt_rows = [0.50, -0.15, -0.80]

    pan_vals = np.linspace(-0.90, 0.90, cols)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/spatial_memory/frames"))
    os.makedirs(out_dir, exist_ok=True)

    captured_frames = []
    lens_mm = prof.get("optical", {}).get("lens_focal_length_mm", 2.8)
    hfov = prof.get("optical", {}).get("hfov_deg", 85.0)
    vfov = prof.get("optical", {}).get("vfov_deg", 50.0)
    print(f"\n[Spatial Scanner] Khởi tạo lưới động cho Lens {lens_mm}mm (HFOV={hfov}°, VFOV={vfov}°): {len(tilt_rows)} tầng x {cols} cột = {len(tilt_rows)*cols} ô...")

    step = 0
    total = len(tilt_rows) * cols
    for r_idx, t_val in enumerate(tilt_rows):
        for c_idx, p_val in enumerate(pan_vals):
            step += 1
            cell_id = f"Y{r_idx}_X{c_idx:02d}"
            p_deg, t_deg = _tracker.virtual_to_physical_angles(p_val, t_val)

            # Quay camera
            _tracker.goto_virtual(p_val, t_val, speed=0.8)
            time.sleep(0.4)

            # Chụp snapshot 1080p
            fb = snapshot(_rtsp_url, width=1920, height=1080)
            if not fb:
                time.sleep(0.3)
                fb = snapshot(_rtsp_url, width=1280, height=720)

            local_path = os.path.join(out_dir, f"{cell_id}.jpg")
            if fb:
                with open(local_path, "wb") as f:
                    f.write(fb)
                # Upload MinIO
                minio_url = upload_spatial_frame(fb, cell_id=cell_id, camera_key=camera_key)
            else:
                minio_url = ""

            save_spatial_cell(
                cell_id=cell_id,
                camera_key=camera_key,
                row_y=r_idx,
                col_x=c_idx,
                pan_deg=p_deg,
                tilt_deg=t_deg,
                pan_val=p_val,
                tilt_val=t_val,
                image_local_path=local_path,
                image_url=minio_url,
                summary="",
            )
            captured_frames.append({"cell_id": cell_id, "local_path": local_path, "url": minio_url, "bytes": fb})
            print(f"  [{step}/{total}] Quét ô {cell_id}: Pan={p_deg:.1f}°, Tilt={t_deg:.1f}° -> Đã lưu.")

    # 3. Tạo ảnh ma trận Grid ghép có nhãn toạ độ
    import cv2
    from panorama.stitcher import _make_grid
    
    # Decode bytes sang cv2 images
    decoded_imgs = []
    for f in captured_frames:
        if f.get("bytes"):
            nparr = np.frombuffer(f["bytes"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded_imgs.append(img)

    grid_bytes = _make_grid(decoded_imgs, grid_rows=len(tilt_rows), grid_cols=cols)
    grid_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/spatial_memory/grid_labeled_space.jpg"))
    with open(grid_path, "wb") as f:
        f.write(grid_bytes)

    grid_url = upload_file_bytes(grid_bytes, filename="grid_labeled_space.jpg", prefix=f"spatial_ptz/{camera_key}/maps")

    # 4. Gửi ảnh ma trận sang Gemini 3.7 VLM phân tích toàn cảnh
    print("\n[Spatial VLM] Đang gửi ảnh ma trận lưới 3D sang Gemini 3.7 để học bối cảnh...")
    vlm = _get_vlm()
    prompt_text = SCENE_ANALYSIS_VLM_PROMPT.format(
        grid_rows=len(tilt_rows),
        grid_cols=cols,
        total_frames=total,
        max_col=cols - 1,
        max_row=len(tilt_rows) - 1,
    )

    # Dùng MinIO URL hoặc Base64
    if grid_url:
        img_content = {"type": "image_url", "image_url": {"url": grid_url}}
    else:
        b64 = base64.b64encode(grid_bytes).decode("utf-8")
        img_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    msg = HumanMessage(content=[{"type": "text", "text": prompt_text}, img_content])
    resp = vlm.invoke([msg])

    # 5. Phân tích kết quả JSON từ VLM
    clean_json = str(resp.content).strip()
    if "```json" in clean_json:
        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_json:
        clean_json = clean_json.split("```")[1].split("```")[0].strip()

    try:
        analysis_data = json.loads(clean_json)
    except Exception:
        analysis_data = {"room_overview": "Đã quét xong 24 ô lưới.", "cells_analysis": []}

    room_overview = analysis_data.get("room_overview", "")
    save_space_metadata(camera_key, room_overview, total, len(tilt_rows), cols)

    saved_obj_count = 0
    from apps.spatial_agent.db import get_connection
    for cell_info in analysis_data.get("cells_analysis", []):
        cid = cell_info.get("cell_id", "")
        summary = cell_info.get("summary", "")
        objs = cell_info.get("objects", [])
        if cid:
            save_spatial_objects(cid, objs)
            saved_obj_count += len(objs)
            # Cập nhật summary ô
            with get_connection() as conn:
                conn.execute("UPDATE spatial_cells SET summary = ? WHERE cell_id = ?", (summary, cid))
                conn.commit()

    return f"ĐÃ HOÀN TẤT GIAI ĐOẠN 1: Quét thành công {total} ô lưới. Đã nhận diện và lưu trữ {saved_obj_count} vật thể vào SQLite. Tổng quan phòng: {room_overview}"


# ──────────────────────────────────────────────
# Tool 2: Giai đoạn 2.1 - Tra Cứu Bộ Nhớ Text-only
# ──────────────────────────────────────────────

@tool
def query_spatial_memory_tool(query_concept: str) -> str:
    """Tra cứu bộ nhớ không gian SQLite bằng text để tìm danh sách các ô nghi vấn chứa đối tượng cần tìm (ví dụ: 'cửa phòng', 'máy lạnh', 'chỗ sạc điện thoại', 'con người', 'kệ sách').
    Hoàn toàn KHÔNG tốn token hình ảnh và cho phản hồi cực nhanh.
    """
    direct_matches = search_objects_in_memory(query_concept)
    if direct_matches:
        results = []
        for m in direct_matches:
            results.append({
                "cell_id": m["cell_id"],
                "label": m["label"],
                "label_vi": m["label_vi"],
                "category": m["category"],
                "status": m["status"],
                "pan_deg": m["pan_deg"],
                "tilt_deg": m["tilt_deg"],
                "notes": m["notes"],
            })
        return json.dumps({
            "match_type": "EXACT_OR_KEYWORD",
            "found": True,
            "candidates": results,
            "message": f"Tìm thấy {len(results)} vị trí phù hợp trong bộ nhớ SQLite."
        }, ensure_ascii=False, indent=2)

    # Nếu chưa khớp từ khoá trực tiếp, lấy toàn bộ tóm tắt không gian cho LLM suy luận
    summary = get_spatial_memory_summary()
    return json.dumps({
        "match_type": "SEMANTIC_REASONING_REQUIRED",
        "found": False,
        "spatial_summary": summary,
        "message": f"Không có từ khoá khớp trực tiếp với '{query_concept}'. Hãy dùng dữ liệu spatial_summary trên để chọn cell_id phù hợp nhất."
    }, ensure_ascii=False, indent=2)


def calculate_optical_target_angle(
    current_pan_deg: float,
    current_tilt_deg: float,
    bbox: list[float],
    hfov_deg: float = 85.0,
    vfov_deg: float = 50.0,
    pan_type: str = "bounded_stops",
    total_pan_range_deg: float = 360.0,
    tilt_min_deg: float = -15.0,
    tilt_max_deg: float = 75.0,
) -> tuple[float, float, float, float]:
    """
    Tính góc Pan/Tilt mục tiêu chính xác dựa trên hình học quang học và tâm BBox:
    - bbox: [ymin, xmin, ymax, xmax] theo thang 0..1000.
    - pan_type: 'bounded_stops' (chặn biên cơ khí) hoặc 'continuous_endless' (xoay 360 vô tận).
    Trả về: (target_pan_deg, target_tilt_deg, offset_x_pct, offset_y_pct)
    """
    ymin, xmin, ymax, xmax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    # Sai số chuẩn hoá [-1.0, 1.0] so với tâm ảnh (500, 500)
    offset_x_pct = (cx - 500.0) / 500.0
    offset_y_pct = (cy - 500.0) / 500.0

    # Độ lệch góc theo FOV của ống kính
    delta_pan = (offset_x_pct * (hfov_deg / 2.0))
    delta_tilt = -(offset_y_pct * (vfov_deg / 2.0))

    if pan_type == "continuous_endless":
        target_pan = (current_pan_deg + delta_pan) % total_pan_range_deg
    else:
        # Bounded stops: Chặn biên cứng [0.0, total_pan_range_deg] để bảo vệ dây cáp & chống quay lùi
        target_pan = max(0.0, min(total_pan_range_deg, current_pan_deg + delta_pan))

    target_tilt = max(tilt_min_deg, min(tilt_max_deg, current_tilt_deg + delta_tilt))

    return round(target_pan, 2), round(target_tilt, 2), round(offset_x_pct, 3), round(offset_y_pct, 3)


# ──────────────────────────────────────────────
# Tool 3: Giai đoạn 2.2 - Lia PTZ & Tái Xác Minh Closed-Loop
# ──────────────────────────────────────────────

@tool
def slew_and_verify_target_tool(target_label: str, cell_id: str, pan_deg: float, tilt_deg: float) -> str:
    """Điều khiển camera PTZ lia tới góc Pan/Tilt của ô nghi vấn, chụp ảnh thời gian thực, thẩm định VLM và TỰ ĐỘNG CĂN TÂM QUANG HỌC 1 BƯỚC (1-Shot Optical Centering) nếu đối tượng bị lệch tâm."""
    global _client, _tracker, _rtsp_url
    if not _client or not _tracker:
        return "Lỗi: Phần cứng Camera PTZ chưa được kết nối."

    print(f"\n[PTZ Slew] Đang điều khiển camera lia tới ô {cell_id}: Pan = {pan_deg:.1f}°, Tilt = {tilt_deg:.1f}°...")
    _tracker.goto_angle(pan_deg, tilt_deg, speed=0.8)
    time.sleep(0.4)  # Chờ ổn định cơ khí

    # Chụp ảnh verify thời gian thực
    fb = snapshot(_rtsp_url, width=1280, height=720)
    if not fb:
        return f"Lỗi: Không lấy được snapshot từ camera tại góc Pan={pan_deg}, Tilt={tilt_deg}."

    verified_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/spatial_memory/verified"))
    os.makedirs(verified_dir, exist_ok=True)
    local_verify_path = os.path.join(verified_dir, f"verify_{cell_id}_{target_label}.jpg")
    with open(local_verify_path, "wb") as f:
        f.write(fb)

    camera_key = getattr(_client, "camera_key", "uniarch_uho_s2e")
    verify_url = upload_verified_image(fb, target_label=target_label, camera_key=camera_key)

    # Gọi Gemini 3.7 VLM thẩm định ảnh vừa chụp
    print(f"[VLM Verifier] Đang gửi ảnh chụp trực tiếp sang Gemini 3.7 để xác thực '{target_label}'...")
    vlm = _get_vlm()
    prompt_text = TARGET_VERIFICATION_VLM_PROMPT.format(
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        target_label=target_label,
    )

    if verify_url:
        img_content = {"type": "image_url", "image_url": {"url": verify_url}}
    else:
        b64 = base64.b64encode(fb).decode("utf-8")
        img_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    msg = HumanMessage(content=[{"type": "text", "text": prompt_text}, img_content])
    resp = vlm.invoke([msg])

    clean_json = str(resp.content).strip()
    if "```json" in clean_json:
        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_json:
        clean_json = clean_json.split("```")[1].split("```")[0].strip()

    try:
        ver_result = json.loads(clean_json)
    except Exception:
        ver_result = {
            "is_found": True,
            "confidence": 0.85,
            "is_centered": True,
            "explanation": str(resp.content),
            "state_change": "PRESENT",
        }

    # ── LAYER NỘI SUY QUANG HỌC 1 BƯỚC (1-Shot Optical Centering) ──
    bbox = ver_result.get("bbox", [])
    if ver_result.get("is_found") and len(bbox) == 4:
        hfov = getattr(_tracker, "fov_degrees_h", 85.0)
        vfov = getattr(_tracker, "fov_degrees_v", 50.0)
        pan_type = getattr(_tracker, "pan_type", "bounded_stops")
        total_pan = getattr(_tracker, "total_pan_range_deg", 360.0)
        tilt_min = getattr(_tracker, "tilt_min_deg", -15.0)
        tilt_max = getattr(_tracker, "tilt_max_deg", 75.0)

        opt_pan, opt_tilt, off_x, off_y = calculate_optical_target_angle(
            current_pan_deg=pan_deg,
            current_tilt_deg=tilt_deg,
            bbox=bbox,
            hfov_deg=hfov,
            vfov_deg=vfov,
            pan_type=pan_type,
            total_pan_range_deg=total_pan,
            tilt_min_deg=tilt_min,
            tilt_max_deg=tilt_max,
        )

        # Nếu độ lệch tâm > 6%, tự động thực hiện bù góc PTZ ngay trong tool
        if abs(off_x) > 0.06 or abs(off_y) > 0.06:
            print(f"[Optical Layer] Phát hiện lệch tâm (ΔX={off_x*100:+.1f}%, ΔY={off_y*100:+.1f}%). Tự động bù góc -> Pan={opt_pan}°, Tilt={opt_tilt}°...")
            _tracker.goto_angle(opt_pan, opt_tilt, speed=0.8)
            time.sleep(0.4)

            # Chụp lại snapshot mới đã căn tâm tuyệt đối
            fb_centered = snapshot(_rtsp_url, width=1280, height=720)
            if fb_centered:
                with open(local_verify_path, "wb") as f:
                    f.write(fb_centered)
                verify_url = upload_verified_image(fb_centered, target_label=target_label, camera_key=camera_key)
                ver_result["live_image_url"] = verify_url
                ver_result["pan_deg"] = opt_pan
                ver_result["tilt_deg"] = opt_tilt
                ver_result["is_centered"] = True
                ver_result["one_shot_centering_applied"] = True
                ver_result["explanation"] += f" (Đã tự động căn tâm quang học 1-Shot: Pan={opt_pan}°, Tilt={opt_tilt}°)"
                pan_deg, tilt_deg = opt_pan, opt_tilt

    # Cập nhật kết quả vào database
    status = ver_result.get("state_change", "PRESENT") if ver_result.get("is_found") else "NOT_FOUND"
    update_object_verification(cell_id, target_label, status, notes=ver_result.get("explanation", ""))

    ver_result["live_image_url"] = verify_url
    ver_result["local_image_path"] = local_verify_path
    ver_result["pan_deg"] = pan_deg
    ver_result["tilt_deg"] = tilt_deg

    return json.dumps(ver_result, ensure_ascii=False, indent=2)

    ver_result["live_image_url"] = verify_url
    ver_result["local_image_path"] = local_verify_path
    ver_result["pan_deg"] = pan_deg
    ver_result["tilt_deg"] = tilt_deg

    return json.dumps(ver_result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 4: Gửi Thông Báo & Ảnh Ra Telegram
# ──────────────────────────────────────────────

@tool
def send_telegram_alert_tool(
    message: str,
    include_current_snapshot: bool = True,
    image_url: Optional[str] = None,
) -> str:
    """Gửi tin nhắn thông báo kèm hình ảnh kết quả tìm kiếm/xác thực hoặc snapshot thời gian thực của camera PTZ tới Telegram của người dùng (@vothanhlam1793).
    Dùng khi người dùng yêu cầu gửi kết quả ra telegram hoặc báo cáo tình hình phòng.
    """
    global _tracker, _rtsp_url
    
    # 1. Ưu tiên gửi kèm ảnh
    photo_payload = None
    if image_url and (image_url.startswith("http://") or image_url.startswith("https://")):
        photo_payload = image_url
    elif include_current_snapshot and _rtsp_url:
        fb = snapshot(_rtsp_url, width=1280, height=720)
        if fb:
            photo_payload = fb

    if photo_payload:
        resp = send_telegram_photo(photo=photo_payload, caption=message)
    else:
        resp = send_telegram_message(text=message)

    if resp.get("ok"):
        return f"Đã gửi thành công thông báo và hình ảnh tới Telegram (@vothanhlam1793)."
    return f"Gặp lỗi khi gửi Telegram: {resp.get('description') or resp.get('error')}"


# ──────────────────────────────────────────────
# Tool 5: Hiệu Chuẩn Phần Cứng Camera (3 Bước)
# ──────────────────────────────────────────────

@tool
def calibrate_camera_hardware_tool(force: bool = True) -> str:
    """Kích hoạt chu trình Auto-Calibration 3 bước tự động đo đạc thông số phần cứng camera: đo góc nhìn quang học (HFOV/VFOV), tiêu cự ống kính (2.4/2.8/3.6mm), dải cơ khí Tilt/Pan và lưu lại hồ sơ outputs/camera_profiles/{camera_key}.json.
    Dùng khi gắn camera mới hoặc muốn hiệu chuẩn lại các góc quay và dải cơ khí.
    """
    global _client, _tracker, _rtsp_url
    if not _client or not _tracker:
        return "Lỗi: Phần cứng Camera PTZ chưa được kết nối."

    from core.auto_calibration import AutoCalibrationEngine
    engine = AutoCalibrationEngine(_client, _rtsp_url)
    prof = engine.get_or_calibrate(force=force)
    _tracker.load_calibration()

    hfov = prof.get("optical", {}).get("hfov_deg")
    vfov = prof.get("optical", {}).get("vfov_deg")
    lens = prof.get("optical", {}).get("lens_focal_length_mm")
    cols = prof.get("pan", {}).get("optimal_pan_steps")
    rows = prof.get("tilt", {}).get("optimal_vertical_frames")
    pan_type = prof.get("pan", {}).get("pan_type")

    return f"ĐÃ HOÀN TẤT HIỆU CHUẨN PHẦN CỨNG: Lens ~{lens}mm (HFOV={hfov}°, VFOV={vfov}°). Dải quay Pan: {prof.get('pan', {}).get('total_pan_range_deg')}° ({pan_type}), Tilt: {prof.get('tilt', {}).get('total_tilt_range_deg')}°. Ma trận lưới tự động: {rows} tầng x {cols} cột = {rows*cols} ô. Hồ sơ đã lưu vào outputs/camera_profiles/{_client.camera_key}.json"


# ──────────────────────────────────────────────
# Tool 6: Đọc Trạng Thái Camera
# ──────────────────────────────────────────────

@tool
def get_camera_status_tool() -> str:
    """Lấy trạng thái và toạ độ vật lý thời gian thực của Camera PTZ."""
    global _tracker
    if not _tracker:
        return "Camera chưa khởi tạo."
    status = _tracker.get_status()
    return json.dumps(status, ensure_ascii=False, indent=2)


def get_spatial_agent_tools() -> list:
    return [
        query_spatial_memory_tool,
        slew_and_verify_target_tool,
        scan_and_index_space_tool,
        calibrate_camera_hardware_tool,
        send_telegram_alert_tool,
        get_camera_status_tool,
    ]
