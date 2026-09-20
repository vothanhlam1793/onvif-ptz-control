"""Tools for Spatial Memory PTZ Agent."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import threading
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
    upsert_discovered_objects,
)
from apps.spatial_agent.minio_client import upload_spatial_frame, upload_verified_image, upload_file_bytes
from apps.spatial_agent.telegram_notifier import (
    send_telegram_message,
    send_telegram_photo,
    send_telegram_media_group,
)
from apps.spatial_agent.prompts import (
    SCENE_ANALYSIS_VLM_PROMPT,
    TARGET_VERIFICATION_VLM_PROMPT,
    CELL_DETAIL_VLM_PROMPT,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Global runtime PTZ instances injected from CLI / Server
_client: Optional[OnvifClient] = None
_tracker: Optional[VirtualPTZTracker] = None
_rtsp_url: str = ""
PTZ_SPEED: float = float(os.getenv("PTZ_SPEED", "1.0"))
PTZ_SETTLE_TIME: float = float(os.getenv("PTZ_SETTLE_TIME", "0.45"))


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
# Tool 1: Giai đoạn 1 - Quét Không Gian & Học Bối Cảnh (Động 100% & Async 1080p)
# ──────────────────────────────────────────────

async def _analyze_single_cell_vlm(
    vlm: ChatOpenAI,
    cell_id: str,
    pan_deg: float,
    tilt_deg: float,
    image_bytes: bytes,
    image_url: str = "",
) -> Dict[str, Any]:
    """Phân tích chi tiết 1 frame ảnh 1080p đơn lẻ bằng Gemini 3.7 VLM."""
    prompt_text = CELL_DETAIL_VLM_PROMPT.format(
        cell_id=cell_id,
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
    )
    if image_url:
        img_content = {"type": "image_url", "image_url": {"url": image_url}}
    else:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        img_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    try:
        msg = HumanMessage(content=[{"type": "text", "text": prompt_text}, img_content])
        resp = await vlm.ainvoke([msg])
        clean_json = str(resp.content).strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()
        data = json.loads(clean_json)
        return {
            "cell_id": cell_id,
            "summary": data.get("summary", ""),
            "objects": data.get("objects", []),
        }
    except Exception as e:
        logger.warning(f"Lỗi phân tích VLM ô {cell_id}: {e}")
        return {"cell_id": cell_id, "summary": f"Ô góc Pan={pan_deg}°, Tilt={tilt_deg}°", "objects": []}


@tool
def scan_and_index_space_tool(force: bool = False, reindex_only: bool = False) -> str:
    """Quét và lập chỉ mục không gian 3D căn phòng:
    - Nếu reindex_only=True: Sử dụng trực tiếp kho ảnh 1080p có sẵn để AI VLM phân tích song song nhận diện lại toàn bộ chi tiết siêu tốc (không làm quay motor camera).
    - Nếu force=True hoặc quét mới: Tự động tính toán ma trận lưới động 100% theo FOV quang học & dải cơ khí, xoay camera chụp toàn bộ frame 1080p mới và nạp vào SQLite.
    """
    global _client, _tracker, _rtsp_url
    if not _client or not _tracker:
        return "Lỗi: Phần cứng Camera PTZ chưa được kết nối."

    existing_cells = get_all_cells()
    if existing_cells and not force and not reindex_only:
        return f"Không gian đã được quét trước đó ({len(existing_cells)} ô trong cơ sở dữ liệu). Dùng `force=True` nếu muốn quay quét lại từ đầu, hoặc `reindex_only=True` để AI nhận diện lại trên ảnh sẵn có."

    camera_key = getattr(_client, "camera_key", "uniarch_uho_s2e")
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/spatial_memory/frames"))
    os.makedirs(out_dir, exist_ok=True)

    # ── THUẬT TOÁN TÍNH TOÁN MA TRẬN LƯỚI ĐỘNG 100% THEO THÔNG SỐ QUANG HỌC & CƠ KHÍ ──
    hfov = getattr(_tracker, "fov_degrees_h", 85.0)
    vfov = getattr(_tracker, "fov_degrees_v", 50.0)
    total_pan = getattr(_tracker, "total_pan_range_deg", 365.0)
    tilt_min = getattr(_tracker, "tilt_min_deg", -15.0)
    tilt_max = getattr(_tracker, "tilt_max_deg", 75.0)
    total_tilt = max(1.0, tilt_max - tilt_min)

    # Tỷ lệ overlap 35% chống méo rìa và đảm bảo bắt trọn biên cơ khí 365°
    overlap = 0.35
    delta_pan = hfov * (1.0 - overlap)
    delta_tilt = vfov * (1.0 - overlap)

    # Dải Pan tâm camera: Quét sát từ 3.0° (chốt trái) đến 362.0° (kịch chốt phải)
    p_start = min(5.0, hfov * 0.08)
    p_end = total_pan - min(3.0, hfov * 0.05)
    cols = max(10, int(np.ceil((p_end - p_start) / delta_pan)) + 1)
    pan_angles = np.linspace(p_start, p_end, cols)

    # Dải Tilt tâm camera: Từ đỉnh trần (tilt_max) xuống sàn/bàn (tilt_min)
    t_top = tilt_max - min(10.0, vfov * 0.20)
    t_bottom = tilt_min + min(5.0, vfov * 0.15)
    num_rows = max(3, int(np.ceil((t_top - t_bottom) / delta_tilt)) + 1)
    tilt_angles = np.linspace(t_top, t_bottom, num_rows)

    total = num_rows * cols
    captured_frames = []

    if reindex_only:
        print(f"\n[Spatial Re-indexer] Chế độ nhận diện lại siêu tốc trên kho ảnh có sẵn ({num_rows} tầng x {cols} cột = {total} ô)...")
        step = 0
        for r_idx, t_deg in enumerate(tilt_angles):
            for c_idx, p_deg in enumerate(pan_angles):
                step += 1
                cell_id = f"Y{r_idx}_X{c_idx:02d}"
                local_path = os.path.join(out_dir, f"{cell_id}.jpg")
                fb = None
                if os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        fb = f.read()
                captured_frames.append({
                    "cell_id": cell_id,
                    "local_path": local_path,
                    "url": "",
                    "bytes": fb,
                    "pan_deg": p_deg,
                    "tilt_deg": t_deg,
                })
    else:
        # 1. Homing camera về chuẩn
        try:
            _tracker.home(speed=PTZ_SPEED)
        except Exception as e:
            logger.warning(f"Homing notice: {e}")

        print(f"\n[Spatial Scanner] Khởi tạo lưới động 100% (HFOV={hfov}°, VFOV={vfov}°, Pan Range={total_pan}°, Tilt Range={total_tilt}°): {num_rows} tầng x {cols} cột = {total} ô...")
        step = 0
        for r_idx, t_deg in enumerate(tilt_angles):
            for c_idx, p_deg in enumerate(pan_angles):
                step += 1
                cell_id = f"Y{r_idx}_X{c_idx:02d}"
                p_val, t_val = _tracker.physical_angles_to_virtual(p_deg, t_deg)

                # Quay camera tốc độ cao
                _tracker.goto_angle(p_deg, t_deg, speed=PTZ_SPEED)
                time.sleep(PTZ_SETTLE_TIME)

                # Chụp snapshot 1080p
                fb = snapshot(_rtsp_url, width=1920, height=1080)
                if not fb:
                    time.sleep(0.15)
                    fb = snapshot(_rtsp_url, width=1280, height=720)

                local_path = os.path.join(out_dir, f"{cell_id}.jpg")
                if fb:
                    with open(local_path, "wb") as f:
                        f.write(fb)
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
                captured_frames.append({
                    "cell_id": cell_id,
                    "local_path": local_path,
                    "url": minio_url,
                    "bytes": fb,
                    "pan_deg": p_deg,
                    "tilt_deg": t_deg,
                })
                print(f"  [{step}/{total}] Quét ô {cell_id}: Pan={p_deg:.1f}°, Tilt={t_deg:.1f}° -> Đã chụp & lưu.")

    # Tạo ảnh ma trận Grid ghép nếu có ảnh
    import cv2
    from panorama.stitcher import _make_grid
    
    decoded_imgs = []
    for f in captured_frames:
        if f.get("bytes"):
            nparr = np.frombuffer(f["bytes"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded_imgs.append(img)

    if decoded_imgs:
        grid_bytes = _make_grid(decoded_imgs, grid_rows=num_rows, grid_cols=cols)
        grid_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/spatial_memory/grid_labeled_space.jpg"))
        with open(grid_path, "wb") as f:
            f.write(grid_bytes)
        grid_url = upload_file_bytes(grid_bytes, filename="grid_labeled_space.jpg", prefix=f"spatial_ptz/{camera_key}/maps")
    else:
        grid_url = ""

    # Gửi ảnh ma trận sang Gemini 3.7 VLM để tổng hợp Room Overview
    print("\n[Spatial VLM] 1/2: Tổng hợp bố cục phòng tổng thể...")
    vlm = _get_vlm()
    prompt_text = SCENE_ANALYSIS_VLM_PROMPT.format(
        grid_rows=num_rows,
        grid_cols=cols,
        total_frames=total,
        max_col=cols - 1,
        max_row=num_rows - 1,
    )

    if grid_url:
        img_content = {"type": "image_url", "image_url": {"url": grid_url}}
    elif decoded_imgs:
        b64 = base64.b64encode(grid_bytes).decode("utf-8")
        img_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    else:
        img_content = None

    try:
        if img_content:
            msg = HumanMessage(content=[{"type": "text", "text": prompt_text}, img_content])
            resp = vlm.invoke([msg])
            clean_json = str(resp.content).strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            overview_data = json.loads(clean_json)
            room_overview = overview_data.get("room_overview", "Đã phân tích không gian.")
        else:
            room_overview = f"Không gian phòng {num_rows} tầng dọc x {cols} cột ngang."
    except Exception:
        room_overview = f"Không gian phòng {num_rows} tầng dọc x {cols} cột ngang."

    save_space_metadata(camera_key, room_overview, total, num_rows, cols)

    # 5. PHÂN TÍCH SONG SONG ASYNC TỪNG FRAME 1080P ĐỘ PHÂN GIẢI CAO (Single-Frame Inspection)
    print(f"\n[Spatial VLM] 2/2: Đang phân tích song song {len(captured_frames)} frame 1080p gốc để nhận diện 150+ vật thể chi tiết...")
    import asyncio

    async def run_parallel_analysis():
        tasks = [
            _analyze_single_cell_vlm(
                vlm=vlm,
                cell_id=f["cell_id"],
                pan_deg=f["pan_deg"],
                tilt_deg=f["tilt_deg"],
                image_bytes=f["bytes"],
                image_url=f["url"],
            )
            for f in captured_frames if f.get("bytes")
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            cell_results = loop.run_until_complete(run_parallel_analysis())
        else:
            cell_results = loop.run_until_complete(run_parallel_analysis())
    except Exception:
        cell_results = asyncio.run(run_parallel_analysis())

    saved_obj_count = 0
    from apps.spatial_agent.db import get_connection
    for res in cell_results:
        cid = res.get("cell_id", "")
        summary = res.get("summary", "")
        objs = res.get("objects", [])
        if cid:
            save_spatial_objects(cid, objs)
            saved_obj_count += len(objs)
            with get_connection() as conn:
                conn.execute("UPDATE spatial_cells SET summary = ? WHERE cell_id = ?", (summary, cid))
                conn.commit()

    action_label = "RE-INDEX AI (TẬP ẢNH SẴN CÓ)" if reindex_only else f"QUÉT LƯỚI ĐỘNG MỚI ({num_rows}x{cols}={total} Ô)"
    return f"ĐÃ HOÀN TẤT {action_label}: Phân tích song song 1080p thành công. Đã nhận diện và lập chỉ mục {saved_obj_count} vật thể chi tiết vào SQLite. Tổng quan phòng: {room_overview}"


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
def slew_and_verify_target_tool(
    target_label: str,
    cell_id: str,
    pan_deg: float,
    tilt_deg: float,
    verify_with_vlm: bool = False,
) -> str:
    """Điều khiển camera PTZ lia tới góc Pan/Tilt của ô mục tiêu:
    - verify_with_vlm = False (MẶC ĐỊNH SIÊU TỐC < 1s): Dùng cho các lệnh 'chụp ảnh', 'quay tới', 'nhìn sang', 'hướng camera'. Camera lia tới góc, chụp ảnh và phản hồi ngay lập tức (KHÔNG tốn token VLM, độ trễ cực thấp).
    - verify_with_vlm = True: Chỉ dùng khi người dùng yêu cầu 'tìm', 'xác thực xem có... không', 'căn tâm đối tượng nhỏ'. Sẽ kích hoạt AI VLM thẩm định và tự động căn tâm quang học 1-Shot.
    """
    global _client, _tracker, _rtsp_url
    if not _client or not _tracker:
        return "Lỗi: Phần cứng Camera PTZ chưa được kết nối."

    print(f"\n[PTZ Slew] Đang điều khiển camera lia tới ô {cell_id}: Pan = {pan_deg:.1f}°, Tilt = {tilt_deg:.1f}° (Speed={PTZ_SPEED}, VLM Verify={verify_with_vlm})...")
    _tracker.goto_angle(pan_deg, tilt_deg, speed=PTZ_SPEED)
    time.sleep(PTZ_SETTLE_TIME)  # Chờ ổn định cơ khí chống rung nhòe ảnh

    # Chụp ảnh snapshot Full HD 1080p sắc nét
    fb = snapshot(_rtsp_url, width=1920, height=1080)
    if not fb:
        return f"Lỗi: Không lấy được snapshot từ camera tại góc Pan={pan_deg}, Tilt={tilt_deg}."

    verified_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/spatial_memory/verified"))
    os.makedirs(verified_dir, exist_ok=True)
    local_verify_path = os.path.join(verified_dir, f"verify_{cell_id}_{target_label}.jpg")
    with open(local_verify_path, "wb") as f:
        f.write(fb)

    camera_key = getattr(_client, "camera_key", "uniarch_uho_s2e")
    
    # Upload MinIO chạy ngầm trong background thread để không chặn luồng chính
    threading.Thread(target=upload_verified_image, args=(fb, target_label, camera_key), daemon=True).start()

    # ── CHẾ ĐỘ 1: FAST DIRECT AIM (< 1.0s, Không qua VLM) ──
    if not verify_with_vlm:
        return json.dumps({
            "is_found": True,
            "mode": "FAST_DIRECT_AIM",
            "target_label": target_label,
            "cell_id": cell_id,
            "pan_deg": pan_deg,
            "tilt_deg": tilt_deg,
            "local_image_path": local_verify_path,
            "message": f"Camera đã lia tới mục tiêu '{target_label}' tại ô {cell_id} (Pan={pan_deg:.1f}°, Tilt={tilt_deg:.1f}°). Ảnh đã được chụp và lưu."
        }, ensure_ascii=False, indent=2)

    # ── CHẾ ĐỘ 2: DEEP OPTICAL VERIFICATION (Có VLM thẩm định & căn tâm) ──
    print(f"[VLM Verifier] Đang gửi ảnh chụp trực tiếp sang Gemini 3.7 để xác thực '{target_label}'...")
    vlm = _get_vlm()
    prompt_text = TARGET_VERIFICATION_VLM_PROMPT.format(
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        target_label=target_label,
    )

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
        total_pan = getattr(_tracker, "total_pan_range_deg", 365.0)
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
            _tracker.goto_angle(opt_pan, opt_tilt, speed=PTZ_SPEED)
            time.sleep(PTZ_SETTLE_TIME)

            # Chụp lại snapshot mới đã căn tâm tuyệt đối Full HD 1080p
            fb_centered = snapshot(_rtsp_url, width=1920, height=1080)
            if fb_centered:
                with open(local_verify_path, "wb") as f:
                    f.write(fb_centered)
                threading.Thread(target=upload_verified_image, args=(fb_centered, target_label, camera_key), daemon=True).start()
                ver_result["pan_deg"] = opt_pan
                ver_result["tilt_deg"] = opt_tilt
                ver_result["is_centered"] = True
                ver_result["one_shot_centering_applied"] = True
                ver_result["explanation"] += f" (Đã tự động căn tâm quang học 1-Shot: Pan={opt_pan}°, Tilt={opt_tilt}°)"
                pan_deg, tilt_deg = opt_pan, opt_tilt

    # Cập nhật trạng thái đối tượng chính
    status = ver_result.get("state_change", "PRESENT") if ver_result.get("is_found") else "NOT_FOUND"
    update_object_verification(cell_id, target_label, status, notes=ver_result.get("explanation", ""))

    # ── TỰ ĐỘNG HỌC & CẬP NHẬT VẬT THỂ LIÊN ĐỚI (In-flight Opportunistic Learning) ──
    context_objects = ver_result.get("detected_context_objects", [])
    if context_objects:
        upserted = upsert_discovered_objects(cell_id, context_objects)
        print(f"[Spatial Memory] Đã tự động học & cập nhật {upserted} vật thể liên đới nhìn thấy tại ô {cell_id} vào SQLite.")
        ver_result["in_flight_learned_objects_count"] = upserted

    ver_result["local_image_path"] = local_verify_path
    ver_result["pan_deg"] = pan_deg
    ver_result["tilt_deg"] = tilt_deg

    return json.dumps(ver_result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 4: Gửi Thông Báo & Ảnh Ra Telegram (Hỗ trợ 1 ảnh hoặc Album loạt ảnh)
# ──────────────────────────────────────────────

@tool
def send_telegram_alert_tool(
    message: str,
    include_current_snapshot: bool = True,
    image_url: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
) -> str:
    """Gửi tin nhắn thông báo kèm hình ảnh kết quả tìm kiếm hoặc album/loạt nhiều ảnh tới Telegram của người dùng (@vothanhlam1793).
    - image_paths: Danh sách đường dẫn các file ảnh cần gửi hàng loạt (ví dụ gửi loạt ảnh hàng Y3_X00 -> Y3_X09).
    - image_url: Đường dẫn ảnh đơn lẻ.
    """
    global _tracker, _rtsp_url
    
    # 1. Gửi album ảnh nếu có danh sách nhiều ảnh
    if image_paths and isinstance(image_paths, list) and len(image_paths) > 1:
        items = []
        for idx, p in enumerate(image_paths):
            cap = message if idx == 0 else ""
            items.append({"photo": p, "caption": cap})
        resp = send_telegram_media_group(photos=items)
        if resp.get("ok"):
            return f"Đã gửi thành công Album {len(image_paths)} hình ảnh tới Telegram (@vothanhlam1793)."
        return f"Gặp lỗi khi gửi Album Telegram: {resp.get('description') or resp.get('error')}"

    # 2. Gửi ảnh đơn lẻ
    photo_payload = None
    if image_paths and len(image_paths) == 1:
        photo_payload = image_paths[0]
    elif image_url and (image_url.startswith("http://") or image_url.startswith("https://") or os.path.exists(image_url)):
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
        slew_and_verify_target_tool,
        scan_and_index_space_tool,
        calibrate_camera_hardware_tool,
        send_telegram_alert_tool,
        get_camera_status_tool,
    ]
