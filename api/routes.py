"""
FastAPI routes: PTZ control, stream, snapshot, presets, AI aim endpoint, panorama.
"""

import io
import os
import time
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

# Injected from main.py
ptz_service = None
stream_relay = None
rtsp_url_main = ""
onvif_client  = None   # raw OnvifClient for panorama

# Patrol state
_patrol_state = {
    "running": False,
    "stop_event": None,
    "presets": [],
    "dwell_sec": 3.0,
}

# Panorama state (one job at a time)
_panorama_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "latest_frame": None,   # bytes JPEG of latest captured frame
    "result": None,         # final result dict
    "error": None,
    "completed": False,     # True chỉ khi vừa hoàn tất run_panorama
}

# Active WebSockets
_active_websockets: List[WebSocket] = []


router = APIRouter()


# ──────────────────────────────────────────────
# WebSocket Telemetry
# ──────────────────────────────────────────────

@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    _active_websockets.append(websocket)
    try:
        while True:
            # Thu thập dữ liệu telemetry tổng hợp
            ptz_stat = {}
            if ptz_service:
                try:
                    ptz_stat = ptz_service.status()
                except Exception:
                    pass

            pano_res = _panorama_state.get("result") or {}
            payload = {
                "ptz": ptz_stat,
                "patrol": {
                    "running": _patrol_state["running"],
                    "presets": _patrol_state["presets"],
                    "dwell_sec": _patrol_state["dwell_sec"],
                },
                "panorama": {
                    "running": _panorama_state["running"],
                    "progress": _panorama_state["progress"],
                    "total": _panorama_state["total"],
                    "error": _panorama_state["error"],
                    "done": _panorama_state.get("completed", False) and not _panorama_state["running"] and _panorama_state["result"] is not None,
                    "calibration_note": pano_res.get("calibration_note", ""),
                    "num_steps": pano_res.get("num_steps", 0),
                    "frame_count": pano_res.get("frame_count", 0),
                }
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if websocket in _active_websockets:
            _active_websockets.remove(websocket)


# ──────────────────────────────────────────────
# Stream & Snapshot
# ──────────────────────────────────────────────

@router.get("/stream")
def mjpeg_stream():
    """MJPEG stream cho thẻ <img> trên Web UI."""
    if stream_relay is None:
        raise HTTPException(503, "Stream not ready")
    return StreamingResponse(
        stream_relay.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/snapshot")
def get_snapshot():
    """Chụp 1 frame JPEG chất lượng cao kèm PTZ Telemetry qua Header."""
    from streaming.stream_relay import snapshot as do_snapshot
    data = do_snapshot(rtsp_url_main)
    if not data:
        raise HTTPException(503, "Snapshot failed")
    
    headers = {}
    if ptz_service and ptz_service.virtual_tracker:
        stat = ptz_service.virtual_tracker.get_status()
        headers["X-PTZ-Pan"] = str(stat.get("pan", 0.0))
        headers["X-PTZ-Tilt"] = str(stat.get("tilt", 0.0))
        headers["X-PTZ-Zoom"] = str(stat.get("zoom", 0.0))
        headers["X-PTZ-Pan-Deg"] = str(stat.get("pan_deg", 0.0))
        headers["X-PTZ-Tilt-Deg"] = str(stat.get("tilt_deg", 0.0))
        headers["X-PTZ-Homed"] = str(stat.get("is_homed", False))

    return Response(content=data, media_type="image/jpeg", headers=headers)


@router.get("/ptz/snapshot_with_telemetry")
def get_snapshot_with_telemetry():
    """
    Chụp 1 frame JPEG kèm metadata toạ độ đầy đủ dạng JSON Base64
    dành cho các hệ thống mở rộng (Panorama, AI Person Detection, Slew-to-Cue).
    """
    import base64
    from streaming.stream_relay import snapshot as do_snapshot
    data = do_snapshot(rtsp_url_main)
    if not data:
        raise HTTPException(503, "Snapshot failed")
    
    telemetry = {}
    if ptz_service and ptz_service.virtual_tracker:
        telemetry = ptz_service.virtual_tracker.get_status()

    return {
        "timestamp": asyncio.get_event_loop().time(),
        "telemetry": telemetry,
        "image_base64": base64.b64encode(data).decode("utf-8"),
        "image_format": "jpeg",
    }


# ──────────────────────────────────────────────
# PTZ D-pad
# ──────────────────────────────────────────────

class MoveRequest(BaseModel):
    direction: str           # left | right | up | down | up-left | ...
    speed: Optional[float] = None


@router.post("/ptz/move")
def ptz_move(req: MoveRequest):
    """Start continuous move."""
    try:
        ptz_service.move(req.direction, req.speed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@router.post("/ptz/stop")
def ptz_stop():
    """Stop all movement."""
    try:
        ptz_service.stop()
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


class SpeedRequest(BaseModel):
    speed: float  # 0.05 - 1.0


@router.post("/ptz/speed")
def ptz_set_speed(req: SpeedRequest):
    ptz_service.set_speed(req.speed)
    return {"speed": ptz_service._speed}


# ──────────────────────────────────────────────
# Click-to-Center
# ──────────────────────────────────────────────

class ClickRequest(BaseModel):
    x: float
    y: float
    frame_width: float  = 1280.0
    frame_height: float = 720.0
    sensitivity: float  = 0.8


@router.post("/ptz/click")
def ptz_click(req: ClickRequest):
    """Nhận toạ độ click pixel -> RelativeMove camera về tâm."""
    try:
        ptz_service.click_to_center(
            req.x, req.y, req.frame_width, req.frame_height, req.sensitivity
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


# ──────────────────────────────────────────────
# AI Aim (Phase 2: LangGraph / Detector gọi vào)
# ──────────────────────────────────────────────

class AimBBoxRequest(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: float  = 1280.0
    frame_height: float = 720.0
    sensitivity: float  = 1.0


@router.post("/ptz/aim/bbox")
def ptz_aim_bbox(req: AimBBoxRequest):
    """
    AI Detector gửi BBox -> camera quay cực nhanh đưa mục tiêu vào tâm hình.
    Đây là endpoint chính cho Phase 2 Slew-to-Cue.
    """
    try:
        ptz_service.aim_at_bbox(
            req.x1, req.y1, req.x2, req.y2,
            req.frame_width, req.frame_height, req.sensitivity
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


class AimPointRequest(BaseModel):
    x: float
    y: float
    frame_width: float  = 1280.0
    frame_height: float = 720.0


@router.post("/ptz/aim/point")
def ptz_aim_point(req: AimPointRequest):
    """AI gửi điểm pixel -> camera aim vào đó."""
    try:
        ptz_service.aim_at_point(req.x, req.y, req.frame_width, req.frame_height)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


# ──────────────────────────────────────────────
# Virtual PTZ & Auto-Calibration Engine
# ──────────────────────────────────────────────

@router.get("/ptz/calibration/profile")
def get_calibration_profile():
    """Lấy profile camera hiện tại (nếu đã hiệu chuẩn)."""
    from core.auto_calibration import load_camera_profile
    key = getattr(onvif_client, "camera_key", "")
    prof = load_camera_profile(key)
    return {
        "camera_key": key,
        "is_calibrated": prof is not None,
        "profile": prof,
    }


@router.post("/ptz/calibration/run")
def run_camera_calibration(force: bool = False):
    """Kích hoạt chạy quy trình hiệu chuẩn 3 bước tự động."""
    from core.auto_calibration import AutoCalibrationEngine
    if not onvif_client:
        raise HTTPException(503, "Camera client not connected")
    try:
        engine = AutoCalibrationEngine(onvif_client, rtsp_url_main)
        prof = engine.get_or_calibrate(force=force)
        if ptz_service and ptz_service.virtual_tracker:
            ptz_service.virtual_tracker.load_calibration()
        return {"ok": True, "profile": prof}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/ptz/virtual/home")
def virtual_home(speed: float = 0.8):
    """Quy trình Homing cho Virtual PTZ Tracker."""
    if not ptz_service or not ptz_service.virtual_tracker:
        raise HTTPException(503, "Virtual tracker not initialized")
    try:
        res = ptz_service.virtual_tracker.home(speed=speed)
        return {"ok": True, "data": res}
    except Exception as e:
        raise HTTPException(500, str(e))


class VirtualGotoRequest(BaseModel):
    pan: float   # [-1.0, 1.0]
    tilt: float  # [-1.0, 1.0]
    speed: float = 0.8


class AngleGotoRequest(BaseModel):
    pan_deg: float   # Ví dụ: 0° -> 366°
    tilt_deg: float  # Ví dụ: -5° -> 80°
    speed: float = 0.8


@router.post("/ptz/virtual/goto")
def virtual_goto(req: VirtualGotoRequest):
    """Quay tới toạ độ ảo định trước [-1.0, 1.0]."""
    if not ptz_service or not ptz_service.virtual_tracker:
        raise HTTPException(503, "Virtual tracker not initialized")
    try:
        ptz_service.virtual_tracker.goto_virtual(req.pan, req.tilt, req.speed)
        return {"ok": True, "status": ptz_service.virtual_tracker.get_status()}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/ptz/virtual/goto_angle")
def virtual_goto_angle(req: AngleGotoRequest):
    """Quay tới góc vật lý thực tế (pan_deg, tilt_deg)."""
    if not ptz_service or not ptz_service.virtual_tracker:
        raise HTTPException(503, "Virtual tracker not initialized")
    try:
        ptz_service.virtual_tracker.goto_angle(req.pan_deg, req.tilt_deg, req.speed)
        return {"ok": True, "status": ptz_service.virtual_tracker.get_status()}
    except Exception as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────
# Presets
# ──────────────────────────────────────────────

@router.get("/ptz/presets")
def list_presets():
    return {"presets": ptz_service.list_presets()}


class SavePresetRequest(BaseModel):
    name: str
    token: Optional[str] = None


@router.post("/ptz/presets/save")
def save_preset(req: SavePresetRequest):
    token = ptz_service.save_preset(req.name, req.token)
    return {"token": token}


class GotoPresetRequest(BaseModel):
    token: str
    speed: float = 1.0


@router.post("/ptz/presets/goto")
def goto_preset(req: GotoPresetRequest):
    ptz_service.goto_preset(req.token, req.speed)
    return {"ok": True}


class RemovePresetRequest(BaseModel):
    token: str


@router.post("/ptz/presets/remove")
def remove_preset(req: RemovePresetRequest):
    ptz_service.remove_preset(req.token)
    return {"ok": True}


# ──────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────

@router.get("/ptz/status")
def get_status():
    try:
        return ptz_service.status()
    except Exception as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────
# Patrol Tour
# ──────────────────────────────────────────────

class PatrolStartRequest(BaseModel):
    preset_tokens: List[str]
    dwell_sec: float = 3.0


@router.post("/ptz/patrol/start")
def start_patrol(req: PatrolStartRequest):
    if _patrol_state["running"]:
        raise HTTPException(409, "Patrol already running")
    if not req.preset_tokens:
        raise HTTPException(400, "preset_tokens cannot be empty")

    stop_ev = ptz_service.patrol_presets(req.preset_tokens, req.dwell_sec)
    _patrol_state["running"] = True
    _patrol_state["stop_event"] = stop_ev
    _patrol_state["presets"] = req.preset_tokens
    _patrol_state["dwell_sec"] = req.dwell_sec
    return {"ok": True, "message": "Patrol tour started"}


@router.post("/ptz/patrol/stop")
def stop_patrol():
    if _patrol_state["stop_event"]:
        _patrol_state["stop_event"].set()
    _patrol_state["running"] = False
    _patrol_state["stop_event"] = None
    return {"ok": True, "message": "Patrol tour stopped"}


@router.get("/ptz/patrol/status")
def get_patrol_status():
    return {
        "running": _patrol_state["running"],
        "presets": _patrol_state["presets"],
        "dwell_sec": _patrol_state["dwell_sec"],
    }


# ──────────────────────────────────────────────
# Panorama
# ──────────────────────────────────────────────

class PanoramaStartRequest(BaseModel):
    mode: Optional[str] = "full_space"  # "horizon" (1 tầng 8 frames) | "full_space" (2 tầng 16 frames)


@router.post("/panorama/start")
def panorama_start(req: Optional[PanoramaStartRequest] = None):
    """Bắt đầu quét panorama. Chạy background thread."""
    if _panorama_state["running"]:
        raise HTTPException(409, "Panorama already running")

    pano_mode = req.mode if req else "full_space"

    def _on_progress(step, total, frame_bytes):
        _panorama_state["progress"] = step
        _panorama_state["total"] = total
        _panorama_state["latest_frame"] = frame_bytes

    def _run():
        from panorama.agent import run_panorama
        _panorama_state.update({
            "running": True, "progress": 0, "total": 0,
            "latest_frame": None, "result": None, "error": None,
            "completed": False,
        })
        try:
            tracker = ptz_service.virtual_tracker if ptz_service else None
            result = run_panorama(
                client=onvif_client,
                rtsp_url=rtsp_url_main,
                tracker=tracker,
                mode=pano_mode,
                on_progress=_on_progress,
            )
            _panorama_state["result"] = result
            _panorama_state["error"] = result.get("error")
            _panorama_state["completed"] = True
        except Exception as e:
            _panorama_state["error"] = str(e)
            _panorama_state["completed"] = False
        finally:
            _panorama_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "message": f"Panorama {pano_mode} sweep started"}


@router.get("/panorama/status")
def panorama_status():
    """Trả về tiến độ hiện tại."""
    s = _panorama_state
    result = s.get("result") or {}
    return {
        "running": s["running"],
        "progress": s["progress"],
        "total": s["total"],
        "error": s["error"],
        "done": s.get("completed", False) and not s["running"] and s["result"] is not None,
        "calibration_note": result.get("calibration_note", ""),
        "num_steps": result.get("num_steps", 0),
        "frame_count": result.get("frame_count", 0),
        "saved_paths": result.get("saved_paths", {}),
    }


@router.get("/panorama/preview")
def panorama_preview():
    """JPEG của frame mới nhất trong lúc đang sweep (realtime preview)."""
    frame = _panorama_state.get("latest_frame")
    if not frame:
        raise HTTPException(404, "No preview available")
    return Response(content=frame, media_type="image/jpeg")


@router.get("/panorama/result/panorama")
def panorama_result_panorama():
    """Trả về ảnh panorama ghép sau khi xong."""
    result = _panorama_state.get("result")
    if not result or not result.get("panorama"):
        raise HTTPException(404, "Panorama not ready")
    return Response(content=result["panorama"], media_type="image/jpeg")


# ──────────────────────────────────────────────
# Settings & Configuration Endpoints
# ──────────────────────────────────────────────

class CameraTestSyncRequest(BaseModel):
    host: str
    port: int = 80
    username: str
    password: str


class LLMTestRequest(BaseModel):
    base_url: str
    api_key: str
    model: str


class SettingsUpdateRequest(BaseModel):
    camera_host: str
    camera_port: int = 80
    camera_user: str
    camera_pass: Optional[str] = None
    stream_width: int = 1280
    stream_height: int = 720
    ninerouter_base_url: str
    ninerouter_api_key: Optional[str] = None
    vlm_model: str


@router.get("/settings")
def get_system_settings():
    """Lấy thông tin cấu hình hiện tại (ẩn password/api_key)."""
    camera_host = os.getenv("CAMERA_HOST", "192.168.110.14")
    camera_port = int(os.getenv("CAMERA_PORT", "80"))
    camera_user = os.getenv("CAMERA_USER", "admin")
    camera_pass = os.getenv("CAMERA_PASS", "")
    stream_width = int(os.getenv("STREAM_WIDTH", "1280"))
    stream_height = int(os.getenv("STREAM_HEIGHT", "720"))
    ninerouter_base_url = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
    ninerouter_api_key = os.getenv("NINEROUTER_API_KEY", "")
    vlm_model = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")

    # Mask key & pass
    masked_key = (ninerouter_api_key[:6] + "..." + ninerouter_api_key[-4:]) if len(ninerouter_api_key) > 10 else "********"
    masked_pass = "********" if camera_pass else ""

    device_info = {}
    if onvif_client:
        device_info = {
            "manufacturer": onvif_client.manufacturer or "LC",
            "model": onvif_client.model or "IPC-K2E-3H3W",
            "firmware_version": onvif_client.firmware_version,
            "serial_number": onvif_client.serial_number,
            "camera_key": onvif_client.camera_key,
        }

    return {
        "camera_host": camera_host,
        "camera_port": camera_port,
        "camera_user": camera_user,
        "camera_pass_masked": masked_pass,
        "stream_width": stream_width,
        "stream_height": stream_height,
        "ninerouter_base_url": ninerouter_base_url,
        "ninerouter_api_key_masked": masked_key,
        "vlm_model": vlm_model,
        "device_info": device_info,
    }


@router.post("/camera/test_sync")
def camera_test_sync(req: CameraTestSyncRequest):
    """Bắt tay ONVIF thử nghiệm, đọc thông tin phần cứng & kiểm tra luồng RTSP."""
    from core.onvif_client import OnvifClient
    t0 = time.time()
    try:
        client = OnvifClient(req.host, req.port, req.username, req.password)
        info = client.discover()
        rtsp_uri = client.get_stream_uri()
        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "ok": True,
            "latency_ms": elapsed,
            "profile_token": info.get("profile_token"),
            "ptz_node_token": info.get("ptz_node_token"),
            "manufacturer": client.manufacturer,
            "model": client.model,
            "firmware": client.firmware_version,
            "serial_number": client.serial_number,
            "mac_address": client.mac_address,
            "rtsp_url": rtsp_uri,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }


@router.post("/settings/test_llm")
def settings_test_llm(req: LLMTestRequest):
    """Gửi prompt kiểm tra tới LLM / 9Router."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    t0 = time.time()
    try:
        api_key = req.api_key
        # Nếu truyền masked key thì fallback lấy từ .env
        if api_key.startswith("sk-") and "..." in api_key:
            api_key = os.getenv("NINEROUTER_API_KEY", "")

        llm = ChatOpenAI(
            base_url=req.base_url,
            api_key=api_key,
            model=req.model,
            temperature=0.0,
            max_tokens=30,
            timeout=10.0,
        )
        resp = llm.invoke([HumanMessage(content="Reply with exactly 'OK_CONNECTED'")])
        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "ok": True,
            "latency_ms": elapsed,
            "reply": resp.content.strip(),
            "model": req.model,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }


def hot_reload_camera(host: str, port: int, user: str, password: str, width: int, height: int) -> dict:
    """Tái kết nối sang camera ONVIF mới và khởi động lại Stream Relay."""
    global onvif_client, stream_relay, rtsp_url_main, ptz_service
    from core.onvif_client import OnvifClient
    from core.ptz_service import PTZService
    from streaming.stream_relay import StreamRelay

    # 1. Dừng stream relay cũ
    if stream_relay is not None:
        try:
            stream_relay.stop()
        except Exception as e:
            print(f"[hot_reload] Lỗi dừng relay cũ: {e}")

    # 2. Khởi tạo OnvifClient mới
    new_client = OnvifClient(host, port, user, password)
    info = new_client.discover()
    new_rtsp = new_client.get_stream_uri()

    # 3. Khởi tạo StreamRelay mới
    new_relay = StreamRelay(new_rtsp, width, height)
    new_relay.start()

    # 4. Gán biến toàn cục
    onvif_client = new_client
    stream_relay = new_relay
    rtsp_url_main = new_rtsp
    ptz_service = PTZService(new_client, new_rtsp)

    return {
        "camera_key": new_client.camera_key,
        "manufacturer": new_client.manufacturer,
        "model": new_client.model,
        "rtsp_url": new_rtsp,
        "profile_token": info.get("profile_token"),
    }


@router.post("/settings/save")
def settings_save(req: SettingsUpdateRequest):
    """Lưu cấu hình mới vào .env và Hot-Swap Camera kết nối ngay lập tức."""
    env_path = Path(__file__).parent.parent / ".env"
    
    # Giữ nguyên pass/key nếu người dùng không đổi
    cam_pass = req.camera_pass if req.camera_pass and req.camera_pass != "********" else os.getenv("CAMERA_PASS", "")
    llm_key = req.ninerouter_api_key if req.ninerouter_api_key and "..." not in req.ninerouter_api_key and req.ninerouter_api_key != "********" else os.getenv("NINEROUTER_API_KEY", "")

    # Kiểm tra xem có thay đổi camera không
    old_host = os.getenv("CAMERA_HOST", "")
    old_port = int(os.getenv("CAMERA_PORT", "80"))
    old_user = os.getenv("CAMERA_USER", "")
    old_pass = os.getenv("CAMERA_PASS", "")
    old_w = int(os.getenv("STREAM_WIDTH", "1280"))
    old_h = int(os.getenv("STREAM_HEIGHT", "720"))

    cam_changed = (
        req.camera_host != old_host or
        req.camera_port != old_port or
        req.camera_user != old_user or
        (cam_pass and cam_pass != old_pass) or
        req.stream_width != old_w or
        req.stream_height != old_h
    )

    # Cập nhật os.environ
    os.environ["CAMERA_HOST"] = req.camera_host
    os.environ["CAMERA_PORT"] = str(req.camera_port)
    os.environ["CAMERA_USER"] = req.camera_user
    os.environ["CAMERA_PASS"] = cam_pass
    os.environ["STREAM_WIDTH"] = str(req.stream_width)
    os.environ["STREAM_HEIGHT"] = str(req.stream_height)
    os.environ["NINEROUTER_BASE_URL"] = req.ninerouter_base_url
    os.environ["NINEROUTER_API_KEY"] = llm_key
    os.environ["VLM_MODEL"] = req.vlm_model

    # Ghi file .env
    env_content = f"""# ONVIF Camera Config
CAMERA_HOST={req.camera_host}
CAMERA_PORT={req.camera_port}
CAMERA_USER={req.camera_user}
CAMERA_PASS={cam_pass}

# Stream Config
STREAM_WIDTH={req.stream_width}
STREAM_HEIGHT={req.stream_height}

# VLM / 9Router Config
NINEROUTER_BASE_URL={req.ninerouter_base_url}
NINEROUTER_API_KEY={llm_key}
VLM_MODEL={req.vlm_model}
"""
    try:
        env_path.write_text(env_content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"Lỗi ghi .env: {e}")

    hot_swap_info = None
    if cam_changed:
        try:
            print(f"[settings] Đang Hot-Swap kết nối sang camera mới: {req.camera_host}:{req.camera_port}...")
            hot_swap_info = hot_reload_camera(
                host=req.camera_host,
                port=req.camera_port,
                user=req.camera_user,
                password=cam_pass,
                width=req.stream_width,
                height=req.stream_height,
            )
            print(f"[settings] Hot-Swap thành công: {hot_swap_info}")
        except Exception as e:
            print(f"[settings] Hot-Swap thất bại: {e}")
            return {
                "ok": True,
                "message": f"Đã lưu .env nhưng kết nối trực tiếp camera lỗi: {e}",
                "hot_swap_success": False,
                "error": str(e)
            }

    return {
        "ok": True,
        "message": "Đã lưu cài đặt và chuyển đổi camera thành công!",
        "hot_swap_success": True,
        "camera_info": hot_swap_info
    }
