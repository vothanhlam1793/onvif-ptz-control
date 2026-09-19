"""
FastAPI routes: PTZ control, stream, snapshot, presets, AI aim endpoint, panorama.
"""

import io
import json
import asyncio
import threading
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
    """Chụp 1 frame JPEG chất lượng cao. Dùng cho AI inspection."""
    from streaming.stream_relay import snapshot as do_snapshot
    data = do_snapshot(rtsp_url_main)
    if not data:
        raise HTTPException(503, "Snapshot failed")
    return Response(content=data, media_type="image/jpeg")


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


@router.post("/ptz/virtual/goto")
def virtual_goto(req: VirtualGotoRequest):
    """Quay tới toạ độ ảo định trước."""
    if not ptz_service or not ptz_service.virtual_tracker:
        raise HTTPException(503, "Virtual tracker not initialized")
    try:
        ptz_service.virtual_tracker.goto_virtual(req.pan, req.tilt, req.speed)
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


@router.get("/panorama/result/grid")
def panorama_result_grid():
    """Trả về ảnh grid (các frame riêng lẻ) sau khi xong."""
    result = _panorama_state.get("result")
    if not result or not result.get("grid"):
        raise HTTPException(404, "Grid not ready")
    return Response(content=result["grid"], media_type="image/jpeg")
