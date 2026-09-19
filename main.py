"""
Entrypoint - ONVIF PTZ Controller.
Chạy: python main.py
Mở: http://localhost:8080
"""

import sys
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ── Camera config từ environment
CAMERA_HOST   = os.getenv("CAMERA_HOST", "192.168.110.14")
CAMERA_PORT   = int(os.getenv("CAMERA_PORT", "80"))
CAMERA_USER   = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS   = os.getenv("CAMERA_PASS", "a12345678")

# ── Stream config
STREAM_WIDTH  = int(os.getenv("STREAM_WIDTH", "1280"))
STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "720"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("onvif-ptz")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import api.routes as routes_module
    from core.onvif_client import OnvifClient
    from core.ptz_service import PTZService
    from streaming.stream_relay import StreamRelay

    log.info(f"Connecting to camera {CAMERA_HOST}:{CAMERA_PORT} ...")

    client = OnvifClient(CAMERA_HOST, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
    try:
        info = client.discover()
        log.info(f"ONVIF discovered: profile={info['profile_token']} ptz={info['ptz_node_token']} time_offset={info['time_offset_s']:.1f}s")
    except Exception as e:
        log.error(f"ONVIF discovery failed: {e}")
        sys.exit(1)

    rtsp_url = client.get_stream_uri()
    log.info(f"RTSP stream: {rtsp_url}")

    relay = StreamRelay(rtsp_url, STREAM_WIDTH, STREAM_HEIGHT)
    relay.start()
    log.info("Stream relay started.")

    # Inject vào routes module
    routes_module.ptz_service   = PTZService(client, rtsp_url)
    routes_module.stream_relay  = relay
    routes_module.rtsp_url_main = rtsp_url
    routes_module.onvif_client  = client

    yield

    log.info("Shutting down...")
    relay.stop()


app = FastAPI(title="ONVIF PTZ Controller", lifespan=lifespan)

# ── Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── API routes
from api.routes import router as ptz_router
app.include_router(ptz_router)

# ── Root -> Web UI
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False,
    )
