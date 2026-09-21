"""
ONVIF SOAP Client - Low-latency, pure Python, no wsdl dependency.
Hỗ trợ: WS-Security Digest Auth, session keep-alive, dynamic capability discovery.
"""

import hashlib
import base64
import os
import time
import datetime
import re
import xml.etree.ElementTree as ET
import urllib.parse
from typing import Optional

import requests


NS = {
    "s":    "http://www.w3.org/2003/05/soap-envelope",
    "tt":   "http://www.onvif.org/ver10/schema",
    "tptz": "http://www.onvif.org/ver20/ptz/wsdl",
    "tds":  "http://www.onvif.org/ver10/device/wsdl",
    "trt":  "http://www.onvif.org/ver10/media/wsdl",
}

WSSE_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU_NS  = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"


class OnvifClient:
    """
    Thin ONVIF client. Tái sử dụng requests.Session cho keep-alive.
    Tự động discover PTZ URL, Profile Token, time offset khi khởi tạo.
    """

    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.base_url = f"http://{host}:{port}"
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/soap+xml; charset=utf-8"})
        self._time_offset: datetime.timedelta = datetime.timedelta(0)

        # Populated by discover()
        self.device_url: str = f"{self.base_url}/onvif/device_service"
        self.media_url: str = ""
        self.ptz_url: str = ""
        self.profile_token: str = ""
        self.ptz_node_token: str = ""
        
        # Hardware & Identity
        self.manufacturer: str = ""
        self.model: str = ""
        self.firmware_version: str = ""
        self.serial_number: str = ""
        self.hardware_id: str = ""
        self.mac_address: str = ""
        self.camera_key: str = ""

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _make_digest(self) -> tuple[str, str, str]:
        """Return (nonce_b64, created, digest_b64) for WS-Security."""
        cam_now = datetime.datetime.utcnow() + self._time_offset
        created = cam_now.strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce_raw = os.urandom(16)
        nonce_b64 = base64.b64encode(nonce_raw).decode()
        sha1 = hashlib.sha1()
        sha1.update(nonce_raw + created.encode() + self.password.encode())
        digest = base64.b64encode(sha1.digest()).decode()
        return nonce_b64, created, digest

    def _security_header(self) -> str:
        nonce, created, digest = self._make_digest()
        return f"""<s:Header>
  <wsse:Security xmlns:wsse="{WSSE_NS}" xmlns:wsu="{WSU_NS}">
    <wsse:UsernameToken>
      <wsse:Username>{self.username}</wsse:Username>
      <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</wsse:Password>
      <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce}</wsse:Nonce>
      <wsu:Created>{created}</wsu:Created>
    </wsse:UsernameToken>
  </wsse:Security>
</s:Header>"""

    def _post(self, url: str, body: str, auth: bool = True) -> ET.Element:
        soap = self._envelope(body, auth=auth)
        for attempt in range(3):
            try:
                resp = self._session.post(url, data=soap.encode("utf-8"), timeout=10)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                # Check SOAP Fault
                fault = root.find(".//s:Fault", NS)
                if fault is not None:
                    reason = fault.findtext(".//s:Text", default="Unknown", namespaces=NS)
                    raise RuntimeError(f"SOAP Fault: {reason}")
                return root
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(0.3)

    def _envelope(self, body: str, auth: bool = True) -> str:
        header = self._security_header() if auth else "<s:Header/>"
        return f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  {header}
  <s:Body>{body}</s:Body>
</s:Envelope>"""

    # ──────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────

    def sync_time(self):
        """Sync time offset với camera để tránh WS-Security 401."""
        body = '<GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
        root = self._post(self.device_url, body, auth=False)
        utc = root.find(".//tt:UTCDateTime", NS)
        if utc is None:
            return
        try:
            Y = int(utc.findtext("tt:Date/tt:Year", "0", NS))
            M = int(utc.findtext("tt:Date/tt:Month", "0", NS))
            D = int(utc.findtext("tt:Date/tt:Day", "0", NS))
            h = int(utc.findtext("tt:Time/tt:Hour", "0", NS))
            m = int(utc.findtext("tt:Time/tt:Minute", "0", NS))
            s = int(utc.findtext("tt:Time/tt:Second", "0", NS))
            cam_utc = datetime.datetime(Y, M, D, h, m, s)
            self._time_offset = cam_utc - datetime.datetime.utcnow()
        except Exception:
            pass

    def get_device_information(self) -> dict:
        """Lấy thông tin nhà sản xuất, model, serial, firmware từ GetDeviceInformation."""
        body = '<GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
        root = self._post(self.device_url, body)
        self.manufacturer     = root.findtext(".//tds:Manufacturer", "Generic", NS)
        self.model            = root.findtext(".//tds:Model", "PTZ_Camera", NS)
        self.firmware_version = root.findtext(".//tds:FirmwareVersion", "", NS)
        self.serial_number    = root.findtext(".//tds:SerialNumber", "", NS)
        self.hardware_id      = root.findtext(".//tds:HardwareId", "", NS)
        
        # Lấy thêm MAC address nếu có
        try:
            body_net = '<GetNetworkInterfaces xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
            root_net = self._post(self.device_url, body_net)
            for iface in root_net.findall(".//tds:NetworkInterfaces", NS):
                hw = iface.findtext(".//tt:HwAddress", "", NS)
                if hw:
                    self.mac_address = hw.replace(":", "").replace("-", "").lower()
                    break
        except Exception:
            pass

        # Tạo camera_key duy nhất (ưu tiên Manufacturer + Model, fallback Serial/MAC)
        clean_mfg = re.sub(r"[^a-zA-Z0-9_]", "_", self.manufacturer.strip().lower())
        clean_model = re.sub(r"[^a-zA-Z0-9_]", "_", self.model.strip().lower())
        
        if clean_mfg and clean_model:
            self.camera_key = f"{clean_mfg}_{clean_model}"
        elif self.serial_number:
            self.camera_key = f"cam_{self.serial_number.lower()}"
        elif self.mac_address:
            self.camera_key = f"cam_{self.mac_address}"
        else:
            self.camera_key = f"cam_{self.host.replace('.', '_')}"

        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "serial_number": self.serial_number,
            "hardware_id": self.hardware_id,
            "mac_address": self.mac_address,
            "camera_key": self.camera_key,
        }

    def discover(self):
        """
        Tự động lấy: device info, media_url, ptz_url, profile_token, ptz_node_token.
        Gọi 1 lần khi khởi tạo.
        """
        self.sync_time()
        self.get_device_information()

        # GetCapabilities
        body = '<GetCapabilities xmlns="http://www.onvif.org/ver10/device/wsdl"><Category>All</Category></GetCapabilities>'
        root = self._post(self.device_url, body)
        self.media_url = root.findtext(".//tt:Media/tt:XAddr", default=f"{self.base_url}/onvif/media_service", namespaces=NS)
        self.ptz_url   = root.findtext(".//tt:PTZ/tt:XAddr",   default=f"{self.base_url}/onvif/ptz_service",   namespaces=NS)

        # GetProfiles -> first video profile
        body = '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>'
        root = self._post(self.media_url, body)
        profile = root.find(".//trt:Profiles", NS)
        if profile is not None:
            self.profile_token = profile.get("token", "Profile000")
        else:
            self.profile_token = "Profile000"

        # GetNodes -> first PTZ node
        body = '<GetNodes xmlns="http://www.onvif.org/ver20/ptz/wsdl"/>'
        root = self._post(self.ptz_url, body)
        node = root.find(".//tptz:PTZNode", NS)
        if node is not None:
            self.ptz_node_token = node.get("token", "PTZ000")
        else:
            self.ptz_node_token = "PTZ000"

        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "camera_key": self.camera_key,
            "media_url": self.media_url,
            "ptz_url": self.ptz_url,
            "profile_token": self.profile_token,
            "ptz_node_token": self.ptz_node_token,
            "time_offset_s": self._time_offset.total_seconds(),
        }

    # ──────────────────────────────────────────────
    # PTZ Commands
    # ──────────────────────────────────────────────

    def continuous_move(self, pan: float, tilt: float, zoom: float = 0.0,
                        timeout_s: float = 30.0):
        """
        Quay liên tục. pan/tilt/zoom: [-1.0, 1.0].
        Dừng bằng stop(). Một số firmware Imou/LC bỏ qua lệnh không có Timeout,
        nên luôn gửi timeout như một failsafe nếu client không gọi stop().
        """
        zoom_xml = ""
        if abs(zoom) > 0.0001:
            zoom_xml = f'''\n    <Zoom xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/ZoomSpaces/VelocityGenericSpace"
      x="{zoom:.4f}"/>'''
        body = f"""<ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  <Velocity>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace"
      x="{pan:.4f}" y="{tilt:.4f}"/>{zoom_xml}
  </Velocity>
  <Timeout>PT{max(0.1, timeout_s):.1f}S</Timeout>
</ContinuousMove>"""
        self._post(self.ptz_url, body)

    def relative_move(self, pan: float, tilt: float, zoom: float = 0.0,
                      pan_speed: float = 0.5, tilt_speed: float = 0.5):
        """
        Dịch chuyển tương đối. Dùng cho Click-to-Center và AI Aim.
        pan/tilt: [-1.0, 1.0] - độ lệch so với vị trí hiện tại.
        """
        body = f"""<RelativeMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  <Translation>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationGenericSpace"
      x="{pan:.4f}" y="{tilt:.4f}"/>
    <Zoom xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/ZoomSpaces/TranslationGenericSpace"
      x="{zoom:.4f}"/>
  </Translation>
  <Speed>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/GenericSpeedSpace"
      x="{pan_speed:.4f}" y="{tilt_speed:.4f}"/>
  </Speed>
</RelativeMove>"""
        self._post(self.ptz_url, body)

    def absolute_move(self, pan: float, tilt: float, zoom: float = 0.0,
                      speed: float = 0.8):
        """Quay đến toạ độ tuyệt đối. pan/tilt: [-1.0, 1.0]."""
        body = f"""<AbsoluteMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  <Position>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace"
      x="{pan:.4f}" y="{tilt:.4f}"/>
    <Zoom xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace"
      x="{zoom:.4f}"/>
  </Position>
  <Speed>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/GenericSpeedSpace"
      x="{speed:.4f}" y="{speed:.4f}"/>
  </Speed>
</AbsoluteMove>"""
        self._post(self.ptz_url, body)

    def stop(self, pan_tilt: bool = True, zoom: bool = False):
        """Dừng chuyển động."""
        body = f"""<Stop xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  <PanTilt>{"true" if pan_tilt else "false"}</PanTilt>
  <Zoom>{"true" if zoom else "false"}</Zoom>
</Stop>"""
        self._post(self.ptz_url, body)

    def get_status(self) -> dict:
        """Trả về vị trí hiện tại và trạng thái IDLE/MOVING."""
        body = f"""<GetStatus xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
</GetStatus>"""
        root = self._post(self.ptz_url, body)
        pt = root.find(".//tt:PanTilt", NS)
        zoom = root.find(".//tt:Zoom", NS)
        move_pt = root.findtext(".//tt:MoveStatus/tt:PanTilt", "UNKNOWN", NS)
        move_z  = root.findtext(".//tt:MoveStatus/tt:Zoom",    "UNKNOWN", NS)
        return {
            "pan":         float(pt.get("x", 0)) if pt is not None else 0.0,
            "tilt":        float(pt.get("y", 0)) if pt is not None else 0.0,
            "zoom":        float(zoom.get("x", 0)) if zoom is not None else 0.0,
            "pan_tilt_status": move_pt,
            "zoom_status": move_z,
        }

    def get_presets(self) -> list[dict]:
        """Lấy danh sách preset đã lưu."""
        body = f"""<GetPresets xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
</GetPresets>"""
        root = self._post(self.ptz_url, body)
        presets = []
        for p in root.findall(".//tptz:Preset", NS):
            presets.append({
                "token": p.get("token"),
                "name":  p.findtext("tt:Name", "", NS),
            })
        return presets

    def goto_preset(self, preset_token: str, speed: float = 1.0):
        """Chuyển camera đến preset đã lưu."""
        body = f"""<GotoPreset xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  <PresetToken>{preset_token}</PresetToken>
  <Speed>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema"
      space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/GenericSpeedSpace"
      x="{speed:.4f}" y="{speed:.4f}"/>
  </Speed>
</GotoPreset>"""
        self._post(self.ptz_url, body)

    def set_preset(self, name: str, preset_token: Optional[str] = None) -> str:
        """Lưu vị trí hiện tại thành preset. Trả về token mới."""
        token_xml = f"<PresetToken>{preset_token}</PresetToken>" if preset_token else ""
        body = f"""<SetPreset xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  {token_xml}
  <PresetName>{name}</PresetName>
</SetPreset>"""
        root = self._post(self.ptz_url, body)
        return root.findtext(".//tptz:PresetToken", "", NS)

    def remove_preset(self, preset_token: str):
        """Xoá preset."""
        body = f"""<RemovePreset xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{self.profile_token}</ProfileToken>
  <PresetToken>{preset_token}</PresetToken>
</RemovePreset>"""
        self._post(self.ptz_url, body)

    def get_stream_uri(self) -> str:
        """Lấy RTSP URI stream chính."""
        body = f"""<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">
  <StreamSetup>
    <Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>
    <Transport xmlns="http://www.onvif.org/ver10/schema">
      <Protocol>RTSP</Protocol>
    </Transport>
  </StreamSetup>
  <ProfileToken>{self.profile_token}</ProfileToken>
</GetStreamUri>"""
        root = self._post(self.media_url, body)
        uri = root.findtext(".//tt:Uri", "", NS)
        # Inject credentials vào URI (URL encode password để tránh lỗi ký tự đặc biệt @, #, !)
        if uri and "://" in uri:
            scheme, rest = uri.split("://", 1)
            # Tách nếu đã có user:pass cũ trong rest
            if "@" in rest:
                rest = rest.split("@", 1)[1]
            safe_user = urllib.parse.quote(self.username, safe="")
            safe_pass = urllib.parse.quote(self.password, safe="")
            uri = f"{scheme}://{safe_user}:{safe_pass}@{rest}"
        return uri
