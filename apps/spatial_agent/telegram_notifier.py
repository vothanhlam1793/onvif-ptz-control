"""Telegram Notification Service for Spatial PTZ Agent."""

from __future__ import annotations

import io
import logging
import os
import requests
from typing import Optional, Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_config():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "8286157581:AAHaIG07LQbTiqQ6XJroYvrZ_kezhnzZrhQ")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "6884649143")
    api_url = f"https://api.telegram.org/bot{token}"
    return token, chat_id, api_url


def send_telegram_message(
    text: str,
    chat_id: Optional[Union[str, int]] = None,
    parse_mode: str = "HTML",
) -> dict:
    """Send text message via Telegram Bot API."""
    token, default_chat_id, api_url = _get_config()
    target_chat_id = str(chat_id or default_chat_id)
    url = f"{api_url}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return {"ok": False, "error": str(e)}


def send_telegram_photo(
    photo: Union[bytes, str],
    caption: str = "",
    chat_id: Optional[Union[str, int]] = None,
    parse_mode: str = "HTML",
) -> dict:
    """
    Send photo to Telegram chat.
    `photo` can be raw bytes or a public HTTPS URL.
    """
    token, default_chat_id, api_url = _get_config()
    target_chat_id = str(chat_id or default_chat_id)
    url = f"{api_url}/sendPhoto"

    try:
        if isinstance(photo, str) and (photo.startswith("http://") or photo.startswith("https://")):
            payload = {
                "chat_id": target_chat_id,
                "photo": photo,
                "caption": caption[:1024],
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, data=payload, timeout=15)
            return resp.json()
        elif isinstance(photo, bytes):
            files = {"photo": ("snapshot.jpg", io.BytesIO(photo), "image/jpeg")}
            data = {
                "chat_id": target_chat_id,
                "caption": caption[:1024],
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, data=data, files=files, timeout=20)
            return resp.json()
        else:
            return {"ok": False, "error": "Invalid photo format"}
    except Exception as e:
        logger.error(f"Failed to send Telegram photo: {e}")
        return {"ok": False, "error": str(e)}


def send_telegram_media_group(
    photos: list[dict],
    chat_id: Optional[Union[str, int]] = None,
) -> dict:
    """
    Gửi một album/loạt ảnh (Media Group) lên Telegram.
    photos: danh sách dict [{"photo": bytes_or_path, "caption": "..."}]
    """
    token, default_chat_id, api_url = _get_config()
    target_chat_id = str(chat_id or default_chat_id)
    url = f"{api_url}/sendMediaGroup"

    import json
    media_json = []
    files = {}

    for idx, item in enumerate(photos[:10]): # Telegram cho phép tối đa 10 ảnh / 1 media group
        attach_name = f"photo_{idx}"
        p = item.get("photo")
        cap = item.get("caption", "")
        
        media_item = {
            "type": "photo",
            "media": f"attach://{attach_name}",
            "caption": cap[:1024],
            "parse_mode": "HTML"
        }
        media_json.append(media_item)

        if isinstance(p, str) and os.path.exists(p):
            with open(p, "rb") as f:
                files[attach_name] = (f"{attach_name}.jpg", f.read(), "image/jpeg")
        elif isinstance(p, bytes):
            files[attach_name] = (f"{attach_name}.jpg", p, "image/jpeg")

    if not files:
        return {"ok": False, "error": "No valid photos to send"}

    data = {
        "chat_id": target_chat_id,
        "media": json.dumps(media_json)
    }

    try:
        resp = requests.post(url, data=data, files=files, timeout=30)
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram media group: {e}")
        return {"ok": False, "error": str(e)}
