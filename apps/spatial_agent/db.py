"""SQLite persistent database for Spatial Memory PTZ Agent."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("SPATIAL_DB_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/spatial_memory.sqlite3")))


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables for spatial cells, objects, and chat sessions."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Bảng lưu trữ 24 ô lưới không gian
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spatial_cells (
                cell_id TEXT PRIMARY KEY,
                camera_key TEXT,
                row_y INTEGER,
                col_x INTEGER,
                pan_deg REAL,
                tilt_deg REAL,
                pan_val REAL,
                tilt_val REAL,
                image_local_path TEXT,
                image_url TEXT,
                captured_at REAL,
                summary TEXT
            )
        """)

        # 2. Bảng lưu trữ các vật thể đã nhận diện trong không gian
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spatial_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cell_id TEXT,
                label TEXT,
                label_vi TEXT,
                category TEXT, -- 'STATIC' hoặc 'DYNAMIC'
                confidence REAL,
                bbox_json TEXT, -- [ymin, xmin, ymax, xmax] trong ô
                notes TEXT,
                status TEXT, -- 'PRESENT', 'MOVED', 'UNKNOWN'
                last_verified_at REAL,
                FOREIGN KEY (cell_id) REFERENCES spatial_cells(cell_id)
            )
        """)

        # 3. Bảng lưu trữ phiên chat và message history (chuẩn CRETA)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                channel TEXT DEFAULT 'cli',
                user_id TEXT DEFAULT 'guest',
                user_name TEXT DEFAULT 'Khách',
                title TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tool_name TEXT,
                tool_args_json TEXT,
                tool_call_id TEXT,
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
            )
        """)

        # 4. Metadata tổng hợp căn phòng
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS space_metadata (
                camera_key TEXT PRIMARY KEY,
                room_overview TEXT,
                last_full_scan REAL,
                total_cells INTEGER,
                grid_rows INTEGER,
                grid_cols INTEGER
            )
        """)
        conn.commit()


# ──────────────────────────────────────────────
# Spatial Cells & Objects Operations
# ──────────────────────────────────────────────

def save_spatial_cell(
    cell_id: str,
    camera_key: str,
    row_y: int,
    col_x: int,
    pan_deg: float,
    tilt_deg: float,
    pan_val: float,
    tilt_val: float,
    image_local_path: str,
    image_url: str = "",
    summary: str = "",
) -> None:
    now = time.time()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO spatial_cells (
                cell_id, camera_key, row_y, col_x, pan_deg, tilt_deg, pan_val, tilt_val,
                image_local_path, image_url, captured_at, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cell_id) DO UPDATE SET
                pan_deg=excluded.pan_deg,
                tilt_deg=excluded.tilt_deg,
                pan_val=excluded.pan_val,
                tilt_val=excluded.tilt_val,
                image_local_path=excluded.image_local_path,
                image_url=excluded.image_url,
                captured_at=excluded.captured_at,
                summary=excluded.summary
        """, (cell_id, camera_key, row_y, col_x, pan_deg, tilt_deg, pan_val, tilt_val, image_local_path, image_url, now, summary))
        conn.commit()


def save_spatial_objects(cell_id: str, objects: List[Dict[str, Any]]) -> None:
    """Xóa các object cũ của cell_id và cập nhật danh sách mới."""
    now = time.time()
    with get_connection() as conn:
        conn.execute("DELETE FROM spatial_objects WHERE cell_id = ?", (cell_id,))
        for obj in objects:
            conn.execute("""
                INSERT INTO spatial_objects (
                    cell_id, label, label_vi, category, confidence, bbox_json, notes, status, last_verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cell_id,
                obj.get("label", "").lower().strip(),
                obj.get("label_vi", ""),
                obj.get("category", "STATIC").upper(),
                obj.get("confidence", 0.9),
                json.dumps(obj.get("bbox", [])),
                obj.get("notes", ""),
                obj.get("status", "PRESENT"),
                now,
            ))
        conn.commit()


def save_space_metadata(camera_key: str, room_overview: str, total_cells: int, grid_rows: int, grid_cols: int) -> None:
    now = time.time()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO space_metadata (
                camera_key, room_overview, last_full_scan, total_cells, grid_rows, grid_cols
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_key) DO UPDATE SET
                room_overview=excluded.room_overview,
                last_full_scan=excluded.last_full_scan,
                total_cells=excluded.total_cells,
                grid_rows=excluded.grid_rows,
                grid_cols=excluded.grid_cols
        """, (camera_key, room_overview, now, total_cells, grid_rows, grid_cols))
        conn.commit()


def get_all_cells() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM spatial_cells ORDER BY row_y ASC, col_x ASC").fetchall()
        return [dict(r) for r in rows]


def get_cell_by_id(cell_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM spatial_cells WHERE cell_id = ?", (cell_id,)).fetchone()
        return dict(row) if row else None


def get_all_objects() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT o.*, c.pan_deg, c.tilt_deg, c.image_url, c.image_local_path
            FROM spatial_objects o
            JOIN spatial_cells c ON o.cell_id = c.cell_id
            ORDER BY o.category ASC, o.confidence DESC
        """).fetchall()
        return [dict(r) for r in rows]


def search_objects_in_memory(keyword: str) -> List[Dict[str, Any]]:
    """Tra cứu đối tượng trong SQLite bằng từ khoá."""
    kw = f"%{keyword.lower().strip()}%"
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT o.*, c.pan_deg, c.tilt_deg, c.image_url, c.image_local_path, c.summary as cell_summary
            FROM spatial_objects o
            JOIN spatial_cells c ON o.cell_id = c.cell_id
            WHERE lower(o.label) LIKE ? OR lower(o.label_vi) LIKE ? OR lower(o.notes) LIKE ?
            ORDER BY o.confidence DESC
        """, (kw, kw, kw)).fetchall()
        return [dict(r) for r in rows]


def update_object_verification(cell_id: str, label: str, status: str, notes: str = "") -> None:
    now = time.time()
    with get_connection() as conn:
        conn.execute("""
            UPDATE spatial_objects
            SET status = ?, last_verified_at = ?, notes = CASE WHEN ? != '' THEN ? ELSE notes END
            WHERE cell_id = ? AND lower(label) = lower(?)
        """, (status, now, notes, notes, cell_id, label))
        conn.commit()


def get_spatial_memory_summary() -> Dict[str, Any]:
    """Tạo bản tóm tắt text ngắn gọn về toàn bộ không gian cho LLM reasoning."""
    with get_connection() as conn:
        meta = conn.execute("SELECT * FROM space_metadata LIMIT 1").fetchone()
        meta_dict = dict(meta) if meta else {}

        cells = conn.execute("SELECT * FROM spatial_cells ORDER BY row_y ASC, col_x ASC").fetchall()
        objects = conn.execute("SELECT * FROM spatial_objects ORDER BY cell_id ASC").fetchall()

        cell_map = {}
        for c in cells:
            cell_map[c["cell_id"]] = {
                "cell_id": c["cell_id"],
                "pan_deg": c["pan_deg"],
                "tilt_deg": c["tilt_deg"],
                "summary": c["summary"],
                "static_objects": [],
                "dynamic_objects": [],
            }

        for o in objects:
            cid = o["cell_id"]
            if cid in cell_map:
                item = {
                    "label": o["label"],
                    "label_vi": o["label_vi"],
                    "category": o["category"],
                    "status": o["status"],
                    "notes": o["notes"],
                }
                if o["category"] == "STATIC":
                    cell_map[cid]["static_objects"].append(item)
                else:
                    cell_map[cid]["dynamic_objects"].append(item)

        return {
            "metadata": meta_dict,
            "total_cells": len(cells),
            "cells": list(cell_map.values()),
        }


# ──────────────────────────────────────────────
# Chat Sessions & Messages (CRETA Pattern)
# ──────────────────────────────────────────────

def ensure_session(session_id: str, channel: str = "cli", user_id: str = "guest", user_name: str = "Khách", title: str = "Phiên mới") -> None:
    now = time.time()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO chat_sessions (session_id, channel, user_id, user_name, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
        """, (session_id, channel, user_id, user_name, title, now, now))
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_args_json: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> None:
    now = time.time()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO chat_messages (session_id, role, content, tool_name, tool_args_json, tool_call_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, tool_name, tool_args_json, tool_call_id, now))
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        conn.commit()


def load_session_messages(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,)).fetchall()
        messages = [dict(r) for r in rows]
        if limit and len(messages) > limit:
            return messages[-limit:]
        return messages


# Tự động init DB khi nạp module
init_db()
