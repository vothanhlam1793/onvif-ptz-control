"""Interactive CLI Runner for Spatial Memory PTZ Agent."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()

from core.onvif_client import OnvifClient
from core.virtual_ptz import VirtualPTZTracker
from apps.spatial_agent.tools import set_ptz_hardware
from apps.spatial_agent.agent import build_spatial_agent, execute_spatial_turn
from apps.spatial_agent.db import get_all_cells, get_all_objects, get_spatial_memory_summary
from apps.spatial_agent.telegram_notifier import send_telegram_message, send_telegram_photo
from streaming.stream_relay import snapshot

console = Console()

CAMERA_HOST = os.getenv("CAMERA_HOST", "192.168.110.110")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "80"))
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS", "asrkpVg10!@#")


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🤖 SPATIAL MEMORY PTZ AGENT (Active Visual Search & Cueing)[/bold cyan]\n"
        "[dim]Hệ thống định vị không gian 360° kết hợp SQLite + MinIO + Gemini 3.7 VLM[/dim]",
        border_style="cyan"
    ))
    console.print("[dim]Lệnh nhanh: [bold]scan[/bold] (quét phòng mới) | [bold]reindex[/bold] (AI nhận diện lại không xoay cam) | [bold]calib[/bold] (hiệu chuẩn) | [bold]map[/bold] (vật thể) | [bold]status[/bold] (góc) | [bold]q[/bold] (thoát)[/dim]\n")


def display_objects_table():
    objs = get_all_objects()
    if not objs:
        console.print("[yellow]⚠️ Chưa có dữ liệu không gian. Hãy gõ 'scan' để bắt đầu quét 24 ô lưới![/yellow]")
        return

    table = Table(title="🗺️ BẢN ĐỒ TRI THỨC VẬT THỂ TRONG PHÒNG", border_style="green")
    table.add_column("Cell ID", style="cyan", justify="center")
    table.add_column("Tên vật thể (EN)", style="white")
    table.add_column("Tên tiếng Việt", style="bold green")
    table.add_column("Phân loại", style="magenta", justify="center")
    table.add_column("Góc Pan/Tilt", style="yellow")
    table.add_column("Trạng thái", style="blue", justify="center")
    table.add_column("Ghi chú", style="dim")

    for o in objs:
        cat_badge = "[bold red]DYNAMIC[/bold red]" if o["category"] == "DYNAMIC" else "[bold blue]STATIC[/bold blue]"
        angle_str = f"P:{o['pan_deg']:.1f}° | T:{o['tilt_deg']:.1f}°"
        table.add_row(
            o["cell_id"],
            o["label"],
            o["label_vi"] or o["label"],
            cat_badge,
            angle_str,
            o["status"],
            o["notes"] or "",
        )
    console.print(table)


async def async_main():
    print_banner()

    # 1. Khởi tạo kết nối ONVIF & Tracker
    with console.status("[bold green]Đang kết nối Camera ONVIF PTZ...[/bold green]"):
        try:
            client = OnvifClient(CAMERA_HOST, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
            info = client.discover()
            rtsp_url = client.get_stream_uri()
            tracker = VirtualPTZTracker(client, rtsp_url=rtsp_url)
            tracker.load_calibration()
            set_ptz_hardware(client, tracker, rtsp_url)
            console.print(f"[green]✓ Kết nối thành công: {client.manufacturer} {client.model} ({CAMERA_HOST})[/green]")
        except Exception as e:
            console.print(f"[red]❌ Lỗi kết nối camera: {e}[/red]")
            return

    # 2. Xây dựng Agent LangGraph
    agent_app = build_spatial_agent()
    session_id = f"cli_session_{int(time.time())}"

    # 3. Vòng lặp tương tác CLI
    while True:
        try:
            # Dùng input() chuẩn và import readline nếu có để tránh lỗi raw terminal ^M
            try:
                import readline
            except ImportError:
                pass

            user_input = input("\nBạn > ").strip()
            # Dọn dẹp ký tự điều khiển \r (^M) nếu terminal bị dính
            user_input = user_input.replace("\r", "").replace("^M", "").strip()
            if not user_input:
                continue

            if user_input.lower() in ("q", "quit", "exit"):
                console.print("[dim]Tạm biệt![/dim]")
                break

            if user_input.lower() == "map":
                display_objects_table()
                continue

            if user_input.lower() == "status":
                st = tracker.get_status()
                console.print(f"[yellow]Toạ độ hiện tại: Pan = {st['pan_deg']}°, Tilt = {st['tilt_deg']}° (Homed: {st['is_homed']})[/yellow]")
                continue

            if user_input.lower() == "calib":
                user_input = "Hãy hiệu chuẩn lại thông số phần cứng của camera ngay bây giờ."

            if user_input.lower() == "reindex":
                user_input = "Hãy chạy re-index AI phân tích lại toàn bộ kho ảnh có sẵn không cần xoay camera."

            t_start = time.time()
            with console.status("[bold yellow]🤖 Agent đang suy luận và điều khiển PTZ...[/bold yellow]"):
                result = await execute_spatial_turn(
                    agent_app=agent_app,
                    session_id=session_id,
                    user_text=user_input,
                )
            latency = time.time() - t_start

            # In chi tiết các tool call đã chạy
            for action in result.get("tool_actions", []):
                t_name = action.get("name")
                console.print(f"[dim]⚙️ [Tool Action] {t_name}[/dim]")

            # In câu trả lời chính thức kèm độ trễ phản hồi
            reply = result.get("reply", "")
            console.print(Panel(
                Markdown(reply),
                title=f"[bold green]PTZ Agent[/bold green] [dim](⏱️ {latency:.2f}s)[/dim]",
                border_style="green",
                subtitle=f"[dim]Latency: {latency:.2f}s[/dim]"
            ))

            # ── TỰ ĐỘNG GỬI KẾT QUẢ + HÌNH ẢNH TELEGRAM SAU MỖI LỆNH ──
            # Nếu trong turn chưa có tool send_telegram_alert_tool được gọi, tự động gửi ảnh live + kết quả
            tool_names = [a.get("name") for a in result.get("tool_actions", [])]
            if "send_telegram_alert_tool" not in tool_names and rtsp_url:
                try:
                    fb = snapshot(rtsp_url, width=1280, height=720)
                    caption = f"🤖 [PTZ CLI Agent] ({latency:.2f}s)\n{reply}"
                    if len(caption) > 1024:
                        caption = caption[:1020] + "..."
                    if fb:
                        send_telegram_photo(photo=fb, caption=caption)
                    else:
                        send_telegram_message(text=caption)
                except Exception as ex:
                    logger.warning(f"Lỗi auto-send Telegram: {ex}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[red]Lỗi thực thi: {e}[/red]")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
