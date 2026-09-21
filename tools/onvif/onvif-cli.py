#!/usr/bin/env python3
"""Interactive, login-gated ONVIF/PTZ operator console."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(os.getenv("ONVIF_PTZ_CONFIG", Path(__file__).with_name("config.json")))
SNAPSHOT_DIR = ROOT / "outputs" / "ptz_snapshots"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from tools.onvif.ptz_tool import PtzConnection, PtzRuntimeAdapter, PtzTool, PtzToolError


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="ONVIF PTZ operator console")
    command.add_argument("--env-file", default=ROOT / ".env", type=Path)
    command.add_argument("--config", default=DEFAULT_CONFIG, type=Path, help="Local device profile file")
    sub = command.add_subparsers(dest="command")
    probe = sub.add_parser("probe", help="Diagnose selected/profile device; does not move it")
    probe.add_argument("--profile", help="Saved device profile name")
    return command


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(config.get("devices"), list):
            return config
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        print(f"Cảnh báo: không đọc được config {path}; tạo danh sách mới.")
    return {"version": 2, "selected": "", "devices": []}


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def selected_device(config: dict, name: str = "") -> dict:
    selection = name or config.get("selected", "")
    matches = [item for item in config["devices"] if item.get("name", "").casefold() == selection.casefold()]
    if len(matches) != 1:
        raise PtzToolError("Chưa chọn thiết bị. Hãy chọn một profile trong menu.")
    return matches[0]


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{text}{suffix}: ").strip() or default


def add_device(config: dict, path: Path) -> None:
    print("\nThêm thiết bị ONVIF")
    name = prompt("Tên thiết bị")
    if not name or any(item.get("name", "").casefold() == name.casefold() for item in config["devices"]):
        raise PtzToolError("Tên thiết bị trống hoặc đã tồn tại.")
    host = prompt("Host/IP")
    if not host:
        raise PtzToolError("Host/IP là bắt buộc.")
    port = prompt("ONVIF port", "80")
    username = prompt("ONVIF username", "admin")
    password = getpass.getpass("ONVIF password: ")
    try:
        port_number = int(port)
    except ValueError as exc:
        raise PtzToolError("ONVIF port phải là số.") from exc
    if not 1 <= port_number <= 65535 or not username or not password:
        raise PtzToolError("Port, username và password là bắt buộc.")
    config["version"] = 2
    config["devices"].append({"name": name, "host": host, "port": port_number, "username": username, "password": password})
    config["selected"] = name
    save_config(path, config)
    print(f"Đã lưu {name}.")


def edit_device(config: dict, path: Path, device: dict) -> dict:
    print("\nSửa thiết bị, Enter để giữ giá trị hiện tại.")
    name = prompt("Tên thiết bị", device["name"])
    if any(item is not device and item.get("name", "").casefold() == name.casefold() for item in config["devices"]):
        raise PtzToolError("Tên thiết bị đã tồn tại.")
    host = prompt("Host/IP", device["host"])
    port = prompt("ONVIF port", str(device["port"]))
    username = prompt("ONVIF username", device["username"])
    password = getpass.getpass("ONVIF password [Enter để giữ]: ") or device.get("password", "")
    try:
        port_number = int(port)
    except ValueError as exc:
        raise PtzToolError("ONVIF port phải là số.") from exc
    if not name or not host or not username or not password or not 1 <= port_number <= 65535:
        raise PtzToolError("Thông tin thiết bị không hợp lệ.")
    config["version"] = 2
    device.update({"name": name, "host": host, "port": port_number, "username": username, "password": password})
    if config.get("selected", "").casefold() != name.casefold():
        config["selected"] = name
    save_config(path, config)
    print("Đã cập nhật thiết bị.")
    return device


def delete_device(config: dict, path: Path, device: dict) -> bool:
    confirmation = input(f"Xóa {device['name']}? Gõ XOA để xác nhận: ").strip()
    if confirmation != "XOA":
        print("Đã hủy.")
        return False
    config["devices"].remove(device)
    if config.get("selected") == device["name"]:
        config["selected"] = config["devices"][0]["name"] if config["devices"] else ""
    save_config(path, config)
    print(f"Đã xóa {device['name']}.")
    return True


def connection_for(device: dict, config: dict | None = None, config_path: Path | None = None) -> PtzConnection:
    password = device.get("password", "") or os.getenv("CAMERA_PASS", "")
    if not password and sys.stdin.isatty():
        password = getpass.getpass(f"ONVIF password cho {device['username']}@{device['host']}: ")
    if not password:
        raise PtzToolError("Profile chưa có password. Sửa profile hoặc đặt CAMERA_PASS trong .env để migrate.")
    if not device.get("password") and config is not None and config_path is not None:
        device["password"] = password
        config["version"] = 2
        save_config(config_path, config)
        print(f"Đã lưu password vào profile {device['name']}.")
    return PtzConnection(device["host"], device["port"], device["username"], password)


def direction_from_key(key: str) -> str:
    mapping = {
        "7": "up-left", "8": "up", "9": "up-right", "4": "left",
        "6": "right", "1": "down-left", "2": "down", "3": "down-right",
        "+": "zoom-in", "-": "zoom-out",
    }
    if key not in mapping:
        raise PtzToolError("Hướng không hợp lệ. Dùng 789/456/123 hoặc +/-. ")
    return mapping[key]


def direction_vector(direction: str, speed: float) -> str:
    vectors = {
        "left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1),
        "up-left": (-1, 1), "up-right": (1, 1), "down-left": (-1, -1), "down-right": (1, -1),
        "zoom-in": (0, 0), "zoom-out": (0, 0),
    }
    pan, tilt = vectors[direction]
    if direction.startswith("zoom"):
        return f"Zoom={speed if direction == 'zoom-in' else -speed:+.2f}"
    return f"PanTilt(x={pan * speed:+.2f}, y={tilt * speed:+.2f})"


def save_snapshot(adapter: PtzRuntimeAdapter, device: dict) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in device["name"])
    output = SNAPSHOT_DIR / f"{safe_name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    output.write_bytes(adapter.snapshot())
    print(f"Đã lưu snapshot: {output.relative_to(ROOT)}")
    return output


def show_probe(result: dict) -> None:
    device = result["device"]
    print("\nKiểm tra kết nối")
    print(f"Camera: {device.get('manufacturer', '?')} {device.get('model', '?')}")
    print(f"RTSP: {'Sẵn sàng' if result['rtsp_ready'] else 'Không khả dụng'}")
    print("PTZ: điều khiển theo nudge (camera không hỗ trợ absolute position đáng tin cậy).")


def diagnose(config: dict, config_path: Path, device: dict) -> None:
    tool = PtzTool(connection=connection_for(device, config, config_path))
    show_probe(tool.probe())


def action_menu(adapter: PtzRuntimeAdapter, device: dict, settings: dict) -> None:
    while True:
        print("\nTác vụ · [1] Speed  [2] Step  [3] Move một bước  [4] Lưu snapshot  [5] Motion check  [6] Thông tin  [0] Quay lại keypad")
        choice = input("Chọn: ").strip()
        try:
            if choice == "0":
                return
            if choice == "1":
                settings["speed"] = min(1.0, max(0.05, float(prompt("Speed 0.05-1.0", str(settings["speed"])))) )
                print(f"Speed: {settings['speed']:.2f}")
            elif choice == "2":
                settings["step"] = min(2.0, max(0.1, float(prompt("Step seconds 0.1-2.0", str(settings["step"])))) )
                print(f"Step: {settings['step']:.1f}s")
            elif choice == "3":
                print("7 8 9 / 4 6 / 1 2 3 / + -")
                direction = direction_from_key(input("Hướng: ").strip())
                adapter.move(direction, settings["speed"], settings["step"])
                print(f"Đã quay {direction}.")
            elif choice == "4":
                save_snapshot(adapter, device)
            elif choice == "5":
                direction = direction_from_key(input("Hướng test: ").strip())
                if input(f"Camera sẽ quay {direction} trong {settings['step']:.1f}s. Tiếp tục? [y/N]: ").strip().lower() != "y":
                    continue
                result = adapter.probe_motion(direction, settings["speed"], settings["step"])
                status = "PASS: phát hiện chuyển động" if result["motion_detected"] else "FAIL: không thấy thay đổi hình ảnh"
                print(f"{status} (difference={result['difference']:.1f})")
            elif choice == "6":
                status = adapter.status()
                print(f"{device['name']} · {device['host']}:{device['port']} · {status['state']}")
            else:
                print("Lựa chọn không hợp lệ.")
        except (PtzToolError, ValueError) as exc:
            print(f"Không thực hiện được: {exc}")


def keypad(adapter: PtzRuntimeAdapter, device: dict, settings: dict | None = None) -> None:
    settings = settings or {"speed": 0.3, "step": 0.5}
    while adapter.state == adapter.READY:
        print("\nKẾT NỐI THÀNH CÔNG")
        print("       [7] [8] [9]\n       [4] [5] [6]\n       [1] [2] [3]")
        print("+/- zoom · 5 hoặc x dừng · s snapshot · m tác vụ · q ngắt kết nối")
        key = input("Chọn nút rồi nhấn Enter: ").strip().lower()
        try:
            if key == "q":
                adapter.logout()
                print("Đã ngắt kết nối.")
                return
            if key == "m":
                action_menu(adapter, device, settings)
                continue
            if key in ("5", "x"):
                adapter.stop()
                print(f"Đã nhận [{key}] → STOP")
                continue
            if key == "s":
                print("Đã nhận [s] → SNAPSHOT")
                save_snapshot(adapter, device)
                continue
            direction = direction_from_key(key)
            vector = direction_vector(direction, settings["speed"])
            print(f"Đã nhận [{key}] → {direction} → {vector}")
            adapter.move(direction, settings["speed"], settings["step"])
        except PtzToolError as exc:
            print(f"Không thực hiện được: {exc}")


def connect_and_control(config: dict, config_path: Path, device: dict) -> None:
    adapter = PtzRuntimeAdapter()
    try:
        info = adapter.login(connection_for(device, config, config_path))
        print(f"Đã kết nối {info.get('manufacturer', '')} {info.get('model', '')}.")
        keypad(adapter, device)
    except (PtzToolError, ValueError) as exc:
        print(f"Không kết nối được: {exc}")
    finally:
        adapter.logout()


def device_dashboard(config: dict, path: Path, device: dict) -> None:
    while True:
        print(f"\n{device['name']} · {device['host']}:{device['port']} · {device['username']}")
        print("[C] Connect & control  [D] Diagnose  [E] Edit  [X] Delete  [B] Back")
        choice = input("Chọn: ").strip().lower()
        try:
            if choice == "b":
                return
            if choice == "c":
                connect_and_control(config, path, device)
            elif choice == "d":
                diagnose(config, path, device)
            elif choice == "e":
                device = edit_device(config, path, device)
            elif choice == "x" and delete_device(config, path, device):
                return
            else:
                print("Lựa chọn không hợp lệ.")
        except (PtzToolError, ValueError) as exc:
            print(f"Không thực hiện được: {exc}")


def manage_devices(config: dict, path: Path) -> None:
    if not config["devices"]:
        print("Chưa có thiết bị để quản lý.")
        return
    for index, device in enumerate(config["devices"], 1):
        print(f"[{index}] {device['name']} · {device['host']}:{device['port']}")
    choice = input("Chọn thiết bị, hoặc 0 để quay lại: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(config["devices"]):
        device = config["devices"][int(choice) - 1]
        config["selected"] = device["name"]
        save_config(path, config)
        device_dashboard(config, path, device)


def main_menu(config_path: Path) -> int:
    config = load_config(config_path)
    while True:
        print("\nONVIF PTZ")
        for index, device in enumerate(config["devices"], 1):
            selected = " *" if device["name"] == config.get("selected") else ""
            print(f"[{index}] {device['name']} · {device['host']}:{device['port']}{selected}")
        print("[A] Add device  [M] Manage devices  [0] Exit")
        choice = input("Chọn: ").strip().lower()
        try:
            if choice == "0":
                return 0
            if choice == "a":
                add_device(config, config_path)
            elif choice == "m":
                manage_devices(config, config_path)
            elif choice.isdigit() and 1 <= int(choice) <= len(config["devices"]):
                device = config["devices"][int(choice) - 1]
                config["selected"] = device["name"]
                save_config(config_path, config)
                device_dashboard(config, config_path, device)
            else:
                print("Chọn một thiết bị, A, M hoặc 0.")
        except (PtzToolError, ValueError) as exc:
            print(f"Không thực hiện được: {exc}")


def probe_command(config_path: Path, profile: str | None) -> None:
    config = load_config(config_path)
    if profile or config.get("selected"):
        device = selected_device(config, profile or "")
        show_probe(PtzTool(connection=connection_for(device, config, config_path)).probe())
    else:
        show_probe(PtzTool.from_environment().probe())


def main() -> int:
    args = parser().parse_args()
    load_dotenv(args.env_file)
    try:
        if args.command == "probe":
            probe_command(args.config, args.profile)
        else:
            return main_menu(args.config)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except PtzToolError as exc:
        print(f"Không thực hiện được: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
