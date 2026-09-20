# StationWatch · ONVIF PTZ Spatial & Mechanical Control System

Hệ thống điều khiển cơ khí, định vị không gian ảo (Virtual PTZ Spatial Engine) và ghép toàn cảnh cầu 3D (Spherical Panorama) cho các dòng Camera an ninh PTZ giá rẻ qua giao thức chuẩn ONVIF.

---

## 🌟 Tính Năng Nổi Bật

1. **Virtual PTZ Spatial Coordinate Engine:**
   - Tạo hệ toạ độ ảo chuẩn hoá $(Pan, Tilt, Zoom) \in [-1.0, 1.0]$ cho camera phổ thông không có optical encoder.
   - Chu trình **Mechanical Homing** tự động định vị điểm gốc vật lý.
   - Ánh xạ song ánh 2 chiều giữa toạ độ ảo và góc quay vật lý thực tế $(Pan^\circ, Tilt^\circ)$.

2. **Snapshot Kèm Metadata Không Gian:**
   - Mỗi frame chụp đều đính kèm toạ độ $(P, T, Z)$ và góc thực $(Pan^\circ, Tilt^\circ)$ phục vụ bài toán AI Person Detection, Slew-to-Cue, Radar Mapping.

3. **Điều Khiển Ngắm Mục Tiêu Tức Thì (Click-to-Point / Slew-to-Cue):**
   - Click chuột vào bất kỳ điểm nào trên khung hình $\rightarrow$ Camera tự động căn tâm vào đối tượng.
   - API nhận toạ độ góc quay hoặc pixel từ AI ngoài để quay camera ngay lập tức.

4. **Hot-Swap & Live Reconnect Camera:**
   - Đổi camera hoặc thông tin kết nối ONVIF trực tiếp trên Web UI không cần khởi động lại server.
   - Tự động mã hoá URL Password có ký tự đặc biệt (`!`, `@`, `#`...).

5. **Xưởng Ghép Ảnh Panorama AI (Spherical Projection):**
   - Chiếu cầu ngược (Backward Equirectangular Warping) theo góc PTZ ground-truth.
   - Hoạt động ổn định trên cả tường trơn và trần trắng (không phụ thuộc SIFT/Feature matching).

---

## 📂 Cấu Trúc Dự Án (Clean Architecture)

```text
onvif-ptz/
├── apps/                   # Ứng dụng AI Demo độc lập
│   └── spatial_agent/      # Demo: Active Spatial Memory & Visual Cueing Agent
│       ├── agent.py        # LangGraph ReAct Workflow + compact_messages
│       ├── tools.py        # LangChain Tools (Scan, Query, Slew, Telegram)
│       ├── prompts.py      # System prompts & VLM templates
│       ├── db.py           # SQLite storage (data/spatial_memory.sqlite3)
│       ├── minio_client.py # Upload S3 MinIO (minio.nvlit.asia)
│       ├── telegram_notifier.py # Gửi ảnh và alert tới Telegram (@vothanhlam1793)
│       └── cli.py          # Interactive Terminal Runner
├── api/                    # FastAPI REST & WebSocket API endpoints
│   ├── __init__.py
│   └── routes.py
├── core/                   # ONVIF SOAP Client & PTZ Mechanical Engine
│   ├── __init__.py
│   ├── onvif_client.py     # SOAP WS-Security Client thuần Python
│   ├── ptz_service.py      # Dịch vụ quản lý di chuyển, presets, patrol
│   ├── virtual_ptz.py      # Bộ máy toạ độ ảo & Homing vật lý
│   ├── coordinate.py       # Chuyển đổi Pixel -> Vector PTZ
│   └── auto_calibration.py # Tự động đo FOV, thời gian quay và lập hồ sơ
├── data/                   # CSDL SQLite cho Spatial Agent (spatial_memory.sqlite3)
├── panorama/               # Module quét và ghép ảnh toàn cảnh cầu
│   ├── __init__.py
│   ├── agent.py            # LangGraph điều phối quy trình quét
│   ├── stitcher.py         # Pipeline ghép ảnh đa tầng
│   ├── sweeper.py          # Quét lưới 2D/3D điều khiển camera
│   └── spherical_projector.py # Chiếu cầu ngược Equirectangular Warping
├── streaming/              # RTSP -> MJPEG Low-latency Relay
│   ├── __init__.py
│   └── stream_relay.py
├── static/                 # Giao diện Web Console (StationWatch)
├── tests/                  # Bộ kiểm thử Unit Test
├── cli.py                  # CLI Launcher cho Spatial Agent
├── main.py                 # Entrypoint Web Server PTZ Core
└── README.md
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt môi trường
Yêu cầu: `Python 3.10+` và đã cài đặt `ffmpeg`.

```bash
# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Thiết lập cấu hình `.env`
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```
Nội dung cấu hình mẫu:
```ini
# ONVIF Camera Config
CAMERA_HOST=192.168.110.110
CAMERA_PORT=80
CAMERA_USER=admin
CAMERA_PASS=your_password_here

# Stream Config
STREAM_WIDTH=1280
STREAM_HEIGHT=720

# VLM / 9Router Config
NINEROUTER_BASE_URL=https://9router.camerangochoang.com/v1
NINEROUTER_API_KEY=your_api_key_here
VLM_MODEL=ag/gemini-3.7-flash-high
```

### 3. Khởi chạy Server
```bash
python main.py
```
Mở trình duyệt truy cập: `http://localhost:8080`.

---

## 📡 Danh Mục API Chuẩn

### 1. Điều Khiển Cơ Khí & Toạ Độ Không Gian
| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/api/ptz/virtual/home` | Chạy quy trình Homing cơ học |
| `GET` | `/api/ptz/status` | Lấy toạ độ ảo $(P, T)$, góc thực tế $(Pan^\circ, Tilt^\circ)$ |
| `POST` | `/api/ptz/virtual/goto` | Di chuyển tới toạ độ ảo `{"pan": 0.5, "tilt": -0.2, "speed": 0.8}` |
| `POST` | `/api/ptz/virtual/goto_angle` | Di chuyển tới góc vật lý `{"pan_deg": 180.0, "tilt_deg": 35.0}` |
| `POST` | `/api/ptz/click` | Căn tâm theo pixel click `{"x": 640, "y": 360}` |

### 2. Snapshot & Telemetry
| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/snapshot` | Chụp frame JPEG (đính kèm telemetry trong HTTP Headers) |
| `GET` | `/ptz/snapshot_with_telemetry` | Chụp frame kèm metadata JSON $(P, T, Z, Pan^\circ, Tilt^\circ)$ |

### 3. Cài Đặt & Hot-Swap Camera
| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/settings` | Đọc cấu hình hệ thống & RTSP URL |
| `POST` | `/camera/test_sync` | Bắt tay ONVIF thử nghiệm & lấy device info |
| `POST` | `/settings/test_llm` | Kiểm tra kết nối model AI LLM / VLM |
| `POST` | `/settings/save` | Lưu cấu hình & Hot-Swap camera trực tiếp |

---

## 🧪 Chạy Kiểm Thử (Unit Tests)

```bash
python tests/test_virtual_engine.py
python tests/test_settings.py
```

---

## 🔮 Lộ Trình Phát Triển: Version 2 (Full PTZ & Hierarchical Scale-Space)

### 1. Mở Rộng Hệ Trục Toạ Độ 3 Chiều $(Pan, Tilt, Zoom)$
- **Hệ trục:** $Pan \in [-1.0, 1.0]$, $Tilt \in [-1.0, 1.0]$, $Zoom \in [0.0, 1.0]$.
- **Không gian Mặt Cầu Đa Tầng Phân Giải (Hierarchical Spherical Scale-Space):**
  - $Z = 0.0$ (Wide Base): Ảnh toàn cảnh 360° bao quát, góc nhìn lớn ($HFOV \approx 85^\circ$).
  - $Z \in (0.0, 1.0]$ (Tele Scale): Zoom quang học/kỹ thuật số chi tiết ($HFOV \approx 10^\circ \rightarrow 15^\circ$).
- **Công thức biến thiên tiêu cự và trường nhìn:**
  $$HFOV(Z) = 2 \cdot \arctan\left(\frac{\tan(HFOV_{wide} / 2)}{1 + Z \cdot (M_{max} - 1)}\right), \quad f(Z) = f_{wide} \cdot (1 + Z \cdot (M_{max} - 1))$$

### 2. Mô Hình Ma Trận Biến Đổi Tuyến Tính Đa Tầng (Homography Transform)
Vì camera PTZ quay và zoom tại tâm quang học cố định ($\mathbf{t} = \mathbf{0}$, không có sai lệch thị sai Parallax), quan hệ giữa ảnh góc rộng và ảnh zoom cận cảnh được tính hoàn toàn bằng ma trận đại số tuyến tính $3 \times 3$:

$$\mathbf{H}_{wide \to zoom} = \mathbf{K}_{zoom} \cdot \mathbf{R}_{zoom}^{-1} \cdot \mathbf{R}_{wide} \cdot \mathbf{K}_{wide}^{-1}$$

$$\begin{bmatrix} u_{zoom} \\ v_{zoom} \\ 1 \end{bmatrix} \sim \mathbf{H}_{wide \to zoom} \begin{bmatrix} u_{wide} \\ v_{wide} \\ 1 \end{bmatrix}$$

- **Ứng dụng:**
  - **Slew-to-Cue 2 Chiều:** Phát hiện đối tượng/người khả nghi trên bản đồ Panorama toàn cảnh $\rightarrow$ Tính vector ma trận $\rightarrow$ Quay camera và Zoom cận cảnh $Z$ vào mục tiêu với độ nét cao.
  - **Auto Re-mapping:** Chiếu ngược các Bounding Box nhận diện từ tầng Zoom sắc nét về toạ độ neo của bản đồ toàn cảnh gốc.

