/* ============================================================
   StationWatch PTZ Controller - app.js
   Architecture: WebSocket Telemetry + Low-latency REST Commands
   ============================================================ */

const API = "";
let FRAME_W = 1280;
let FRAME_H = 720;
let currentSpeed = 0.40;
let currentPresets = [];
let isPatrolRunning = false;

// ──────────────────────────────────────────────
// Navigation & Tab Switching
// ──────────────────────────────────────────────
const navButtons = document.querySelectorAll(".nav-btn");
const viewSections = document.querySelectorAll(".view-section");
const pageBreadcrumb = document.getElementById("page-breadcrumb");
const pageTitle = document.getElementById("page-title");

const TAB_TITLES = {
  live: { breadcrumb: "GIÁM SÁT · CAMERA PTZ", title: "Trực tiếp & Điều khiển PTZ" },
  panorama: { breadcrumb: "CÔNG CỤ AI · TOÀN CẢNH", title: "Xưởng Ghép Ảnh Panorama AI" },
  patrol: { breadcrumb: "TỰ ĐỘNG HÓA · TUẦN TRA", title: "Thiết lập Tuần tra & Điểm nhớ" }
};

navButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const tabKey = btn.dataset.tab;
    navButtons.forEach(b => b.classList.remove("active"));
    viewSections.forEach(s => s.classList.remove("active"));
    
    btn.classList.add("active");
    const targetSection = document.getElementById(`tab-${tabKey}`);
    if (targetSection) targetSection.classList.add("active");

    if (TAB_TITLES[tabKey]) {
      pageBreadcrumb.textContent = TAB_TITLES[tabKey].breadcrumb;
      pageTitle.textContent = TAB_TITLES[tabKey].title;
    }
  });
});

// ──────────────────────────────────────────────
// REST Utilities
// ──────────────────────────────────────────────
async function post(path, body = {}) {
  try {
    const res = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    return await res.json();
  } catch (err) {
    console.error("POST Error:", path, err);
  }
}

async function get(path) {
  try {
    const res = await fetch(API + path);
    return await res.json();
  } catch (err) {
    console.error("GET Error:", path, err);
  }
}

// ──────────────────────────────────────────────
// Stream & Click-to-Center HUD
// ──────────────────────────────────────────────
const streamImg = document.getElementById("stream-img");
const videoWrapper = document.getElementById("video-wrapper");

streamImg.addEventListener("load", () => {
  FRAME_W = streamImg.naturalWidth || 1280;
  FRAME_H = streamImg.naturalHeight || 720;
});

videoWrapper.addEventListener("click", async (ev) => {
  const rect = streamImg.getBoundingClientRect();
  const scaleX = FRAME_W / rect.width;
  const scaleY = FRAME_H / rect.height;
  const clickX = (ev.clientX - rect.left) * scaleX;
  const clickY = (ev.clientY - rect.top) * scaleY;

  // Visual Ping Marker
  const marker = document.createElement("div");
  marker.className = "click-marker";
  marker.style.left = (ev.clientX - rect.left) + "px";
  marker.style.top = (ev.clientY - rect.top) + "px";
  videoWrapper.appendChild(marker);
  setTimeout(() => marker.remove(), 800);

  await post("/ptz/click", {
    x: clickX,
    y: clickY,
    frame_width: FRAME_W,
    frame_height: FRAME_H,
    sensitivity: 0.8
  });
});

// Snapshot Instant Button
document.getElementById("btn-snapshot-instant").addEventListener("click", () => {
  window.open("/snapshot", "_blank");
});

// Virtual Homing Button
const btnVirtualHome = document.getElementById("btn-virtual-home");
if (btnVirtualHome) {
  btnVirtualHome.addEventListener("click", async () => {
    btnVirtualHome.disabled = true;
    btnVirtualHome.textContent = "⌂ Đang chuẩn hoá...";
    try {
      await post("/ptz/virtual/home", {});
    } catch (e) {
      console.error("Home error:", e);
    } finally {
      btnVirtualHome.disabled = false;
      btnVirtualHome.textContent = "⌂ Calibrate Homing";
    }
  });
}

// ──────────────────────────────────────────────
// PTZ Speed Slider
// ──────────────────────────────────────────────
const speedRange = document.getElementById("speed-range");
const speedDisplay = document.getElementById("speed-display");

speedRange.addEventListener("input", async () => {
  currentSpeed = parseFloat(speedRange.value);
  speedDisplay.textContent = currentSpeed.toFixed(2);
  await post("/ptz/speed", { speed: currentSpeed });
});

// ──────────────────────────────────────────────
// D-pad & Continuous Movement
// ──────────────────────────────────────────────
let activeMoveDir = null;

function startMove(dir) {
  if (activeMoveDir === dir) return;
  activeMoveDir = dir;
  post("/ptz/move", { direction: dir, speed: currentSpeed });
  document.querySelectorAll(`.dpad-key[data-dir="${dir}"], .zoom-key[data-dir="${dir}"]`).forEach(el => {
    el.classList.add("active");
  });
}

function stopMove() {
  if (!activeMoveDir) return;
  activeMoveDir = null;
  post("/ptz/stop");
  document.querySelectorAll(".dpad-key.active, .zoom-key.active").forEach(el => {
    el.classList.remove("active");
  });
}

document.querySelectorAll(".dpad-key[data-dir], .zoom-key[data-dir]").forEach(btn => {
  btn.addEventListener("mousedown", () => startMove(btn.dataset.dir));
  btn.addEventListener("touchstart", (e) => { e.preventDefault(); startMove(btn.dataset.dir); }, { passive: false });
  btn.addEventListener("mouseup", stopMove);
  btn.addEventListener("mouseleave", stopMove);
  btn.addEventListener("touchend", stopMove);
  btn.addEventListener("touchcancel", stopMove);
});

document.getElementById("dpad-stop").addEventListener("click", stopMove);

// Keyboard Listeners
const KEY_MAP = {
  "w": "up", "a": "left", "s": "down", "d": "right",
  "ArrowUp": "up", "ArrowLeft": "left", "ArrowDown": "down", "ArrowRight": "right",
  "+": "zoom-in", "=": "zoom-in", "-": "zoom-out",
  "q": "up-left", "e": "up-right", "z": "down-left", "c": "down-right",
};

const pressedKeys = new Set();

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === " ") {
    ev.preventDefault();
    stopMove();
    return;
  }
  const dir = KEY_MAP[ev.key];
  if (dir && !pressedKeys.has(ev.key)) {
    pressedKeys.add(ev.key);
    startMove(dir);
  }
});

document.addEventListener("keyup", (ev) => {
  if (KEY_MAP[ev.key]) {
    pressedKeys.delete(ev.key);
    if (pressedKeys.size === 0) stopMove();
  }
});

// ──────────────────────────────────────────────
// Presets Management
// ──────────────────────────────────────────────
async function loadPresets() {
  const data = await get("/ptz/presets");
  if (!data || !data.presets) return;
  currentPresets = data.presets;

  // 1. Dock Preset List
  const dockList = document.getElementById("dock-preset-list");
  dockList.innerHTML = "";
  if (currentPresets.length === 0) {
    dockList.innerHTML = `<span style="font-size:12px;color:#8fa3bb;">Chưa có điểm nhớ nào.</span>`;
  } else {
    currentPresets.forEach(p => {
      const chip = document.createElement("div");
      chip.className = "preset-chip";
      chip.innerHTML = `<span>${p.name || p.token}</span><span class="chip-del" title="Xoá">&times;</span>`;
      chip.querySelector("span:first-child").addEventListener("click", async () => {
        await post("/ptz/presets/goto", { token: p.token, speed: 1.0 });
      });
      chip.querySelector(".chip-del").addEventListener("click", async (e) => {
        e.stopPropagation();
        if (confirm(`Xoá điểm nhớ "${p.name}"?`)) {
          await post("/ptz/presets/remove", { token: p.token });
          loadPresets();
        }
      });
      dockList.appendChild(chip);
    });
  }

  // 2. Patrol Checkbox List
  const patrolList = document.getElementById("patrol-checkbox-list");
  patrolList.innerHTML = "";
  if (currentPresets.length === 0) {
    patrolList.innerHTML = `<span style="font-size:13px;color:#8fa3bb;padding:8px;">Hãy lưu ít nhất 2 điểm nhớ để tuần tra.</span>`;
  } else {
    currentPresets.forEach(p => {
      const item = document.createElement("label");
      item.className = "patrol-item-check";
      item.innerHTML = `<input type="checkbox" value="${p.token}" checked> <span>${p.name || p.token}</span>`;
      patrolList.appendChild(item);
    });
  }

  // 3. Preset Manager Tab
  const managerItems = document.getElementById("preset-manager-items");
  managerItems.innerHTML = "";
  if (currentPresets.length === 0) {
    managerItems.innerHTML = `<p style="color:#71869e;font-size:13px;">Chưa có điểm nhớ được cấu hình.</p>`;
  } else {
    currentPresets.forEach((p, idx) => {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:#f8fafc;border:1px solid #e7edf4;border-radius:8px;";
      row.innerHTML = `
        <div>
          <strong style="font-size:14px;color:#1e293b;">#${idx + 1}. ${p.name || p.token}</strong>
          <small style="display:block;color:#64748b;font-size:12px;margin-top:2px;">Token: ${p.token}</small>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="primary-btn" style="padding:6px 12px;font-size:12px;" onclick="gotoPresetToken('${p.token}')">Đến vị trí</button>
          <button class="secondary-btn" style="padding:6px 10px;font-size:12px;color:#dc2626;" onclick="deletePresetToken('${p.token}', '${p.name}')">Xoá</button>
        </div>
      `;
      managerItems.appendChild(row);
    });
  }
}

window.gotoPresetToken = async function(token) {
  await post("/ptz/presets/goto", { token, speed: 1.0 });
};

window.deletePresetToken = async function(token, name) {
  if (confirm(`Xác nhận xoá điểm nhớ "${name}"?`)) {
    await post("/ptz/presets/remove", { token });
    loadPresets();
  }
};

document.getElementById("btn-save-preset").addEventListener("click", async () => {
  const nameInput = document.getElementById("input-preset-name");
  const name = nameInput.value.trim();
  if (!name) return;
  await post("/ptz/presets/save", { name });
  nameInput.value = "";
  loadPresets();
});

document.getElementById("btn-refresh-presets").addEventListener("click", loadPresets);

// ──────────────────────────────────────────────
// Patrol Tour Controls
// ──────────────────────────────────────────────
const btnStartPatrol = document.getElementById("btn-start-patrol");
const btnStopPatrol = document.getElementById("btn-stop-patrol");
const dwellInput = document.getElementById("patrol-dwell-input");

btnStartPatrol.addEventListener("click", async () => {
  const selectedCheckboxes = document.querySelectorAll("#patrol-checkbox-list input[type='checkbox']:checked");
  const tokens = Array.from(selectedCheckboxes).map(cb => cb.value);
  const dwell = parseFloat(dwellInput.value) || 3.0;

  if (tokens.length === 0) {
    alert("Vui lòng chọn ít nhất 1 điểm nhớ để tuần tra!");
    return;
  }

  const res = await post("/ptz/patrol/start", {
    preset_tokens: tokens,
    dwell_sec: dwell
  });

  if (res && res.ok) {
    isPatrolRunning = true;
    updatePatrolUI(true);
  }
});

btnStopPatrol.addEventListener("click", async () => {
  const res = await post("/ptz/patrol/stop");
  if (res && res.ok) {
    isPatrolRunning = false;
    updatePatrolUI(false);
  }
});

function updatePatrolUI(running) {
  btnStartPatrol.disabled = running;
  btnStopPatrol.disabled = !running;
}

// ──────────────────────────────────────────────
// Panorama Studio (AI Workflow)
// ──────────────────────────────────────────────
const btnStartPano = document.getElementById("btn-start-panorama");
const step1 = document.getElementById("step-1");
const step2 = document.getElementById("step-2");
const step3 = document.getElementById("step-3");
const panoFill = document.getElementById("pano-progress-bar");
const panoStatusMsg = document.getElementById("pano-status-msg");
const panoPctMsg = document.getElementById("pano-pct-msg");
const panoNotesText = document.getElementById("pano-notes-text");
const imgResultPano = document.getElementById("img-result-pano");
const imgResultGrid = document.getElementById("img-result-grid");
const hintEmptyPano = document.getElementById("hint-empty-pano");
const hintEmptyGrid = document.getElementById("hint-empty-grid");
const btnDownloadPano = document.getElementById("btn-download-pano");

let panoDoneHandled = false;

btnStartPano.addEventListener("click", async () => {
  btnStartPano.disabled = true;
  panoDoneHandled = false;
  panoStatusMsg.textContent = "Đang khởi tạo chu trình Panorama...";
  panoPctMsg.textContent = "0%";
  panoFill.style.width = "0%";
  panoNotesText.textContent = "";
  imgResultPano.style.display = "none";
  imgResultGrid.style.display = "none";
  hintEmptyPano.style.display = "block";
  hintEmptyGrid.style.display = "block";
  
  [step1, step2, step3].forEach(s => s.classList.remove("active"));
  step1.classList.add("active");

  const modeSelect = document.getElementById("select-pano-mode");
  const selectedMode = modeSelect ? modeSelect.value : "full_space";

  await post("/panorama/start", { mode: selectedMode });
});

function updatePanoramaProgress(data) {
  if (!data) return;

  if (data.running) {
    btnStartPano.disabled = true;
    panoDoneHandled = false;
    const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
    panoPctMsg.textContent = pct + "%";
    panoFill.style.width = pct + "%";

    if (data.total === 0) {
      step1.classList.add("active");
      step2.classList.remove("active");
      step3.classList.remove("active");
      panoStatusMsg.textContent = "Bước 1/3: AI Gemini 3.7 đang đo đạc FOV tại 2 biên...";
    } else {
      step1.classList.remove("active");
      step2.classList.add("active");
      step3.classList.remove("active");
      panoStatusMsg.textContent = `Bước 2/3: Đang quét & chụp frame ${data.progress}/${data.total} (${pct}%)...`;
    }
  } else if (data.done && !panoDoneHandled) {
    panoDoneHandled = true;
    btnStartPano.disabled = false;
    step1.classList.remove("active");
    step2.classList.remove("active");
    step3.classList.add("active");
    
    panoPctMsg.textContent = "100%";
    panoFill.style.width = "100%";
    panoStatusMsg.textContent = `Hoàn tất! Đã ghép thành công ${data.frame_count} frames.`;
    panoNotesText.textContent = data.calibration_note ? `Ghi chú AI: ${data.calibration_note}` : "";

    // Show Images only once
    imgResultPano.src = "/panorama/result/panorama?t=" + Date.now();
    imgResultPano.style.display = "block";
    hintEmptyPano.style.display = "none";
    btnDownloadPano.href = "/panorama/result/panorama";

    imgResultGrid.src = "/panorama/result/grid?t=" + Date.now();
    imgResultGrid.style.display = "block";
    hintEmptyGrid.style.display = "none";
    document.getElementById("pano-frames-count").textContent = `${data.frame_count} frames`;
  } else if (data.error) {
    btnStartPano.disabled = false;
    panoStatusMsg.textContent = `Lỗi: ${data.error}`;
    [step1, step2, step3].forEach(s => s.classList.remove("active"));
  }
}

// ──────────────────────────────────────────────
// Realtime WebSocket Telemetry Client
// ──────────────────────────────────────────────
let ws = null;
const wsDot = document.getElementById("ws-dot");
const wsStatusText = document.getElementById("ws-status-text");
const hudPan = document.getElementById("hud-pan");
const hudTilt = document.getElementById("hud-tilt");
const hudZoom = document.getElementById("hud-zoom");
const badgeStatus = document.getElementById("badge-status");
const badgeStatusText = document.getElementById("badge-status-text");

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    wsDot.classList.add("connected");
    wsStatusText.textContent = "Telemetry Live (WebSocket)";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      // PTZ Status
      if (data.ptz) {
        if (data.ptz.virtual_pan !== undefined) {
          hudPan.textContent = Number(data.ptz.virtual_pan).toFixed(3);
        } else if (data.ptz.pan !== undefined) {
          hudPan.textContent = Number(data.ptz.pan).toFixed(3);
        }

        if (data.ptz.virtual_tilt !== undefined) {
          hudTilt.textContent = Number(data.ptz.virtual_tilt).toFixed(3);
        } else if (data.ptz.tilt !== undefined) {
          hudTilt.textContent = Number(data.ptz.tilt).toFixed(3);
        }

        if (data.ptz.zoom !== undefined) hudZoom.textContent = Number(data.ptz.zoom).toFixed(1);

        const status = data.ptz.pan_tilt_status || "IDLE";
        if (status === "MOVING") {
          badgeStatus.className = "badge warn";
          badgeStatusText.textContent = "Camera đang quay";
        } else {
          badgeStatus.className = "badge good";
          badgeStatusText.textContent = data.ptz.is_homed ? "Virtual PTZ Đã chuẩn hoá" : "PTZ Sẵn sàng";
        }
      }

      // Patrol Status
      if (data.patrol) {
        updatePatrolUI(data.patrol.running);
      }

      // Panorama Progress
      if (data.panorama) {
        updatePanoramaProgress(data.panorama);
      }

    } catch (e) {
      console.error("WS Parse Error:", e);
    }
  };

  ws.onclose = () => {
    wsDot.classList.remove("connected");
    wsStatusText.textContent = "Mất kết nối telemetry. Đang thử lại...";
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

// ──────────────────────────────────────────────
// Camera Profile & Auto-Calibration
// ──────────────────────────────────────────────
async function loadCameraProfile() {
  const data = await get("/ptz/calibration/profile");
  if (!data) return;

  const profTitle = document.getElementById("prof-device-title");
  const profBadge = document.getElementById("prof-badge-status");
  const step1Desc = document.getElementById("pano-step1-desc");

  if (data.is_calibrated && data.profile) {
    const dev = data.profile.device_info || {};
    const pan = data.profile.pan || {};
    const tilt = data.profile.tilt || {};
    profTitle.textContent = `${dev.manufacturer || 'ONVIF'} ${dev.model || ''} (S/N: ${dev.serial_number || 'N/A'})`;
    profBadge.className = "badge good";
    profBadge.textContent = "Profile Sẵn sàng (0s overhead)";
    if (step1Desc) {
      step1Desc.textContent = `Đã nạp profile: Pan ${pan.total_pan_range_deg || 360}°, ${pan.optimal_pan_steps || 6} góc, Tilt ${tilt.optimal_horizon_tilt_val || -0.8}`;
    }
  } else {
    profTitle.textContent = data.camera_key ? `Thiết bị: ${data.camera_key} (Chưa hiệu chuẩn)` : "Chưa có profile";
    profBadge.className = "badge warn";
    profBadge.textContent = "Cần hiệu chuẩn 3 bước";
  }
}

const btnRecalibrate = document.getElementById("btn-recalibrate");
if (btnRecalibrate) {
  btnRecalibrate.addEventListener("click", async () => {
    if (confirm("Chạy lại toàn bộ chu trình hiệu chuẩn 3 bước tự động (Tilt, Horizon, Pan 360°)? Camera sẽ tự động xoay trong ~20 giây.")) {
      btnRecalibrate.disabled = true;
      btnRecalibrate.textContent = "⚙ Đang hiệu chuẩn...";
      try {
        const res = await post("/ptz/calibration/run?force=true");
        if (res && res.ok) {
          alert("Hiệu chuẩn thành công!");
          loadCameraProfile();
        }
      } catch (e) {
        alert("Lỗi hiệu chuẩn: " + e);
      } finally {
        btnRecalibrate.disabled = false;
        btnRecalibrate.textContent = "⚙ Hiệu chuẩn lại 3 bước";
      }
    }
  });
}

// ──────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────
loadPresets();
loadCameraProfile();
connectWebSocket();
