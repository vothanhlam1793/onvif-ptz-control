# ONVIF PTZ Operator Console

Start the interactive console:

```bash
python tools/onvif/onvif-cli.py
```

Select a saved camera, then choose `C` to connect. The profile supplies host,
port, username, and password; the session starts without credential prompts.
`config.json` is saved with file permission `0600` and is ignored by Git.

Existing profiles without a password are migrated on first use from
`CAMERA_PASS` in `.env`, or by entering the password once.

After connecting, the console opens live keypad control. Type one key and press
Enter for each movement, so holding a key cannot repeat PTZ commands:

```text
       [7] [8] [9]
       [4] [5] [6]
       [1] [2] [3]

+/- zoom · 5 or x stop · s snapshot · m actions · q disconnect
```

The console confirms the received key, direction, and ONVIF PanTilt vector
before every move. Press `m` without ending the session for speed/step settings, a one-step move,
snapshot save, motion check, or device details. Snapshots are saved under
`outputs/ptz_snapshots/`.

Device dashboard options are `C` Connect & control, `D` Diagnose, `E` Edit,
and `X` Delete. Deletion requires typing `XOA`.

Run a non-moving diagnostic from a shell:

```bash
python tools/onvif/onvif-cli.py probe
```

`probe` uses the selected saved camera when one exists. Otherwise it falls back
to `CAMERA_HOST`, `CAMERA_PORT`, `CAMERA_USER`, and `CAMERA_PASS` in `.env`.

## Calibrate a new physical camera

Each device gets a profile keyed by manufacturer, model, and serial number (or
MAC address when serial is unavailable). Profiles are never shared silently by
two cameras of the same model.

Print `outputs/checkerboard-10x7-a4.pdf` at **Actual size / 100%**, place the
whole board in view, then run:

```bash
python -m tools.onvif.calibrate_device --capture-checkerboard
```

The wizard performs these required stages:

1. ONVIF discovery and physical device identification.
2. Checkerboard intrinsic and distortion calibration.
3. Four-direction `ContinuousMove`/`Stop` validation.
4. Minimum pulse, post-Stop drift, and surveillance tolerance measurement.
5. Atomic profile save only after every required stage passes.

To reuse checkerboard images captured separately:

```bash
python -m tools.onvif.calibrate_device \
  --checkerboard-images /path/to/checkerboard/views
```

The generated profile enables relative-angle control. Mechanical four-endstop
calibration remains optional and is required only for absolute room coordinates.
