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
