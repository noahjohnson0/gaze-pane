"""`gaze-pane cameras`: enumerate webcams so you can pick the right --camera.

Two views, side by side:
  * AVFoundation device names (what macOS thinks is attached — "FaceTime HD
    Camera", "iPhone Camera (Continuity)", ...). Order here matches the
    order cv2's AVFoundation backend uses, so AV device #N usually opens
    at `--camera N`.
  * cv2 probe: which indices actually open, frame shape, mean brightness,
    and a JPEG snapshot saved under /tmp/ so you can eyeball which lens
    is which.
"""
from __future__ import annotations

import os
import time


def _list_avfoundation_devices() -> list[str]:
    try:
        from AVFoundation import (
            AVCaptureDevice,
            AVMediaTypeVideo,
        )
    except ImportError:
        return []
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or []
    return [str(d.localizedName()) for d in devices]


def cmd_cameras(_args) -> int:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import cv2

    names = _list_avfoundation_devices()
    if names:
        print("AVFoundation video devices (macOS, in enumeration order):")
        for i, name in enumerate(names):
            print(f"  [{i}] {name}")
    else:
        print("(AVFoundation device list unavailable; falling back to cv2 probe)")

    print("\ncv2 probe (indices 0..4):")
    any_ok = False
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"  [{idx}] (not available)")
            continue
        # Warm-up: some cameras hand out a black frame on the first read.
        for _ in range(3):
            cap.read()
            time.sleep(0.03)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  [{idx}] opened but read failed")
            cap.release()
            continue
        any_ok = True
        h, w = frame.shape[:2]
        brightness = float(frame.mean())
        snap = f"/tmp/gaze-pane-cam{idx}.jpg"
        cv2.imwrite(snap, frame)
        name = names[idx] if idx < len(names) else "?"
        print(f"  [{idx}] {w}x{h}  brightness={brightness:.1f}/255  "
              f"({name})  snapshot: {snap}")
        cap.release()

    if not any_ok:
        print("\nNo working cameras found. On macOS, grant Camera permission "
              "to your terminal in System Settings -> Privacy & Security.")
        return 1
    print("\nPick the index whose snapshot shows your face, then:")
    print("  gaze-pane run --camera <index> --overlay --webcam-overlay")
    return 0
