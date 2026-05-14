"""9-point fullscreen calibration via cv2. Press SPACE at each dot; ESC quits.

We use OpenCV for the UI because Homebrew's python@3.11 ships without tkinter,
and we already depend on cv2 for the webcam. Screen size comes from osascript
so we don't pull in pyobjc just for one query.
"""
from __future__ import annotations

import subprocess
import sys
import time

import cv2
import numpy as np

from .gaze import GazeTracker, GazeFeatures, SmoothedTracker
from .mapper import GazeMapper, default_calibration_path


GRID = [
    (0.05, 0.05), (0.50, 0.05), (0.95, 0.05),
    (0.05, 0.50), (0.50, 0.50), (0.95, 0.50),
    (0.05, 0.95), (0.50, 0.95), (0.95, 0.95),
]

WIN_NAME = "gaze-pane calibration"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def get_screen_size() -> tuple[int, int]:
    """Width, height of the main display in points (via AppleScript)."""
    try:
        out = subprocess.run(
            ["osascript", "-e",
             'tell application "Finder" to get bounds of window of desktop'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            return int(parts[2]), int(parts[3])
    except Exception:
        pass
    return 1920, 1080


def _render(sw: int, sh: int, dot_xy: tuple[float, float],
            dot_color: tuple[int, int, int],
            status: str, counter: str) -> np.ndarray:
    img = np.zeros((sh, sw, 3), dtype=np.uint8)
    cv2.circle(img, (int(dot_xy[0]), int(dot_xy[1])), 14, dot_color, -1)
    cv2.putText(img, counter, (sw // 2 - 40, sh // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, status, (40, sh - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "SPACE = capture   ESC = quit", (40, sh - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def run_calibration(tracker, samples_per_point: int = 12) -> list:
    sw, sh = get_screen_size()
    cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WIN_NAME, 0, 0)

    samples: list[tuple[float, float, GazeFeatures]] = []
    try:
        for idx, (nx, ny) in enumerate(GRID):
            dot_xy = (nx * sw, ny * sh)
            captured: list[GazeFeatures] = []
            state = "wait"
            print(f"[{_ts()}] point {idx + 1}/{len(GRID)}: look at "
                  f"({nx:.2f},{ny:.2f})", flush=True)
            while True:
                if state == "capture":
                    color = (0, 200, 255)  # BGR -> orange-yellow
                    status = f"capturing... {len(captured)}/{samples_per_point}"
                else:
                    color = (255, 255, 255)
                    status = "look at the dot, press SPACE"
                counter_text = f"{idx + 1} / {len(GRID)}"
                cv2.imshow(WIN_NAME, _render(sw, sh, dot_xy, color, status, counter_text))
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print(f"[{_ts()}] aborted", flush=True)
                    return []
                if key == 32 and state == "wait":  # SPACE
                    if tracker.get_latest(max_age_s=0.5) is None:
                        # Flash red so the user knows.
                        for _ in range(8):
                            cv2.imshow(WIN_NAME, _render(
                                sw, sh, dot_xy, (0, 0, 200),
                                "no face detected", counter_text))
                            cv2.waitKey(60)
                        continue
                    state = "capture"
                    captured = []
                if state == "capture":
                    f = tracker.get_latest(max_age_s=0.3)
                    if f is not None:
                        captured.append(f)
                    if len(captured) >= samples_per_point:
                        arr = np.array([s.as_vec() for s in captured])
                        mean = arr.mean(axis=0)
                        samples.append((nx, ny, GazeFeatures(
                            iris_x=float(mean[0]), iris_y=float(mean[1]),
                            yaw=float(mean[2]), pitch=float(mean[3]),
                            timestamp=time.time(),
                        )))
                        print(f"[{_ts()}] captured @ ({nx:.2f},{ny:.2f}) "
                              f"feat={mean.round(3).tolist()}", flush=True)
                        break
    finally:
        cv2.destroyAllWindows()
        # On macOS one extra waitKey tick is needed to actually tear down the window.
        cv2.waitKey(1)
    return samples


def cmd_calibrate(args) -> int:
    print(f"[{_ts()}] starting webcam + face landmarker...", flush=True)
    base = GazeTracker(camera_index=args.camera)
    base.start()
    tracker = SmoothedTracker(base, alpha=0.6)

    # Wait briefly for the first face detection.
    for _ in range(40):
        if tracker.get_latest(max_age_s=0.3) is not None:
            break
        time.sleep(0.05)
    if tracker.get_latest(max_age_s=0.5) is None:
        print(f"[{_ts()}] no face detected yet. Continuing — center yourself "
              "in front of the camera; press SPACE only after the dot is white "
              "and a face is visible.", flush=True)

    try:
        samples = run_calibration(tracker, samples_per_point=args.samples)
    finally:
        base.stop()

    if len(samples) < len(GRID):
        print(f"[{_ts()}] only got {len(samples)}/{len(GRID)} samples; not saving.",
              file=sys.stderr)
        return 1

    mapper = GazeMapper()
    mapper.fit(samples)
    out = default_calibration_path()
    mapper.save(out)
    rx, ry = mapper.residuals or (0.0, 0.0)
    print(f"[{_ts()}] calibration saved -> {out}", flush=True)
    print(f"[{_ts()}] RMS residual: x={rx:.3f} y={ry:.3f}  "
          f"(~0.05 good, >0.15 redo)", flush=True)
    return 0
