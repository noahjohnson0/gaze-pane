"""9-point fullscreen calibration via cv2. Press SPACE at each dot; ESC quits.

We use OpenCV for the UI because Homebrew's python@3.11 ships without tkinter,
and we already depend on cv2 for the webcam. Screen size comes from osascript
so we don't pull in pyobjc just for one query.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .gaze import GazeTracker, GazeFeatures, SmoothedTracker
from .mapper import GazeMapper, default_calibration_path


WIN_NAME = "gaze-pane calibration"


def make_grid(n: int) -> list[tuple[float, float]]:
    """An n x n grid of points in [0.05, 0.95] on each axis (inset from edges)."""
    if n < 2:
        raise ValueError("grid must be at least 2x2")
    if n == 1:
        return [(0.5, 0.5)]
    step_lo, step_hi = 0.05, 0.95
    coords = [step_lo + (step_hi - step_lo) * i / (n - 1) for i in range(n)]
    return [(x, y) for y in coords for x in coords]


# Test points for the validation phase: 4 corners + center. Hits all the regions
# where the linear model is most likely to err.
VALIDATION_POINTS = [
    (0.05, 0.05), (0.95, 0.05),
    (0.05, 0.95), (0.95, 0.95),
    (0.50, 0.50),
]


def _features_from_mean(mean: np.ndarray) -> GazeFeatures:
    return GazeFeatures(
        l_iris_x=float(mean[0]), l_iris_y=float(mean[1]),
        r_iris_x=float(mean[2]), r_iris_y=float(mean[3]),
        yaw=float(mean[4]), pitch=float(mean[5]),
        l_openness=float(mean[6]), r_openness=float(mean[7]),
        face_scale=float(mean[8]),
        timestamp=time.time(),
    )


def run_validation(tracker, mapper, *, points: list[tuple[float, float]] | None = None,
                   countdown_s: int = 3, capture_s: float = 4.0,
                   min_samples: int = 30,
                   ) -> tuple[str, list[tuple[float, float, GazeFeatures]],
                              list[float]] | None:
    """Passive-capture validation. Phases per point: COUNTDOWN -> CAPTURE -> RESULT.

    The capture phase shows only the target dot (no prediction overlay) so your
    gaze isn't pulled toward the green dot. The result phase briefly shows where
    the prediction landed. After all points, the user gets a summary and picks
    REFINE / KEEP / ABORT.

    Returns (decision, samples, errors). decision is "refine" | "keep" | "abort".
    None if interrupted before any meaningful response.
    """
    pts = points if points is not None else VALIDATION_POINTS
    sw, sh = get_screen_size()
    cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    new_samples: list[tuple[float, float, GazeFeatures]] = []
    errs: list[float] = []
    point_status: list[str] = []   # human-readable per-point summary

    def blank() -> np.ndarray:
        img = np.zeros((sh, sw, 3), dtype=np.uint8)
        for px, py in pts:
            cv2.circle(img, (int(px * sw), int(py * sh)), 5, (35, 35, 35), -1)
        return img

    def draw_target(img: np.ndarray, xy: tuple[int, int], pulse: float = 1.0,
                    color: tuple[int, int, int] = (255, 255, 255)) -> None:
        r = int(16 + 4 * pulse)
        cv2.circle(img, xy, r, color, -1)
        cv2.circle(img, xy, r + 8, (220, 220, 220), 2)
        cv2.line(img, (xy[0] - r - 14, xy[1]), (xy[0] - r - 4, xy[1]), (220, 220, 220), 1)
        cv2.line(img, (xy[0] + r + 4, xy[1]), (xy[0] + r + 14, xy[1]), (220, 220, 220), 1)
        cv2.line(img, (xy[0], xy[1] - r - 14), (xy[0], xy[1] - r - 4), (220, 220, 220), 1)
        cv2.line(img, (xy[0], xy[1] + r + 4), (xy[0], xy[1] + r + 14), (220, 220, 220), 1)

    def status_line(img: np.ndarray, text: str, subtext: str = "") -> None:
        cv2.putText(img, text, (40, sh - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2.LINE_AA)
        if subtext:
            cv2.putText(img, subtext, (40, sh - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    try:
        for idx, (tx_n, ty_n) in enumerate(pts):
            target = (int(tx_n * sw), int(ty_n * sh))

            # Phase 1: COUNTDOWN. White target dot, big countdown number.
            for s in range(countdown_s, 0, -1):
                t_end = time.time() + 1.0
                while time.time() < t_end:
                    img = blank()
                    pulse = 1.0 - (t_end - time.time())  # 0 -> 1 each second
                    draw_target(img, target, pulse=pulse)
                    cv2.putText(img, str(s),
                                (target[0] + 40, target[1] + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 0),
                                2, cv2.LINE_AA)
                    face = tracker.get_latest(max_age_s=0.5) is not None
                    face_text = "face: ok" if face else "face: NOT DETECTED"
                    status_line(img,
                                f"point {idx+1}/{len(pts)}  look at the dot",
                                f"capturing in {s}...   {face_text}   ESC to abort")
                    cv2.imshow(WIN_NAME, img)
                    if (cv2.waitKey(33) & 0xFF) == 27:
                        return ("abort", new_samples, errs)

            # Phase 2: CAPTURE. ONLY target, no prediction (prevents your eyes from chasing it).
            captured: list[GazeFeatures] = []
            face_seen = 0
            total_ticks = 0
            capture_end = time.time() + capture_s
            while time.time() < capture_end:
                f = tracker.get_latest(max_age_s=0.3)
                if f is not None:
                    captured.append(f)
                    face_seen += 1
                total_ticks += 1
                remaining = capture_end - time.time()
                img = blank()
                draw_target(img, target, pulse=(remaining % 0.6) / 0.6,
                            color=(0, 200, 255))   # orange-yellow during capture
                face_ratio = face_seen / max(total_ticks, 1)
                status_line(img,
                            f"capturing... {remaining:.1f}s   samples={len(captured)}",
                            f"face seen {int(face_ratio*100)}% of ticks   ESC to abort")
                cv2.imshow(WIN_NAME, img)
                if (cv2.waitKey(33) & 0xFF) == 27:
                    return ("abort", new_samples, errs)

            # Decide whether this point is usable.
            if len(captured) < min_samples:
                point_status.append(
                    f"  {idx+1}: ({tx_n:.2f},{ty_n:.2f}) SKIPPED (only {len(captured)} "
                    f"samples, need {min_samples})"
                )
                print(f"[{_ts()}] {point_status[-1].strip()}", flush=True)
                continue

            arr = np.array([s.as_vec() for s in captured])
            mean = arr.mean(axis=0)
            feat = _features_from_mean(mean)
            new_samples.append((tx_n, ty_n, feat))
            try:
                pnx, pny = mapper.predict(feat)
            except Exception:
                pnx = pny = -1.0
            err = math.hypot(pnx - tx_n, pny - ty_n)
            errs.append(err)
            point_status.append(
                f"  {idx+1}: ({tx_n:.2f},{ty_n:.2f}) -> predicted ({pnx:+.2f},{pny:+.2f}) "
                f"err={err:.3f}"
            )
            print(f"[{_ts()}]{point_status[-1]}", flush=True)

            # Phase 3: RESULT. Show prediction landing for ~1s.
            result_end = time.time() + 1.0
            pred_xy = (int(max(0, min(sw - 1, pnx * sw))),
                       int(max(0, min(sh - 1, pny * sh))))
            while time.time() < result_end:
                img = blank()
                draw_target(img, target, pulse=0.5, color=(255, 255, 255))
                cv2.line(img, target, pred_xy, (140, 140, 140), 1)
                # prediction dot
                cv2.circle(img, pred_xy, 14, (0, 200, 100), -1)
                cv2.circle(img, pred_xy, 14, (255, 255, 255), 2)
                status_line(img,
                            f"point {idx+1}/{len(pts)} done  error={err:.3f}",
                            "next point shortly...   ESC to abort")
                cv2.imshow(WIN_NAME, img)
                if (cv2.waitKey(33) & 0xFF) == 27:
                    return ("abort", new_samples, errs)

        # Summary screen + decision.
        return _validation_decision(blank, status_line, point_status, errs, new_samples)
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def _validation_decision(blank_fn, status_line_fn,
                         point_status: list[str],
                         errs: list[float],
                         new_samples) -> tuple[str, list, list[float]]:
    """Show per-point error summary, wait for SPACE/ENTER/ESC."""
    while True:
        img = blank_fn()
        sh = img.shape[0]
        cv2.putText(img, "validation results",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (220, 220, 220), 2, cv2.LINE_AA)
        y = 110
        for line in point_status:
            cv2.putText(img, line.strip(), (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 180, 180), 1, cv2.LINE_AA)
            y += 26
        if errs:
            summary = (f"max error {max(errs):.3f}   mean {sum(errs)/len(errs):.3f}   "
                       f"samples added: {len(new_samples)}")
            cv2.putText(img, summary, (40, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 80), 1, cv2.LINE_AA)
        status_line_fn(img,
                       "SPACE = use refined fit   ENTER = keep initial   ESC = abort",
                       "")
        cv2.imshow(WIN_NAME, img)
        key = cv2.waitKey(33) & 0xFF
        if key == 32:        # SPACE
            return ("refine", new_samples, errs)
        if key in (13, 10):  # ENTER / Return
            return ("keep", new_samples, errs)
        if key == 27:        # ESC
            return ("abort", new_samples, errs)


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


def run_calibration(tracker, *, grid: list[tuple[float, float]],
                    samples_per_point: int = 12) -> list:
    sw, sh = get_screen_size()
    cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WIN_NAME, 0, 0)

    samples: list[tuple[float, float, GazeFeatures]] = []
    try:
        for idx, (nx, ny) in enumerate(grid):
            dot_xy = (nx * sw, ny * sh)
            captured: list[GazeFeatures] = []
            state = "wait"
            print(f"[{_ts()}] point {idx + 1}/{len(grid)}: look at "
                  f"({nx:.2f},{ny:.2f})", flush=True)
            while True:
                if state == "capture":
                    color = (0, 200, 255)  # BGR -> orange-yellow
                    status = f"capturing... {len(captured)}/{samples_per_point}"
                else:
                    color = (255, 255, 255)
                    status = "look at the dot, press SPACE"
                counter_text = f"{idx + 1} / {len(grid)}"
                cv2.imshow(WIN_NAME, _render(sw, sh, dot_xy, color, status, counter_text))
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print(f"[{_ts()}] aborted", flush=True)
                    return []
                if key == 32 and state == "wait":  # SPACE
                    if tracker.get_latest(max_age_s=0.5) is None:
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
                        samples.append((nx, ny, _features_from_mean(mean)))
                        print(f"[{_ts()}] captured @ ({nx:.2f},{ny:.2f}) "
                              f"feat={mean.round(3).tolist()}", flush=True)
                        break
    finally:
        cv2.destroyAllWindows()
        # On macOS one extra waitKey tick is needed to actually tear down the window.
        cv2.waitKey(1)
    return samples


def _archive_existing(out: Path) -> Path | None:
    """Move the existing calibration into history/ keyed by its finished_at stamp."""
    if not out.exists():
        return None
    history = out.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    try:
        prev = json.loads(out.read_text())
        finished = (prev.get("metadata") or {}).get("finished_at")
    except Exception:
        finished = None
    if finished:
        stamp = finished.replace(":", "-")
    else:
        stamp = datetime.fromtimestamp(out.stat().st_mtime).isoformat(timespec="seconds").replace(":", "-")
    archive = history / f"{stamp}.json"
    # Don't clobber an existing archive with the same stamp.
    i = 1
    while archive.exists():
        archive = history / f"{stamp}-{i}.json"
        i += 1
    shutil.move(str(out), str(archive))
    return archive


def cmd_calibrate(args) -> int:
    started_at = datetime.now().isoformat(timespec="seconds")
    start_wall = time.time()

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

    grid = make_grid(args.grid)
    print(f"[{_ts()}] {args.grid}x{args.grid} grid ({len(grid)} points), "
          f"{args.samples} frames per point", flush=True)

    try:
        samples = run_calibration(tracker, grid=grid, samples_per_point=args.samples)

        if len(samples) < len(grid):
            print(f"[{_ts()}] only got {len(samples)}/{len(grid)} samples; "
                  "not saving.", file=sys.stderr)
            return 1

        # Initial fit.
        mapper = GazeMapper()
        mapper.fit(samples)
        rx0, ry0 = mapper.residuals or (0.0, 0.0)
        print(f"[{_ts()}] initial fit RMS: x={rx0:.3f} y={ry0:.3f}", flush=True)

        # Validation + refinement.
        decision = "keep"
        val_samples: list = []
        if not args.skip_validate:
            print(f"[{_ts()}] validation phase: 5 points, 4s passive each",
                  flush=True)
            result = run_validation(tracker, mapper)
            if result is not None:
                decision, val_samples, _errs = result
            if decision == "abort":
                print(f"[{_ts()}] aborted during validation; nothing saved.",
                      file=sys.stderr)
                return 1
            if decision == "refine" and val_samples:
                mapper.fit(samples + val_samples)
                rx1, ry1 = mapper.residuals or (0.0, 0.0)
                print(f"[{_ts()}] refined fit RMS: x={rx1:.3f} y={ry1:.3f}  "
                      f"(was x={rx0:.3f} y={ry0:.3f})", flush=True)
            else:
                print(f"[{_ts()}] keeping initial fit (no refinement applied)",
                      flush=True)
    finally:
        base.stop()

    finished_at = datetime.now().isoformat(timespec="seconds")
    duration = time.time() - start_wall

    mapper.metadata = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration, 2),
        "grid": int(args.grid),
        "n_points": len(grid),
        "samples_per_point": int(args.samples),
        "camera": int(args.camera),
        "validation_points": len(val_samples) if val_samples else 0,
        "validation_decision": decision if not args.skip_validate else "skipped",
    }

    out = default_calibration_path()
    archived = _archive_existing(out)
    if archived is not None:
        print(f"[{_ts()}] archived previous calibration -> {archived}", flush=True)
    mapper.save(out)

    rx, ry = mapper.residuals or (0.0, 0.0)
    print(f"[{_ts()}] calibration saved -> {out}", flush=True)
    print(f"[{_ts()}] duration: {duration:.1f}s  "
          f"final RMS residual: x={rx:.3f} y={ry:.3f}  "
          f"(~0.05 good, >0.15 redo)", flush=True)
    return 0
