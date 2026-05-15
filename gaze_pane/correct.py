"""Quick mid-session correction.

User presses a global hotkey while looking at their mouse cursor. We read the
real cursor position from AppKit and the current gaze prediction from the
mapper, compute `bias = mouse - predicted`, and apply that bias additively to
every subsequent prediction until the next correction (or until the runner
restarts).

This is translation-only correction -- can't recover from scale / rotation
drift -- but it's cheap, robust, and exactly the right tool for "calibration
was fine but my head shifted" cases.

Needs macOS Accessibility permission for global key capture. First use will
trigger the system prompt; grant it to whichever terminal launched gaze-pane
and re-run.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _mouse_xy_norm() -> Optional[tuple[float, float]]:
    """Top-left-normalized cursor position on the main display (0..1)."""
    try:
        from AppKit import NSEvent, NSScreen
        p = NSEvent.mouseLocation()
        f = NSScreen.mainScreen().frame()
        sw = float(f.size.width)
        sh = float(f.size.height)
        nx = float(p.x) / sw
        ny = (sh - float(p.y)) / sh  # Quartz bottom-left -> top-left
        return nx, ny
    except Exception:
        return None


class CorrectionListener:
    """Global hotkey -> compute and apply a bias correction.

    `tracker.get_latest()` and `mapper.predict()` are called when the hotkey
    fires, so we capture the *current* gaze at the moment of the chord and
    compare it against the *current* mouse position.
    """

    def __init__(
        self,
        *,
        hotkey: str,
        tracker,
        mapper,
        on_apply: Callable[[float, float, tuple[float, float]], None],
    ) -> None:
        self.hotkey = hotkey
        self.tracker = tracker
        self.mapper = mapper
        self.on_apply = on_apply
        self._listener = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise RuntimeError(
                "pynput not installed; install with `pip install pynput`"
            ) from e
        self._listener = keyboard.GlobalHotKeys({self.hotkey: self._on_chord})
        self._listener.start()
        print(f"[{_ts()}] [correct] global hotkey {self.hotkey!r} active "
              "(if it doesn't fire, grant Accessibility permission to your "
              "terminal in System Settings -> Privacy & Security -> "
              "Accessibility, then restart).", flush=True)

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_chord(self) -> None:
        # Acquire lock so two rapid presses don't race.
        if not self._lock.acquire(blocking=False):
            return
        try:
            f = self.tracker.get_latest(max_age_s=0.5)
            if f is None:
                print(f"[{_ts()}] [correct] no gaze data; ignoring",
                      flush=True)
                return
            m = _mouse_xy_norm()
            if m is None:
                print(f"[{_ts()}] [correct] couldn't read mouse; ignoring",
                      flush=True)
                return
            mx, my = m
            try:
                px, py = self.mapper.predict(f)
            except Exception as e:
                print(f"[{_ts()}] [correct] mapper.predict failed: {e!r}",
                      flush=True)
                return
            bias_x = mx - px
            bias_y = my - py
            self.on_apply(bias_x, bias_y, (mx, my))
        finally:
            self._lock.release()
