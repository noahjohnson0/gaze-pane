"""Transparent always-on-top dot showing where gaze-pane thinks you're looking.

We use AppKit (PyObjC) to create a borderless, click-through NSWindow at the
main screen size and host an NSView that draws a translucent circle at the
gaze position.

This module must drive the main thread because AppKit's runloop owns it. The
rest of gaze-pane (webcam thread + asyncio iTerm2 loop) runs on background
threads while the overlay is on.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowCollectionBehaviorTransient,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSMakeRect, NSObject, NSTimer


class _GazeView(NSView):
    """NSView that draws a filled, outlined circle at (gaze_x, gaze_y) in points."""

    def initWithFrame_(self, frame):
        self = objc.super(_GazeView, self).initWithFrame_(frame)
        if self is None:
            return None
        # AppKit uses bottom-left origin. We store gaze in the same coords.
        self._x = -1.0
        self._y = -1.0
        self._radius = 14.0
        self._hit = True   # whether the gaze landed in a known pane (green) or not (red)
        self._frame_size = (frame.size.width, frame.size.height)
        # Display-level EMA on the rendered position. Independent of the
        # feature-level smoothing used for pane hit-testing, so the dot looks
        # calm without making pane-switch decisions laggier.
        self._smooth_alpha = 0.2
        self._smooth_x = -1.0
        self._smooth_y = -1.0
        return self

    def isOpaque(self):
        return False

    def setSmoothAlpha_(self, alpha: float):
        a = float(alpha)
        if a < 0.01:
            a = 0.01
        if a > 1.0:
            a = 1.0
        self._smooth_alpha = a

    def setX_y_hit_(self, x: float, y: float, hit: bool):
        if x < 0 or self._smooth_x < 0:
            # No prior position (or hidden); snap, don't interpolate.
            self._smooth_x = x
            self._smooth_y = y
        else:
            a = self._smooth_alpha
            self._smooth_x = a * x + (1.0 - a) * self._smooth_x
            self._smooth_y = a * y + (1.0 - a) * self._smooth_y
        self._x = self._smooth_x
        self._y = self._smooth_y
        self._hit = bool(hit)
        self.setNeedsDisplay_(True)

    def drawRect_(self, _rect):
        # Always draw a small status indicator in the bottom-right so the
        # overlay is visible even before we have gaze data. Helps confirm the
        # window is actually on top of fullscreen iTerm2.
        sw, sh = self._frame_size
        ind_rect = NSMakeRect(sw - 26, 18, 8, 8)
        ind = NSBezierPath.bezierPathWithOvalInRect_(ind_rect)
        if self._x < 0:
            # No gaze yet — pulse a faint amber dot.
            NSColor.colorWithRed_green_blue_alpha_(1.0, 0.8, 0.2, 0.6).setFill()
        else:
            NSColor.colorWithRed_green_blue_alpha_(0.3, 1.0, 0.5, 0.6).setFill()
        ind.fill()

        if self._x < 0:
            return
        r = self._radius
        rect = NSMakeRect(self._x - r, self._y - r, 2 * r, 2 * r)
        path = NSBezierPath.bezierPathWithOvalInRect_(rect)
        if self._hit:
            fill = NSColor.colorWithRed_green_blue_alpha_(0.30, 0.95, 0.45, 0.45)
            edge = NSColor.colorWithRed_green_blue_alpha_(1.00, 1.00, 1.00, 0.95)
        else:
            fill = NSColor.colorWithRed_green_blue_alpha_(0.95, 0.30, 0.30, 0.40)
            edge = NSColor.colorWithRed_green_blue_alpha_(1.00, 1.00, 1.00, 0.80)
        fill.setFill()
        path.fill()
        edge.setStroke()
        path.setLineWidth_(2.0)
        path.stroke()


class _Ticker(NSObject):
    """Lightweight NSObject wrapper so NSTimer can call us with a selector."""

    def initWithCallback_(self, cb):
        self = objc.super(_Ticker, self).init()
        if self is None:
            return None
        self._cb = cb
        return self

    def tick_(self, _timer):
        try:
            self._cb()
        except Exception as e:  # don't kill the runloop on a bad tick
            import sys
            print(f"[overlay] tick error: {e!r}", file=sys.stderr)


class Overlay:
    """Runs an AppKit runloop with a translucent gaze indicator on top.

    `get_gaze` is called every tick. It must return either:
      - None if no gaze (hides the dot), or
      - (norm_x, norm_y, in_pane: bool) with coords in top-left normalized space.
    `stop_event` lets the asyncio thread signal a graceful shutdown.
    """

    def __init__(
        self,
        get_gaze: Callable[[], Optional[tuple[float, float, bool]]],
        stop_event: threading.Event,
        fps: float = 30.0,
        smoothing_alpha: float = 0.2,
    ):
        self._get_gaze = get_gaze
        self._stop_event = stop_event
        self._fps = max(5.0, fps)
        self._smoothing_alpha = float(smoothing_alpha)

    def run(self) -> None:
        # Make Ctrl-C exit even though the main thread is inside AppKit.run().
        # Python signal handlers only fire at bytecode boundaries; the NSTimer
        # below provides those, and the handler asks AppKit to stop cleanly.
        import signal

        def _on_sigint(_signum, _frame):
            self._stop_event.set()
            NSApplication.sharedApplication().stop_(None)

        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)

        app = NSApplication.sharedApplication()
        # No Dock icon, doesn't steal focus.
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        screen = NSScreen.mainScreen()
        frame = screen.frame()
        screen_w = float(frame.size.width)
        screen_h = float(frame.size.height)

        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setLevel_(NSScreenSaverWindowLevel + 1)
        window.setIgnoresMouseEvents_(True)
        window.setHasShadow_(False)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorTransient
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        view = _GazeView.alloc().initWithFrame_(frame)
        view.setSmoothAlpha_(self._smoothing_alpha)
        window.setContentView_(view)
        # orderFrontRegardless_ forces this window in front even when iTerm2 in
        # fullscreen owns the active Space. Plain orderFront_ won't cross over.
        window.orderFrontRegardless()

        def on_tick():
            if self._stop_event.is_set():
                NSApplication.sharedApplication().stop_(None)
                return
            data = self._get_gaze()
            if data is None:
                view.setX_y_hit_(-1.0, -1.0, True)
                return
            nx, ny, hit = data
            # Convert top-left normalized -> Cocoa bottom-left points on the main screen.
            x = nx * screen_w
            y = screen_h - ny * screen_h
            view.setX_y_hit_(x, y, hit)

        ticker = _Ticker.alloc().initWithCallback_(on_tick)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / self._fps, ticker, "tick:", None, True
        )
        # `stop_` only takes effect after the current event; nudge the loop with
        # a wakeup timer so shutdown isn't laggy.
        app.run()
