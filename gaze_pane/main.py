"""Runtime: webcam in a thread, iTerm2 API in asyncio, dwell-based activation."""
from __future__ import annotations

import asyncio
import sys
import time

import iterm2

from .calibrate import get_screen_size
from .gaze import GazeTracker, SmoothedTracker
from .iterm import activate_session, get_pane_rects
from .mapper import GazeMapper, default_calibration_path


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def cmd_run(args) -> int:
    cal_path = default_calibration_path()
    if not cal_path.exists():
        print(f"[{_ts()}] no calibration at {cal_path}", file=sys.stderr)
        print("        run:  python -m gaze_pane calibrate", file=sys.stderr)
        return 1
    mapper = GazeMapper()
    mapper.load(cal_path)
    print(f"[{_ts()}] loaded calibration ({cal_path})", flush=True)
    rx, ry = mapper.residuals or (0.0, 0.0)
    print(f"[{_ts()}] calibration RMS residual: x={rx:.3f} y={ry:.3f}", flush=True)

    screen_w, screen_h = (float(x) for x in get_screen_size())
    print(f"[{_ts()}] main display: {int(screen_w)}x{int(screen_h)} pts", flush=True)

    base = GazeTracker(camera_index=args.camera)
    base.start()
    print(f"[{_ts()}] webcam + face landmarker running", flush=True)
    tracker = SmoothedTracker(base, alpha=args.alpha)

    async def loop(connection):
        last_activated: str | None = None
        candidate: str | None = None
        candidate_since: float = 0.0
        pane_rects: list = []
        active_session: str | None = None
        next_pane_refresh: float = 0.0
        pane_refresh_period = max(0.25, args.pane_refresh)
        tick_period = 1.0 / max(args.hz, 1.0)
        print(f"[{_ts()}] running. dwell={args.dwell_ms}ms  hz={args.hz}  "
              f"alpha={args.alpha}  ctrl-c to quit.", flush=True)

        debug_print_every = max(1, args.hz // 4) if args.debug else 0
        debug_counter = 0

        while True:
            now = time.monotonic()

            if now >= next_pane_refresh:
                try:
                    pane_rects, active_session = await get_pane_rects(
                        connection, screen_w, screen_h,
                        chrome_top=args.chrome_top)
                except Exception as e:
                    print(f"[{_ts()}] pane query failed: {e!r}", file=sys.stderr)
                next_pane_refresh = now + pane_refresh_period
                # If iTerm focus moved on its own, sync our notion of "active".
                if active_session is not None:
                    last_activated = active_session

            if not pane_rects:
                await asyncio.sleep(tick_period)
                continue

            f = tracker.get_latest(max_age_s=0.5)
            if f is None:
                await asyncio.sleep(tick_period)
                continue

            nx, ny = mapper.predict(f)
            hit = next((r for r in pane_rects if r.contains(nx, ny)), None)

            if args.debug:
                debug_counter += 1
                if debug_counter % debug_print_every == 0:
                    where = hit.session_id[:8] if hit else "—"
                    print(f"[{_ts()}] gaze=({nx:+.2f},{ny:+.2f}) "
                          f"yaw={f.yaw:+.2f} pitch={f.pitch:+.2f} "
                          f"iris=({f.iris_x:+.2f},{f.iris_y:+.2f}) -> {where}",
                          flush=True)

            if hit is None:
                await asyncio.sleep(tick_period)
                continue

            if hit.session_id == last_activated:
                # Already focused there; keep candidate aligned so we don't bounce.
                candidate = hit.session_id
                candidate_since = now
                await asyncio.sleep(tick_period)
                continue

            if hit.session_id != candidate:
                candidate = hit.session_id
                candidate_since = now
            elif (now - candidate_since) * 1000.0 >= args.dwell_ms:
                try:
                    await activate_session(hit.session)
                    last_activated = hit.session_id
                    print(f"[{_ts()}] -> activated {hit.session_id[:8]} "
                          f"(gaze {nx:+.2f},{ny:+.2f})", flush=True)
                except Exception as e:
                    print(f"[{_ts()}] activate failed: {e!r}", file=sys.stderr)
                # Force a re-query of "active" on next refresh so our state stays honest.
                next_pane_refresh = now

            await asyncio.sleep(tick_period)

    try:
        iterm2.run_until_complete(loop, retry=False)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] stopping...", flush=True)
    except ConnectionRefusedError:
        print(f"[{_ts()}] could not connect to iTerm2's Python API.", file=sys.stderr)
        print("        Enable it: iTerm2 > Settings > General > Magic > "
              "Enable Python API", file=sys.stderr)
        return 2
    finally:
        base.stop()
    return 0
