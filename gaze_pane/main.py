"""Runtime: webcam in a thread, iTerm2 API in asyncio, dwell-based activation.

If `--overlay` is set, the asyncio loop is moved to a background thread so the
main thread can run AppKit for a translucent gaze indicator (see overlay.py).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Callable

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
    tracker = SmoothedTracker(
        base,
        alpha=args.alpha,
        mode=args.smoothing,
        window=args.median_window,
    )

    voice = None
    if args.voice:
        from .voice import VoiceListener
        voice = VoiceListener(
            wake_phrase=args.wake_phrase,
            end_phrase=args.end_phrase,
            model=args.voice_model,
            device=args.voice_device,
        )
        try:
            voice.start()
        except Exception as e:
            print(f"[{_ts()}] voice listener failed to start: {e!r}",
                  file=sys.stderr)
            voice = None

    # When --overlay is set, the asyncio loop publishes the latest gaze + hit
    # to this shared state; the AppKit thread reads it to drive the dot.
    gaze_state = {"nx": -1.0, "ny": -1.0, "hit": False, "ts": 0.0}
    gaze_lock = threading.Lock()

    def publish_gaze(nx: float, ny: float, hit: bool) -> None:
        if not args.overlay:
            return
        with gaze_lock:
            gaze_state["nx"] = nx
            gaze_state["ny"] = ny
            gaze_state["hit"] = hit
            gaze_state["ts"] = time.time()

    async def loop(connection):
        last_activated: str | None = None
        candidate: str | None = None
        candidate_since: float = 0.0
        pane_rects: list = []
        active_session: str | None = None
        next_pane_refresh: float = 0.0
        pane_refresh_period = max(0.25, args.pane_refresh)
        tick_period = 1.0 / max(args.hz, 1.0)
        prev_pane_ids: tuple[str, ...] | None = None
        last_status_print = 0.0
        status_interval = max(0.5, args.status_every)
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
                cur_ids = tuple(r.session_id for r in pane_rects)
                if cur_ids != prev_pane_ids:
                    print(f"[{_ts()}] {len(pane_rects)} pane(s):", flush=True)
                    for r in pane_rects:
                        mark = " (active)" if r.session_id == active_session else ""
                        print(f"        {r.session_id[:8]}  "
                              f"x=[{r.norm_left:.2f},{r.norm_right:.2f}]  "
                              f"y=[{r.norm_top:.2f},{r.norm_bottom:.2f}]{mark}",
                              flush=True)
                    prev_pane_ids = cur_ids

            if not pane_rects:
                await asyncio.sleep(tick_period)
                continue

            f = tracker.get_latest(max_age_s=0.5)
            if f is None:
                await asyncio.sleep(tick_period)
                continue

            nx, ny = mapper.predict(f)
            hit = next((r for r in pane_rects if r.contains(nx, ny)), None)
            publish_gaze(nx, ny, hit is not None)

            if (now - last_status_print) >= status_interval:
                where = hit.session_id[:8] if hit else "—"
                marker = " (focused)" if hit and hit.session_id == last_activated else ""
                print(f"[{_ts()}] looking at: {where}{marker}  "
                      f"gaze=({nx:+.2f},{ny:+.2f})", flush=True)
                last_status_print = now

            if args.debug:
                debug_counter += 1
                if debug_counter % debug_print_every == 0:
                    where = hit.session_id[:8] if hit else "—"
                    print(f"[{_ts()}] gaze=({nx:+.2f},{ny:+.2f}) "
                          f"yaw={f.yaw:+.2f} pitch={f.pitch:+.2f} "
                          f"iris=({f.iris_x:+.2f},{f.iris_y:+.2f}) -> {where}",
                          flush=True)

            # Drain any voice commands; send them to whatever pane we currently
            # consider focused. We don't gate this on "hit" because the user
            # may have looked away after issuing the command.
            if voice is not None:
                cmd = voice.get_command()
                if cmd:
                    target_id = last_activated or active_session
                    target_sess = next(
                        (r.session for r in pane_rects
                         if r.session_id == target_id), None)
                    if target_sess is None and pane_rects:
                        target_sess = pane_rects[0].session
                    if target_sess is not None:
                        try:
                            # \r (CR) is what Enter sends in a TTY; \n (LF) is
                            # what Shift+Enter sends and lands as a literal
                            # newline in zsh/bash instead of submitting.
                            await target_sess.async_send_text(cmd + "\r")
                            print(f"[{_ts()}] voice -> "
                                  f"{target_sess.session_id[:8]}: {cmd!r}",
                                  flush=True)
                        except Exception as e:
                            print(f"[{_ts()}] voice send failed: {e!r}",
                                  file=sys.stderr)
                    else:
                        print(f"[{_ts()}] voice command dropped (no pane): "
                              f"{cmd!r}", file=sys.stderr)

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

    if args.overlay:
        from .overlay import Overlay  # lazy import; only needs PyObjC on this path
        stop_event = threading.Event()

        def asyncio_thread() -> None:
            try:
                iterm2.run_until_complete(loop, retry=False)
            except ConnectionRefusedError:
                print(f"[{_ts()}] could not connect to iTerm2's Python API.",
                      file=sys.stderr)
                print("        Enable it: iTerm2 > Settings > General > Magic > "
                      "Enable Python API", file=sys.stderr)
            except Exception as e:
                print(f"[{_ts()}] asyncio thread crashed: {e!r}", file=sys.stderr)
            finally:
                stop_event.set()

        th = threading.Thread(target=asyncio_thread, daemon=True, name="gp-asyncio")
        th.start()

        def get_gaze() -> tuple[float, float, bool] | None:
            with gaze_lock:
                nx = gaze_state["nx"]
                ny = gaze_state["ny"]
                hit = gaze_state["hit"]
                ts = gaze_state["ts"]
            if nx < 0 or (time.time() - ts) > 0.5:
                return None
            return (nx, ny, hit)

        print(f"[{_ts()}] overlay enabled (translucent dot, main screen)",
              flush=True)
        try:
            Overlay(get_gaze, stop_event, fps=args.overlay_fps).run()
        except KeyboardInterrupt:
            print(f"\n[{_ts()}] stopping...", flush=True)
        finally:
            stop_event.set()
            if voice is not None:
                voice.stop()
            base.stop()
        return 0

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
        if voice is not None:
            voice.stop()
        base.stop()
    return 0
