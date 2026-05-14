"""CLI: `python -m gaze_pane {calibrate,run}`."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(
        prog="gaze_pane",
        description="Auto-select the iTerm2 pane you're looking at.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="run 9-point calibration (fullscreen)")
    pc.add_argument("--camera", type=int, default=0, help="cv2 camera index (default 0)")
    pc.add_argument("--samples", type=int, default=12,
                    help="frames to average per calibration point (default 12)")

    pr = sub.add_parser("run", help="watch gaze and activate panes")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--dwell-ms", dest="dwell_ms", type=int, default=350,
                    help="ms of stable gaze inside a new pane before switching (default 350)")
    pr.add_argument("--hz", type=float, default=20.0,
                    help="control loop ticks per second (default 20)")
    pr.add_argument("--alpha", type=float, default=0.45,
                    help="EMA smoothing 0..1, lower = smoother (default 0.45)")
    pr.add_argument("--pane-refresh", dest="pane_refresh", type=float, default=1.0,
                    help="seconds between re-querying iTerm pane bounds (default 1.0)")
    pr.add_argument("--chrome-top", dest="chrome_top", type=float, default=52.0,
                    help="points to deduct from top of iTerm window for title+tab bar "
                         "(default 52). Lower if you've hidden the tab bar; raise if you have a "
                         "status bar at the top.")
    pr.add_argument("--debug", action="store_true",
                    help="print gaze coords + pane hits every ~250ms")

    args = p.parse_args()
    if args.cmd == "calibrate":
        from .calibrate import cmd_calibrate
        return cmd_calibrate(args)
    if args.cmd == "run":
        from .main import cmd_run
        return cmd_run(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
