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

    pc = sub.add_parser("calibrate", help="run N x N grid calibration (fullscreen)")
    pc.add_argument("--camera", type=int, default=0, help="cv2 camera index (default 0)")
    pc.add_argument("--grid", type=int, default=4,
                    help="grid side (default 4 -> 16 points). 3=fast, 4=balanced, 5=most accurate")
    pc.add_argument("--samples", type=int, default=24,
                    help="frames to median-aggregate per calibration point "
                         "(default 24, was 12). Higher = more robust to "
                         "blinks/saccades/reflection spikes, slower capture.")
    pc.add_argument("--validation-grid", dest="validation_grid", type=int, default=0,
                    help="if >0, the validation phase uses this NxN grid "
                         "(e.g. 4 = 16 points, same as the main grid) instead "
                         "of the default 5-point (4 corners + center) sweep.")
    pc.add_argument("--skip-validate", dest="skip_validate", action="store_true",
                    help="skip the post-calibration validation/refinement phase")

    pr = sub.add_parser("run", help="watch gaze and activate panes")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--dwell-ms", dest="dwell_ms", type=int, default=350,
                    help="ms of stable gaze inside a new pane before switching (default 350)")
    pr.add_argument("--hz", type=float, default=20.0,
                    help="control loop ticks per second (default 20)")
    pr.add_argument("--alpha", type=float, default=0.45,
                    help="EMA smoothing 0..1, lower = smoother. Ignored if "
                         "--smoothing=median. (default 0.45)")
    pr.add_argument("--smoothing", choices=["ema", "median"], default="ema",
                    help="EMA = low-latency exponential average (default). "
                         "median = per-feature rolling median over "
                         "--median-window frames; rejects single-frame outliers "
                         "and helps with glasses/reflection spikes at the cost "
                         "of a couple frames of extra latency.")
    pr.add_argument("--median-window", dest="median_window", type=int, default=5,
                    help="window size for --smoothing=median (default 5).")
    pr.add_argument("--pane-refresh", dest="pane_refresh", type=float, default=1.0,
                    help="seconds between re-querying iTerm pane bounds (default 1.0)")
    pr.add_argument("--chrome-top", dest="chrome_top", type=float, default=52.0,
                    help="points to deduct from top of iTerm window for title+tab bar "
                         "(default 52). Lower if you've hidden the tab bar; raise if you have a "
                         "status bar at the top.")
    pr.add_argument("--status-every", dest="status_every", type=float, default=2.0,
                    help="seconds between 'looking at: <pane>' status lines (default 2.0). "
                         "0.5 minimum.")
    pr.add_argument("--overlay", action="store_true",
                    help="show a translucent always-on-top dot at the predicted gaze "
                         "position (green when inside a known pane, red when outside)")
    pr.add_argument("--overlay-fps", dest="overlay_fps", type=float, default=30.0,
                    help="overlay redraw rate (default 30). Ignored without --overlay.")
    pr.add_argument("--overlay-smoothing", dest="overlay_smoothing",
                    type=float, default=0.2,
                    help="EMA alpha applied to the dot's screen position "
                         "(default 0.2; lower = smoother / laggier). This is "
                         "independent of --alpha, which controls the feature-"
                         "level smoothing that pane switches actually use.")
    pr.add_argument("--voice", action="store_true",
                    help="enable voice control: speak commands and they're sent to "
                         "the pane you're focused on")
    pr.add_argument("--wake-phrase", dest="wake_phrase", default="hey claude",
                    help="phrase that begins a voice command (default: 'hey claude')")
    pr.add_argument("--end-phrase", dest="end_phrase", default="send it",
                    help="phrase that submits the voice command + Enter "
                         "(default: 'send it')")
    pr.add_argument("--lock-phrase", dest="lock_phrase", default="lock the pane",
                    help="phrase that locks the runner to the current pane so "
                         "gaze stops switching it (default: 'lock the pane'). "
                         "Bypasses wake/end.")
    pr.add_argument("--unlock-phrase", dest="unlock_phrase", default="unlock the pane",
                    help="phrase that re-enables gaze-driven pane switching "
                         "(default: 'unlock the pane'). Bypasses wake/end.")
    pr.add_argument("--voice-model", dest="voice_model",
                    default="mlx-community/whisper-small-mlx",
                    help="MLX Whisper model id (default whisper-small-mlx). "
                         "Try mlx-community/whisper-medium-mlx for harder accents "
                         "/ shell tokens, at the cost of ~300 ms extra latency.")
    pr.add_argument("--voice-device", dest="voice_device", type=int, default=None,
                    help="audio input device index for --voice. "
                         "Inspect the list printed at startup; default is the "
                         "system default mic.")
    pr.add_argument("--correct", action="store_true",
                    help="enable mid-session bias correction. Look at your "
                         "mouse cursor and press --correct-hotkey to capture a "
                         "(mouse - predicted-gaze) offset that's added to all "
                         "subsequent predictions. Needs macOS Accessibility "
                         "permission for your terminal app.")
    pr.add_argument("--correct-hotkey", dest="correct_hotkey",
                    default="<ctrl>+<shift>+g",
                    help="global hotkey for --correct (pynput syntax; default "
                         "ctrl+shift+g). Examples: '<cmd>+<shift>+g', "
                         "'<alt>+<f1>'.")
    pr.add_argument("--debug", action="store_true",
                    help="print gaze + head features every ~250ms (in addition to status)")

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
