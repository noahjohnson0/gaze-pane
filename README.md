# gaze-pane

Auto-select the iTerm2 pane you're looking at. Webcam tracks your eyes, a small
affine model maps gaze to screen position, and the iTerm2 Python API focuses
whichever pane the gaze lands in.

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.10, 3.11, or 3.12 (MediaPipe wheels don't ship for 3.13+ yet)
- iTerm2 with the Python API enabled:
  `Settings > General > Magic > Enable Python API`
- A working webcam. macOS will prompt for Camera permission for your terminal
  the first time.

## Install

```bash
cd ~/repos/gaze-pane
./setup.sh                  # creates .venv, installs deps, downloads the MediaPipe model
source .venv/bin/activate
```

## Use

```bash
python -m gaze_pane calibrate   # one-time: 9 fullscreen dots, press SPACE at each
python -m gaze_pane run         # watch gaze + switch panes
```

Tuning flags on `run`:

| flag | default | what it does |
| --- | --- | --- |
| `--dwell-ms` | 350 | how long your gaze must stay in a new pane before switching |
| `--alpha` | 0.45 | EMA smoothing on the gaze vector, lower = smoother |
| `--hz` | 20 | control loop rate |
| `--pane-refresh` | 1.0 | seconds between re-querying iTerm pane bounds |
| `--chrome-top` | 52 | points to deduct from the top of the iTerm window (title bar + tab bar). Set lower if you hide the tab bar. |
| `--debug` | off | prints `gaze=(nx,ny) -> pane <id>` so you can see what it sees |

Recalibrate any time you move the camera, change monitors, or sit differently.
Calibration is per user, saved to `~/.config/gaze-pane/calibration.json`.

## How it works

1. **Capture.** OpenCV grabs frames from the webcam in a background thread.
2. **Landmarks.** MediaPipe's `FaceLandmarker` Tasks API gives 478 face landmarks
   (including iris) plus a 4x4 head transformation matrix per frame.
3. **Features.** We reduce each frame to a 4-dim vector:
   eye-corner-normalized mean iris offset `(x, y)` plus head `(yaw, pitch)`
   extracted from the transformation matrix.
4. **Map.** A 9-point calibration fits a least-squares affine
   `(features) -> (normalized screen x, y)`. Five unknowns per axis vs nine
   equations stays well-conditioned without overfitting.
5. **Pane hit-test.** iTerm2's `Session` doesn't expose a pixel frame, so we
   walk the tab's splitter tree, weight each subtree by its `grid_size` in
   cells, and recursively assign each pane a proportional rect inside the
   window's content area (window frame minus title/tab-bar chrome). Then we
   check which rect contains the gaze point.
6. **Activate.** After `dwell-ms` of stable gaze in a non-active pane, call
   `session.async_activate()`.

## Known limitations

- Single monitor. The mapper assumes the gaze lands on the main display.
- Webcam gaze is inherently coarse. Expect roughly 2-3 inch precision on a 14"
  laptop, which is fine for typical pane sizes (two-up or four-quadrant), less
  fine for many small panes.
- The calibration is sensitive to head distance. Try to sit at roughly the same
  distance you calibrated at, or recalibrate.
- The Python API connection prompts iTerm for permission the first time the
  script runs. Click "Always allow".

## Layout

```
gaze-pane/
  setup.sh                 bootstraps venv + deps + model
  requirements.txt
  pyproject.toml
  face_landmarker.task     downloaded by setup.sh (gitignored)
  gaze_pane/
    __main__.py            CLI: calibrate / run
    gaze.py                webcam thread + MediaPipe -> feature vector
    mapper.py              affine fit + save/load
    iterm.py               iTerm2 Python API helpers
    calibrate.py           fullscreen Tk calibration UI
    main.py                runtime orchestrator
```
