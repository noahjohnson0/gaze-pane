#!/usr/bin/env bash
# Bootstrap the gaze-pane project: venv, deps, and the MediaPipe model.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[setup] $PYTHON not found. Install with: brew install python@3.11" >&2
    exit 1
fi
echo "[setup] Using $($PYTHON --version) at $(command -v "$PYTHON")"

if [ ! -d .venv ]; then
    echo "[setup] Creating .venv..."
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

MODEL=face_landmarker.task
if [ ! -f "$MODEL" ]; then
    echo "[setup] Downloading $MODEL..."
    curl -L -o "$MODEL" \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
fi

cat <<EOF

[setup] Done.

Next steps:
  1. In iTerm2, enable the Python API:
        Settings (Cmd+,) > General > Magic > Enable Python API
     iTerm2 will prompt the first time gaze-pane connects; click "Always allow"
     for this script.
  2. Activate the venv:
        source .venv/bin/activate
  3. Calibrate (one time, or after moving your camera/monitor):
        python -m gaze_pane calibrate
  4. Run:
        python -m gaze_pane run

EOF
