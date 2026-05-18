"""Webcam capture + MediaPipe face/iris landmarks -> 4-dim gaze feature vector.

The feature vector is intentionally compact (mean iris offset x/y + head yaw/pitch)
so that a 9-point affine calibration is well-conditioned (5 unknowns per axis,
9 equations).
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe Face Mesh landmark indices we care about.
# (Index conventions vary across docs; what matters is consistency between
#  calibration and runtime, which is guaranteed because we use the same code path.)
L_EYE_OUTER, L_EYE_INNER = 33, 133
R_EYE_OUTER, R_EYE_INNER = 263, 362
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473
L_UPPER_LID, L_LOWER_LID = 159, 145
R_UPPER_LID, R_LOWER_LID = 386, 374

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _resolve_model_path() -> Path:
    """Return a usable path to face_landmarker.task, downloading if missing.

    Search order:
      1. repo-root sibling (development checkout, fetched by setup.sh)
      2. ~/.cache/gaze-pane/face_landmarker.task (download cache)
    The cache path is created on demand by fetching from MediaPipe's public
    Google Storage bucket — same URL setup.sh uses.
    """
    dev_path = Path(__file__).resolve().parent.parent / "face_landmarker.task"
    if dev_path.exists():
        return dev_path

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME",
                                    Path.home() / ".cache")) / "gaze-pane"
    cache_path = cache_dir / "face_landmarker.task"
    if cache_path.exists():
        return cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gaze-pane] downloading MediaPipe face landmarker model "
          f"(~3.6 MB) -> {cache_path}", flush=True)
    import urllib.request
    tmp = cache_path.with_suffix(".task.partial")
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    tmp.replace(cache_path)
    return cache_path


DEFAULT_MODEL = _resolve_model_path  # callable; resolved lazily in GazeTracker


@dataclass
class GazeFeatures:
    l_iris_x: float    # left iris x-offset, eye-width normalized
    l_iris_y: float    # left iris y-offset, eye-width normalized
    r_iris_x: float    # right iris x-offset, eye-width normalized
    r_iris_y: float    # right iris y-offset, eye-width normalized
    yaw: float         # head yaw in radians (Y rotation)
    pitch: float       # head pitch in radians (X rotation)
    # Eye openness: (lower_lid_y - upper_lid_y) / eye_width. Drops when the user
    # looks down because the upper lid lowers slightly; better vertical signal
    # than iris-y alone.
    l_openness: float
    r_openness: float
    # Inter-iris distance in image-normalized coords. Acts as a "how close is
    # the face to the camera" feature so calibration doesn't fall apart when
    # you lean in or away.
    face_scale: float
    timestamp: float

    @property
    def iris_x(self) -> float:
        return (self.l_iris_x + self.r_iris_x) / 2

    @property
    def iris_y(self) -> float:
        return (self.l_iris_y + self.r_iris_y) / 2

    def as_vec(self) -> np.ndarray:
        return np.array([
            self.l_iris_x, self.l_iris_y,
            self.r_iris_x, self.r_iris_y,
            self.yaw, self.pitch,
            self.l_openness, self.r_openness,
            self.face_scale,
        ], dtype=np.float64)


def _yaw_pitch_from_matrix(m: np.ndarray) -> tuple[float, float]:
    """Extract yaw (Y) and pitch (X) from a 4x4 rigid transformation matrix."""
    r = m[:3, :3]
    yaw = math.atan2(r[0, 2], r[2, 2])
    pitch = math.atan2(-r[1, 2], math.sqrt(r[1, 0] ** 2 + r[1, 1] ** 2))
    return yaw, pitch


def _extract_features(result, ts: float) -> Optional[GazeFeatures]:
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    if len(lms) < 478:
        # iris landmarks (468+) need refine_landmarks; FaceLandmarker emits 478 by default.
        return None

    def pt(i: int) -> tuple[float, float]:
        return lms[i].x, lms[i].y

    lo, li = pt(L_EYE_OUTER), pt(L_EYE_INNER)
    ro, ri = pt(R_EYE_OUTER), pt(R_EYE_INNER)
    l_iris = pt(L_IRIS_CENTER)
    r_iris = pt(R_IRIS_CENTER)

    l_w = max(abs(lo[0] - li[0]), 1e-6)
    r_w = max(abs(ro[0] - ri[0]), 1e-6)
    l_cx, l_cy = (lo[0] + li[0]) / 2, (lo[1] + li[1]) / 2
    r_cx, r_cy = (ro[0] + ri[0]) / 2, (ro[1] + ri[1]) / 2
    l_dx, l_dy = (l_iris[0] - l_cx) / l_w, (l_iris[1] - l_cy) / l_w
    r_dx, r_dy = (r_iris[0] - r_cx) / r_w, (r_iris[1] - r_cy) / r_w

    # Eye openness: vertical lid gap divided by eye width. Drops a few percent
    # when the user looks down because the upper lid sags slightly with gaze.
    l_op = (lms[L_LOWER_LID].y - lms[L_UPPER_LID].y) / l_w
    r_op = (lms[R_LOWER_LID].y - lms[R_UPPER_LID].y) / r_w

    # Inter-iris distance: larger when face is closer to camera.
    face_scale = math.hypot(l_iris[0] - r_iris[0], l_iris[1] - r_iris[1])

    if result.facial_transformation_matrixes:
        m = np.asarray(result.facial_transformation_matrixes[0])
        yaw, pitch = _yaw_pitch_from_matrix(m)
    else:
        yaw = pitch = 0.0

    return GazeFeatures(
        l_iris_x=l_dx, l_iris_y=l_dy,
        r_iris_x=r_dx, r_iris_y=r_dy,
        yaw=yaw, pitch=pitch,
        l_openness=l_op, r_openness=r_op,
        face_scale=face_scale,
        timestamp=ts,
    )


class GazeTracker:
    """Background thread that owns the webcam and the FaceLandmarker.

    Call `start()` to begin capturing; `get_latest()` returns the most recent
    features (or None if no face or stale). Call `stop()` to release the camera.
    """

    def __init__(self, *, model_path: Optional[Path] = None, camera_index: int = 0):
        self.model_path = (Path(model_path) if model_path
                           else _resolve_model_path())
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {self.model_path}."
            )
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._landmarker = None
        self._latest: Optional[GazeFeatures] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Quiet TensorFlow/MediaPipe spam.
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    def start(self) -> None:
        if self._running:
            return
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.camera_index}. "
                "On first run macOS will ask for Camera permission for your terminal."
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="gaze-tracker", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None and self._landmarker is not None
        last_ts_ms = -1
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Mirror so left-right of head correlates intuitively with left-right of screen.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            # FaceLandmarker.VIDEO mode requires strictly increasing timestamps.
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            try:
                result = self._landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                continue
            features = _extract_features(result, ts=ts_ms / 1000.0)
            with self._lock:
                # Always publish the latest frame for the webcam preview, even
                # when no face is detected — keeps the thumbnail live.
                self._latest_frame = frame
                if features is not None:
                    self._latest = features

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Most recent (mirrored, BGR) webcam frame, or None if not started."""
        with self._lock:
            return self._latest_frame

    def get_latest(self, max_age_s: float = 0.5) -> Optional[GazeFeatures]:
        with self._lock:
            f = self._latest
        if f is None:
            return None
        if (time.time() - f.timestamp) > max_age_s:
            return None
        return f

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "GazeTracker":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class SmoothedTracker:
    """Smoothing wrapper over GazeTracker. Two modes:

    - "ema" (default): exponential moving average; low latency, low CPU,
      but a single noisy frame still pulls the output partway.
    - "median": rolling per-feature median over the last `window` samples.
      Rejects single-frame outliers cleanly (e.g., specular reflections on
      glasses or sudden iris-landmark jumps). Costs `window`/2 frames of
      extra latency.
    """

    def __init__(self, base: GazeTracker, alpha: float = 0.5,
                 mode: str = "ema", window: int = 5):
        if mode not in ("ema", "median"):
            raise ValueError(f"unknown smoothing mode {mode!r}")
        self.base = base
        self.alpha = float(alpha)
        self.mode = mode
        self.window = int(max(1, window))
        self._smoothed: Optional[np.ndarray] = None
        self._buf: list[np.ndarray] = []

    def get_latest(self, max_age_s: float = 0.5) -> Optional[GazeFeatures]:
        f = self.base.get_latest(max_age_s=max_age_s)
        if f is None:
            return None
        v = f.as_vec()
        if self.mode == "median":
            self._buf.append(v)
            if len(self._buf) > self.window:
                self._buf.pop(0)
            s = np.median(np.stack(self._buf), axis=0)
        else:  # ema
            if self._smoothed is None:
                self._smoothed = v.copy()
            else:
                self._smoothed = self.alpha * v + (1 - self.alpha) * self._smoothed
            s = self._smoothed
        return GazeFeatures(
            l_iris_x=float(s[0]), l_iris_y=float(s[1]),
            r_iris_x=float(s[2]), r_iris_y=float(s[3]),
            yaw=float(s[4]), pitch=float(s[5]),
            l_openness=float(s[6]), r_openness=float(s[7]),
            face_scale=float(s[8]),
            timestamp=f.timestamp,
        )
