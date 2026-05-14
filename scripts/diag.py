"""Webcam + MediaPipe diagnostic. Prints frame stats and face counts."""
import os
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "face_landmarker.task"


def main() -> int:
    print(f"[diag] opening camera index 0...", flush=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[diag] FAIL: cv2.VideoCapture could not open camera", flush=True)
        return 1
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[diag] FAIL: cv2.VideoCapture.read() returned no frame", flush=True)
        cap.release()
        return 1
    print(f"[diag] frame shape={frame.shape}  mean brightness={frame.mean():.1f}/255",
          flush=True)
    cv2.imwrite("/tmp/diag-frame.jpg", frame)
    print(f"[diag] sample frame saved to /tmp/diag-frame.jpg", flush=True)

    print(f"[diag] creating FaceLandmarker from {MODEL}...", flush=True)
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    lm = vision.FaceLandmarker.create_from_options(opts)

    n_frames = 0
    n_face = 0
    landmark_count = None
    last_iris_ok = None
    last_ts = -1
    print(f"[diag] capturing 30 frames (about 3s)...", flush=True)
    for i in range(30):
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        n_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms
        res = lm.detect_for_video(img, ts_ms)
        if res.face_landmarks:
            n_face += 1
            if landmark_count is None:
                lms = res.face_landmarks[0]
                landmark_count = len(lms)
                last_iris_ok = landmark_count >= 478
                print(f"[diag] first detection: {landmark_count} landmarks, "
                      f"iris={'YES' if last_iris_ok else 'NO'}", flush=True)
                if last_iris_ok:
                    print(f"[diag]   left iris  (468): x={lms[468].x:.3f} y={lms[468].y:.3f}",
                          flush=True)
                    print(f"[diag]   right iris (473): x={lms[473].x:.3f} y={lms[473].y:.3f}",
                          flush=True)
                if res.facial_transformation_matrixes:
                    print(f"[diag]   transformation matrix: present", flush=True)
                else:
                    print(f"[diag]   transformation matrix: MISSING", flush=True)
        time.sleep(0.05)

    cap.release()
    lm.close()
    print(f"[diag] summary: {n_frames} frames captured, {n_face} with a face "
          f"({100 * n_face / max(n_frames, 1):.0f}%)", flush=True)
    if n_frames == 0:
        print("[diag] CAUSE: camera not producing frames", flush=True)
    elif n_face == 0:
        print("[diag] CAUSE: camera works but MediaPipe finds no face", flush=True)
        print("[diag]        Check /tmp/diag-frame.jpg — is your face visible?", flush=True)
    elif landmark_count is not None and landmark_count < 478:
        print(f"[diag] CAUSE: model returns only {landmark_count} landmarks "
              "(no iris). Wrong model file.", flush=True)
    else:
        print("[diag] looks healthy. The earlier 'no face' was likely just a timing miss.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
