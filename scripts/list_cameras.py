"""Try camera indices 0..3, report which work, brightness, and face-detection."""
import os
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "face_landmarker.task"
opts = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
lm = vision.FaceLandmarker.create_from_options(opts)

ts_seed = int(time.time() * 1000)
for idx in range(4):
    print(f"\n=== camera index {idx} ===", flush=True)
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[idx={idx}] not available", flush=True)
        continue
    # Some cameras need a few frames to warm up.
    for _ in range(5):
        cap.read()
        time.sleep(0.05)
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[idx={idx}] opened but read failed", flush=True)
        cap.release()
        continue
    brightness = frame.mean()
    cv2.imwrite(f"/tmp/diag-cam{idx}.jpg", frame)
    print(f"[idx={idx}] shape={frame.shape}  brightness={brightness:.1f}/255  "
          f"saved /tmp/diag-cam{idx}.jpg", flush=True)
    # Test face detection over 10 frames
    n_face = 0
    for i in range(10):
        ok, f = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_seed += 50
        res = lm.detect_for_video(img, ts_seed)
        if res.face_landmarks:
            n_face += 1
        time.sleep(0.05)
    print(f"[idx={idx}] face detected in {n_face}/10 frames", flush=True)
    cap.release()

lm.close()
print("\nDone.", flush=True)
