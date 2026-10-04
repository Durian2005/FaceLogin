import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from model_paths import DETECTOR_NAME, loadable, source

DETECTOR = loadable(source(DETECTOR_NAME))

det = cv2.FaceDetectorYN.create(DETECTOR, "", (320, 320), 0.5, 0.3, 5000)
print("DETECTOR ready, threshold=0.5")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("CAMERA_OPEN", cap.isOpened())
if not cap.isOpened():
    raise SystemExit(3)

try:
    for i in range(12):
        ok, frame = cap.read()
        if not ok:
            print(f"frame {i}: READ_FAIL")
            continue
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        n = 0 if faces is None else len(faces)
        best = 0.0
        if n:
            best = max(float(f[14]) for f in faces)
        gray_mean = frame.mean()
        print(f"frame {i}: size={w}x{h} faces={n} best_score={round(best,3)} brightness={round(gray_mean,1)}")
        time.sleep(0.3)
finally:
    cap.release()
    print("CAMERA_RELEASED True")
print("NO_IMAGE_SAVED True")
