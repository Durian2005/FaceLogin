import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from model_paths import DETECTOR_NAME, RECOGNIZER_NAME, loadable, source

DETECTOR = loadable(source(DETECTOR_NAME))
RECOGNIZER = loadable(source(RECOGNIZER_NAME))

det = cv2.FaceDetectorYN.create(DETECTOR, "", (320, 320), 0.5, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(RECOGNIZER, "")
print("ENGINE ready (detector threshold=0.5)")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("CAMERA_OPEN_FAIL")
    raise SystemExit(3)
print("CAMERA_OPEN ok -> please face the camera, only one face in view")


def wait_for_face(cap, seconds, label):
    deadline = time.time() + seconds
    hit = None
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        n = 0 if faces is None else len(faces)
        if n:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            print(f"[{label}] face found: faces={n} score={round(float(faces[0][14]),3)}")
            hit = (frame, faces[0])
            break
        time.sleep(0.15)
    if hit is None:
        print(f"[{label}] no face within {seconds}s")
    return hit


try:
    a = wait_for_face(cap, 15, "A")
    time.sleep(1.0)
    b = wait_for_face(cap, 15, "B")
finally:
    cap.release()
    print("CAMERA_RELEASED True")

if not a or not b:
    print("VERDICT blocked: camera view had no detectable face")
    raise SystemExit(5)

fa = rec.alignCrop(a[0], a[1])
fb = rec.alignCrop(b[0], b[1])
va = rec.feature(fa)
vb = rec.feature(fb)
score = rec.match(va, vb, cv2.FaceRecognizerSF_FR_COSINE)
print("SAME_PERSON_COSINE", round(float(score), 4))
print("REFERENCE_DOC_THRESHOLD 0.363")
print("VERDICT", "above_doc_threshold" if float(score) >= 0.363 else "below_doc_threshold")
print("NO_IMAGE_OR_FEATURE_SAVED True")
