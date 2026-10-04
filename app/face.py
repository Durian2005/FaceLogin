"""人脸引擎：摄像头读取、YuNet 检测、SFace 对齐与特征、相似度比较。

模型位置由 `model_paths` 解析：默认取仓库内的 `models/`，
若该路径含非 ASCII 字符会自动镜像到 ASCII 目录 —— 原因见 model_paths 模块文档。
"""

import os
import threading
import time

import cv2
import numpy as np

from model_paths import DETECTOR_NAME, RECOGNIZER_NAME, loadable, source

DETECTOR_SRC = source(DETECTOR_NAME)
RECOGNIZER_SRC = source(RECOGNIZER_NAME)


class FaceEngine:
    def __init__(self):
        # 先用**原始路径**判断文件在不在，这样报错信息对用户是可操作的
        missing = [p for p in (DETECTOR_SRC, RECOGNIZER_SRC) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("model_not_found: " + ", ".join(missing))

        self.detector = cv2.FaceDetectorYN.create(
            loadable(DETECTOR_SRC), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(loadable(RECOGNIZER_SRC), "")

        self.camera_lock = threading.Lock()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ------------------------------------------------------------ camera

    def camera_ready(self):
        return bool(self.cap.isOpened())

    def read_frame(self):
        with self.camera_lock:
            ok, frame = self.cap.read()
        return frame if ok else None

    # ------------------------------------------------------------ detection

    def detect_largest(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    def draw(self, frame, face):
        out = frame.copy()
        if face is not None:
            x, y, w, h = [int(v) for v in face[:4]]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(out, "{:.2f}".format(float(face[14])), (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        else:
            cv2.putText(out, "no face", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    # ------------------------------------------------------------ features

    def capture_feature(self):
        """采集一帧并提取特征。返回 (特征 or None, 状态, 耗时毫秒)。"""
        t0 = time.perf_counter()
        frame = self.read_frame()
        if frame is None:
            return None, "camera_read_failed", 0.0
        face = self.detect_largest(frame)
        if face is None:
            return None, "no_face_detected", round((time.perf_counter() - t0) * 1000, 1)
        aligned = self.recognizer.alignCrop(frame, face)
        feat = self.recognizer.feature(aligned).copy()
        return feat, "ok", round((time.perf_counter() - t0) * 1000, 1)

    def capture_many(self, count=3, interval=0.35):
        feats = []
        for _ in range(count):
            feat, _status, _ms = self.capture_feature()
            if feat is not None:
                feats.append(feat)
            time.sleep(interval)
        return feats

    def best_cosine(self, refs, feat):
        """refs: 已存模板的特征数组列表；返回最高余弦相似度。"""
        best = -1.0
        for ref in refs:
            score = self.recognizer.match(ref.reshape(1, -1), feat,
                                          cv2.FaceRecognizerSF_FR_COSINE)
            best = max(best, float(score))
        return best


def pack(vec):
    """特征向量 -> BLOB；维度放进数组头部，读取时校验。"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr.tobytes(), int(arr.shape[0])


def unpack(blob, dim):
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != dim:
        return None
    return arr
