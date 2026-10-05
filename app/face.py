"""人脸引擎：摄像头读取、YuNet 检测、SFace 对齐与特征、相似度比较、头部姿态代理值。

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

# 单张人脸的检测输出是 15 个浮点数，关键点顺序经真实摄像头逐点核对确认（非照抄文档）：
#   [0:4]   人脸框 x, y, w, h
#   [4:6]   眼睛 A      [6:8]   眼睛 B     —— y 最小，位于画面最上方
#   [8:10]  鼻尖        —— y 居中
#   [10:12] 嘴角 A      [12:14] 嘴角 B     —— y 最大，位于画面最下方
#   [14]    检测置信度
# 注意：只有这 5 个点、**不含眼睑**，所以眨眼类活体检测做不了（见 README 已知限制）。
_EYE_A, _EYE_B, _NOSE = slice(4, 6), slice(6, 8), slice(8, 10)
_MOUTH_A, _MOUTH_B = slice(10, 12), slice(12, 14)


class FaceEngine:
    def __init__(self, open_camera=True):
        # 先用**原始路径**判断文件在不在，这样报错信息对用户是可操作的
        missing = [p for p in (DETECTOR_SRC, RECOGNIZER_SRC) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("model_not_found: " + ", ".join(missing))

        self.detector = cv2.FaceDetectorYN.create(
            loadable(DETECTOR_SRC), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(loadable(RECOGNIZER_SRC), "")

        self.camera_lock = threading.Lock()
        # OpenCV 的 DetectorYN / RecognizerSF 实例不保证线程安全，而 MJPEG 推流线程
        # 与登录/录入/活体采样线程会同时用到它们 —— 不加锁属于潜在的随机崩溃。
        self.model_lock = threading.Lock()

        # open_camera=False 用于**批量离线评测**（tools/eval_threshold.py 要跑几千张
        # LFW 图片）。那些场景只需要检测与特征提取，没有理由占住摄像头 ——
        # 否则跑一次评测就会把摄像头指示灯点亮，还可能和正在用的程序抢设备。
        self.cap = None
        if open_camera:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ------------------------------------------------------------ camera

    def camera_ready(self):
        return bool(self.cap is not None and self.cap.isOpened())

    def read_frame(self):
        if self.cap is None:
            return None
        with self.camera_lock:
            ok, frame = self.cap.read()
        return frame if ok else None

    # ------------------------------------------------------------ detection

    def detect_largest(self, frame):
        """返回画面中面积最大的那张脸（15 个浮点数的数组），没有则返回 None。"""
        h, w = frame.shape[:2]
        with self.model_lock:
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

    def capture_from(self, frame, face):
        """从**已指定**的一帧与一张脸上提特征。

        活体检测需要「动作做完、人已回正」的那一帧来做比对，所以不能重新抓帧
        —— 重新抓帧会让「通过动作的脸」和「用于比对的脸」不是同一帧。
        """
        with self.model_lock:
            aligned = self.recognizer.alignCrop(frame, face)
            return self.recognizer.feature(aligned).copy()

    def capture_feature(self):
        """采集一帧并提取特征。返回 (特征 or None, 状态, 耗时毫秒)。"""
        t0 = time.perf_counter()
        frame = self.read_frame()
        if frame is None:
            return None, "camera_read_failed", 0.0
        face = self.detect_largest(frame)
        if face is None:
            return None, "no_face_detected", round((time.perf_counter() - t0) * 1000, 1)
        feat = self.capture_from(frame, face)
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


def landmarks(face):
    """从检测结果中取出 5 个关键点，返回形状 (5, 2) 的数组。"""
    f = np.asarray(face, dtype=np.float32).reshape(-1)
    return np.array([f[_EYE_A], f[_EYE_B], f[_NOSE], f[_MOUTH_A], f[_MOUTH_B]],
                    dtype=np.float32)


def pose_metrics(face):
    """由 5 个关键点估算头部姿态的**代理值**（不是真实角度，无需相机标定）。

    返回 (yaw, pitch, iod)：

      yaw   = (鼻尖 x - 双眼中点 x) / 瞳距
              正对镜头时接近 0；**增大 = 头转向使用者自己的左侧**。
              推导：面对镜头时人的右侧落在画面左侧，因此头转向自己左侧时
              鼻尖朝画面右侧偏移。鼻尖前凸约 4–6cm，绕垂直轴转 25° 时，
              在 60cm 距离、640px 宽的画面上约位移 24px，除以瞳距（实测约 60px）≈ 0.4。

      pitch = (鼻尖 y - 双眼中点 y) / (双嘴角中点 y - 双眼中点 y)
              即鼻尖在「眼线 → 嘴线」这段竖向跨度上的相对高度。
              正视约 0.4–0.55（取值因人脸型而异，所以判定一律使用**相对基线的变化量**）；
              **减小 = 抬头，增大 = 低头**。

    iod 是瞳距的像素值，可用于判断人脸是否离镜头过远。
    """
    pts = landmarks(face)
    eye_mid = (pts[0] + pts[1]) / 2.0
    mouth_mid = (pts[3] + pts[4]) / 2.0
    nose = pts[2]

    iod = float(np.linalg.norm(pts[0] - pts[1]))
    yaw = float(nose[0] - eye_mid[0]) / max(iod, 1e-6)
    span = float(mouth_mid[1] - eye_mid[1])
    pitch = float(nose[1] - eye_mid[1]) / max(span, 1e-6)
    return yaw, pitch, iod


def pack(vec):
    """特征向量 -> BLOB；维度放进数组头部，读取时校验。"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr.tobytes(), int(arr.shape[0])


def unpack(blob, dim):
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != dim:
        return None
    return arr
