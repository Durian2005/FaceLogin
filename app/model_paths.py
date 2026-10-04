"""模型文件路径解析（全项目单一来源）。

默认位置是仓库内的 `models/` 目录，可用环境变量 `FACELOGIN_MODELS` 覆盖。

为什么需要单独一个模块
----------------------
OpenCV 无法从**非 ASCII 路径**读取 ONNX 模型。本项目在 Windows + OpenCV 5.0.0 下实测：

1. 直接把含中文的路径交给 `cv2.FaceDetectorYN.create()`
   → `error: Can't read ONNX file`（-5:Bad argument）
2. 改传内存 bytes（C++ 侧有 buffer 重载）
   → Python 绑定不接受，进程直接**段错误**退出(exit 139)，比抛异常更糟
3. 用 Windows 8.3 短路径（`GetShortPathNameW`）
   → 数据卷默认不生成短名，函数原样返回中文路径，无效

可行且唯一稳妥的办法：路径非 ASCII 时，把模型**镜像到纯 ASCII 目录**再交给 OpenCV
（本模块的 `loadable()`）。国内中文项目目录、中文用户名都很常见，这层兜底是必需的。
"""

import os
import shutil
import tempfile

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

DETECTOR_NAME = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_NAME = "face_recognition_sface_2021dec.onnx"

#: 模型所在目录。默认 = 仓库根目录下的 models/，除非环境变量指定。
MODELS_DIR = os.environ.get("FACELOGIN_MODELS") or os.path.join(PROJECT_DIR, "models")

_cache_resolved = False
_cache_dir = None


def _is_ascii(path):
    """OpenCV 的路径处理只看字节，任何非 ASCII 字符都可能出问题。"""
    return all(ord(ch) < 128 for ch in path)


def ascii_cache_dir():
    """挑一个**纯 ASCII 且可写**的缓存目录；没有则返回 None。

    按优先级尝试：%LOCALAPPDATA%\\FaceLogin\\models → 临时目录 → C:\\ProgramData。
    每个候选都要先过 ASCII 检查，因为用户名本身也可能是中文。
    """
    global _cache_resolved, _cache_dir
    if _cache_resolved:
        return _cache_dir

    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(os.path.join(local, "FaceLogin", "models"))
    candidates.append(os.path.join(tempfile.gettempdir(), "facelogin_models"))
    candidates.append(os.path.join("C:\\", "ProgramData", "FaceLogin", "models"))

    for candidate in candidates:
        if not _is_ascii(candidate):
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        _cache_dir = candidate
        break

    _cache_resolved = True
    return _cache_dir


def loadable(path):
    """返回 OpenCV 能真正读取的路径。

    - 路径是纯 ASCII  → 原样返回，零开销
    - 路径含非 ASCII  → 镜像到 ASCII 缓存目录，返回副本路径（大小/时间戳一致则复用）

    镜像失败时退回原路径，让 OpenCV 抛它自己的错误 —— 至少错误信息指向的是真实文件。
    """
    if _is_ascii(path):
        return path

    cache = ascii_cache_dir()
    if not cache:
        print("[WARN] no ascii cache dir available, opencv may fail on this path")
        return path

    dst = os.path.join(cache, os.path.basename(path))
    try:
        reusable = (os.path.exists(dst)
                    and os.path.getsize(dst) == os.path.getsize(path)
                    and os.path.getmtime(dst) >= os.path.getmtime(path))
        if not reusable:
            shutil.copy2(path, dst)
            print("[INFO] model mirrored to ascii path:", dst)
        return dst
    except OSError as exc:
        print("[WARN] model mirror failed (%s), using original path" % exc)
        return path


def source(name):
    """模型在 MODELS_DIR 下的原始路径。"""
    return os.path.join(MODELS_DIR, name)
