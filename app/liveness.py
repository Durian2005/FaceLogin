"""活体检测：随机动作挑战。

为什么是「动作挑战」而不是「眨眼检测」
- YuNet 只输出人脸框 + 5 个关键点（双眼、鼻尖、左右嘴角），**不含眼睑**，
  无法判断眼睛开合 —— 眨眼检测在这一技术栈里做不到，除非再引入一个关键点模型。
- 用这 5 个点可以粗估头部姿态：鼻尖相对双眼中点的水平偏移 ≈ yaw，
  鼻尖在「眼线→嘴线」上的相对高度 ≈ pitch。零额外模型、零额外依赖。

能防什么、不能防什么（README 与使用教程里同样写明，不要夸大）
- 能防：拿一张静态照片、或一段**事先录好**的视频对着摄像头。挑战动作由服务端
  随机生成、一次性、带有效期，预录素材对不上随机指令。
- 不能防：攻击者拿着照片 / 手机屏**实时**配合动作旋转，以及实时换脸与深度伪造。
  它的作用是把门槛从「举张照片」提高到「必须实时配合随机动作」，属于轻量反欺骗，
  **不等于完整的活体检测**。

判定为什么一律用「相对基线的变化量」
- yaw 的正视取值因人、因坐姿而异（实测同一个人不同时刻在 +0.01 ~ +0.07 之间波动），
  pitch 的正视取值还取决于鼻长与嘴位。用绝对阈值会误判，所以先测基线再比变化。
"""

import math
import random
import secrets
import threading
import time
from statistics import median

from face import pose_metrics

# ---------------------------------------------------------------- 动作定义

ACTIONS = ("left", "right", "up", "down")

ACTION_TEXT = {
    "left": "把头转向你的左侧",
    "right": "把头转向你的右侧",
    "up": "把头略微抬起（向上看）",
    "down": "把头略微低下（向下看）",
}

ACTION_HINT = "幅度大一点，保持约 1 秒，然后转回正对摄像头"

# ---------------------------------------------------------------- 判定参数

SAMPLE_INTERVAL = 0.07   # 采样间隔，约 14 Hz
BASELINE_FRAMES = 4      # 求基线所需的帧数
MAX_SECONDS = 8.0        # 单次挑战最长时间
HIT_WINDOW = 5           # 判定动作时滑窗长度
HIT_NEEDED = 3           # 滑窗内至少命中几帧才算动作成立
NEUTRAL_FRAMES = 2       # 回正需要连续几帧
MAX_MISS = 10            # 连续读帧失败 / 检测不到人脸的上限
MAX_JUMPS = 3            # 允许的人脸位置跳变次数
JUMP_RATIO = 0.6         # 相邻采样帧人脸中心位移 / 人脸宽度 的上限
MIN_SAMPLES = 6          # 判定所需的最少有效帧

YAW_TURN = 0.13          # 「明显转头」的 yaw 偏移阈值
PITCH_TILT = 0.10        # 「明显抬头/低头」的 pitch 偏移阈值
NEUTRAL_YAW = 0.07       # 视为已回正的 yaw 容差
NEUTRAL_PITCH = 0.05     # 视为已回正的 pitch 容差

CHALLENGE_TTL = 90.0     # 挑战有效期（秒）
MAX_PENDING = 128        # 未被消费的挑战上限，防无界增长


# ---------------------------------------------------------------- 挑战票据

_pending = {}
_pending_lock = threading.Lock()


def new_challenge(session_key):
    """生成一次性挑战。返回 (token, action)。

    动作只随机、不加密 —— 用户必须知道要做什么；安全性来自「随机 + 新鲜 + 一次有效」，
    而不是来自保密。
    """
    now = time.time()
    token = secrets.token_urlsafe(18)
    action = random.choice(ACTIONS)
    with _pending_lock:
        for tok in [t for t, it in _pending.items() if now - it["created"] > CHALLENGE_TTL]:
            _pending.pop(tok, None)
        if len(_pending) >= MAX_PENDING:
            oldest = min(_pending, key=lambda t: _pending[t]["created"])
            _pending.pop(oldest, None)
        _pending[token] = {"action": action, "created": now, "session": session_key}
    return token, action


def consume(token, session_key):
    """取出并作废一张挑战票据。返回 (item, 错误码)。

    先 pop 再校验：无论后续校验成功与否，这张票据都不会被第二次使用。
    """
    with _pending_lock:
        item = _pending.pop(token, None)
    if item is None:
        return None, "unknown_challenge"
    if time.time() - item["created"] > CHALLENGE_TTL:
        return None, "challenge_expired"
    if item["session"] != session_key:
        return None, "challenge_mismatch"
    return item, None


def pending_count():
    with _pending_lock:
        return len(_pending)


# ---------------------------------------------------------------- 判定

def satisfied(action, dy, dp):
    """给定相对基线的偏移，判断本帧是否满足指定动作。"""
    if action == "left":
        return dy >= YAW_TURN
    if action == "right":
        return dy <= -YAW_TURN
    if action == "up":
        return dp <= -PITCH_TILT
    return dp >= PITCH_TILT          # down


def _result(passed, action, error, feature, diag):
    return {"passed": bool(passed), "action": action, "error": error,
            "feature": feature, "diag": diag}


def diag_text(diag):
    """把诊断量压成一行，写进审计。"""
    return ("yaw %+.2f..%+.2f pitch %+.2f..%+.2f in %.1fs"
            % (diag["min_yaw_delta"], diag["max_yaw_delta"],
               diag["min_pitch_delta"], diag["max_pitch_delta"], diag["elapsed"]))


def run_challenge(engine, action, max_seconds=MAX_SECONDS, sleep=time.sleep):
    """采样摄像头画面，判断指定动作是否真实发生，并从「已回正」的那一帧提特征。

    engine 需要提供 read_frame() / detect_largest(frame) / capture_from(frame, face)，
    以便测试时注入假引擎。

    分三阶段：
      1) 基线：取若干帧求 yaw / pitch 的中位数
      2) 等动作：滑窗内命中足够帧数即认为动作成立
      3) 等回正：动作成立后继续采样，等头部回到正对镜头，用那一帧做特征比对
         —— 这一步同时保证「通过动作的脸」和「用于比对的脸」是同一段连续画面。

    返回 dict(passed, action, error, feature, diag)。失败时不产生 feature，
    也不消耗登录失败次数（调用方决定）。
    """
    t0 = time.monotonic()
    base_yaw_samples = []
    base_pitch_samples = []
    base_yaw = base_pitch = None

    hits = []
    neutral_run = 0
    miss = 0
    jumps = 0
    samples = 0
    action_seen = False
    prev_center = None
    best = None            # (代价, frame, face)：动作成立后最接近正对的一帧

    max_dy = min_dy = max_dp = min_dp = 0.0
    face_width = 0.0

    def make_diag():
        return {
            "elapsed": round(time.monotonic() - t0, 2),
            "samples": samples,
            "misses": miss,
            "jumps": jumps,
            "baseline_yaw": None if base_yaw is None else round(base_yaw, 4),
            "baseline_pitch": None if base_pitch is None else round(base_pitch, 4),
            "min_yaw_delta": round(min_dy, 4),
            "max_yaw_delta": round(max_dy, 4),
            "min_pitch_delta": round(min_dp, 4),
            "max_pitch_delta": round(max_dp, 4),
            "face_width": round(face_width, 1),
            "action_seen": action_seen,
        }

    while time.monotonic() - t0 < max_seconds:
        frame = engine.read_frame()
        if frame is None:
            miss += 1
            if miss >= MAX_MISS:
                return _result(False, action, "camera_read_failed", None, make_diag())
            sleep(SAMPLE_INTERVAL)
            continue

        face = engine.detect_largest(frame)
        if face is None:
            miss += 1
            if miss >= MAX_MISS:
                return _result(False, action, "no_face_detected", None, make_diag())
            sleep(SAMPLE_INTERVAL)
            continue
        miss = 0

        # 画面跳变检测：正常转头时人脸中心是连续移动的，被切换成另一路视频
        # / 换一张照片会表现为突然的整体位移。属启发式，只拦截明显的跳变。
        face_width = float(face[2]) or 1.0
        cx = float(face[0]) + face_width / 2.0
        cy = float(face[1]) + float(face[3]) / 2.0
        if prev_center is not None:
            if math.hypot(cx - prev_center[0], cy - prev_center[1]) / face_width > JUMP_RATIO:
                jumps += 1
                if jumps > MAX_JUMPS:
                    return _result(False, action, "face_changed", None, make_diag())
        prev_center = (cx, cy)

        yaw, pitch, _iod = pose_metrics(face)
        samples += 1

        if base_yaw is None:
            base_yaw_samples.append(yaw)
            base_pitch_samples.append(pitch)
            if len(base_yaw_samples) >= BASELINE_FRAMES:
                base_yaw = float(median(base_yaw_samples))
                base_pitch = float(median(base_pitch_samples))
            sleep(SAMPLE_INTERVAL)
            continue

        dy = yaw - base_yaw
        dp = pitch - base_pitch
        max_dy = max(max_dy, dy)
        min_dy = min(min_dy, dy)
        max_dp = max(max_dp, dp)
        min_dp = min(min_dp, dp)

        if not action_seen:
            hits.append(1 if satisfied(action, dy, dp) else 0)
            if len(hits) > HIT_WINDOW:
                hits.pop(0)
            if sum(hits) >= HIT_NEEDED:
                action_seen = True
                neutral_run = 0
        else:
            if abs(dy) <= NEUTRAL_YAW and abs(dp) <= NEUTRAL_PITCH:
                neutral_run += 1
                cost = abs(dy) + abs(dp)
                if best is None or cost < best[0]:
                    best = (cost, frame, face)
            else:
                neutral_run = 0
            if neutral_run >= NEUTRAL_FRAMES:
                break

        sleep(SAMPLE_INTERVAL)

    diag = make_diag()

    if samples < MIN_SAMPLES:
        return _result(False, action, "not_enough_samples", None, diag)
    if not action_seen:
        return _result(False, action, "liveness_timeout", None, diag)
    if neutral_run < NEUTRAL_FRAMES or best is None:
        return _result(False, action, "action_not_returned", None, diag)

    feature = engine.capture_from(best[1], best[2])
    if feature is None:
        return _result(False, action, "no_face_detected", None, diag)
    return _result(True, action, None, feature, diag)
