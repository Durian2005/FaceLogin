"""活体检测自测：姿态数学 + 挑战票据 + 采样状态机。

**完全离线、确定性**：不打开摄像头、不起服务、不需要采集真人样本。
用合成关键点喂进判定逻辑，用脚本化的假引擎驱动状态机。
真机表现（真做出动作能否通过）需要真人配合，见 tools/head_pose_probe.py。

跑法：
    .venv\\Scripts\\python.exe tools\\test_liveness.py
"""

import os
import sys
import time

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))

import liveness  # noqa: E402
from face import landmarks, pose_metrics  # noqa: E402

PASSED = 0
FAILED = 0


def check(name, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("  [OK]   %s" % name)
    else:
        FAILED += 1
        print("  [FAIL] %s %s" % (name, extra))


# ---------------------------------------------------------------- 合成人脸

def synth_face(yaw=0.0, pitch=0.5, cx=320.0, cy=240.0, w=130.0, h=170.0):
    """按 pose_metrics 的定义反向构造一张 15 元素的检测结果。

    yaw   = (鼻尖x - 双眼中点x) / 瞳距       瞳距固定 60
    pitch = (鼻尖y - 双眼中点y) / (嘴中点y - 双眼中点y)   跨度固定 50
    """
    iod, span = 60.0, 50.0
    eye_y = cy - 40.0
    mouth_y = eye_y + span
    eye_mid_x = cx
    nose_x = eye_mid_x + yaw * iod
    nose_y = eye_y + pitch * span
    return [
        cx - w / 2, cy - h / 2, w, h,
        eye_mid_x - iod / 2, eye_y,          # 眼睛 A
        eye_mid_x + iod / 2, eye_y,          # 眼睛 B
        nose_x, nose_y,                      # 鼻尖
        eye_mid_x - 25.0, mouth_y,           # 嘴角 A
        eye_mid_x + 25.0, mouth_y,           # 嘴角 B
        0.95,
    ]


print("=" * 68)
print("一、姿态代理值的数学")
print("=" * 68)

pts = landmarks(synth_face())
check("landmarks 形状为 (5,2)", pts.shape == (5, 2), str(pts.shape))

yaw, pitch, iod = pose_metrics(synth_face(yaw=0.0, pitch=0.5))
check("正视: yaw == 0", abs(yaw) < 1e-9, "yaw=%r" % yaw)
check("正视: pitch == 0.5", abs(pitch - 0.5) < 1e-9, "pitch=%r" % pitch)
check("正视: 瞳距 == 60", abs(iod - 60.0) < 1e-9, "iod=%r" % iod)

y, _, _ = pose_metrics(synth_face(yaw=0.25))
check("鼻尖右移 0.25 瞳距 -> yaw = +0.25", abs(y - 0.25) < 1e-9, "yaw=%r" % y)
y, _, _ = pose_metrics(synth_face(yaw=-0.25))
check("鼻尖左移 -> yaw 为负", abs(y + 0.25) < 1e-9, "yaw=%r" % y)

_, pi, _ = pose_metrics(synth_face(pitch=0.30))
check("鼻尖更靠眼线 -> pitch 变小", abs(pi - 0.30) < 1e-9, "pitch=%r" % pi)
_, pi, _ = pose_metrics(synth_face(pitch=0.70))
check("鼻尖更靠嘴线 -> pitch 变大", abs(pi - 0.70) < 1e-9, "pitch=%r" % pi)

# yaw 与 iod 的比值关系不受缩放影响（同一张脸拉远拉近结果一致）
ya, pa, ia = pose_metrics(synth_face(yaw=0.2, pitch=0.6, cx=200, cy=150, w=65, h=85))
check("同一姿态换尺寸后 yaw 不变", abs(ya - 0.2) < 1e-9, "yaw=%r" % ya)
check("同一姿态换尺寸后 pitch 不变", abs(pa - 0.6) < 1e-9, "pitch=%r" % pa)

print()
print("=" * 68)
print("二、动作判定 satisfied()")
print("=" * 68)

Y, P = liveness.YAW_TURN, liveness.PITCH_TILT
check("left  : +阈值 命中", liveness.satisfied("left", Y, 0.0) is True)
check("left  : 阈值之下不命中", liveness.satisfied("left", Y - 0.01, 0.0) is False)
check("left  : 反方向不命中", liveness.satisfied("left", -Y, 0.0) is False)
check("right : -阈值 命中", liveness.satisfied("right", -Y, 0.0) is True)
check("right : 正方向不命中", liveness.satisfied("right", Y, 0.0) is False)
check("up    : pitch 减小命中", liveness.satisfied("up", 0.0, -P) is True)
check("up    : pitch 增大不命中", liveness.satisfied("up", 0.0, P) is False)
check("down  : pitch 增大命中", liveness.satisfied("down", 0.0, P) is True)
check("down  : pitch 减小不命中", liveness.satisfied("down", 0.0, -P) is False)
check("yaw 大也不能冒充 up/down", not liveness.satisfied("up", 0.9, 0.0)
      and not liveness.satisfied("down", 0.9, 0.0))
check("pitch 大也不能冒充 left/right", not liveness.satisfied("left", 0.0, 0.9)
      and not liveness.satisfied("right", 0.0, 0.9))

print()
print("=" * 68)
print("三、挑战票据：一次性 / 过期 / 会话绑定 / 随机性")
print("=" * 68)

tok, act = liveness.new_challenge("sid-A")
check("返回的 token 非空且够长", isinstance(tok, str) and len(tok) >= 20, repr(tok))
check("动作在合法集合内", act in liveness.ACTIONS, act)

item, why = liveness.consume(tok, "sid-A")
check("正确会话可取出", item is not None and item["action"] == act, str(why))
item2, why2 = liveness.consume(tok, "sid-A")
check("同一票据不能用第二次", item2 is None and why2 == "unknown_challenge", str(why2))

tok2, _ = liveness.new_challenge("sid-A")
_, why3 = liveness.consume(tok2, "sid-B")
check("换会话取出会被拒", why3 == "challenge_mismatch", str(why3))
_, why4 = liveness.consume(tok2, "sid-A")
check("被拒后票据即作废（不能换会话重试）", why4 == "unknown_challenge", str(why4))

tok3, _ = liveness.new_challenge("sid-A")
_saved = liveness.CHALLENGE_TTL
liveness.CHALLENGE_TTL = -1.0          # 强制判定为已过期
_, why5 = liveness.consume(tok3, "sid-A")
liveness.CHALLENGE_TTL = _saved
check("过期票据被拒", why5 == "challenge_expired", str(why5))

seen = set()
for _ in range(400):
    _, a = liveness.new_challenge("sid-R")
    seen.add(a)
check("四种动作都能随机到", seen == set(liveness.ACTIONS), str(sorted(seen)))

for _ in range(50):
    liveness.new_challenge("sid-B")
check("票据数量有上限保护", liveness.pending_count() <= liveness.MAX_PENDING,
      str(liveness.pending_count()))


# ---------------------------------------------------------------- 假引擎

class FakeEngine:
    """按脚本逐帧返回关键点；脚本用完后重复最后一帧。"""

    def __init__(self, script, missing=0, jump=None, width=130.0):
        self.script = list(script)
        self.i = 0
        self.missing = missing
        self.jump = jump
        self.width = width
        self.calls = []
        self.capture_args = []
        self._flip = False

    def read_frame(self):
        return "frame-%d" % self.i

    def detect_largest(self, frame):
        if self.i < self.missing:
            self.i += 1
            return None
        idx = min(self.i - self.missing, len(self.script) - 1)
        self.i += 1
        spec = self.script[max(idx, 0)]
        cx = 320.0
        if self.jump:
            self._flip = not self._flip
            cx = 320.0 + (self.jump if self._flip else -self.jump) / 2.0
        return synth_face(yaw=spec[0], pitch=spec[1], cx=cx, w=self.width)

    def capture_from(self, frame, face):
        self.capture_args.append((frame, face))
        return "FEATURE"


N = 60          # 够长的中性序列


def run(script, action, **kw):
    eng = FakeEngine(script, **kw)
    res = liveness.run_challenge(eng, action, max_seconds=0.4, sleep=lambda s: None)
    return eng, res


print()
print("=" * 68)
print("四、采样状态机")
print("=" * 68)

neutral = [(0.0, 0.52)] * N

# 1) 正常完成：转向自己左侧，然后回正
eng, res = run(neutral[:6] + [(0.35, 0.52)] * 8 + neutral, "left")
check("left 成功：passed", res["passed"] is True, str(res["error"]))
check("left 成功：error 为空", res["error"] is None, str(res["error"]))
check("left 成功：拿到特征", res["feature"] == "FEATURE")
check("left 成功：max_yaw_delta 达标", res["diag"]["max_yaw_delta"] >= liveness.YAW_TURN,
      str(res["diag"]["max_yaw_delta"]))
used_frame = eng.capture_args[0][0] if eng.capture_args else None
used_idx = int(used_frame.split("-")[1]) if used_frame else -1
check("left 成功：用于比对的帧取自『动作之后』", used_idx >= 14,
      "frame=%s" % used_frame)

# 2) 反方向：要左转却右转
_, res = run(neutral[:6] + [(-0.35, 0.52)] * 8 + neutral, "left")
check("要左转却右转 -> 超时", res["passed"] is False
      and res["error"] == "liveness_timeout", str(res["error"]))

# 3) 完全不动
_, res = run(neutral, "left")
check("完全不动 -> 超时", res["error"] == "liveness_timeout", str(res["error"]))

# 4) 四个方向各自成功
for act, spec in [("right", (-0.35, 0.52)), ("up", (0.0, 0.32)), ("down", (0.0, 0.72))]:
    _, res = run(neutral[:6] + [spec] * 8 + neutral, act)
    check("%s 成功" % act, res["passed"] is True, str(res["error"]))

# 5) 动作幅度不够（只有阈值的一半）
_, res = run(neutral[:6] + [(liveness.YAW_TURN / 2, 0.52)] * 8 + neutral, "left")
check("动作幅度不足 -> 不通过", res["passed"] is False, str(res["error"]))

# 6) 做完动作但不回正
_, res = run(neutral[:6] + [(0.35, 0.52)] * N, "left")
check("不回正 -> action_not_returned",
      res["error"] == "action_not_returned", str(res["error"]))
check("不回正时不产生特征", res["feature"] is None)

# 7) 全程没有人脸
_, res = run(neutral, "left", missing=10 ** 6)
check("无人脸 -> no_face_detected", res["error"] == "no_face_detected", str(res["error"]))

# 8) 画面位置突变（模拟被切换成另一路视频）
_, res = run(neutral, "left", jump=500.0)
check("人脸位置突变 -> face_changed", res["error"] == "face_changed", str(res["error"]))

# 9) 有效帧太少
eng2 = FakeEngine(neutral, missing=3)
res = liveness.run_challenge(eng2, "left", max_seconds=0.001, sleep=lambda s: None)
check("有效帧不足 -> not_enough_samples",
      res["error"] in ("not_enough_samples", "liveness_timeout"), str(res["error"]))

# 10) 诊断文本可生成
_, res = run(neutral[:6] + [(0.35, 0.52)] * 8 + neutral, "left")
txt = liveness.diag_text(res["diag"])
check("diag_text 可生成且含关键量", "yaw" in txt and "pitch" in txt and "s" in txt, txt)

print()
print("=" * 68)
print("结果：%d 项通过，%d 项失败" % (PASSED, FAILED))
print("=" * 68)
sys.exit(1 if FAILED else 0)
