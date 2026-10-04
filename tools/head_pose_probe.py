"""头姿实时探针：用真人动作核对 yaw / pitch 的符号方向，并顺带校准阈值。

背景：活体检测靠「鼻尖相对双眼中点的偏移」估 yaw、「鼻尖在眼线→嘴线上的相对高度」
估 pitch。数学推导给出的方向约定是：

    转向自己的**左侧** -> yaw 相对基线**增大**
    转向自己的**右侧** -> yaw 相对基线**减小**
    抬头               -> pitch 相对基线**减小**
    低头               -> pitch 相对基线**增大**

但推导毕竟只是推导，所以这个工具让你真做一遍动作，直接看数值怎么变。
它同时给出每个动作实测到的最大偏移量，可用来判断现有阈值是否合适。

用法（面对着摄像头坐正，跟着提示做动作）：
    .venv\\Scripts\\python.exe tools\\head_pose_probe.py

不保存任何图像，只打印数值。
"""

import os
import sys
import time

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))

import liveness  # noqa: E402
from face import FaceEngine, pose_metrics  # noqa: E402

BASELINE_SECONDS = 3.0
SEGMENT_SECONDS = 6.0
SAMPLE_INTERVAL = 0.07


def out(text=""):
    """控制台编码兜底：GBK 终端下打印中文可能抛异常，退化成替换字符而不是崩掉。"""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


SEGMENTS = [
    ("left", "把头转向你自己的左侧  (turn to YOUR LEFT)", "yaw 应增大 / yaw should INCREASE"),
    ("right", "把头转向你自己的右侧  (turn to YOUR RIGHT)", "yaw 应减小 / yaw should DECREASE"),
    ("up", "把头抬起向上看        (look UP)", "pitch 应减小 / pitch should DECREASE"),
    ("down", "把头低下向下看        (look DOWN)", "pitch 应增大 / pitch should INCREASE"),
]


def collect(engine, seconds, base_yaw=None, base_pitch=None, on_sample=None):
    """采样若干秒，返回 (yaw 列表, pitch 列表, 有效帧数, 丢帧数)。"""
    ys, ps = [], []
    miss = 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        frame = engine.read_frame()
        if frame is None:
            miss += 1
            time.sleep(SAMPLE_INTERVAL)
            continue
        face = engine.detect_largest(frame)
        if face is None:
            miss += 1
            time.sleep(SAMPLE_INTERVAL)
            continue
        yaw, pitch, _iod = pose_metrics(face)
        ys.append(yaw)
        ps.append(pitch)
        if on_sample:
            on_sample(yaw, pitch, base_yaw, base_pitch)
        time.sleep(SAMPLE_INTERVAL)
    return ys, ps, len(ys), miss


def main():
    engine = FaceEngine()
    if not engine.camera_ready():
        out("摄像头打不开：可能被其它程序占用。")
        return 1

    out("=" * 70)
    out("头姿实时探针 —— 用于核对动作方向与阈值")
    out("=" * 70)
    out("第一步：请正对摄像头坐好，保持不动 %.0f 秒，正在测基线…" % BASELINE_SECONDS)
    out()

    ys, ps, ok, miss = collect(engine, BASELINE_SECONDS)
    if ok < 5:
        out("基线阶段只采到 %d 帧（丢失 %d 帧），请确认人脸在画面中、光线足够。" % (ok, miss))
        return 1

    ys.sort()
    ps.sort()
    base_yaw = ys[len(ys) // 2]
    base_pitch = ps[len(ps) // 2]
    out("基线完成：yaw=%.4f  pitch=%.4f   （有效帧 %d，丢失 %d）" % (base_yaw, base_pitch, ok, miss))
    out()
    out("接下来依次让你做 4 个动作，每个 %.0f 秒。屏幕上会实时显示偏移量。" % SEGMENT_SECONDS)
    out()

    results = []
    for key, prompt, expect in SEGMENTS:
        out("-" * 70)
        out("请准备：%s" % prompt)
        out("预期：%s" % expect)
        out("开始 —— 请缓慢、明显地做这个动作，并保持住")
        out()

        extreme = {"yaw": 0.0, "pitch": 0.0}
        last = [0.0]

        def on_sample(yaw, pitch, by, bp, extreme=extreme, last=last):
            dy = yaw - by
            dp = pitch - bp
            if abs(dy) > abs(extreme["yaw"]):
                extreme["yaw"] = dy
            if abs(dp) > abs(extreme["pitch"]):
                extreme["pitch"] = dp
            now = time.monotonic()
            if now - last[0] >= 0.5:
                last[0] = now
                out("    yaw %+.3f    pitch %+.3f" % (dy, dp))

        ys, ps, ok, miss = collect(engine, SEGMENT_SECONDS, base_yaw, base_pitch, on_sample)
        dy, dp = extreme["yaw"], extreme["pitch"]

        if key == "left":
            got = dy >= liveness.YAW_TURN
            value = dy
            need = liveness.YAW_TURN
        elif key == "right":
            got = -dy >= liveness.YAW_TURN
            value = -dy
            need = liveness.YAW_TURN
        elif key == "up":
            got = -dp >= liveness.PITCH_TILT
            value = -dp
            need = liveness.PITCH_TILT
        else:
            got = dp >= liveness.PITCH_TILT
            value = dp
            need = liveness.PITCH_TILT

        out()
        out("    实测极值：yaw 偏移 %+.3f ，pitch 偏移 %+.3f" % (dy, dp))
        out("    本次动作判定：%s（方向强度 %.3f，阈值 %.3f）"
            % ("通过 OK" if got else "不通过", value, need))
        out()
        results.append((key, got, value, need))
        time.sleep(0.4)

    out("=" * 70)
    out("汇总")
    out("=" * 70)
    all_ok = True
    for key, got, value, need in results:
        all_ok &= got
        out("  %-6s %-4s  方向强度 %+.3f / 阈值 %.3f"
            % (key, "OK" if got else "FAIL", value, need))
    out()
    if all_ok:
        out("四个动作全部按预期方向识别 —— 符号约定与阈值都成立。")
    else:
        out("有动作没有按预期方向识别。若四个动作**全部反向**，说明符号约定需要翻转：")
        out("  请把 app/liveness.py 里 satisfied() 的 left / right 与 up / down 判定互换。")
        out("若只是某个动作强度不够，说明阈值偏高或动作幅度太小，可先调大动作幅度再试。")
    out()
    out("提示：上述结论只说明『方向约定对不对』。真机能否稳定通过活体登录，")
    out("      还要在页面上实际登录一次确认。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        out()
        out("已中断。")
        sys.exit(130)
