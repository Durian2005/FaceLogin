"""跑穿 eval_threshold.py 的 main()，验证"报告能真的生成出来"。

为什么需要这一层
----------------
只测纯函数是不够的。曾经出过一次：`build_rows()` 的结果在 main() 里被用了，
但那个局部变量**根本忘了赋值** —— 所有纯函数测试全绿，因为测试是手工把
`rows=...` 传给 `.format()` 的，恰好绕过了出错的那一行。纯函数测试测不出
"编排层忘了接线"，必须真的跑一遍 main()。

人脸引擎在这里是**桩**：不为识别精度负责，只负责让 main() 的整条管线
（读 parquet → 提特征 → 配对打分 → 算曲线 → 画图 → 填模板 → 落盘）走完。
识别精度由 LFW 上的真实运行负责。

    .venv\\Scripts\\python.exe tools\\test_eval_report.py
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOOL = os.path.join(ROOT, "tools", "eval_threshold.py")

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print("  PASS  " + name + (("  " + extra) if extra else ""))
    else:
        fail += 1
        print("  FAIL  " + name + (("  " + extra) if extra else ""))


# ---------------------------------------------------------------- 人脸引擎桩
#
# 约定：合成图用「填充色」编码身份，颜色差别大 = 不同人。
# 同人的不同照片在填充色上叠一点随机噪声，于是同人相似度高、异人相似度低 ——
# 足以让 FAR/FRR 两条曲线在一个像样的阈值范围内相交。


class _StubEngine(object):
    def __init__(self, open_camera=True):
        self.open_camera = open_camera
        self.cap = None

    def detect_largest(self, frame):
        return np.zeros(15, np.float32) if frame is not None else None

    def capture_from(self, frame, face):
        med = np.median(frame.reshape(-1, 3).astype(np.float32), axis=0)
        v = med / (np.linalg.norm(med) + 1e-9)
        return v.astype(np.float32)

    def best_cosine(self, refs, feat):
        return float(np.max([np.dot(r, feat) for r in refs]))


def _install_stub():
    mod = types.ModuleType("face")
    mod.FaceEngine = _StubEngine
    sys.modules["face"] = mod
    return mod


# ---------------------------------------------------------------- 合成 parquet

def make_parquet(path, n_identities=40, per_identity=3, seed=20261005):
    """构造一份"长得像 LFW 官方 parquet"的文件。

    schema 刻意照抄真数据：`label` 是 **int64**（人物编号），`image` 是
    `struct<bytes, path>`。第一版合成数据用的是字符串标签，于是"标签列是整数"
    这个真实形态没被测到 —— 上真数据时才发现认不出来。真实形态必须进测试。
    """
    rng = np.random.default_rng(seed)
    blobs, labels, paths = [], [], []
    for i in range(n_identities):
        base = np.array([(i * 17) % 200 + 28,
                         (i * 53) % 200 + 28,
                         (i * 97) % 200 + 28], np.float32)
        name = "person_%03d" % i
        for j in range(per_identity):
            color = np.clip(base + rng.normal(0, 22, 3), 0, 255)
            img = np.tile(color.astype(np.uint8), (72, 72, 1))
            good, buf = cv2.imencode(".jpg", img)
            assert good
            blobs.append(buf.tobytes())
            labels.append(i)
            paths.append("%s/%s_%04d.jpg" % (name, name, j))
    struct_col = pa.StructArray.from_arrays(
        [pa.array(blobs, type=pa.binary()), pa.array(paths)],
        names=["bytes", "path"])
    pq.write_table(pa.table({"label": pa.array(labels, type=pa.int64()),
                             "image": struct_col}), path)
    return n_identities * per_identity


# ---------------------------------------------------------------- 主流程

def main():
    TMP = os.path.join(ROOT, ".workbuddy", "tmp")
    os.makedirs(TMP, exist_ok=True)
    work = tempfile.mkdtemp(prefix="evalreport_", dir=TMP)
    try:
        parq = os.path.join(work, "synth.parquet")
        report = os.path.join(work, "report.md")
        curve = os.path.join(work, "curve.png")

        n = make_parquet(parq)
        print("合成 parquet：%d 张图" % n)

        spec = importlib.util.spec_from_file_location("eval_threshold", TOOL)
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)

        # ---------------------------------------------------- [0] 列识别
        print("\n[0] parquet 列识别（含 LFW 真实形态）")
        schema = pq.ParquetFile(parq).schema_arrow
        print("  合成 schema:", [(f.name, str(f.type)) for f in schema])
        image_col, label_col = ev.pick_columns(schema)
        check("认得出 struct{bytes,path} 图像列", image_col == "image", str(image_col))
        check("认得出 int64 标签列（LFW 官方 parquet 就是这种）",
              label_col == "label", str(label_col))

        # 字符串标签优先于整数（人名比编号可读）
        strq = os.path.join(work, "strlab.parquet")
        pq.write_table(pa.table({"name": pa.array(["alice", "bob"]),
                                 "num": pa.array([1, 2], type=pa.int64())}), strq)
        _i, l_s = ev.pick_columns(pq.ParquetFile(strq).schema_arrow)
        check("字符串与整数并存时优先取字符串", l_s == "name", str(l_s))

        # 纯 binary 图像列的形态也要认得
        binq = os.path.join(work, "binimg.parquet")
        pq.write_table(pa.table({"face": pa.array([b"x", b"y"], type=pa.binary()),
                                 "label": pa.array([0, 0], type=pa.int64())}), binq)
        i_b, l_b = ev.pick_columns(pq.ParquetFile(binq).schema_arrow)
        check("也认得出纯 binary 图像列", (i_b, l_b) == ("face", "label"),
              "%s/%s" % (i_b, l_b))

        # ---------------------------------------------------- [1] build_rows
        print("\n[1] 报告表格行")
        grid = np.round(np.arange(0.20, 0.951, 0.005), 3)
        rr = np.linspace(0.9, 0.0, len(grid))
        rows = ev.build_rows(grid, rr, 1.0 - rr)
        lines = [ln for ln in rows.splitlines() if ln.strip()]
        check("表格行数抽稀过（不是 151 行）", 10 <= len(lines) <= 25, "%d 行" % len(lines))
        check("当前默认阈值 0.500 在表内", "| 0.500 |" in rows)
        check("默认阈值被标出来", "当前默认" in rows)
        check("每行都是合法 markdown 表格行",
              all(ln.startswith("|") and ln.endswith("|") for ln in lines))

        # 默认阈值恰好落在采样目标里 —— 若哪天改了采样步长导致 0.50 被跳过，
        # 报告会悄悄少掉"0.50 表现如何"这个核心答案，这里拦住。
        check("0.500 是网格点之一",
              bool(np.any(np.abs(grid - 0.50) < 1e-9)))

        # ---------------------------------------------------- [2] eer 兜底
        print("\n[2] EER 两曲线不相交时的兜底")
        no_cross_far = np.full(len(grid), 0.30)
        no_cross_frr = np.full(len(grid), 0.10)
        t_fb, v_fb = ev.eer_point(grid, no_cross_far, no_cross_frr)
        check("兜底也给出 0.2~0.95 内的阈值", 0.20 <= t_fb <= 0.95, "t=%.3f" % t_fb)
        check("兜底错误率是错误率量级（不是 |FAR-FRR| 那种差值）",
              0.05 <= v_fb <= 0.45, "EER=%.3f" % v_fb)

        # ---------------------------------------------------- [3] main() 端到端
        print("\n[3] main() 端到端")
        _install_stub()
        argv_backup = sys.argv[:]
        sys.argv = ["eval_threshold.py", "--parquet", parq,
                    "--identities", "40", "--max-per-identity", "3",
                    "--pairs", "300", "--report", report, "--curve", curve]
        try:
            rc = ev.main()
        finally:
            sys.argv = argv_backup

        check("main() 返回 0", rc == 0, "rc=%s" % rc)
        check("报告文件已生成", os.path.exists(report))
        check("曲线图已生成", os.path.exists(curve))

        body = ""
        if os.path.exists(report):
            with open(report, "r", encoding="utf-8") as fh:
                body = fh.read()

        check("五个小节齐全",
              all(k in body for k in ["## 1. 结论先说", "## 2. 相似度分布",
                                      "## 3. 各阈值下的错误率", "## 4. 方法与局限",
                                      "## 5. 怎么用这个结论"]))
        # 上一版就是因为 rows 没赋值而在这里炸掉，且纯函数测试全绿。
        check("没有残留未填充的占位符",
              not any(p in body for p in ["{eer_v}", "{rec}", "{rows}", "{dprime", "{ci"]))
        check("表格里有真实数据行（rows 确实被接线了）", "| 0.500 |" in body)
        # 图片链接必须指向**实际**的曲线文件名，写死 threshold-curve.png 会在
        # 换路径时变成坏链。
        check("曲线图链接指向真实文件名", "(curve.png)" in body,
              "写了 curve.png 之外的链接" if "(curve.png)" not in body else "")
        check("写明了自建协议这条局限", "自建配对协议" in body)
        check("写明了 FAR 置信上界", "置信上界" in body)

        img = None
        if os.path.exists(curve):
            img = cv2.imdecode(np.fromfile(curve, dtype=np.uint8), cv2.IMREAD_COLOR)
        check("曲线图可解码且尺寸正确",
              img is not None and img.shape[:2] == (600, 960),
              str(None if img is None else img.shape))

        print()
        print("通过 %d 项，失败 %d 项" % (ok, fail))
        return 1 if fail else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
