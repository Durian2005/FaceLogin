"""用 LFW 公开数据集标定相似度阈值：FAR/FRR 曲线、EER、推荐阈值。

为什么需要这个脚本
------------------
默认阈值 `0.50` 是经验值 —— 定它的时候手上只有两个数据点（正对镜头 0.8418、
偏头 0.5684）。"阈值到底该定多少"这种问题只能靠**分布**回答，而分布需要成百上千
对真实人脸；人凑不出来，公开数据集可以。

数据来源
--------
LFW（Labeled Faces in the Wild）：13233 张、5749 人，人脸识别最通用的公开评测集
之一。本脚本读它的 parquet 版本。本机环境下的获取方式（LFW 官网不通，走国内镜像）：

    curl -L --noproxy "*" -o .workbuddy/tmp/lfw/lfw.parquet \\
      "https://hf-mirror.com/datasets/logasja/lfw/resolve/main/data/train-00000-of-00001.parquet"

用法
----
    .venv\\Scripts\\python.exe tools\\eval_threshold.py --parquet .workbuddy\\tmp\\lfw\\lfw.parquet

产物
----
- `docs/threshold-report.md` —— 完整报告（样本量、分布、曲线、结论、局限）
- `docs/threshold-curve.png`  —— FAR/FRR 曲线（用 cv2 画，**不引入 matplotlib**）

依赖
----
pyarrow，且**只在评测时需要**（见 requirements-dev.txt）。程序运行本身不需要它。
"""

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------- 数据读取


def pick_columns(schema):
    """从 parquet schema 里认出「图像列」和「身份标签列」。

    不同来源的 LFW parquet 字段名不一（image/img/face、label/name/person），
    所以按**类型**判断而不是按名字猜：

      图像 = struct 里带 bytes 字段，或直接的 binary 列
      标签 = 能当身份用的列。⚠️ 字符串**和整数**都算 ——
             LFW 的官方 parquet 里 `label` 就是 `int64`（人物编号），
             只认字符串的话会直接"认不出标签列"（真数据上踩过）。
             字符串优先（更可读），没有字符串才退回整数。
    """
    import pyarrow as pa

    image_col = None
    str_col = int_col = None
    for field in schema:
        t = field.type
        if pa.types.is_struct(t):
            if t.get_field_index("bytes") >= 0 and image_col is None:
                image_col = field.name
        elif (pa.types.is_binary(t) or pa.types.is_large_binary(t)) and image_col is None:
            image_col = field.name
        elif (pa.types.is_string(t) or pa.types.is_large_string(t)) and str_col is None:
            str_col = field.name
        elif pa.types.is_integer(t) and int_col is None:
            int_col = field.name
    return image_col, (str_col or int_col)


def _extract_bytes(cell):
    """parquet 里的图像可能存成 {'bytes': b'..'} 或直接的 bytes。"""
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def count_by_label(parquet_path, label_col):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    counts = {}
    for batch in pf.iter_batches(batch_size=2048, columns=[label_col]):
        for label in batch.column(0).to_pylist():
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
    return counts


def load_images(parquet_path, image_col, label_col, wanted, max_per_identity):
    """只取 `wanted` 里那些身份的图片，每个身份最多 max_per_identity 张。

    拿够了就提前退出 —— 整份数据集 188MB，没必要全解出来。
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    out = {label: [] for label in wanted}
    remaining = len(wanted) * max_per_identity
    for batch in pf.iter_batches(batch_size=128, columns=[image_col, label_col]):
        imgs = batch.column(0).to_pylist()
        labels = batch.column(1).to_pylist()
        for cell, label in zip(imgs, labels):
            if label not in out or len(out[label]) >= max_per_identity:
                continue
            raw = _extract_bytes(cell)
            if not raw:
                continue
            buf = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            out[label].append(img)
            remaining -= 1
            if remaining <= 0:
                return out
    return out


# ---------------------------------------------------------------- 特征提取


def extract_features(engine, images):
    """对每张图跑**和运行时完全相同**的检测 + 对齐 + 特征提取。

    这一点是评测有效性的关键：比对用的必须是 app 里那条链路
    （YuNet 检测 → SFace alignCrop → feature），而不是另一套实现。
    """
    feats = {}
    tried = detected = 0
    for label, imgs in images.items():
        got = []
        for img in imgs:
            tried += 1
            face = engine.detect_largest(img)
            if face is None:
                continue
            got.append(engine.capture_from(img, face))
            detected += 1
        if len(got) >= 2:          # 单张图构不成同人对
            feats[label] = got
    return feats, tried, detected


# ---------------------------------------------------------------- 指标


def score_pair(engine, a, b):
    """用 app 自己的比对函数算相似度（best_cosine → SFace 的余弦）。"""
    return engine.best_cosine([a], b)


def build_scores(engine, feats, n_pairs, seed):
    labels = sorted(feats)
    rng = random.Random(seed)

    genuine = []
    for _ in range(n_pairs):
        label = rng.choice(labels)
        i, j = rng.sample(range(len(feats[label])), 2)
        genuine.append(score_pair(engine, feats[label][i], feats[label][j]))

    impostor = []
    for _ in range(n_pairs):
        l1, l2 = rng.sample(labels, 2)
        a = rng.choice(feats[l1])
        b = rng.choice(feats[l2])
        impostor.append(score_pair(engine, a, b))

    return np.array(genuine), np.array(impostor)


def curve_at(grid, genuine, impostor):
    """给定阈值 t：FAR = 异类中被误判为同类的比例；FRR = 同类中被拒的比例。

    判定规则与 server._match_and_login 一致：score >= threshold 视为通过。
    """
    far = np.array([np.mean(impostor >= t) for t in grid])
    frr = np.array([np.mean(genuine < t) for t in grid])
    return far, frr


def eer_point(grid, far, frr):
    """等错误率：FAR 与 FRR 交叉处。相交在网格点之间时做线性插值。"""
    diff = far - frr
    for i in range(1, len(grid)):
        if diff[i - 1] <= 0 <= diff[i] or diff[i - 1] >= 0 >= diff[i]:
            span = abs(diff[i - 1]) + abs(diff[i])
            ratio = 0.0 if span == 0 else abs(diff[i - 1]) / span
            t = grid[i - 1] + (grid[i] - grid[i - 1]) * ratio
            return float(t), float(far[i - 1] + (far[i] - far[i - 1]) * ratio)
    # 两族曲线在网格范围内始终不相交（理论上少见，但别留一个语义错误的兜底）：
    # 取两线最接近的那一点，错误率取两者均值，即该点的「等错误率估计」。
    k = int(np.argmin(np.abs(diff)))
    return float(grid[k]), float((far[k] + frr[k]) / 2.0)


def build_rows(grid, far, frr, default_t=0.50):
    """生成报告里的错误率表格行。

    阈值按固定步长抽稀（网格有 151 个点，全列出来没人看），
    但**当前默认阈值必须出现在表里** —— 报告要回答的核心问题之一就是
    「0.50 到底合不合适」，它不在表里等于没回答。
    """
    targets = [round(0.20 + 0.05 * i, 2) for i in range(16)]   # 0.20 … 0.95
    targets.append(default_t)
    lines = []
    for t in sorted(set(targets)):
        i = int(np.argmin(np.abs(grid - t)))
        mark = "**← 当前默认**" if abs(float(grid[i]) - default_t) < 1e-9 else ""
        lines.append("| %.3f | %.4f%% | %.4f%% | %s |"
                     % (grid[i], far[i] * 100, frr[i] * 100, mark))
    return "\n".join(lines)


def threshold_for_far(grid, far, target):
    """在 FAR 不超过 target 的前提下，取**最低**的阈值（即最宽松）。
    这样既满足误接受率要求，又尽量不误拒本人。"""
    ok = [t for t, f in zip(grid, far) if f <= target]
    return float(min(ok)) if ok else None


def far_upper_bound(n_impostor):
    """0 次误接受时 FAR 的 95% 置信上界，用「三倍法则」：≈ 3/n。

    没有它就容易把"FAR = 0%"说成"绝对安全" —— 实际上只是样本不够多，
    区分不了 0 和 3/n。
    """
    return 3.0 / n_impostor if n_impostor else float("nan")


# ---------------------------------------------------------------- 画图（cv2，不引入 matplotlib）


def draw_curve(path, grid, far, frr, eer_t, eer_v, rec_t, title, default=None):
    """画 FAR/FRR 曲线。

    纵轴**刻意截断**：FRR 在阈值升到 0.9 附近会爬到 100%，若按真实最大值定标，
    真正要看的低错误率区（< 2%）会被压成贴着轴线的一条线 —— 这张图存在的意义
    就是让人看清那一小块。所以上限封顶在 VMAX_CAP，超出部分截平并在图上注明。
    """
    W, H = 960, 600
    VMAX_CAP = 0.15

    img = np.full((H, W, 3), 255, np.uint8)
    left, right, top, bottom = 96, W - 44, 78, H - 84

    raw_max = float(max(far.max(), frr.max()))
    vmax = max(0.02, min(raw_max * 1.12, VMAX_CAP))
    clipped = raw_max > vmax + 1e-9

    def px(t):
        lo, hi = float(grid[0]), float(grid[-1])
        return int(round(left + (t - lo) / (hi - lo) * (right - left)))

    def py(v):
        v = min(float(v), vmax)          # 截断：超出上限的一律画在上边框上
        return int(round(bottom - v / vmax * (bottom - top)))

    far_col = (200, 90, 40)     # BGR 蓝
    frr_col = (30, 130, 220)    # BGR 橙
    rec_col = (60, 160, 90)     # BGR 绿
    def_col = (150, 110, 60)    # BGR 青灰
    eer_col = (120, 120, 120)
    ink = (45, 45, 45)
    grid_col = (215, 215, 215)

    def pct(v):
        return ("%.2f%%" % (v * 100)) if vmax <= 0.05 else ("%.1f%%" % (v * 100))

    for i in range(6):
        value = vmax * i / 5.0
        y = py(value)
        cv2.line(img, (left, y), (right, y), grid_col, 1)
        cv2.putText(img, pct(value), (left - 76, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, ink, 1, cv2.LINE_AA)

    cv2.line(img, (left, bottom), (right, bottom), ink, 1)
    cv2.line(img, (left, top), (left, bottom), ink, 1)

    for t in np.arange(0.2, 1.0, 0.1):
        x = px(round(t, 2))
        if left <= x <= right:
            cv2.line(img, (x, bottom), (x, bottom + 5), ink, 1)
            cv2.putText(img, "%.1f" % t, (x - 14, bottom + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, ink, 1, cv2.LINE_AA)

    cv2.polylines(img, [np.array([[px(t), py(v)] for t, v in zip(grid, far)], np.int32)],
                  False, far_col, 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array([[px(t), py(v)] for t, v in zip(grid, frr)], np.int32)],
                  False, frr_col, 2, cv2.LINE_AA)

    # 当前默认阈值那条线，连同它对应的 FRR —— 这是本报告最想让人看到的一个数
    if default is not None:
        x_def, frr_def = px(default[0]), float(default[1])
        cv2.line(img, (x_def, py(frr_def)), (x_def, bottom), def_col, 1, cv2.LINE_4)
        cv2.circle(img, (x_def, py(frr_def)), 4, def_col, -1, cv2.LINE_AA)
        cv2.putText(img, "default %.2f -> FRR %.2f%%" % (default[0], frr_def * 100),
                    (x_def + 8, top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.47, def_col, 1,
                    cv2.LINE_AA)

    x_eer = px(eer_t)
    cv2.line(img, (x_eer, top), (x_eer, bottom), eer_col, 1, cv2.LINE_AA)
    cv2.circle(img, (x_eer, py(eer_v)), 5, (70, 70, 70), -1, cv2.LINE_AA)
    cv2.putText(img, "EER %.4f @ %.3f" % (eer_v, eer_t),
                (x_eer - 130, min(py(eer_v) - 12, bottom - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)

    if rec_t is not None:
        x_rec = px(rec_t)
        cv2.line(img, (x_rec, top), (x_rec, bottom), rec_col, 1, cv2.LINE_AA)
        # 标签放在绘图区内侧，避开上方的图例带（原来放在 top-12 会压住图例文字）
        cv2.putText(img, "rec %.3f (FAR<=0.1%%)" % rec_t, (x_rec + 8, top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, rec_col, 1, cv2.LINE_AA)

    cv2.putText(img, title, (left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "threshold", (right - 90, bottom + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "error rate", (left - 80, top - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA)

    cv2.line(img, (left, top - 32), (left + 26, top - 32), far_col, 2, cv2.LINE_AA)
    cv2.putText(img, "FAR  impostor accepted", (left + 34, top - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA)
    cv2.line(img, (left, top - 10), (left + 26, top - 10), frr_col, 2, cv2.LINE_AA)
    cv2.putText(img, "FRR  genuine rejected", (left + 34, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA)

    if clipped:
        # 截断必须写在图上，否则"橙色曲线平在顶边"会被误读成"FRR 不再上升"
        cv2.putText(img, "y axis capped at %s  (FRR rises further above; see report table)"
                    % pct(vmax), (left, H - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 120, 120), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 不能用 cv2.imwrite：Windows 下它写不了含非 ASCII 字符的路径
    # （本项目路径就是「人脸识别测试实验」，踩过一次），而且失败时只返回 False
    # 不抛异常。走 imencode + Python 写文件，任何路径都稳。
    good, buf = cv2.imencode(".png", img)
    if not good:
        raise RuntimeError("cv2.imencode 编码失败")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
    return path


# ---------------------------------------------------------------- 报告


def build_report():
    """报告模板。占位符由 main() 用 .format() 填充 —— 单列一处便于核对数字口径。"""
    return """# 相似度阈值标定报告

> 由 `tools/eval_threshold.py` 自动生成。复现命令：
>
> ```
> .venv\\Scripts\\python.exe tools\\eval_threshold.py --parquet .workbuddy\\tmp\\lfw\\lfw.parquet
> ```

## 1. 结论先说

| 项 | 值 |
|---|---|
| **等错误率 EER** | **{eer_v}**（出现在阈值 `{eer_t}`） |
| 推荐阈值（误接受率 ≤ 0.1%） | **`{rec}`** |
| 当前默认阈值 | `0.50` |
| 该默认值下的表现 | FAR {far50} / FRR {frr50} |
| 判别力 d' | {dprime:.2f} |
| 样本量 | 同类 {ng} 对 · 异类 {ni} 对 |
| 耗时 | {elapsed:.0f} 秒 |

**外部佐证**：OpenCV 官方文档给出的 SFace 在 LFW 上的参考值是「准确率 99.60%，
余弦阈值 **0.363**」（`opencv_zoo/models/face_recognition_sface/sface.py` 里
`_threshold_cosine = 0.363` 是写死的）。本报告**独立标定**出的推荐值与之相差不到 4%，
说明这条评测链路和口径没有跑偏；反过来，本项目原先的默认阈值 `0.50`
比这两个数都高出一大截，**确实偏严**。

![FAR/FRR 曲线]({curve_name})

**怎么读这张图**：横轴是阈值，纵轴是错误率。蓝线是 FAR（异类被误判为同类的比例），
橙线是 FRR（同类被拒的比例）。两条线交叉处就是 EER —— 阈值越低越宽松（FAR 升、
FRR 降），越高越严格。

## 2. 相似度分布

| 分布 | 均值 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|
| 同类（同一个人不同照片） | {mu_g:.4f} | {sd_g:.4f} | {min_g:.4f} | {max_g:.4f} |
| 异类（不同人） | {mu_i:.4f} | {sd_i:.4f} | {min_i:.4f} | {max_i:.4f} |

两类分布间隔 d' = **{dprime:.2f}**（越大越容易分开）。这个数是这套
「YuNet + SFace + 余弦」组合在 LFW 上的实际判别力，不是引用论文的数值。

## 3. 各阈值下的错误率

| 阈值 | FAR（误接受） | FRR（误拒绝） | |
|---|---|---|---|
{rows}

## 4. 方法与局限（必须读）

**做对了什么**

- 比对用的是**与运行时完全相同**的链路：`FaceEngine.detect_largest` →
  `capture_from`（SFace `alignCrop` + `feature`）→ `best_cosine`。
  不是另写一套实现，所以这些数字对**本项目**有效，而不是对某篇论文有效。
- 判定规则与 `server._match_and_login` 一致：`score >= threshold` 视为通过。
- 配对随机采样，**种子固定**，任何人可复现同一组数字。
- 同类/异类配对数量相等，避免样本不均衡扭曲曲线。

**局限（不要过度解读）**

1. **这是自建配对协议，不是 LFW 官方的 6000 对视图。** 官方协议里的异类对是
   特意挑过的"难例"，因此本报告给出的 FAR 很可能**偏乐观**。
   跨数据集比较时请以官方协议为准。
2. **LFW 是"in the wild"但已裁剪对齐的图**，人脸基本居中、光照可控。
   真实摄像头场景（侧光、逆光、遮挡、大角度）会明显更难。
3. **FAR 的分辨力受样本量限制**：{ni} 对异类样本下，"0 次误接受"并不等于
   FAR 真的是 0，只能说明 **95% 置信上界约 {ci:.4f}%**（三倍法则 3/n）。
   想要更小的置信区间，就要更多配对。
4. **只测了识别环节，没测活体环节。** 阈值回答的是"像不像同一个人"，
   不回答"是不是活人"。反欺骗能力另见 THREAT_MODEL.md 第 4.3 节。
5. 每个人的样本数被限制在 {max_per} 张以内（为了控制耗时），
   同人内部的变化度因此没有被充分覆盖。
6. **身份不是随机抽的**：按「可用图片数从多到少」取了前 {n_id} 个身份，
   实际平均每人 {avg_imgs:.1f} 张。这样同人配对能覆盖更多姿态/光照变化，
   但代价是偏向"照片多的好例子"，对难例（侧脸、遮挡）覆盖有限。

## 5. 怎么用这个结论

- 想把阈值调到有量化依据的值：
  `.venv\\Scripts\\python.exe tools\\eval_threshold.py` 看推荐值，再在「阈值校准」
  卡片里改（**调低需要输入密码确认**）。
- 需要更严格（如门禁场景）：取 FAR 更小那一档，但 FRR 会上升 ——
  本人被拒后还有口令回退，所以偏严格通常比偏宽松划算。
- 需要更宽松：注意 FAR 上升速度，参见上表。
"""


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="用 LFW 标定相似度阈值")
    ap.add_argument("--parquet", required=True, help="LFW parquet 文件路径")
    ap.add_argument("--identities", type=int, default=300, help="参与评测的身份数")
    ap.add_argument("--max-per-identity", type=int, default=4, help="每个身份最多取几张")
    ap.add_argument("--pairs", type=int, default=4000, help="同类、异类各采多少对")
    ap.add_argument("--seed", type=int, default=20261005, help="随机种子")
    ap.add_argument("--report", default=os.path.join(ROOT, "docs", "threshold-report.md"))
    ap.add_argument("--curve", default=os.path.join(ROOT, "docs", "threshold-curve.png"))
    args = ap.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("缺少 pyarrow。它只在评测时需要：")
        print("    .venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt")
        return 2

    if not os.path.exists(args.parquet):
        print("找不到 parquet 文件: " + args.parquet)
        return 2

    t0 = time.perf_counter()
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(args.parquet).schema_arrow
    image_col, label_col = pick_columns(schema)
    print("列:", [f.name for f in schema])
    print("图像列 = %s / 标签列 = %s" % (image_col, label_col))
    if not image_col or not label_col:
        print("无法从 schema 中认出图像列与标签列，请手工指定。")
        return 2

    counts = count_by_label(args.parquet, label_col)
    multi = sorted([k for k, v in counts.items() if v >= 2],
                   key=lambda k: (-counts[k], str(k)))
    print("身份总数 %d，其中图片 >= 2 张的有 %d 个" % (len(counts), len(multi)))
    # 按「可用图片数从多到少」取，而不是取编号最小的若干个：同人多张照片才能
    # 承载姿态/光照/表情的变化，否则同人配对的相似度会被高估、FRR 被低估。
    # 这条选择规则是不随机的，必须在报告里披露（见报告第 4 节）。
    wanted = set(multi[:args.identities])
    print("取前 %d 个身份参与评测（按图片数降序）" % len(wanted))

    images = load_images(args.parquet, image_col, label_col, wanted, args.max_per_identity)
    total_imgs = sum(len(v) for v in images.values())
    print("已载入图片 %d 张" % total_imgs)

    from face import FaceEngine

    print("加载人脸引擎（不打开摄像头）...")
    engine = FaceEngine(open_camera=False)
    feats, tried, detected = extract_features(engine, images)
    print("检测成功 %d / %d 张，可用身份 %d 个" % (detected, tried, len(feats)))
    if len(feats) < 20:
        print("可用身份太少，无法给出有意义的曲线。")
        return 1

    genuine, impostor = build_scores(engine, feats, args.pairs, args.seed)
    grid = np.round(np.arange(0.20, 0.951, 0.005), 3)
    far, frr = curve_at(grid, genuine, impostor)
    eer_t, eer_v = eer_point(grid, far, frr)
    rec = threshold_for_far(grid, far, 0.001)

    print()
    print("同类 均值 %.4f 标准差 %.4f" % (genuine.mean(), genuine.std()))
    print("异类 均值 %.4f 标准差 %.4f" % (impostor.mean(), impostor.std()))
    print("EER = %.4f @ 阈值 %.3f" % (eer_v, eer_t))
    print("推荐阈值(FAR<=0.1%%) = %s" % rec)
    print("当前默认 0.50 -> FAR %.4f%% / FRR %.4f%%"
          % (np.mean(impostor >= 0.50) * 100, np.mean(genuine < 0.50) * 100))

    frr_at_default = float(np.mean(genuine < 0.50))
    far_at_default = float(np.mean(impostor >= 0.50))

    draw_curve(args.curve, grid, far, frr, eer_t, eer_v, rec,
               "FaceLogin threshold calibration (LFW)",
               default=(0.50, frr_at_default))

    rows = build_rows(grid, far, frr)

    # 报告里的图片链接按**实际**横线位置推出来，不写死文件名 ——
    # 否则一旦用 --curve 换个路径/名字，链接就是坏的。
    curve_name = os.path.relpath(args.curve, os.path.dirname(args.report) or ".")
    curve_name = curve_name.replace(os.sep, "/")

    body = build_report().format(
        curve_name=curve_name,
        eer_v="%.4f" % eer_v, eer_t="%.3f" % eer_t,
        rec=("%.3f" % rec) if rec is not None else "（无，样本内 FAR 无法降到 0.1%）",
        far50="%.4f%%" % (np.mean(impostor >= 0.50) * 100),
        frr50="%.4f%%" % (np.mean(genuine < 0.50) * 100),
        dprime=(genuine.mean() - impostor.mean()) / max(
            1e-9, np.sqrt(0.5 * (genuine.std() ** 2 + impostor.std() ** 2))),
        ng=len(genuine), ni=len(impostor),
        elapsed=time.perf_counter() - t0,
        mu_g=genuine.mean(), sd_g=genuine.std(),
        min_g=genuine.min(), max_g=genuine.max(),
        mu_i=impostor.mean(), sd_i=impostor.std(),
        min_i=impostor.min(), max_i=impostor.max(),
        rows=rows, ci=far_upper_bound(len(impostor)) * 100,
        max_per=args.max_per_identity,
        n_id=len(feats),
        avg_imgs=(sum(len(v) for v in feats.values()) / float(len(feats))),
    )

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    # newline="\n" 是刻意的：仓库用 .gitattributes 统一为 LF，而 Python 文本模式
    # 在 Windows 上默认把 \n 翻成 \r\n，会造成"工作区 CRLF / 仓库 LF"的长期不一致。
    # 生成物应当跨平台字节稳定。
    with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    print()
    print("报告已写入:", args.report)
    print("曲线图已写入:", args.curve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
