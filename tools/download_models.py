"""下载人脸模型与许可证到 models/ 目录，并校验 SHA-256。

为什么需要这个脚本
------------------
OpenCV Zoo 的模型文件由 **Git LFS** 托管。直接请求
`https://raw.githubusercontent.com/...` 只会得到 131 字节的 LFS 指针文件
（内容形如 `version https://git-lfs.github.com/spec/v1`），而不是模型本体。
必须走 `media.githubusercontent.com/media/...` 这条 LFS 直链。

用法
----
    python tools/download_models.py              # 缺失或校验不符时才下载
    python tools/download_models.py --force      # 无条件重新下载
    python tools/download_models.py --no-proxy   # 忽略系统代理设置

代理说明：脚本默认沿用系统的 HTTP(S) 代理环境变量。若你的代理已失效
（环境里残留了 http_proxy 但端口不通），会卡住或报 SSL/连接错误，
此时加 `--no-proxy` 绕过。
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

REPO = "opencv/opencv_zoo"
REF = "main"
RAW = "https://raw.githubusercontent.com/%s/%s" % (REPO, REF)
LFS = "https://media.githubusercontent.com/media/%s/%s" % (REPO, REF)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# (本地文件名, 上游子目录, 期望 SHA-256, 走 LFS 直链?)
ASSETS = [
    ("face_detection_yunet_2023mar.onnx", "face_detection_yunet",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", True),
    ("face_recognition_sface_2021dec.onnx", "face_recognition_sface",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", True),
    # 许可证不是 LFS 对象，走 raw 即可（体积很小）
    ("LICENSE_yunet.txt", "face_detection_yunet", None, False),
    ("LICENSE_sface.txt", "face_recognition_sface", None, False),
]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_opener(no_proxy):
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def download(opener, url, dest):
    """带进度地把 url 下载到 dest（先写 .part，成功后再改名）。"""
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "face-login-fetch"})
    with opener.open(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write("\r    %5.1f%%  (%d / %d bytes)"
                                     % (done * 100.0 / total, done, total))
                    sys.stdout.flush()
    if total:
        sys.stdout.write("\n")
    os.replace(tmp, dest)


def fetch_one(opener, name, subdir, expected, use_lfs, force):
    dest = os.path.join(MODELS_DIR, name)
    base = LFS if use_lfs else RAW
    url = "%s/models/%s/%s" % (base, subdir, name)

    if not force and os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected is None:
            print("  [跳过] %s 已存在 (%d bytes)" % (name, size))
            return True
        if size == 131:
            print("  [重下] %s 是 LFS 指针文件（131 bytes），不是模型本体" % name)
        elif sha256_of(dest) == expected:
            print("  [跳过] %s 已存在且校验通过" % name)
            return True
        else:
            print("  [重下] %s 校验不符" % name)

    print("  [下载] %s" % name)
    print("         %s" % url)
    try:
        download(opener, url, dest)
    except (urllib.error.URLError, OSError) as exc:
        print("         [失败] %s" % exc)
        print("         提示：代理不通时试试 --no-proxy")
        return False

    if expected is not None:
        actual = sha256_of(dest)
        if actual != expected:
            print("         [失败] SHA-256 不符")
            print("                期望 %s" % expected)
            print("                实际 %s" % actual)
            return False
        print("         [OK] SHA-256 校验通过 (%d bytes)" % os.path.getsize(dest))
    else:
        print("         [OK] %d bytes" % os.path.getsize(dest))
    return True


def main():
    parser = argparse.ArgumentParser(description="下载人脸模型与许可证到 models/")
    parser.add_argument("--force", action="store_true", help="无条件重新下载")
    parser.add_argument("--no-proxy", action="store_true",
                        help="忽略系统代理环境变量（代理失效时使用）")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("目标目录: %s" % MODELS_DIR)
    print("来源: OpenCV Zoo (%s @ %s)" % (REPO, REF))
    print()

    opener = build_opener(args.no_proxy)
    results = []
    for name, subdir, expected, use_lfs in ASSETS:
        results.append(fetch_one(opener, name, subdir, expected, use_lfs, args.force))

    print()
    if all(results):
        print("全部就绪。")
        return 0
    print("有文件未能就绪，请检查上面的错误信息。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
