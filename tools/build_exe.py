"""把项目打包成免安装的 Windows 文件夹（PyInstaller onedir）。

产物结构（dist/FaceLogin/）：
    FaceLogin.exe     双击启动，无控制台窗口
    models/           人脸模型，从项目根复制过来 —— 放在这里是为了让用户
                      看得见、也能自己替换；不打进 exe 内部，省掉每次启动解压
    _internal/        Python 运行时与第三方依赖，不要手动改
    data/             首次运行后自动生成，存 faces.db
    launch_log.txt    首次运行后自动生成

为什么用 onedir 而不是 onefile
    onefile 每次启动都要把上百 MB 解压到临时目录，冷启动 5-15 秒；
    onedir 直接映射磁盘文件，启动 1-2 秒，也不依赖临时目录的写入权限。

用法（在项目根执行，或用项目根的「打包.bat」双击）：
    .venv\\Scripts\\python.exe tools\\build_exe.py
    .venv\\Scripts\\python.exe tools\\build_exe.py --console   # 保留控制台，排查用
    .venv\\Scripts\\python.exe tools\\build_exe.py --slim      # 额外裁掉用不到的大文件
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
APP_DIR = os.path.join(ROOT, "app")
ENTRY = os.path.join(APP_DIR, "run.py")
TEMPLATES = os.path.join(APP_DIR, "templates")
MODELS_SRC = os.path.join(ROOT, "models")
DIST_DIR = os.path.join(ROOT, "dist")
BUILD_DIR = os.path.join(ROOT, "build")
OUT_DIR = os.path.join(DIST_DIR, "FaceLogin")
APP_NAME = "FaceLogin"

# 整个项目都用不到、却会被依赖链顺带扫进来的模块。
# 只做「不打进去」，不做任何代码改动 —— 运行时不 import 就不会触发。
EXCLUDES = [
    "tkinter",          # GUI 工具包，本项目界面全在浏览器里
    "unittest", "pydoc", "doctest",
    "setuptools", "pip", "distutils", "lib2to3",
    "numpy.f2py", "numpy.testing",
    "PIL", "matplotlib", "pandas", "scipy", "IPython",
    "pytest", "setuptools._vendor",
]

# cffi / argon2 的底层模块不会被静态扫描到，必须显式声明。
HIDDEN = ["cv2", "argon2", "argon2._ffi", "_cffi_backend"]


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0


def dir_size_and_top(path, top=10):
    """返回 (总字节数, [(大小, 相对路径)] 前 top 大)。"""
    total = 0
    files = []
    for base, _dirs, names in os.walk(path):
        for name in names:
            full = os.path.join(base, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            total += size
            files.append((size, os.path.relpath(full, path)))
    files.sort(reverse=True)
    return total, files[:top]


def slim_trim(out_dir):
    """裁掉运行时用不到的大文件，返回 (已删除, 删除失败)。

    目前只动一处：OpenCV 的 ffmpeg 视频解码后端（约 30 MB）。本项目只用摄像头
    实时取帧（Windows 上走 MSMF/DSHOW），不解码任何视频文件，所以它是纯负重。
    它是延迟加载的 —— 只有真的用 VideoCapture 打开**文件**时才需要，删掉不影响
    实时取帧。

    删除失败时**不能中断打包**：产物本身是好的，只是多占体积。受限环境
    （只读盘、企业审计策略、沙箱批量删除保护）都可能拒绝删除，所以这里把
    异常全部收下并如实上报，由用户决定是否手动删。
    """
    internal = os.path.join(out_dir, "_internal")
    removed = []
    failed = []
    for base, _dirs, names in os.walk(internal):
        for name in names:
            if name.startswith("opencv_videoio_ffmpeg") and name.endswith(".dll"):
                full = os.path.join(base, name)
                try:
                    size = os.path.getsize(full)
                    os.remove(full)
                    removed.append((size, name))
                except BaseException as exc:   # 权限/只读/沙箱策略，一律不致命
                    failed.append((name, exc))
    return removed, failed


def main():
    parser = argparse.ArgumentParser(description="打包成免安装的 Windows 文件夹")
    parser.add_argument("--console", action="store_true",
                        help="保留控制台窗口（默认无窗口，便于排查问题）")
    parser.add_argument("--slim", action="store_true",
                        help="额外裁掉已知用不到的大文件（见 slim_trim）")
    args = parser.parse_args()

    # ---- 前置检查：宁可现在报错，也不要打出一个跑不起来的包 ----
    problems = []
    if not os.path.exists(ENTRY):
        problems.append("找不到入口：%s" % ENTRY)
    if not os.path.exists(TEMPLATES):
        problems.append("找不到前端模板目录：%s" % TEMPLATES)
    for name in ("face_detection_yunet_2023mar.onnx",
                 "face_recognition_sface_2021dec.onnx"):
        if not os.path.exists(os.path.join(MODELS_SRC, name)):
            problems.append("缺少模型文件：models\\%s（可运行 tools\\download_models.py 下载）" % name)
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        problems.append("未安装 PyInstaller，请先执行："
                        ".venv\\Scripts\\python.exe -m pip install pyinstaller")
    if problems:
        for p in problems:
            print("[ERROR] " + p)
        return 1

    print("=" * 62)
    print("  打包 FaceLogin（onedir%s）" % ("，无控制台" if not args.console else "，带控制台"))
    print("=" * 62)
    print("  入口    : %s" % ENTRY)
    print("  输出    : %s" % OUT_DIR)
    print()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir",
        "--name", APP_NAME,
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        "--specpath", BUILD_DIR,
        # app/ 加进搜索路径，run.py 里的 `import server` 才能被静态分析到
        "--paths", APP_DIR,
        # 前端模板是只读资源，打包后要落在 _MEIPASS/templates/
        "--add-data", "%s%stemplates" % (TEMPLATES, os.pathsep),
    ]
    if not args.console:
        cmd.append("--noconsole")
    for name in HIDDEN:
        cmd += ["--hidden-import", name]
    for name in EXCLUDES:
        cmd += ["--exclude-module", name]
    cmd.append(ENTRY)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print()
        print("[ERROR] PyInstaller 打包失败，返回码 %d" % result.returncode)
        return result.returncode

    # ---- 把模型放到 exe 旁边（不打进包内）----
    dst_models = os.path.join(OUT_DIR, "models")
    if os.path.exists(dst_models):
        shutil.rmtree(dst_models)
    shutil.copytree(MODELS_SRC, dst_models)

    print()
    print("模型已复制到 : %s" % dst_models)

    if args.slim:
        removed, failed = slim_trim(OUT_DIR)
        if removed:
            print("已裁剪       : " + "、".join(
                "%s（%s）" % (n, human(s)) for s, n in removed))
        for name, exc in failed:
            print("[WARN] 裁剪失败：%s" % name)
            print("       原因：%s" % exc)
            print("       不影响运行，只是多占体积；可手动删除该文件。")
        if not removed and not failed:
            print("已裁剪       : 未发现可裁剪项")

    # ---- 体积报告 ----
    total, top = dir_size_and_top(OUT_DIR)
    print()
    print("=" * 62)
    print("  打包完成，用时 %.0f 秒" % (time.time() - t0))
    print("  总大小 : %s" % human(total))
    print("=" * 62)
    print("  占体积最大的文件：")
    for size, rel in top:
        print("    %10s  %s" % (human(size), rel))

    print()
    print("  下一步：双击 %s 启动" % os.path.join("dist", "FaceLogin", APP_NAME + ".exe"))
    print("  注意：第一次运行会在 exe 旁边创建 data\\ 目录与 launch_log.txt")
    print("  分发：整个 dist\\FaceLogin 文件夹压缩后即可发给别人（无需安装 Python）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
