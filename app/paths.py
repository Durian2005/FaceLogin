"""运行形态相关的路径解析（源码运行 / PyInstaller 打包后）。

为什么单独抽一个模块
--------------------
同一份代码有两种运行形态，路径基准不一样：

- **源码运行**：程序根 = 项目根（`app/` 的上一级）。
  模型在 `<项目根>/models`，数据在 `<项目根>/data`。
- **打包运行**（PyInstaller onedir）：程序根 = **exe 所在目录**。
  模型与数据都放在 exe 旁边 —— 用户看得见，也能自己替换 `models/` 里的模型
  文件、直接备份 `data/`。工程上刻意不让它们藏进 exe 内部，这样
  「复制文件夹 = 备份全部数据」，不依赖任何安装过程。

另有一类路径不能混进来：**只读资源**（Flask 的 HTML 模板）。打包后它们位于
PyInstaller 的解压目录（`sys._MEIPASS`，onedir 模式下是 `_internal/`），
既不在项目根、也不在 exe 旁边，必须用 `resource_path()` 取。
"""

import os
import sys


def is_frozen():
    """是否运行在 PyInstaller 打出来的 exe 里。"""
    return bool(getattr(sys, "frozen", False))


def app_root():
    """程序根目录：源码运行 = 项目根；打包运行 = exe 所在目录。

    可写、且对用户可见的那一层。`data/`、`models/`、`launch_log.txt` 都在这里。
    """
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts):
    """只读资源（HTML 模板等）的绝对路径。

    打包后位于 PyInstaller 的解压目录，由打包时的 `--add-data` 决定；
    源码运行时就是 `app/` 下的相对路径。
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, *parts)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)
