# -*- coding: utf-8 -*-
"""打包产物烟测：对着 dist 里的 exe 发真实 HTTP 请求，验证「二进制里」到底有什么。

为什么需要这个脚本
------------------
PyInstaller 把纯 Python 模块压进 exe 内的 PYZ 归档（zlib 压缩），所以
「在包里 find 一下某个 .py」是**问不出答案**的 —— 按文件名既找不到存在、
也找不到不存在。唯一可靠的办法是**对着 exe 本身发真实请求**。

这一层是 tools/test_security.py 覆盖不到的：它跑的是源码。
源码全绿而二进制是坏的（比如重打包漏了模块、或干脆忘了重打包、
发出去的还是加固前的旧包）只有这里能发现。

最关键的一条是**正向对照**：只测「该被拒的都拒了」是不够的 ——
服务整个坏掉也会让所有请求返回 403，看起来像「加固生效」。
所以必须有一条「带正确凭证要能通过」的用例。

关于「关页自动退出」这条探测
----------------------------
它会**被动观察**而不是主动轮询（发任何请求都会把退出计划取消掉，包括我自己发的）。
并且会先判断环境里有没有**第二个客户端**（比如你忘了关的浏览器标签）：
那种情况下它会每 2 秒轮询一次 /api/state，不断给服务续命，
退出永远不会触发 —— 那是环境问题，不是包的缺陷，所以标记为「跳过」而不是「失败」。

用法
----
    .venv\\Scripts\\python.exe tools\\smoke_frozen.py
    .venv\\Scripts\\python.exe tools\\smoke_frozen.py --exe <exe 路径>

退出码：0 = 没有失败项（可能含跳过）；1 = 有探测失败（这个包不可信，别分发）
"""

import argparse
import http.client
import os
import re
import shutil
import socket
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.path.join(PROJECT_ROOT, "dist", "FaceLogin", "FaceLogin.exe")

HOST = "127.0.0.1"
PORT = int(os.environ.get("FACELOGIN_PORT", "5000"))
HOST_HEADER = "%s:%d" % (HOST, PORT)

_REQ_LINE = re.compile(rb'"(GET|POST|HEAD|OPTIONS) ')

_passed = 0
_skipped = []
_failures = []


def check(label, ok, detail=""):
    global _passed
    line = "  [%s] %s" % ("OK  " if ok else "FAIL", label)
    if detail:
        line += "  <- %s" % detail
    print(line, flush=True)
    if ok:
        _passed += 1
    else:
        _failures.append(label)
    return ok


def skip(label, reason):
    print("  [SKIP] %s" % label, flush=True)
    print("         %s" % reason, flush=True)
    _skipped.append(label)


def port_busy():
    """5000 端口是否已被占用。

    这个检查是**必须的**：如果端口上已经跑着别的实例（比如你自己开着的开发服务），
    探测打中的是那个进程，会给出一片 PASS —— 但那验的根本不是刚打出来的包。
    **一个会误报通过的检查比没有检查更危险**，所以宁可拒绝运行。
    """
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex((HOST, PORT)) == 0
    finally:
        s.close()


def newest_source():
    """app/ 下最新的源文件与它的 mtime（排除 __pycache__）。"""
    newest, newest_path = 0.0, ""
    for base, dirs, names in os.walk(os.path.join(PROJECT_ROOT, "app")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if not (name.endswith(".py") or name.endswith(".html")):
                continue
            full = os.path.join(base, name)
            mtime = os.path.getmtime(full)
            if mtime > newest:
                newest, newest_path = mtime, full
    return newest, newest_path


def request(method, path, headers=None, body=None, read=True, timeout=6):
    """发一个裸请求。read=False 用于 MJPEG 流 —— 它永远不会结束，不能 read()。"""
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        status = resp.status
        data = resp.read() if read else b""
        return status, data
    finally:
        conn.close()


def wait_ready(seconds):
    """等服务开始监听。不算失败项 —— 起不来由调用方统一报错并打日志。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            status, _ = request("GET", "/", {"Host": HOST_HEADER}, timeout=3)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------- 日志观察

def count_requests(log_path):
    try:
        with open(log_path, "rb") as fh:
            return len(_REQ_LINE.findall(fh.read()))
    except OSError:
        return 0


def request_lines(log_path):
    try:
        with open(log_path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return []
    return [ln.decode("utf-8", "replace") for ln in raw.splitlines()
            if _REQ_LINE.search(ln)]


def detect_competing_client(log_path, settle=3.6):
    """静默观察一小段，看有没有「不是本脚本发的」请求。

    心跳间隔是 2 秒，所以静默 3.6 秒足以确认。**全程只读日志、不发请求** ——
    发请求本身就会把退出计划取消掉。
    """
    before = count_requests(log_path)
    time.sleep(settle)
    return count_requests(log_path) > before


# ---------------------------------------------------------------- 探测

def probe_binary():
    """对着 exe 的真实 HTTP 探测。"""
    print("\n-- 探测：二进制里到底有没有那道闸门 --")

    status, _ = request("GET", "/", {"Host": HOST_HEADER})
    check("正常 Host 访问首页 => 200", status == 200, "got %d" % status)

    # 真分页之后模板落在 _MEIPASS/templates/，由 server 用 paths.resource_path()
    # 指过去。这一条专门验"包里也认得那个目录"，且页面不是空壳。
    status, body = request("GET", "/login", {"Host": HOST_HEADER})
    check("/login 也是真页面（多页路由与模板都进了包）",
          status == 200 and b'name="page-token"' in body, "got %d" % status)

    status, _ = request("GET", "/settings", {"Host": HOST_HEADER})
    check("未登录访问 /settings => 302（状态机在服务端，包里同样生效）",
          status == 302, "got %d" % status)

    status, _ = request("GET", "/", {"Host": "evil.com"})
    check("Host: evil.com => 403（DNS 重绑定防护在包里）",
          status == 403, "got %d" % status)

    status, _ = request(
        "POST", "/api/login/password",
        {"Host": HOST_HEADER, "Origin": "http://evil.com",
         "Content-Type": "application/json"},
        b'{"username":"smoke","password":"Smoke#12345"}')
    check("Origin: evil.com 打登录接口 => 403（CSRF 防护在包里）",
          status == 403, "got %d" % status)

    status, _ = request("GET", "/api/stream", {"Host": HOST_HEADER})
    check("/api/stream 不带页面令牌 => 403", status == 403, "got %d" % status)

    status, body = request("GET", "/", {"Host": HOST_HEADER})
    m = re.search(rb'name="page-token" content="([^"]*)"', body)
    token = m.group(1).decode("ascii", "replace") if m else ""
    check("从响应体里抠出页面令牌（证明值真的送到了浏览器）",
          len(token) >= 20, "len=%d" % len(token))

    status, _ = request("GET", "/api/stream?t=" + token,
                        {"Host": HOST_HEADER}, read=False)
    check("带正确令牌访问 /api/stream => 200（正向对照：防「服务全坏」被误判成「加固生效」）",
          status == 200, "got %d" % status)

    return token


def cleanup(exe_dir, created_data, created_log):
    """只删**本次运行自己产生**的文件。

    如果这两个东西在启动前就存在（比如用户真在用这个文件夹），那是别人的数据，
    一个字节都不能碰 —— 这个脚本是为了防事故，不能自己变成事故源。
    """
    removed = []
    if created_data:
        data_dir = os.path.join(exe_dir, "data")
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir, ignore_errors=True)
            removed.append("data/")
    if created_log:
        log_path = os.path.join(exe_dir, "launch_log.txt")
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
                removed.append("launch_log.txt")
            except OSError:
                pass
    if removed:
        print("\n已清掉本次运行痕迹：%s（产物恢复「首次运行前」状态）" % "、".join(removed))
    else:
        print("\n（无需清理：data/ 与 launch_log.txt 启动前就存在，未触碰）")


def read_log_tail(path, limit=4000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[-limit:]
    except OSError:
        return ""


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description="对打包产物做真机 HTTP 探测")
    parser.add_argument("--exe", default=DEFAULT_EXE, help="要验证的 exe 路径")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="等待服务就绪的秒数")
    parser.add_argument("--grace-wait", type=float, default=20.0,
                        help="等待「关页退出」生效的秒数")
    args = parser.parse_args()

    exe = os.path.abspath(args.exe)
    exe_dir = os.path.dirname(exe)
    log_path = os.path.join(exe_dir, "launch_log.txt")

    print("=" * 62)
    print("  打包产物烟测：%s" % exe)
    print("=" * 62)

    print("\n-- 前置检查 --")

    if not os.path.isfile(exe):
        print("  [FAIL] 找不到 exe：%s" % exe)
        print("\n先打包：\n    .venv\\Scripts\\python.exe tools\\build_exe.py --slim")
        return 1

    # 防呆：exe 若就在项目根，它的 data/ 会落到**真实的** data/faces.db 上。
    if os.path.normcase(exe_dir) == os.path.normcase(PROJECT_ROOT):
        print("  [FAIL] exe 位于项目根，它会把数据写进真实的 data/faces.db —— 拒绝运行")
        return 1
    check("产物目录不是项目根（不会写进真实数据库）", True, exe_dir)

    if port_busy():
        print("  [FAIL] %s:%d 已被占用" % (HOST, PORT))
        print("\n端口上已经有别的服务在跑 —— 探测会打中它并给出假 PASS，"
              "所以拒绝运行。\n请先关掉那个实例，或用 FACELOGIN_PORT 指定别的端口。")
        return 1
    check("端口 %d 空闲（保证探测打的是刚打包的这个 exe）" % PORT, True)

    src_mtime, src_path = newest_source()
    exe_mtime = os.path.getmtime(exe)
    fresh = exe_mtime >= src_mtime
    detail = "" if fresh else "exe %s 早于 %s" % (
        time.strftime("%m-%d %H:%M", time.localtime(exe_mtime)),
        os.path.relpath(src_path, PROJECT_ROOT))
    if not check("exe 不比源码旧（防「改了代码但忘了重打包」）", fresh, detail):
        print("\n     这份包是旧的 —— 源码改过之后没有重新打包，"
              "现在所验证的是一个与源码不符的二进制。")
        return 1

    created_data = not os.path.isdir(os.path.join(exe_dir, "data"))
    created_log = not os.path.exists(log_path)

    print("\n-- 启动 --")
    proc = subprocess.Popen([exe], cwd=exe_dir)
    print("  已启动 pid=%d，等待监听 %s:%d ..." % (proc.pid, HOST, PORT))

    try:
        if not wait_ready(args.timeout):
            print("  [FAIL] %g 秒内没有起来" % args.timeout)
            tail = read_log_tail(log_path)
            if tail:
                print("\n---- launch_log.txt 末尾 ----\n%s" % tail)
            return 1
        print("  服务就绪")

        token = probe_binary()

        # 先判断环境里有没有第二个客户端在轮询，再决定这条探测能不能做。
        print("\n-- 探测：关页自动退出（打包后是否仍然生效）--")
        print("  静默观察 3.6 秒，看有没有别的客户端在轮询 ...")
        competing = detect_competing_client(log_path)
        if competing:
            print("  检测到外部客户端在轮询 /api/state —— 它会不断给服务续命。")

        status, body = request("POST", "/api/bye?t=" + token, {"Host": HOST_HEADER})
        check("/api/bye 带正确令牌 => 200", status == 200, "got %d" % status)
        check("/api/bye 确实排了退出计划（scheduled=true）",
              b'"scheduled": true' in body or b'"scheduled":true' in body,
              body[:120].decode("utf-8", "replace"))

        if competing:
            skip("关页 5 秒后进程自行退出、摄像头被释放",
                 "环境里有第二个客户端在轮询（最可能是你忘关的浏览器标签或预览面板），"
                 "它会不断取消退出计划 —— 这不是包的缺陷。关掉那个页面即可验证。")
        else:
            # bye 之后**一个请求都不能再发**，只能被动看日志 + 轮询进程。
            base = count_requests(log_path)
            start = time.time()
            while time.time() - start < args.grace_wait:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            waited = time.time() - start
            new_reqs = count_requests(log_path) - base
            if proc.poll() is not None:
                check("关页 5 秒后进程自行退出、摄像头被释放", True,
                      "用了 %.1f 秒" % waited)
            elif new_reqs > 0:
                skip("关页 5 秒后进程自行退出、摄像头被释放",
                     "等待 %.1f 秒期间日志里多出 %d 条不是本脚本发的请求 —— 有客户端在续命"
                     % (waited, new_reqs))
                for line in request_lines(log_path)[-3:]:
                    print("         %s" % line.strip())
            else:
                check("关页 5 秒后进程自行退出、摄像头被释放", False,
                      "等待 %.1f 秒仍未退出，且日志里没有外部请求 —— 退出机制本身有问题"
                      % args.grace_wait)

    finally:
        if proc.poll() is None:
            print("\n  （进程仍在运行，兜底终止）")
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        cleanup(exe_dir, created_data, created_log)

    print("\n" + "=" * 62)
    total = _passed + len(_skipped) + len(_failures)
    if _failures:
        print("  结果：%d 项中 %d 通过、%d 跳过、%d 失败 —— 这个包不可信，别分发"
              % (total, _passed, len(_skipped), len(_failures)))
        for name in _failures:
            print("    x %s" % name)
        print("=" * 62)
        return 1
    if _skipped:
        print("  结果：%d 项中 %d 通过、%d 跳过（跳过项见上方说明）" % (total, _passed, len(_skipped)))
    else:
        print("  结果：%d 项全部通过" % total)
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
