"""Face Login launcher.

Works both with python.exe (console visible) and pythonw.exe (no console).
When run under pythonw.exe, sys.stdout / sys.stderr are None, so we redirect
them to launch_log.txt -- otherwise the first print() would crash silently.
"""

import io
import os
import sys
import threading
import time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
LOG_PATH = os.path.join(PROJECT_DIR, "launch_log.txt")

sys.path.insert(0, BASE_DIR)

URL = "http://127.0.0.1:5000/"


def _ensure_streams():
    """pythonw.exe gives us None streams; point them at a log file instead."""
    needs_redirect = sys.stdout is None or sys.stderr is None
    if needs_redirect:
        try:
            log = open(LOG_PATH, "a", encoding="utf-8", errors="replace")
        except Exception:
            log = io.StringIO()
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _pause_if_possible(msg="Press Enter to exit..."):
    """input() only makes sense when a real console/stdin is attached."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input(msg)
    except Exception:
        pass


def _has_console():
    """True only when a real console window is attached (python.exe, not pythonw.exe)."""
    try:
        import ctypes

        return ctypes.windll.kernel32.GetConsoleWindow() != 0
    except Exception:
        return False


def _alert(title, message, style=0x10):
    """Under pythonw.exe there is no console, so errors would vanish silently.
    Pop a dialog instead -- the user has to be able to see what went wrong."""
    if _has_console():
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, style)
    except Exception:
        pass


def open_browser_later():
    time.sleep(2.5)
    try:
        webbrowser.open(URL)
    except Exception:
        pass


def _service_already_running():
    """On Windows a second process can still bind the same port, which makes
    two competing services. Detect it first and just reuse the running one."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", 5000)) == 0


def main():
    _ensure_streams()

    if _service_already_running():
        print("[INFO] Face Login service is already running at " + URL)
        print("[INFO] Opening the browser instead of starting a second copy.")
        sys.stdout.flush()
        try:
            webbrowser.open(URL)
        except Exception:
            pass
        return 0

    try:
        import server
    except Exception as exc:
        print("[ERROR] failed to import server module:", exc)
        sys.stdout.flush()
        _alert("Face Login - startup failed",
               "Could not load the service module.\n\n"
               "Reason:\n" + str(exc) + "\n\n"
               "See launch_log.txt in the project folder for details.")
        _pause_if_possible()
        return 1

    print("=" * 46)
    print("  Face Login System is running")
    print("  URL: " + URL)
    print("  Close this window to stop the service")
    print("=" * 46)
    sys.stdout.flush()

    if not server.engine.camera_ready():
        print("[WARN] camera not opened, it may be used by another program.")
        sys.stdout.flush()
        _alert("Face Login - camera not ready",
               "The camera could not be opened.\n\n"
               "Another program (WeChat / Tencent Meeting / camera app) may be "
               "holding it. Close that program, then stop and restart the service.",
               0x30)

    threading.Thread(target=open_browser_later, daemon=True).start()

    try:
        server.app.run(host="127.0.0.1", port=5000, threaded=True)
    except Exception as exc:
        print("[ERROR] server stopped:", exc)
        sys.stdout.flush()
        _alert("Face Login - service stopped",
               "The web service exited unexpectedly.\n\n"
               "Reason:\n" + str(exc) + "\n\n"
               "If the port is already in use, run the Stop script first.")
        _pause_if_possible()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
