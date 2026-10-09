"""本地人脸登录服务：注册、密码登录、人脸 1:1 登录、会话、失败限速、审计。"""

import os
import re
import secrets
import threading
import time
from datetime import datetime

import cv2
from flask import (Flask, Response, jsonify, redirect, render_template, request,
                   session)

import db
import liveness
import paths
import reqguard
import security
from face import FaceEngine, pack, unpack
from model_paths import MODELS_DIR

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")

app = Flask(__name__)

db.init_db()
app.secret_key = db.get_or_create_secret()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=3600 * 8,
)

# ------------------------------------------------- 请求来源校验
# 本地服务最现实的攻击面是「你自己的浏览器」：恶意页面可以在你不知情时向
# 127.0.0.1:5000 发请求。详见 app/reqguard.py 的模块说明。
PORT = int(os.environ.get("FACELOGIN_PORT", "5000"))
ALLOWED_HOSTS = reqguard.allowed_hosts(PORT)
#: 每进程随机，重启即失效。只有本服务渲染出去的页面才知道它，
#: 用来保护那些「不能要求登录态」的接口（登录前就要显示摄像头画面）。
PAGE_TOKEN = reqguard.new_page_token()


@app.before_request
def _guard_request():
    """Host / Origin 双闸门。

    必须注册在 _touch_liveness 之前：被拒绝的跨站请求不该顺手续上心跳，
    否则攻击者只要持续发垃圾请求就能让服务永远不自动退出。
    """
    if not reqguard.host_allowed(request.host, ALLOWED_HOSTS):
        # Host 是攻击者域名 => DNS rebinding 的典型特征，直接掐断
        return jsonify({"ok": False, "error": "bad_host",
                        "message": "请求的 Host 不在允许列表内"}), 403
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        if not reqguard.origin_allowed(request.headers.get("Origin"),
                                       request.headers.get("Referer"),
                                       ALLOWED_HOSTS):
            return jsonify({"ok": False, "error": "bad_origin",
                            "message": "请求来源不被允许"}), 403
    return None


def _page_token_ok():
    supplied = request.args.get("t") or request.headers.get("X-Page-Token")
    return reqguard.token_matches(supplied, PAGE_TOKEN)


engine = FaceEngine()


def current_threshold():
    try:
        return float(db.get_setting("threshold", security.DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return security.DEFAULT_THRESHOLD


THRESHOLD = {"value": current_threshold()}
last_state = {"faces": 0, "score": 0.0}


# ------------------------------------------------- 关掉网页就自动退出（释放摄像头）
# 机制：页面每 2 秒轮询 /api/state，天然就是心跳；关闭/刷新页面时前端再用
# sendBeacon 发一次 /api/bye。收到 bye 先给一个宽限期，期间只要还有任何请求
# 进来就取消退出（这样"刷新页面"不会被误判成"关闭"），超时才真正退出。

def _env_flag(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


AUTO_EXIT = _env_flag("FACELOGIN_AUTO_EXIT", True)
# 人脸登录是否要求通过随机动作挑战。默认开启；关闭后登录退化为「直接比对一次」，
# 便于无人值守的自测。**关闭即等于放弃反照片欺骗能力**，README 里已写明。
LIVENESS_ENABLED = _env_flag("FACELOGIN_LIVENESS", True)
try:
    EXIT_GRACE = float(os.environ.get("FACELOGIN_EXIT_GRACE", "5"))
except ValueError:
    EXIT_GRACE = 5.0
try:
    EXIT_IDLE = float(os.environ.get("FACELOGIN_EXIT_IDLE", "120"))
except ValueError:
    EXIT_IDLE = 120.0

_liveness = {
    "seen": False,      # 页面是否连上来过（没连过就永不自动退出）
    "last_seen": 0.0,   # 最后一次页面活动时间
    "exit_at": 0.0,     # 非 0 表示已排好退出时间，等待被取消
}
_liveness_lock = threading.Lock()


def _log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _mark_alive():
    now = time.time()
    with _liveness_lock:
        _liveness["seen"] = True
        _liveness["last_seen"] = now
        _liveness["exit_at"] = 0.0


def _mark_leaving():
    with _liveness_lock:
        if not _liveness["seen"]:
            return False
        _liveness["exit_at"] = time.time() + EXIT_GRACE
        return True


@app.before_request
def _touch_liveness():
    # /api/bye 是"我要走了"的信号，不能当成活动
    if request.path != "/api/bye":
        _mark_alive()


@app.route("/api/bye", methods=["POST"])
def api_bye():
    # 同样用页面令牌：这个接口能触发服务退出，不能让任意网页随便调。
    if not _page_token_ok():
        return jsonify({"ok": False, "error": "bad_token"}), 403
    scheduled = _mark_leaving()
    return jsonify({"ok": True, "scheduled": scheduled, "grace": EXIT_GRACE})


def _exit_watcher():
    """兜底看护：宽限期到点、或长时间没有任何页面活动，就退出释放摄像头。"""
    while True:
        time.sleep(1.0)
        now = time.time()
        with _liveness_lock:
            seen = _liveness["seen"]
            last = _liveness["last_seen"]
            exit_at = _liveness["exit_at"]
        if not seen:
            continue
        if exit_at and now >= exit_at:
            _log("[INFO] page closed -> shutting down to release the camera")
            os._exit(0)
        if last and now - last > EXIT_IDLE:
            _log("[INFO] no page activity for %.0fs -> shutting down" % (now - last))
            os._exit(0)


if AUTO_EXIT:
    threading.Thread(target=_exit_watcher, daemon=True).start()


# ---------------------------------------------------------------- helpers

def now_text(ts):
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")


def lock_remaining(user):
    if user is None:
        return 0
    return max(0, int(round(user["locked_until"] - time.time())))


def lock_remaining_by_name(username):
    return lock_remaining(db.get_user(username))


def logged_in_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.get_user_by_id(uid)


def start_session(user):
    session.clear()  # 防会话固定
    session.permanent = True
    session["user_id"] = user["id"]
    session["username"] = user["username"]


def require_login():
    user = logged_in_user()
    if user is None:
        return None, (jsonify({"ok": False, "error": "login_required"}), 401)
    return user, None


# ---------------------------------------------------------------- camera stream

def gen_frames():
    while True:
        frame = engine.read_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        face = engine.detect_largest(frame)
        last_state["faces"] = 0 if face is None else 1
        last_state["score"] = 0.0 if face is None else round(float(face[14]), 4)
        ok, jpg = cv2.imencode(".jpg", engine.draw(frame, face))
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.route("/api/stream")
def api_stream():
    # 摄像头画面是这套系统里最敏感的输出，但它在登录前就要显示（注册与
    # 活体挑战都要看得见自己），所以不能用登录态来保护 —— 改用页面令牌。
    if not _page_token_ok():
        return jsonify({"ok": False, "error": "bad_token",
                        "message": "缺少或错误的页面令牌"}), 403
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    user = logged_in_user()
    return jsonify({
        "faces": last_state["faces"],
        "det_score": last_state["score"],
        "threshold": THRESHOLD["value"],
        "password_backend": security.hash_backend(),
        "liveness": LIVENESS_ENABLED,
        "logged_in": user is not None,
        "username": user["username"] if user else None,
        "face_samples": db.count_templates(user["id"]) if user else 0,
        # 未登录时 face_samples 恒为 0，光靠它分不清「本机没账号」和「只是没登录」。
        # 页面需要这个布尔才能决定是引导注册还是引导登录。只增字段，不动上面任何一个。
        "has_any_user": db.count_users() > 0,
    })


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    # 这里刻意**不要求**页面令牌：全项目没有任何调用方（停止服务.bat 是按
    # PID 直接杀的），保留无令牌是为了让本机脚本还能用它。安全性由
    # _guard_request 的 Origin 校验兜住 —— 跨站表单一定带别人的 Origin，
    # 会被 403；而一个已经能在本机跑命令的进程，本来就能 taskkill。
    def stop():
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=stop, daemon=True).start()
    return jsonify({"ok": True, "message": "server stopping"})


# ---------------------------------------------------------------- register

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not USERNAME_RE.match(username):
        return jsonify({"ok": False, "error": "invalid_username",
                        "message": "用户名需 3-32 位，仅限字母、数字、下划线、点、连字符"})
    problem = security.validate_password(password)
    if problem:
        return jsonify({"ok": False, "error": problem,
                        "message": "密码至少 6 位，最多 128 位"})

    ok, result = db.create_user(username, security.hash_password(password))
    if not ok:
        db.add_event(username, "register", False, detail="username_taken")
        return jsonify({"ok": False, "error": "username_taken", "message": "该用户名已被占用"})

    user = db.get_user_by_id(result)
    db.add_event(username, "register", True, detail="account_created")
    start_session(user)
    return jsonify({"ok": True, "username": user["username"], "user_id": user["id"],
                    "message": "注册成功，已登录。接下来请录入人脸。"})


# ---------------------------------------------------------------- login

@app.route("/api/login/password", methods=["POST"])
def api_login_password():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"ok": False, "error": "missing_field", "message": "请填写用户名和密码"})

    user = db.get_user(username)
    remaining = lock_remaining(user)
    if remaining > 0:
        db.add_event(username, "password", False, detail="locked")
        return jsonify({"ok": False, "error": "locked", "retry_after": remaining,
                        "message": "账号已临时锁定，请 %d 秒后再试" % remaining})

    if user is None:
        security.dummy_verify()  # 与真实校验耗时接近，避免暴露账号是否存在
        db.add_event(username, "password", False, detail="unknown_user")
        return jsonify({"ok": False, "error": "auth_failed", "message": "用户名或密码不正确"})

    if security.verify_password(user["pwd_hash"], password):
        db.set_last_login(user["id"])
        db.add_event(username, "password", True, detail="ok")
        start_session(user)
        return jsonify({"ok": True, "username": user["username"],
                        "face_samples": db.count_templates(user["id"]),
                        "message": "密码验证通过"})

    failed, locked_until = db.bump_failure(user["id"], security.MAX_FAILS,
                                           security.LOCK_SECONDS)
    db.add_event(username, "password", False, detail="bad_password_%d" % failed)
    left = max(0, security.MAX_FAILS - failed)
    if locked_until > time.time():
        db.add_event(username, "lockout", False, detail="locked_%ds" % security.LOCK_SECONDS)
        return jsonify({"ok": False, "error": "locked", "retry_after": security.LOCK_SECONDS,
                        "message": "连续失败 %d 次，账号锁定 %d 秒"
                                   % (security.MAX_FAILS, security.LOCK_SECONDS)})
    return jsonify({"ok": False, "error": "auth_failed", "failed": failed, "remaining": left,
                    "message": "用户名或密码不正确，还可尝试 %d 次" % left})


def _face_login_guard(username):
    """人脸登录的公共前置检查：锁定状态、账号是否存在、是否已录入人脸。

    返回 (user, stored, 错误响应)。错误响应非 None 时直接原样返回给客户端。
    抽出来是为了让「直接人脸登录」和「活体挑战后人脸登录」走完全相同的判定。
    """
    user = db.get_user(username)
    remaining = lock_remaining(user)
    if remaining > 0:
        db.add_event(username, "face", False, detail="locked")
        return None, None, jsonify({"ok": False, "error": "locked", "retry_after": remaining,
                                    "message": "账号已临时锁定，请 %d 秒后再试" % remaining})

    if user is None:
        db.add_event(username, "face", False, detail="unknown_user")
        return None, None, jsonify({"ok": False, "error": "auth_failed",
                                    "message": "用户名或人脸不匹配"})

    stored = db.templates(user["id"])
    if not stored:
        db.add_event(username, "face", False, detail="no_template")
        return None, None, jsonify({"ok": False, "error": "no_template",
                                    "message": "该账号还没有录入人脸，请先登录后录入"})
    return user, stored, None


def _match_and_login(user, stored, feat, ms, extra=None):
    """特征与模板比对，按结果决定登录或累计失败次数。"""
    refs = [unpack(blob, dim) for blob, dim in stored]
    refs = [r for r in refs if r is not None]
    score = engine.best_cosine(refs, feat)
    extra = extra or {}

    if score >= THRESHOLD["value"]:
        db.set_last_login(user["id"])
        db.add_event(user["username"], "face", True, similarity=round(score, 4), detail="ok")
        start_session(user)
        body = {"ok": True, "username": user["username"], "cosine": round(score, 4),
                "threshold": THRESHOLD["value"], "ms": ms, "message": "人脸验证通过"}
        body.update(extra)
        return jsonify(body)

    failed, locked_until = db.bump_failure(user["id"], security.MAX_FAILS,
                                           security.LOCK_SECONDS)
    db.add_event(user["username"], "face", False, similarity=round(score, 4),
                 detail="below_threshold_%d" % failed)
    if locked_until > time.time():
        db.add_event(user["username"], "lockout", False,
                     detail="locked_%ds" % security.LOCK_SECONDS)
        body = {"ok": False, "error": "locked", "retry_after": security.LOCK_SECONDS,
                "cosine": round(score, 4), "threshold": THRESHOLD["value"], "ms": ms,
                "message": "连续失败 %d 次，账号锁定 %d 秒"
                           % (security.MAX_FAILS, security.LOCK_SECONDS)}
        body.update(extra)
        return jsonify(body)

    body = {"ok": False, "error": "auth_failed", "cosine": round(score, 4),
            "threshold": THRESHOLD["value"], "ms": ms, "failed": failed,
            "message": "人脸相似度 %.4f 低于阈值 %.2f" % (score, THRESHOLD["value"])}
    body.update(extra)
    return jsonify(body)


@app.route("/api/login/face", methods=["POST"])
def api_login_face():
    """直接人脸登录（不带动作挑战）。

    活体检测开启时这个入口会被拒绝：否则随机动作挑战形同虚设 ——
    攻击者绕过前端、直接打这个接口就能跳过挑战。要做免挑战的自动化自测，
    需显式设置 FACELOGIN_LIVENESS=0，并在 README 里如实说明这是降级状态。
    """
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "missing_field", "message": "请填写用户名"})

    if LIVENESS_ENABLED:
        db.add_event(username, "face", False, detail="liveness_required")
        return jsonify({"ok": False, "error": "liveness_required",
                        "message": "已开启活体检测，请点击「人脸登录」完成动作挑战"})

    user, stored, err = _face_login_guard(username)
    if err:
        return err

    feat, status, ms = engine.capture_feature()
    if feat is None:
        db.add_event(username, "face", False, detail=status)
        return jsonify({"ok": False, "error": status,
                        "message": "没有检测到人脸，请正对摄像头再试"})

    return _match_and_login(user, stored, feat, ms)


# ---------------------------------------------------------------- liveness

LIVENESS_ERROR_TEXT = {
    "no_face_detected": "挑战过程中没有检测到人脸，请正对摄像头再试",
    "camera_read_failed": "读取摄像头失败，请检查摄像头是否被其他程序占用",
    "face_changed": "画面中的人脸位置发生突变，动作挑战未通过",
    "liveness_timeout": "没有检测到指定动作，请照着提示转动头部再试",
    "action_not_returned": "动作已识别，但没取到正面画面 —— 做完动作后请转回正对摄像头",
    "not_enough_samples": "采样帧数不足，请保持人脸在画面中再试",
}


@app.route("/api/liveness/start", methods=["POST"])
def api_liveness_start():
    """生成一次性随机动作挑战。动作只随机、不保密：用户得知道要做什么。"""
    if not LIVENESS_ENABLED:
        return jsonify({"ok": False, "error": "liveness_disabled",
                        "message": "活体检测已关闭"})

    sid = session.get("live_sid")
    if not sid:
        sid = secrets.token_urlsafe(12)
        session["live_sid"] = sid          # 挑战绑定到当前浏览器会话
    token, action = liveness.new_challenge(sid)
    return jsonify({"ok": True, "challenge_id": token, "action": action,
                    "action_text": liveness.ACTION_TEXT[action],
                    "hint": liveness.ACTION_HINT,
                    "timeout": liveness.MAX_SECONDS})


@app.route("/api/liveness/verify", methods=["POST"])
def api_liveness_verify():
    """执行动作挑战并完成人脸 1:1 比对。

    注意这里用的是挑战过程中采到的那一帧特征，不是重新抓一帧 ——
    否则「通过动作验证的脸」和「用于比对的脸」可能不是同一个人。
    """
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    token = (data.get("challenge_id") or "").strip()
    if not username or not token:
        return jsonify({"ok": False, "error": "missing_field",
                        "message": "缺少用户名或挑战标识"})

    user, stored, err = _face_login_guard(username)
    if err:
        return err

    item, why = liveness.consume(token, session.get("live_sid"))
    if item is None:
        db.add_event(username, "liveness", False, detail=why)
        return jsonify({"ok": False, "error": why,
                        "message": "挑战已失效，请重新点击「人脸登录」"})

    action = item["action"]
    result = liveness.run_challenge(engine, action)
    diag = result["diag"]
    info = {"action": action, "action_text": liveness.ACTION_TEXT[action],
            "elapsed": diag["elapsed"]}

    if not result["passed"]:
        # 动作没做出来 ≠ 身份验证失败，所以**不累计失败次数**（否则动作不熟练的
        # 用户会被锁号）。只有真正跑到特征比对且不通过时才计数。重试活体挑战
        # 对攻击者没有帮助，不会因此削弱防护。
        db.add_event(username, "liveness", False,
                     detail="%s_%s" % (action, result["error"]))
        info["diag"] = liveness.diag_text(diag)
        return jsonify({"ok": False, "error": result["error"], "liveness": info,
                        "message": LIVENESS_ERROR_TEXT.get(
                            result["error"], "动作挑战未通过，请重试")})

    db.add_event(username, "liveness", True,
                 detail="action_%s | %s" % (action, liveness.diag_text(diag)))
    info["passed"] = True
    return _match_and_login(user, stored, result["feature"],
                            round(diag["elapsed"] * 1000, 1),
                            extra={"liveness": info})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    username = session.get("username")
    session.clear()
    if username:
        db.add_event(username, "logout", True, detail="ok")
    return jsonify({"ok": True, "message": "已退出登录"})


# ---------------------------------------------------------------- face data

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    user, err = require_login()
    if err:
        return err

    # count 是为了让页面能逐组采集、把 1/3 → 3/3 显示成**真实进度**，
    # 而不是拿客户端计时器假装。默认 3 与原行为完全一致，老调用方不受影响。
    raw = (request.get_json(silent=True) or {}).get("count", 3)
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_value",
                        "message": "count 需要是一个整数"})
    count = max(1, min(3, count))

    feats = engine.capture_many(count=count)
    if not feats:
        db.add_event(user["username"], "enroll", False, detail="no_face_detected")
        return jsonify({"ok": False, "error": "no_face_detected",
                        "message": "没有采集到人脸，请正对摄像头、保持画面中只有一张脸"})

    for feat in feats:
        blob, dim = pack(feat)
        db.add_template(user["id"], blob, dim)
    samples = db.count_templates(user["id"])
    db.add_event(user["username"], "enroll", True, detail="samples_%d" % len(feats))
    return jsonify({"ok": True, "created": len(feats), "samples": samples,
                    "message": "录入成功，本次采集 %d 组，当前共 %d 组" % (len(feats), samples)})


@app.route("/api/face/delete", methods=["POST"])
def api_face_delete():
    user, err = require_login()
    if err:
        return err
    password = (request.get_json(silent=True) or {}).get("password") or ""
    if not password:
        return jsonify({"ok": False, "error": "password_required",
                        "message": "删除人脸数据需要输入密码确认"})
    if not security.verify_password(user["pwd_hash"], password):
        db.add_event(user["username"], "delete_face", False, detail="bad_password")
        return jsonify({"ok": False, "error": "auth_failed", "message": "密码不正确，未删除"})

    removed = db.delete_templates(user["id"])
    db.add_event(user["username"], "delete_face", True, detail="removed_%d" % removed)
    return jsonify({"ok": True, "removed": removed,
                    "message": "已删除 %d 组人脸模板" % removed})


# ---------------------------------------------------------------- audit / users / config

@app.route("/api/audit")
def api_audit():
    user, err = require_login()
    if err:
        return err
    limit = min(100, max(5, int(request.args.get("limit", 20))))
    rows = db.recent_events(limit=limit, username=user["username"])
    for row in rows:
        row["time"] = now_text(row["ts"])
        row["ts"] = None
    return jsonify({"ok": True, "events": rows})


@app.route("/api/users")
def api_users():
    user, err = require_login()
    if err:
        return err
    rows = db.list_users()
    users = [{"username": r["username"], "samples": r["samples"],
              "created": now_text(r["created_at"]),
              "last_login": now_text(r["last_login_at"]) if r["last_login_at"] else "-"}
             for r in rows]
    return jsonify({"ok": True, "users": users})


@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    user, err = require_login()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_value"})
    if not 0.20 <= value <= 0.95:
        return jsonify({"ok": False, "error": "out_of_range",
                        "message": "阈值请设在 0.20 到 0.95 之间"})

    current = THRESHOLD["value"]
    if value < current - 1e-9:
        # 调低阈值 = 放宽认人条件，是一次**安全降级**。若任何一个已登录用户
        # 都能随手调低，人脸登录这道关就等于形同虚设；所以要求重新输入本人
        # 口令确认。调高不设门槛（那是收紧，不会削弱任何人的防护）。
        password = data.get("password") or ""
        if not password:
            db.add_event(user["username"], "threshold", False, detail="password_required")
            return jsonify({"ok": False, "error": "password_required",
                            "message": "调低阈值会放宽人脸判定，需要输入密码确认"})
        if not security.verify_password(user["pwd_hash"], password):
            db.add_event(user["username"], "threshold", False, detail="bad_password")
            return jsonify({"ok": False, "error": "auth_failed",
                            "message": "密码不正确，阈值未修改"})

    db.set_setting("threshold", value)
    THRESHOLD["value"] = value
    db.add_event(user["username"], "threshold", True,
                 detail="set_%.2f_from_%.2f" % (value, current))
    return jsonify({"ok": True, "threshold": value, "message": "阈值已更新为 %.2f" % value})


# ---------------------------------------------------------------- 页面（真分页）

# 模板是只读资源：源码运行时在 app/templates/，打包后由 --add-data 放到
# PyInstaller 的解压目录里 —— 必须用 paths.resource_path() 取，不能用程序根。
app.template_folder = paths.resource_path("templates")
# 改完模板刷新即生效：Flask 默认只在 debug 下自动重载模板，
# 而这里刻意要保住「改一行 HTML、刷新就能看到」的手感。
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

#: 上手流程的三步。步骤条由**服务端**渲染 —— 每张页面自己知道"我在第几步"，
#: 不需要前端拿状态去猜。
STEPS = ("注册账号", "录入人脸", "登录验证")


def _steps(step):
    """step：当前进行到第几步（从 0 数）。-1 = 还没开始；3 = 三步都已完成。"""
    return [{"label": label,
             "index": i + 1,
             "state": "done" if i < step else ("now" if i == step else "todo")}
            for i, label in enumerate(STEPS)]


def render_page(name, step=-1, title="", **ctx):
    """渲染一张页面。

    模板拆成 layout.html + 每页一个文件，用 Jinja2 继承 —— 它是 Flask 自带的，
    零构建、离线可用，与本项目「不引 CDN、不引构建工具」这条底线不冲突。

    注入值一律走 HTML（meta 与可见文本），**不写进 <script>** —— 这样内联 JS
    始终是纯静态文本，node --check 还能继续当语法门禁。
    """
    return Response(render_template(
        name,
        token=PAGE_TOKEN,
        port=PORT,
        auto_exit=AUTO_EXIT,
        grace_text="%.0f" % EXIT_GRACE,
        steps=_steps(step),
        stage=name.split(".")[0],
        title=title,
        **ctx), mimetype="text/html")


def _landing_for(user):
    """已登录的人此刻该待在哪张页面：没录人脸先去录入，否则去主控台。"""
    if db.count_templates(user["id"]) == 0:
        return "/enroll"
    return "/console"


# 下面六条路由合起来就是这套系统的状态机。**能不能进某张页面由服务端判定**，
# 不满足条件就重定向到该去的地方 —— 这是真分页最大的收益：用户不可能落到一张
# 自己用不了的页面上，也就不需要靠前端置灰去"提示"他。
# 每张页面只干一件事，页面之间用跳转衔接（完成后自动进入下一张）。

@app.route("/")
def page_welcome():
    user = logged_in_user()
    if user is not None:
        return redirect(_landing_for(user))
    return render_page("welcome.html", title="开始",
                       has_any_user=db.count_users() > 0)


@app.route("/register")
def page_register():
    user = logged_in_user()
    if user is not None:
        return redirect(_landing_for(user))
    return render_page("register.html", step=0, title="创建账号")


@app.route("/login")
def page_login():
    user = logged_in_user()
    if user is not None:
        return redirect(_landing_for(user))
    has_any = db.count_users() > 0
    # 本机连账号都还没有时把步骤条停在第 1 步 —— 否则会显示"注册已完成"。
    return render_page("login.html", step=2 if has_any else 0, title="登录",
                       has_any_user=has_any, liveness=LIVENESS_ENABLED)


@app.route("/enroll")
def page_enroll():
    user = logged_in_user()
    if user is None:
        return redirect("/login")
    return render_page("enroll.html", step=1, title="录入人脸",
                       cam_note="采集中", show_guide=True,
                       username=user["username"],
                       samples=db.count_templates(user["id"]))


@app.route("/console")
def page_console():
    user = logged_in_user()
    if user is None:
        return redirect("/login")
    samples = db.count_templates(user["id"])
    if samples == 0:
        # 没人脸模板就没有"主控台"可言，直接送到该去的地方
        return redirect("/enroll")
    return render_page("console.html", step=3, title="主控台",
                       username=user["username"], samples=samples,
                       threshold=THRESHOLD["value"])


@app.route("/settings")
def page_settings():
    user = logged_in_user()
    if user is None:
        return redirect("/login")
    return render_page("settings.html", step=3, title="设置",
                       username=user["username"],
                       samples=db.count_templates(user["id"]),
                       threshold=THRESHOLD["value"])




if __name__ == "__main__":
    print("DB", db.DB_PATH)
    print("THRESHOLD", THRESHOLD["value"])
    print("PASSWORD_BACKEND", security.hash_backend())
    print("CAMERA_OPEN", engine.camera_ready())
    print("LIVENESS", LIVENESS_ENABLED)
    print("MODELS_DIR", MODELS_DIR)
    app.run(host="127.0.0.1", port=PORT, threaded=True)
