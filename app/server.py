"""本地人脸登录服务：注册、密码登录、人脸 1:1 登录、会话、失败限速、审计。"""

import os
import re
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, session

import db
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
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    user = logged_in_user()
    return jsonify({
        "faces": last_state["faces"],
        "det_score": last_state["score"],
        "threshold": THRESHOLD["value"],
        "password_backend": security.hash_backend(),
        "logged_in": user is not None,
        "username": user["username"] if user else None,
        "face_samples": db.count_templates(user["id"]) if user else 0,
    })


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
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


@app.route("/api/login/face", methods=["POST"])
def api_login_face():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "missing_field", "message": "请填写用户名"})

    user = db.get_user(username)
    remaining = lock_remaining(user)
    if remaining > 0:
        db.add_event(username, "face", False, detail="locked")
        return jsonify({"ok": False, "error": "locked", "retry_after": remaining,
                        "message": "账号已临时锁定，请 %d 秒后再试" % remaining})

    if user is None:
        db.add_event(username, "face", False, detail="unknown_user")
        return jsonify({"ok": False, "error": "auth_failed", "message": "用户名或人脸不匹配"})

    stored = db.templates(user["id"])
    if not stored:
        db.add_event(username, "face", False, detail="no_template")
        return jsonify({"ok": False, "error": "no_template",
                        "message": "该账号还没有录入人脸，请先登录后录入"})

    feat, status, ms = engine.capture_feature()
    if feat is None:
        db.add_event(username, "face", False, detail=status)
        return jsonify({"ok": False, "error": status,
                        "message": "没有检测到人脸，请正对摄像头再试"})

    refs = [unpack(blob, dim) for blob, dim in stored]
    refs = [r for r in refs if r is not None]
    score = engine.best_cosine(refs, feat)
    passed = score >= THRESHOLD["value"]

    if passed:
        db.set_last_login(user["id"])
        db.add_event(username, "face", True, similarity=round(score, 4), detail="ok")
        start_session(user)
        return jsonify({"ok": True, "username": user["username"], "cosine": round(score, 4),
                        "threshold": THRESHOLD["value"], "ms": ms,
                        "message": "人脸验证通过"})

    failed, locked_until = db.bump_failure(user["id"], security.MAX_FAILS,
                                           security.LOCK_SECONDS)
    db.add_event(username, "face", False, similarity=round(score, 4),
                 detail="below_threshold_%d" % failed)
    if locked_until > time.time():
        db.add_event(username, "lockout", False, detail="locked_%ds" % security.LOCK_SECONDS)
        return jsonify({"ok": False, "error": "locked", "retry_after": security.LOCK_SECONDS,
                        "cosine": round(score, 4), "threshold": THRESHOLD["value"], "ms": ms,
                        "message": "连续失败 %d 次，账号锁定 %d 秒"
                                   % (security.MAX_FAILS, security.LOCK_SECONDS)})
    return jsonify({"ok": False, "error": "auth_failed", "cosine": round(score, 4),
                    "threshold": THRESHOLD["value"], "ms": ms, "failed": failed,
                    "message": "人脸相似度 %.4f 低于阈值 %.2f"
                               % (score, THRESHOLD["value"])})


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

    feats = engine.capture_many(count=3)
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
    try:
        value = float((request.get_json(silent=True) or {}).get("value"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_value"})
    if not 0.20 <= value <= 0.95:
        return jsonify({"ok": False, "error": "out_of_range",
                        "message": "阈值请设在 0.20 到 0.95 之间"})
    db.set_setting("threshold", value)
    THRESHOLD["value"] = value
    db.add_event(user["username"], "threshold", True, detail="set_%.2f" % value)
    return jsonify({"ok": True, "threshold": value, "message": "阈值已更新为 %.2f" % value})


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>人脸登录系统</title>
<style>
:root{--bg:#15181c;--card:#232a32;--line:#3a434e;--line2:#4a5563;--tx:#e8e8e8;--tx2:#b8c0ca}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);margin:0;padding:20px}
h1{font-size:18px;font-weight:500;margin:0 0 4px}
.sub{font-size:12px;color:var(--tx2);margin-bottom:16px}
.wrap{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
.col{display:flex;flex-direction:column;gap:14px;flex:1;min-width:300px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h2{font-size:14px;font-weight:500;margin:0 0 10px;color:#d5dde6}
img{width:100%;max-width:520px;border-radius:8px;background:#000;display:block}
label{display:block;font-size:12px;color:var(--tx2);margin:8px 0 4px}
input{width:100%;padding:8px;border-radius:6px;border:1px solid var(--line2);background:#1b2026;color:var(--tx);font-size:13px}
button{margin:10px 8px 0 0;padding:8px 13px;border-radius:6px;border:1px solid var(--line2);background:#2d3742;color:var(--tx);cursor:pointer;font-size:13px}
button:hover{background:#38434f}
button:disabled{opacity:.45;cursor:not-allowed}
.bar{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--tx2);margin-top:10px;line-height:1.8}
.bar b{color:#e8e8e8;font-weight:500}
.msg{margin-top:10px;font-size:13px;line-height:1.6;min-height:20px}
.ok{color:#4ade80}.bad{color:#f87171}.warn{color:#fbbf24}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid #313a44;color:var(--tx2)}
th{color:#8d97a3;font-weight:500}
td.y{color:#4ade80}td.n{color:#f87171}
.hint{font-size:12px;color:#8d97a3;line-height:1.7;margin-top:8px}
.row{display:flex;gap:10px}.row>div{flex:1}
</style></head><body>
<h1>本地人脸登录系统</h1>
<div class="sub">离线运行 · 模板存 SQLite · 密码用 Argon2id 哈希 · 失败 3 次锁定 3 分钟 <!--AUTOEXIT--></div>
<div class="wrap">
  <div class="col" style="max-width:540px">
    <div class="card">
      <img src="/api/stream" alt="camera">
      <div class="bar">
        <span>检测：<b id="det">-</b></span>
        <span>阈值：<b id="thr">-</b></span>
        <span>密码哈希：<b id="pwd">-</b></span>
      </div>
    </div>
    <div class="card">
      <h2>登录状态</h2>
      <div class="bar"><span>当前用户：<b id="who">未登录</b></span><span>人脸模板：<b id="samples">0</b> 组</span></div>
      <button onclick="enroll()">录入人脸</button>
      <button onclick="logout()">退出登录</button>
      <label>删除人脸数据需再次输入密码</label>
      <input id="delpwd" type="password" placeholder="输入密码以删除本人人脸数据">
      <button onclick="deleteFace()">删除我的所有人脸数据</button>
      <div class="hint">人脸模板属于敏感生物特征信息，删除后该账号将无法再用人脸登录，只能改用密码。</div>
    </div>
  </div>

  <div class="col">
    <div class="card">
      <h2>登录</h2>
      <label>用户名</label><input id="lu" placeholder="请输入用户名" autocomplete="username" autofocus>
      <label>密码</label><input id="lp" type="password" placeholder="密码登录时填写">
      <button onclick="loginFace()">人脸登录</button>
      <button onclick="loginPassword()">密码登录</button>
      <div class="hint">人脸登录为「账号先行 + 1:1 比对」，无需遍历所有用户；密码登录可随时回退。</div>
    </div>
    <div class="card">
      <h2>注册新账号</h2>
      <label>用户名</label><input id="ru" placeholder="3-32 位字母数字下划线">
      <label>密码</label><input id="rp" type="password" placeholder="至少 6 位">
      <button onclick="register()">注册并登录</button>
      <div class="hint">注册后请立刻录入人脸；否则该账号只能用密码登录。</div>
    </div>
    <div class="card">
      <h2>阈值校准</h2>
      <div class="bar"><span>当前阈值：<b id="thr2">-</b></span></div>
      <input id="thrv" type="number" step="0.01" min="0.2" max="0.95" value="0.5">
      <button onclick="setThreshold()">更新阈值</button>
      <div class="hint">阈值越高越严格。先用本人多次样本测下限，再用他人样本测误接受，再定值。</div>
    </div>
    <div class="card">
      <h2>审计记录（本人）</h2>
      <table><thead><tr><th>时间</th><th>方式</th><th>结果</th><th>相似度</th><th>说明</th></tr></thead>
      <tbody id="audit"><tr><td colspan="5">登录后可见</td></tr></tbody></table>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let msgEl=null;
function flash(text,cls){
  if(!msgEl){msgEl=document.createElement('div');msgEl.className='card';msgEl.style.marginTop='14px';
    msgEl.innerHTML='<h2>操作结果</h2><div class="msg" id="msgtxt"></div>';document.body.appendChild(msgEl);}
  const t=$('msgtxt');t.className='msg '+(cls||'');t.textContent=text;
}
async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  return r.json();
}
async function guard(fn){try{await fn();}catch(e){flash('请求失败：'+e,'bad');}finally{refresh();}}

function register(){guard(async()=>{
  flash('正在注册…');
  const r=await post('/api/register',{username:$('ru').value.trim(),password:$('rp').value});
  flash(r.message||r.error, r.ok?'ok':'bad');
  if(r.ok)$('lu').value=r.username;
});}
function loginPassword(){guard(async()=>{
  flash('正在校验密码…');
  const r=await post('/api/login/password',{username:$('lu').value.trim(),password:$('lp').value});
  flash(r.message||r.error, r.ok?'ok':(r.error==='locked'?'warn':'bad'));
});}
function loginFace(){guard(async()=>{
  flash('正在验证，请正对摄像头…');
  const r=await post('/api/login/face',{username:$('lu').value.trim()});
  let extra=r.cosine!==undefined?'（相似度 '+r.cosine+' / 阈值 '+r.threshold+'，'+(r.ms||0)+'ms）':'';
  flash((r.message||r.error)+extra, r.ok?'ok':(r.error==='locked'?'warn':'bad'));
});}
function enroll(){guard(async()=>{
  flash('正在采集 3 组样本，请保持正对摄像头…');
  const r=await post('/api/enroll');
  flash(r.message||r.error, r.ok?'ok':'bad');
});}
function deleteFace(){guard(async()=>{
  if(!confirm('确定删除你的人脸模板吗？删除后需重新录入才能人脸登录。'))return;
  const r=await post('/api/face/delete',{password:$('delpwd').value});
  flash(r.message||r.error, r.ok?'ok':'bad');
  $('delpwd').value='';
});}
function logout(){guard(async()=>{const r=await post('/api/logout');flash(r.message,'ok');});}
function setThreshold(){guard(async()=>{
  const r=await post('/api/threshold',{value:parseFloat($('thrv').value)});
  flash(r.message||r.error, r.ok?'ok':'bad');
});}

async function refresh(){
  try{
    const s=await (await fetch('/api/state')).json();
    $('det').textContent = s.faces>0 ? ('1 张 · '+s.det_score) : '未检测到人脸';
    $('thr').textContent = s.threshold; $('thr2').textContent = s.threshold;
    $('pwd').textContent = s.password_backend;
    $('who').textContent = s.logged_in ? s.username : '未登录';
    $('samples').textContent = s.face_samples;
    if(s.logged_in){
      const a=await (await fetch('/api/audit?limit=15')).json();
      if(a.ok){
        $('audit').innerHTML = a.events.length? a.events.map(e=>
          '<tr><td>'+e.time+'</td><td>'+e.method+'</td><td class="'+(e.success?'y':'n')+'">'+
          (e.success?'成功':'失败')+'</td><td>'+(e.similarity??'-')+'</td><td>'+(e.detail||'')+'</td></tr>'
        ).join('') : '<tr><td colspan="5">暂无记录</td></tr>';
      }
    }else{
      $('audit').innerHTML='<tr><td colspan="5">登录后可见</td></tr>';
    }
  }catch(e){}
}
function notifyBye(){try{navigator.sendBeacon('/api/bye');}catch(e){}}
addEventListener('pagehide',notifyBye);
addEventListener('beforeunload',notifyBye);
setInterval(refresh,2000); refresh();
</script></body></html>"""


@app.route("/")
def index():
    if AUTO_EXIT:
        hint = ('<span style="color:#fbbf24">关闭本页约 %.0f 秒后自动停止服务、释放摄像头</span>'
                % EXIT_GRACE)
    else:
        hint = ""
    return Response(PAGE.replace("<!--AUTOEXIT-->", hint),
                    mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    print("DB", db.DB_PATH)
    print("THRESHOLD", THRESHOLD["value"])
    print("PASSWORD_BACKEND", security.hash_backend())
    print("CAMERA_OPEN", engine.camera_ready())
    print("MODELS_DIR", MODELS_DIR)
    app.run(host="127.0.0.1", port=5000, threaded=True)
