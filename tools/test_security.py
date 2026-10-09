"""安全加固的离线自测：请求来源校验、页面令牌、阈值降级确认、审计哈希链。

全部离线：不碰摄像头、不碰正式数据库、不加载 38MB 模型。

    .venv\\Scripts\\python.exe tools\\test_security.py

两个刻意的设计：
- **把 face 模块换成桩**。真实 FaceEngine 在构造时就会打开摄像头，
  跑一次测试把摄像头指示灯点亮是不合适的，而且没必要加载模型。
- **数据目录指向临时目录**（FACELOGIN_DATA），所以怎么折腾都不会动到
  你自己的 data/faces.db。测试最后会故意把临时库改坏以验证篡改检出。
"""

import os
import sys
import tempfile
import types

TMP_DATA = tempfile.mkdtemp(prefix="facelogin-sec-")
os.environ["FACELOGIN_DATA"] = TMP_DATA
# 关掉"关页自动退出"的看护线程，否则它会直接 os._exit 掉测试进程
os.environ["FACELOGIN_AUTO_EXIT"] = "0"

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))

import reqguard  # noqa: E402

import db  # noqa: E402

# ---- 用桩替掉 face（必须在 import server 之前）----
_stub = types.ModuleType("face")


class _StubEngine:
    def __init__(self, *a, **k):
        pass

    def camera_ready(self):
        return False

    def read_frame(self):
        return None

    def detect_largest(self, frame):
        return None

    def draw(self, frame, face):
        return frame

    def capture_many(self, count=3):
        return []

    def capture_feature(self):
        return None, "no_face_detected", 0.0

    def best_cosine(self, refs, feat):
        return 0.0


_stub.FaceEngine = _StubEngine
_stub.pack = lambda vec: (b"", 0)
_stub.unpack = lambda blob, dim: None
# liveness 直接 import 了这个函数，桩必须一并提供，否则 server 导入就会失败
_stub.pose_metrics = lambda landmarks: {"yaw": 0.0, "pitch": 0.0}
sys.modules["face"] = _stub

import server  # noqa: E402

ok_count = 0
fail_count = 0


def check(name, condition, extra=""):
    global ok_count, fail_count
    if condition:
        ok_count += 1
        print("  PASS  " + name + (("  " + extra) if extra else ""))
    else:
        fail_count += 1
        print("  FAIL  " + name + (("  " + extra) if extra else ""))


def chain_snapshot():
    """把整条链的哈希抓一份快照，用来证明校验过程确实没写库。"""
    with db.connect() as conn:
        return [tuple(r) for r in conn.execute(
            "SELECT id, prev_hash, entry_hash FROM login_events ORDER BY id")]


print("临时数据目录:", TMP_DATA)
HOSTS = reqguard.allowed_hosts(5000)
SAME_ORIGIN = "http://127.0.0.1:5000"

# ---------------------------------------------------------------- Host 白名单

print("\n[1] Host 白名单（防 DNS rebinding）")
check("接受 127.0.0.1:5000", reqguard.host_allowed("127.0.0.1:5000", HOSTS))
check("接受 localhost:5000", reqguard.host_allowed("localhost:5000", HOSTS))
check("接受裸 localhost", reqguard.host_allowed("localhost", HOSTS))
check("大小写不敏感", reqguard.host_allowed("LOCALHOST:5000", HOSTS))
check("拒绝攻击者域名", not reqguard.host_allowed("evil.com:5000", HOSTS))
check("拒绝看起来像 localhost 的域名",
      not reqguard.host_allowed("attacker.localhost.evil.com", HOSTS))
check("拒绝空 Host", not reqguard.host_allowed("", HOSTS))
check("拒绝 None", not reqguard.host_allowed(None, HOSTS))
# Host 校验刻意**不看端口** —— 详细理由见 reqguard.host_allowed 的注释。
# 这条是改端口就全站 403 那个自伤故障的回归锁。
check("回环名 + 任意端口 => 放行（端口不是承重点）",
      reqguard.host_allowed("127.0.0.1:49975", HOSTS))
check("IPv6 回环 + 任意端口 => 放行",
      reqguard.host_allowed("[::1]:49975", HOSTS))
# 但「主机名对、尾巴是垃圾」不能骗过去（手工按冒号切就会栽在这）
check("拒绝 127.0.0.1:8080.evil.com 这类拼接",
      not reqguard.host_allowed("127.0.0.1:8080.evil.com", HOSTS))
check("拒绝带 userinfo 的 Host（evil.com@127.0.0.1）",
      not reqguard.host_allowed("evil.com@127.0.0.1", HOSTS))

# ---------------------------------------------------------------- Origin

print("\n[2] Origin / Referer 校验（防跨站请求伪造）")
check("两个头都缺 => 放行（curl 这类非浏览器客户端）",
      reqguard.origin_allowed(None, None, HOSTS))
check("同源 Origin => 放行", reqguard.origin_allowed(SAME_ORIGIN, None, HOSTS))
check("同源 Referer => 放行",
      reqguard.origin_allowed(None, "http://127.0.0.1:5000/index.html", HOSTS))
check("跨站 Origin => 拒绝", not reqguard.origin_allowed("http://evil.com", None, HOSTS))
check("跨站 Referer => 拒绝", not reqguard.origin_allowed(None, "https://evil.com/x", HOSTS))
check("Origin: null（sandbox iframe / file://）=> 拒绝",
      not reqguard.origin_allowed("null", None, HOSTS))
check("同主机但别的端口 => 拒绝",
      not reqguard.origin_allowed("http://127.0.0.1:8080", None, HOSTS))

# ---------------------------------------------------------------- 页面令牌

print("\n[3] 页面令牌")
TOKEN = reqguard.new_page_token()
check("正确令牌 => 通过", reqguard.token_matches(TOKEN, TOKEN))
check("错误令牌 => 拒绝", not reqguard.token_matches("guessed", TOKEN))
check("空令牌 => 拒绝", not reqguard.token_matches("", TOKEN))
check("None => 拒绝", not reqguard.token_matches(None, TOKEN))
check("每次生成都不同", reqguard.new_page_token() != reqguard.new_page_token())

# ---------------------------------------------------------------- HTTP 层

print("\n[4] HTTP 层：Host / Origin 真的生效")
client = server.app.test_client()
GOOD = server.PAGE_TOKEN

resp = client.get("/", headers={"Host": "evil.com:5000"})
check("Host 是攻击者域名时访问首页 => 403", resp.status_code == 403,
      "got %d" % resp.status_code)
resp = client.get("/api/state", headers={"Host": "evil.com:5000"})
check("Host 非法时 /api/state 也被拦 => 403", resp.status_code == 403,
      "got %d" % resp.status_code)

home = client.get("/")
check("正常 Host 访问首页 => 200", home.status_code == 200, "got %d" % home.status_code)
html = home.get_data(as_text=True)
check("页面令牌已注入（占位符被替换掉）",
      "<!--PAGETOKEN-->" not in html and GOOD in html)
# 这条是真机探针抓出来的坑：占位符如果只是被替换成裸文本，HTML 里就多出一行
# 无意义的令牌，而前端 querySelector 取不到它 —— 摄像头画面会一直 403。
check("令牌确实落在 meta 的 content 里（不是裸文本）",
      ('name="page-token" content="%s"' % GOOD) in html)
check("模板其余部分未被破坏（自动退出提示位仍在处理）",
      "<!--AUTOEXIT-->" not in html)

# 真分页之后模板换成 Jinja 继承，`<!--AUTOEXIT-->` 这个占位符本身就不存在了
# —— 上面那句会**永远为真**、等于空转。所以这里补一条真正检查渲染结果的断言，
# 而不是让一句空转的断言留在套件里假装覆盖了。
check("首页是多页之一（含步骤条，不是空壳）", 'class="steps"' in html)
if server.AUTO_EXIT:
    check("首页确实渲染出「关页即停止服务」提示",
          "自动停止服务" in html)
else:
    check("AUTO_EXIT 关闭时不渲染退出提示", "自动停止服务" not in html)

# ---------------------------------------------------------------- 真分页路由
# 这套系统的状态机现在跑在**服务端**：能不能进某张页面由它判定，走不进去就
# 重定向到该去的地方。这一节就是那道门禁的看门人。
print("\n[4b] 真分页路由：状态机在服务端")
OPEN_WHEN_ANON = ("/", "/register", "/login")
for path in ("/", "/register", "/login", "/enroll", "/console", "/settings"):
    r = client.get(path, follow_redirects=False)
    if path in OPEN_WHEN_ANON:
        check("未登录可直接打开 %-10s => 200" % path, r.status_code == 200,
              "got %d" % r.status_code)
    else:
        check("未登录访问 %-10s => 302 /login" % path,
              r.status_code == 302 and r.headers.get("Location", "").endswith("/login"),
              "%d %s" % (r.status_code, r.headers.get("Location")))

resp = client.post("/api/register",
                   json={"username": "sec_probe", "password": "Probe#12345"},
                   headers={"Origin": "http://evil.com"})
check("跨站 Origin 注册 => 403", resp.status_code == 403, "got %d" % resp.status_code)
check("被拒的请求没有建号", db.get_user("sec_probe") is None)

resp = client.post("/api/register",
                   json={"username": "sec_probe", "password": "Probe#12345"},
                   headers={"Origin": SAME_ORIGIN})
check("同源 Origin 注册 => 成功（且已登录）",
      resp.status_code == 200 and resp.get_json().get("ok"),
      str(resp.get_json())[:80])

# 已登录但这个人还没录人脸：主控台不该给他一张空页面，而应送回录入页。
r = client.get("/console", follow_redirects=False)
check("已登录但 0 组模板 => /console 送回 /enroll",
      r.status_code == 302 and r.headers.get("Location", "").endswith("/enroll"),
      "%d %s" % (r.status_code, r.headers.get("Location")))
r = client.get("/enroll", follow_redirects=False)
check("已登录 => /enroll 可进入", r.status_code == 200, "got %d" % r.status_code)
r = client.get("/register", follow_redirects=False)
check("已登录再访问 /register => 被送回流程内（不会重复注册）",
      r.status_code == 302 and r.headers.get("Location", "").endswith("/enroll"),
      "%d %s" % (r.status_code, r.headers.get("Location")))

# ---------------------------------------------------------------- 摄像头画面

print("\n[5] 摄像头画面必须带令牌")
resp = client.get("/api/stream")
check("无令牌 => 403", resp.status_code == 403, "got %d" % resp.status_code)
resp = client.get("/api/stream?t=wrong-token")
check("错误令牌 => 403", resp.status_code == 403, "got %d" % resp.status_code)

# 正向用例：把帧生成器换成有限序列 —— 真实的 MJPEG 是无限流，
# 测试客户端会把迭代器读到天荒地老。
_real_gen = server.gen_frames
server.gen_frames = lambda: iter([b"--frame\r\n\r\n"])
try:
    resp = client.get("/api/stream?t=" + GOOD)
    check("正确令牌 => 200", resp.status_code == 200, "got %d" % resp.status_code)
    check("响应类型是 MJPEG 流",
          "multipart/x-mixed-replace" in resp.headers.get("Content-Type", ""),
          resp.headers.get("Content-Type", ""))
finally:
    server.gen_frames = _real_gen

resp = client.post("/api/bye")
check("关页通知无令牌 => 403", resp.status_code == 403, "got %d" % resp.status_code)
resp = client.post("/api/bye?t=" + GOOD)
check("关页通知带令牌 => 200", resp.status_code == 200 and resp.get_json().get("ok"))

# ---------------------------------------------------------------- 阈值降级

print("\n[6] 阈值降级需要口令确认")
resp = client.post("/api/threshold", json={"value": 0.30})
body = resp.get_json()
check("调低但不给口令 => password_required",
      (not body.get("ok")) and body.get("error") == "password_required",
      str(body.get("error")))

resp = client.post("/api/threshold", json={"value": 0.30, "password": "wrong-one"})
body = resp.get_json()
check("调低但口令错误 => auth_failed",
      (not body.get("ok")) and body.get("error") == "auth_failed", str(body.get("error")))
check("阈值确实没被改动", abs(server.THRESHOLD["value"] - 0.50) < 1e-9,
      "=%s" % server.THRESHOLD["value"])

resp = client.post("/api/threshold", json={"value": 0.30, "password": "Probe#12345"})
body = resp.get_json()
check("调低且口令正确 => 生效", body.get("ok") and abs(body.get("threshold", 0) - 0.30) < 1e-9,
      str(body.get("message")))

resp = client.post("/api/threshold", json={"value": 0.60})
body = resp.get_json()
check("调高不收门槛 => 生效", body.get("ok") and abs(body.get("threshold", 0) - 0.60) < 1e-9)

# ---------------------------------------------------------------- 审计哈希链

print("\n[7] 审计哈希链")
good, detail = db.verify_audit_chain()
check("初始链条完好", good, str(detail))
baseline = detail["checked"]
check("链上已有记录（注册/阈值事件）", baseline > 0, "%d 条" % baseline)

db.add_event("chain_probe", "test", True, detail="one")
db.add_event("chain_probe", "test", False, detail="two")
good, detail = db.verify_audit_chain()
check("追加后仍然完好", good)
check("条数 +2", detail["checked"] == baseline + 2,
      "%d -> %d" % (baseline, detail["checked"]))

snap_before = chain_snapshot()
db.verify_audit_chain()
check("校验是只读的（前后哈希完全一致）", chain_snapshot() == snap_before)

with db.connect() as conn:
    victim = conn.execute("SELECT id FROM login_events ORDER BY id LIMIT 1").fetchone()["id"]
    conn.execute("UPDATE login_events SET success = 1, detail = 'forged' WHERE id = ?",
                 (victim,))

good, detail = db.verify_audit_chain()
check("把一条失败改成成功 => 检出", not good, "reason=%s" % detail.get("reason"))
check("断点精确定位到那一条", detail.get("broken_at") == victim,
      "broken_at=%s / victim=%s" % (detail.get("broken_at"), victim))
check("检出原因是内容被改", detail.get("reason") == "entry_hash_mismatch",
      str(detail.get("reason")))

with db.connect() as conn:
    conn.execute("UPDATE login_events SET prev_hash = ? WHERE id = ?", ("f" * 64, victim))
good, detail = db.verify_audit_chain()
check("只改链字段（prev_hash）=> 也检出",
      (not good) and detail.get("reason") == "prev_hash_mismatch", str(detail.get("reason")))

# ---------------------------------------------------------------- 真机 socket
#
# 为什么非要有这一节：Flask 的 test_client() 是**进程内直调 WSGI**，
# 它绕过"值是怎么送进浏览器"的那段链路。本项目就栽过一次 —— 模板里
# <!--PAGETOKEN--> 被替换成了一行裸文本而不是塞进 <meta content="...">，
# querySelector 取不到 → 令牌为空 → 摄像头画面一直 403。而 test_client
# 直接拿 server.PAGE_TOKEN 做断言，全绿。
# 所以这一节起一个**真 socket**，用 http.client 发原始请求，按浏览器实际
# 会在线上发出的字节来验。

print("\n[8] 真机 socket 探测（补 test_client 的盲区）")

import http.client  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

_srv = make_server("127.0.0.1", 0, server.app, threaded=True)
PORT = _srv.server_port
_srv_thread = threading.Thread(target=_srv.serve_forever, daemon=True)
_srv_thread.start()
print("  真服务已起在 127.0.0.1:%d" % PORT)

# 生产里 ALLOWED_HOSTS 由唯一的端口来源 PORT 推出，和 app.run 绑的端口天然一致。
# 测试为了不占用固定端口，绑的是随机端口，所以这里按「实际监听的端口」重建一次
# 白名单 —— 这不是绕过校验，而是把服务配置成它真实监听的地址，和生产同一个做法。
# （Host 那半不看端口，所以本来就不受影响；受影响的是要求端口精确匹配的 Origin。）
_orig_allowed = server.ALLOWED_HOSTS
server.ALLOWED_HOSTS = reqguard.allowed_hosts(PORT)
assert server.ALLOWED_HOSTS is not _orig_allowed


def raw(method, path, headers=None, body=None):
    """发一个原始 HTTP 请求。Host 头由调用方完全掌控（skip_host=True）。"""
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in (headers or {}).items():
            conn.putheader(k, v)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", "replace")
    finally:
        conn.close()


REAL_HOST = "127.0.0.1:%d" % PORT

try:
    # --- 真机上的 Host 校验 ---
    st, _ = raw("GET", "/", {"Host": "evil.com:5000"})
    check("真机上伪造 Host => 403", st == 403, "got %d" % st)
    st, _ = raw("GET", "/api/state", {"Host": "evil.com:%d" % PORT})
    check("真机上伪造 Host 打 /api/state => 403", st == 403, "got %d" % st)

    st, body = raw("GET", "/", {"Host": REAL_HOST})
    check("真机上正常 Host 访问首页 => 200", st == 200, "got %d" % st)
    # 这一条是整节的重点：从**真响应体**里把令牌抠出来，再拿它去打流接口。
    # 只有它能证明"令牌真的流到了浏览器能看到的位置"。
    m = re.search(r'name="page-token"\s+content="([^"]+)"', body)
    check("真响应体里能正则抠出页面令牌", m is not None)
    sock_token = m.group(1) if m else ""
    check("抠出来的令牌与实际一致", sock_token == server.PAGE_TOKEN,
          "sock=%s.. server=%s.." % (sock_token[:8], server.PAGE_TOKEN[:8]))

    # --- 真机上的 Origin 校验 ---
    st, _ = raw("POST", "/api/register",
                {"Host": REAL_HOST, "Origin": "http://evil.com",
                 "Content-Type": "application/json"},
                b'{"username":"sock_evil","password":"Probe#12345"}')
    check("真机上跨站 Origin 注册 => 403", st == 403, "got %d" % st)
    check("该请求确实没建号", db.get_user("sock_evil") is None)

    st, _ = raw("POST", "/api/register",
                {"Host": REAL_HOST, "Origin": "null",
                 "Content-Type": "application/json"},
                b'{"username":"sock_null","password":"Probe#12345"}')
    check("真机上 Origin: null => 403", st == 403, "got %d" % st)

    # Origin 校验必须**端口精确**：本机别的端口上的服务不得冒充本站。
    # 这条同时证明上面重建白名单没有把 Origin 检查放松掉。
    st, _ = raw("POST", "/api/register",
                {"Host": REAL_HOST, "Origin": "http://127.0.0.1:8080",
                 "Content-Type": "application/json"},
                b'{"username":"sock_other","password":"Probe#12345"}')
    check("真机上别的回环端口当 Origin => 403（端口精确匹配）",
          st == 403, "got %d" % st)
    check("该请求确实没建号", db.get_user("sock_other") is None)

    # 不带 Origin/Referer（curl、本机脚本）应当放行 —— 否则把自动化全挡死了
    st, body = raw("POST", "/api/register",
                   {"Host": REAL_HOST, "Content-Type": "application/json",
                    "Origin": "http://" + REAL_HOST},
                   b'{"username":"sock_ok","password":"Probe#12345"}')
    check("真机上同源 Origin 注册 => 200", st == 200, "got %d" % st)
    st, body = raw("POST", "/api/register",
                   {"Host": REAL_HOST, "Content-Type": "application/json"},
                   b'{"username":"sock_curl","password":"Probe#12345"}')
    check("真机上不带 Origin（curl 类客户端）=> 放行", st == 200, "got %d" % st)
    check("curl 类客户端确实建了号", db.get_user("sock_curl") is not None)

    # --- 真机上的流接口令牌 ---
    st, _ = raw("GET", "/api/stream", {"Host": REAL_HOST})
    check("真机上无令牌访问摄像头流 => 403", st == 403, "got %d" % st)
    st, _ = raw("GET", "/api/stream?t=wrong", {"Host": REAL_HOST})
    check("真机上错误令牌访问流 => 403", st == 403, "got %d" % st)

    _real_gen2 = server.gen_frames
    server.gen_frames = lambda: iter([b"--frame\r\n\r\n"])
    try:
        st, _ = raw("GET", "/api/stream?t=" + sock_token, {"Host": REAL_HOST})
        check("真机上用**从 HTML 抠出的**令牌访问流 => 200", st == 200, "got %d" % st)
    finally:
        server.gen_frames = _real_gen2

    # --- 真机上的 bye 令牌 ---
    st, _ = raw("POST", "/api/bye", {"Host": REAL_HOST})
    check("真机上无令牌 bye => 403", st == 403, "got %d" % st)
    st, body = raw("POST", "/api/bye?t=" + sock_token, {"Host": REAL_HOST})
    # 注意别断言 '"ok": true'：Flask 在非 debug/testing 下用紧凑 JSON（无空格），
    # 带空格的断言会在真机上永远失败。解析后判字段才稳。
    bye_ok = False
    try:
        bye_ok = bool(json.loads(body).get("ok"))
    except Exception:
        pass
    check("真机上带正确令牌 bye => 200", st == 200 and bye_ok, "got %d" % st)
finally:
    _srv.shutdown()
    server.ALLOWED_HOSTS = _orig_allowed

# ---------------------------------------------------------------- 汇总

print()
print("通过 %d 项，失败 %d 项" % (ok_count, fail_count))
sys.exit(1 if fail_count else 0)
