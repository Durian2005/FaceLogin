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

# ---------------------------------------------------------------- 汇总

print()
print("通过 %d 项，失败 %d 项" % (ok_count, fail_count))
sys.exit(1 if fail_count else 0)
