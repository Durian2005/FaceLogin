"""存储层自测：用户、模板、失败锁定、审计事件。

用临时数据目录运行，不碰正式数据库：
    FACELOGIN_DATA=<临时目录> python tools/test_storage.py
"""

import os
import sys
import time

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))

import db  # noqa: E402
import security  # noqa: E402
from face import pack, unpack  # noqa: E402

import numpy as np  # noqa: E402

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


print("DB:", db.init_db())

# ---------------------------------------------------------------- 用户
created, uid = db.create_user("probe_user", security.hash_password("Probe#12345"))
check("create_user", created, "id=%s" % uid)

again, err = db.create_user("PROBE_USER", security.hash_password("other"))
check("用户名大小写不敏感去重", (not again) and err == "username_taken", str(err))

user = db.get_user("probe_user")
check("get_user", user is not None and user["username"] == "probe_user")
check("密码哈希可校验", security.verify_password(user["pwd_hash"], "Probe#12345"))
check("错误密码不通过", not security.verify_password(user["pwd_hash"], "wrong-pass"))

# ---------------------------------------------------------------- 模板
feat = np.random.rand(1, 128).astype(np.float32)
blob, dim = pack(feat)
check("特征打包维度", dim == 128 and len(blob) == 128 * 4, "bytes=%d" % len(blob))
check("特征解包一致", np.allclose(unpack(blob, dim), feat.reshape(-1)))

n = db.add_template(uid, blob, dim)
check("add_template", n == 1, "count=%d" % n)

for _ in range(12):
    b, d = pack(np.random.rand(1, 128).astype(np.float32))
    n = db.add_template(uid, b, d)
check("模板数量上限 8", n == db.MAX_TEMPLATES_PER_USER == 8, "count=%d" % n)
check("templates 读取", len(db.templates(uid)) == 8)

# ---------------------------------------------------------------- 失败与锁定
f1, lock1 = db.bump_failure(uid, security.MAX_FAILS, security.LOCK_SECONDS)
f2, lock2 = db.bump_failure(uid, security.MAX_FAILS, security.LOCK_SECONDS)
f3, lock3 = db.bump_failure(uid, security.MAX_FAILS, security.LOCK_SECONDS)
check("失败计数累加", (f1, f2, f3) == (1, 2, 3), str((f1, f2, f3)))
check("第 1、2 次未锁定", lock1 == 0 and lock2 == 0)
check("第 3 次触发锁定", lock3 > time.time(), "+%ds" % int(lock3 - time.time()))

user = db.get_user("probe_user")
check("锁定状态已落库", user["locked_until"] > time.time())

db.clear_failure(uid)
user = db.get_user("probe_user")
check("clear_failure 清零", user["failed_count"] == 0 and user["locked_until"] == 0)

# ---------------------------------------------------------------- 审计
db.add_event("probe_user", "password", False, detail="bad_password_1")
db.add_event("probe_user", "face", True, similarity=0.8123, detail="ok")
db.add_event("other_user", "password", False, detail="unknown_user")
events = db.recent_events(limit=10, username="probe_user")
check("按用户查审计", len(events) == 2, "n=%d" % len(events))
check("审计相似度已保存", any(e["similarity"] == 0.8123 for e in events))
stats = db.event_stats()
check("全局事件统计", stats["total"] == 3 and stats["failed"] == 2, str(stats))

# ---------------------------------------------------------------- 删除
removed = db.delete_templates(uid)
check("删除模板", removed == 8 and db.count_templates(uid) == 0, "removed=%d" % removed)
check("删除用户", db.delete_user("probe_user") and db.get_user("probe_user") is None)

# ---------------------------------------------------------------- 会话密钥
s1 = db.get_or_create_secret()
s2 = db.get_or_create_secret()
check("会话密钥持久且稳定", s1 == s2 and len(s1) == 64)

print()
print("结果: %d 通过 / %d 失败" % (ok_count, fail_count))
sys.exit(1 if fail_count else 0)
