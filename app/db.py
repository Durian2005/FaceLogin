"""SQLite 存储层：用户、人脸模板、审计事件、设置。

设计要点
- 只存人脸特征向量（BLOB），不存原始照片。
- 密码只存哈希，不存明文。
- 失败计数与锁定时间跟用户记录放在一起，重启后仍然有效。
"""

import hashlib
import os
import secrets
import sqlite3
import time

import paths

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 打包成 exe 后 __file__ 会指向解压目录，所以程序根统一交给 paths 判断：
# 源码运行 = 项目根，打包运行 = exe 所在目录（数据跟着用户走，不会写进临时目录）。
PROJECT_DIR = paths.app_root()
DATA_DIR = os.environ.get("FACELOGIN_DATA", os.path.join(PROJECT_DIR, "data"))
DB_PATH = os.path.join(DATA_DIR, "faces.db")

# 每个用户最多保留的人脸模板数（超出后淘汰最旧的）
MAX_TEMPLATES_PER_USER = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    pwd_hash      TEXT    NOT NULL,
    created_at    REAL    NOT NULL,
    last_login_at REAL,
    failed_count  INTEGER NOT NULL DEFAULT 0,
    locked_until  REAL    NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS face_templates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vec        BLOB    NOT NULL,
    dim        INTEGER NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_user ON face_templates(user_id);

CREATE TABLE IF NOT EXISTS login_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL    NOT NULL,
    username   TEXT,
    method     TEXT    NOT NULL,
    success    INTEGER NOT NULL,
    similarity REAL,
    detail     TEXT,
    prev_hash  TEXT,
    entry_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON login_events(ts DESC);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate_audit_chain(conn)
    return DB_PATH


# ---------------------------------------------------------------- settings

def get_setting(key, default=None):
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key, value):
    with connect() as conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def get_or_create_secret():
    """会话签名密钥，落库保存，重启后已登录状态不会失效。"""
    existing = get_setting("session_secret")
    if existing:
        return existing
    value = secrets.token_hex(32)
    set_setting("session_secret", value)
    return value


# ---------------------------------------------------------------- users

def create_user(username, pwd_hash):
    now = time.time()
    try:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO users(username, pwd_hash, created_at) VALUES(?, ?, ?)",
                (username, pwd_hash, now),
            )
            return True, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, "username_taken"


def get_user(username):
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()


def get_user_by_id(user_id):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def list_users():
    with connect() as conn:
        return conn.execute(
            "SELECT u.id, u.username, u.created_at, u.last_login_at, "
            "       (SELECT COUNT(*) FROM face_templates t WHERE t.user_id = u.id) AS samples "
            "FROM users u ORDER BY u.id"
        ).fetchall()


def set_last_login(user_id):
    with connect() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ?, failed_count = 0, locked_until = 0 "
            "WHERE id = ?",
            (time.time(), user_id),
        )


def bump_failure(user_id, max_fails, lock_seconds):
    """记一次失败；达到上限则设置锁定时间。返回 (失败次数, 锁定截止时间)。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT failed_count FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        failed = (row["failed_count"] if row else 0) + 1
        locked_until = time.time() + lock_seconds if failed >= max_fails else 0
        conn.execute(
            "UPDATE users SET failed_count = ?, locked_until = ? WHERE id = ?",
            (failed, locked_until, user_id),
        )
    return failed, locked_until


def clear_failure(user_id):
    with connect() as conn:
        conn.execute(
            "UPDATE users SET failed_count = 0, locked_until = 0 WHERE id = ?", (user_id,)
        )


def delete_user(username):
    with connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE username = ? COLLATE NOCASE", (username,))
    return cur.rowcount > 0


# ---------------------------------------------------------------- templates

def add_template(user_id, vec_bytes, dim, max_keep=MAX_TEMPLATES_PER_USER):
    now = time.time()
    with connect() as conn:
        conn.execute(
            "INSERT INTO face_templates(user_id, vec, dim, created_at) VALUES(?, ?, ?, ?)",
            (user_id, vec_bytes, dim, now),
        )
        # 只保留最近的 max_keep 条
        conn.execute(
            "DELETE FROM face_templates WHERE user_id = ? AND id NOT IN ("
            "  SELECT id FROM face_templates WHERE user_id = ? ORDER BY id DESC LIMIT ?"
            ")",
            (user_id, user_id, max_keep),
        )
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM face_templates WHERE user_id = ?", (user_id,)
        ).fetchone()["c"]
    return n


def templates(user_id):
    with connect() as conn:
        rows = conn.execute(
            "SELECT vec, dim FROM face_templates WHERE user_id = ? ORDER BY id", (user_id,)
        ).fetchall()
    return [(row["vec"], row["dim"]) for row in rows]


def count_templates(user_id):
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM face_templates WHERE user_id = ?", (user_id,)
        ).fetchone()["c"]


def delete_templates(user_id):
    with connect() as conn:
        cur = conn.execute("DELETE FROM face_templates WHERE user_id = ?", (user_id,))
    return cur.rowcount


# ---------------------------------------------------------------- audit
#
# 审计日志用**哈希链**串起来：每条记录都带上前一条的哈希，任何一条被改动
# （哪怕只改一个字符），从它往后的所有 entry_hash 都会对不上。
#
# 这解决的是一个很具体的弱点：审计表的价值全在"事后能证明发生过什么"，
# 而在此之前，谁拿到 data/faces.db 都能直接 UPDATE 一条记录把登录失败
# 改成成功，或者删掉自己的痕迹，事后完全看不出来。
#
# 边界（必须说清楚）：哈希链只能证明"被改过"，不能阻止改动。攻击者可以
# 把整条链重算一遍 —— 除非链头被锚定在别处。本项目是本地单机，链头就存在
# 同一个库里，所以它防的是"随手改一条"，不是"有备而来的重算"。

GENESIS_HASH = "0" * 64


def _entry_payload(prev_hash, ts, username, method, success, similarity, detail):
    """把一条审计事件序列化成确定性的字节串（同一行永远得到同一结果）。

    用 \\x1f 当分隔符，避免字段内容里恰好有分隔符导致不同记录哈希相同。
    浮点用 repr 而非格式化：repr 是往返精确的，重新读回来后必然一致。
    """
    return "\x1f".join([
        prev_hash or "",
        repr(float(ts)),
        username or "",
        method or "",
        "1" if success else "0",
        "" if similarity is None else repr(float(similarity)),
        detail or "",
    ])


def _hash_entry(prev_hash, ts, username, method, success, similarity, detail):
    payload = _entry_payload(prev_hash, ts, username, method, success, similarity, detail)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _migrate_audit_chain(conn):
    """首次升级到此版本时，给历史记录补算哈希链。

    只在**列刚被创建出来的这一次**回填。之后无论出现什么情况都不再自动重算 ——
    否则"启动时自动修复链条"就等于把篡改也一并修好，反而抹掉了证据。
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(login_events)")}
    if "entry_hash" in cols:
        return 0
    conn.execute("ALTER TABLE login_events ADD COLUMN prev_hash TEXT")
    conn.execute("ALTER TABLE login_events ADD COLUMN entry_hash TEXT")

    rows = conn.execute(
        "SELECT id, ts, username, method, success, similarity, detail "
        "FROM login_events ORDER BY id"
    ).fetchall()
    prev = GENESIS_HASH
    for row in rows:
        entry = _hash_entry(prev, row["ts"], row["username"], row["method"],
                            row["success"], row["similarity"], row["detail"])
        conn.execute("UPDATE login_events SET prev_hash = ?, entry_hash = ? WHERE id = ?",
                     (prev, entry, row["id"]))
        prev = entry
    return len(rows)


def add_event(username, method, success, similarity=None, detail=None):
    ts = time.time()
    with connect() as conn:
        # BEGIN IMMEDIATE：先拿到写锁再读链尾。否则两个线程可能都读到同一个
        # 链尾、各接一支，链条就分叉了（推流线程与登录线程是真并发）。
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT entry_hash FROM login_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev = (row["entry_hash"] if row and row["entry_hash"] else None) or GENESIS_HASH
        entry = _hash_entry(prev, ts, username, method, success, similarity, detail)
        conn.execute(
            "INSERT INTO login_events(ts, username, method, success, similarity, "
            "                        detail, prev_hash, entry_hash) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, username, method, 1 if success else 0, similarity, detail, prev, entry),
        )


def verify_audit_chain():
    """校验链条完整性。**只读**：发现问题只报告，绝不自动修复。

    返回 (是否完好, 详情)。详情里带失败原因与断点 id，
    便于 tools/verify_audit_chain.py 定位到具体是哪一条被动过。
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, username, method, success, similarity, detail, "
            "       prev_hash, entry_hash FROM login_events ORDER BY id"
        ).fetchall()

    prev = GENESIS_HASH
    checked = 0
    for row in rows:
        if (row["prev_hash"] or "") != prev:
            return False, {"checked": checked, "broken_at": row["id"],
                           "reason": "prev_hash_mismatch",
                           "expected": prev[:16], "found": (row["prev_hash"] or "")[:16]}
        expect = _hash_entry(prev, row["ts"], row["username"], row["method"],
                             row["success"], row["similarity"], row["detail"])
        if row["entry_hash"] != expect:
            return False, {"checked": checked, "broken_at": row["id"],
                           "reason": "entry_hash_mismatch",
                           "expected": expect[:16], "found": (row["entry_hash"] or "")[:16]}
        prev = expect
        checked += 1
    return True, {"checked": checked, "head": prev}


def recent_events(limit=20, username=None):
    sql = ("SELECT ts, username, method, success, similarity, detail FROM login_events "
           "{where} ORDER BY id DESC LIMIT ?").format(
        where="WHERE username = ? COLLATE NOCASE" if username else "")
    params = (username, limit) if username else (limit,)
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def event_stats():
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed "
            "FROM login_events"
        ).fetchone()
    return {"total": row["total"] or 0, "failed": row["failed"] or 0}
