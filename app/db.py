"""SQLite 存储层：用户、人脸模板、审计事件、设置。

设计要点
- 只存人脸特征向量（BLOB），不存原始照片。
- 密码只存哈希，不存明文。
- 失败计数与锁定时间跟用户记录放在一起，重启后仍然有效。
"""

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
    detail     TEXT
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

def add_event(username, method, success, similarity=None, detail=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO login_events(ts, username, method, success, similarity, detail) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (time.time(), username, method, 1 if success else 0, similarity, detail),
        )


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
