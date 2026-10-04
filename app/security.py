"""密码哈希与登录策略。

- 优先使用 Argon2id（argon2-cffi）；不可用时退回标准库 scrypt，绝不退回明文。
- 校验失败一律返回 False，不区分「哈希损坏」和「密码错误」，避免泄露信息。
"""

import hashlib
import hmac
import os
import secrets

try:  # pragma: no cover - 取决于运行环境
    from argon2 import PasswordHasher
    from argon2.exceptions import Argon2Error

    _hasher = PasswordHasher()
    HAVE_ARGON2 = True
except Exception:  # pragma: no cover
    _hasher = None
    HAVE_ARGON2 = False
    Argon2Error = Exception  # 占位，保证下面的 except 子句始终可解析

_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_LEN = 32

# 登录策略（可用环境变量覆盖）
MAX_FAILS = int(os.environ.get("FACELOGIN_MAX_FAILS", "3"))
LOCK_SECONDS = int(os.environ.get("FACELOGIN_LOCK_SECONDS", "180"))
DEFAULT_THRESHOLD = float(os.environ.get("FACELOGIN_THRESHOLD", "0.5"))
MAX_PASSWORD_LEN = 128
MIN_PASSWORD_LEN = 6


def hash_password(password: str) -> str:
    if HAVE_ARGON2:
        return _hasher.hash(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N,
        r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_LEN,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N, _SCRYPT_R, _SCRYPT_P, salt.hex(), digest.hex()
    )


def verify_password(stored: str, password: str) -> bool:
    if not stored or not password:
        return False
    try:
        if stored.startswith("$argon2") and HAVE_ARGON2:
            _hasher.verify(stored, password)
            return True
        if stored.startswith("scrypt$"):
            _, n, r, p, salt_hex, digest_hex = stored.split("$")
            digest = hashlib.scrypt(
                password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                n=int(n), r=int(r), p=int(p), dklen=len(digest_hex) // 2,
            )
            return hmac.compare_digest(digest.hex(), digest_hex)
    except Argon2Error:
        return False
    except Exception:
        return False
    return False


def dummy_verify():
    """用户名不存在时也消耗一次哈希时间，避免通过响应快慢猜出账号是否存在。"""
    try:
        hash_password("dummy-password-for-timing")
    except Exception:
        pass


def validate_password(password: str):
    if not password or len(password) < MIN_PASSWORD_LEN:
        return "password_too_short"
    if len(password) > MAX_PASSWORD_LEN:
        return "password_too_long"
    return None


def hash_backend() -> str:
    return "argon2id" if HAVE_ARGON2 else "scrypt"
