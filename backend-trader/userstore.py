"""Sign-in storage: users.db (kept separate from the account/ledger database)."""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
USERS_DB = DATA_DIR / "users.db"

SESSION_SECONDS = 30 * 24 * 3600
MAX_FAILURES = 5
LOCKOUT_SECONDS = 300

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}

_failures = {}  # username -> [count, first_failure_time]


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(USERS_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at    INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL
                           REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            """
        )


def _hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt,
        n=_SCRYPT["n"], r=_SCRYPT["r"], p=_SCRYPT["p"],
        dklen=_SCRYPT["dklen"],
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify_password(password, stored):
    try:
        _, salt_hex, digest_hex = stored.split("$")
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex),
            n=_SCRYPT["n"], r=_SCRYPT["r"], p=_SCRYPT["p"],
            dklen=_SCRYPT["dklen"],
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


_DUMMY_HASH = _hash_password("not-a-real-password")


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def validate(username, password):
    if not USERNAME_RE.match(username or ""):
        raise ValueError(
            "Username must be 3-32 characters: letters, numbers, . _ -"
        )
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters")
    if len(password) > 200:
        raise ValueError("Password is too long")


def create_user(username, password):
    validate(username, password)

    try:
        with connect() as db:
            cur = db.execute(
                "INSERT INTO users (username, password_hash, created_at) "
                "VALUES (?, ?, ?)",
                (username, _hash_password(password), int(time.time())),
            )
            return {"id": cur.lastrowid, "username": username}
    except sqlite3.IntegrityError:
        raise ValueError("That username is already taken")


def is_locked(username):
    entry = _failures.get(username.lower())

    if not entry:
        return False

    count, first = entry

    if time.time() - first > LOCKOUT_SECONDS:
        _failures.pop(username.lower(), None)
        return False

    return count >= MAX_FAILURES


def authenticate(username, password):
    """Returns the user dict, or None. Raises PermissionError if locked."""
    key = (username or "").lower()

    if is_locked(key):
        raise PermissionError("Too many attempts. Try again in a few minutes.")

    with connect() as db:
        row = db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username or "",),
        ).fetchone()

    # always run a hash so unknown users take the same time as known ones
    ok = _verify_password(password or "", row["password_hash"] if row else _DUMMY_HASH)

    if row and ok:
        _failures.pop(key, None)
        return {"id": row["id"], "username": row["username"]}

    count, first = _failures.get(key, [0, time.time()])
    _failures[key] = [count + 1, first]
    return None


def create_session(user_id):
    token = secrets.token_urlsafe(32)

    with connect() as db:
        db.execute("DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))
        db.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), user_id, int(time.time()) + SESSION_SECONDS),
        )

    return token


def user_for_token(token):
    if not token:
        return None

    with connect() as db:
        row = db.execute(
            "SELECT u.id, u.username FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_token_hash(token), int(time.time())),
        ).fetchone()

    return {"id": row["id"], "username": row["username"]} if row else None


def delete_session(token):
    with connect() as db:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
