"""Persistent web-only users, sessions, document access labels and conversation state.

The CSV judge core has no user identities and does not import this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time

SESSION_SECONDS = 8 * 60 * 60
PBKDF2_ITERATIONS = 310_000


class AccessError(ValueError):
    def __init__(self, message, status=403):
        super().__init__(message)
        self.status = status


def _level(value):
    if type(value) is not int or value not in {1, 2}:
        raise AccessError('Уровень доступа должен быть 1 или 2.', 400)
    return value


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), PBKDF2_ITERATIONS).hex()


def _identifier(value, limit=128):
    return isinstance(value, str) and 1 <= len(value) <= limit and bool(re.fullmatch(r'[A-Za-z0-9_.:-]+', value))


class AccessStore:
    def __init__(self, path, users=None):
        self.lock = threading.RLock()
        self.credentials_path = None
        self._attempts = {}
        if str(path) == ':memory:':
            filename = ':memory:'
            parent = None
        else:
            candidate = Path(path)
            filename = candidate if candidate.suffix in {'.sqlite', '.sqlite3', '.db'} else candidate / 'access.sqlite'
            parent = filename.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = filename
        self.db = sqlite3.connect(str(filename), check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA busy_timeout = 10000')
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, salt TEXT NOT NULL,
                password_hash TEXT NOT NULL, level INTEGER NOT NULL CHECK(level IN (1,2)));
            CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL,
                csrf_token TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS document_access (doc_id TEXT PRIMARY KEY,
                level INTEGER NOT NULL CHECK(level IN (1,2)));
            CREATE TABLE IF NOT EXISTS conversations (username TEXT NOT NULL, conversation_id TEXT NOT NULL,
                revision INTEGER NOT NULL, history TEXT NOT NULL, dialogue_state TEXT, display TEXT,
                PRIMARY KEY(username,conversation_id));
        """)
        if 'display' not in {row[1] for row in self.db.execute('PRAGMA table_info(conversations)')}:
            self.db.execute('ALTER TABLE conversations ADD COLUMN display TEXT')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('revision', '0')")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('secret', ?)", (secrets.token_hex(32),))
        self.db.commit()
        if parent is not None:
            os.chmod(filename, 0o600)
        existing = self.db.execute('SELECT count(*) FROM users').fetchone()[0]
        if not existing:
            generated = users is None
            if users is None:
                users = {
                    'user': {'level': 1, 'password': os.getenv('PSB_USER_PASSWORD') or secrets.token_urlsafe(18)},
                    'admin': {'level': 2, 'password': os.getenv('PSB_ADMIN_PASSWORD') or secrets.token_urlsafe(18)},
                }
            if not isinstance(users, dict) or not users:
                raise ValueError('At least one user is required')
            checked = []
            for username, item in users.items():
                if not _identifier(username, 64) or not isinstance(item, dict):
                    raise ValueError('Invalid account configuration')
                level = _level(item.get('level'))
                password = item.get('password')
                if not isinstance(password, str) or not 8 <= len(password) <= 1024:
                    raise ValueError('Passwords must contain 8–1024 characters')
                salt = secrets.token_hex(16)
                checked.append((username, salt, _password_hash(password, salt), level))
            with self.db:
                self.db.executemany('INSERT INTO users VALUES (?,?,?,?)', checked)
            if generated and parent is not None:
                self.credentials_path = parent / 'initial-credentials.txt'
                lines = ['Учётные записи созданы при первом запуске. Не публикуйте этот файл.', '']
                for username, item in users.items():
                    lines.extend([f"Уровень {item['level']}: {username}", f"Пароль: {item['password']}", ''])
                temporary = parent / ('.credentials-' + secrets.token_hex(8))
                try:
                    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                        stream.write('\n'.join(lines))
                    os.replace(temporary, self.credentials_path)
                finally:
                    temporary.unlink(missing_ok=True)
        self._secret = bytes.fromhex(self.db.execute("SELECT value FROM metadata WHERE key='secret'").fetchone()[0])
        self._dummy_salt = secrets.token_hex(16)
        self._dummy_hash = _password_hash(secrets.token_urlsafe(20), self._dummy_salt)

    def close(self):
        with self.lock:
            self.db.close()

    def revision(self):
        with self.lock:
            return int(self.db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])

    def check_revision(self, revision):
        if type(revision) is not int or revision != self.revision():
            raise AccessError('Права доступа изменились. Начните новый диалог.', 409)

    def history_scope(self, username, revision):
        self.check_revision(revision)
        return hmac.new(self._secret, f'{username}:{revision}'.encode(), hashlib.sha256).hexdigest()

    def login(self, username, password):
        if not _identifier(username, 64) or not isinstance(password, str) or len(password) > 1024:
            raise AccessError('Неверный логин или пароль.', 401)
        with self.lock:
            now = time.time()
            attempts, until = self._attempts.get(username, (0, 0))
            if until > now:
                raise AccessError('Слишком много попыток входа. Повторите позже.', 429)
            if until:
                attempts = 0
            row = self.db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
            salt, expected = (row['salt'], row['password_hash']) if row else (self._dummy_salt, self._dummy_hash)
            actual = _password_hash(password, salt)
            if not hmac.compare_digest(actual, expected) or row is None:
                attempts += 1
                self._attempts[username] = (attempts, now + 60 if attempts >= 10 else 0)
                raise AccessError('Неверный логин или пароль.', 401)
            self._attempts.pop(username, None)
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            with self.db:
                self.db.execute('DELETE FROM sessions WHERE expires<=?', (now,))
                self.db.execute('INSERT INTO sessions VALUES (?,?,?,?)',
                    (hashlib.sha256(token.encode()).hexdigest(), username, csrf, now + SESSION_SECONDS))
            return token, csrf, {'username': username, 'level': row['level']}

    def authenticate(self, token):
        if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
            raise AccessError('Войдите в систему.', 401)
        with self.lock:
            row = self.db.execute('SELECT users.username, users.level, sessions.csrf_token, sessions.expires '
                'FROM sessions JOIN users USING(username) WHERE token_hash=?',
                (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
            if row is None or row['expires'] <= time.time():
                raise AccessError('Сеанс завершён. Войдите снова.', 401)
            return {key: row[key] for key in ('username', 'level', 'csrf_token')}

    def logout(self, token):
        if not isinstance(token, str):
            return
        with self.lock, self.db:
            self.db.execute('DELETE FROM sessions WHERE token_hash=?',
                (hashlib.sha256(token.encode()).hexdigest(),))

    def snapshot(self, level):
        _level(level)
        with self.lock:
            revision = self.revision()
            labels = dict(self.db.execute('SELECT doc_id, level FROM document_access'))
            return revision, labels

    def set_level(self, doc_id, level):
        _level(level)
        if not isinstance(doc_id, str) or not re.fullmatch(r'doc_[a-f0-9]{16}', doc_id):
            raise AccessError('Некорректный документ.', 400)
        with self.lock, self.db:
            old = self.db.execute('SELECT level FROM document_access WHERE doc_id=?', (doc_id,)).fetchone()
            if old is not None and old['level'] == level:
                return self.revision()
            self.db.execute('INSERT INTO document_access VALUES (?,?) ON CONFLICT(doc_id) DO UPDATE SET level=excluded.level', (doc_id, level))
            self.db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
            # Old conversations remain stored but are unavailable to every
            # current-revision read, including the model context.
            return self.revision()

    def get_conversation(self, username, conversation_id, revision):
        if not _identifier(username, 64) or not _identifier(conversation_id):
            raise AccessError('Некорректный диалог.', 400)
        with self.lock:
            self.check_revision(revision)
            row = self.db.execute('SELECT * FROM conversations WHERE username=? AND conversation_id=?',
                (username, conversation_id)).fetchone()
            if row is None:
                return {'history': [], 'dialogue_state': None, 'display': None}
            if row['revision'] != revision:
                raise AccessError('Права доступа изменились. Начните новый диалог.', 409)
            return {'history': json.loads(row['history']),
                    'dialogue_state': json.loads(row['dialogue_state']) if row['dialogue_state'] else None,
                    'display': json.loads(row['display']) if row['display'] else None}

    def save_conversation(self, username, conversation_id, revision, history, dialogue_state=None, display=None):
        if not _identifier(username, 64) or not _identifier(conversation_id):
            raise AccessError('Некорректный диалог.', 400)
        if not isinstance(history, list) or len(history) > 16:
            raise AccessError('История слишком длинная.', 400)
        checked = []
        for item in history:
            if (not isinstance(item, dict) or set(item) != {'role', 'content'}
                    or item.get('role') not in {'user', 'assistant'}
                    or not isinstance(item.get('content'), str) or len(item['content']) > 12040):
                raise AccessError('Некорректная история диалога.', 400)
            checked.append(item)
        if sum(len(x['content']) for x in checked) > 60000:
            raise AccessError('История слишком длинная.', 400)
        raw_state = json.dumps(dialogue_state, ensure_ascii=False) if dialogue_state is not None else None
        if raw_state and (not isinstance(dialogue_state, dict) or len(raw_state) > 40000):
            raise AccessError('Некорректное состояние диалога.', 400)
        raw_display = json.dumps(display, ensure_ascii=False) if display is not None else None
        if raw_display and (not isinstance(display, dict) or display.get('id') != conversation_id or len(raw_display) > 2_000_000):
            raise AccessError('История диалога слишком большая. Начните новый диалог.', 413)
        with self.lock, self.db:
            self.check_revision(revision)
            self.db.execute('INSERT INTO conversations VALUES (?,?,?,?,?,?) '
                'ON CONFLICT(username,conversation_id) DO UPDATE SET revision=excluded.revision, history=excluded.history, dialogue_state=excluded.dialogue_state, display=excluded.display',
                (username, conversation_id, revision, json.dumps(checked, ensure_ascii=False), raw_state, raw_display))

    def list_conversations(self, username, revision):
        with self.lock:
            self.check_revision(revision)
            rows = self.db.execute('SELECT display FROM conversations WHERE username=? AND revision=? AND display IS NOT NULL',
                (username, revision)).fetchall()
            result = [json.loads(row['display']) for row in rows]
            return sorted(result, key=lambda item: item.get('updatedAt', 0), reverse=True)

    def clear_conversations(self, username):
        with self.lock, self.db:
            self.db.execute('DELETE FROM conversations WHERE username=?', (username,))

    def delete_conversation(self, username, conversation_id):
        if not _identifier(username, 64) or not _identifier(conversation_id):
            raise AccessError('Некорректный диалог.', 400)
        with self.lock, self.db:
            self.db.execute('DELETE FROM conversations WHERE username=? AND conversation_id=?',
                (username, conversation_id))
