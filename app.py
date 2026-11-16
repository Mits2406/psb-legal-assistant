"""Local web workbench. Serves only the three public assets and a bounded JSON API.

Run: python app.py --host 127.0.0.1 --port 8000 --data-dir data
Users, document labels and account-scoped history are kept in a private SQLite
state directory. API credentials are never returned to the browser.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
import hmac
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import uuid
from urllib.parse import unquote, urlsplit

from llm import load_env
from access_control import AccessStore, AccessError, SESSION_SECONDS
from scoped_assistant import ScopedManager

ROOT = Path(__file__).resolve().parent
MAX_BODY_BYTES = 262_144
MAX_QUESTION_CHARS = 12_000
MAX_HISTORY_MESSAGE_CHARS = MAX_QUESTION_CHARS + 40  # Browser attaches an explicit event date.
MAX_HISTORY_MESSAGES = 16
MAX_HISTORY_CHARS = 60_000
MAX_DIALOGUE_STATE_CHARS = 40_000
STATIC_FILES = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/static/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/static/styles.css': ('styles.css', 'text/css; charset=utf-8'),
    '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
    '/static/history.js': ('history.js', 'text/javascript; charset=utf-8'),
}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_chat(data):
    if not isinstance(data, dict):
        raise RequestError('Ожидается JSON-объект с вопросом.')
    if set(data) - {'question', 'history', 'event_date', 'dialogue_state', 'reference_date'}:
        raise RequestError('В запросе есть неизвестные поля.')
    question = data.get('question')
    if not isinstance(question, str) or not question.strip():
        raise RequestError('Введите непустой вопрос.')
    if len(question) > MAX_QUESTION_CHARS:
        raise RequestError(f'Вопрос должен быть не длиннее {MAX_QUESTION_CHARS} символов.', 413)
    history = data.get('history', [])
    if not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES:
        raise RequestError(f'История должна содержать не более {MAX_HISTORY_MESSAGES} сообщений.')
    checked = []
    for item in history:
        if not isinstance(item, dict) or set(item) != {'role', 'content'}:
            raise RequestError('Каждое сообщение истории должно содержать role и content.')
        if not isinstance(item['role'], str) or item['role'] not in {'user', 'assistant'}:
            raise RequestError('Допустимые роли истории: user и assistant.')
        if not isinstance(item['content'], str) or not item['content'].strip():
            raise RequestError('Сообщения истории должны содержать непустой текст.')
        if len(item['content']) > MAX_HISTORY_MESSAGE_CHARS:
            raise RequestError('Сообщение истории слишком длинное.', 413)
        checked.append({'role': item['role'], 'content': item['content']})
    if sum(len(item['content']) for item in checked) > MAX_HISTORY_CHARS:
        raise RequestError('История слишком длинная. Начните новый диалог.', 413)
    event_date = data.get('event_date')
    for field in ('event_date', 'reference_date'):
        value = data.get(field)
        if value is not None:
            if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                raise RequestError('Дата должна иметь формат YYYY-MM-DD.')
            try:
                date.fromisoformat(value)
            except ValueError:
                raise RequestError('Укажите существующую календарную дату.') from None
    state = data.get('dialogue_state')
    if state is not None:
        if not isinstance(state, dict) or state.get('version') != 1:
            raise RequestError('Некорректное состояние диалога.')
        if len(json.dumps(state, ensure_ascii=False)) > MAX_DIALOGUE_STATE_CHARS:
            raise RequestError('Состояние диалога слишком большое.', 413)
    return question.strip(), checked, event_date


def public_document(document):
    return document.to_dict(include_text=False)


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, assistant, max_workers=2, access_store=None):
        super().__init__(address, WorkbenchHandler)
        self.assistant = assistant
        self.access = access_store if access_store is not None else AccessStore(os.getenv('PSB_STATE_DIR') or ROOT / 'state')
        self.scopes = ScopedManager(assistant, self.access)
        self.user_locks = {}
        self.user_locks_guard = threading.Lock()
        self.chat_slots = threading.BoundedSemaphore(max(1, min(8, int(max_workers))))
        self.allowed_hosts = {'localhost', '127.0.0.1', '::1', str(address[0]).lower()}

    def user_lock(self, username):
        with self.user_locks_guard:
            return self.user_locks.setdefault(username, threading.RLock())


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = 'PSB-Assistant'
    sys_version = ''
    protocol_version = 'HTTP/1.0'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_args):
        # Do not log request paths, query strings, questions, or model responses.
        return

    def _headers(self, code, content_type, length):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        if getattr(self, 'response_cookie', None):
            self.send_header('Set-Cookie', self.response_cookie)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()

    def _send(self, code, value, head=False):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self._headers(code, 'application/json; charset=utf-8', len(body))
        if not head:
            self.wfile.write(body)

    def _path(self):
        try:
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc:
                raise RequestError('Ожидается локальный путь запроса.')
            return unquote(parsed.path, errors='strict')
        except (UnicodeDecodeError, ValueError):
            raise RequestError('Некорректный путь запроса.') from None

    def _check_host(self):
        host_header = self.headers.get('Host', '')
        try:
            hostname = urlsplit('http://' + host_header).hostname
        except ValueError:
            hostname = None
        if hostname not in self.server.allowed_hosts:
            raise RequestError('Недопустимый адрес сервера.', 403)

    def _token(self):
        try:
            cookie = SimpleCookie(self.headers.get('Cookie', ''))
            return cookie['psb_session'].value if 'psb_session' in cookie else ''
        except Exception:
            return ''

    def _principal(self, optional=False):
        try:
            return self.server.access.authenticate(self._token())
        except AccessError as error:
            if optional and error.status == 401:
                return None
            raise

    def _session_payload(self, principal):
        if principal is None:
            return {'authenticated': False}
        revision = self.server.access.revision()
        return {'authenticated': True,
                'user': {'username': principal['username'], 'level': principal['level']},
                'csrf_token': principal['csrf_token'], 'acl_revision': revision,
                'history_scope': self.server.access.history_scope(principal['username'], revision)}

    def _cookie(self, token, seconds):
        secure = '; Secure' if os.getenv('PSB_COOKIE_SECURE', '0') == '1' else ''
        self.response_cookie = f'psb_session={token}; Path=/; Max-Age={seconds}; HttpOnly; SameSite=Strict{secure}'

    def _same_origin(self):
        origin = self.headers.get('Origin')
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {'http', 'https'} or parsed.netloc.lower() != self.headers.get('Host', '').lower():
                raise RequestError('Запрос разрешён только из этого приложения.', 403)
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise RequestError('Межсайтовый запрос запрещён.', 403)

    def _csrf(self, principal):
        supplied = self.headers.get('X-CSRF-Token', '')
        if not supplied or not hmac.compare_digest(supplied, principal['csrf_token']):
            raise RequestError('Обновите страницу и повторите действие.', 403)

    def _scoped(self, principal):
        revision, labels = self.server.access.snapshot(principal['level'])
        return self.server.scopes.get(principal['level'], revision, labels), revision, labels

    def _send_scoped(self, value, revision, head=False):
        # An ACL edit cannot complete between the final check and delivery.
        with self.server.access.lock:
            self._principal()
            self.server.access.check_revision(revision)
            return self._send(200, value, head)

    def _get(self, head=False):
        self._check_host()
        path = self._path()
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            body = (ROOT / 'static' / filename).read_bytes()
            self._headers(200, content_type, len(body))
            if not head:
                self.wfile.write(body)
            return
        if path == '/api/session':
            return self._send(200, self._session_payload(self._principal(optional=True)), head)
        principal = self._principal(optional=path == '/api/health')
        if principal is None:
            return self._send(200, {'status': 'ok', 'version': 'v21', 'authentication_required': True}, head)
        if path == '/api/history':
            revision = self.server.access.revision()
            chats = self.server.access.list_conversations(principal['username'], revision)
            return self._send_scoped({'history': {'version': 1, 'activeId': chats[0]['id'] if chats else None, 'chats': chats},
                                     'acl_revision': revision}, revision, head)
        assistant, revision, labels = self._scoped(principal)
        corpus = assistant.corpus
        if path == '/api/health':
            client = getattr(assistant, 'client', None)
            enabled = bool(client and client.enabled)
            return self._send_scoped({
                'documents': len(corpus.documents), 'chunks': len(corpus.chunks),
                'mode': 'llm' if enabled else 'extractive',
                'model': client.model if enabled else None,
                'version': 'v21',
                'core_revision': 'v20-quality-defaults+access',
                'acl_revision': revision,
                'retrieval': {'lexical': 'bm25f',
                              'dense_enabled': bool(getattr(client, 'embeddings_enabled', False)),
                              'reranker_enabled': bool(getattr(client, 'reranker_enabled', False)),
                              'reranker_model': getattr(client, 'reranker_model', '') or None},
                'features': {
                    'self_check_enabled': os.getenv('ANSWER_VERIFY', '1').strip().lower() not in {'0', 'false', 'off'},
                    'self_check_available': enabled,
                    'clarifications_before_search': True,
                    'action_steps': True,
                    'history': 'server-account',
                    'access_levels': [1, 2],
                    'query_normalization_enabled': os.getenv('QUERY_NORMALIZE', '0').strip().lower() not in {'0', 'false', 'off'},
                    'query_normalization_available': enabled,
                    'dialogue_state': True,
                },
            }, revision, head)
        if path == '/api/documents':
            docs = sorted(corpus.documents.values(), key=lambda d: (d.kind == 'law', d.title))
            return self._send_scoped({'documents': [{**public_document(d), 'access_level': labels.get(d.id)} for d in docs],
                                     'acl_revision': revision}, revision, head)
        if path == '/api/registry':
            registry=assistant.registry
            return self._send_scoped({
                'profiles':[asdict(p) for p in registry.profiles.values()],
                'amendments':[asdict(a) for a in registry.amendments],
                'issues':registry.issues,
            }, revision, head)
        match = re.fullmatch(r'/api/documents/(doc_[a-f0-9]{16})', path)
        if match:
            document = corpus.get_document(match[1])
            if document:
                payload = document.to_dict()
                payload['access_level'] = labels.get(document.id)
                payload['acl_revision'] = revision
                payload['chunks'] = [c.to_dict() for c in corpus.chunks.values() if c.doc_id == document.id]
                return self._send_scoped(payload, revision, head)
        self._send(404, {'error': 'Страница или документ не найдены.'}, head)

    def do_GET(self):
        self._handle_get()

    def do_HEAD(self):
        self._handle_get(head=True)

    def _handle_get(self, head=False):
        try:
            self._get(head)
        except (RequestError, AccessError) as error:
            self._send(error.status, {'error': str(error)}, head)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            print('HTTP GET failed: ' + type(error).__name__, file=sys.stderr)
            self._send(500, {'error': 'Не удалось обработать запрос.'}, head)

    def _read_json(self):
        if self.headers.get('Transfer-Encoding'):
            raise RequestError('Передайте JSON с Content-Length.')
        if self.headers.get_content_type() != 'application/json':
            raise RequestError('Используйте Content-Type: application/json.', 415)
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise RequestError('Нужен корректный Content-Length.', 411)
        if len(lengths[0]) > 9:
            raise RequestError('Слишком большой запрос.', 413)
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise RequestError('Слишком большой запрос.', 413)
        if length == 0:
            raise RequestError('Тело запроса не может быть пустым.')
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise RequestError('Запрос передан не полностью.')
            return json.loads(raw.decode('utf-8'), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise RequestError('Ожидается корректный JSON в UTF-8.') from None

    def do_POST(self):
        acquired = False
        try:
            self._check_host()
            self._same_origin()
            path = self._path()
            if path == '/api/login':
                payload = self._read_json()
                if not isinstance(payload, dict) or set(payload) != {'username', 'password'}:
                    raise RequestError('Укажите логин и пароль.')
                token, csrf, principal = self.server.access.login(payload['username'], payload['password'])
                self.server.access.logout(self._token())
                principal['csrf_token'] = csrf
                self._cookie(token, SESSION_SECONDS)
                return self._send(200, self._session_payload(principal))
            principal = self._principal()
            self._csrf(principal)
            payload = self._read_json()
            if not isinstance(payload, dict):
                raise RequestError('Ожидается JSON-объект.')
            if path == '/api/logout':
                self.server.access.logout(self._token())
                self._cookie('', 0)
                return self._send(200, {'authenticated': False})
            match = re.fullmatch(r'/api/documents/(doc_[a-f0-9]{16})/access', path)
            if match:
                if principal['level'] != 2:
                    raise RequestError('Изменять уровни доступа может только пользователь уровня 2.', 403)
                if match[1] not in self.server.assistant.corpus.documents:
                    raise RequestError('Документ не найден.', 404)
                if set(payload) != {'access_level'}:
                    raise RequestError('Укажите уровень доступа 1 или 2.')
                revision = self.server.access.set_level(match[1], payload['access_level'])
                return self._send(200, {'access_level': payload['access_level'], 'acl_revision': revision})
            if path in {'/api/history/clear', '/api/history/delete'}:
                with self.server.user_lock(principal['username']):
                    if path.endswith('/clear'):
                        if payload:
                            raise RequestError('Неизвестные поля запроса.')
                        self.server.access.clear_conversations(principal['username'])
                    else:
                        if set(payload) != {'conversation_id'}:
                            raise RequestError('Укажите диалог.')
                        self.server.access.delete_conversation(principal['username'], self._identifier(payload['conversation_id']))
                return self._send(200, {'ok': True})
            if path != '/api/chat':
                return self._send(404, {'error': 'Маршрут не найден.'})
            if set(payload) - {'question', 'event_date', 'reference_date', 'conversation_id', 'turn_id', 'acl_revision'}:
                raise RequestError('История и уровень доступа определяются сервером; неизвестные поля запроса.')
            revision = payload.get('acl_revision')
            self.server.access.check_revision(revision)
            question, _, event_date = validate_chat({key: payload[key] for key in ('question', 'event_date', 'reference_date') if key in payload})
            conversation_id = self._identifier(payload.get('conversation_id', uuid.uuid4().hex))
            turn_id = self._identifier(payload.get('turn_id', uuid.uuid4().hex))
            acquired = self.server.chat_slots.acquire(blocking=False)
            if not acquired:
                raise RequestError('Сервер обрабатывает другие вопросы. Повторите немного позже.', 429)
            with self.server.user_lock(principal['username']):
                self.server.access.check_revision(revision)
                saved = self.server.access.get_conversation(principal['username'], conversation_id, revision)
                now = int(time.time() * 1000)
                display = saved.get('display') or {'id': conversation_id, 'createdAt': now, 'updatedAt': now, 'turns': []}
                for turn in display['turns']:
                    if turn['id'] == turn_id:
                        if turn['question'] != question or turn.get('date') != event_date:
                            raise RequestError('Повтор запроса не соответствует сохранённому вопросу.', 409)
                        return self._send_scoped(turn['answerData'], revision)
                assistant, actual_revision, _ = self._scoped(principal)
                if actual_revision != revision:
                    raise AccessError('Права доступа изменились. Начните новый диалог.', 409)
                options = {'dialogue_state': saved.get('dialogue_state')}
                if payload.get('reference_date') is not None:
                    options['reference_date'] = payload['reference_date']
                result = assistant.answer(question, history=saved['history'], event_date=event_date,
                                          deadline=time.monotonic() + 120, **options)
                if not isinstance(result, dict) or not isinstance(result.get('answer'), str):
                    raise RuntimeError('Unexpected answer contract')
                self.server.access.check_revision(revision)
                self.server.access.authenticate(self._token())
                if any(c.get('doc_id') not in assistant.corpus.documents for c in result.get('citations', [])):
                    raise RuntimeError('Citation outside authenticated scope')
                result = {**result, 'conversation_id': conversation_id, 'turn_id': turn_id, 'acl_revision': revision}
                history = saved['history'] + [{'role': 'user', 'content': question + (f'\nДата события: {event_date}' if event_date else '')}]
                if result.get('status') != 'technical_error' and not result['answer'].startswith('DIAG_'):
                    history.append({'role': 'assistant', 'content': result['answer'][:MAX_HISTORY_MESSAGE_CHARS]})
                history = history[-MAX_HISTORY_MESSAGES:]
                while history and (history[0]['role'] == 'assistant' or sum(len(m['content']) for m in history) > MAX_HISTORY_CHARS):
                    history.pop(0)
                display['updatedAt'] = now
                display['turns'].append({'id': turn_id, 'question': question, 'date': event_date, 'at': now,
                                         'status': 'complete', 'answerData': result})
                # Bound displayed history without truncating source quotes inside a turn.
                display['turns'] = display['turns'][-100:]
                with self.server.access.lock:
                    self._principal()
                    self.server.access.check_revision(revision)
                    self.server.access.save_conversation(principal['username'], conversation_id, revision,
                                                         history, result.get('dialogue_state'), display)
                    self._send(200, result)
        except (RequestError, AccessError) as error:
            self._send(error.status, {'error': str(error)})
        except (socket.timeout, TimeoutError):
            self._send(408, {'error': 'Истекло время обработки запроса. Попробуйте ещё раз.'})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            print('HTTP chat failed: ' + type(error).__name__, file=sys.stderr)
            self._send(500, {'error': 'Не удалось сформировать ответ из-за технического сбоя. Повторите вопрос.'})
        finally:
            if acquired:
                self.server.chat_slots.release()

    @staticmethod
    def _identifier(value):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
            raise RequestError('Некорректный идентификатор диалога или вопроса.')
        return value

    def do_OPTIONS(self):
        self._send(405, {'error': 'Метод не поддерживается.'})


def create_server(host, port, assistant, max_workers=2, access_store=None):
    return WorkbenchServer((host, port), assistant, max_workers=max_workers, access_store=access_store)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Юридический ассистент ПСБ: локальный веб-интерфейс')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('--port должен быть от 0 до 65535')
    load_env(ROOT / '.env')
    # Diagnostic nonsense is useful only for the explicitly diagnostic batch
    # experiment. The workbench defaults to honest source excerpts offline.
    os.environ.setdefault('DIAGNOSTIC_API_FAILURES', '0')
    data_dir = Path(args.data_dir or os.getenv('DATA_DIR') or ROOT / 'data')
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    from answer import LegalAssistant
    assistant = LegalAssistant(data_dir=data_dir)
    try:
        workers = int(os.getenv('MAX_WORKERS', '2'))
    except ValueError:
        workers = 2
    access_store = AccessStore(os.getenv('PSB_STATE_DIR') or ROOT / 'state')
    server = create_server(args.host, args.port, assistant, max_workers=workers, access_store=access_store)
    host, port = server.server_address[:2]
    print(f'Юридический ассистент: http://{host}:{port}', flush=True)
    print(f'Документов: {len(assistant.corpus.documents)}. Для остановки нажмите Ctrl+C.', flush=True)
    if access_store.credentials_path:
        print(f'Учётные записи: {access_store.credentials_path}. Не публикуйте этот файл.', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        access_store.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
