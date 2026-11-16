"""Local web workbench. Serves only the three public assets and a bounded JSON API.

Run: python app.py --host 127.0.0.1 --port 8000 --data-dir data
The browser persists conversations locally. This server keeps no user sessions
and writes neither questions nor API credentials to disk.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

from llm import load_env

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

    def __init__(self, address, assistant, max_workers=2):
        super().__init__(address, WorkbenchHandler)
        self.assistant = assistant
        self.chat_slots = threading.BoundedSemaphore(max(1, min(8, int(max_workers))))
        self.allowed_hosts = {'localhost', '127.0.0.1', '::1', str(address[0]).lower()}


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
        assistant = self.server.assistant
        corpus = assistant.corpus
        if path == '/api/health':
            client = getattr(assistant, 'client', None)
            enabled = bool(client and client.enabled)
            return self._send(200, {
                'documents': len(corpus.documents), 'chunks': len(corpus.chunks),
                'mode': 'llm' if enabled else 'extractive',
                'model': client.model if enabled else None,
                'version': 'v20',
                'core_revision': 'v20-quality-defaults',
                'retrieval': {'lexical': 'bm25f',
                              'dense_enabled': bool(getattr(client, 'embeddings_enabled', False)),
                              'reranker_enabled': bool(getattr(client, 'reranker_enabled', False)),
                              'reranker_model': getattr(client, 'reranker_model', '') or None},
                'features': {
                    'self_check_enabled': os.getenv('ANSWER_VERIFY', '1').strip().lower() not in {'0', 'false', 'off'},
                    'self_check_available': enabled,
                    'clarifications_before_search': True,
                    'action_steps': True,
                    'history': 'browser',
                    'query_normalization_enabled': os.getenv('QUERY_NORMALIZE', '0').strip().lower() not in {'0', 'false', 'off'},
                    'query_normalization_available': enabled,
                    'dialogue_state': True,
                },
            }, head)
        if path == '/api/documents':
            docs = sorted(corpus.documents.values(), key=lambda d: (d.kind == 'law', d.title))
            return self._send(200, {'documents': [public_document(d) for d in docs]}, head)
        if path == '/api/registry':
            registry=assistant.registry
            return self._send(200, {
                'profiles':[asdict(p) for p in registry.profiles.values()],
                'amendments':[asdict(a) for a in registry.amendments],
                'issues':registry.issues,
            }, head)
        match = re.fullmatch(r'/api/documents/(doc_[a-f0-9]{16})', path)
        if match:
            document = corpus.get_document(match[1])
            if document:
                payload = document.to_dict()
                payload['chunks'] = [c.to_dict() for c in corpus.chunks.values() if c.doc_id == document.id]
                return self._send(200, payload, head)
        self._send(404, {'error': 'Страница или документ не найдены.'}, head)

    def do_GET(self):
        self._handle_get()

    def do_HEAD(self):
        self._handle_get(head=True)

    def _handle_get(self, head=False):
        try:
            self._get(head)
        except RequestError as error:
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
            if self._path() != '/api/chat':
                return self._send(404, {'error': 'Маршрут не найден.'})
            origin = self.headers.get('Origin')
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme not in {'http', 'https'} or parsed.netloc.lower() != self.headers.get('Host', '').lower():
                    raise RequestError('Запрос разрешён только из этого приложения.', 403)
            payload = self._read_json()
            question, history, event_date = validate_chat(payload)
            acquired = self.server.chat_slots.acquire(blocking=False)
            if not acquired:
                raise RequestError('Сервер обрабатывает другие вопросы. Повторите немного позже.', 429)
            options = {key: payload[key] for key in ('dialogue_state', 'reference_date')
                       if payload.get(key) is not None}
            result = self.server.assistant.answer(question, history=history, event_date=event_date,
                                                  deadline=time.monotonic() + 120, **options)
            if not isinstance(result, dict) or not isinstance(result.get('answer'), str):
                raise RuntimeError('Unexpected answer contract')
            self._send(200, result)
        except RequestError as error:
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

    def do_OPTIONS(self):
        self._send(405, {'error': 'Метод не поддерживается.'})


def create_server(host, port, assistant, max_workers=2):
    return WorkbenchServer((host, port), assistant, max_workers=max_workers)


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
    server = create_server(args.host, args.port, assistant, max_workers=workers)
    host, port = server.server_address[:2]
    print(f'Юридический ассистент: http://{host}:{port}', flush=True)
    print(f'Документов: {len(assistant.corpus.documents)}. Для остановки нажмите Ctrl+C.', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
