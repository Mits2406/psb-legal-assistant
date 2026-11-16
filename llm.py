"""OpenAI-compatible organiser proxy. Standard library only; never logs secrets."""
from __future__ import annotations

import json
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def load_env(path: Path | str = '.env') -> None:
    """Load simple KEY=value without evaluating shell expressions or overriding env."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.removeprefix('export ').strip()
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            value = value.strip()
            if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def number(name: str, default: float, low: float, high: float) -> float:
    try:
        n = float(os.getenv(name, str(default)))
        return max(low, min(high, n)) if math.isfinite(n) else default
    except ValueError:
        return default


def env_first(*names: str) -> str:
    """Match organiser aliases without treating empty/template values as config."""
    for name in names:
        value = os.getenv(name, '').strip()
        if value and value not in {name, '${' + name + '}', '{' + name + '}'}:
            return value
    return ''


def diagnostic_enabled() -> bool:
    """Diagnostic markers are opt-in; normal submissions use grounded recovery."""
    return os.getenv('DIAGNOSTIC_API_FAILURES', '0').strip().lower() not in {'0', 'false', 'off'}


def diagnostic_answer(category: str, reason: str, stage: str) -> str:
    """Visible, deliberately irrelevant output; never include external errors."""
    if category not in {'API_FAILURE', 'API_NOT_CALLED', 'MODEL_OUTPUT', 'RUN_FAILURE'}:
        category = 'RUN_FAILURE'
    reason = reason if isinstance(reason, str) and re.fullmatch(r'[a-z0-9_]{1,64}', reason) else 'unknown'
    stage = stage if stage in {'configuration', 'draft', 'review', 'parser', 'batch', 'initialization'} else 'unknown'
    return f'DIAG_{category}:{reason}; stage={stage}. Ответ: фиолетовый трактор.'


class APIError(RuntimeError):
    def __init__(self, message, *, code='api_error'):
        super().__init__(message)
        self.code = code


class ModelOutputError(APIError):
    """A completed HTTP response cannot be accepted as a model JSON answer."""


def parse_json_object(content: str) -> dict:
    """Accept one complete object, optionally fenced or surrounded by prose.

    Decode only the first outer object. Never balance missing braces, strip a
    trailing comma, or recover an inner object from truncated outer JSON.
    """
    if not isinstance(content, str) or not content.strip():
        raise ModelOutputError('Модель вернула пустой ответ.', code='empty_content')
    text = content.strip().lstrip('\ufeff')
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text, flags=re.I)
    if fence:
        text = fence.group(1).strip()
    def invalid_constant(_):
        raise ValueError('Nonfinite JSON value')
    decoder = json.JSONDecoder(parse_constant=invalid_constant)
    try:
        start = text.find('{')
        if start < 0 or any(char in text[:start] for char in '{}[]'):
            raise ValueError()
        result, end = decoder.raw_decode(text, start)
        trailing = text[end:].strip()
        # Multiple objects, arrays and unmatched closing braces are ambiguous.
        if not isinstance(result, dict) or any(char in trailing for char in '{}[]'):
            raise ValueError()
        return result
    except (ValueError, TypeError) as exc:
        raise ModelOutputError('Модель не вернула один завершённый JSON-объект.', code='invalid_json') from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise APIError('Прокси вернул перенаправление; проверьте LLM_BASE_URL.', code='redirect')


class ModelClient:
    def __init__(self):
        self.base_url = env_first('LLM_BASE_URL', 'OPENAI_BASE_URL', 'BASE_URL').rstrip('/')
        self.api_key = env_first('LLM_API_KEY', 'OPENAI_API_KEY')
        self.model = env_first('LLM_MODEL', 'OPENAI_MODEL', 'GEN_MODEL')
        self.embedding_model = env_first('EMB_MODEL') or 'baai/bge-m3'
        self.reranker_model = env_first('RERANKER_MODEL')
        self.reranker_api_key = env_first('RERANKER_API_KEY') or self.api_key
        self.enabled = bool(self.base_url and self.api_key and self.model)
        self.embeddings_enabled = self.enabled and os.getenv('ENABLE_EMBEDDINGS', '1').strip().lower() not in {'0', 'false', 'off'}
        self.reranker_enabled = (bool(self.base_url and self.reranker_api_key and self.reranker_model)
                                and os.getenv('ENABLE_RERANKER', '1').strip().lower() not in {'0', 'false', 'off'})
        if self.base_url and urllib.parse.urlsplit(self.base_url).scheme not in {'http', 'https'}:
            raise APIError('LLM_BASE_URL должен быть HTTP(S)-адресом прокси.', code='invalid_configuration')
        self.timeout = number('API_TIMEOUT_SECONDS', 90, 1, 110)
        self._local = threading.local()

    @property
    def last_chat_diagnostics(self):
        """Per-thread metadata only: never prompts, content, URLs or keys."""
        return dict(getattr(self._local, 'chat', {}))

    @property
    def last_rerank_diagnostics(self):
        """Per-thread rerank metadata; no source text, query, URL or credential."""
        return dict(getattr(self._local, 'rerank', {}))

    def post(self, path: str, payload: dict, deadline: float | None = None, key: str | None = None,
             *, retry_limit: int | None = None) -> dict:
        # A separate reranker credential/model can work without a configured
        # generative model. Other endpoints preserve their existing contract.
        if not (self.reranker_enabled if path == '/rerank' else self.enabled):
            raise APIError('Не настроены адрес, ключ и модель API.', code='not_configured')
        retries = int(number('API_RETRIES', 1, 0, 2)) if retry_limit is None else max(0, min(2, int(retry_limit)))
        diagnostics = (getattr(self._local, 'chat', {}) if path == '/chat/completions'
                       else getattr(self._local, 'rerank', {}) if path == '/rerank' else {})
        request_key = key or (self.reranker_api_key if path == '/rerank' else self.api_key)
        for attempt in range(retries + 1):
            remaining = (deadline - time.monotonic()) if deadline else self.timeout
            if remaining <= .3:
                raise APIError('Исчерпан бюджет времени обращения к модели.', code='deadline')
            req = urllib.request.Request(self.base_url + path, data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={'Authorization': 'Bearer ' + request_key, 'Content-Type': 'application/json'}, method='POST')
            try:
                opener = urllib.request.build_opener(NoRedirect())
                diagnostics['api_attempted'] = True
                diagnostics['api_attempts'] = diagnostics.get('api_attempts', 0) + 1
                with opener.open(req, timeout=min(self.timeout, remaining)) as response:
                    diagnostics['api_http_success'] = True
                    status = getattr(response, 'status', 200)
                    if type(status) is int:
                        diagnostics['http_status'] = status
                    data = response.read(12_000_001)
                    if len(data) > 12_000_000:
                        raise ModelOutputError('Слишком большой ответ прокси.', code='response_too_large')
                    try:
                        result = json.loads(data)
                    except (ValueError, UnicodeError):
                        raise ModelOutputError('Прокси вернул некорректный JSON.', code='invalid_response_json') from None
                    if not isinstance(result, dict):
                        raise ModelOutputError('Прокси вернул некорректную структуру.', code='response_shape')
                    return result
            except urllib.error.HTTPError as exc:
                diagnostics['http_status'] = exc.code
                exc.close()
                if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                    raise APIError(f'Ошибка прокси HTTP {exc.code}.', code='http_' + str(exc.code)) from None
            except (urllib.error.URLError, OSError, ValueError) as exc:
                if attempt == retries:
                    timed_out = isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(getattr(exc, 'reason', None), (TimeoutError, socket.timeout))
                    raise APIError('Прокси недоступен или истекло время ожидания.', code='timeout' if timed_out else 'transport_error') from None
            pause = .4 * (2 ** attempt)
            if deadline and time.monotonic() + pause >= deadline:
                raise APIError('Исчерпан бюджет времени обращения к модели.', code='deadline')
            time.sleep(pause)
        raise APIError('Не удалось обратиться к прокси.')

    def chat_text(self, system: str, data: dict, deadline: float | None = None,
                  *, max_tokens: int | None = None) -> str:
        """One normal completion; structural and semantic checks are separate.

        Return nonempty text even at the token limit. The caller can preserve
        complete cited paragraphs instead of discarding the entire answer.
        """
        started = time.monotonic()
        # Allow headroom for a complete answer/review. This is a ceiling, not
        # a requested answer length; an explicit platform value takes priority.
        cap = int(number('MAX_TOKENS', 8192, 1, 8192))
        budget = cap if max_tokens is None else min(cap, max(1, int(max_tokens)))
        self._local.chat = {'requested_tokens': budget, 'finish_reason': 'unavailable',
                            'api_attempted': False, 'api_attempts': 0,
                            'api_http_success': False, 'model_text_received': False,
                            'model_text_usable': False}
        try:
            result = self.post('/chat/completions', {
                'model': self.model,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(data, ensure_ascii=False, separators=(',', ':'))}],
                'temperature': number('TEMPERATURE', 0, 0, 1),
                'max_tokens': budget,
            }, deadline, retry_limit=int(number('API_RETRIES', 1, 0, 1)))
            choice = result['choices'][0]
            reason = choice.get('finish_reason')
            self._local.chat['finish_reason'] = reason if reason in {'stop', 'length', 'content_filter', 'tool_calls', 'function_call', None} else 'other'
            usage = result.get('usage', {})
            if isinstance(usage, dict):
                self._local.chat['usage'] = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                           if type(usage.get(key)) is int and usage[key] >= 0}
            message = choice['message']
            content = message.get('content')
            self._local.chat['model_text_received'] = isinstance(content, str) and bool(content.strip())
            if reason in {'content_filter', 'tool_calls', 'function_call'}:
                raise ModelOutputError('Модель не завершила текстовый ответ.', code='nontext_output')
            if message.get('refusal'):
                raise ModelOutputError('Модель отказалась формировать ответ.', code='model_refusal')
            self._local.chat['reasoning_present'] = bool(message.get('reasoning_content'))
            if not isinstance(content, str) or not content.strip():
                raise ModelOutputError('Модель вернула пустой ответ.', code='empty_content')
            self._local.chat['model_text_usable'] = True
            return content.strip()
        except APIError as exc:
            self._local.chat['error_code'] = exc.code
            raise
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            self._local.chat['error_code'] = 'response_shape'
            raise ModelOutputError('Прокси вернул неожиданный формат ответа модели.', code='response_shape') from None
        finally:
            self._local.chat['elapsed_ms'] = round((time.monotonic() - started) * 1000)

    def chat_json(self, system: str, data: dict, deadline: float | None = None,
                  *, max_tokens: int | None = None) -> dict:
        content = self.chat_text(system, data, deadline, max_tokens=max_tokens)
        try:
            return parse_json_object(content)
        except APIError as exc:
            self._local.chat['error_code'] = exc.code
            raise

    def embed(self, texts: list[str], deadline=None) -> list[list[float]]:
        result = self.post('/embeddings', {'model': self.embedding_model, 'input': texts, 'encoding_format': 'float'},
                           deadline, env_first('EMBEDDER_API_KEY', 'EMB_API_KEY', 'EMB_KEY') or None)
        try:
            rows = result['data']
            vectors = [None] * len(texts)
            for pos, row in enumerate(rows):
                idx = int(row.get('index', pos))
                if not 0 <= idx < len(vectors) or vectors[idx] is not None:
                    raise ValueError()
                v = [float(x) for x in row['embedding']]
                if not v or not all(math.isfinite(x) for x in v):
                    raise ValueError()
                norm = math.sqrt(sum(x*x for x in v))
                if not norm:
                    raise ValueError()
                vectors[idx] = [x/norm for x in v]
            if any(v is None for v in vectors) or len({len(v) for v in vectors}) != 1:
                raise ValueError()
            return vectors
        except (KeyError, ValueError, TypeError, IndexError):
            raise APIError('Некорректные эмбеддинги прокси.') from None

    def rerank(self, query, chunks, deadline=None):
        """Ask the configured reranker once and return canonical input objects.

        A short valid top-N ranking is intentional. The planner is responsible
        for retaining omitted evidence and mandatory related conditions.
        """
        started = time.monotonic()
        self._local.rerank = {'model': self.reranker_model, 'status': 'pending',
                              'input_count': len(chunks) if isinstance(chunks, (list, tuple)) else 0,
                              'top_n': 0, 'returned': 0, 'api_attempted': False,
                              'api_attempts': 0, 'api_http_success': False}
        diagnostics = self._local.rerank
        try:
            if not isinstance(chunks, (list, tuple)):
                raise APIError('Некорректные кандидаты реранкера.', code='invalid_rerank_input')
            if not chunks:
                diagnostics['status'] = 'skipped_empty'
                return []
            if (not isinstance(query, str) or not query.strip()
                    or any(not isinstance(getattr(c, 'title', None), str)
                           or not isinstance(getattr(c, 'text', None), str)
                           or not c.text.strip() for c in chunks)):
                raise APIError('Некорректные кандидаты реранкера.', code='invalid_rerank_input')
            top_n = int(number('RERANKER_TOP_N', 8, 1, min(64, len(chunks))))
            diagnostics['top_n'] = top_n
            if not self.reranker_enabled:
                raise APIError('Реранкер не настроен или выключен.', code='not_configured')
            if deadline is not None and deadline - time.monotonic() <= .3:
                raise APIError('Исчерпан бюджет реранжирования.', code='deadline')
            result = self.post('/rerank', {'model': self.reranker_model, 'query': query,
                'documents': [c.title + '\n' + c.text for c in chunks], 'top_n': top_n},
                deadline, self.reranker_api_key, retry_limit=0)
            if not isinstance(result, dict):
                raise ValueError()
            rows = result.get('results', result.get('data'))
            if not isinstance(rows, list) or not 0 < len(rows) <= top_n:
                raise ValueError()
            indices = []
            for row in rows:
                if not isinstance(row, dict) or type(row.get('index')) is not int:
                    raise ValueError()
                index = row['index']
                if not 0 <= index < len(chunks) or index in indices:
                    raise ValueError()
                for field in ('relevance_score', 'score'):
                    if field in row and (type(row[field]) not in (int, float)
                                         or not math.isfinite(row[field])):
                        raise ValueError()
                indices.append(index)
            diagnostics.update(status='completed', returned=len(indices))
            return [chunks[i] for i in indices]
        except APIError as exc:
            diagnostics.update(status='failed', error_code=exc.code)
            raise
        except (KeyError, ValueError, TypeError, OverflowError):
            diagnostics.update(status='failed', error_code='invalid_rerank_response')
            raise APIError('Некорректный ответ реранкера.', code='invalid_rerank_response') from None
        finally:
            diagnostics['elapsed_ms'] = round((time.monotonic() - started) * 1000)
