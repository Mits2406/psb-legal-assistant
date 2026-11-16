"""Per-request authorization scope, applied before every RAG/model stage.

No global corpus or shared index is mutated. The archived V20 CSV core remains
unchanged; the authenticated web API selects an isolated view of that core.
"""
from __future__ import annotations

import copy
from hashlib import sha256
from pathlib import Path
import threading
import time

from access_control import AccessError, _level
from corpus import Corpus
from registry import RuleRegistry
from clarification import PreRetrievalClarifier
from evidence import EvidencePlanner
from llm import APIError, ModelClient, number


def scoped_corpus(original, allowed_ids):
    """Build an independent lexical index and metadata graph from allowed docs."""
    corpus = object.__new__(Corpus)
    corpus.path = getattr(original, 'path', Path('data'))
    corpus.documents = {key: value for key, value in original.documents.items() if key in allowed_ids}
    corpus.chunks = {key: value for key, value in original.chunks.items() if value.doc_id in corpus.documents}
    corpus._build_index()
    company_count = sum(d.kind == 'company' for d in corpus.documents.values())
    suspicious_count = sum(c.suspicious for c in corpus.chunks.values())
    manifest = [did + ':' + d.sha256 for did, d in sorted(corpus.documents.items())]
    corpus.stats = {
        'documents': len(corpus.documents), 'chunks': len(corpus.chunks),
        'document_count': len(corpus.documents), 'chunk_count': len(corpus.chunks),
        'company_count': company_count, 'law_count': len(corpus.documents) - company_count,
        'bytes': sum(len(d.text.encode('utf-8')) for d in corpus.documents.values()),
        'characters': sum(len(d.text) for d in corpus.documents.values()),
        'words': sum(len(d.text.split()) for d in corpus.documents.values()),
        'suspicious_chunks': suspicious_count, 'suspicious_chunk_count': suspicious_count,
        'indexed_chunks': len(corpus._lengths),
        'metadata_chunks': sum(c.content_role == 'metadata' for c in corpus.chunks.values()),
        'snapshot_dates': sorted({d.snapshot_date for d in corpus.documents.values() if d.snapshot_date}),
        'corpus_hash': sha256('\n'.join(manifest).encode()).hexdigest(),
    }
    return corpus


class GuardedClient:
    """Reject further model calls after an ACL edit; in-flight calls cannot undo."""
    API_METHODS = frozenset({'post', 'chat_text', 'chat_json', 'embed', 'rerank'})

    def __init__(self, base, store, revision):
        self._base = base
        self._store = store
        self._revision = revision

    def _check(self):
        try:
            self._store.check_revision(self._revision)
        except AccessError:
            raise APIError('Права доступа изменились.', code='access_changed') from None

    def _post(self, path, payload, deadline=None, key=None, *, retry_limit=None):
        """Keep core transport parsing, but authorize each individual attempt."""
        retries = int(number('API_RETRIES', 1, 0, 2)) if retry_limit is None else max(0, min(2, int(retry_limit)))
        for attempt in range(retries + 1):
            self._check()
            try:
                return ModelClient.post(self, path, payload, deadline, key, retry_limit=0)
            except APIError as exc:
                if exc.code not in {'http_429', 'http_500', 'http_502', 'http_503', 'http_504',
                                    'transport_error', 'timeout'} or attempt == retries:
                    raise
            self._check()
            pause = .4 * (2 ** attempt)
            if deadline and time.monotonic() + pause >= deadline:
                raise APIError('Исчерпан бюджет времени обращения к модели.', code='deadline')
            time.sleep(pause)

    def __getattr__(self, name):
        value = getattr(self._base, name)
        if name not in self.API_METHODS or not callable(value):
            return value

        def guarded(*args, **kwargs):
            self._check()
            # Bind real ModelClient methods to this proxy: internal chat_json →
            # chat_text → post calls also pass the revision guard. Fixture and
            # adapter clients retain their original behaviour behind the guard.
            if isinstance(self._base, ModelClient):
                if name == 'post':
                    return self._post(*args, **kwargs)
                function = getattr(type(self._base), name)
                return function(self, *args, **kwargs)
            return value(*args, **kwargs)
        return guarded


class ScopedManager:
    def __init__(self, assistant, store):
        self.base = assistant
        self.store = store
        self.lock = threading.RLock()
        self._cache = {}
        self._generation = None

    def get(self, level, revision, labels):
        _level(level)
        actual_revision, actual_labels = self.store.snapshot(level)
        if type(revision) is not int or actual_revision != revision:
            raise AccessError('Права доступа изменились. Повторите запрос.', 409)
        if labels != actual_labels:
            raise AccessError('Неверная область доступа.', 403)
        with self.lock:
            generation = (revision, id(self.base.client))
            if self._generation != generation:
                self._cache.clear()
                self._generation = generation
            if level in self._cache:
                return self._cache[level]
            allowed = {did for did in self.base.corpus.documents if actual_labels.get(did, 1) <= level}
            assistant = copy.copy(self.base)
            assistant.corpus = scoped_corpus(self.base.corpus, allowed)
            assistant.client = GuardedClient(self.base.client, self.store, revision)
            assistant.registry = RuleRegistry(assistant.corpus)
            assistant.clarifier = PreRetrievalClarifier(assistant.registry)
            assistant.planner = EvidencePlanner(assistant.corpus, assistant.registry, assistant.client)
            retriever = assistant.planner.structure
            if retriever.cache_dir is not None:
                # A lower-level scope never reads another scope's cache/probes.
                identity = f'{level}:{revision}:{assistant.corpus.stats["corpus_hash"]}'
                retriever.cache_dir /= 'access-' + sha256(identity.encode()).hexdigest()
            self._cache[level] = assistant
            return assistant
