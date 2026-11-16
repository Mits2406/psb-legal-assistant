"""Scoped hybrid retrieval and loss-aware source-context packing (stdlib only)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time

from chunk import Chunk


@dataclass(frozen=True, slots=True)
class SearchScope:
    doc_ids: frozenset[str] | None = None
    chunk_ids: frozenset[str] | None = None

    def __post_init__(self):
        for name in ('doc_ids', 'chunk_ids'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, frozenset((value,)) if isinstance(value, str) else frozenset(value))


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    trace: dict
    warnings: list[str]
    complete: bool = True
    ranked: list[Chunk] = field(default_factory=list)


class Retriever:
    SCHEMA = 'psb-content-embedding-v2'
    PROBE_TEXTS = ('Срок оплаты товара определяется договором.', 'Как оформить передачу персональных данных?')
    BATCH_SIZE = 32
    SPARSE_CANDIDATES = 48
    DENSE_CANDIDATES = 48
    FUSION_CANDIDATES = 40

    def __init__(self, corpus, client, registry=None, cache_dir=None, portable_index_path=None):
        self.corpus, self.client, self.registry = corpus, client, registry
        directory = cache_dir if cache_dir is not None else os.getenv('RETRIEVAL_CACHE_DIR', '')
        self.cache_dir = Path(directory) if directory else None
        self._builder_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cache_key = None
        self._vectors = {}
        self._dimension = None
        self._context_prefixes = {}
        self.portable_index_path = Path(portable_index_path) if portable_index_path is not None else Path(__file__).with_name('precomputed_index.py')
        self._portable = None
        self._portable_state = {}
        self._space_ready = False
        self._live_probes = []
        self._probe_attempts = 0
        self._probe_retry_after = 0.0
        # Bounded, process-local cache: vectors only, never dialog/answers.
        self._query_vectors = OrderedDict()

    def _allowed(self, scope):
        scope = scope or SearchScope()
        return {c.id: c for c in self.corpus.chunks.values()
                if not c.suspicious and c.searchable and c.content_role != 'metadata'
                and (scope.doc_ids is None or c.doc_id in scope.doc_ids)
                and (scope.chunk_ids is None or c.id in scope.chunk_ids)}

    def _inputs(self, values, allowed):
        accepted, rejected = [], []
        for value in values:
            key = value if isinstance(value, str) else getattr(value, 'id', None)
            if key in allowed:
                if key not in {c.id for c in accepted}:
                    accepted.append(allowed[key])
            else:
                rejected.append(key if key in self.corpus.chunks else None)
        return accepted, rejected

    def _new_trace(self, scope, allowed, strategy):
        scope = scope or SearchScope()
        return {'strategy': strategy,
                'scope': {'doc_ids': sorted(scope.doc_ids) if scope.doc_ids is not None else None,
                          'chunk_ids': sorted(scope.chunk_ids) if scope.chunk_ids is not None else None,
                          'allowed_count': len(allowed)},
                'stage_ids': {}, 'stage_counts': {}, 'timings_ms': {},
                'cache': {'mode': 'disk' if self.cache_dir else 'memory', 'status': 'disabled'}}

    @staticmethod
    def _stage(trace, name, chunks):
        trace['stage_ids'][name] = [c.id for c in chunks]
        trace['stage_counts'][name] = len(chunks)

    def contextual_text(self, chunk):
        """Deterministic context, never an LLM-authored summary or rewritten text."""
        lines = [f'Документ: {chunk.title}', f'Раздел: {chunk.section}', f'Положение: {chunk.clause}']
        profile = getattr(self.registry, 'profiles', {}).get(chunk.doc_id) if self.registry else None
        document = self.corpus.get_document(chunk.doc_id)
        cached = self._context_prefixes.get(chunk.doc_id)
        if cached and cached[0] is document and cached[1] is profile:
            return '\n'.join(lines + cached[2]) + '\n\n' + chunk.text
        extra = []
        if profile and document:
            evidence = getattr(profile, 'evidence', ())
            proven = bool(evidence) and all(
                e.doc_id == document.id and 0 <= e.start < e.end <= len(document.text)
                and document.text[e.start:e.end] == e.quote for e in evidence)
            if proven:
                kind = getattr(profile, 'document_type', '')
                if kind in {'contract', 'amendment', 'regulation', 'letter', 'memo', 'law', 'law_article'}:
                    extra.append('Тип: ' + kind)
                normalize = lambda text: re.sub(r'\s+', '', text.upper().translate(str.maketrans('‐‑–—', '----')))
                source = normalize(document.text)
                numbers = [str(n) for n in getattr(profile, 'contract_numbers', ()) if normalize(str(n)) in source]
                if numbers:
                    extra.append('Номера договора: ' + ', '.join(numbers))
        self._context_prefixes[chunk.doc_id] = (document, profile, extra)
        return '\n'.join(lines + extra) + '\n\n' + chunk.text

    @staticmethod
    def _text_hash(text):
        return sha256(text.encode('utf-8')).hexdigest()

    def _fingerprint(self):
        texts = {c.id: self.contextual_text(c) for c in self._allowed(None).values()}
        payload = {'schema': self.SCHEMA, 'model': str(getattr(self.client, 'embedding_model', '')).lower(),
                   'endpoint': str(getattr(self.client, 'base_url', ''))}
        # Provider identity isolates ONLINE caches. The portable artifact is
        # independent of URL and corpus IDs; its exact-input hashes + live
        # compatibility probes establish whether its vectors can be reused.
        key = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return key, texts

    @staticmethod
    def _vector(raw, dimension=None):
        if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= 65536:
            raise ValueError('Invalid vector length')
        if dimension is not None and len(raw) != dimension:
            raise ValueError('Inconsistent vector dimension')
        if any(type(value) not in (int, float) for value in raw):
            raise ValueError('Invalid vector value')
        vector = [float(value) for value in raw]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError('Nonfinite vector')
        length = math.hypot(*vector)
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Zero vector')
        return tuple(value / length for value in vector)

    def _cache_path(self, key):
        return self.cache_dir / ('embeddings-' + key + '.json') if self.cache_dir else None

    def _load_disk(self, key, warnings):
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            if path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError('Oversized cache')
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('schema') != self.SCHEMA or data.get('key') != key:
                raise ValueError('Cache identity mismatch')
            dimension, values, probes = data.get('dimension'), data.get('vectors'), data.get('probes')
            if type(dimension) is not int or not 1 <= dimension <= 65536 or not isinstance(values, dict):
                raise ValueError('Invalid cache shape')
            if any(not re.fullmatch(r'[0-9a-f]{64}', digest) for digest in values):
                raise ValueError('Invalid input hash')
            if not isinstance(probes, list) or not 2 <= len(probes) <= 8:
                raise ValueError('Invalid cache probes')
            vectors = {digest: self._vector(vector, dimension) for digest, vector in values.items()}
            validated_probes = []
            for probe in probes:
                if not isinstance(probe['text'], str) or not probe['text']:
                    raise ValueError('Invalid probe text')
                validated_probes.append({'text': probe['text'], 'vector': self._vector(probe['vector'], dimension)})
            return {'vectors': vectors, 'dimension': dimension, 'probes': validated_probes}
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            warnings.append('Дисковый кэш эмбеддингов не прошёл проверку; используются доступные источники и переносимый индекс.')
            return None

    def _save_disk(self, key, warnings):
        path = self._cache_path(key)
        if path is None:
            return
        temporary = None
        try:
            with self._state_lock:
                payload = {'schema': self.SCHEMA, 'key': key, 'dimension': self._dimension,
                           'vectors': dict(self._vectors), 'probes': self._live_probes}
            if not payload['vectors'] or not payload['probes']:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='.embedding-', suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except (OSError, ValueError, TypeError):
            message = 'Не удалось сохранить кэш эмбеддингов; текущий индекс остаётся доступным в памяти.'
            if message not in warnings:
                warnings.append(message)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _bounded_setting(name, default, maximum):
        try:
            return max(0, min(maximum, int(os.getenv(name, str(default)))))
        except ValueError:
            return default

    def _embed(self, texts, deadline, cache):
        if time.monotonic() >= deadline - .25:
            raise TimeoutError('Embedding deadline')
        cache['api_calls'] += 1
        return self.client.embed(texts, deadline)

    def _initialize_space(self, key, texts, deadline, cache, warnings):
        """Called under builder lock; publish only verified, single-space vectors."""
        with self._state_lock:
            if self._cache_key != key:
                self._probe_attempts, self._probe_retry_after = 0, 0.0
            self._cache_key, self._vectors, self._dimension = key, {}, None
            self._space_ready, self._portable, self._live_probes = False, None, []
            self._query_vectors.clear()
        state = {'status': 'absent', 'probe_verified': False, 'invalid': False}
        portable = None
        try:
            if self.portable_index_path.is_file():
                from vector_index import load_module
                portable = load_module(self.portable_index_path, expected_model=self.client.embedding_model)
                state['status'] = 'awaiting_probe'
                state['model'] = portable.manifest['model_id']
                state['revision'] = portable.manifest['model_revision']
        except (OSError, ValueError, ImportError):
            state.update(status='invalid_artifact', invalid=True)
            warnings.append('Переносимый индекс не соответствует модели или повреждён; его векторы не используются.')
        disk = self._load_disk(key, warnings)
        probe_texts = (list(portable.probe_texts) if portable else
                       [p['text'] for p in disk['probes']] if disk else list(self.PROBE_TEXTS))
        try:
            self._probe_attempts += 1
            raw = self._embed(probe_texts, deadline, cache)
            if not isinstance(raw, (list, tuple)) or len(raw) != len(probe_texts):
                raise ValueError('Probe coverage mismatch')
            actual, dimension = [], None
            for row in raw:
                vector = self._vector(row, dimension)
                dimension = len(vector)
                actual.append(vector)
            vectors = {}
            if portable:
                report = portable.validate_probe_vectors(actual, threshold=.999)
                state.update(status='verified' if report['compatible'] else 'incompatible',
                             probe_verified=report['compatible'], invalid=not report['compatible'],
                             probe_cosines=report['cosines'], probe_threshold=report['threshold'])
                if report['compatible']:
                    matched, _ = portable.match_inputs(texts)
                    vectors.update({self._text_hash(texts[cid]): tuple(row) for cid, row in matched.items()})
                else:
                    portable = None
                    warnings.append('Векторы API не совпали с переносимым индексом; смешивание запрещено, доступен поиск по словам и ограниченная перестройка.')
            # Persisted online rows also require live verification, even if the
            # endpoint/model strings match: providers can replace model weights.
            if (disk and disk['dimension'] == dimension
                    and [p['text'] for p in disk['probes']] == probe_texts
                    and all(sum(a*b for a,b in zip(p['vector'], actual[i])) >= .999
                            for i,p in enumerate(disk['probes']))):
                vectors.update(disk['vectors'])
                state['runtime_disk_verified'] = True
            with self._state_lock:
                self._vectors, self._dimension = vectors, dimension
                self._portable, self._space_ready = portable, True
                self._live_probes = [{'text': text, 'vector': vector} for text,vector in zip(probe_texts, actual)]
        except Exception:
            state.update(status='probe_error', invalid=True, probe_verified=False)
            self._probe_retry_after = time.monotonic() + 5
            warnings.append('Не удалось проверить совместимость эмбеддингов; используется поиск по словам.')
        state['probe_attempts'] = self._probe_attempts
        with self._state_lock:
            self._portable_state = state

    def dense_search(self, query, *, scope=None, deadline=None):
        """Return dense candidates only; caller preserves lexical coverage/RRF."""
        started = time.monotonic()
        allowed = self._allowed(scope)
        trace = self._new_trace(scope, allowed, 'lexical')
        trace['cache'].update(api_calls=0, reused=0, missing=len(allowed), query_success=False)
        warnings, dense = [], []
        if (os.getenv('ENABLE_EMBEDDINGS', '1').strip().lower() not in {'0','false','off'}
                and getattr(self.client, 'embeddings_enabled', False) and allowed
                and isinstance(query, str) and query.strip()):
            try:
                dense = self._dense(query, allowed, deadline if deadline is not None else started + 15, trace, warnings)
            except Exception:
                trace['cache']['status'] = 'error'
                warnings.append('Семантический поиск недоступен; использован поиск по словам.')
        if dense:
            trace['strategy'] = 'dense'
        self._stage(trace, 'dense', dense)
        trace['timings_ms']['dense'] = round((time.monotonic() - started) * 1000, 2)
        return RetrievalResult(dense, trace, warnings, complete=not trace['cache'].get('missing_vectors', 0), ranked=dense)

    def _dense(self, query, allowed, deadline, trace, warnings):
        key, texts = self._fingerprint()
        texts = {cid: texts[cid] for cid in allowed}
        hashes = {cid: self._text_hash(texts[cid]) for cid in allowed}
        cache = trace['cache']
        cache.update(status='hit', key=key[:16], requested=len(allowed), batches=0,
                     api_calls=0, reused=0, missing=len(allowed), query_success=False)
        if time.monotonic() >= deadline - .25:
            cache['status'] = 'deadline'
            return []
        if not self._builder_lock.acquire(timeout=min(1.5, max(0., deadline - time.monotonic() - 2))):
            cache['status'] = 'building'
            warnings.append('Семантический индекс готовится другим запросом; использован поиск по словам.')
            return []
        try:
            retry_probe = (self._portable_state.get('status') == 'probe_error'
                           and self._probe_attempts < 2 and time.monotonic() >= self._probe_retry_after)
            if self._cache_key != key or retry_probe:
                self._initialize_space(key, texts, deadline, cache, warnings)
            with self._state_lock:
                cache['portable'] = dict(self._portable_state)
                cache['probe_verified'] = self._portable_state.get('probe_verified', False)
                cache['invalid'] = self._portable_state.get('invalid', False)
                if not self._space_ready:
                    cache['status'] = 'unverified'
                    return []
                if self._portable is not None:
                    matches, counts = self._portable.match_inputs(texts)
                    self._vectors.update({self._text_hash(texts[cid]): tuple(row) for cid,row in matches.items()})
                    cache['portable']['matches'] = counts
                missing = sorted(cid for cid in allowed if hashes[cid] not in self._vectors)
                cache['reused'] = len(allowed) - len(missing)
                cache['missing'] = len(missing)
                dimension = self._dimension
            # Duplicate exact inputs share one row, even after document IDs move.
            by_hash = {hashes[cid]: cid for cid in missing}
            limit = self._bounded_setting('EMBEDDING_MAX_NEW_CHUNKS', 64, 1024)
            pending = list(by_hash)[:limit]
            for offset in range(0, len(pending), self.BATCH_SIZE):
                # Leave a query call a chance; generation has its own outer budget.
                if time.monotonic() >= deadline - 2:
                    cache['status'] = 'partial'
                    break
                batch = pending[offset:offset + self.BATCH_SIZE]
                try:
                    raw = self._embed([texts[by_hash[digest]] for digest in batch], deadline - 1, cache)
                    if not isinstance(raw, (list, tuple)) or len(raw) != len(batch):
                        raise ValueError('Embedding coverage mismatch')
                    vectors = [self._vector(row, dimension) for row in raw]
                    with self._state_lock:
                        self._vectors.update(zip(batch, vectors))
                    cache['batches'] += 1
                    self._save_disk(key, warnings)
                except Exception:
                    cache['status'] = 'partial'
                    warnings.append('Часть новых эмбеддингов недоступна; поиск по словам сохраняет все допустимые документы.')
                    break
            with self._state_lock:
                vectors = {cid: self._vectors[digest] for cid,digest in hashes.items() if digest in self._vectors}
                cache['cached_vectors'] = len(vectors)
                cache['missing_vectors'] = len(allowed) - len(vectors)
            if cache['missing_vectors']:
                cache['status'] = 'partial'
            elif cache['batches']:
                cache['status'] = 'extended' if cache['reused'] else 'built'
        finally:
            self._builder_lock.release()
        if not vectors or time.monotonic() >= deadline - .25:
            cache['query_status'] = 'no_vectors' if not vectors else 'deadline'
            return []
        try:
            vector = self._query_vector(query, key, dimension, deadline, cache)
            scored = [(sum(x*y for x,y in zip(vector,row)), cid) for cid,row in vectors.items()]
            scored.sort(key=lambda item: (-item[0], item[1]))
            cache['query_success'] = True
            cache['query_status'] = 'success'
            return [replace(allowed[cid], score=score) for score,cid in scored[:self.DENSE_CANDIDATES]]
        except Exception:
            cache['query_status'] = 'error'
            warnings.append('Эмбеддинг запроса не получен или не прошёл проверку; использован поиск по словам.')
            return []

    def _query_vector(self, query, corpus_key, dimension, deadline, trace):
        identity = (corpus_key, dimension, query)
        with self._state_lock:
            cached = self._query_vectors.get(identity)
            if cached is not None:
                self._query_vectors.move_to_end(identity)
                trace['query_embedding_cache'] = 'hit'
                return cached
        raw = self._embed([query], deadline, trace)
        if not isinstance(raw, (list, tuple)) or len(raw) != 1:
            raise ValueError('Query embedding coverage')
        vector = self._vector(raw[0], dimension)
        with self._state_lock:
            self._query_vectors[identity] = vector
            self._query_vectors.move_to_end(identity)
            while len(self._query_vectors) > 256:
                self._query_vectors.popitem(last=False)
        trace['query_embedding_cache'] = 'miss'
        return vector

    @staticmethod
    def _dedup(chunks, protected=frozenset()):
        """Remove source-span containment, never distinct same-clause fragments."""
        kept = []
        for chunk in chunks:
            if any(c.id == chunk.id for c in kept):
                continue
            if chunk.id not in protected and any(c.doc_id == chunk.doc_id and c.start <= chunk.start and chunk.end <= c.end for c in kept):
                continue
            contained = [i for i, c in enumerate(kept) if c.id not in protected and c.doc_id == chunk.doc_id
                         and chunk.start <= c.start and c.end <= chunk.end]
            if contained:
                position = min(contained)
                kept = [c for i, c in enumerate(kept) if i not in contained]
                kept.insert(position, chunk)
            else:
                kept.append(chunk)
        return kept

    def _component(self, anchor, allowed):
        # "текст" labels and editorial labels do not identify legal provisions.
        if anchor.content_role != 'body' or anchor.clause in {'', 'текст'}:
            return [anchor]
        peers = [c for c in allowed.values() if c.doc_id == anchor.doc_id and c.clause == anchor.clause
                 and c.section == anchor.section and c.kind == anchor.kind and c.content_role == 'body']
        connected = {anchor.id: anchor}
        lo, hi = anchor.start, anchor.end
        changed = True
        while changed:
            changed = False
            for chunk in peers:
                if chunk.id not in connected and chunk.start < hi and lo < chunk.end:
                    connected[chunk.id] = chunk
                    lo, hi = min(lo, chunk.start), max(hi, chunk.end)
                    changed = True
        return sorted(connected.values(), key=lambda c: (c.start, c.end))

    def _parent(self, anchor, allowed):
        clause = anchor.clause
        parent = None
        prefix_number = None
        if re.fullmatch(r'\d+(?:\.\d+)+', clause):
            profile = getattr(self.registry, 'profiles', {}).get(anchor.doc_id) if self.registry else None
            document = self.corpus.get_document(anchor.doc_id)
            if (not anchor.section or getattr(profile, 'document_type', None) == 'amendment'
                    or (document and (document.metadata.get('is_amendment')
                                      or document.metadata.get('document_type') == 'amendment'))):
                return None
            parent = clause.rsplit('.', 1)[0]
            prefix_number = parent
        elif ', подп. ' in clause and ', п. ' in clause:
            parent = clause.rsplit(', подп. ', 1)[0]
            prefix_number = parent.rsplit(', п. ', 1)[1]
        if parent is None:
            return None
        candidates = [c for c in allowed.values() if c.doc_id == anchor.doc_id and c.clause == parent
                      and c.content_role in {'body', 'heading'} and c.start < anchor.start
                      and c.section == anchor.section and c.kind == anchor.kind
                      and (anchor.kind == 'law' or c.text.strip() == anchor.section.strip())
                      and re.match(r'^\s*[«"]?\s*' + re.escape(prefix_number) + r'\.\s', c.text)]
        return max(candidates, key=lambda c: c.start) if candidates else None

    def _expand(self, anchor, allowed):
        items = self._component(anchor, allowed)
        parent = self._parent(anchor, allowed)
        if parent is not None:
            items = self._component(parent, allowed) + items
        return list({c.id: c for c in items}.values())

    @staticmethod
    def _cost(chunk):
        return len(chunk.text) + len(chunk.title) + len(chunk.clause) + 50

    def _pack(self, ranked, allowed, required, rejected_required, k, max_chars, trace, warnings):
        k, max_chars = max(0, int(k)), max(0, int(max_chars))
        required_ids = {c.id for c in required}
        selected, used = {}, 0
        missing_required, dropped, expanded_ids, blocked_ids = [], [], [], []
        complete = not rejected_required
        safe = self._allowed(None)

        def add(chunk, essential=False):
            nonlocal used, complete
            if chunk.id in selected:
                return True
            cost = self._cost(chunk)
            if len(selected) >= k or used + cost > max_chars:
                complete = False
                if chunk.id not in dropped:
                    dropped.append(chunk.id)
                if essential and chunk.id not in missing_required:
                    missing_required.append(chunk.id)
                return False
            selected[chunk.id] = chunk
            used += cost
            return True

        # Preserve explicitly required IDs before spending budget on context.
        for chunk in required:
            add(chunk, True)
        anchors = list({c.id: c for c in required + ranked[:k]}.values())
        for anchor in anchors:
            add(anchor, anchor.id in required_ids)
            # Scope can forbid a needed split/lead; detect the loss without
            # crossing the scope or sending the blocked text to a model.
            members = self._expand(anchor, safe)
            for chunk in members:
                if chunk.id not in allowed:
                    complete = False
                    if chunk.id not in blocked_ids:
                        blocked_ids.append(chunk.id)
                    continue
                if chunk.id not in expanded_ids:
                    expanded_ids.append(chunk.id)
                add(chunk, chunk.id in required_ids)
        chunks = list(selected.values())
        self._stage(trace, 'expanded', [allowed[cid] for cid in expanded_ids])
        self._stage(trace, 'packed', chunks)
        trace['budget'] = {'k': k, 'max_chars': max_chars, 'used_chars': used,
                           'required_ids': [c.id for c in required], 'missing_required_ids': missing_required,
                           'rejected_required_count': len(rejected_required), 'dropped_ids': dropped,
                           'blocked_context_ids': blocked_ids,
                           'complete': complete}
        if not complete:
            warnings.append('Не весь обязательный или связанный контекст поместился в лимит либо разрешённую область; полнота ответа не подтверждена.')
        return RetrievalResult(chunks=chunks, ranked=ranked, trace=trace, warnings=warnings, complete=complete)

    def pack(self, chunks, *, scope=None, required=(), k=8, max_chars=8500):
        started = time.monotonic()
        allowed = self._allowed(scope)
        trace = self._new_trace(scope, allowed, 'policy_context')
        warnings = []
        anchors, _ = self._inputs(chunks, allowed)
        mandatory, rejected = self._inputs(required, allowed)
        ranked = self._dedup(anchors, {c.id for c in mandatory})
        self._stage(trace, 'ranked', ranked)
        result = self._pack(ranked, allowed, mandatory, rejected, k, max_chars, trace, warnings)
        trace['timings_ms']['total'] = round((time.monotonic() - started) * 1000, 2)
        return result

    def search(self, query, *, scope=None, seeds=(), required=(), k=8, max_chars=8500, deadline=None):
        started = time.monotonic()
        allowed = self._allowed(scope)
        trace = self._new_trace(scope, allowed, 'lexical')
        warnings = []
        seed_chunks, rejected_seeds = self._inputs(seeds, allowed)
        mandatory, rejected = self._inputs(required, allowed)
        self._stage(trace, 'seeds', seed_chunks)
        trace['rejected_seed_count'] = len(rejected_seeds)
        deadline = deadline if deadline is not None else started + 35
        if not allowed or not isinstance(query, str) or not query.strip():
            return self._pack([], allowed, mandatory, rejected, k, max_chars, trace, warnings)
        if mandatory and {c.id for c in mandatory} == set(allowed):
            trace['strategy'] = 'policy_exact'
            ranked = self._dedup(mandatory, {c.id for c in mandatory})
            self._stage(trace, 'ranked', ranked)
            return self._pack(ranked, allowed, mandatory, rejected, k, max_chars, trace, warnings)
        before = time.monotonic()
        sparse = self.corpus.search(query, k=self.SPARSE_CANDIDATES,
                                    doc_ids=frozenset(c.doc_id for c in allowed.values()),
                                    chunk_ids=frozenset(allowed))
        sparse = [replace(allowed[c.id], score=c.score) for c in sparse if c.id in allowed]
        self._stage(trace, 'sparse', sparse)
        trace['timings_ms']['sparse'] = round((time.monotonic() - before) * 1000, 2)
        dense = []
        if (os.getenv('ENABLE_EMBEDDINGS', '1').strip().lower() not in {'0','false','off'}
                and getattr(self.client, 'embeddings_enabled', False)):
            before = time.monotonic()
            dense = self._dense(query, allowed, deadline, trace, warnings)
            trace['timings_ms']['dense'] = round((time.monotonic() - before) * 1000, 2)
            if dense:
                trace['strategy'] = 'hybrid'
        self._stage(trace, 'dense', dense)
        scores = {}
        for ranking in (sparse, dense, seed_chunks):
            for position, chunk in enumerate(ranking, 1):
                if chunk.id in allowed:
                    scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (60 + position)
        candidates = [replace(allowed[cid], score=scores[cid]) for cid in sorted(scores, key=lambda cid: (-scores[cid], cid))[:self.FUSION_CANDIDATES]]
        self._stage(trace, 'fused', candidates)
        if getattr(self.client, 'reranker_enabled', False) and candidates and time.monotonic() < deadline - 0.25:
            before = time.monotonic()
            try:
                reordered = self.client.rerank(query, candidates, deadline)
                ids = [c.id for c in reordered]
                permitted = {c.id: c for c in candidates}
                if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(permitted):
                    raise ValueError('Invalid rerank coverage')
                candidates = [permitted[cid] for cid in ids]
                trace['rerank_returned'] = len(ids)
            except Exception:
                warnings.append('Реранжирование недоступно или вернуло недопустимые источники; сохранён исходный порядок.')
            trace['timings_ms']['rerank'] = round((time.monotonic() - before) * 1000, 2)
        self._stage(trace, 'reranked', candidates)
        ranked = self._dedup(candidates, {c.id for c in mandatory})
        self._stage(trace, 'ranked', ranked)
        before = time.monotonic()
        result = self._pack(ranked, allowed, mandatory, rejected, k, max_chars, trace, warnings)
        trace['timings_ms']['pack'] = round((time.monotonic() - before) * 1000, 2)
        trace['timings_ms']['total'] = round((time.monotonic() - started) * 1000, 2)
        return result
