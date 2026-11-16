"""Authorization before every RAG stage, including cache/retry boundaries.

Responses/vectors are deterministic fixtures; these tests make no model-quality
claims and send no requests to an external service.
"""
import copy
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from access_control import AccessError, AccessStore
from answer import LegalAssistant
from chunk import Chunk
from llm import APIError, ModelClient
from scoped_assistant import GuardedClient, ScopedManager

HIDDEN = 'doc_b8d10bac552077f9'
SENTINEL = 'PRIVATE_ACL_SENTINEL_85A9'
QUESTION = 'Когда платить по договору А-08/2024?'
USERS = {'user': {'password': 'user-test-password', 'level': 1},
         'admin': {'password': 'admin-test-password', 'level': 2}}


class CapturingClient:
    enabled = embeddings_enabled = reranker_enabled = True
    embedding_model = 'fixture-embedding'
    reranker_model = 'fixture-reranker'
    base_url = 'https://fixture.invalid/v1'

    def __init__(self):
        self.records = []
        self._local = threading.local()
        self.last_rerank_diagnostics = {}

    @property
    def last_chat_diagnostics(self):
        return getattr(self._local, 'diagnostics', {})

    def record(self, stage, payload):
        self.records.append((stage, copy.deepcopy(payload)))
        self._local.diagnostics = {'api_attempted': True, 'api_attempts': 1,
            'api_http_success': True, 'model_text_received': True,
            'model_text_usable': True, 'finish_reason': 'stop'}

    def chat_json(self, system, data, deadline=None, *, max_tokens=None):
        self.record('normalization', {'system': system, 'data': data})
        user = [m for m in data['messages'] if m['role'] == 'user'][-1]
        return {'task_type': 'application', 'topic': 'payment', 'question': user['content'],
            'task_message_ids': [user['id']], 'topic_changed': False,
            'subquestions': [{'text': user['content'], 'evidence': [{'message_id': user['id'], 'quote': user['content']}]}],
            'facts': [], 'unknown_facts': [], 'corrections': []}

    def chat_text(self, system, data, deadline=None, *, max_tokens=None):
        stage = 'review' if 'draft' in data else 'draft'
        self.record(stage, {'system': system, 'data': data})
        if stage == 'review':
            return data['draft']
        source = next((s for s in data['sources'] if '5-го числа' in s['text']), data['sources'][0])
        return source['text'] + ' [' + source['id'] + ']'

    def embed(self, texts, deadline=None):
        self.record('embedding', texts)
        return [[1.0, 0.0] for _ in texts]

    def rerank(self, query, chunks, deadline=None):
        self.record('rerank', {'query': query, 'chunks': [c.to_dict() for c in chunks]})
        self.last_rerank_diagnostics = {'api_attempted': True, 'api_attempts': 1,
                                       'api_http_success': True, 'returned': len(chunks)}
        return list(chunks)


class ScopedAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {'QUERY_NORMALIZE': '1', 'ANSWER_VERIFY': '1',
            'ENABLE_EMBEDDINGS': '1', 'ENABLE_RERANKER': '1', 'EMBEDDING_MAX_NEW_CHUNKS': '1024',
            'RETRIEVAL_CACHE_DIR': str(Path(self.temp.name) / 'vectors')}, clear=True)
        env.start(); self.addCleanup(env.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No external network'))
        network.start(); self.addCleanup(network.stop)
        self.store = AccessStore(Path(self.temp.name) / 'state', users=USERS)
        self.addCleanup(self.store.close)
        self.base = LegalAssistant(ROOT / 'data')
        # Keep the real amendment and add a unique marker to its title/body;
        # any cross-stage leak is detectable independently of candidate ranks.
        original = self.base.corpus.documents[HIDDEN]
        extra = '\n99.9. ' + SENTINEL + ': закрытое условие оплаты.\n'
        content = original.text + extra
        title = original.title + ' ' + SENTINEL
        self.base.corpus.documents[HIDDEN] = replace(original, title=title, text=content,
                                                   sha256=sha256(content.encode()).hexdigest())
        for cid, chunk in list(self.base.corpus.chunks.items()):
            if chunk.doc_id == HIDDEN:
                self.base.corpus.chunks[cid] = replace(chunk, title=title)
        self.base.corpus.chunks['private-extra'] = Chunk('private-extra', HIDDEN, title, '99.9', extra,
            len(original.text), len(content), False, 'Условия', 'company', 'body', True)
        self.base.corpus._build_index()
        self.client = CapturingClient()
        self.base.client = self.client
        self.manager = ScopedManager(self.base, self.store)

    def scoped(self, level=1):
        revision, labels = self.store.snapshot(level)
        scoped = self.manager.get(level, revision, labels)
        # No unrelated artifact/model compatibility behaviour in fixture tests.
        scoped.planner.structure.portable_index_path = Path(self.temp.name) / 'no-artifact.py'
        return scoped

    def test_unmarked_and_level_one_are_visible_level_two_is_not(self):
        original_ids = set(self.base.corpus.documents)
        self.store.set_level(HIDDEN, 2)
        explicit = next(did for did in original_ids if did != HIDDEN)
        self.store.set_level(explicit, 1)
        limited, full = self.scoped(1), self.scoped(2)
        self.assertEqual(set(limited.corpus.documents), original_ids - {HIDDEN})
        self.assertEqual(set(full.corpus.documents), original_ids)
        self.assertEqual(set(self.base.corpus.documents), original_ids)
        self.assertNotEqual(limited.corpus.stats['corpus_hash'], full.corpus.stats['corpus_hash'])
        self.assertEqual(limited.corpus.stats['documents'], 20)
        self.assertFalse(any(c.doc_id == HIDDEN for c in limited.corpus.chunks.values()))
        self.assertTrue(any(c.doc_id == HIDDEN for c in full.corpus.chunks.values()))
        self.assertIn(SENTINEL, self.base.corpus.documents[HIDDEN].text)

    def test_hidden_amendment_is_absent_before_registry_dependency_and_search(self):
        self.store.set_level(HIDDEN, 2)
        limited, full = self.scoped(1), self.scoped(2)
        self.assertNotIn(HIDDEN, limited.registry.profiles)
        self.assertFalse(any(a.doc_id == HIDDEN for a in limited.registry.amendments))
        self.assertTrue(any(a.doc_id == HIDDEN for a in full.registry.amendments))
        self.assertNotIn(HIDDEN, limited.planner.by_doc)
        self.assertFalse(any(c.doc_id == HIDDEN for c in limited.planner.safe.values()))
        self.assertFalse(any(c.doc_id == HIDDEN for c in limited.corpus.search('оплата ' + SENTINEL)))
        plan = limited.planner.build('Как оплатить поставку по договору П-17/2024?', {},
                                     deadline=time.monotonic() + 90)
        self.assertNotIn(HIDDEN, {s['doc_id'] for s in plan['sources']})
        self.assertNotIn(SENTINEL, json.dumps(plan, ensure_ascii=False))

    def test_all_live_model_stages_receive_only_authorized_corpus(self):
        self.store.set_level(HIDDEN, 2)
        assistant = self.scoped()
        result = assistant.answer(QUESTION, reference_date='2026-11-15', deadline=time.monotonic() + 90)
        self.assertEqual(result['mode'], 'llm')
        self.assertTrue({'normalization', 'embedding', 'rerank', 'draft', 'review'} <=
                        {stage for stage, _ in self.client.records})
        recorded = json.dumps(self.client.records, ensure_ascii=False)
        self.assertNotIn(SENTINEL, recorded)
        self.assertNotIn(HIDDEN, recorded)
        self.assertNotIn(SENTINEL, json.dumps(result, ensure_ascii=False))
        self.assertNotIn(HIDDEN, json.dumps(result, ensure_ascii=False))

    def test_level_two_can_use_closed_document_then_level_one_cannot_reuse_it(self):
        self.store.set_level(HIDDEN, 2)
        question = 'Как оплатить поставку по договору П-17/2024?'
        self.scoped(2).answer(question, reference_date='2026-11-15', deadline=time.monotonic() + 90)
        for stage in ('normalization', 'embedding', 'rerank', 'draft', 'review'):
            self.assertTrue(any(name == stage and SENTINEL in str(payload)
                                for name, payload in self.client.records), stage)
        self.client.records.clear()
        limited = self.scoped(1)
        limited.answer(question, reference_date='2026-11-15', deadline=time.monotonic() + 90)
        self.assertTrue(self.client.records)
        self.assertNotIn(SENTINEL, json.dumps(self.client.records, ensure_ascii=False))
        self.assertNotIn(HIDDEN, json.dumps(self.client.records, ensure_ascii=False))

    def test_all_hidden_has_empty_corpus_and_safe_no_source_answer(self):
        for did in self.base.corpus.documents:
            self.store.set_level(did, 2)
        limited = self.scoped()
        self.assertEqual(limited.corpus.stats['documents'], 0)
        self.assertEqual(limited.corpus.stats['chunks'], 0)
        self.assertEqual(limited.registry.profiles, {})
        self.assertEqual(limited.corpus.search('оплата'), [])
        result = limited.answer(QUESTION, deadline=time.monotonic() + 90)
        self.assertFalse(result.get('citations'))
        self.assertNotIn(SENTINEL, json.dumps(result, ensure_ascii=False))
        self.assertNotIn('embedding', {stage for stage, _ in self.client.records})
        self.assertNotIn('rerank', {stage for stage, _ in self.client.records})

    def test_revision_rebuilds_memory_and_disk_cache_without_mutating_old_scope(self):
        old = self.scoped(1)
        old.planner.structure.dense_search('оплата', deadline=time.monotonic() + 90)
        self.assertTrue(old.planner.structure._vectors)
        old_key = old.corpus.stats['corpus_hash']
        old_dir = old.planner.structure.cache_dir
        self.store.set_level(HIDDEN, 2)
        new = self.scoped(1)
        self.assertIsNot(new, old)
        self.assertNotEqual(new.corpus.stats['corpus_hash'], old_key)
        self.assertNotEqual(new.planner.structure.cache_dir, old_dir)
        self.assertEqual(new.planner.structure._vectors, {})
        self.assertEqual(new.planner.structure._query_vectors, {})
        self.client.records.clear()
        dense = new.planner.structure.dense_search('оплата', deadline=time.monotonic() + 90)
        self.assertTrue(dense.chunks)
        self.assertFalse(any(c.doc_id == HIDDEN for c in dense.chunks))
        self.assertNotIn(SENTINEL, json.dumps(self.client.records, ensure_ascii=False))
        self.assertIn(HIDDEN, old.corpus.documents)
        self.assertIs(self.scoped(1), new)
        self.assertIsNot(self.scoped(2), new)
        self.base.client = CapturingClient()
        self.assertIsNot(self.scoped(1), new)

    def test_guard_blocks_each_model_entry_point_after_revision_change(self):
        old = self.scoped()
        self.store.set_level(HIDDEN, 2)
        for name, args in [('chat_text', ('system', {})), ('chat_json', ('system', {})),
                           ('embed', (['text'],)), ('rerank', ('query', []))]:
            with self.subTest(method=name), self.assertRaises(APIError) as raised:
                getattr(old.client, name)(*args)
            self.assertEqual(raised.exception.code, 'access_changed')
        self.assertEqual(self.client.records, [])
        with self.assertRaises(AccessError) as stale:
            self.manager.get(1, 0, {})
        self.assertEqual(stale.exception.status, 409)
        revision, labels = self.store.snapshot(1)
        with self.assertRaises(AccessError) as forged:
            self.manager.get(1, revision, {})
        self.assertEqual(forged.exception.status, 403)

    def test_real_client_retry_never_resends_after_access_revocation(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://fixture.invalid/v1',
            'LLM_API_KEY': 'fixture-key', 'LLM_MODEL': 'fixture-model', 'API_RETRIES': '1'}):
            actual = ModelClient()
        guarded = GuardedClient(actual, self.store, 0)
        def fail_and_revoke(*args, **kwargs):
            self.store.set_level(HIDDEN, 2)
            raise urllib.error.URLError('fixture interruption')
        with patch('urllib.request.OpenerDirector.open', side_effect=fail_and_revoke) as network:
            with self.assertRaises(APIError) as raised:
                guarded.chat_text('system', {'question': 'query'})
        self.assertEqual(raised.exception.code, 'access_changed')
        self.assertEqual(network.call_count, 1)

    def test_real_client_successful_chat_embedding_and_rerank_keep_wire_contract(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://fixture.invalid/v1',
            'LLM_API_KEY': 'fixture-key', 'LLM_MODEL': 'fixture-model',
            'RERANKER_MODEL': 'fixture-reranker', 'API_RETRIES': '1'}):
            actual = ModelClient()
        guarded = GuardedClient(actual, self.store, 0)
        calls = []
        def response(request, **kwargs):
            payload = json.loads(request.data)
            calls.append((request.full_url, payload))
            if request.full_url.endswith('/embeddings'):
                value = {'data': [{'index': 0, 'embedding': [3, 4]}]}
            elif request.full_url.endswith('/rerank'):
                value = {'results': [{'index': 0, 'relevance_score': 1.0}]}
            elif len(calls) == 1:
                raise urllib.error.URLError('transient fixture error')
            else:
                value = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}]}
            return io.BytesIO(json.dumps(value).encode())
        chunk = next(iter(self.base.corpus.chunks.values()))
        with patch('urllib.request.OpenerDirector.open', side_effect=response), patch('scoped_assistant.time.sleep'):
            self.assertEqual(guarded.chat_json('system', {'question': 'test'}), {'ok': True})
            self.assertEqual(guarded.embed(['test']), [[.6, .8]])
            self.assertEqual(guarded.rerank('test', [chunk]), [chunk])
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][1]['model'], 'fixture-model')
        self.assertEqual(calls[-1][1]['documents'], [chunk.title + '\n' + chunk.text])
        self.assertEqual(actual.last_chat_diagnostics['api_attempts'], 2)

    def test_nested_real_client_post_is_rechecked_after_chat_entry(self):
        actual = ModelClient()
        guarded = GuardedClient(actual, self.store, 0)
        check = self.store.check_revision
        calls = []
        def revoke_before_nested(revision):
            calls.append(revision)
            if len(calls) == 2:
                self.store.set_level(HIDDEN, 2)
            check(revision)
        with patch.object(self.store, 'check_revision', side_effect=revoke_before_nested):
            with self.assertRaises(APIError) as raised:
                guarded.chat_json('system', {})
        self.assertEqual(raised.exception.code, 'access_changed')
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
