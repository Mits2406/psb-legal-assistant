"""Behavioral contracts with controlled vectors; not semantic-quality evidence."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chunk import Chunk
from document import Document
from evidence import EvidencePlanner
from retrieval import Retriever, SearchScope
from vector_index import build_artifact, write_module


def basis(index):
    result = [0.] * 1024
    result[index] = 1.
    return result


class TinyCorpus:
    def __init__(self):
        self.chunks = {}
        self.documents = {}
        self.lexical = {}
        self.stats = {'corpus_hash': 'a' * 64}
        self.add('a', 'Покупатель принимает поставку в течение установленного срока и подписывает акт.')
        self.add('b', 'Исполнитель передает заказчику все исходные материалы результата работы по акту.')
        self.add('c', 'При нарушении обязательства сторона направляет письменную претензию контрагенту.')

    def add(self, cid, text):
        self.chunks[cid] = Chunk(cid, cid, 'Документ ' + cid, '1', text, 0, len(text), False, 'Условия', 'company', 'body', True)
        self.documents[cid] = Document(cid, 'Документ ' + cid, cid + '.txt', text, 'company', None, {}, 'a' * 64)

    def get_document(self, did):
        return self.documents.get(did)

    def search(self, query, **kwargs):
        permitted = kwargs.get('chunk_ids', self.chunks)
        return [replace(self.chunks[cid], score=score) for cid,score in self.lexical.get(query, [('a', 10.), ('c', 4.)])
                if cid in permitted]


class ControlledClient:
    embeddings_enabled = enabled = True
    reranker_enabled = False
    embedding_model = 'BAAI/bge-m3'
    base_url = 'https://new-proxy.example/v1'

    def __init__(self):
        self.calls = []
        self.vectors = {}
        self.default = basis(1)
        self.failure = None

    def embed(self, texts, deadline):
        self.calls.append(list(texts))
        if self.failure:
            raise self.failure
        return [self.vectors.get(text, self.default) for text in texts]


class HybridIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'precomputed_index.py'
        self.corpus, self.client = TinyCorpus(), ControlledClient()
        self.retriever = Retriever(self.corpus, self.client, portable_index_path=self.path)
        self.inputs = {cid: self.retriever.contextual_text(c) for cid,c in self.corpus.chunks.items()}
        self.probes = [{'text': 'probe alpha', 'vector': basis(3)}, {'text': 'probe beta', 'vector': basis(4)}]
        rows = {cid: basis(index) for index,cid in enumerate(self.inputs)}
        self.client.vectors.update({self.inputs[cid]: row for cid,row in rows.items()})
        self.client.vectors.update({p['text']: p['vector'] for p in self.probes})
        write_module(self.path, build_artifact(self.inputs, rows,
                     model_revision='unit-fixture-only', corpus_hash='a' * 64, chunker_fingerprint='b' * 64,
                     weights_provenance={'source': 'controlled fixtures, no semantic quality claim'}, probes=self.probes))
        self.env = patch.dict('os.environ', {'ENABLE_EMBEDDINGS': '1', 'EMBEDDING_MAX_NEW_CHUNKS': '64'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def dense(self, query='query', retriever=None, **kwargs):
        return (retriever or self.retriever).dense_search(query, deadline=time.monotonic() + 10, **kwargs)

    def test_reuses_portable_rows_across_endpoint_and_checks_probes_once(self):
        result = self.dense()
        self.assertEqual(result.chunks[0].id, 'b')
        self.assertEqual(self.client.calls, [['probe alpha', 'probe beta'], ['query']])
        self.assertEqual(result.trace['cache']['reused'], 3)
        self.assertEqual(result.trace['cache']['missing'], 0)
        self.assertTrue(result.trace['cache']['probe_verified'])
        self.assertTrue(result.trace['cache']['query_success'])
        second = self.dense('query two')
        self.assertEqual(second.trace['cache']['api_calls'], 1)
        self.assertEqual(len(self.client.calls), 3)
        third = self.dense('query two')
        self.assertEqual(third.trace['cache']['api_calls'], 0)
        self.assertEqual(third.trace['cache']['query_embedding_cache'], 'hit')

    def test_changed_inputs_embed_only_delta_removed_and_renamed_ids(self):
        self.dense()
        self.client.calls.clear()
        original = self.corpus.chunks.pop('a')
        self.corpus.chunks['renamed'] = replace(original, id='renamed')
        self.corpus.chunks.pop('c')
        self.corpus.add('b', 'Совершенно новая действующая редакция обязательства стороны договора с иной процедурой.')
        new_input = self.retriever.contextual_text(self.corpus.chunks['b'])
        result = self.dense('new question')
        self.assertEqual(self.client.calls, [[new_input], ['new question']])
        self.assertEqual(result.trace['cache']['reused'], 1)
        self.assertEqual(result.trace['cache']['missing'], 1)
        self.assertEqual({c.id for c in result.chunks}, {'renamed', 'b'})

    def test_mismatching_probe_never_uses_portable_rows(self):
        self.client.vectors['probe alpha'] = basis(9)
        with patch.dict('os.environ', {'EMBEDDING_MAX_NEW_CHUNKS': '0'}):
            result = self.dense()
        self.assertEqual(result.chunks, [])
        self.assertTrue(result.trace['cache']['invalid'])
        self.assertFalse(result.trace['cache']['probe_verified'])
        self.assertEqual(result.trace['cache']['reused'], 0)
        self.assertEqual(self.client.calls, [['probe alpha', 'probe beta']])
        self.assertTrue(result.warnings)

    def test_mismatching_probe_rebuilds_only_current_provider(self):
        self.client.vectors['probe alpha'] = basis(9)
        for text in self.inputs.values():
            self.client.vectors[text] = basis(8)
        result = self.dense()
        self.assertTrue(result.trace['cache']['invalid'])
        self.assertEqual(result.trace['cache']['reused'], 0)
        self.assertEqual(result.trace['cache']['batches'], 1)
        self.assertEqual(len(result.chunks), 3)
        self.assertTrue(all(c.score == 0. for c in result.chunks))

    def test_probe_error_has_safe_fallback_and_is_not_retried_per_part(self):
        self.client.failure = TimeoutError('fixture')
        first, second = self.dense(), self.dense('another question')
        self.assertEqual(first.chunks + second.chunks, [])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(first.trace['cache']['portable']['status'], 'probe_error')
        self.assertFalse(second.trace['cache']['query_success'])

    def test_transient_probe_failure_can_retry_once_after_backoff(self):
        self.client.failure = TimeoutError('fixture')
        self.dense()
        self.client.failure = None
        self.assertFalse(self.dense('early retry').chunks)
        self.assertEqual(len(self.client.calls), 1)
        self.retriever._probe_retry_after = time.monotonic() - 1
        recovered = self.dense('recovered')
        self.assertTrue(recovered.chunks)
        self.assertEqual(recovered.trace['cache']['portable']['probe_attempts'], 2)
        self.assertEqual(len(self.client.calls), 3)

    def test_repeated_probe_failure_stops_after_two_attempts(self):
        self.client.failure = TimeoutError('fixture')
        for _ in range(4):
            self.retriever._probe_retry_after = time.monotonic() - 1
            self.assertFalse(self.dense().chunks)
        self.assertEqual(len(self.client.calls), 2)

    def test_disabled_and_expired_budget_do_not_call_provider(self):
        with patch.dict('os.environ', {'ENABLE_EMBEDDINGS': '0'}):
            self.assertFalse(self.dense().chunks)
        result = self.retriever.dense_search('query', deadline=time.monotonic() - 1)
        self.assertFalse(result.chunks)
        self.assertEqual(self.client.calls, [])

    def test_finite_delta_budget_preserves_partial_dense_candidates(self):
        self.corpus.add('d', 'Дополнительное обязательство, отсутствующее среди предвычисленных источников и положений.')
        with patch.dict('os.environ', {'EMBEDDING_MAX_NEW_CHUNKS': '0'}):
            result = self.dense()
        self.assertEqual({c.id for c in result.chunks}, {'a','b','c'})
        self.assertEqual(result.trace['cache']['missing_vectors'], 1)
        self.assertFalse(result.complete)
        self.assertTrue(result.trace['cache']['query_success'])

    def test_concurrent_builder_falls_back_without_duplicate_probe(self):
        entered, release = threading.Event(), threading.Event()
        original = self.client.embed
        def blocked(texts, deadline):
            if texts == ['probe alpha', 'probe beta']:
                entered.set()
                release.wait(2)
            return original(texts, deadline)
        self.client.embed = blocked
        results = []
        worker = threading.Thread(target=lambda: results.append(self.dense()))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            fallback = self.dense('concurrent')
            self.assertEqual(fallback.trace['cache']['status'], 'building')
            self.assertFalse(fallback.chunks)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].chunks)
        self.assertEqual(len(self.client.calls), 2)

    def test_disk_rows_are_verified_before_reuse(self):
        cache_dir = Path(self.temp.name) / 'cache'
        retriever = Retriever(self.corpus, self.client, cache_dir=cache_dir, portable_index_path=self.path)
        self.corpus.add('d', 'Дополнительный документ, который может быть переиспользован после перезапуска процесса.')
        self.dense(retriever=retriever)
        self.client.calls.clear()
        restarted = Retriever(self.corpus, self.client, cache_dir=cache_dir, portable_index_path=self.path)
        result = self.dense(retriever=restarted)
        self.assertEqual(len(self.client.calls), 2)
        self.assertTrue(result.trace['cache']['portable']['runtime_disk_verified'])
        self.assertEqual(result.trace['cache']['reused'], 4)

    def planner(self):
        registry = SimpleNamespace(profiles={}, amendments=[])
        planner = EvidencePlanner(self.corpus, registry, self.client)
        planner.structure.portable_index_path = self.path
        return planner

    def test_evidence_rrf_keeps_lexical_union_and_leader(self):
        planner = self.planner()
        plan = planner.build('query', {}, deadline=time.monotonic() + 10)
        self.assertEqual(plan['trace']['strategy'], 'global_hybrid_rrf_parts_v15')
        self.assertEqual(set(plan['trace']['ranked_chunk_ids']), {'a','b','c'})
        self.assertEqual(plan['trace']['lexical_leaders_preserved'], ['a'])
        self.assertIn('a', {s['chunk_id'] for s in plan['sources']})
        self.assertIn('b', {s['chunk_id'] for s in plan['sources']})
        self.assertEqual(plan['trace']['api_calls'], 2)
        for source in plan['sources']:
            self.assertEqual(source['text'], self.corpus.chunks[source['chunk_id']].text)

    def test_contract_scope_is_shared_by_lexical_and_dense(self):
        planner = self.planner()
        planner.registry.profiles = {
            'a': SimpleNamespace(contract_numbers=('11-А',), document_type='contract'),
            'b': SimpleNamespace(contract_numbers=('22-Б',), document_type='contract'),
        }
        ranked, parts, _, _, searches, state = planner._retrieve(['query'], {'contracts':['11-А']}, time.monotonic()+10)
        self.assertNotIn('b', {c.id for c in ranked})
        self.assertIn('a', {c.id for c in ranked})
        dense = next(s for s in searches if s['kind'] == 'dense')
        self.assertNotIn('b', dense['candidate_ids'])

    def test_each_question_part_keeps_lexical_leader_after_fusion(self):
        self.corpus.lexical = {'first': [('a', 10.)], 'second': [('c', 10.)]}
        planner = self.planner()
        plan = planner.build('first', {'_normalization_queries':['first','second']}, deadline=time.monotonic()+10)
        self.assertEqual(plan['trace']['lexical_leaders_preserved'], ['a', 'c'])
        ids = {s['chunk_id'] for s in plan['sources']}
        self.assertTrue({'a','c'} <= ids)


if __name__ == '__main__':
    unittest.main()
