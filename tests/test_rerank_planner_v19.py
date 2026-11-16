"""Active-planner reranking contracts using controlled rankings, no model claims."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chunk import Chunk
from document import Document
from evidence import EvidencePlanner
from llm import APIError


class CorpusFixture:
    def __init__(self):
        self.documents, self.chunks, self.lexical = {}, {}, {}
        self.stats = {'corpus_hash': 'fixture'}

    def add(self, did, title, clauses):
        text = '\n'.join(body for _, _, body in clauses)
        self.documents[did] = Document(did, title, did + '.txt', text, 'company', None, {}, sha256(text.encode()).hexdigest())
        position = 0
        for cid, clause, body in clauses:
            self.chunks[cid] = Chunk(cid, did, title, clause, body, position, position + len(body),
                                     False, 'Условия', 'company', 'body', True)
            position += len(body) + 1

    def get_document(self, did):
        return self.documents.get(did)

    def search(self, query, **kwargs):
        permitted = kwargs.get('chunk_ids', self.chunks)
        ranking = self.lexical.get(query, self.lexical.get('*', []))
        return [replace(self.chunks[cid], score=score) for cid, score in ranking if cid in permitted]


class RerankerFixture:
    enabled = True
    embeddings_enabled = False
    reranker_enabled = True
    reranker_model = 'fixture-model'
    last_rerank_diagnostics = {}

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls = []
        self.last_rerank_diagnostics = {}

    def rerank(self, query, chunks, deadline):
        self.calls.append((query, list(chunks), deadline))
        self.last_rerank_diagnostics = {'api_attempted': True, 'api_attempts': 1,
                                       'api_http_success': self.error is None,
                                       'top_n': 1}
        if self.error:
            raise self.error
        result = self.result(chunks) if callable(self.result) else self.result
        if result is None:
            result = chunks[:1]
        if isinstance(result, list):
            self.last_rerank_diagnostics['returned'] = len(result)
        return result


class RerankPlannerV19Tests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {'ENABLE_EMBEDDINGS': '0', 'ENABLE_RERANKER': '1',
                                            'RERANK_CANDIDATES': '32', 'RERANK_TIMEOUT_SECONDS': '5'})
        self.env.start(); self.addCleanup(self.env.stop)
        blocker = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No external API'))
        blocker.start(); self.addCleanup(blocker.stop)

    def planner(self, client=None):
        corpus = CorpusFixture()
        for cid in ('a', 'b', 'c'):
            corpus.add(cid, 'Учебный документ ' + cid,
                       [(cid, '1.1', '1.1. Сторона передает предусмотренный результат другой стороне по подписанному акту ' + cid + '.')])
        corpus.lexical['*'] = [('a', 10.), ('b', 9.), ('c', 8.)]
        planner = EvidencePlanner(corpus, SimpleNamespace(profiles={}, amendments=[]), client or RerankerFixture())
        planner._parts = lambda question, ctx: ctx.get('fixture_parts', ['topic'])
        return planner

    def retrieve(self, planner, parts=None, ctx=None):
        parts = parts or ['topic']
        ranked, by_part, scores, _, _, state = planner._retrieve(parts, ctx or {}, time.monotonic() + 10)
        return ranked, by_part, scores, state

    def apply(self, planner, baseline=None, parts=None, deadline=None):
        ranked, by_part, scores, state = baseline or self.retrieve(planner, parts)
        result = planner._rerank_candidates(parts or ['topic'], ranked, by_part, scores,
                                             state, time.monotonic() + 10 if deadline is None else deadline)
        return result, state

    def test_success_changes_packed_source_and_keeps_canonical_original(self):
        client = RerankerFixture(lambda items: [next(c for c in items if c.id == 'b')])
        planner = self.planner(client)
        planner.SHORT_DOCUMENT_CHARS = 0
        planner.MAX_ANCHORS = 1
        result = planner.build('topic', {}, deadline=time.monotonic() + 10)
        self.assertEqual(result['sources'][0]['chunk_id'], 'b')
        self.assertIn('a', {s['chunk_id'] for s in result['sources']})
        self.assertEqual(result['trace']['rerank']['status'], 'applied')
        self.assertEqual(result['trace']['api_calls'], 1)
        self.assertEqual(len(client.calls), 1)
        self.assertLessEqual(client.calls[0][2], time.monotonic() + 5)
        for source in result['sources']:
            chunk = planner.safe[source['chunk_id']]
            self.assertEqual(source['text'], chunk.text)
            self.assertEqual(source['text'], planner.corpus.documents[chunk.doc_id].text[chunk.start:chunk.end])

    def test_success_updates_continuation_answer_anchor_not_only_trace(self):
        client = RerankerFixture(lambda items: [next(c for c in items if c.id == 'b')])
        planner = self.planner(client)
        planner.SHORT_DOCUMENT_CHARS = 0
        context = {'_clarification_continuation': {'original_task': 'previous topic', 'latest_user': 'topic'}}
        result = planner.build('topic', context, deadline=time.monotonic() + 10)
        by_id = {s['id']: s for s in result['sources']}
        first = result['context']['answer_anchor_ids'][0]
        self.assertEqual(by_id[first]['chunk_id'], 'b')
        self.assertTrue(result['context']['fallback_anchor_only'])
        self.assertIn('a', {s['chunk_id'] for s in result['sources']})

    def test_top_one_retains_independent_question_leaders_and_tail(self):
        client = RerankerFixture(lambda items: [next(c for c in items if c.id == 'b')])
        planner = self.planner(client)
        planner.corpus.lexical = {'first': [('a', 10.), ('b', 9.)], 'second': [('c', 10.)]}
        baseline = self.retrieve(planner, ['first', 'second'])
        (ranked, by_part, _), state = self.apply(planner, baseline, ['first', 'second'])
        self.assertEqual([c.id for c in ranked], ['b', 'a', 'c'])
        self.assertEqual([c.id for c in by_part[0]], ['b', 'a'])
        self.assertEqual([c.id for c in by_part[1]], ['c'])
        self.assertEqual(state['pre_rerank_part_leaders'], ['a', 'c'])
        result = planner.build('first', {'fixture_parts': ['first', 'second']}, deadline=time.monotonic() + 10)
        self.assertTrue({'a', 'b', 'c'} <= {s['chunk_id'] for s in result['sources']})

    def test_partial_response_keeps_unpromoted_part_order(self):
        planner = self.planner(RerankerFixture(lambda items: [next(c for c in items if c.id == 'c')]))
        baseline = self.retrieve(planner)
        ranked, _, scores, state = baseline
        by_part = [[replace(planner.safe['b'], score=9.), replace(planner.safe['a'], score=8.)],
                   [replace(planner.safe['c'], score=10.)]]
        (ranked, parts, _), _ = self.apply(planner, (ranked, by_part, scores, state))
        self.assertEqual([c.id for c in parts[0]], ['b', 'a'])
        self.assertEqual([c.id for c in ranked][0], 'c')

    def test_ordinal_scores_do_not_admit_unpromoted_weak_document(self):
        client = RerankerFixture(lambda items: [next(c for c in items if c.id == 'a')])
        planner = self.planner(client)
        planner.corpus.lexical['*'] = [('a', 10.), ('b', 8.), ('c', 3.5)]
        planner.SHORT_DOCUMENT_CHARS = 0
        client.reranker_enabled = False
        baseline = planner.build('topic', {}, deadline=time.monotonic() + 10)
        client.reranker_enabled = True
        reranked = planner.build('topic', {}, deadline=time.monotonic() + 10)
        self.assertEqual({s['chunk_id'] for s in baseline['sources']}, {'a', 'b'})
        self.assertEqual({s['chunk_id'] for s in reranked['sources']}, {'a', 'b'})

    def test_explicitly_promoted_weak_document_can_enter_context(self):
        client = RerankerFixture(lambda items: [next(c for c in items if c.id == 'c')])
        planner = self.planner(client)
        planner.corpus.lexical['*'] = [('a', 10.), ('b', 8.), ('c', 3.5)]
        planner.SHORT_DOCUMENT_CHARS = 0
        result = planner.build('topic', {}, deadline=time.monotonic() + 10)
        self.assertEqual(result['sources'][0]['chunk_id'], 'c')
        self.assertTrue({'a', 'b', 'c'} <= {s['chunk_id'] for s in result['sources']})

    def test_top_one_keeps_other_edition_and_unranked_exception(self):
        corpus = CorpusFixture()
        corpus.add('old', 'Порядок закупок ред 2038', [
            ('old-rule', '2.4', '2.4. Руководитель согласует закупку до подписания договора в течение пяти рабочих дней.'),
            ('exception', '2.5', '2.5. Для срочной закупки требуется дополнительное письменное согласование руководителя.')])
        corpus.add('new', 'Порядок закупок ред 2040', [
            ('new-rule', '2.4', '2.4. Директор согласует закупку до подписания договора в течение восьми рабочих дней.')])
        corpus.lexical['*'] = [('old-rule', 10.), ('new-rule', 9.)]
        profiles = {key: SimpleNamespace(contract_numbers=(), family='procurement-rules',
                    document_type='regulation', entities=(), conditions=(), issues=()) for key in ('old', 'new')}
        client = RerankerFixture(lambda items: [next(c for c in items if c.id == 'new-rule')])
        planner = EvidencePlanner(corpus, SimpleNamespace(profiles=profiles, amendments=[]), client)
        planner._parts = lambda *_: ['topic']
        result = planner.build('topic', {}, deadline=time.monotonic() + 10)
        self.assertEqual(result['sources'][0]['chunk_id'], 'new-rule')
        self.assertTrue({'old-rule', 'new-rule', 'exception'} <= {s['chunk_id'] for s in result['sources']})
        self.assertTrue(result['context']['evidence_complete'])

    def test_failure_preserves_exact_baseline_order_scores_and_part_lists(self):
        planner = self.planner(RerankerFixture(error=APIError('fixture timeout', code='timeout')))
        baseline = self.retrieve(planner)
        result, state = self.apply(planner, baseline)
        for got, original in zip(result, baseline[:3]):
            self.assertIs(got, original)
        self.assertEqual(state['rerank']['status'], 'failed')
        self.assertEqual(state['api_calls'], 1)
        self.assertTrue(state['warnings'])
        self.assertNotIn('pre_rerank_part_leaders', state)

    def test_foreign_duplicate_or_altered_sources_fail_without_publishing_them(self):
        def wrong(items):
            return [replace(items[0], id='outside-candidate-set')]
        def modified(items):
            return [replace(items[0], text='X' * len(items[0].text))]
        for response in (wrong, modified, lambda items: [items[0], items[0]], lambda items: []):
            with self.subTest(response=response):
                planner = self.planner(RerankerFixture(response))
                baseline = self.retrieve(planner)
                result, state = self.apply(planner, baseline)
                self.assertEqual(state['rerank']['status'], 'failed')
                for got, original in zip(result, baseline[:3]):
                    self.assertIs(got, original)

    def test_response_arriving_after_subdeadline_keeps_baseline(self):
        clock = [100.]
        def late(items):
            clock[0] = 106.
            return [items[-1]]
        planner = self.planner(RerankerFixture(late))
        baseline = self.retrieve(planner)
        with patch('evidence.time.monotonic', side_effect=lambda: clock[0]):
            result, state = self.apply(planner, baseline, deadline=110.)
        self.assertEqual(state['rerank']['status'], 'failed')
        for got, original in zip(result, baseline[:3]):
            self.assertIs(got, original)

    def test_disabled_empty_or_expired_never_calls_provider(self):
        for kind in ('disabled', 'unconfigured', 'empty', 'expired'):
            with self.subTest(kind=kind):
                client = RerankerFixture()
                planner = self.planner(client)
                baseline = self.retrieve(planner)
                deadline = time.monotonic() + 10
                if kind == 'disabled': client.reranker_enabled = False
                if kind == 'unconfigured': client.reranker_model = ''
                if kind == 'empty': baseline = ([], [], {}, baseline[3])
                if kind == 'expired': deadline = time.monotonic() - 1
                result, state = self.apply(planner, baseline, deadline=deadline)
                self.assertEqual(client.calls, [])
                self.assertFalse(state['rerank']['request_attempted'])
                for got, original in zip(result, baseline[:3]): self.assertIs(got, original)

    def test_candidate_cap_reserves_independent_low_rank_leaders(self):
        corpus = CorpusFixture()
        for index in range(40):
            cid = f'c{index:02d}'
            corpus.add(cid, 'Документ ' + cid,
                       [(cid, '4.1', '4.1. Сторона направляет установленный документ контрагенту после согласования условия.')])
        corpus.lexical['*'] = [(f'c{i:02d}', 50. - i) for i in range(40)]
        client = RerankerFixture()
        planner = EvidencePlanner(corpus, SimpleNamespace(profiles={}, amendments=[]), client)
        ranked = list(corpus.chunks.values())
        by_part = [[corpus.chunks['c00']], [corpus.chunks['c38']], [corpus.chunks['c39']]]
        state = {'warnings': [], 'api_calls': 0, 'lexical_leaders': [], 'dense_leaders': [], 'focused_lexical_leaders': []}
        with patch.dict('os.environ', {'RERANK_CANDIDATES': '8'}):
            result, state = self.apply(planner, (ranked, by_part, {c.id: 1. for c in ranked}, state))
        requested = {c.id for c in client.calls[0][1]}
        self.assertEqual(len(requested), 8)
        self.assertTrue({'c00', 'c38', 'c39'} <= requested)
        self.assertEqual(len(result[0]), 40)


if __name__ == '__main__':
    unittest.main()
