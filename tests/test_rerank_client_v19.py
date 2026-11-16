"""Reranker API boundary tests; no external service or model is called."""
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import APIError, ModelClient


class Response:
    status = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, limit):
        return json.dumps(self.value).encode()[:limit]


class RerankClientV19Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://reranker-fixture.invalid/v1',
            'LLM_API_KEY': 'fixture-chat-key', 'LLM_MODEL': 'fixture-generator',
            'RERANKER_MODEL': 'provider/qwen-reranker-fixture',
            'RERANKER_API_KEY': 'fixture-rerank-key', 'API_RETRIES': '2',
        }, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.client = ModelClient()
        self.chunks = [SimpleNamespace(id=str(i), title=f'Документ {i}',
                                       text=f'{i}. Исходный текст.', score=.12 + i)
                       for i in range(3)]
        opener = patch('urllib.request.build_opener')
        self.build_opener = opener.start(); self.addCleanup(opener.stop)
        self.open = self.build_opener.return_value.open
        self.open.side_effect = AssertionError('No unconfigured network request')

    def response(self, value):
        self.open.side_effect = None
        self.open.return_value = Response(value)

    def test_http_contract_model_credentials_and_canonical_items(self):
        os.environ['RERANKER_TOP_N'] = '2'
        self.response({'results': [{'index': 2, 'relevance_score': .8},
                                   {'index': 0, 'relevance_score': -.2}]})
        result = self.client.rerank('Точный вопрос', self.chunks, time.monotonic() + 2)
        self.assertIs(result[0], self.chunks[2])
        self.assertIs(result[1], self.chunks[0])
        self.assertEqual(result[0].score, 2.12)
        request = self.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://reranker-fixture.invalid/v1/rerank')
        self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-rerank-key')
        self.assertEqual(json.loads(request.data), {
            'model': 'provider/qwen-reranker-fixture', 'query': 'Точный вопрос',
            'documents': [c.title + '\n' + c.text for c in self.chunks], 'top_n': 2})
        self.assertLessEqual(self.open.call_args.kwargs['timeout'], 2)
        diagnostics = self.client.last_rerank_diagnostics
        self.assertEqual((diagnostics['status'], diagnostics['api_attempts'],
                          diagnostics['input_count'], diagnostics['returned'],
                          diagnostics['top_n'], diagnostics['http_status']),
                         ('completed', 1, 3, 2, 2, 200))
        serialized = json.dumps(diagnostics, ensure_ascii=False)
        for private in ('Точный вопрос', 'Исходный текст', 'fixture-chat-key',
                        'fixture-rerank-key', 'reranker-fixture.invalid'):
            self.assertNotIn(private, serialized)

    def test_data_response_and_short_top_n_are_valid(self):
        self.response({'data': [{'index': 1}]})
        self.assertEqual(self.client.rerank('Запрос', self.chunks), [self.chunks[1]])
        self.assertEqual(self.client.last_rerank_diagnostics['returned'], 1)

    def test_top_n_is_bounded_and_invalid_config_uses_default(self):
        many = self.chunks * 30
        for configured, count, expected in [('999', 3, 3), ('0', 3, 1),
                                             ('-2', 3, 1), ('nan', 90, 8),
                                             ('invalid', 90, 8), ('999', 90, 64)]:
            with self.subTest(configured=configured, count=count):
                os.environ['RERANKER_TOP_N'] = configured
                self.response({'results': [{'index': 0}]})
                self.client.rerank('Запрос', many[:count])
                self.assertEqual(json.loads(self.open.call_args.args[0].data)['top_n'], expected)

    def test_malformed_responses_never_create_candidate_indices(self):
        cases = [None, [], {}, {'results': None}, {'results': {}}, {'results': []},
                 {'results': ['bad']}, {'results': [{}]},
                 *({'results': [{'index': value}]} for value in
                   (True, False, .5, 1.0, '1', 'garbage', -1, 3)),
                 {'results': [{'index': 0}, {'index': 0}]},
                 *({'results': [{'index': 0, 'relevance_score': value}]}
                   for value in (float('nan'), float('inf'), -float('inf'), True, '0.4', None)),
                 {'results': [{'index': 0, 'score': float('nan')}]},
                 {'results': None, 'data': [{'index': 0}]}]
        for value in cases:
            with self.subTest(value=value):
                self.response(value)
                with self.assertRaises(APIError) as error:
                    self.client.rerank('Запрос', self.chunks)
                self.assertIn(error.exception.code, ('response_shape', 'invalid_rerank_response'))
                self.assertEqual(self.client.last_rerank_diagnostics['status'], 'failed')
                self.assertEqual(self.client.last_rerank_diagnostics['returned'], 0)

    def test_more_results_than_requested_is_rejected(self):
        os.environ['RERANKER_TOP_N'] = '1'
        self.response({'results': [{'index': 1}, {'index': 0}]})
        with self.assertRaises(APIError) as error:
            self.client.rerank('Запрос', self.chunks)
        self.assertEqual(error.exception.code, 'invalid_rerank_response')

    def test_empty_input_returns_without_call(self):
        self.assertEqual(self.client.rerank('Запрос', []), [])
        self.assertFalse(self.build_opener.called)
        self.assertEqual(self.client.last_rerank_diagnostics['status'], 'skipped_empty')
        self.assertEqual(self.client.last_rerank_diagnostics['api_attempts'], 0)

    def test_bad_input_and_expired_deadline_do_not_call_api(self):
        cases = [('', self.chunks, None, 'invalid_rerank_input'),
                 ('Запрос', None, None, 'invalid_rerank_input'),
                 ('Запрос', [SimpleNamespace(title='Документ', text=42)], None, 'invalid_rerank_input'),
                 ('Запрос', self.chunks, time.monotonic() - 1, 'deadline')]
        for query, chunks, deadline, code in cases:
            with self.subTest(code=code, query=query):
                with self.assertRaises(APIError) as error:
                    self.client.rerank(query, chunks, deadline)
                self.assertEqual(error.exception.code, code)
                self.assertEqual(self.client.last_rerank_diagnostics['api_attempts'], 0)
        self.assertFalse(self.build_opener.called)

    def test_http_failure_has_one_attempt_despite_configured_retries(self):
        self.open.side_effect = HTTPError('http://private.invalid', 503, 'private-error-text', {}, BytesIO())
        with self.assertRaises(APIError) as error:
            self.client.rerank('Запрос', self.chunks)
        self.assertEqual(error.exception.code, 'http_503')
        self.assertEqual(self.open.call_count, 1)
        diagnostics = self.client.last_rerank_diagnostics
        self.assertEqual((diagnostics['api_attempts'], diagnostics['http_status']), (1, 503))
        self.assertNotIn('private', json.dumps(diagnostics))

    def test_independent_reranker_works_without_generator_config(self):
        os.environ.pop('LLM_MODEL'); os.environ.pop('LLM_API_KEY')
        client = ModelClient()
        self.assertFalse(client.enabled)
        self.assertTrue(client.reranker_enabled)
        self.response({'results': [{'index': 0}]})
        self.assertEqual(client.rerank('Запрос', self.chunks), [self.chunks[0]])
        self.assertEqual(self.open.call_args.args[0].get_header('Authorization'), 'Bearer fixture-rerank-key')
        with self.assertRaises(APIError):
            client.chat_text('system', {})

    def test_key_falls_back_to_generator_key_and_disable_is_respected(self):
        os.environ.pop('RERANKER_API_KEY')
        client = ModelClient()
        self.response({'results': [{'index': 0}]})
        client.rerank('Запрос', self.chunks)
        self.assertEqual(self.open.call_args.args[0].get_header('Authorization'), 'Bearer fixture-chat-key')
        for value in ('0', 'false', 'off'):
            with self.subTest(value=value):
                os.environ['ENABLE_RERANKER'] = value
                disabled = ModelClient()
                self.assertFalse(disabled.reranker_enabled)
                with self.assertRaises(APIError) as error:
                    disabled.rerank('Запрос', self.chunks)
                self.assertEqual(error.exception.code, 'not_configured')
                self.assertEqual(disabled.last_rerank_diagnostics['api_attempts'], 0)

    def test_diagnostics_are_thread_local_and_return_copies(self):
        self.response({'results': [{'index': 0}]})
        self.client.rerank('Запрос', self.chunks)
        snapshot = self.client.last_rerank_diagnostics
        snapshot['status'] = 'modified-outside'
        worker_results = []
        worker = threading.Thread(target=lambda: (self.client.rerank('Запрос', []),
                    worker_results.append(self.client.last_rerank_diagnostics)))
        worker.start(); worker.join()
        self.assertEqual(worker_results[0]['status'], 'skipped_empty')
        self.assertEqual(self.client.last_rerank_diagnostics['status'], 'completed')
        self.assertEqual(self.client.last_rerank_diagnostics['returned'], 1)


if __name__ == '__main__':
    unittest.main()
