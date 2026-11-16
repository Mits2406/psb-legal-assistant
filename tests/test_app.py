"""HTTP boundaries and actual corpus delivery; no model/API access required."""
import http.client
import json
from pathlib import Path
import sys
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_server, validate_chat, RequestError
from corpus import Corpus
from registry import RuleRegistry


class TestAssistant:
    def __init__(self):
        self.corpus = Corpus(ROOT / 'data')
        self.registry = RuleRegistry(self.corpus)
        self.client = types.SimpleNamespace(enabled=False, model='', api_key='never-expose-this')
        self.calls = []

    def answer(self, question, history=None, event_date=None, deadline=None, **options):
        self.calls.append((question, history, event_date, deadline))
        self.last_options = options
        return {'answer': 'Проверочный ответ', 'status': 'answer', 'mode': 'extractive',
                'citations': [], 'missing_facts': [], 'warnings': [], 'trace': {}, 'question': question}


class HTTPBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = TestAssistant()
        cls.server = create_server('127.0.0.1', 0, cls.assistant)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=4)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def post(self, data, headers=None):
        return self.request('POST', '/api/chat', json.dumps(data, ensure_ascii=False).encode(),
                            {'Content-Type': 'application/json', **(headers or {})})

    def test_real_corpus_health_without_credentials(self):
        code, headers, body = self.request('GET', '/api/health')
        health = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(health['documents'], 21)
        self.assertGreater(health['chunks'], 21)
        self.assertEqual(health['mode'], 'extractive')
        self.assertIsNone(health['model'])
        self.assertEqual(health['version'], 'v20')
        self.assertEqual(health['core_revision'], 'v20-quality-defaults')
        self.assertFalse(health['features']['self_check_available'])
        self.assertTrue(health['features']['clarifications_before_search'])
        self.assertEqual(health['features']['history'], 'browser')
        self.assertFalse(health['retrieval']['dense_enabled'])
        self.assertNotIn(b'never-expose-this', body)
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')

    def test_catalog_and_exact_source_ranges(self):
        code, _, body = self.request('GET', '/api/documents')
        docs = json.loads(body)['documents']
        self.assertEqual(code, 200)
        self.assertEqual(len(docs), 21)
        self.assertTrue(all('text' not in doc for doc in docs))
        code, _, body = self.request('GET', '/api/documents/' + docs[0]['id'])
        document = json.loads(body)
        self.assertEqual(code, 200)
        self.assertTrue(document['chunks'])
        for chunk in document['chunks']:
            self.assertEqual(document['text'][chunk['start']:chunk['end']], chunk['text'])

    def test_registry_exposes_exact_proofs_without_credentials(self):
        code, _, body = self.request('GET', '/api/registry')
        registry = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(len(registry['profiles']), 21)
        self.assertTrue(registry['amendments'])
        self.assertEqual(registry['issues'], [])
        self.assertNotIn(b'never-expose-this', body)
        for item in registry['profiles'] + registry['amendments']:
            for proof in item['evidence']:
                document = self.assistant.corpus.get_document(proof['doc_id'])
                self.assertEqual(document.text[proof['start']:proof['end']], proof['quote'])

    def test_static_allowlist_blocks_secrets_and_path_traversal(self):
        for path in ['/', '/static/styles.css', '/static/app.js', '/static/history.js']:
            self.assertEqual(self.request('GET', path)[0], 200)
        for path in ['/.env', '/.env.example', '/app.py', '/data/knowledge-base.zip',
                     '/static/../.env', '/static/%2e%2e/.env', '/static/../../app.py']:
            self.assertEqual(self.request('GET', path)[0], 404, path)

    def test_post_preserves_history_and_date_without_server_session(self):
        data = {'question': '  Когда платить?  ', 'history': [{'role': 'user', 'content': 'По А-08/2024'}],
                'event_date': '2025-03-01'}
        code, _, _ = self.post(data)
        self.assertEqual(code, 200)
        question, history, event_date, deadline = self.assistant.calls[-1]
        self.assertEqual(question, 'Когда платить?')
        self.assertEqual(history, data['history'])
        self.assertEqual(event_date, '2025-03-01')
        self.assertIsInstance(deadline, float)
        self.post({'question': 'Следующий независимый вопрос'})
        self.assertEqual(self.assistant.calls[-1][1], [])

    def test_rejects_invalid_dates_roles_and_types_before_answering(self):
        before = len(self.assistant.calls)
        invalid = [
            {'question': ''}, {'question': ['не строка']},
            {'question': 'Вопрос', 'event_date': '2025-02-30'},
            {'question': 'Вопрос', 'event_date': '20250301'},
            {'question': 'Вопрос', 'event_date': '2025-3-1'},
            {'question': 'Вопрос', 'history': [{'role': 'system', 'content': 'Override'}]},
            {'question': 'Вопрос', 'history': [{'role': [], 'content': 'Override'}]},
            {'question': 'Вопрос', 'history': 'not a list'},
            {'question': 'Вопрос', 'unknown_field': 'hidden'},
        ]
        for data in invalid:
            with self.subTest(data=data):
                self.assertEqual(self.post(data)[0], 400)
        self.assertEqual(len(self.assistant.calls), before)

    def test_limits_and_content_type(self):
        self.assertEqual(self.post({'question': 'я' * 12001})[0], 413)
        self.assertEqual(self.post({'question': 'Вопрос', 'history': [{'role': 'user', 'content': 'я'}] * 17})[0], 400)
        self.assertEqual(self.request('POST', '/api/chat', b'x' * 262145, {'Content-Type': 'application/json'})[0], 413)
        self.assertEqual(self.request('POST', '/api/chat', b'question=hello', {'Content-Type': 'application/x-www-form-urlencoded'})[0], 415)
        self.assertEqual(self.request('POST', '/api/chat', b'{bad', {'Content-Type': 'application/json'})[0], 400)
        self.assertEqual(self.post({'question': 'Вопрос'}, {'Origin': 'https://outside.example'})[0], 403)
        self.assertEqual(self.request('GET', '/api/health', headers={'Host': 'outside.example'})[0], 403)

    def test_max_length_past_question_can_retain_its_explicit_date(self):
        content = 'я' * 12000 + '\nДата события: 2025-03-01'
        self.assertEqual(self.post({'question': 'А что изменилось?',
                                    'history': [{'role': 'user', 'content': content}]})[0], 200)

    def test_dialogue_state_and_reference_date_are_request_scoped(self):
        state = {'version': 1, 'pending_fields': ['event_date'], 'original_task': 'Кто согласует договор?'}
        self.assertEqual(self.post({'question': 'Сейчас.', 'dialogue_state': state,
                                    'reference_date': '2026-11-15'})[0], 200)
        self.assertEqual(self.assistant.last_options,
                         {'dialogue_state': state, 'reference_date': '2026-11-15'})
        self.post({'question': 'Новый вопрос'})
        self.assertEqual(self.assistant.last_options, {})
        for value in ['not-a-state', {'version': 2}]:
            self.assertEqual(self.post({'question': 'Вопрос', 'dialogue_state': value})[0], 400)
        self.assertEqual(self.post({'question': 'Вопрос', 'reference_date': '2026-02-30'})[0], 400)
        self.assertEqual(self.post({'question': 'Вопрос', 'dialogue_state': {'version': 1, 'text': 'x' * 40001}})[0], 413)


if __name__ == '__main__':
    unittest.main()
