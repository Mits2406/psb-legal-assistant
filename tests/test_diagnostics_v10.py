"""Diagnostic output and real HTTP-client code with a controlled transport."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator
from llm import ModelClient, diagnostic_enabled
from solution import run_batch

SOURCES = [{'id': 'S001', 'doc_id': 'rent', 'title': 'Договор аренды', 'clause': '3.1',
            'text': '3.1. Арендная плата составляет 450 000 рублей в месяц, включая НДС.'}]
ENV = {'LLM_BASE_URL': 'http://127.0.0.1:1/v1', 'LLM_API_KEY': 'test-secret-do-not-leak',
       'LLM_MODEL': 'controlled-test-model', 'API_RETRIES': '1', 'ANSWER_VERIFY': '0'}


class Response:
    status = 200
    def __init__(self, raw): self.raw = raw
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit): return self.raw


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', ENV, clear=True)
        env.start(); self.addCleanup(env.stop)
        sleeper = patch('llm.time.sleep')
        sleeper.start(); self.addCleanup(sleeper.stop)
        self.attempts = []

    def opener(self, mode):
        outer = self
        class Opener:
            def open(self, request, timeout):
                outer.attempts.append(mode)
                if mode == 'transport':
                    raise urllib.error.URLError('test-secret-do-not-leak')
                if mode == 'timeout':
                    raise urllib.error.URLError(TimeoutError('test-secret-do-not-leak'))
                if mode == 'http':
                    raise urllib.error.HTTPError(request.full_url, 503, 'test-secret-do-not-leak', {}, None)
                if mode == 'malformed_json':
                    return Response(b'not JSON test-secret-do-not-leak')
                if mode == 'shape':
                    return Response(b'{}')
                content = 'Арендная плата составляет 450 000 рублей в месяц, включая НДС. [S001]'
                if mode == 'empty': content = ''
                if mode == 'parser': content = 'Этот вывод не имеет ссылки на предоставленный источник.'
                if mode == 'missing': content = 'Недостаточно данных: пришлите текст договора К-77/2037 и его условия о гарантии.'
                finish = 'tool_calls' if mode == 'nontext' else 'stop'
                return Response(json.dumps({'choices': [{'message': {'content': content},
                                                         'finish_reason': finish}]}).encode())
        return Opener()

    def generate(self, mode, *, sources=None, deadline=None):
        client = ModelClient()
        with patch('urllib.request.build_opener', return_value=self.opener(mode)):
            result = GroundedGenerator(client).generate('Какая арендная плата?',
                SOURCES if sources is None else sources, deadline=deadline or time.monotonic()+10)
        self.assertNotIn('test-secret-do-not-leak', json.dumps(result, ensure_ascii=False))
        return result

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_opt_in_diagnostic_transport_http_timeout_and_bounded_retries(self):
        for mode, reason in [('transport', 'transport_error'), ('timeout', 'timeout'), ('http', 'http_503')]:
            with self.subTest(mode=mode):
                self.attempts.clear()
                result = self.generate(mode)
                self.assertTrue(result['answer'].startswith('DIAG_API_FAILURE:' + reason))
                self.assertIn('фиолетовый трактор', result['answer'])
                self.assertEqual(len(self.attempts), 2)
                self.assertEqual(result['trace']['api_attempts'], 2)
                self.assertTrue(result['trace']['api_attempted'])
                self.assertFalse(result['trace']['api_http_success'])
                self.assertFalse(result['trace']['model_text_received'])

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_successful_http_with_unusable_output_is_not_transport_failure(self):
        for mode, reason in [('empty', 'empty_content'), ('shape', 'response_shape'),
                             ('malformed_json', 'invalid_response_json'), ('nontext', 'nontext_output')]:
            with self.subTest(mode=mode):
                self.attempts.clear()
                result = self.generate(mode)
                self.assertTrue(result['answer'].startswith('DIAG_MODEL_OUTPUT:' + reason))
                self.assertTrue(result['trace']['api_http_success'])
                self.assertEqual(result['trace']['model_text_received'], mode == 'nontext')
                self.assertFalse(result['trace']['model_text_usable'])
                self.assertEqual(len(self.attempts), 1)
                self.assertEqual(result['trace']['stages'][0]['stage'], 'draft')

    def test_missing_configuration_is_explicit_without_call(self):
        with patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'}, clear=True):
            result = self.generate('success')
        self.assertIn('DIAG_API_FAILURE:not_configured; stage=configuration', result['answer'])
        self.assertFalse(result['trace']['api_attempted'])
        self.assertEqual(self.attempts, [])

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_budget_expiry_is_distinct_and_makes_no_request(self):
        result = self.generate('success', deadline=time.monotonic()-1)
        self.assertIn('DIAG_API_NOT_CALLED:deadline_before_call', result['answer'])
        self.assertFalse(result['trace']['api_attempted'])
        self.assertEqual(self.attempts, [])

    def test_actual_absence_of_sources_is_not_api_failure_even_without_configuration(self):
        with patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'}, clear=True):
            result = self.generate('success', sources=[])
        self.assertNotIn('DIAG_', result['answer'])
        self.assertEqual(result['mode'], 'abstain')
        self.assertEqual(result['trace']['not_called_reason'], 'no_sources')
        self.assertEqual(self.attempts, [])

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_good_answer_and_concrete_missing_document_are_not_diagnostic(self):
        for mode in ('success', 'missing'):
            with self.subTest(mode=mode):
                result = self.generate(mode)
                self.assertNotIn('DIAG_', result['answer'])
                self.assertTrue(result['trace']['api_http_success'])
                self.assertTrue(result['trace']['model_text_received'])
                self.assertEqual(result['trace']['stages'][0]['finish_reason'], 'stop')
                if mode == 'missing': self.assertIn('К-77/2037', result['answer'])

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_parser_rejection_keeps_api_success_separate(self):
        result = self.generate('parser')
        self.assertTrue(result['trace']['api_http_success'])
        self.assertTrue(result['trace']['model_text_received'])
        self.assertNotIn('DIAG_API_FAILURE', result['answer'])
        self.assertEqual(result['mode'], 'extractive')
        self.assertIn('450 000', result['answer'])
        self.assertEqual(result['trace']['output_status'], 'rejected_draft')

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_per_row_diagnostic_preserves_other_rows_and_trace(self):
        outer = self
        class Assistant:
            def answer(self, question, deadline):
                mode = question
                result = outer.generate(mode, sources=[] if mode == 'no_sources' else None, deadline=deadline)
                return {**result, 'trace': {'generation': result['trace']}}
        # Serial control avoids overlapping patch contexts; production concurrency
        # is exercised separately by the isolated ZIP runner.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            run_batch([('001', 'transport'), ('02-good', 'success'), ('absent', 'no_sources')],
                      Assistant(), path/'submission.csv', trace_path=path/'trace.jsonl', max_workers=1)
            with (path/'submission.csv').open() as handle: rows = list(csv.DictReader(handle))
            self.assertEqual([r['id'] for r in rows], ['001', '02-good', 'absent'])
            self.assertIn('DIAG_API_FAILURE:transport_error', rows[0]['answer'])
            self.assertIn('450 000', rows[1]['answer'])
            self.assertNotIn('DIAG_', rows[1]['answer'] + rows[2]['answer'])
            records = [json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 3)
            by_id = {record['id']: record['trace']['generation'] for record in records}
            self.assertTrue(by_id['001']['api_attempted'])
            self.assertTrue(by_id['02-good']['model_text_received'])
            self.assertFalse(by_id['absent']['api_attempted'])

    def test_can_explicitly_disable_diagnostic_for_ordinary_recovery(self):
        with patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '0'}):
            result = self.generate('transport')
        self.assertNotIn('DIAG_', result['answer'])
        self.assertIn('450 000', result['answer'])

    def test_default_off_recovers_sources_and_preserves_failure_trace(self):
        self.assertFalse(diagnostic_enabled())
        for mode, reason, http_success in (
            ('transport', 'transport_error', False),
            ('timeout', 'timeout', False),
            ('empty', 'empty_content', True),
        ):
            with self.subTest(mode=mode):
                self.attempts.clear()
                result = self.generate(mode)
                self.assertEqual(result['mode'], 'extractive')
                self.assertNotIn('DIAG_', result['answer'])
                self.assertIn(SOURCES[0]['text'], result['answer'])
                trace = result['trace']
                self.assertEqual(trace['fallback_reason'], 'draft_unavailable')
                self.assertTrue(trace['api_attempted'])
                self.assertEqual(trace['api_attempts'], len(self.attempts))
                self.assertEqual(trace['api_http_success'], http_success)
                self.assertFalse(trace['model_text_received'])
                self.assertEqual(trace['stages'][0]['stage'], 'draft')
                self.assertEqual(trace['stages'][0]['error_code'], reason)

    @patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_preexpired_batch_does_not_dispatch_and_has_exact_not_called_code(self):
        class NeverCalled:
            def answer(self, *args, **kwargs):
                raise AssertionError('No worker should run after an already expired budget')
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            summary = run_batch([('001', 'q'), ('002', 'q')], NeverCalled(), p/'out.csv',
                                deadline=time.monotonic()-1, trace_path=p/'trace.jsonl')
            with (p/'out.csv').open() as handle: rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all('DIAG_API_NOT_CALLED:batch_deadline' in r['answer'] for r in rows))
            self.assertEqual(summary['timed_out'], 2)
            records = [json.loads(line) for line in (p/'trace.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['trace']['generation']['api_attempted'] is False for r in records))


if __name__ == '__main__':
    unittest.main()
