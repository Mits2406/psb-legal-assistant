"""Whole-answer normalization faults, budget and independent batch rows.

All responses here are controlled fixtures, not real organizer model results.
"""
import copy
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from answer import LegalAssistant
from llm import APIError
from solution import run_batch


QUESTION = 'Когда платить по договору А-08/2024?'


class ControlledClient:
    enabled = True

    def __init__(self, normal_fault=None, draft_fault=None):
        self.normal_fault, self.draft_fault = normal_fault, draft_fault
        self._local = threading.local()
        self.records = []
        self.lock = threading.Lock()

    @property
    def last_chat_diagnostics(self):
        return getattr(self._local, 'diagnostics', {})

    def record(self, stage, data, deadline, max_tokens=None):
        self._local.diagnostics = {'api_attempted': True, 'api_attempts': 1,
            'api_http_success': True, 'model_text_received': True,
            'model_text_usable': True, 'finish_reason': 'stop'}
        with self.lock:
            self.records.append({'stage': stage, 'data': copy.deepcopy(data),
                                 'deadline': deadline, 'max_tokens': max_tokens})

    def chat_json(self, system, data, deadline=None, *, max_tokens=None):
        self.record('normalization', data, deadline, max_tokens)
        if isinstance(self.normal_fault, Exception):
            self._local.diagnostics.update(api_http_success=False, model_text_received=False,
                                          model_text_usable=False, error_code=self.normal_fault.code)
            raise self.normal_fault
        user = [m for m in data['messages'] if m['role'] == 'user'][-1]
        result = {'task_type': 'application', 'topic': 'payment', 'question': user['content'],
            'task_message_ids': [user['id']], 'topic_changed': False,
            'subquestions': [{'text': user['content'], 'evidence': [{'message_id': user['id'], 'quote': user['content']}]}],
            'facts': [], 'unknown_facts': [], 'corrections': []}
        if self.normal_fault == 'shape':
            return {'question': QUESTION}
        if self.normal_fault == 'invented':
            result['question'] += ' По договору А-999/2099.'
        if self.normal_fault == 'assistant_provenance':
            result['facts'] = [{'kind': 'amount', 'value': '999 рублей',
                'evidence': [{'message_id': 'm0', 'quote': '999 рублей'}]}]
        return result

    def chat_text(self, system, data, deadline=None, *, max_tokens=None):
        stage = 'review' if 'draft' in data else 'draft'
        self.record(stage, data, deadline, max_tokens)
        if stage == 'draft' and self.draft_fault:
            self._local.diagnostics.update(api_http_success=False, model_text_received=False,
                                          model_text_usable=False, error_code=self.draft_fault.code)
            raise self.draft_fault
        if stage == 'review':
            return data['draft']
        source = next(s for s in data['sources'] if '5-го числа' in s['text'])
        return source['text'] + ' [' + source['id'] + ']'


class PipelineV12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = LegalAssistant(ROOT / 'data')

    def setUp(self):
        env = patch.dict(os.environ, {'QUERY_NORMALIZE': '1', 'ANSWER_VERIFY': '1',
                                     'DIAGNOSTIC_API_FAILURES': '1'}, clear=True)
        env.start(); self.addCleanup(env.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No real network in tests'))
        network.start(); self.addCleanup(network.stop)

    def run_answer(self, client, *, history=None, seconds=90):
        self.assistant.client = client
        return self.assistant.answer(QUESTION, history=history, reference_date='2026-11-15',
                                     deadline=time.monotonic() + seconds)

    def assert_good_answer(self, result):
        self.assertEqual(result['mode'], 'llm')
        self.assertIn('5-го числа текущего месяца', result['answer'])
        self.assertNotIn('DIAG_', result['answer'])
        self.assertTrue(result['self_check']['completed'])
        self.assertTrue(result['citations'])

    def test_bad_normalization_does_not_turn_successful_answer_into_api_failure(self):
        for failure in ['shape', 'invented', APIError('private proxy detail', code='timeout')]:
            with self.subTest(failure=type(failure).__name__):
                client = ControlledClient(normal_fault=failure)
                result = self.run_answer(client)
                self.assert_good_answer(result)
                self.assertTrue(result['trace']['normalization']['status'].startswith('fallback_'))
                self.assertEqual([x['stage'] for x in client.records], ['normalization', 'draft', 'review'])
                self.assertEqual(result['trace']['generation']['stage'], 'review')
                self.assertEqual(result['trace']['model_calls'], 3)
                self.assertNotIn('private proxy detail', str(result))
                self.assertNotIn('А-999/2099', client.records[1]['data']['question'])

    def test_assistant_assertion_cannot_supply_normalized_fact(self):
        client = ControlledClient(normal_fault='assistant_provenance')
        result = self.run_answer(client, history=[{'role': 'assistant', 'content': '999 рублей'}])
        self.assert_good_answer(result)
        self.assertEqual(result['trace']['normalization']['status'], 'fallback_invalid')
        self.assertNotIn('999 рублей', result['answer'])

    def test_valid_normalization_and_three_calls_stay_within_one_deadline(self):
        client = ControlledClient()
        started = time.monotonic()
        result = self.run_answer(client)
        self.assert_good_answer(result)
        self.assertEqual(result['trace']['normalization']['status'], 'accepted')
        self.assertLessEqual(client.records[0]['deadline'] - started, 12.1)
        self.assertLessEqual(client.records[0]['max_tokens'], 1800)
        self.assertTrue(all(x['deadline'] <= started + 90.1 for x in client.records))
        self.assertLess(client.records[1]['deadline'], client.records[2]['deadline'])
        self.assertEqual(client.records[1]['data']['context']['reference_date'], '2026-11-15')

    def test_small_budget_skips_normalization_but_can_answer(self):
        client = ControlledClient()
        result = self.run_answer(client, seconds=5)
        self.assert_good_answer(result)
        self.assertEqual(result['trace']['normalization']['status'], 'skipped_budget')
        self.assertEqual([x['stage'] for x in client.records], ['draft', 'review'])

    def test_draft_failure_keeps_requested_diagnostic_and_correct_stage(self):
        client = ControlledClient(draft_fault=APIError('secret transport detail', code='timeout'))
        result = self.run_answer(client)
        self.assertEqual(result['trace']['normalization']['status'], 'accepted')
        self.assertTrue(result['answer'].startswith('DIAG_API_FAILURE:timeout; stage=draft.'))
        self.assertNotIn('secret transport detail', str(result))

    def test_parallel_csv_rows_never_inherit_other_rows_state(self):
        client = ControlledClient(); self.assistant.client = client
        rows = [('alpha', QUESTION), ('beta', 'До какого числа платить по А-08/2024?')]
        with tempfile.TemporaryDirectory() as temp:
            summary = run_batch(rows, self.assistant, Path(temp) / 'answers.csv', max_workers=2,
                                deadline=time.monotonic() + 90)
        self.assertEqual(summary['technical_errors'], 0)
        requests = [x['data'] for x in client.records if x['stage'] == 'normalization']
        self.assertEqual(len(requests), 2)
        self.assertEqual({x['messages'][0]['content'] for x in requests}, {q for _, q in rows})
        self.assertTrue(all(len(x['messages']) == 1 and x['pending_state'] is None for x in requests))


if __name__ == '__main__':
    unittest.main()
