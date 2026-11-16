"""Active self-check and evidence-based workflows; controlled model, no network."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator, _wants_steps
from llm import APIError


SOURCES = [
    {'id': 'S001', 'title': 'Учебный договор', 'clause': '4.1',
     'text': '4.1. Арендная плата составляет 123 000 рублей в месяц.'},
    {'id': 'S002', 'title': 'Учебный договор', 'clause': '4.2',
     'text': '4.2. Арендная плата вносится не позднее 8-го числа текущего месяца.'},
]
DRAFT = 'Арендная плата составляет 123 000 рублей в месяц. [S001]'


class Client:
    enabled = True

    def __init__(self, responses, finishes=(), after_call=None):
        self.responses = list(responses)
        self.finishes = list(finishes)
        self.calls = []
        self.after_call = after_call
        self.last_chat_diagnostics = {}

    def chat_text(self, system, payload, deadline):
        self.calls.append((system, payload, deadline))
        self.last_chat_diagnostics = {
            'finish_reason': self.finishes.pop(0) if self.finishes else 'stop',
            'api_attempted': True, 'api_attempts': 1, 'api_http_success': True,
        }
        result = self.responses.pop(0)
        if self.after_call:
            self.after_call()
        if isinstance(result, Exception):
            raise result
        return result


class ReviewV11Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No real API'))
        network.start()
        self.addCleanup(network.stop)

    def generate(self, responses, *, question='Какая арендная плата?', sources=None, finishes=()):
        self.client = Client(responses, finishes)
        return GroundedGenerator(self.client).generate(question, SOURCES if sources is None else sources,
            deadline=time.monotonic() + 30)

    def test_default_review_corrects_wrong_amount_and_restores_missing_condition(self):
        wrong = 'Арендная плата составляет 999 рублей. [S001]'
        corrected = DRAFT + '\n\nВносится не позднее 8-го числа текущего месяца. [S002]'
        result = self.generate([wrong, corrected], question='Сколько и когда платить аренду?')
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(self.client.calls[1][1]['draft'], wrong)
        self.assertEqual(self.client.calls[0][1]['sources'], self.client.calls[1][1]['sources'])
        self.assertNotIn('999', result['answer'])
        self.assertIn('8-го числа', result['answer'])
        self.assertEqual(result['self_check'], {'enabled': True, 'status': 'completed', 'completed': True, 'changed': True})
        self.assertFalse(result['trace']['structural_only'])
        self.assertEqual([s['stage'] for s in result['trace']['stages']], ['draft', 'review'])
        self.assertEqual(result['used_ids'], ['S001', 'S002'])

    def test_unchanged_review_is_reported_as_completed_not_as_correctness_guarantee(self):
        result = self.generate([DRAFT, DRAFT])
        self.assertTrue(result['self_check']['completed'])
        self.assertFalse(result['self_check']['changed'])
        self.assertNotIn('безошибоч', result['answer'])
        self.assertNotIn('проверен', result['answer'])

    def test_review_api_failure_preserves_draft_and_exposes_unchecked_status(self):
        result = self.generate([DRAFT, APIError('secret-key-never-expose', code='timeout')])
        self.assertIn(DRAFT, result['answer'])
        self.assertEqual(result['mode'], 'llm')
        self.assertEqual(result['self_check']['status'], 'unavailable')
        self.assertFalse(result['self_check']['completed'])
        self.assertTrue(result['trace']['structural_only'])
        self.assertIn('Самопроверка не выполнена', result['warnings'][0])
        self.assertNotIn('secret-key', str(result))
        self.assertNotIn('DIAG_', result['answer'])

    def test_damaged_reviews_cannot_erase_intact_draft(self):
        for review, finish in [
            ('Другая сумма. [S999]', 'stop'),
            ('Плата составляет 123 000 рублей. [S001]\n\nНо только если', 'length'),
            ('Недостаточно данных: поскольку платить не нужно.', 'stop'),
            ('Уточните: договор действует бессрочно.', 'stop'),
            ('{}', 'stop'),
        ]:
            with self.subTest(review=review):
                result = self.generate([DRAFT, review], finishes=['stop', finish])
                self.assertIn(DRAFT, result['answer'])
                self.assertEqual(result['self_check']['status'], 'invalid')
                self.assertFalse(result['self_check']['completed'])
                self.assertTrue(result['warnings'])
                self.assertNotIn('S999', result['answer'])

    def test_review_deadline_reports_skipped_without_second_call(self):
        clock = [100.0]
        client = Client([DRAFT], after_call=lambda: clock.__setitem__(0, 999.5))
        with patch('generation.time.monotonic', side_effect=lambda: clock[0]):
            result = GroundedGenerator(client).generate('Какая арендная плата?', SOURCES, deadline=1000.0)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][2], 640.0)
        self.assertEqual(result['self_check']['status'], 'skipped_deadline')
        self.assertIn(DRAFT, result['answer'])
        self.assertTrue(result['warnings'])

    def test_explicit_opt_out_is_visible(self):
        for value in ('0', 'false', 'off', ' OFF '):
            with self.subTest(value=value), patch.dict(os.environ, {'ANSWER_VERIFY': value}):
                result = self.generate([DRAFT])
            self.assertEqual(len(self.client.calls), 1)
            self.assertEqual(result['self_check']['status'], 'disabled')
            self.assertFalse(result['self_check']['enabled'])
            self.assertFalse(result['self_check']['completed'])

    def test_no_sources_never_calls_model_or_claims_check(self):
        result = self.generate([], sources=[])
        self.assertEqual(self.client.calls, [])
        self.assertEqual(result['self_check']['status'], 'skipped_no_sources')
        self.assertFalse(result['self_check']['completed'])

    @patch.dict(os.environ, {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_no_model_keeps_opt_in_diagnostic_contract_and_exposes_missing_check(self):
        client = Client([])
        client.enabled = False
        result = GroundedGenerator(client).generate('Какая арендная плата?', SOURCES, deadline=time.monotonic()+10)
        self.assertEqual(client.calls, [])
        self.assertIn('DIAG_API_FAILURE:not_configured', result['answer'])
        self.assertEqual(result['self_check']['status'], 'skipped_no_model')

    @patch.dict(os.environ, {'DIAGNOSTIC_API_FAILURES': '1'})
    def test_draft_failure_stays_diagnostic_and_does_not_call_review(self):
        result = self.generate([APIError('unavailable', code='timeout')])
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn('DIAG_API_FAILURE:timeout', result['answer'])
        self.assertEqual(result['self_check']['status'], 'skipped_draft_failure')

    def test_default_draft_failure_recovers_and_preserves_review_and_error_trace(self):
        self.assertNotIn('DIAGNOSTIC_API_FAILURES', os.environ)
        result = self.generate([APIError('unavailable', code='timeout')])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(result['mode'], 'extractive')
        self.assertNotIn('DIAG_', result['answer'])
        self.assertIn(SOURCES[0]['text'], result['answer'])
        self.assertEqual(result['self_check']['status'], 'skipped_draft_failure')
        self.assertFalse(result['self_check']['completed'])
        self.assertEqual(result['trace']['fallback_reason'], 'draft_unavailable')
        self.assertEqual(result['trace']['stages'][0]['error_code'], 'timeout')
        self.assertTrue(result['trace']['api_attempted'])
        self.assertEqual(result['trace']['api_attempts'], 1)

    def test_workflow_is_reviewed_as_cited_numbered_actions(self):
        sources = [
            {'id': 'S001', 'title': 'Учебный регламент', 'clause': '2.1',
             'text': '2.1. Инициатор готовит проект договора и направляет его в юридический отдел.'},
            {'id': 'S002', 'title': 'Учебный регламент', 'clause': '2.2',
             'text': '2.2. После согласования юридическим отделом проект направляется руководителю подразделения. '
                     'При сумме свыше 600 000 рублей также требуется согласование финансового директора.'},
        ]
        steps = ('1. Инициатор готовит проект договора и направляет его в юридический отдел. [S001]\n\n'
                 '2. После юридического согласования направьте проект руководителю подразделения; '
                 'при сумме свыше 600 000 рублей получите также согласование финансового директора. [S002]')
        result = self.generate([steps, steps], question='Как согласовать договор? Опиши шаги.', sources=sources)
        for system, payload, _ in self.client.calls:
            self.assertEqual(payload['response_format'], 'steps')
            self.assertIn('Каждый шаг, роль, срок, документ и переход', system)
            self.assertIn('Не добавляй типовой маршрут', system)
        self.assertEqual(result['trace']['answer_format'], 'steps')
        self.assertTrue(result['trace']['action_steps_requested'])
        self.assertIn('1.', result['answer'])
        self.assertIn('2.', result['answer'])
        self.assertEqual(result['used_ids'], ['S001', 'S002'])
        self.assertTrue(result['self_check']['completed'])

    def test_unsubstantiated_action_step_does_not_pass_as_a_cosmetic_list(self):
        draft = '1. Внесите арендную плату до 8-го числа. [S002]\n\n2. Затем обратитесь к нотариусу.'
        result = self.generate([draft, APIError('controlled review outage', code='timeout')],
                               question='Как оплатить аренду пошагово?')
        self.assertNotIn('нотариусу', result['answer'])
        self.assertEqual(result['mode'], 'extractive')
        self.assertEqual(result['self_check']['status'], 'recovery_after_review')
        self.assertFalse(result['self_check']['completed'])
        self.assertEqual(len(self.client.calls), 2)

    def test_short_factual_question_does_not_trigger_workflow(self):
        for question in ('Какая арендная плата?', 'Когда вносить плату?', 'Нужно ли согласование директора?',
                         'Какие документы есть в базе?'):
            with self.subTest(question=question):
                self.assertFalse(_wants_steps(question))
        result = self.generate([DRAFT, DRAFT])
        self.assertEqual(self.client.calls[0][1]['response_format'], 'prose')
        self.assertFalse(result['trace']['action_steps_requested'])

    def test_generic_workflow_intents_do_not_depend_on_case_names(self):
        for question in ('Как мне оформить претензию?', 'Что надо сделать для расторжения договора?',
                         'Каков порядок передачи документов?', 'Опиши процедуру согласования.',
                         'Пошаговый план возврата оборудования', 'Какая последовательность действий?',
                         'Дай план согласования договора', 'Как подготовить документы?'):
            with self.subTest(question=question):
                self.assertTrue(_wants_steps(question))


if __name__ == '__main__':
    unittest.main()
