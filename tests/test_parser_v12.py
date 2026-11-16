"""Whole-answer repair and independent speech-act controls; no network calls."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation import GroundedGenerator, _nonclaim, _cosmetic
from llm import APIError


RULE = '4.3. Плату можно увеличить только по соглашению сторон, не более чем на 6% в год.'
SOURCES = [{'id': 'S001', 'title': 'Учебный договор', 'clause': '4.3', 'text': RULE}]
SAFE_DRAFT = 'Увеличение платы требует соглашения сторон и ограничено 6% в год. [S001]'


class ControlledClient:
    enabled = True

    def __init__(self, answers, finishes=()):
        self.answers = list(answers)
        self.finishes = list(finishes)
        self.calls = []
        self.last_chat_diagnostics = {}

    def chat_text(self, system, payload, deadline):
        self.calls.append({'system': system, 'data': payload, 'deadline': deadline})
        self.last_chat_diagnostics = {'finish_reason': self.finishes.pop(0) if self.finishes else 'stop',
                                      'api_attempted': False, 'api_attempts': 0,
                                      'transport': 'controlled_local_response_not_API'}
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class ParserV12Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'ANSWER_VERIFY': '1'}, clear=True)
        env.start(); self.addCleanup(env.stop)
        blocked = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No real API'))
        blocked.start(); self.addCleanup(blocked.stop)

    def run_generator(self, responses, *, question='Как изменяется плата?', sources=None, finishes=(), deadline=None):
        self.client = ControlledClient(responses, finishes)
        return GroundedGenerator(self.client).generate(question, sources or SOURCES,
            deadline=time.monotonic() + 30 if deadline is None else deadline)

    def test_independent_requests_are_whole_speech_acts_not_legal_claims(self):
        requests = [
            'Уточните: категорию договора, способ направления акта и сведения об участниках сделки.',
            'Уточните: сможете вы прислать копию договора и дату получения заявления?',
            'Уточните: готовы ли вы предоставить полный текст приложения или подтвердить факт его подписания?',
            'Уточните: предмет и сумму сделки. Для проверки нужен документ об особенностях согласования.',
            'Уточните: сумму, форму и особые условия договора для определения дополнительных согласований, а также дату получения запроса комплаенсом.',
            'Уточните: фактическую дату получения акта, и полное наименование получателя.',
            'Уточните: существенные условия заявки, а также исходный текст приложения.',
            'Уточните: если нужна конечная дата возврата, когда получено уведомление?',
            'Уточните: подтверждённый банком момент получения перевода.',
            'Уточните: дату получения акта, если нужны конкретные даты расчёта.',
            'Уточните: укажите дату заявления и предоставьте внутренний маршрут рассмотрения.',
            'Недостаточно данных: по представленным источникам нельзя достоверно рассчитать сумму. Нужен текст применимых правил расчёта.',
            'Уточните: когда получено уведомление? Без этого нельзя назвать конкретную дату.',
            'Уточните: есть ли документ, описывающий очередность или параллельность рассмотрения заявок?',
            'Уточните: наличие поручительства или залога, аффилированность, сведения о предыдущих поставках и уведомлениях.',
            'Уточните: оформлены ли прекращение договора и акт возврата, есть ли отдельное соглашение об обеспечительном платеже?',
        ]
        generator = GroundedGenerator(None)
        for request in requests:
            with self.subTest(request=request):
                self.assertTrue(_nonclaim(request))
                paragraphs, rejected = generator._parse(SAFE_DRAFT + '\n\n' + request,
                                                        {'S001': SOURCES[0]}, normalize_caution=False)
                self.assertEqual(rejected, 0)
                self.assertEqual(paragraphs[-1], request)

    def test_a_request_prefix_does_not_license_norms_actions_or_hidden_tails(self):
        requests = [
            'Уточните: сумму договора — 15 млн рублей, согласование не требуется.',
            'Уточните: можете предоставить договор и не платить задолженность?',
            'Уточните: пришлите документ и согласуйте его с нотариусом.',
            'Уточните: предмет договора, остальные согласования уже выполнены.',
            'Уточните: фактическую дату получения акта, и оплатите задолженность.',
            'Уточните: существенные условия заявки, а также штраф не более 3%.',
            'Недостаточно данных: по представленным источникам нельзя платить задолженность.',
            'Недостаточно данных: нельзя рассчитать сумму, поэтому оплату можно пропустить.',
            'Уточните: дату получения акта, если его нет — не платите долг.',
            'Уточните: предмет договора и все согласования уже выполнены.',
            'Уточните: сумму и деньги перечислены.',
            'Уточните: можете предоставить договор, штраф 10%?',
            'Уточните: можете предоставить акт, неустойка не более 2%?',
            'Уточните: готовы ли вы подтвердить, что договор считается заключенным без подписи?',
            'Недостаточно данных: для расчёта нужен документ. Пособие положено всем.',
        ]
        for request in requests:
            with self.subTest(request=request):
                self.assertFalse(_nonclaim(request))
                _, rejected = GroundedGenerator(None)._parse(SAFE_DRAFT + '\n\n' + request,
                    {'S001': SOURCES[0]}, normalize_caution=False)
                self.assertGreater(rejected, 0)

    def test_heading_grammar_does_not_remove_a_normative_proposition(self):
        for heading in ('Пошаговый план:', 'Практический алгоритм действий:', 'Краткий план согласования договора:'):
            self.assertTrue(_cosmetic(heading), heading)
        for heading in ('Пошаговый план: согласование не требуется', 'План обязательного согласования',
                        'Шаги: получите одобрение директора', 'Максимальное повышение — 8%'):
            self.assertFalse(_cosmetic(heading), heading)

    def test_both_independent_audit_drafts_keep_their_entire_answers(self):
        fixtures = json.loads((ROOT / 'tests/fixtures/parser_v11_audit.json').read_text())
        for name, case in fixtures.items():
            with self.subTest(case=name), patch.dict(os.environ, {'ANSWER_VERIFY': '0'}):
                result = self.run_generator([case['draft']], question=case['question'], sources=case['sources'])
                self.assertEqual(result['mode'], 'llm')
                self.assertEqual(result['trace']['discarded_paragraphs'], 0)
                self.assertTrue(result['answer'].startswith(case['draft'].strip()))
                self.assertIn('Уточните:', result['answer'])
                self.assertNotIn('По найденным положениям:', result['answer'])
                if name == 'f01_foreign_steps':
                    self.assertIn('4.', result['answer'])
                    self.assertIn('сумму, предмет и особенности договора', result['answer'])
                else:
                    self.assertIn('сама по себе не определяет место доставки', result['answer'])
                    self.assertIn('или раньше', result['answer'])

    def test_uncertain_full_draft_reaches_repair_without_deleting_restriction(self):
        draft = ('Увеличение платы возможно. [S001]\n\n'
                 'Но обязательно требуется соглашение сторон, и максимум — 6% за год.')
        result = self.run_generator([draft, SAFE_DRAFT])
        self.assertEqual(self.client.calls[1]['data']['draft'], draft)
        self.assertTrue(self.client.calls[1]['data']['validation_issues']['repair_required'])
        self.assertEqual(result['mode'], 'llm')
        self.assertTrue(result['self_check']['completed'])
        self.assertTrue(result['trace']['review_repaired'])
        self.assertIn('соглашения сторон', result['answer'])
        self.assertIn('6% в год', result['answer'])

    def test_invalid_or_unavailable_repair_never_leaves_unqualified_broad_rule(self):
        bad = 'Увеличение возможно без каких-либо условий. [S001]\n\nОграничение. [S999]'
        for review in ('Неверное условие. [S999]', APIError('private-detail', code='timeout')):
            with self.subTest(review=type(review).__name__):
                result = self.run_generator([bad, review])
                self.assertEqual(result['mode'], 'extractive')
                self.assertIn(RULE, result['answer'])
                self.assertNotIn('без каких-либо', result['answer'])
                self.assertNotIn('private-detail', str(result))
                self.assertEqual(result['self_check']['status'], 'recovery_after_review')
                self.assertFalse(result['self_check']['completed'])
                self.assertEqual(len(self.client.calls), 2)

    def test_truncated_draft_can_be_rebuilt_once_from_sources(self):
        draft = 'Увеличить плату можно. [S001]\n\nНо при этом'
        result = self.run_generator([draft, SAFE_DRAFT], finishes=['length', 'stop'])
        self.assertEqual(self.client.calls[1]['data']['draft'], draft)
        self.assertTrue(self.client.calls[1]['data']['validation_issues']['truncated'])
        self.assertTrue(result['trace']['draft_truncated'])
        self.assertFalse(result['trace']['truncated'])
        self.assertTrue(result['trace']['review_repaired'])
        self.assertIn('соглашения сторон', result['answer'])

    def test_unverified_recovery_preserves_conditions_if_review_disabled(self):
        with patch.dict(os.environ, {'ANSWER_VERIFY': '0'}):
            result = self.run_generator(['Общее право повысить плату. [S001]\n\nОграничение. [S999]'])
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn(RULE, result['answer'])
        self.assertFalse(result['self_check']['completed'])
        self.assertNotIn('Общее право', result['answer'])

    def test_deadline_prevents_repair_but_not_whole_source_recovery(self):
        now = [100.0]
        class SlowClient(ControlledClient):
            def chat_text(self, system, payload, deadline):
                answer = super().chat_text(system, payload, deadline)
                now[0] = 109.6
                return answer
        client = SlowClient(['Общее право. [S001]\n\nОграничение. [S999]'])
        with patch('generation.time.monotonic', side_effect=lambda: now[0]):
            result = GroundedGenerator(client).generate('Как меняется плата?', SOURCES, deadline=110.0)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result['trace']['review_error'], 'review_deadline')
        self.assertIn(RULE, result['answer'])
        self.assertFalse(result['self_check']['completed'])

    def test_sources_heading_cannot_hide_an_uncited_qualification(self):
        draft = 'Увеличение платы возможно. [S001]\n\nИсточники:\nНо только с согласия сторон и не более 6%.'
        result = self.run_generator([draft, SAFE_DRAFT])
        self.assertEqual(self.client.calls[1]['data']['draft'], draft)
        self.assertTrue(result['trace']['review_repair_requested'])
        self.assertTrue(result['trace']['review_repaired'])
        self.assertIn('соглашения сторон', result['answer'])


if __name__ == '__main__':
    unittest.main()
