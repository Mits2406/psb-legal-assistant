"""Neutral headings must not erase a complete, cited calculation; no API calls."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator, _cosmetic


RULE = ('Неустойка составляет 0,1% суммы долга за каждый день просрочки, '
        'но не более 10% суммы долга.')
SOURCES = [{'id': 'S001', 'title': 'Учебный договор', 'clause': '7.2', 'text': RULE}]
ANSWER = ('Неустойка за 150 дней при долге 500000 рублей составляет 50000 рублей: '
          '500000 × 0,1% × 150 = 75000 рублей, но предел — 500000 × 10% = 50000 рублей. [S001]')
NEUTRAL = (
    '**Расчёт неустойки**', '### Расчет штрафа:', 'Предварительный расчет суммы штрафа',
    'Пример расчета пеней', 'Итоговый расчет арендной платы', 'Правовое обоснование',
    'Юридическое обоснование расчета неустойки', 'Обоснование суммы штрафа',
    'Краткое обоснование вывода',
)
CLAIMS = (
    '**Расчёт неустойки: максимум 3%**', '### Штраф не более 2%',
    '**Обоснование: согласование не требуется**', 'Расчет штрафа при отсутствии договора',
    'Расчет неустойки без ограничения суммы', 'Обоснование освобождения от оплаты',
    '**Правовое обоснование: платить не нужно**', 'Расчет штрафа — 500 рублей',
    'Обоснование суммы штрафа\nСогласование не требуется.',
)


class ControlledClient:
    enabled = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.last_chat_diagnostics = {}

    def chat_text(self, system, payload, deadline):
        self.calls.append(payload)
        self.last_chat_diagnostics = {'finish_reason': 'stop'}
        return self.responses.pop(0)


class HeadingV14Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'ANSWER_VERIFY': '1'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No real API'))
        network.start()
        self.addCleanup(network.stop)

    def generate(self, responses):
        self.client = ControlledClient(responses)
        return GroundedGenerator(self.client).generate(
            'Рассчитайте неустойку при долге 500000 рублей и просрочке 150 дней.',
            SOURCES, deadline=time.monotonic() + 30)

    def test_neutral_calculation_and_reasoning_headings_preserve_final_answer(self):
        for heading in NEUTRAL:
            with self.subTest(heading=heading):
                self.assertTrue(_cosmetic(heading))
                draft = heading + '\n\n' + ANSWER
                result = self.generate([draft, draft])
                self.assertEqual(len(self.client.calls), 2)
                self.assertEqual(self.client.calls[1]['draft'], draft)
                self.assertFalse(self.client.calls[1]['validation_issues']['repair_required'])
                self.assertEqual(result['mode'], 'llm')
                self.assertTrue(result['self_check']['completed'])
                self.assertEqual(result['trace']['discarded_paragraphs'], 0)
                self.assertIn(ANSWER, result['answer'])
                self.assertNotIn('По найденным положениям:', result['answer'])

    def test_uncited_limits_and_conditions_still_require_repair(self):
        for claim in CLAIMS:
            with self.subTest(claim=claim):
                self.assertFalse(_cosmetic(claim))
                draft = claim + '\n\n' + ANSWER
                result = self.generate([draft, draft])
                self.assertEqual(len(self.client.calls), 2)
                self.assertTrue(self.client.calls[1]['validation_issues']['repair_required'])
                self.assertEqual(result['mode'], 'extractive')
                self.assertFalse(result['self_check']['completed'])
                self.assertEqual(result['trace']['fallback_reason'], 'structurally_incomplete_draft')
                self.assertIn(RULE, result['answer'])
                self.assertNotIn(claim, result['answer'])

    def test_review_can_repair_an_uncited_claim_without_losing_calculation(self):
        result = self.generate([
            CLAIMS[0] + '\n\n' + ANSWER,
            '**Обоснование расчета неустойки**\n\n' + ANSWER,
        ])
        self.assertTrue(self.client.calls[1]['validation_issues']['repair_required'])
        self.assertEqual(result['mode'], 'llm')
        self.assertTrue(result['self_check']['completed'])
        self.assertTrue(result['trace']['review_repaired'])
        self.assertIn(ANSWER, result['answer'])
        self.assertNotIn('максимум 3%', result['answer'])


if __name__ == '__main__':
    unittest.main()
