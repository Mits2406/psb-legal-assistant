"""Retrieval preserves an omitted legal subject, not an earlier asserted norm."""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from answer import LegalAssistant
from evidence import _unnamed_rule_followup


class HistorySubjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict('os.environ', {'QUERY_NORMALIZE': '0', 'DIAGNOSTIC_API_FAILURES': '0'}):
            cls.assistant = LegalAssistant(ROOT / 'data')
        cls.assistant.client.enabled = False

    def ask(self, question, history=None):
        with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No network')):
            with patch.dict('os.environ', {'QUERY_NORMALIZE': '0', 'DIAGNOSTIC_API_FAILURES': '0'}):
                return self.assistant.answer(question, history, deadline=time.monotonic()+10)

    def test_historical_followup_retrieves_original_user_subject(self):
        history = [
            {'role': 'user', 'content': 'Какой МРОТ прямо написан в загруженном законе с января 2026?'},
            {'role': 'assistant', 'content': 'Недостоверный ответ: закон о зелёных дирижаблях устанавливает 999999 рублей.'},
        ]
        for q in [
            'А какую сумму устанавливал закон на 1 января 2021 года? Нужна именно норма того времени, не пересчёт сегодняшней суммы.',
            'А какой размер предусматривала норма на 1 января 2022 года?',
        ]:
            with self.subTest(question=q):
                result = self.ask(q, history)
                parts = '\n'.join(result['trace']['retrieval']['query_parts'])
                self.assertIn('МРОТ', parts)
                self.assertNotIn('дирижабл', parts)
                self.assertNotIn('999999', parts)
                self.assertTrue(any('МРОТ' in c['title'] for c in result['citations']))
                self.assertIn(q, parts)

    def test_explicit_new_subject_does_not_inherit_old_law(self):
        history = [{'role': 'user', 'content': 'Какой МРОТ установлен законом?'}]
        for q in ['А какой размер НДС действовал по закону на 1 января 2021 года?',
                  'А какую сумму штрафа устанавливал закон на 1 января 2021 года?',
                  'А какой размер устанавливает статья 224 на 1 января 2021 года?']:
            with self.subTest(question=q):
                self.assertFalse(_unnamed_rule_followup(q))
                result = self.ask(q, history)
                self.assertNotIn('МРОТ', '\n'.join(result['trace']['retrieval']['query_parts']))

    def test_independent_row_does_not_reuse_previous_question(self):
        self.ask('Какой МРОТ установлен законом?')
        q = 'А какую сумму устанавливал закон на 1 января 2021 года?'
        result = self.ask(q)
        self.assertNotIn('МРОТ', '\n'.join(result['trace']['retrieval']['query_parts']))


if __name__ == '__main__':
    unittest.main()
