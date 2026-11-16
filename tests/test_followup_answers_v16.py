"""Full offline answer regressions for organizer clarification continuations.

These test fallback relevance independently of an answer model or embedding API.
"""
import csv
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from answer import LegalAssistant


class FollowupAnswersTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'ENABLE_EMBEDDINGS': '0', 'QUERY_NORMALIZE': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        with patch('answer.load_env'):
            self.assistant = LegalAssistant(ROOT / 'data')
        self.assistant.client.enabled = self.assistant.client.embeddings_enabled = False
        with (ROOT / 'data/demo_questions.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.questions = {row['id']: row['question'] for row in csv.DictReader(stream)}

    def check_leading_evidence(self, case, clause, fragment):
        result = self.assistant.answer(self.questions[case], reference_date='2026-11-16')
        self.assertEqual(result['mode'], 'extractive')
        self.assertEqual(result['citations'][0]['source_clause'], clause)
        self.assertIn(fragment, result['citations'][0]['quote'])
        self.assertNotIn('персональных данных', result['answer'])
        return result

    def test_rental_return_after_contract_clarification(self):
        self.check_leading_evidence('15', '6.3', 'возвращает Помещение по акту')

    def test_included_delivery_and_packaging_after_contract_clarification(self):
        self.check_leading_evidence('16', '2.1', 'стоимость упаковки и доставки')

    def test_supplier_clarification_keeps_manufacturer_warranty(self):
        self.check_leading_evidence('17', '6.1', 'в гарантийном талоне')

    def test_confirmed_subject_keeps_transfer_not_approval(self):
        result = self.check_leading_evidence('18', '3.2', 'Подписанный оригинал договора передаётся')
        self.assertNotIn('Срок согласования юридическим отделом', result['citations'][0]['quote'])
        self.assertNotIn('комплаенс', result['answer'].lower())
        self.assertNotIn('иностранными контрагентами', result['answer'])

    def test_web_history_and_embedded_csv_have_same_leading_evidence(self):
        for case in ('15', '16', '17', '18'):
            with self.subTest(case=case):
                text = self.questions[case]
                messages = [{'role': 'user' if role == 'Пользователь' else 'assistant', 'content': content.strip()}
                            for role, content in re.findall(r'(Пользователь|Ассистент):\s*(.*?)(?=\n(?:Пользователь|Ассистент):|\Z)', text, re.S)]
                embedded = self.assistant.answer(text, reference_date='2026-11-16')
                structured = self.assistant.answer(messages[-1]['content'], history=messages[:-1], reference_date='2026-11-16')
                self.assertEqual(embedded['status'], structured['status'])
                self.assertEqual(embedded['citations'][0]['id'], structured['citations'][0]['id'])


if __name__ == '__main__':
    unittest.main()
