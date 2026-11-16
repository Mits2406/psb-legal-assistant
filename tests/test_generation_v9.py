"""Diagnostic change: formatting must not erase an intact, cited answer."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator


SOURCE = ('4.2. Поставщик вправе изменить цену не чаще одного раза в полугодие, '
          'не более чем на 4%, после письменного согласия покупателя.')
ANSWER = ('Цена может меняться раз в полугодие, максимум на 4%, '
          'после письменного согласия покупателя. [S001]')
SOURCES = [{'id': 'S001', 'doc_id': 'contract', 'title': 'Учебный договор поставки',
            'clause': '4.2', 'text': SOURCE}]


class Client:
    enabled = True
    def __init__(self, draft, finish='stop'):
        self.draft = draft
        self.finish = finish
        self.calls = 0
        self.last_chat_diagnostics = {}
    def chat_text(self, system, payload, deadline):
        self.calls += 1
        self.last_chat_diagnostics = {'finish_reason': self.finish}
        return self.draft


class FormattingV9Tests(unittest.TestCase):
    def run_answer(self, draft, finish='stop'):
        client = Client(draft, finish)
        with patch.dict(os.environ, {'ANSWER_VERIFY': '0'}):
            result = GroundedGenerator(client).generate(
                'Как изменить цену по договору поставки?', SOURCES,
                deadline=time.monotonic()+10)
        self.assertEqual(client.calls, 1)
        self.assertLessEqual(len(result['answer']), 10000)
        return result

    def test_neutral_labels_do_not_destroy_an_intact_answer(self):
        for label in ('**Размер арендной платы**', '## Порядок оплаты',
                      '__Срок рассмотрения документов__', 'Условия изменения договора:',
                      'Применимая редакция', 'Ответ'):
            with self.subTest(label=label):
                result = self.run_answer(label+'\n\n'+ANSWER)
                self.assertEqual(result['mode'], 'llm')
                self.assertIn(ANSWER, result['answer'])
                self.assertNotIn(label+'\n', result['answer'])
                self.assertEqual(result['trace']['discarded_paragraphs'], 0)

    def test_neutral_introductions_do_not_destroy_an_intact_answer(self):
        for intro in ('По договору действуют следующие условия:',
                      'По предоставленным документам можно выделить следующие положения:'):
            with self.subTest(intro=intro):
                result = self.run_answer(intro+'\n\n'+ANSWER)
                self.assertEqual(result['mode'], 'llm')
                self.assertIn(ANSWER, result['answer'])
                self.assertNotIn(intro, result['answer'])

    def test_specific_fact_questions_survive_with_the_answer(self):
        for question in ('Уточните: когда арендатор получил письменное уведомление о повышении?',
                         'Уточните: в какую дату поставщик передал товар?',
                         'Уточните: когда стороны подписали соглашение?',
                         'Уточните: была ли подписана спецификация?'):
            with self.subTest(question=question):
                result = self.run_answer(ANSWER+'\n\n'+question)
                self.assertEqual(result['mode'], 'llm')
                self.assertIn(ANSWER, result['answer'])
                self.assertIn(question, result['answer'])

    def test_standalone_specific_question_preserves_the_requested_fact(self):
        question = 'Уточните: когда покупатель получил счет?'
        result = self.run_answer(question)
        self.assertEqual(result['mode'], 'clarify')
        self.assertEqual(result['answer'], question)

    def test_legal_assertions_disguised_as_labels_are_not_cosmetic(self):
        for label in ('**Повышение платы без уведомления**',
                      '**Условия повышения без ограничений**',
                      'Размер платы: 900 рублей',
                      '# Договор продлевается автоматически',
                      'По договору действуют следующие условия: согласие не требуется'):
            with self.subTest(label=label):
                result = self.run_answer(label+'\n\n'+ANSWER)
                self.assertEqual(result['mode'], 'extractive')
                self.assertIn(SOURCE, result['answer'])
                self.assertNotIn(label, result['answer'])

    def test_question_prefix_cannot_smuggle_claims_or_conditions(self):
        for question in ('Уточните: когда поставщик вправе увеличить цену без согласия?',
                         'Уточните: когда покупатель получил счет, поскольку согласие не требуется?',
                         'Уточните: когда покупатель получил счет? Повышение не ограничено.',
                         'Уточните: когда покупатель получил счет без уведомления?',
                         'Недостаточно данных: когда покупатель получил счет?',
                         'Уточните: когда покупатель получил счет? [S999]'):
            with self.subTest(question=question):
                result = self.run_answer(ANSWER+'\n\n'+question)
                self.assertEqual(result['mode'], 'extractive')
                self.assertIn(SOURCE, result['answer'])
                self.assertNotIn(question, result['answer'])

    def test_neutral_heading_does_not_hide_a_damaged_restriction(self):
        draft = ('**Порядок оплаты**\n\nПоставщик вправе повысить цену. [S001]\n\n'
                 'Повышение требует согласия. [S999]')
        result = self.run_answer(draft)
        self.assertEqual(result['mode'], 'extractive')
        self.assertIn(SOURCE, result['answer'])
        self.assertNotIn('Поставщик вправе повысить цену.', result['answer'])
        self.assertNotIn('S999', result['answer'])

    def test_even_complete_looking_length_response_still_recovers_sources(self):
        result = self.run_answer('Ответ\n\n'+ANSWER, finish='length')
        self.assertEqual(result['mode'], 'extractive')
        self.assertEqual(result['trace']['fallback_reason'], 'truncated_draft')
        self.assertIn(SOURCE, result['answer'])

    def test_unfinished_condition_is_not_ignored_as_formatting(self):
        result = self.run_answer('Ответ\n\nПоставщик вправе повысить цену. [S001]\n\nНо только если',
                                 finish='length')
        self.assertEqual(result['mode'], 'extractive')
        self.assertIn(SOURCE, result['answer'])
        self.assertNotIn('Но только если', result['answer'])

    def test_formatting_alone_is_not_an_answer(self):
        result = self.run_answer('## Ответ')
        self.assertEqual(result['mode'], 'extractive')
        self.assertIn(SOURCE, result['answer'])


if __name__ == '__main__':
    unittest.main()
