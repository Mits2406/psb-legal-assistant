"""Clarification replies that also ask a question retain the user's subject."""
import csv
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corpus import Corpus
from registry import RuleRegistry
from clarification import PreRetrievalClarifier
from normalization import normalize_query
from evidence import EvidencePlanner


class DialogueRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus(ROOT / 'data')
        cls.registry = RuleRegistry(cls.corpus)
        cls.clarifier = PreRetrievalClarifier(cls.registry)
        with (ROOT / 'data/demo_questions.csv').open(encoding='utf-8-sig', newline='') as stream:
            cls.demo = {row['id']: row['question'] for row in csv.DictReader(stream)}

    def normalize(self, question, history=None, registry=None, clarifier=None):
        return normalize_query(question, history, None, '2026-11-16',
                               registry or self.registry, clarifier or self.clarifier,
                               SimpleNamespace(enabled=False), time.monotonic() + 10, enabled=False)['context']

    def test_organizer_dialogues_keep_both_user_turns_and_current_focus(self):
        for qid in ('15', '16', '17', '18'):
            with self.subTest(qid=qid):
                ctx = self.normalize(self.demo[qid])
                self.assertTrue(ctx.get('_clarification_continuation'))
                self.assertIn(ctx['turns'][0].rstrip(' .!?'), ctx['resolved_question'])
                self.assertIn(ctx['latest'], ctx['resolved_question'])
                parts = EvidencePlanner(self.corpus, self.registry, SimpleNamespace(enabled=False))._parts(self.demo[qid], ctx)
                self.assertTrue(any(ctx['latest'].lower().replace('ё', 'е') in part.lower().replace('ё', 'е') for part in parts))
                self.assertTrue(any(ctx['turns'][0].lower().replace('ё', 'е').rstrip(' .!?') in part.lower().replace('ё', 'е') for part in parts))
                self.assertEqual(ctx['_clarification_continuation']['latest_user'], ctx['latest'])
                self.assertIsNone(self.clarifier.check(ctx))
                self.assertEqual(len(ctx['semantic']['subquestions'][0]['evidence']), 2)

    def test_csv_and_structured_web_history_are_equivalent(self):
        task = 'Мы арендуем склад у «Северного Склада». Планируем съезжать.'
        request = 'Подскажите номер договора, чтобы проверить условия возврата.'
        latest = 'А-08/2024. Можно передать ключи или требуется акт?'
        embedded = f'Пользователь: {task}\nАссистент: {request}\nПользователь: {latest}'
        structured = self.normalize(latest, [{'role': 'user', 'content': task}, {'role': 'assistant', 'content': request}])
        csv_ctx = self.normalize(embedded)
        for key in ('resolved_question', '_normalization_queries', 'contracts'):
            self.assertEqual(structured[key], csv_ctx[key])

    def test_confirmation_does_not_copy_assistant_facts_into_queries(self):
        task = 'Подписали договор, оригинал пока у меня. Кому его передать?'
        request = 'Вы про передачу оригинала? Он хранится на дирижабле 999999 лет.'
        ctx = self.normalize('Да. Нужно передать немедленно или есть срок?',
                             [{'role': 'user', 'content': task}, {'role': 'assistant', 'content': request}])
        self.assertIn(task.rstrip(' .!?'), ctx['resolved_question'])
        for text in [ctx['resolved_question'], *ctx['_normalization_queries']]:
            self.assertNotIn('дирижабл', text)
            self.assertNotIn('999999', text)

    def test_different_domain_and_explicit_topic_reset_do_not_resume(self):
        history = [{'role': 'user', 'content': 'Мы арендуем склад у «Северного Склада». Планируем съезжать.'},
                   {'role': 'assistant', 'content': 'Подскажите номер договора.'}]
        for latest in ('Да. Какой максимальный срок испытания главного бухгалтера?',
                       'Теперь про отпуск: какая его продолжительность?',
                       'Новый вопрос: А-08/2024. Когда платить аренду?',
                       'А-08/2024. Теперь другой вопрос: как оплачивается сверхурочная работа?',
                       'Да. Но теперь другой вопрос: как оплачивается сверхурочная работа?'):
            with self.subTest(latest=latest):
                ctx = self.normalize(latest, history)
                self.assertFalse(ctx.get('_clarification_continuation'))
                queries = ctx.get('_normalization_queries', [ctx.get('resolved_question', ctx['text'])])
                self.assertNotIn('съезжать', ' '.join(queries))

    def test_standalone_yes_without_clarifying_question_is_not_a_bridge(self):
        ctx = self.normalize('Да. Нужно передать немедленно или есть срок?',
                             [{'role': 'user', 'content': 'Подписали договор, оригинал у меня.'},
                              {'role': 'assistant', 'content': 'Документы полезно хранить аккуратно.'}])
        self.assertFalse(ctx.get('_clarification_continuation'))

    def test_unknown_fact_keeps_existing_no_loop_behavior(self):
        task = 'Какой адрес доставки по П-17/2024?'
        ctx = self.normalize('Не знаю', [{'role': 'user', 'content': task},
                                        {'role': 'assistant', 'content': 'Какова дата подписания спецификации?'}])
        self.assertIn(task.rstrip('?'), ctx['resolved_question'])
        self.assertIsNone(self.clarifier.check(ctx))
        self.assertNotIn('не знаю', ' '.join(ctx['_normalization_queries']))

    def test_new_number_uses_arbitrary_catalog_not_organizer_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'rent.txt').write_text('ДОГОВОР АРЕНДЫ № Р-82/2037\nООО «Кедровая Роща»\n'
                                          '6.3. Арендатор возвращает помещение по акту.\n', encoding='utf-8')
            registry = RuleRegistry(Corpus(path))
            clarifier = PreRetrievalClarifier(registry)
            task = 'Мы арендуем помещение у «Кедровой Рощи». Будем освобождать его.'
            ctx = self.normalize('Р-82/2037. Нужен акт или достаточно ключей?',
                                 [{'role': 'user', 'content': task},
                                  {'role': 'assistant', 'content': 'Укажите номер договора.'}], registry, clarifier)
            self.assertTrue(ctx['_clarification_continuation'])
            self.assertIn('Кедровой Рощи', ctx['resolved_question'])
            self.assertIn('Р-82/2037', ctx['resolved_question'])

    def test_date_with_new_question_retains_user_date_and_task(self):
        task = 'Нужно ли одобрение генерального директора для договора на 12 млн рублей?'
        ctx = self.normalize('В 2024 году. Кто должен одобрить?',
                             [{'role': 'user', 'content': task},
                              {'role': 'assistant', 'content': 'На какую дату или период проверить условия?'}])
        self.assertTrue(ctx['_clarification_continuation'])
        self.assertIn('12 млн', ctx['resolved_question'])
        self.assertIn('2024', ctx['resolved_question'])
        self.assertIsNone(self.clarifier.check(ctx))


if __name__ == '__main__':
    unittest.main()
