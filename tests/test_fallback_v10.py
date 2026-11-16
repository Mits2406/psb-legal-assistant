"""Concise recovery keeps relevant complete provisions and their prerequisites."""
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from answer import LegalAssistant
from generation import GroundedGenerator
from query import inspect_query


class FallbackV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = LegalAssistant(ROOT / 'data')

    def recover(self, question):
        a = self.assistant
        ctx = inspect_query(question, registry=a.registry)
        ctx['_answer_context'] = {}
        plan = a.planner.build(question, ctx, deadline=time.monotonic() + 3)
        g = GroundedGenerator(a.client)
        safe = g._safe_sources(a._checked_sources(plan['sources']))
        return g._fallback(question, safe, plan.get('fallback_sources', []), {}, 'synthetic_parser_rejection')

    def test_rent_amount_and_due_survive_without_requisites_or_unrelated_laws(self):
        result = self.recover('Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?')
        answer = result['answer']
        for value in ('450 000', '5-го числа', '7%', 'одного раза в год', 'письменно'):
            self.assertIn(value, answer)
        self.assertNotIn('Реквизиты', answer)
        self.assertLess(len(answer), 1500)

    def test_warranty_keeps_definition_of_delivery_without_other_contract_topics(self):
        result = self.recover('Какой гарантийный срок по договору П-17/2024?')
        answer = result['answer']
        self.assertIn('24 (двадцать четыре) месяца', answer)
        self.assertIn('Дата', answer.replace('Датой', 'Дата'))
        self.assertIn('подписания Покупателем', answer)
        self.assertIn('УПД', answer)
        self.assertNotIn('неустойк', answer)
        self.assertLess(len(answer), 1000)

    def test_payment_keeps_advance_agreement_exception(self):
        answer = self.recover('За сколько дней нужно оплатить товар по договору ПТ-05/2023?')['answer']
        for value in ('45 (сорока пяти)', 'с даты поставки', 'По соглашению Сторон', 'предоплата'):
            self.assertIn(value, answer)
        self.assertNotIn('неустойк', answer)
        self.assertLess(len(answer), 1200)

    def test_requested_penalty_is_not_treated_as_irrelevant(self):
        answer = self.recover('Какая неустойка за просрочку оплаты по ПТ-05/2023?')['answer']
        self.assertIn('0,05%', answer)
        self.assertIn('неоплаченной суммы', answer)

    def test_payment_timing_paraphrases_select_actual_due_rule_in_real_corpus(self):
        for question in ('Когда платить по договору А-08/2024?',
                         'Когда вносить арендную плату по А-08/2024?',
                         'До какого числа нужно заплатить аренду по А-08/2024?'):
            with self.subTest(question=question):
                answer = self.recover(question)['answer']
                self.assertIn('ежемесячно, не позднее 5-го числа текущего месяца', answer)
                self.assertNotIn('Коммунальные платежи', answer)
                self.assertNotIn('соблюдать правила пожарной безопасности', answer)
                self.assertNotIn('0,1%', answer)
                self.assertLess(len(answer), 1000)

    def test_timing_anchor_does_not_substitute_ordinary_rent_for_deposit(self):
        safe = {
            'S001': {'title': 'Учебный договор', 'clause': '2.1',
                     'text': '2.1. Арендная плата вносится ежемесячно до 8-го числа.'},
            'S002': {'title': 'Учебный договор', 'clause': '2.2',
                     'text': '2.2. При подписании договора арендатор вносит обеспечительный платёж.'},
        }
        result = GroundedGenerator(None)._fallback('Когда вносить обеспечительный платёж?', safe,
                                                   ['S001'], {}, 'probe')
        self.assertEqual(result['used_ids'], ['S002'])
        self.assertIn('При подписании договора', result['answer'])

    def test_rent_amount_paraphrases_do_not_select_penalty_instead_of_amount(self):
        for question in ('Какая сумма аренды указана в А-08/2024?',
                         'Каков размер аренды по А-08/2024?',
                         'Какая стоимость аренды по А-08/2024?'):
            with self.subTest(question=question):
                answer = self.recover(question)['answer']
                self.assertIn('450 000', answer)
                self.assertNotIn('0,1%', answer)
                self.assertLess(len(answer), 1500)

    def test_amendment_is_kept_with_its_exact_target(self):
        answer = self.recover('На какой адрес доставлять товар по договору П-17/2024?')['answer']
        self.assertIn('после 1 октября 2024', answer)
        self.assertIn('Химки', answer)
        self.assertIn('Складская', answer)
        self.assertLess(len(answer), 2500)

    def test_shared_definition_does_not_pull_other_users_of_that_definition(self):
        safe = {
            'S001': {'doc_id': 'contract', 'clause': '5.2', 'text': 'Гарантия.', 'dependency_ids': ['S002']},
            'S002': {'doc_id': 'contract', 'clause': '3.3', 'text': 'Определение даты.'},
            'S003': {'doc_id': 'contract', 'clause': '6.1', 'text': 'Проверка качества.', 'dependency_ids': ['S002']},
        }
        self.assertEqual(GroundedGenerator._source_group('S001', safe), ['S001', 'S002'])
        self.assertEqual(GroundedGenerator._source_group('S002', safe), ['S002'])

    def test_no_sources_is_a_normal_absence_not_an_api_failure(self):
        result = GroundedGenerator(None)._fallback('Неизвестный договор?', {}, [], {}, 'no_sources')
        self.assertEqual(result['mode'], 'abstain')
        self.assertNotIn('DIAG_', result['answer'])

    def test_omitted_large_group_does_not_hide_a_complete_independent_prerequisite(self):
        safe = {
            'S001': {'title': 'Учебный договор', 'doc_id': 'contract', 'clause': '7.1',
                     'text': 'Ответственность регулируется следующими условиями. ' * 400,
                     'dependency_ids': ['S002']},
            'S002': {'title': 'Учебный договор', 'doc_id': 'contract', 'clause': '2.1',
                     'text': 'Цена договора составляет 750 000 рублей.'},
        }
        result = GroundedGenerator(None)._fallback('Ответственность и цена договора?', safe,
                    ['S002'], {}, 'probe', cited_ids=['S001'])
        self.assertIn('750 000', result['answer'])
        self.assertNotIn('Ответственность регулируется', result['answer'])
        self.assertIn('Полное связанное положение не помещается', result['answer'])


if __name__ == '__main__':
    unittest.main()
