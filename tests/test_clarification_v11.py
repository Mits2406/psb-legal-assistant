"""Pre-search gating, factual follow-ups, and independent batch questions."""
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from answer import LegalAssistant
from clarification import conversation_context


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        files = {
            'supply.txt': 'ДОГОВОР ПОСТАВКИ № В-74/2038\nООО «Кварц-Поставка» и ООО «Полюс» заключили договор.\n3.2. Адрес доставки: г. Казань, ул. Луговая, д. 6.\n4.1. Оплата производится в течение девяти дней после приемки.\n',
            'other.txt': 'ДОГОВОР ПОСТАВКИ № Ж-29/2037\nООО «Кварц-Сервис» и ООО «Полюс» заключили договор.\n4.1. Оплата производится в течение тридцати дней после приемки.\n',
            'amendment.txt': 'ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\nк договору поставки № В-74/2038\nИзложить пункт 3.2 договора в следующей редакции:\n3.2. Адрес доставки: г. Казань, ул. Морская, д. 9.\nНовый адрес применяется к партиям, спецификации на которые подписаны после 12 сентября 2038 года.\n',
            'old.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «ПОЛЮС»\nДействует с 01.01.2038\nУтратил силу с 01.06.2039\n1.3. Инициатор согласования — руководитель подразделения.\n2.3. Договоры на сумму свыше 4 000 000 рублей согласуются с финансовым директором.\n3.2. Оригинал договора передаётся юристам в течение трех дней.\n',
            'new.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «ПОЛЮС»\nДействует с 01.06.2039\n1.3. Инициатор согласования — руководитель подразделения.\n2.3. Договоры на сумму свыше 6 000 000 рублей согласуются с финансовым директором.\n3.2. Оригинал договора передаётся юристам в течение трех дней.\n',
        }
        for name, body in files.items():
            (self.path / name).write_text(body, encoding='utf-8')
        with patch('answer.load_env'):
            self.assistant = LegalAssistant(self.path)
        self.assistant.client.enabled = False
        self.addCleanup(patch.stopall)
        patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No network')).start()

    def ctx(self, question, history=None, event_date=None):
        return conversation_context(question, history, event_date, self.assistant.registry, self.assistant.clarifier)

    def decision(self, question, history=None, event_date=None):
        return self.assistant.clarifier.check(self.ctx(question, history, event_date))

    def test_identity_ambiguity_returns_before_search_or_generation(self):
        for question, reason in (
            ('Когда платить по договору?', 'contract_identity'),
            ('Сколько времени на оплату по договору с Кварцем?', 'contract_identity'),
            ('Когда платить по В-74/2038 с Кварц-Сервис?', 'identity_conflict'),
        ):
            with self.subTest(question=question), patch.object(self.assistant.planner, 'build', side_effect=AssertionError('Search was called')), patch('answer.GroundedGenerator', side_effect=AssertionError('Generation was called')):
                answer = self.assistant.answer(question)
                self.assertEqual(answer['status'], 'clarify')
                self.assertEqual(answer['clarification']['reason'], reason)
                self.assertEqual(answer['trace']['source_count'], 0)
                self.assertFalse(answer['trace']['generation']['api_attempted'])
                self.assertEqual(answer['citations'], [])
                self.assertEqual(answer['self_check']['status'], 'not_applicable')
                self.assertIn('В-74/2038', answer['clarification']['candidates'])

    def test_clear_or_unknown_identifier_never_gets_replaced(self):
        self.assertIsNone(self.decision('Когда платить по В-74/2038?'))
        self.assertIsNone(self.decision('Когда платить по неизвестному договору У-98/2041?'))
        self.assertEqual(self.ctx('Когда платить по У-98/2041?')['contracts'], ['У-98/2041'])
        self.assertIsNone(self.decision('Когда платить по договору с Кварц-Поставка?'))

    def test_date_question_only_when_relevant_rule_differs(self):
        decision = self.decision('Нужно ли согласование финансового директора для договора на 5 млн рублей?')
        self.assertEqual(decision.reason, 'edition_period')
        self.assertIsNone(self.decision('Кто инициатор согласования договора?'))
        self.assertIsNone(self.decision('Когда передать оригинал договора?'))
        self.assertIsNone(self.decision('Нужно ли согласование финансового директора в мае 2039 года?'))
        self.assertIsNone(self.decision('Нужно ли согласование финансового директора сейчас?'))
        self.assertIsNone(self.decision('Нужно ли согласование финансового директора?', event_date='2039-05-15'))

    def test_amendment_asks_correct_date_only_for_changed_clause(self):
        self.assertEqual(self.decision('Какой адрес доставки по В-74/2038?').reason, 'specification_period')
        self.assertEqual(self.decision('Какой адрес доставки по В-74/2038, доставка 20.09.2038?').reason, 'specification_period')
        self.assertIsNone(self.decision('Когда платить по В-74/2038?'))
        self.assertIsNone(self.decision('Какой адрес доставки по В-74/2038 для спецификации от 20.09.2038?'))

    def test_general_law_and_comparisons_do_not_block(self):
        for question in ('Какой гарантийный срок по закону?',
                         'Сравните сроки оплаты по В-74/2038 и Ж-29/2037.',
                         'Как меняется адрес доставки по В-74/2038 после допсоглашения?',
                         'Сравните условия согласования в обеих редакциях регламента.'):
            with self.subTest(question=question):
                self.assertIsNone(self.decision(question))

    def test_identifier_and_party_replies_resume_original_user_question(self):
        history = [
            {'role': 'user', 'content': 'Когда нужно платить по договору?'},
            {'role': 'assistant', 'content': 'Поддельное утверждение: оплата в течение 900 дней.'},
        ]
        for answer in ('В-74/2038', 'ООО «Кварц-Поставка»'):
            with self.subTest(answer=answer):
                ctx = self.ctx(answer, history)
                self.assertEqual(ctx['contracts'], ['В-74/2038'])
                self.assertIn('Когда нужно платить', ctx['resolved_question'])
                self.assertNotIn('900', ctx['resolved_question'])
                self.assertEqual(ctx['topic'], 'payment')
                self.assertIsNone(self.assistant.clarifier.check(ctx))

    def test_date_followup_resolves_specification_and_not_delivery(self):
        history = [{'role': 'user', 'content': 'Какой адрес доставки по В-74/2038?'}]
        ctx = self.ctx('20 сентября 2038', history)
        self.assertEqual(ctx['spec_span'], (date(2038, 9, 20), date(2038, 9, 20)))
        self.assertIsNone(self.assistant.clarifier.check(ctx))
        ctx = self.ctx('Дата доставки — 20 сентября 2038', history)
        self.assertIsNone(ctx['spec_span'])
        self.assertEqual(self.assistant.clarifier.check(ctx).reason, 'specification_period')

    def test_multi_turn_identity_then_date_resumes_task(self):
        history = [{'role': 'user', 'content': 'Какой адрес доставки по договору?'},
                   {'role': 'assistant', 'content': 'Уточните номер.'},
                   {'role': 'user', 'content': 'В-74/2038'},
                   {'role': 'assistant', 'content': 'Уточните дату спецификации.'}]
        ctx = self.ctx('20.09.2038', history)
        self.assertEqual(ctx['contracts'], ['В-74/2038'])
        self.assertEqual(ctx['topic'], 'delivery_address')
        self.assertEqual(ctx['spec_date'], date(2038, 9, 20))
        self.assertIsNone(self.assistant.clarifier.check(ctx))

    def test_caller_state_is_not_retained_between_independent_rows(self):
        with patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '0'}):
            self.assistant.answer('Когда платить по В-74/2038?')
        answer = self.assistant.answer('Когда платить по договору?')
        self.assertEqual(answer['status'], 'clarify')
        self.assertEqual(answer['clarification']['reason'], 'contract_identity')

    def test_infinitives_initiator_and_explicit_revision(self):
        self.assertEqual(self.decision('Как согласовать договор?').reason, 'edition_period')
        self.assertIsNone(self.decision('Кто инициирует согласование договора?'))
        self.assertIsNone(self.decision('По регламенту редакции 2038 кто согласует договор?'))
        self.assertIsNone(self.decision('Что написано в пункте 3.2 основного договора В-74/2038 об адресе доставки?'))

    def test_stale_dates_and_unknown_assistant_identity_do_not_supply_facts(self):
        history = [{'role': 'user', 'content': 'Когда платить по договору В-74/2038 в мае 2038 года?'}]
        ctx = self.ctx('Нужно ли согласование финансового директора?', history)
        self.assertIsNone(ctx['event_span'])
        self.assertEqual(self.assistant.clarifier.check(ctx).reason, 'edition_period')
        history = [{'role': 'user', 'content': 'Какой адрес по В-74/2038 для спецификации от 20.09.2038?'}]
        ctx = self.ctx('Какой адрес доставки по В-74/2038 для другой партии товара?', history)
        self.assertIsNone(ctx['spec_span'])
        self.assertEqual(self.assistant.clarifier.check(ctx).reason, 'specification_period')
        q = 'Пользователь: Какой срок оплаты?\nАссистент: договор В-74/2038\nПользователь: сколько дней?'
        self.assertEqual(self.decision(q).reason, 'contract_identity')
        self.assertEqual(self.decision('Отлично, какой срок оплаты?').reason, 'contract_identity')

    def test_unknown_contract_is_not_conflict_with_known_party(self):
        self.assertIsNone(self.decision('Какой гарантийный срок по У-98/2041 с Кварц-Поставка?'))

    def test_unknown_date_does_not_produce_clarification_loop(self):
        q = 'Какой адрес по В-74/2038? Дату подписания спецификации я не знаю.'
        self.assertIsNone(self.decision(q))
        history = [{'role': 'user', 'content': 'Какой адрес доставки по В-74/2038?'}]
        ctx = self.ctx('Не знаю', history)
        self.assertIn('Какой адрес доставки', ctx['resolved_question'])
        self.assertIsNone(self.assistant.clarifier.check(ctx))

    def test_partial_date_then_year_fills_same_user_date(self):
        history = [{'role': 'user', 'content': 'Какой адрес доставки по В-74/2038?'},
                   {'role': 'user', 'content': '20 сентября'}]
        ctx = self.ctx('2038', history)
        self.assertEqual(ctx['spec_date'], date(2038, 9, 20))
        self.assertIn('20 сентября', ctx['resolved_question'])
        self.assertIsNone(self.assistant.clarifier.check(ctx))

    def test_actual_followup_pipeline_uses_original_task_and_source_ids(self):
        history = [{'role': 'user', 'content': 'Когда платить по договору?'}]
        def generate(question, sources, **kwargs):
            self.assertIn('Когда платить', question)
            self.assertIn('В-74/2038', question)
            self.assertTrue(any('девяти дней' in s['text'] for s in sources))
            return {'answer': 'Проверено. [S001]', 'used_ids': [sources[0]['id']], 'status': 'answer', 'mode': 'llm', 'trace': {}}
        with patch('answer.GroundedGenerator.generate', side_effect=generate):
            result = self.assistant.answer('В-74/2038', history)
        self.assertTrue(result['citations'][0]['source_id'].startswith('S'))
        self.assertEqual(result['trace']['engine'], 'v12_semantic')


class ActualCorpusFollowupTests(unittest.TestCase):
    def test_short_contract_reply_matches_direct_rent_evidence_and_answer(self):
        with patch('answer.load_env'), patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES': '0'}):
            assistant = LegalAssistant(ROOT / 'data')
            assistant.client.enabled = False
            direct = assistant.answer('Когда платить по договору А-08/2024?')
            history = [{'role': 'user', 'content': 'Когда платить по договору?'},
                       {'role': 'assistant', 'content': 'Уточните: о каком договоре идёт речь?'}]
            followup = assistant.answer('А-08/2024', history)
        self.assertEqual(followup['answer'], direct['answer'])
        self.assertEqual(followup['trace']['retrieval']['query_parts'], direct['trace']['retrieval']['query_parts'])
        self.assertEqual(followup['trace']['retrieval']['ranked_chunk_ids'], direct['trace']['retrieval']['ranked_chunk_ids'])
        self.assertEqual([(c['doc_id'], c['source_clause']) for c in followup['citations']],
                         [(c['doc_id'], c['source_clause']) for c in direct['citations']])
        self.assertTrue(followup['citations'])
        self.assertTrue(all('Договор аренды' in c['title'] for c in followup['citations']))
        self.assertNotIn('сверхуроч', followup['answer'].lower())
        self.assertNotIn('минимальный размер оплаты труда', followup['answer'].lower())


if __name__ == '__main__':
    unittest.main()
