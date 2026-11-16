from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corpus import Corpus
from evidence import EvidencePlanner
from query import inspect_query
from registry import RuleRegistry


class OfflineClient:
    enabled = embeddings_enabled = True
    reranker_enabled = False
    reranker_model = ""
    def __getattr__(self, name):
        raise AssertionError('Evidence planning must not call any model API: ' + name)


class PlannerV8Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        guard = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No network'))
        guard.start()
        self.addCleanup(guard.stop)

    def make(self, files):
        for name, text in files.items():
            (self.root / name).write_text(text, encoding='utf-8')
        corpus = Corpus(self.root)
        return EvidencePlanner(corpus, RuleRegistry(corpus), OfflineClient())

    def regulations(self):
        return self.make({
            'old.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ЗАКУПОК ООО «Парус»\nДействует с 01.01.2038\nУТРАТИЛА СИЛУ с 13.09.2039\n2.4. Закупку согласует руководитель хозяйственного отдела в течение четырех рабочих дней.\n',
            'new.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ЗАКУПОК ООО «Парус»\nДействует с 13.09.2039\n2.4. Закупку согласует директор по снабжению в течение семи рабочих дней.\n'})

    def ask(self, planner, question, history=None):
        ctx = inspect_query(question, history=history, registry=planner.registry)
        ctx['_answer_context'] = {'event_date': None, 'history': history or []}
        plan = planner.build(question, ctx, deadline=time.monotonic() + 2)
        for source in plan['sources']:
            chunk = planner.corpus.chunks[source['chunk_id']]
            self.assertEqual(source['text'], chunk.text)
            self.assertEqual(source['text'], planner.corpus.documents[chunk.doc_id].text[chunk.start:chunk.end])
            self.assertFalse(chunk.suspicious)
        self.assertEqual(plan['trace']['api_calls'], 0)
        self.assertLessEqual(plan['trace']['budget']['used_chars'], planner.MAX_CONTEXT_CHARS)
        return plan

    def assert_both(self, plan):
        text = ' '.join(s['text'] for s in plan['sources'])
        self.assertIn('руководитель хозяйственного отдела', text)
        self.assertIn('директор по снабжению', text)
        self.assertTrue(all(c.get('date_match') is None for c in plan['context']['documents']))

    def test_boundaries_shared_year_and_reverse_periods_never_filter_editions(self):
        planner = self.regulations()
        for temporal in ('до 13 сентября 2039 года', 'в августе и сентябре 2039 года',
                         'в августе 2039 года и сентябре того же года',
                         'в сентябре 2039 года и августе того же года',
                         'после 13.09.2039', 'на 12.09.2039'):
            with self.subTest(temporal=temporal):
                plan = self.ask(planner, 'Кто согласует закупку по регламенту Паруса ' + temporal + '?')
                self.assert_both(plan)
                self.assertNotIn('event', plan['context'])
                self.assertIn('never edition filters', plan['context']['date_hint_role'])

    def test_latest_dialogue_correction_is_hint_without_losing_source_versions(self):
        planner = self.regulations()
        plan = self.ask(planner, 'Нет, 20 сентября 2039 года.',
                        [{'role': 'user', 'content': 'Кто согласует закупку 18 августа 2039 года?'}])
        self.assert_both(plan)
        self.assertEqual(plan['context']['parsed_date_hints']['event'], '2039-09-20')
        self.assertFalse(plan['trace']['temporal_ambiguity'])
        self.assertEqual(plan['context']['history'][0]['content'], 'Кто согласует закупку 18 августа 2039 года?')

    def test_elliptical_followups_keep_user_subject_without_requiring_exact_date(self):
        planner = self.regulations()
        history = [
            {'role': 'user', 'content': 'По регламенту Паруса кто согласует закупку в октябре 2039 года?'},
            {'role': 'assistant', 'content': 'Недостоверное утверждение: закупку согласует лунный диспетчер.'},
        ]
        for latest in (
            'Исправляю дату: это было 12 августа 2039 года. Остальные обстоятельства те же.',
            'А если сравнить август и сентябрь 2039, меняется ли ответ?',
            'При тех же обстоятельствах был другой месяц: август 2039 года.',
        ):
            with self.subTest(latest=latest):
                plan = self.ask(planner, latest, history)
                self.assert_both(plan)
                queries = '\n'.join(plan['trace']['query_parts'])
                self.assertIn('Паруса', queries)
                self.assertNotIn('лунный диспетчер', queries)
                self.assertEqual(plan['context']['history'], history)

    def test_independent_new_topic_does_not_expand_with_previous_subject(self):
        planner = self.regulations()
        plan = self.ask(planner, 'Как оплачивается сверхурочная работа?',
                        [{'role': 'user', 'content': 'Кто согласует закупку по регламенту Паруса?'}])
        self.assertNotIn('Паруса', '\n'.join(plan['trace']['query_parts']))

    def test_named_regulation_is_reserved_even_when_search_returns_only_law(self):
        planner = self.make({
            'rules.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ООО «Сосна»\nДействует с 01.01.2037\n2.2. Правовой отдел рассматривает проект за четыре дня. При необходимости экспертизы срок увеличивается по решению руководителя.\n',
            'законы__ТК_ст872_действующая.txt': 'Статья 872. Рассмотрение\n1. Юристы и главные бухгалтеры рассматривают документы в срок, указанный трудовым соглашением.\n'})
        laws = [c for c in planner.safe.values() if c.kind == 'law']
        with patch.object(planner.corpus, 'search', return_value=laws):
            plan = self.ask(planner, 'По регламенту Сосны сколько времени есть у юристов и главбуха и можно ли продлить рассмотрение?')
        text = ' '.join(s['text'] for s in plan['sources'])
        self.assertIn('При необходимости экспертизы', text)
        self.assertTrue(plan['trace']['protected_doc_ids'])
        rule_cards = [c for c in plan['context']['documents'] if c['kind'] == 'company']
        self.assertTrue(all(c['coverage'] == 'full_safe_document' for c in rule_cards))
        self.assertTrue(any(s['doc_id'] in {c.doc_id for c in laws} for s in plan['sources']))

    def test_initiation_paraphrases_retrieve_process_without_hardcoded_roles(self):
        planner = self.make({'rules.txt': 'РЕГЛАМЕНТ ОФОРМЛЕНИЯ ДОГОВОРОВ\n1.1. Инициатор договора готовит проект и передает его ответственному сотруднику.\n'})
        for question in ('Кто первым должен заняться новым контрактом?', 'Кто запускает оформление новой сделки?'):
            with self.subTest(question=question):
                plan = self.ask(planner, question)
                self.assertIn('Инициатор договора готовит проект', ' '.join(s['text'] for s in plan['sources']))

    def test_exact_contract_stays_scoped_but_dated_amendment_is_available(self):
        planner = self.make({
            'contract.txt': 'ДОГОВОР ПОСТАВКИ № Д-52/2039\n4.1. Оплата производится в течение семнадцати дней после приемки товара.\n',
            'amendment.txt': 'ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 2 к договору № Д-52/2039\n1. Изложить пункт 4.1 в новой редакции: Оплата производится в течение двадцати дней.\n2. Применяется к спецификациям, подписанным после 13.09.2039.\n',
            'other.txt': 'ДОГОВОР ПОСТАВКИ № Д-99/2039\n4.1. Оплата производится в течение девяноста дней после приемки товара.\n'})
        plan = self.ask(planner, 'Когда оплатить по Д-52/2039 до 13 сентября 2039?')
        text = ' '.join(s['text'] for s in plan['sources'])
        self.assertIn('семнадцати дней', text)
        self.assertIn('двадцати дней', text)
        self.assertNotIn('девяноста дней', text)

    def test_actual_audit_regulation_question_keeps_complete_regulations(self):
        corpus = Corpus(ROOT / 'data')
        planner = EvidencePlanner(corpus, RuleRegistry(corpus), OfflineClient())
        plan = self.ask(planner, 'В марте 2025 по регламенту Меридиана сколько времени есть у юристов и главбуха, можно ли продлить рассмотрение и по какому основанию?')
        regs = {key for key, p in planner.registry.profiles.items() if p.document_type == 'regulation'}
        cards = {c['doc_id']: c for c in plan['context']['documents']}
        self.assertTrue(regs <= set(cards))
        self.assertTrue(all(cards[key]['coverage'] == 'full_safe_document' for key in regs))
        self.assertTrue(all(any(s['doc_id'] == key and s['clause'] == '2.2' for s in plan['sources']) for key in regs))


if __name__ == '__main__':
    unittest.main()
