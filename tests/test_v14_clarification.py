"""Clarify only facts that can change the source-supported requested answer."""
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clarification import PreRetrievalClarifier, _topic
from corpus import Corpus
from normalization import normalize_query
from registry import RuleRegistry


class ApplicabilityGateV14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = RuleRegistry(Corpus(ROOT / 'data'))
        cls.clarifier = PreRetrievalClarifier(cls.registry)

    def setUp(self):
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No API'))
        network.start()
        self.addCleanup(network.stop)

    def decision(self, question, clarifier=None):
        clarifier = clarifier or self.clarifier
        ctx = normalize_query(question, None, None, '2026-11-16', clarifier.registry,
                              clarifier, SimpleNamespace(enabled=False), time.monotonic() + 20,
                              enabled=False)['context']
        return clarifier.check(ctx)

    def test_existing_answerable_cases_reach_retrieval(self):
        cases = {
            'demo10': 'Мы готовим новый договор. Кто должен запустить согласование — юристы или руководитель нашего отдела?',
            'retrieval_P06': 'Регламент 2025 года: вправе ли согласующий попросить дополнительное время и каков предел такого продления?',
            'retrieval_P07': 'Что кроме самого зарубежного контрагента проверяют по санкционному чек-листу из письма К-12?',
            'holdout_H11': 'Коллега говорит: «В П-17/2024 и привезти мебель, и заплатить за неё нужно за 30 календарных дней после спецификации». Какие два срока и события отсчёта записаны на самом деле?',
        }
        for case, question in cases.items():
            with self.subTest(case=case):
                self.assertIsNone(self.decision(question))

    def test_router_specific_approval_intent_is_preserved(self):
        for topic in ('approval_initiator', 'approval_finance', 'approval_general'):
            with self.subTest(topic=topic):
                self.assertEqual(_topic({'topic': topic, 'text': 'Какой порядок предусмотрен?'}), topic)

    def test_regulation_extension_is_not_contract_identity(self):
        self.assertIsNone(self.decision('В регламенте редакции 2025 может ли согласующий продлить срок рассмотрения?'))
        # Without an edition the source really changes: the extension is new.
        self.assertEqual(self.decision('В регламенте может ли согласующий продлить срок рассмотрения?').reason,
                         'edition_period')

    def test_changed_approval_thresholds_still_need_period(self):
        for question in (
            'Нужно ли одобрение генерального директора для договора на 12 млн рублей?',
            'Нужно ли согласование финансового директора для договора на 2 млн рублей?',
            'Нужно ли одобрение генерального директора для договора на 12 млн рублей? В дополнение учитываем письмо К-12.',
        ):
            with self.subTest(question=question):
                self.assertEqual(self.decision(question).reason, 'edition_period')

    def test_address_and_combined_request_still_need_specification_date(self):
        for question in (
            'Какой адрес доставки по П-17/2024?',
            'По П-17/2024 товар отгрузили 10 октября 2024 года. Куда везти эту партию?',
            'По П-17/2024 когда привезти мебель и по какому адресу?',
        ):
            with self.subTest(question=question):
                self.assertEqual(self.decision(question).reason, 'specification_period')

    def test_uncertain_contract_identity_still_requires_clarification(self):
        for question in ('Когда платить по договору?', 'Когда платить по договору с Альфой?'):
            with self.subTest(question=question):
                self.assertEqual(self.decision(question).reason, 'contract_identity')
        self.assertEqual(self.decision('Когда платить по П-17/2024 с Альфа-Трейдинг?').reason,
                         'identity_conflict')

    def synthetic(self, *, changed_timing=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        files = {
            'contract.txt': 'ДОГОВОР ПОСТАВКИ № Ц-42/2038\nООО «Ладога» и ООО «Полюс» заключили договор.\n3.1. Товар поставляется в течение семи дней после спецификации.\n3.2. Адрес доставки: г. Казань, ул. Луговая, д. 6.\n',
            'amendment.txt': 'ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\nк договору поставки № Ц-42/2038\nИзложить пункт 3.2 договора в следующей редакции:\n3.2. Адрес доставки: г. Казань, ул. Морская, д. 9. '
                + ('Доставка производится в течение одиннадцати дней. ' if changed_timing else '')
                + '\nНовый адрес применяется к партиям, спецификации на которые подписаны после 12 сентября 2038 года.\n',
            'old.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «ПОЛЮС»\nДействует с 01.01.2038\nУтратил силу с 01.06.2039\n1.3. Инициатор согласования — руководитель подразделения.\n2.3. Договоры на сумму свыше 4 000 000 рублей согласуются с финансовым директором.\n5.4. Иностранные контрагенты проверяются юридическим отделом.\n',
            'new.txt': 'РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «ПОЛЮС»\nДействует с 01.06.2039\n1.3. Инициатор согласования — руководитель подразделения.\n2.3. Договоры на сумму свыше 6 000 000 рублей согласуются с финансовым директором.\n5.4. Иностранные контрагенты проверяются юридическим и финансовым отделами.\n',
            'letter.txt': 'ООО «ПОЛЮС»\nСлужба комплаенс\nИсх. № М-87 от 15.06.2039\nО санкционной проверке иностранных контрагентов\nС 21 июня 2039 года инициатор обеспечивает санкционную проверку иностранного контрагента.\nПриложение к настоящему письму. Чек-лист: проверить контрагента, его банк и юрисдикцию.\n',
        }
        for name, text in files.items():
            (directory / name).write_text(text, encoding='utf-8')
        return PreRetrievalClarifier(RuleRegistry(Corpus(directory)))

    def test_changed_clause_contents_determine_deadline_relevance(self):
        question = 'По Ц-42/2038 в какой срок привезти товар после спецификации?'
        address_only = self.synthetic()
        self.assertTrue(address_only.registry.amendments)
        self.assertIsNone(self.decision(question, address_only))
        address_and_timing = self.synthetic(changed_timing=True)
        self.assertEqual(self.decision(question, address_and_timing).reason, 'specification_period')

    def test_source_letter_identity_is_independent_of_preposition_and_number(self):
        clarifier = self.synthetic()
        self.assertTrue(clarifier.letter_identifiers)
        for wording in ('из письма М-87', 'приложенном к письму М-87', 'письма № М-87'):
            with self.subTest(wording=wording):
                self.assertIsNone(self.decision('Что проверяют по санкционному чек-листу ' + wording + '?', clarifier))
        # An absent identifier cannot select an existing letter by coincidence.
        self.assertEqual(self.decision('Что проверяют по санкционному чек-листу из письма М-870?', clarifier).reason,
                         'edition_period')


if __name__ == '__main__':
    unittest.main()
