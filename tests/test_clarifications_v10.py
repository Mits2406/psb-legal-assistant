"""Speech-act regression tests independent of actual contract wording."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator, _nonclaim


class ClarificationV10Tests(unittest.TestCase):
    def setUp(self):
        self.generator = GroundedGenerator(None)
        self.source = {'S001': {'id': 'S001', 'title': 'Учебное соглашение', 'clause': '8.2',
                               'text': 'Повышение допускается один раз в год после письменного уведомления.'}}
        self.answer = 'Повышение допускается один раз в год после письменного уведомления. [S001]'

    def test_new_factual_questions_and_document_requests_are_preserved_whole(self):
        paragraphs = [
            'Уточните: в какой день адресат получил уведомление и сохранилась ли квитанция о вручении?',
            'Уточните: кто подписал акт сверки? Пришлите экземпляр с подписями.',
            'Уточните: договор уже подписан или пока существует только проект, и какая дата стоит рядом с подписью?',
            'Уточните: какой адрес указан в подписанной сторонами заявке? Если её копии нет, укажите дату и номер заявки.',
            'Недостаточно данных: для определения гарантийного срока нужен текст приложения к договору Р-842/2037, включая документы изготовителя.',
            'Недостаточно данных: нужны сведения об отправке заявления и копия подтверждения его получения.',
            'Недостаточно данных: для установления даты нужна копия акта; для проверки адреса — подписанная заявка.',
        ]
        for paragraph in paragraphs:
            with self.subTest(paragraph=paragraph):
                self.assertTrue(_nonclaim(paragraph))
                kept, rejected = self.generator._parse(self.answer + '\n\n' + paragraph, self.source)
                self.assertEqual(rejected, 0)
                self.assertEqual(kept, [self.answer, paragraph])

    def test_unknown_contract_request_keeps_arbitrary_identity(self):
        paragraph = ('Недостаточно данных: чтобы проверить условия возврата по соглашению З-912/2039, '
                     'нужны его полный текст и соответствующие приложения.')
        kept, rejected = self.generator._parse(paragraph, self.source)
        self.assertEqual(rejected, 0)
        self.assertEqual(kept, [paragraph])
        self.assertIn('З-912/2039', kept[0])

    def test_question_mark_cannot_license_a_legal_rule(self):
        for paragraph in (
            'Уточните: почему арендодатель вправе повысить плату на 30%?',
            'Уточните: когда применяется ограничение повышения не более чем на 5%?',
            'Уточните: согласны ли вы, что штраф всегда ограничен 1%?',
            'Уточните: какой срок уведомления действует по закону?',
            'Уточните: когда акт подписан, поскольку уведомление не требуется?',
        ):
            with self.subTest(paragraph=paragraph):
                self.assertFalse(_nonclaim(paragraph))
                kept, rejected = self.generator._parse(self.answer + '\n\n' + paragraph, self.source)
                self.assertEqual(rejected, 1)
                self.assertNotIn(paragraph, kept)

    def test_substantive_second_sentence_or_tail_invalidates_whole_request(self):
        for paragraph in (
            'Уточните: кто подписал договор? Долг признан.',
            'Уточните: в какой день получено уведомление? Срок оплаты составляет 30 дней.',
            'Уточните: кто подписал договор, арендодатель свободен от любых ограничений.',
            'Недостаточно данных: нужен текст договора. Уведомление не требуется.',
            'Недостаточно данных: нужно оплатить долг в течение пяти дней.',
            'Недостаточно данных: для арендодателя — право повысить плату без ограничений.',
        ):
            with self.subTest(paragraph=paragraph):
                self.assertFalse(_nonclaim(paragraph))
                _, rejected = self.generator._parse(self.answer + '\n\n' + paragraph, self.source)
                self.assertEqual(rejected, 1)

    def test_legal_premise_before_question_or_after_imperative_is_not_a_request(self):
        for paragraph in (
            'Уточните: по закону повышение — 20%, когда вы заплатите?',
            'Уточните: штраф — 10% по договору, когда подписан акт?',
            'Уточните: предоставьте договор, штраф — 10%.',
            'Уточните: когда подписан акт, размер неустойки — 4%?',
            'Уточните: повышение — любая сумма, когда направили письмо?',
            'Уточните: по закону аренда возобновилась, когда вы освободите помещение?',
            'Недостаточно данных: нужен текст договора; штраф — 20%.',
        ):
            with self.subTest(paragraph=paragraph):
                self.assertFalse(_nonclaim(paragraph))
                _, rejected = self.generator._parse(self.answer + '\n\n' + paragraph, self.source)
                self.assertEqual(rejected, 1)

    def test_neutral_heading_does_not_remove_conditions_or_request(self):
        paragraph = 'Уточните: кто получил письмо и какой адрес доставки указан?'
        kept, rejected = self.generator._parse('## Что нужно уточнить\n\n' + self.answer + '\n\n' + paragraph,
                                               self.source)
        self.assertEqual(rejected, 0)
        self.assertEqual(kept, [self.answer, paragraph])

    def test_substantive_heading_is_not_silently_removed(self):
        for heading in ('**Ограничения не применяются**', '## Максимальное повышение — 30%'):
            kept, rejected = self.generator._parse(heading + '\n\n' + self.answer, self.source)
            self.assertEqual(rejected, 1)
            self.assertNotIn(heading, kept)

    def test_bad_reference_in_question_remains_a_structural_error(self):
        paragraph = 'Уточните: кто подписал акт? [S999]'
        _, rejected = self.generator._parse(self.answer + '\n\n' + paragraph, self.source)
        self.assertEqual(rejected, 1)


if __name__ == '__main__':
    unittest.main()
