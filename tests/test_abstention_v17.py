"""Source limitations must survive parsing; legal claims still need evidence."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abstention import inspect_abstention, normalize_abstention
from generation import GroundedGenerator


class AbstentionTests(unittest.TestCase):
    def test_organizer_style_source_absence(self):
        text = ('В базе знаний нет документов о пособиях по беременности и родам, '
                'поэтому ответить по документам я не могу. По памяти отвечать не буду, '
                'так как нормы могли измениться. Для ответа обратитесь в отдел кадров или к юристу.')
        self.assertTrue(inspect_abstention(text))
        self.assertEqual(normalize_abstention(text), text.replace('В базе знаний', 'В предоставленных источниках'))

    def test_arbitrary_subjects_do_not_need_topic_rules(self):
        for topic in ('субсидиях на городской транспорт', 'порядке утилизации оборудования',
                      'расчете командировочных расходов', 'гарантии по изделию Р-731/2038'):
            with self.subTest(topic=topic):
                self.assertTrue(inspect_abstention('В предоставленных источниках нет сведений о ' + topic + '.'))

    def test_no_numeric_value_is_inferred_from_missing_evidence(self):
        for text in (
            'Недостаточно данных: в источниках не указан размер выплаты.',
            'Размер выплаты в предоставленных источниках не указан. Для ответа нужен применимый документ.',
            'В предоставленных документах отсутствуют правила расчёта. Поэтому не могу рассчитать сумму.',
            'В доступных фрагментах нет сведений о гарантии. Нужен гарантийный талон.',
            'Недостаточно данных: предоставьте текст договора К-31/2038.',
            'В источниках не найдено документов по этому вопросу. По документам ответить не могу.',
        ):
            with self.subTest(text=text):
                self.assertTrue(inspect_abstention(text))

    def test_operative_claims_cannot_hide_in_caution(self):
        for text in (
            'В источниках нет сведений о штрафе, поэтому штраф не применяется.',
            'Недостаточно данных: в источниках не указан размер выплаты. Пособие положено всем.',
            'В базе нет документов, поэтому можно не платить.',
            'В источниках нет сведений, значит согласование не требуется.',
            'Недостаточно данных: штраф — 10%.',
            'В документах нет сведений о том, что работник обязан вернуть выплату.',
            'В источниках нет сведений о выплате 10% зарплаты.',
            'Недостаточно данных: нужен документ. Уведомление не требуется.',
            'В базе нет документов. Перечислите деньги контрагенту.',
            'Недостаточно данных: предоставьте договор и согласуйте его с нотариусом.',
            'В источниках нет сведений о договоре. Договор считается заключенным.',
            'В источниках нет сведений о выплате; деньги перечислены.',
            'В источниках нет сведений о налоге (ставка равна нулю).',
            'В источниках нет сведений о выплате. Для ответа обратитесь к юристу и оплатите долг.',
            'В источниках нет документов о выплате, договор недействителен.',
            'В источниках нет документов о выплате — штраф нулевой.',
            'В источниках нет сведений о выплате и договор недействителен.',
            'В источниках нет сведений о выплате и налог нулевой.',
            'Не могу определить размер выплаты, вы платите налог.',
            'В источниках нет документов о выплате: налог нулевой.',
        ):
            with self.subTest(text=text):
                self.assertFalse(inspect_abstention(text))
                self.assertIsNone(normalize_abstention(text))

    def test_mixed_cited_answer_is_not_a_whole_uncited_abstention(self):
        self.assertFalse(inspect_abstention('В источниках нет сведений о выплате. [S001]'))
        self.assertFalse(inspect_abstention('В источниках нет сведений о выплате. [S999]'))
        self.assertFalse(inspect_abstention('В источниках нет сведений о выплате. https://example.com'))

    def test_truncated_or_empty_absence_is_not_accepted(self):
        for text in ('', None, 'Недостаточно данных:', 'В предоставленных источниках',
                     'В базе нет', 'Нужен документ.', 'Уточните: когда подписан договор?'):
            self.assertFalse(inspect_abstention(text))

    def test_scope_is_only_widened_with_explicit_complete_corpus_flag(self):
        text = 'В предоставленной базе знаний нет документов по этому вопросу.'
        self.assertEqual(normalize_abstention(text, complete_corpus=True), text)
        self.assertEqual(normalize_abstention(text), 'В предоставленных источниках нет документов по этому вопросу.')

    def test_captured_live_refusals_with_unknown_entitlement(self):
        # Verbatim draft/review from work/author_style_v17/refusal_capture.json.
        texts = (
            'В предоставленных источниках нет сведений о размере страховой пенсии по старости, поэтому определить, какая сумма вам положена, по этим документам нельзя. Для ответа нужны применимые пенсионные нормы и данные о вашем пенсионном стаже, пенсионных коэффициентах и обстоятельствах назначения пенсии.',
            'В предоставленных источниках нет сведений о размере страховой пенсии по старости, поэтому определить положенную вам сумму по этим документам нельзя. Для ответа нужны применимые пенсионные нормы и данные о вашем пенсионном стаже и пенсионных коэффициентах.',
        )
        for text in texts:
            with self.subTest(text=text):
                self.assertTrue(inspect_abstention(text))
                self.assertEqual(normalize_abstention(text), text)

    def test_unknown_entitlement_is_not_specific_to_one_benefit(self):
        texts = (
            'В предоставленных источниках нет сведений о размере страхового возмещения, поэтому определить, какая сумма вам положена, по этим документам нельзя. Для ответа нужны применимые страховые нормы и данные о вашем страховом случае, понесенных расходах и условиях договора.',
            'В предоставленных источниках нет сведений о размере компенсации за расходы, поэтому определить положенную вам сумму по этим документам нельзя. Для ответа нужны применимые нормы и данные о понесенных расходах, подтверждающих документах и условиях договора.',
        )
        for text in texts:
            with self.subTest(text=text):
                self.assertTrue(inspect_abstention(text))

    def test_unknown_entitlement_and_nominal_lists_do_not_hide_legal_assertions(self):
        for text in (
            'В источниках нет сведений о выплате. Сумма вам положена.',
            'В источниках нет сведений о выплате. Определить положенную вам сумму по этим документам нельзя, но пособие положено всем.',
            'Определить положенную вам сумму по этим документам нельзя. Согласование не требуется.',
            'В источниках нет сведений о выплате. Для ответа нужны нормы и данные, договор действителен.',
            'В источниках нет сведений о выплате. Для ответа нужны нормы и данные, санкции исключены.',
            'В источниках нет сведений о выплате. Для ответа нужны нормы и данные, суммы поступили.',
            'Определить положенную вам сумму в размере 10% по этим документам нельзя.',
        ):
            with self.subTest(text=text):
                self.assertFalse(inspect_abstention(text))


class AbstentionGeneratorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.generator = GroundedGenerator(None)
        self.safe = {
            'S001': {'id': 'S001', 'title': 'Договор поставки', 'clause': 'п. 1.1',
                     'text': 'Поставщик передает покупателю товар по спецификации.'},
        }

    def test_natural_refusal_keeps_separate_referral_without_irrelevant_citation(self):
        text = ('В базе знаний нет документов о пособиях по беременности и родам, '
                'поэтому ответить по документам я не могу. '
                'По памяти отвечать не буду, так как нормы могли измениться.\n\n'
                'Для ответа обратитесь в отдел кадров или к юристу.')
        for normalize_caution in (True, False):
            with self.subTest(normalize_caution=normalize_caution):
                paragraphs, rejected = self.generator._parse(
                    text, self.safe, normalize_caution=normalize_caution)
                self.assertEqual(rejected, 0)
                rendered, used, omitted = self.generator._render(paragraphs, self.safe)
                self.assertEqual(rendered, text.replace('В базе знаний', 'В предоставленных источниках'))
                self.assertEqual(used, [])
                self.assertEqual(omitted, 0)
                self.assertNotIn('[S001]', rendered)

    def test_refusal_does_not_admit_uncited_legal_claim(self):
        limitation = 'В предоставленных источниках нет сведений о выплате.'
        for claim in ('Пособие положено всем.', 'Согласование не требуется.', 'Штраф составляет 10%.'):
            for separator in (' ', '\n\n'):
                with self.subTest(claim=claim, separator=separator):
                    paragraphs, rejected = self.generator._parse(
                        limitation + separator + claim, self.safe, normalize_caution=False)
                    self.assertGreater(rejected, 0)
                    self.assertNotIn(claim, '\n\n'.join(paragraphs))

    def test_cited_refusal_loses_only_valid_known_irrelevant_markers(self):
        safe = {**self.safe,
                'S002': {**self.safe['S001'], 'id': 'S002'},
                'S003': {**self.safe['S001'], 'id': 'S003'}}
        limitation = 'В базе знаний нет документов о пособиях по беременности и родам.'
        referral = 'Для ответа обратитесь в отдел кадров или к юристу.'
        expected = limitation.replace('В базе знаний', 'В предоставленных источниках') + '\n\n' + referral
        for markers in ('[S001,S002,S003]', '[ S001; S002; S003 ]', '[S001] [S002] [S003]'):
            with self.subTest(markers=markers):
                paragraphs, rejected = self.generator._parse(
                    limitation + ' ' + markers + '\n\n' + referral + ' [S002]',
                    safe, normalize_caution=False)
                self.assertEqual(rejected, 0)
                rendered, used, omitted = self.generator._render(paragraphs, safe)
                self.assertEqual(rendered, expected)
                self.assertEqual((used, omitted), ([], 0))

    def test_refusal_foreign_or_malformed_markers_remain_rejected(self):
        text = 'В предоставленных источниках нет документов по этому вопросу.'
        for marker in ('[S999]', '[S001,S999]', '[S001,]', '[S000001]', '[S001 extra]',
                       '[S001](https://example.com)'):
            with self.subTest(marker=marker):
                paragraphs, rejected = self.generator._parse(
                    text + ' ' + marker, self.safe, normalize_caution=False)
                self.assertGreater(rejected, 0)
                self.assertEqual(paragraphs, [])

    def test_truncated_refusal_is_not_repaired_into_valid_answer(self):
        text = 'В предоставленных источниках нет документов по этому вопросу.'
        paragraphs, rejected = self.generator._parse(text, self.safe, truncated=True)
        self.assertEqual(paragraphs, [])
        self.assertGreater(rejected, 0)
        paragraphs, rejected = self.generator._parse(
            'В предоставленных источниках нет документов, поэтому', self.safe)
        self.assertEqual(paragraphs, [])
        self.assertGreater(rejected, 0)


if __name__ == '__main__':
    unittest.main()
