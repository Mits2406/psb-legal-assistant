"""Citation presentation contracts; no model calls or answer substitution."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from answer_style import render_author_answer


def source(key, title, clause, text, doc='d1', **extra):
    return {'id': key, 'title': title, 'clause': clause, 'text': text, 'doc_id': doc, **extra}


def rendered(body, sources, ids):
    safe = {item['id']: item for item in sources}
    footer = 'Источники: ' + '; '.join('[' + key + '] ' + (safe[key].get('display_label') or
        safe[key]['title'] + ', ' + safe[key]['clause']) for key in ids) + '.'
    original = body + '\n\n' + footer
    return original, render_author_answer(original, sources, ids)


class AnswerStyleTests(unittest.TestCase):
    def test_contract_groups_clauses_without_changing_claims(self):
        sources = [source('S001', 'Договор аренды Л-52-2038 Кедр', '3.2', '3.2. Срок оплаты.'),
                   source('S002', 'Договор аренды Л-52-2038 Кедр', '3.1', '3.1. Размер платы.')]
        body = 'Плата — 271 000 рублей с НДС; срок — не позднее 8-го числа. [S001, S002]'
        _, answer = rendered(body, sources, ['S001', 'S002'])
        self.assertEqual(answer, 'Плата — 271 000 рублей с НДС; срок — не позднее 8-го числа '
                                '(Договор аренды № Л-52/2038, п. 3.1, 3.2).')

    def test_law_uses_article_not_inherited_subparagraph_or_truncated_quote(self):
        sources = [source('S001', 'ТК ст70 действующая', 'ст. 70, подп. 7', 'Обычная норма.',
                          display_label='ТК ст70 действующая, фрагмент «Обычная…»'),
                   source('S002', 'ТК ст70 действующая', 'ст. 70', 'Ещё условие.')]
        _, answer = rendered('Правило и его исключение. [S001] [S002]', sources, ['S001', 'S002'])
        self.assertEqual(answer, 'Правило и его исключение (ТК РФ, ст. 70).')
        self.assertNotIn('подп. 7', answer)

    def test_federal_law_label(self):
        items = [source('S001', '152ФЗ ст9 действующая', 'ст. 9, п. 2', '2. Согласие можно отозвать.')]
        _, answer = rendered('Условие. [S001]', items, ['S001'])
        self.assertEqual(answer, 'Условие (152-ФЗ, ст. 9).')

    def test_regulation_edition_comes_from_same_document_text(self):
        title = 'Регламент согласования договоров ред 2038'
        items = [source('S001', title, '2.3', '2.3. Нужное правило.'),
                 source('S002', title, 'текст', 'Редакция от 12.04.2038. Редакция от 08.02.2036 утрачивает силу.')]
        _, answer = rendered('Нужное правило. [S001]', items, ['S001'])
        self.assertEqual(answer, 'Нужное правило (Регламент согласования договоров (редакция от 12.04.2038), п. 2.3).')

    def test_different_documents_cannot_supply_missing_edition(self):
        items = [source('S001', 'Регламент ред 2038', '2.3', 'Правило.', doc='a'),
                 source('S002', 'Регламент ред 2038', 'текст', 'Редакция от 12.04.2038.', doc='b')]
        _, answer = rendered('Правило. [S001]', items, ['S001'])
        self.assertIn('Регламент ред 2038', answer)
        self.assertNotIn('12.04.2038', answer)

    def test_document_metadata_can_supply_exact_edition(self):
        items = [source('S001', 'Регламент ред 2038', '2.3', 'Правило.',
                        citation_metadata={'revision_date': '2038-04-12'})]
        _, answer = rendered('Правило. [S001]', items, ['S001'])
        self.assertIn('редакция от 12.04.2038', answer)

    def test_memo_identifier_is_taken_from_text(self):
        items = [source('S001', 'Служебная записка об архиве', 'текст',
                        'СЛУЖЕБНАЯ ЗАПИСКА № ЮО-83 от 16.06.2037\nО хранении.')]
        _, answer = rendered('Правило. [S001]', items, ['S001'])
        self.assertEqual(answer, 'Правило (Служебная записка № ЮО-83 от 16.06.2037).')

    def test_distinct_contracts_and_clauses_remain_distinct(self):
        items = [source('S001', 'Договор поставки Ж-21-2038 Берёза', '4.1', 'Первое правило.', doc='a'),
                 source('S002', 'Договор поставки Ж-22-2038 Берёза', '4.1', 'Второе правило.', doc='b')]
        _, answer = rendered('Первое правило. [S001]\n\nВторое правило. [S002]', items, ['S001', 'S002'])
        self.assertEqual(answer, 'Первое правило (Договор поставки № Ж-21/2038, п. 4.1).\n\n'
                                 'Второе правило (Договор поставки № Ж-22/2038, п. 4.1).')

    def test_unknown_or_malformed_ids_are_not_hidden(self):
        items = [source('S001', 'Учебный документ', '4.1', 'Условие.')]
        for body in ('Условие. [S999]', 'Условие. [S001 и S999]', 'Условие. [S001; мусор]',
                     'Условие. [S001] Необъяснённый S004.', 'Условие. [[S001]]'):
            with self.subTest(body=body):
                original, answer = rendered(body, items, ['S001'])
                self.assertEqual(answer, original)

    def test_changed_footer_is_not_removed(self):
        items = [source('S001', 'Учебный документ', '4.1', 'Условие.')]
        original, _ = rendered('Условие. [S001]', items, ['S001'])
        changed = original + '\nВажное исключение без ссылки.'
        self.assertEqual(render_author_answer(changed, items, ['S001']), changed)

    def test_uncited_abstention_and_substantive_brackets_stay_intact(self):
        refusal = 'В базе нет нужных документов; ответить по ним невозможно.'
        self.assertEqual(render_author_answer(refusal, [], []), refusal)
        items = [source('S001', 'Учебный документ', '4.1', 'Условие.')]
        _, answer = rendered('Размер [без НДС] составляет 700 рублей. [S001]', items, ['S001'])
        self.assertIn('[без НДС]', answer)

    def test_clause_order_is_numeric(self):
        items = [source('S001', 'Учебный документ', '4.10', 'Десятый пункт.'),
                 source('S002', 'Учебный документ', '4.2', 'Второй пункт.')]
        _, answer = rendered('Условия. [S001, S002]', items, ['S001', 'S002'])
        self.assertIn('п. 4.2, 4.10', answer)

    def test_only_citation_adjacent_whitespace_is_changed(self):
        items = [source('S001', 'Учебный документ', '4.1', 'Условие.')]
        body = '  Сумма:  7 000 рублей [S001], включая НДС.  \n\n  Срок:  8 дней.'
        _, answer = rendered(body, items, ['S001'])
        self.assertEqual(answer,
                         '  Сумма:  7 000 рублей (Учебный документ, п. 4.1), включая НДС.  \n\n  Срок:  8 дней.')

    def test_each_claim_keeps_its_own_clause_in_the_same_paragraph(self):
        items = [source('S001', 'Договор аренды Л-52-2038 Кедр', '3.1', 'Размер платы.'),
                 source('S002', 'Договор аренды Л-52-2038 Кедр', '3.2', 'Срок оплаты.')]
        _, answer = rendered('Плата — 271 000 рублей [S001]. Оплата до 8-го числа [S002].', items, ['S001', 'S002'])
        self.assertEqual(answer, 'Плата — 271 000 рублей (Договор аренды № Л-52/2038, п. 3.1). '
                                 'Оплата до 8-го числа (п. 3.2).')
        self.assertNotIn('Источники', answer)

    def test_generator_pipeline_preserves_sentence_level_mapping(self):
        from generation import GroundedGenerator
        items = [source('S001', 'Учебный договор', '3.1', 'Размер платы.'),
                 source('S002', 'Учебный договор', '3.2', 'Срок оплаты.')]
        safe = {item['id']: item for item in items}
        generator = GroundedGenerator(None)
        draft = 'Плата — 271 000 рублей [S001]. Оплата до 8-го числа [S002].'
        paragraphs, rejected = generator._parse(draft, safe)
        self.assertEqual(rejected, 0)
        self.assertEqual(paragraphs, [draft])
        canonical, used, omitted = generator._render(paragraphs, safe)
        self.assertEqual((used, omitted), (['S001', 'S002'], 0))
        answer = render_author_answer(canonical, items, used)
        self.assertEqual(answer, 'Плата — 271 000 рублей (Учебный договор, п. 3.1). '
                                 'Оплата до 8-го числа (п. 3.2).')

    def test_repeated_clause_is_not_removed_from_a_later_claim(self):
        items = [source('S001', 'Учебный договор', '4.1', 'Условие и его исключение.')]
        _, answer = rendered('Общее условие. [S001] Исключение. [S001]', items, ['S001'])
        self.assertEqual(answer, 'Общее условие (Учебный договор, п. 4.1). Исключение (п. 4.1).')

    def test_multiple_sources_support_only_the_claim_where_they_occur(self):
        items = [source('S001', 'Договор поставки Ж-21-2038 Берёза', '4.1', 'Первое правило.', doc='a'),
                 source('S002', 'Дополнительное соглашение № 3', '2.1', 'Исключение.', doc='b')]
        _, answer = rendered('Правило с исключением [S001, S002]. Другое условие [S001].', items, ['S001', 'S002'])
        self.assertEqual(answer, 'Правило с исключением (Договор поставки № Ж-21/2038, п. 4.1; '
                                 'Дополнительное соглашение № 3, п. 2.1). '
                                 'Другое условие (Договор поставки № Ж-21/2038, п. 4.1).')

    def test_editions_remain_named_and_switching_documents_restores_full_labels(self):
        items = [source('S001', 'Регламент ред 2036', '2.4', 'Старое правило.', doc='old',
                        citation_metadata={'revision_date': '2036-01-10'}),
                 source('S002', 'Регламент ред 2038', '2.4', 'Новое правило.', doc='new',
                        citation_metadata={'revision_date': '2038-03-01'})]
        _, answer = rendered('Раньше — 10 [S001]; сейчас — 15 [S002]; для старой даты — 10 [S001].',
                             items, ['S001', 'S002'])
        self.assertEqual(answer, 'Раньше — 10 (Регламент (редакция от 10.01.2036), п. 2.4); '
                                 'сейчас — 15 (Регламент (редакция от 01.03.2038), п. 2.4); '
                                 'для старой даты — 10 (Регламент (редакция от 10.01.2036), п. 2.4).')

    def test_steps_and_new_paragraphs_keep_their_own_document_reference(self):
        items = [source('S001', 'Учебный регламент', '1.1', 'Первый шаг.'),
                 source('S002', 'Учебный регламент', '1.2', 'Второй шаг.')]
        _, answer = rendered('1. Подготовьте документ [S001].\n\n2. Передайте документ [S002].', items, ['S001', 'S002'])
        self.assertEqual(answer, '1. Подготовьте документ (Учебный регламент, п. 1.1).\n\n'
                                 '2. Передайте документ (Учебный регламент, п. 1.2).')

    def test_abbreviations_decimal_values_and_ellipses_are_preserved(self):
        items = [source('S001', 'Учебный документ', '4.1', 'Условие.')]
        for before, expected in (
            ('Размер — 2.5 тыс. [S001]', 'Размер — 2.5 тыс. (Учебный документ, п. 4.1)'),
            ('Условие... [S001]', 'Условие... (Учебный документ, п. 4.1)'),
            ('Размер — 2.5 [S001].', 'Размер — 2.5 (Учебный документ, п. 4.1).'),
            ('«Условие». [S001]', '«Условие» (Учебный документ, п. 4.1).'),
        ):
            with self.subTest(before=before):
                _, answer = rendered(before, items, ['S001'])
                self.assertEqual(answer, expected)

    def test_same_normalized_contract_number_preserves_counterparty_identity(self):
        items = [source('S001', 'Договор поставки Ж-21-2038 Берёза', '4.1', 'Первое правило.', doc='a'),
                 source('S002', 'Договор поставки Ж-21-2038 Кедр', '4.1', 'Второе правило.', doc='b')]
        _, answer = rendered('Первое [S001]. Второе [S002].', items, ['S001', 'S002'])
        self.assertIn('(Договор поставки Ж-21-2038 Берёза, п. 4.1)', answer)
        self.assertIn('(Договор поставки Ж-21-2038 Кедр, п. 4.1)', answer)

    def test_indistinguishable_document_identities_fail_closed(self):
        items = [source('S001', 'Учебный документ', '4.1', 'Первое правило.', doc='a'),
                 source('S002', 'Учебный документ', '4.1', 'Второе правило.', doc='b')]
        original, answer = rendered('Первое [S001]. Второе [S002].', items, ['S001', 'S002'])
        self.assertEqual(answer, original)

    def test_invalid_source_types_fail_closed(self):
        items = [source('S001', 'Учебный документ', '4.1', 'Условие.')]
        original, _ = rendered('Условие. [S001]', items, ['S001'])
        for sources, ids in ((None, ['S001']), (items, [[]]),
                             ([dict(items[0], display_label=42)], ['S001'])):
            self.assertEqual(render_author_answer(original, sources, ids), original)


if __name__ == '__main__':
    unittest.main()
