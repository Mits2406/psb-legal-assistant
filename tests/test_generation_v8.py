"""Final-answer degradation regressions; synthetic sources, no model network."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator
from llm import APIError


def src(key, text, clause='3.4', **extra):
    return dict(id=key, text=text, title='Учебный договор аренды', clause=clause,
                doc_id='rent', **extra)


RULE = '3.4. Арендодатель вправе повысить арендную плату.'
BOUNDS = 'Повышение допускается не чаще одного раза в год, максимум на 7%, с письменным уведомлением арендатора.'


class Client:
    enabled = True
    def __init__(self, responses, finishes=()):
        self.responses = list(responses)
        self.finishes = list(finishes)
        self.calls = 0
        self.last_chat_diagnostics = {}
    def chat_text(self, system, data, deadline):
        self.calls += 1
        self.last_chat_diagnostics = {'finish_reason': self.finishes.pop(0) if self.finishes else 'stop'}
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class DegradationV8Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.sources = [src('S001', RULE + ' ' + BOUNDS),
                        src('S002', '3.1. Плата составляет 123 000 рублей в месяц.', clause='3.1')]

    def run_answer(self, draft, *, sources=None, finishes=(), verify=False, fallback_ids=None):
        self.client = Client(draft, finishes)
        with patch.dict(os.environ, {'ANSWER_VERIFY': '1' if verify else '0'}):
            return GroundedGenerator(self.client).generate('Как изменяется арендная плата?',
                sources or self.sources, deadline=time.monotonic()+30, fallback_ids=fallback_ids)

    def assert_constraints(self, result):
        self.assertIn(BOUNDS, result['answer'])
        self.assertNotIn('S999', result['answer'])
        self.assertLessEqual(len(result['answer']), 10000)
        self.assertFalse(result['trace']['semantic_audited'])

    def test_invalid_restriction_restores_complete_source(self):
        result = self.run_answer(['Арендодатель вправе повысить плату. [S001]\n\nОграничения сохраняются. [S999]'])
        self.assert_constraints(result)
        self.assertEqual(result['mode'], 'extractive')
        self.assertEqual(self.client.calls, 1)

    def test_length_cutoff_never_leaves_generalized_rule(self):
        result = self.run_answer(['Арендодатель вправе повысить плату. [S001]\n\nНо повышение возможно только'], finishes=['length'])
        self.assert_constraints(result)
        self.assertNotIn('Но повышение возможно только', result['answer'])

    def test_valid_ids_in_rejected_paragraph_prioritize_repair(self):
        result = self.run_answer(['Оплата регулируется договором. [S002]\n\nПовышение ограничено. [S001, S999]'])
        self.assert_constraints(result)
        self.assertIn('123 000', result['answer'])

    def test_same_clause_pieces_are_atomic_even_without_restriction_citation(self):
        sources = [src('S001', RULE), src('S003', BOUNDS), self.sources[1]]
        result = self.run_answer(['Арендодатель вправе повысить плату. [S001]\n\nОграничения. [S999]'], sources=sources)
        self.assert_constraints(result)
        self.assertIn('S003', result['used_ids'])

    def test_explicit_cross_clause_dependency_is_atomic(self):
        sources = [src('S001', RULE, dependency_ids=['S003']), src('S003', BOUNDS, clause='3.5')]
        result = self.run_answer(['Общее право. [S001]\n\nУсловие'], sources=sources, finishes=['length'])
        self.assert_constraints(result)

    def test_render_overflow_replaces_all_paraphrases(self):
        draft = 'Общее право повысить плату. [S001]\n\n' + ('Условие соблюдается. ' * 600) + '[S001]'
        result = self.run_answer([draft])
        self.assert_constraints(result)
        self.assertEqual(result['trace']['fallback_reason'], 'answer_limit')
        self.assertNotIn('Общее право', result['answer'])

    def test_oversize_dependency_group_does_not_leave_broad_rule(self):
        sources = [src('S001', RULE), src('S003', BOUNDS + ' Полный текст.' * 1000), self.sources[1]]
        result = self.run_answer(['Общее право. [S001]\n\nУсловие'], sources=sources, finishes=['length'], fallback_ids=['S002'])
        self.assertNotIn(RULE, result['answer'])
        self.assertNotIn('Общее право', result['answer'])
        self.assertIn('Полное связанное положение не помещается', result['answer'])
        self.assertIn('123 000', result['answer'])
        self.assertLessEqual(len(result['answer']), 10000)
        self.assertLessEqual(len(result['answer'].encode('utf-8')), 40000)

    def test_corrupt_review_keeps_intact_original_with_conditions(self):
        draft = RULE + ' ' + BOUNDS + ' [S001]'
        result = self.run_answer([draft, 'Общее право. [S001]\n\nОграничения. [S999]'], verify=True)
        self.assert_constraints(result)
        self.assertEqual(result['mode'], 'llm')
        self.assertIn(draft, result['answer'])

    def test_corrupt_original_not_salvaged_by_failed_audit(self):
        result = self.run_answer(['Общее право. [S001]\n\nОграничения. [S999]',
                                  APIError('controlled review outage', code='timeout')], verify=True)
        self.assert_constraints(result)
        self.assertEqual(self.client.calls, 2)
        self.assertEqual(result['trace']['review_status'], 'recovery_after_review')

    def test_nonclaim_prefix_cannot_smuggle_legal_assertion(self):
        result = self.run_answer(['Недостаточно данных: поскольку арендодатель вправе повысить плату на 20% без уведомления.'])
        self.assertNotIn('20%', result['answer'])
        self.assertNotIn('без уведомления', result['answer'])
        self.assertEqual(result['mode'], 'abstain')
        self.assertIn('исходные факты', result['answer'])

    def test_uncited_condition_prefix_cannot_leave_cited_rule_unqualified(self):
        result = self.run_answer(['Общее право повысить плату. [S001]\n\nНедостаточно данных: однако повышение ограничено 7%.'])
        self.assert_constraints(result)
        self.assertNotIn('Общее право', result['answer'])

    def test_standalone_factual_clarification_keeps_specific_request(self):
        result = self.run_answer(['Уточните: в какие даты произошли передача документов и последующая оплата?'])
        self.assertEqual(result['mode'], 'clarify')
        self.assertEqual(result['answer'], 'Уточните: в какие даты произошли передача документов и последующая оплата?')

    def test_honest_short_clarification_survives(self):
        result = self.run_answer(['Уточните: подписано ли уведомление?'])
        self.assertEqual(result['mode'], 'clarify')
        self.assertIn('подписано ли уведомление?', result['answer'])

    def test_valid_id_is_not_claimed_as_semantic_proof(self):
        result = self.run_answer(['Плата составляет 999 рублей. [S002]'])
        self.assertFalse(result['trace']['semantic_audited'])
        self.assertTrue(result['trace']['structural_only'])


if __name__ == '__main__':
    unittest.main()
