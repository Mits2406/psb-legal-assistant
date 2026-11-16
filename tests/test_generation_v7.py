"""Synthetic generation regressions; no frozen question or candidate is loaded."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generation import GroundedGenerator
from llm import APIError


def source(sid,text,title='Учебный договор',clause='4.1',**kw):
    return {'id':sid,'title':title,'clause':clause,'text':text,**kw}


class Model:
    enabled=True
    def __init__(self,responses,finishes=None):
        self.responses=list(responses);self.finishes=list(finishes or [])
        self.calls=[];self.last_chat_diagnostics={}
    def chat_text(self,system,data,deadline,**kwargs):
        self.calls.append((system,data,deadline,kwargs))
        self.last_chat_diagnostics={'finish_reason':self.finishes.pop(0) if self.finishes else 'stop'}
        result=self.responses.pop(0)
        if isinstance(result,Exception):raise result
        return result


class GenerationV7Tests(unittest.TestCase):
    def setUp(self):
        # These regressions exercise ordinary source recovery. The diagnostic
        # default and its distinct failure markers have their own V10 suite.
        # V7 cases isolate the draft parser and fallback; V11 tests exercise
        # the now-enabled-by-default review as a separate stage.
        env=patch.dict(os.environ,{'DIAGNOSTIC_API_FAILURES':'0','ANSWER_VERIFY':'0'},clear=True);env.start();self.addCleanup(env.stop)
        blocker=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('No external API'))
        blocker.start();self.addCleanup(blocker.stop)
        self.rent=[source(f'S{i:03d}',f'{i}. Общие положения',content_role='heading') for i in range(1,10)]
        self.rent += [source('S010','4.1. Арендная плата составляет 275 000 рублей в месяц.',clause='4.1'),
                      source('S011','4.2. Арендная плата вносится не позднее 7-го числа текущего месяца.',clause='4.2')]

    def generate(self,responses,*,sources=None,question='Какова арендная плата и когда её вносить?',ids=None,finishes=None,context=None):
        self.model=Model(responses,finishes)
        return GroundedGenerator(self.model).generate(question,sources or self.rent,
            deadline=time.monotonic()+30,fallback_ids=ids,context=context)

    def test_default_one_normal_prose_call_and_deterministic_labels(self):
        result=self.generate(['Арендная плата — 275 000 рублей в месяц. [S010]\n\nВносится до 7-го числа текущего месяца. [S011]'])
        self.assertEqual(result['mode'],'llm')
        self.assertEqual(result['status'],'answer')
        self.assertEqual(len(self.model.calls),1)
        self.assertEqual(result['used_ids'],['S010','S011'])
        self.assertEqual(result['answer'].count('Учебный договор, 4.1'),1)
        self.assertFalse(result['trace']['semantic_audited'])
        self.assertNotIn('max_tokens',self.model.calls[0][3])

    def test_rent_fallback_prefers_s010_s011_over_first_nine_headings(self):
        result=self.generate([APIError('unavailable')],ids=['S010','S011'])
        self.assertEqual(result['mode'],'extractive')
        self.assertEqual(result['used_ids'],['S010','S011'])
        self.assertIn('275 000',result['answer'])
        self.assertIn('7-го числа',result['answer'])
        self.assertNotIn('Общие положения',result['answer'])

    def test_local_relevance_also_finds_late_rent_without_fallback_ids(self):
        result=self.generate([APIError('unavailable')])
        self.assertEqual(set(result['used_ids']),{'S010','S011'})
        self.assertNotIn('Общие положения',result['answer'])

    def test_absent_api_uses_same_relevance_pipeline(self):
        online_failed=self.generate([APIError('unavailable')],ids=['S010','S011'])
        model=Model([]);model.enabled=False
        offline=GroundedGenerator(model).generate('Какова арендная плата и когда её вносить?',self.rent,
            deadline=time.monotonic()+10,fallback_ids=['S010','S011'])
        self.assertEqual(offline['answer'],online_failed['answer'])
        self.assertEqual(offline['used_ids'],online_failed['used_ids'])
        self.assertEqual(model.calls,[])

    def test_497_character_paragraph_survives_without_point_cap(self):
        paragraph='Арендная плата определяется условиями договора. '+('Существенные условия сохраняются. '*20)
        paragraph=paragraph[:487].rstrip()+'. [S010]'
        paragraph+=' '*(497-len(paragraph))
        self.assertEqual(len(paragraph),497)
        result=self.generate([paragraph])
        self.assertEqual(result['mode'],'llm')
        self.assertIn(paragraph.strip(),result['answer'])
        self.assertEqual(result['trace']['discarded_paragraphs'],0)

    def test_audit_error_retains_usable_cited_draft(self):
        draft='Арендная плата — 275 000 рублей в месяц. [S010]'
        with patch.dict(os.environ,{'ANSWER_VERIFY':'1'}):
            result=self.generate([draft,APIError('private-key-must-not-appear')])
        self.assertEqual(result['mode'],'llm')
        self.assertIn(draft,result['answer'])
        self.assertEqual(len(self.model.calls),2)
        self.assertFalse(result['trace']['semantic_audited'])
        self.assertEqual(result['trace']['review_error'],'review_unavailable')
        self.assertNotIn('private-key',str(result))

    def test_optional_review_replaces_only_with_structurally_cited_text(self):
        with patch.dict(os.environ,{'ANSWER_VERIFY':'1'}):
            result=self.generate(['Арендная плата — 280 000 рублей. [S010]',
                                  'Арендная плата — 275 000 рублей в месяц. [S010]'])
        self.assertTrue(result['trace']['semantic_audited'])
        self.assertIn('275 000',result['answer'])
        self.assertNotIn('280 000',result['answer'])

    def test_invalid_review_does_not_erase_valid_draft(self):
        draft='Арендная плата — 275 000 рублей в месяц. [S010]'
        with patch.dict(os.environ,{'ANSWER_VERIFY':'1'}):
            result=self.generate([draft,'Совсем другая сумма. [S999]'])
        self.assertIn(draft,result['answer'])
        self.assertNotIn('S999',result['answer'])
        self.assertFalse(result['trace']['semantic_audited'])

    def test_truncated_reply_preserves_complete_cited_paragraph(self):
        draft='Арендная плата — 275 000 рублей в месяц. [S010]\n\nПорядок изменения арендной платы предполагает'
        result=self.generate([draft],finishes=['length'])
        self.assertEqual(result['mode'],'extractive')
        self.assertEqual(result['status'],'partial')
        self.assertIn('275 000',result['answer'])
        self.assertNotIn('предполагает',result['answer'])
        self.assertIn('S010',result['used_ids'])

    def test_unknown_citation_and_uncited_claim_are_not_published(self):
        result=self.generate(['Оплата производится до 7-го числа. [S011]\n\nИная норма. [S999]\n\nПроизвольное неподтверждённое утверждение.'])
        self.assertIn('7-го числа',result['answer'])
        self.assertNotIn('S999',result['answer'])
        self.assertNotIn('неподтверждённое утверждение',result['answer'])
        self.assertEqual(result['trace']['discarded_paragraphs'],2)

    def test_unknown_facts_can_remain_conditional_without_claiming_event(self):
        text='Если подписано уведомление, применяется новый порядок; иначе прежний. [S010]\n\nУточните: подписано ли уведомление?'
        result=self.generate([text])
        self.assertEqual(result['status'],'partial')
        self.assertIn('иначе прежний',result['answer'])
        self.assertEqual(result['missing_facts'],['подписано ли уведомление?'])

    def test_context_incomplete_does_not_add_automatic_rest_unconfirmed_text(self):
        draft='Арендная плата — 275 000 рублей в месяц. [S010]'
        result=self.generate([draft],context={'evidence_complete':False})
        self.assertEqual(result['status'],'partial')
        self.assertNotIn('Остальные части',result['answer'])
        self.assertIn(draft,result['answer'])

    def test_oversized_answer_keeps_only_whole_cited_paragraphs(self):
        sentence='Полное условие применяется при наличии согласованного документа. '
        paragraphs=[sentence*30+f'Положение номер {i}. [S010]' for i in range(8)]
        result=self.generate(['\n\n'.join(paragraphs)])
        self.assertLessEqual(len(result['answer']),10000)
        self.assertGreater(result['trace']['omitted_for_size'],0)
        self.assertEqual(result['mode'],'extractive')
        self.assertIn('275 000',result['answer'])
        self.assertNotIn('Положение номер',result['answer'])

    def test_fallback_does_not_cut_a_condition_bearing_chunk(self):
        text='Срок исполнения составляет 18 дней, за исключением согласованной отсрочки; отсрочка возможна только письменно.'
        result=self.generate([APIError('offline')],sources=[source('S001',text)],question='Каков срок исполнения?',ids=['S001'])
        self.assertIn('«'+text+'»',result['answer'])
        self.assertEqual(result['used_ids'],['S001'])

    def test_sources_without_relevance_do_not_produce_arbitrary_first_excerpt(self):
        result=self.generate([APIError('offline')],sources=[source('S001','Нейтральное описание имущества и технических характеристик.')],question='Какой срок оплаты?',ids=[])
        self.assertEqual(result['mode'],'clarify')
        self.assertEqual(result['used_ids'],[])

    def test_expired_deadline_is_reported_without_model_call(self):
        model=Model([])
        result=GroundedGenerator(model).generate('Какова арендная плата?',self.rent,
            deadline=time.monotonic()-1,fallback_ids=['S010'])
        self.assertEqual(result['trace']['calls'],0)
        self.assertEqual(result['trace']['stages'][0]['error_code'],'deadline_before_call')
        self.assertEqual(model.calls,[])

if __name__=='__main__':unittest.main()
