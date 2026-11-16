import sys,time,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from answer import LegalAssistant
from generation import GroundedGenerator

class CapturingClient:
    enabled=True
    last_chat_diagnostics={'finish_reason':'stop'}
    def chat_text(self,system,data,deadline):
        self.payload=data
        return 'Условие подтверждается приведённым фрагментом. [S001]'

class CitationTests(unittest.TestCase):
    def test_law_caption_does_not_relabel_unnumbered_paragraph_as_previous_list_item(self):
        a=LegalAssistant(Path(__file__).resolve().parents[1]/'data')
        c=next(c for c in a.corpus.chunks.values() if 'Продолжительность сверхурочной работы' in c.text)
        source={'id':'S001','doc_id':c.doc_id,'chunk_id':c.id,'text':c.text,'title':c.title,'clause':c.clause}
        checked=a._checked_sources([source])[0]
        self.assertIn(c.title,checked['display_label'])
        self.assertNotIn('подп.',checked['display_label'])
        self.assertIn(c.text.strip()[:45],checked['display_label'])
        self.assertEqual(checked['clause'],c.clause) # Retrieval/prompt semantics unchanged.

    def test_independent_letter_question_survives_pronoun_expansion(self):
        a=LegalAssistant(Path(__file__).resolve().parents[1]/'data')
        a.client.enabled=False
        with patch.dict('os.environ', {'DIAGNOSTIC_API_FAILURES':'0'}):
            answer=a.answer('Когда платить за мебель по П-17/2024 и нужно ли при санкционной проверке иностранного контрагента проверять его банк по письму К-12?')
        titles=[c['title'] for c in answer['citations']]
        self.assertTrue(any('П-17' in t for t in titles))
        self.assertTrue(any('Письмо' in t and 'К-12' in t for t in titles))
        self.assertEqual(answer['mode'],'extractive')

    def test_caption_fix_does_not_change_model_input_or_answer_body(self):
        source={'id':'S001','title':'ТК ст99','clause':'ст. 99, подп. 3','text':'Продолжительность сверхурочной работы ограничена.'}
        first=CapturingClient();second=CapturingClient()
        a=GroundedGenerator(first).generate('Какое условие?',[source],deadline=time.monotonic()+10)
        b=GroundedGenerator(second).generate('Какое условие?',[{**source,'display_label':'ТК ст99, фрагмент «Продолжительность сверхурочной работы…»'}],deadline=time.monotonic()+10)
        self.assertEqual(first.payload,second.payload)
        self.assertEqual(a['answer'].split('Источники:')[0],b['answer'].split('Источники:')[0])
        self.assertNotIn('подп. 3',b['answer'])

if __name__=='__main__':unittest.main()
