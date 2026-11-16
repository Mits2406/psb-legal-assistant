import json, tempfile, time, unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution import run_batch, load_questions
from llm import ModelClient, APIError
from unittest.mock import patch

class RuntimeTests(unittest.TestCase):
    def test_failed_generation_is_observable_even_with_usable_excerpts(self):
        class Assistant:
            def answer(self, q, deadline):
                return {'answer':'3.2. Оплатить до 5-го числа. [Договор, 3.2]', 'status':'partial',
                        'mode':'extractive', 'trace':{'generation':{'fallback_reason':'generation_failed',
                        'stages':[{'stage':'draft','error_code':'transport_error'}]}}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            report=run_batch([('q','Оплата?')],Assistant(),p/'submission.csv',trace_path=p/'trace.jsonl')
            self.assertEqual(report['answer_modes'],{'extractive':1})
            self.assertEqual(report['failure_reasons']['transport_error'],1)
            self.assertEqual(json.loads((p/'trace.jsonl').read_text())['id'],'q')
            self.assertIn('до 5-го',(p/'submission.csv').read_text())

    def test_completed_text_at_token_limit_is_not_discarded(self):
        client=ModelClient()
        client.post=lambda *a,**kw:{'choices':[{'finish_reason':'length','message':{'content':'Оплата в течение 21 дня. [S001]','reasoning_content':'hidden'}}], 'usage':{'completion_tokens':1024}}
        text=client.chat_text('prompt',{})
        self.assertIn('21 дня',text)
        self.assertEqual(client.last_chat_diagnostics['finish_reason'],'length')
        self.assertTrue(client.last_chat_diagnostics['reasoning_present'])
        self.assertNotIn('hidden',json.dumps(client.last_chat_diagnostics))

    def test_failed_call_diagnostics_do_not_leak_content(self):
        client=ModelClient()
        def fail(*a,**kw): raise APIError('secret-containing external error',code='deadline')
        client.post=fail
        with self.assertRaises(APIError): client.chat_text('private prompt',{})
        d=client.last_chat_diagnostics
        self.assertEqual(d['error_code'],'deadline')
        self.assertNotIn('secret',json.dumps(d))

    def test_final_csv_preserves_arbitrary_ids_and_order(self):
        class Assistant:
            def answer(self,q,deadline): return {'answer':q,'status':'answer','mode':'llm','trace':{}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out.csv'
            run_batch([('001','Первый'),('id-x','Второй')],Assistant(),p)
            import csv
            with p.open() as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['id'] for r in rows],['001','id-x'])
            self.assertEqual([r['answer'] for r in rows],['Первый','Второй'])

if __name__=='__main__': unittest.main()
