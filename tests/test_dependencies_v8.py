"""Recovery groups preserve restrictions without swallowing independent topics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from answer import LegalAssistant
from generation import GroundedGenerator


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = LegalAssistant(Path(__file__).resolve().parents[1] / 'data')
        chunks = [c for c in cls.assistant.corpus.chunks.values()
                  if cls.assistant.corpus.get_document(c.doc_id).kind == 'company'
                  and c.searchable and not c.suspicious
                  and c.content_role not in {'metadata', 'editorial'}]
        sources = [dict(id=f'S{i:03d}', chunk_id=c.id, doc_id=c.doc_id,
                        title=c.title, clause=c.clause, text=c.text)
                   for i, c in enumerate(chunks, 1)]
        cls.safe = {s['id']: s for s in cls.assistant._checked_sources(sources)}

    def group(self, title, clause):
        source = next(s for s in self.safe.values() if title in s['title'] and s['clause'] == clause)
        return [self.safe[sid] for sid in GroundedGenerator._source_group(source['id'], self.safe)]

    def test_payment_does_not_pull_delivery_amendment_but_delivery_does(self):
        payment = self.group('Договор поставки П-17', '4.2')
        self.assertTrue(any(s['clause'] == '4.3' for s in payment))
        self.assertFalse(any('Доп соглашение' in s['title'] for s in payment))
        delivery = self.group('Договор поставки П-17', '3.2')
        self.assertTrue(any('после 1 октября 2024' in s['text'] for s in delivery))
        self.assertTrue(any('Химки' in s['text'] for s in delivery))

    def test_rent_and_liability_keep_section_limits_and_explicit_cross_reference(self):
        rent = self.group('Договор аренды', '3.1')
        self.assertTrue(any(s['clause'] == '3.4' and '7%' in s['text'] for s in rent))
        liability = self.group('Договор поставки П-17', '7.1')
        self.assertTrue(any(s['clause'] == '7.5' and '50%' in s['text'] for s in liability))
        self.assertTrue(any(s['clause'] == '2.2' and '8 400 000' in s['text'] for s in liability))


if __name__ == '__main__':
    unittest.main()
