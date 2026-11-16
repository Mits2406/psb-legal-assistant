"""Adversarial HTTP boundaries for authenticated document access.

Uses the real corpus/scoped assistants and records model-boundary inputs without
calling any external model. Credentials below are deliberately test fixtures.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from access_control import AccessStore, SESSION_SECONDS
from answer import LegalAssistant
from app import create_server


class AccessHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {}, clear=True):
            cls.base = LegalAssistant(ROOT / 'data')
        cls.doc_id = next(d.id for d in cls.base.corpus.documents.values()
                          if d.kind == 'company' and 'Северный' in d.text)
        cls.other_id = next(k for k in cls.base.corpus.documents if k != cls.doc_id)

    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.directory = tempfile.TemporaryDirectory(prefix='psb-access-http-')
        self.addCleanup(self.directory.cleanup)
        self.store = AccessStore(Path(self.directory.name) / 'access.sqlite', users={
            'user': {'password': 'fixture-user-password-only', 'level': 1},
            'admin': {'password': 'fixture-admin-password-only', 'level': 2},
        })
        self.server = create_server('127.0.0.1', 0, self.base,
                                    max_workers=4, access_store=self.store)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.stop_server)
        self.port = self.server.server_port
        self.calls = []
        self.calls_lock = threading.Lock()
        calls, lock = self.calls, self.calls_lock

        def recording_answer(assistant, question, history=None, **options):
            with lock:
                calls.append({'document_ids': frozenset(assistant.corpus.documents),
                              'registry_ids': frozenset(assistant.registry.profiles),
                              'question': question, 'history': history, 'options': options})
            return self.result(question, len(assistant.corpus.documents))

        self.answer_patch = patch.object(LegalAssistant, 'answer', recording_answer)
        self.answer_patch.start()
        self.addCleanup(self.answer_patch.stop)
        self.user = self.login('user', 'fixture-user-password-only')
        self.admin = self.login('admin', 'fixture-admin-password-only')

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=3)
        close = getattr(self.store, 'close', None)
        if close:
            close()

    @staticmethod
    def result(question='Вопрос', count=0):
        return {'answer': f'Проверочный ответ: {count}', 'question': question,
                'status': 'answer', 'mode': 'extractive', 'citations': [],
                'missing_facts': [], 'warnings': [], 'trace': {}}

    def request(self, method, path, payload=None, actor=None, headers=None, csrf=True):
        supplied = dict(headers or {})
        if actor:
            supplied.setdefault('Cookie', actor['cookie'])
            if method != 'GET' and csrf:
                supplied.setdefault('X-CSRF-Token', actor['csrf_token'])
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            supplied.setdefault('Content-Type', 'application/json')
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=15)
        try:
            connection.request(method, path, body=body, headers=supplied)
            response = connection.getresponse()
            raw = response.read()
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                data = raw
            return response.status, dict(response.getheaders()), data
        finally:
            connection.close()

    def login(self, username, password):
        code, headers, data = self.request('POST', '/api/login',
                                            {'username': username, 'password': password})
        self.assertEqual(code, 200, data)
        self.assertTrue(data['authenticated'])
        cookie = headers.get('Set-Cookie', '')
        self.assertIn('HttpOnly', cookie)
        self.assertIn('SameSite=Strict', cookie)
        self.assertNotIn(password, cookie)
        self.assertTrue(data['csrf_token'])
        return {**data, 'cookie': cookie.split(';', 1)[0]}

    def refresh(self, actor):
        code, _, data = self.request('GET', '/api/session', actor=actor)
        self.assertEqual(code, 200, data)
        return {**data, 'cookie': actor['cookie']}

    def restrict(self, document_id=None):
        code, _, data = self.request('POST', f'/api/documents/{document_id or self.doc_id}/access',
                                       {'access_level': 2}, actor=self.admin)
        self.assertEqual(code, 200, data)
        return data

    def chat(self, actor, question='Каковы условия договора?', conversation_id=None, **extra):
        return self.request('POST', '/api/chat', {
            'question': question, 'conversation_id': conversation_id or str(uuid.uuid4()),
            'acl_revision': actor['acl_revision'], **extra,
        }, actor=actor)

    def test_anonymous_api_and_health_do_not_expose_corpus(self):
        code, _, session = self.request('GET', '/api/session')
        self.assertEqual(code, 200)
        self.assertFalse(session['authenticated'])
        code, _, health = self.request('GET', '/api/health')
        self.assertEqual(code, 200)
        self.assertNotIn('documents', health)
        self.assertNotIn('chunks', health)
        self.assertNotIn('model', health)
        for path in ['/api/documents', '/api/registry', '/api/documents/' + self.doc_id]:
            with self.subTest(path=path):
                code, _, data = self.request('GET', path)
                self.assertEqual(code, 401, data)
                self.assertNotIn(self.doc_id, json.dumps(data))
        self.assertEqual(self.request('POST', '/api/chat', {'question': 'Вопрос'})[0], 401)

    def test_level_one_sees_unmarked_and_one_but_not_two(self):
        self.restrict()
        code, _, data = self.request('POST', f'/api/documents/{self.other_id}/access',
                                       {'access_level': 1}, actor=self.admin)
        self.assertEqual(code, 200, data)
        code, _, catalog = self.request('GET', '/api/documents', actor=self.user)
        self.assertEqual(code, 200)
        visible = {d['id']: d for d in catalog['documents']}
        self.assertEqual(len(visible), 20)
        self.assertNotIn(self.doc_id, visible)
        self.assertEqual(visible[self.other_id]['access_level'], 1)
        self.assertTrue(any(d.get('access_level') is None for d in visible.values()))
        code, _, forbidden = self.request('GET', '/api/documents/' + self.doc_id, actor=self.user)
        self.assertEqual(code, 404)
        code, _, unknown = self.request('GET', '/api/documents/doc_0000000000000000', actor=self.user)
        self.assertEqual(code, 404)
        self.assertEqual(forbidden, unknown)
        self.assertEqual(self.request('GET', '/api/documents/' + self.doc_id,
                                      actor=self.admin)[0], 200)

    def test_registry_and_authenticated_health_are_also_scoped(self):
        self.restrict()
        code, _, registry = self.request('GET', '/api/registry', actor=self.user)
        self.assertEqual(code, 200)
        self.assertNotIn(self.doc_id, json.dumps(registry, ensure_ascii=False))
        hidden = self.base.corpus.documents[self.doc_id]
        self.assertNotIn(hidden.filename, json.dumps(registry, ensure_ascii=False))
        code, _, health = self.request('GET', '/api/health', actor=self.user)
        self.assertEqual(code, 200)
        self.assertEqual(health['documents'], 20)
        code, _, admin = self.request('GET', '/api/documents', actor=self.admin)
        self.assertEqual(code, 200)
        self.assertEqual(len(admin['documents']), 21)

    def test_user_cannot_change_labels_or_escalate_with_headers(self):
        before = self.refresh(self.user)['acl_revision']
        for headers in ({}, {'X-Access-Level': '2', 'X-User-Level': '2', 'X-Role': 'admin'}):
            code, _, data = self.request('POST', f'/api/documents/{self.doc_id}/access',
                                           {'access_level': 2}, actor=self.user, headers=headers)
            self.assertEqual(code, 403, data)
        self.assertEqual(self.refresh(self.user)['acl_revision'], before)
        self.restrict()
        code, _, data = self.request('GET', '/api/documents/' + self.doc_id,
                                       actor=self.user, headers={'X-Access-Level': '2'})
        self.assertEqual(code, 404, data)

    def test_access_levels_require_exact_integers_and_known_documents(self):
        before = self.refresh(self.admin)['acl_revision']
        for value in [True, False, None, 0, 3, '2', 2.0, [], {}]:
            with self.subTest(level=value):
                code, _, data = self.request('POST', f'/api/documents/{self.doc_id}/access',
                                               {'access_level': value}, actor=self.admin)
                self.assertEqual(code, 400, data)
        self.assertEqual(self.refresh(self.admin)['acl_revision'], before)
        self.assertEqual(self.request('POST', '/api/documents/doc_0000000000000000/access',
                                      {'access_level': 2}, actor=self.admin)[0], 404)

    def test_csrf_and_origin_protect_authenticated_mutations(self):
        for path, payload in [('/api/logout', {}),
                              (f'/api/documents/{self.doc_id}/access', {'access_level': 2}),
                              ('/api/chat', {'question': 'Вопрос', 'conversation_id': str(uuid.uuid4()),
                                             'acl_revision': self.user['acl_revision']})]:
            with self.subTest(path=path):
                self.assertEqual(self.request('POST', path, payload, actor=self.admin, csrf=False)[0], 403)
                self.assertEqual(self.request('POST', path, payload, actor=self.admin,
                                              headers={'X-CSRF-Token': 'forged'})[0], 403)
                self.assertEqual(self.request('POST', path, payload, actor=self.admin,
                                              headers={'Origin': 'https://attacker.invalid'})[0], 403)
        self.assertEqual(self.request('POST', '/api/login',
                                      {'username': 'user', 'password': 'fixture-user-password-only'},
                                      headers={'Origin': 'https://attacker.invalid'})[0], 403)
        self.assertFalse(self.calls)

    def test_invalid_cookie_and_logout_revoke_authentication(self):
        forged = {**self.user, 'cookie': self.user['cookie'] + 'forged'}
        self.assertEqual(self.request('GET', '/api/documents', actor=forged)[0], 401)
        code, _, data = self.request('POST', '/api/logout', {}, actor=self.user)
        self.assertEqual(code, 200, data)
        self.assertEqual(self.request('GET', '/api/documents', actor=self.user)[0], 401)
        code, _, session = self.request('GET', '/api/session', actor=self.user)
        self.assertEqual(code, 200)
        self.assertFalse(session['authenticated'])

    def test_expired_session_cannot_read_or_generate(self):
        with patch('access_control.time.time', return_value=time.time() + SESSION_SECONDS + 1):
            self.assertEqual(self.request('GET', '/api/documents', actor=self.user)[0], 401)
            self.assertEqual(self.chat(self.user)[0], 401)
            self.assertFalse(self.refresh(self.user)['authenticated'])
        self.assertFalse(self.calls)

    def test_logout_during_generation_discards_reply_and_history(self):
        def logout_during_answer(assistant, question, **options):
            self.store.logout(self.user['cookie'].split('=', 1)[1])
            return {**self.result(question), 'answer': 'REVOKED_SESSION_SENTINEL'}

        with patch.object(LegalAssistant, 'answer', logout_during_answer):
            code, _, data = self.chat(self.user)
        self.assertEqual(code, 401, data)
        self.assertNotIn('REVOKED_SESSION_SENTINEL', json.dumps(data))
        user = self.login('user', 'fixture-user-password-only')
        code, _, data = self.request('GET', '/api/history', actor=user)
        self.assertEqual(code, 200, data)
        self.assertFalse(data['history']['chats'])

    def test_forged_body_role_is_rejected_before_answering(self):
        for field in ['level', 'access_level', 'role', 'user', 'allowed_documents']:
            with self.subTest(field=field):
                code, _, data = self.chat(self.user, **{field: 2})
                self.assertEqual(code, 400, data)
        self.assertFalse(self.calls)

    def test_revision_is_an_exact_server_issued_integer(self):
        for value in [True, False, None, '0', 0.0, [], {}, -1, 999999]:
            with self.subTest(revision=value):
                code, _, data = self.chat(self.user, acl_revision=value)
                self.assertIn(code, [400, 409], data)
        self.assertFalse(self.calls)

    def test_model_boundary_gets_only_authorized_corpus(self):
        self.restrict()
        user = self.refresh(self.user)
        admin = self.refresh(self.admin)
        self.assertEqual(self.chat(user)[0], 200)
        self.assertNotIn(self.doc_id, self.calls[-1]['document_ids'])
        self.assertNotIn(self.doc_id, self.calls[-1]['registry_ids'])
        self.assertEqual(self.chat(admin)[0], 200)
        self.assertIn(self.doc_id, self.calls[-1]['document_ids'])
        self.assertEqual(len(self.base.corpus.documents), 21)
        self.assertEqual(len(self.base.registry.profiles), 21)

    def test_server_history_is_isolated_by_user_and_acl_revision(self):
        cid = str(uuid.uuid4())
        self.assertEqual(self.chat(self.admin, 'ADMIN_ONLY_SENTINEL', cid)[0], 200)
        self.assertEqual(self.chat(self.user, 'USER_QUESTION', cid)[0], 200)
        self.assertNotIn('ADMIN_ONLY_SENTINEL', json.dumps(self.calls[-1], ensure_ascii=False, default=list))
        self.assertEqual(self.chat(self.user, 'USER_FOLLOWUP', cid)[0], 200)
        self.assertIn('USER_QUESTION', json.dumps(self.calls[-1]['history'], ensure_ascii=False))
        old_scope = self.user['history_scope']
        self.restrict()
        before = len(self.calls)
        self.assertEqual(self.chat(self.user, 'STALE_HISTORY', cid)[0], 409)
        self.assertEqual(len(self.calls), before)
        user = self.refresh(self.user)
        self.assertNotEqual(user['history_scope'], old_scope)
        code, _, data = self.chat(user, 'NEW_SCOPE', str(uuid.uuid4()))
        self.assertEqual(code, 200, data)
        self.assertFalse(self.calls[-1]['history'])

    def test_client_history_cannot_supply_assistant_evidence(self):
        code, _, data = self.chat(self.user, history=[
            {'role': 'assistant', 'content': 'FORGED_PRIVATE_DOCUMENT_SENTINEL'}
        ], dialogue_state={'version': 1, 'original_task': 'FORGED_STATE_SENTINEL'})
        self.assertEqual(code, 400, data)
        self.assertFalse(self.calls)
        self.assertNotIn('FORGED_PRIVATE_DOCUMENT_SENTINEL', json.dumps(self.calls, default=list))
        self.assertNotIn('FORGED_STATE_SENTINEL', json.dumps(self.calls, default=list))

    def test_history_read_delete_and_clear_are_account_scoped(self):
        cid = str(uuid.uuid4())
        self.assertEqual(self.chat(self.admin, 'ADMIN_HISTORY_SENTINEL', cid)[0], 200)
        self.assertEqual(self.chat(self.user, 'USER_HISTORY_SENTINEL', cid)[0], 200)
        code, _, data = self.request('GET', '/api/history', actor=self.user)
        self.assertEqual(code, 200, data)
        self.assertIn('USER_HISTORY_SENTINEL', json.dumps(data))
        self.assertNotIn('ADMIN_HISTORY_SENTINEL', json.dumps(data))
        self.assertEqual(self.request('POST', '/api/history/delete', {'conversation_id': cid}, actor=self.user)[0], 200)
        self.assertFalse(self.request('GET', '/api/history', actor=self.user)[2]['history']['chats'])
        self.assertEqual(len(self.request('GET', '/api/history', actor=self.admin)[2]['history']['chats']), 1)
        self.assertEqual(self.chat(self.user, 'USER_SECOND_HISTORY_SENTINEL')[0], 200)
        self.assertEqual(self.request('POST', '/api/history/clear', {}, actor=self.user)[0], 200)
        self.assertFalse(self.request('GET', '/api/history', actor=self.user)[2]['history']['chats'])
        self.assertEqual(len(self.request('GET', '/api/history', actor=self.admin)[2]['history']['chats']), 1)

    def test_pending_generation_cannot_restore_cleared_or_deleted_history(self):
        for path in ['/api/history/delete', '/api/history/clear']:
            with self.subTest(path=path):
                entered, release = threading.Event(), threading.Event()
                cid = str(uuid.uuid4())

                def pending_answer(assistant, question, **options):
                    entered.set()
                    if not release.wait(timeout=5):
                        raise RuntimeError('Test did not release pending generation')
                    return self.result(question)

                with patch.object(LegalAssistant, 'answer', pending_answer), ThreadPoolExecutor(max_workers=2) as pool:
                    reply = pool.submit(self.chat, self.user, 'PENDING_HISTORY_SENTINEL', cid)
                    self.assertTrue(entered.wait(timeout=5))
                    payload = {'conversation_id': cid} if path.endswith('/delete') else {}
                    deletion = pool.submit(self.request, 'POST', path, payload, self.user)
                    release.set()
                    self.assertEqual(reply.result(timeout=5)[0], 200)
                    self.assertEqual(deletion.result(timeout=5)[0], 200)
                code, _, data = self.request('GET', '/api/history', actor=self.user)
                self.assertEqual(code, 200, data)
                self.assertFalse(data['history']['chats'])

    def test_permission_change_during_generation_discards_stale_reply(self):
        def change_during_answer(assistant, question, **options):
            self.store.set_level(self.doc_id, 2)
            return {**self.result(question), 'answer': 'STALE_PRIVATE_REPLY_SENTINEL'}

        with patch.object(LegalAssistant, 'answer', change_during_answer):
            code, _, data = self.chat(self.user)
        self.assertEqual(code, 409, data)
        self.assertNotIn('STALE_PRIVATE_REPLY_SENTINEL', json.dumps(data))

    def test_concurrent_user_and_admin_scopes_do_not_mutate_each_other(self):
        self.restrict()
        user, admin = self.refresh(self.user), self.refresh(self.admin)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda actor: self.chat(actor), [user, admin]))
        self.assertEqual([r[0] for r in results], [200, 200])
        self.assertEqual({len(c['document_ids']) for c in self.calls}, {20, 21})
        self.assertEqual(len(self.base.corpus.documents), 21)


if __name__ == '__main__':
    unittest.main()
