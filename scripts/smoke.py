"""Offline HTTP + CSV acceptance check. Does not call or submit to any model API.

Run with: python scripts/smoke.py
Or: docker compose run --rm web python scripts/smoke.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def offline_environment() -> dict[str, str]:
    # Explicit blanks also prevent load_env() from importing local API secrets.
    env = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
    for key in ('LLM_BASE_URL', 'OPENAI_BASE_URL', 'BASE_URL', 'LLM_API_KEY',
                'OPENAI_API_KEY', 'LLM_MODEL', 'OPENAI_MODEL', 'GEN_MODEL',
                'RERANKER_MODEL', 'RERANKER_API_KEY', 'EMBEDDER_API_KEY',
                'EMB_API_KEY', 'EMB_KEY', 'RETRIEVAL_CACHE_DIR', 'REFERENCE_DATE'):
        env[key] = ''
    env.update(DATA_DIR=str(ROOT / 'data'), ENABLE_EMBEDDINGS='0', ENABLE_RERANKER='0',
               QUERY_NORMALIZE='0', DIAGNOSTIC_API_FAILURES='0', MAX_WORKERS='2',
               PYTHONDONTWRITEBYTECODE='1', RUN_TIMEOUT_SECONDS='60')
    return env


def check() -> dict:
    from access_control import AccessStore
    from answer import LegalAssistant
    from app import create_server

    env = offline_environment()
    with tempfile.TemporaryDirectory(prefix='psb-web-smoke-') as state_dir, patch.dict(os.environ, env, clear=True):
        assistant = LegalAssistant(data_dir=ROOT / 'data')
        assert not assistant.client.enabled, 'Smoke must not use an external model'
        access = AccessStore(Path(state_dir) / 'access.sqlite', users={
            'user': {'password': 'fixture-smoke-user-password', 'level': 1},
        })
        server = create_server('127.0.0.1', 0, assistant, access_store=access)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        authentication = {}

        def request(path, payload=None):
            body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
            headers = {'Content-Type': 'application/json'} if body else {}
            if authentication:
                headers['Cookie'] = authentication['cookie']
                if body is not None:
                    headers['X-CSRF-Token'] = authentication['csrf_token']
            req = urllib.request.Request(base + path, data=body, headers=headers)
            with opener.open(req, timeout=15) as response:
                assert response.status == 200
                return response.read(), response.headers

        def api(path, payload=None):
            return json.loads(request(path, payload)[0])

        try:
            html, headers = request('/')
            assert headers.get_content_type() == 'text/html' and b'<html' in html.lower()
            for asset in ('app.js', 'history.js', 'styles.css'):
                assert request('/static/' + asset)[0]
            public_health = api('/api/health')
            assert public_health['authentication_required'] and 'documents' not in public_health
            assert api('/api/session') == {'authenticated': False}
            raw_session, login_headers = request('/api/login', {
                'username': 'user', 'password': 'fixture-smoke-user-password',
            })
            authentication.update(json.loads(raw_session))
            authentication['cookie'] = login_headers['Set-Cookie'].split(';', 1)[0]
            assert authentication['authenticated'] and authentication['user']['level'] == 1
            health = api('/api/health')
            assert health['mode'] == 'extractive' and health['documents'] == 21
            documents = api('/api/documents')['documents']
            assert len(documents) == 21
            document = api('/api/documents/' + documents[0]['id'])
            assert document['text'] and document['chunks']
            for chunk in document['chunks']:
                assert document['text'][chunk['start']:chunk['end']] == chunk['text']

            question = 'Нужно ли одобрение до 12 млн рублей?'
            conversation = {'conversation_id': 'smoke-approval', 'acl_revision': authentication['acl_revision']}
            clarification = api('/api/chat', {'question': question, **conversation})
            assert clarification['status'] == 'clarify'
            continued = api('/api/chat', {
                'question': '2024', **conversation,
            })
            assert continued['answer'] and continued['status'] != 'clarify'
            assert continued.get('citations'), 'Follow-up should expose source fragments'
            history = api('/api/history')['history']['chats']
            assert len(history) == 1 and len(history[0]['turns']) == 2
            assert history[0]['turns'][0]['question'] == question
            assert api('/api/logout', {}) == {'authenticated': False}
            authentication.clear()
            assert api('/api/session') == {'authenticated': False}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            access.close()

    with tempfile.TemporaryDirectory(prefix='psb-v20-smoke-') as temp:
        directory = Path(temp)
        questions = directory / 'questions.csv'
        rows = [
            ('rent-001', 'Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?'),
            ('date-002', 'Нужно ли одобрение до 12 млн рублей?'),
        ]
        with questions.open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(('id', 'question'))
            writer.writerows(rows)
        output = directory / 'submission.csv'
        result = subprocess.run(
            [sys.executable, str(ROOT / 'solution.py'), '--data-dir', str(ROOT / 'data'),
             '--questions', str(questions), '--output', str(output),
             '--trace', str(directory / 'trace.jsonl')],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, 'CSV entry point failed'
        with output.open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ['id', 'answer']
            answers = list(reader)
        assert [row['id'] for row in answers] == [row[0] for row in rows]
        assert all(row['answer'] and len(row['answer']) <= 10000 for row in answers)
        assert '450000' in ''.join(answers[0]['answer'].split())
    return {'status': 'PASS', 'mode': 'offline', 'documents': health['documents'],
            'chunks': health['chunks'], 'http_assets': 4, 'document_offsets': 'PASS',
            'authentication_and_server_history': 'PASS',
            'clarification_and_followup': 'PASS', 'csv_rows': len(answers),
            'external_model_calls': 0, 'submissions_sent': 0}


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
