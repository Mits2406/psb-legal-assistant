"""V20 wire defaults and real batch deadline propagation, without model calls."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import ModelClient
import solution


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, limit):
        return json.dumps({
            "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "Короткий ответ по источнику."
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7},
        }).encode()[:limit]


class ModelDefaultsV20Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "LLM_BASE_URL": "http://quality-fixture.invalid/v1",
            "LLM_API_KEY": "fixture-only-key",
            "LLM_MODEL": "platform-supplied-model",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        opener = patch("urllib.request.build_opener")
        self.build_opener = opener.start()
        self.addCleanup(opener.stop)
        self.open = self.build_opener.return_value.open
        self.open.return_value = Response()

    def call_and_payload(self, **kwargs):
        client = ModelClient()
        result = client.chat_text("Отвечай по источникам.", {"question": "Вопрос?"}, **kwargs)
        self.assertEqual(result, "Короткий ответ по источнику.")
        self.open.assert_called_once()
        return client, json.loads(self.open.call_args.args[0].data)

    def test_default_wire_budget_and_temperature_preserve_platform_model(self):
        client, payload = self.call_and_payload()
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["model"], "platform-supplied-model")
        self.assertEqual(client.last_chat_diagnostics["requested_tokens"], 8192)
        self.assertEqual(self.open.call_args.kwargs["timeout"], 90)
        self.assertEqual(set(payload), {"model", "messages", "temperature", "max_tokens"})

    def test_explicit_platform_values_take_priority_over_quality_defaults(self):
        os.environ.update(MAX_TOKENS="1024", TEMPERATURE="0.25", API_TIMEOUT_SECONDS="7")
        client, payload = self.call_and_payload()
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["temperature"], .25)
        self.assertEqual(self.open.call_args.kwargs["timeout"], 7)
        self.assertEqual(client.last_chat_diagnostics["requested_tokens"], 1024)

    def test_openai_aliases_remain_the_source_of_identity(self):
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "http://alias-fixture.invalid/v1",
            "OPENAI_API_KEY": "fixture-alias-key",
            "GEN_MODEL": "another-platform-model",
        }, clear=True):
            _, payload = self.call_and_payload()
        request = self.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://alias-fixture.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer fixture-alias-key")
        self.assertEqual(payload["model"], "another-platform-model")

    def test_explicit_stage_limit_can_still_reserve_a_smaller_budget(self):
        _, payload = self.call_and_payload(max_tokens=1800)
        self.assertEqual(payload["max_tokens"], 1800)

    def test_deadline_still_takes_priority_over_longer_transport_timeout(self):
        with patch("llm.time.monotonic", return_value=1000.0):
            self.call_and_payload(deadline=1009.0)
        self.assertEqual(self.open.call_args.kwargs["timeout"], 9)


class RecordingAssistant:
    def __init__(self):
        self.deadlines = {}

    def answer(self, question, *, deadline):
        self.deadlines[question] = deadline
        return {"answer": "Готово: " + question, "status": "answer", "mode": "fixture"}


class BatchDeadlinesV20Tests(unittest.TestCase):
    NOW = 1000.0

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        clock = patch("solution.time.monotonic", return_value=self.NOW)
        clock.start()
        self.addCleanup(clock.stop)
        self.directory = tempfile.TemporaryDirectory(prefix="psb-v20-deadline-")
        self.addCleanup(self.directory.cleanup)

    def run_fixture(self, *, count=1, workers=4, remaining=1000):
        assistant = RecordingAssistant()
        output = Path(self.directory.name) / "submission.csv"
        rows = [(str(i), "Вопрос " + str(i)) for i in range(count)]
        summary = solution.run_batch(rows, assistant, output, max_workers=workers,
                                     deadline=self.NOW + remaining)
        self.assertEqual(summary["technical_errors"], 0)
        self.assertEqual(summary["timed_out"], 0)
        with output.open(encoding="utf-8", newline="") as source:
            actual = list(csv.DictReader(source))
        self.assertEqual([row["id"] for row in actual], [qid for qid, _ in rows])
        self.assertEqual(len(assistant.deadlines), count)
        return assistant.deadlines

    def test_default_batch_question_budget_reaches_120_seconds(self):
        deadlines = self.run_fixture()
        self.assertEqual(deadlines["Вопрос 0"], self.NOW + 120)

    def test_explicit_answer_budget_reaches_the_batch_worker(self):
        os.environ["ANSWER_TIMEOUT_SECONDS"] = "37.5"
        deadlines = self.run_fixture()
        self.assertEqual(deadlines["Вопрос 0"], self.NOW + 37.5)

    def test_environment_budget_still_respects_supported_upper_limit(self):
        os.environ["ANSWER_TIMEOUT_SECONDS"] = "999"
        deadlines = self.run_fixture()
        self.assertEqual(deadlines["Вопрос 0"], self.NOW + 180)

    def test_fair_share_protects_other_questions_from_long_default(self):
        deadlines = self.run_fixture(count=4, workers=1, remaining=80)
        self.assertEqual(deadlines["Вопрос 0"], self.NOW + 25)
        self.assertTrue(all(value <= self.NOW + 80 for value in deadlines.values()))

    def test_global_batch_deadline_wins_over_question_default(self):
        deadlines = self.run_fixture(remaining=30)
        self.assertEqual(deadlines["Вопрос 0"], self.NOW + 30)


if __name__ == "__main__":
    unittest.main()
