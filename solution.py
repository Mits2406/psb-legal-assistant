"""Local/batch entry point. Creates CSV only; never uploads or submits it."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable


MAX_ANSWER_CHARS = 10_000
PENDING_ANSWER = (
    "Технический статус: обработка запроса не завершена. "
    "Это не означает, что ответа нет в базе знаний."
)
TIMEOUT_ANSWER = (
    "Техническая ошибка: исчерпан лимит времени обработки. "
    "Ответ по документам не сформирован; повторите запрос. "
    "Это не означает, что нужных сведений нет в базе знаний."
)
ERROR_ANSWER = (
    "Техническая ошибка: запрос не удалось обработать. "
    "Ответ по документам не сформирован; повторите запрос. "
    "Это не является юридическим выводом или подтверждением отсутствия данных."
)


def technical_reply(category: str, reason: str, stage: str, standard: str) -> str:
    from llm import diagnostic_answer, diagnostic_enabled
    return diagnostic_answer(category, reason, stage) if diagnostic_enabled() else standard


def failure_metrics(reason: str, stage: str, *, attempted: bool | None = False) -> dict:
    from llm import diagnostic_enabled
    return {'mode': 'diagnostic' if diagnostic_enabled() else 'technical_error', 'status': 'technical_error', 'reason': reason,
            'trace': {'generation': {'calls': 0, 'api_attempted': attempted,
                'api_attempts': 0 if attempted is False else None,
                'api_http_success': False if attempted is False else None,
                'model_text_received': False if attempted is False else None,
                'model_text_usable': False if attempted is False else None,
                'stage': stage, 'finish_reason': 'not_called' if attempted is False else 'unknown',
                'fallback_reason': reason, 'stages': [{'stage': stage,
                    'error_code': reason, 'api_attempted': attempted,
                    'finish_reason': 'not_called' if attempted is False else 'unknown'}]}}}
LENGTH_ERROR_ANSWER = (
    "Техническая ошибка: ответ превысил допустимые 10 000 символов. "
    "Ответ не был обрезан, чтобы не потерять условия и ссылки. "
    "Повторите запрос с более узким вопросом."
)


def load_questions(path: Path) -> list[tuple[str, str]]:
    """Validate the entire input before starting work; preserve the exact IDs."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if headers is None or len(headers) != 2 or set(headers) != {"id", "question"}:
            raise ValueError("CSV должен содержать ровно колонки id и question.")
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Некорректное число полей в строке {line}.")
            qid, question = row["id"], row["question"]
            if not qid.strip() or not question.strip():
                raise ValueError(f"Пустой id или question в строке {line}.")
            if qid in seen:
                raise ValueError(f"Повторный id в строке {line}.")
            seen.add(qid)
            rows.append((qid, question.strip()))
    if not rows:
        raise ValueError("CSV не содержит вопросов.")
    return rows


def validate_output(rows: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    checked = list(rows)
    seen: set[str] = set()
    for qid, answer in checked:
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError("Некорректный или повторный id в выходных данных.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Выходной ответ не может быть пустым.")
        if len(answer) > MAX_ANSWER_CHARS:
            raise ValueError("Выходной ответ длиннее 10 000 символов.")
        seen.add(qid)
    return checked


def atomic_write_csv(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    """A crash cannot expose a half-written CSV; use the destination filesystem."""
    checked = validate_output(rows)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.writer(handle)
            writer.writerow(("id", "answer"))
            writer.writerows(checked)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def bounded_answer(result: Any) -> tuple[str, bool]:
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        return ERROR_ANSWER, True
    answer = result["answer"].strip()
    if not answer:
        return ERROR_ANSWER, True
    if len(answer) > MAX_ANSWER_CHARS:
        # Never truncate: a suffix can contain a material condition or citation.
        return LENGTH_ERROR_ANSWER, True
    return answer, result.get("status") == "technical_error"


def run_batch(
    rows: list[tuple[str, str]], assistant: Any, output: Path,
    *, max_workers: int = 4, deadline: float | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Use one immutable corpus and pass no state between independent IDs.

    The core must honor the monotonic deadline, including its HTTP timeouts.
    Python threads cannot be killed safely by the caller.
    """
    from llm import number
    question_cap = number('ANSWER_TIMEOUT_SECONDS', 120, 2, 180)
    if not rows or len({qid for qid, _ in rows}) != len(rows):
        raise ValueError("Нужны непустые вопросы с уникальными id.")
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers должен быть от 1 до 8.")
    if deadline is None:
        deadline = time.monotonic() + 1100
    results: dict[str, str] = {}
    technical_errors = 0
    timed_out = 0
    completed = 0
    budget_lock = threading.Lock()
    modes, statuses, failures = Counter(), Counter(), Counter()
    latencies = []
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text('', encoding='utf-8')
    # A pre-expired run never dispatches workers. Otherwise a racing pending
    # future could be mislabeled as an in-flight timeout without any API attempt.
    if time.monotonic() >= deadline:
        metrics = failure_metrics('batch_deadline', 'batch')
        answer = technical_reply('API_NOT_CALLED', 'batch_deadline', 'batch', TIMEOUT_ANSWER)
        atomic_write_csv(output, ((qid, answer) for qid, _ in rows))
        if trace_path is not None:
            with trace_path.open('w', encoding='utf-8') as handle:
                for qid, _ in rows:
                    handle.write(json.dumps({'id': qid, **metrics}, ensure_ascii=False) + '\n')
        return {'questions': len(rows), 'technical_errors': len(rows), 'timed_out': len(rows),
                'answer_modes': {metrics['mode']: len(rows)}, 'statuses': {'technical_error': len(rows)},
                'failure_reasons': {'batch_deadline': len(rows)},
                'mean_question_ms': 0, 'max_question_ms': 0}

    def checkpoint(default: str = PENDING_ANSWER) -> None:
        atomic_write_csv(output, ((qid, results.get(qid, default)) for qid, _ in rows))

    def invoke(question: str) -> tuple[str, bool, dict]:
        nonlocal completed
        now = time.monotonic()
        if now >= deadline:
            return (technical_reply('API_NOT_CALLED', 'batch_deadline', 'batch', TIMEOUT_ANSWER),
                    True, failure_metrics('batch_deadline', 'batch'))
        # Divide the remaining worker-time among unfinished questions. A slow
        # first question must not consume the budget of the entire hidden set.
        with budget_lock:
            unfinished = max(1, len(rows) - completed)
        fair_share = (deadline - now) * max_workers / unfinished
        # Generation and self-check share the question budget. Modest slack
        # prevents early slow requests from being cut exactly at the mean.
        question_deadline = min(deadline, now + max(.5, min(question_cap, 1.25 * fair_share)))
        try:
            # Full short-dialog history, if present, stays inside this question.
            result = assistant.answer(question, deadline=question_deadline)
            answer, error = bounded_answer(result)
            metrics = {'mode': result.get('mode', 'unknown'), 'status': result.get('status', 'unknown'),
                       'elapsed_ms': round((time.monotonic() - now) * 1000),
                       'trace': result.get('trace', {})}
            return answer, error, metrics
        except Exception as exc:
            # Exception messages can contain a URL/key; log only the class.
            print(f"core_error={type(exc).__name__}", file=sys.stderr)
            return (technical_reply('RUN_FAILURE', 'core_error', 'batch', ERROR_ANSWER),
                    True, failure_metrics('core_error', 'batch', attempted=None))
        finally:
            with budget_lock:
                completed += 1

    checkpoint()
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="psb")
    futures = {executor.submit(invoke, question): qid for qid, question in rows}
    pending = set(futures)
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            finished, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not finished:
                break
            for future in finished:
                answer, is_error, metrics = future.result()
                results[futures[future]] = answer
                technical_errors += int(is_error)
                modes[metrics.get('mode', 'unknown')] += 1
                statuses[metrics.get('status', 'unknown')] += 1
                generation = metrics.get('trace', {}).get('generation', {})
                for stage in generation.get('stages', []):
                    if stage.get('error_code'):
                        failures[stage['error_code']] += 1
                reason = generation.get('fallback_reason') or metrics.get('reason')
                if reason:
                    failures['fallback:' + str(reason)] += 1
                if 'elapsed_ms' in metrics:
                    latencies.append(metrics['elapsed_ms'])
                if trace_path is not None:
                    with trace_path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps({'id': futures[future], **metrics}, ensure_ascii=False) + '\n')
            checkpoint()
        for future in pending:
            cancelled = future.cancel()
            reason = 'batch_deadline' if cancelled else 'batch_timeout'
            category = 'API_NOT_CALLED' if cancelled else 'RUN_FAILURE'
            results[futures[future]] = technical_reply(category, reason, 'batch', TIMEOUT_ANSWER)
            metrics = failure_metrics(reason, 'batch', attempted=False if cancelled else None)
            modes['diagnostic'] += 1
            statuses['technical_error'] += 1
            failures[reason] += 1
            if trace_path is not None:
                with trace_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'id': futures[future], **metrics}, ensure_ascii=False) + '\n')
            technical_errors += 1
            timed_out += 1
        checkpoint(TIMEOUT_ANSWER)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return {"questions": len(rows), "technical_errors": technical_errors, "timed_out": timed_out,
            "answer_modes": dict(modes), "statuses": dict(statuses), "failure_reasons": dict(failures),
            "mean_question_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "max_question_ms": max(latencies) if latencies else None}


def env_int(name: str, default: int, low: int, high: int) -> int:
    value = os.environ.get(name)
    value = value.strip() if value is not None else None
    if value in (None, "", name):
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом.") from exc
    if not low <= parsed <= high:
        raise ValueError(f"{name} должен быть в диапазоне {low}..{high}.")
    return parsed


def main(argv: list[str] | None = None) -> int:
    from llm import APIError, load_env, env_first
    load_env(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(env_first("DATA_DIR") or "data"))
    parser.add_argument("--questions", type=Path, help="Default: DATA_DIR/questions.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--trace", type=Path, default=Path("run_trace.jsonl"))
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        workers = env_int("MAX_WORKERS", 4, 1, 8)
        timeout = env_int("RUN_TIMEOUT_SECONDS", 1100, 1, 1190)
        questions = args.questions or args.data_dir / "questions.csv"
        if args.output.resolve() in {
            questions.resolve(), (args.data_dir / "knowledge-base.zip").resolve(),
        }:
            raise ValueError("Выходной CSV не должен заменять входные данные.")
        if args.trace.resolve() in {args.output.resolve(), questions.resolve(),
                                    (args.data_dir / 'knowledge-base.zip').resolve()}:
            raise ValueError('Диагностика не должна заменять входные данные или ответы.')
        if not (args.data_dir / "knowledge-base.zip").is_file():
            raise ValueError("В DATA_DIR отсутствует knowledge-base.zip.")
        rows = load_questions(questions)
        atomic_write_csv(args.output, ((qid, PENDING_ANSWER) for qid, _ in rows))
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"input_error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    deadline = started + timeout
    try:
        from answer import LegalAssistant
        assistant = LegalAssistant(data_dir=args.data_dir)
    except Exception as exc:
        print(f"initialization_error={type(exc).__name__}", file=sys.stderr)
        config_error = isinstance(exc, APIError) and exc.code == 'invalid_configuration'
        reason = 'invalid_configuration' if config_error else 'initialization_error'
        category = 'API_FAILURE' if config_error else 'RUN_FAILURE'
        answer = technical_reply(category, reason, 'initialization', ERROR_ANSWER)
        atomic_write_csv(args.output, ((qid, answer) for qid, _ in rows))
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open('w', encoding='utf-8') as handle:
            for qid, _ in rows:
                handle.write(json.dumps({'id': qid, **failure_metrics(reason, 'initialization')}, ensure_ascii=False) + '\n')
        return 0  # A complete, valid CSV explicitly reports the technical failure.
    summary = run_batch(rows, assistant, args.output, max_workers=workers, deadline=deadline,
                        trace_path=args.trace)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
