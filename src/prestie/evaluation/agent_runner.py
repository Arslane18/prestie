"""Runs the real agent over the evaluation cases, grades each answer, and writes
the results in the layout the eval report builder reads:

    <flow>/_state.json                  metric definitions, prices, harness sha
    <flow>/<variant>/results.jsonl      one graded row per (case, rep)
    <flow>/<variant>/errors.jsonl       attempts that produced nothing gradable
    <flow>/<variant>/traces/<id>_rep<k>.json   full transcript per attempt

Rows are written as attempts finish, and (case, rep) pairs already present in
results.jsonl are skipped, so a crashed or interrupted run can simply resume.
"""

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from prestie.agent.agent import AgentError, AgentReply
from prestie.agent.prompts import build_system_prompt
from prestie.evaluation.agent_cases import AgentCase
from prestie.evaluation.agent_checks import (
    GATING_CHECKS,
    answer_lines,
    programmatic_grades,
)
from prestie.evaluation.agent_judge import (
    CONTEXT_METRIC,
    JUDGE_CRITERIA,
    JudgeError,
    JudgeVerdict,
)

RESULTS_FILE = "results.jsonl"
ERRORS_FILE = "errors.jsonl"
TRACES_DIR = "traces"
STATE_FILE = "_state.json"
CI_Z = 1.96  # 95% normal-approximation interval
# Reported but never gating: they tell a retrieval failure apart from a
# writing failure (an unanswerable case is expected to have invalid context).
DIAGNOSTIC_METRICS = ("retrieved", CONTEXT_METRIC)
# USD per million tokens (input, output); cache writes 1.25x input, reads 0.1x.
PRICES = {
    "claude-opus-5": {"in": 5.0, "out": 25.0},
    "claude-sonnet-5": {"in": 2.0, "out": 10.0},
}


class HarnessChangedError(Exception):
    """The grading code changed since the user last approved it."""


class AsksQuestions(Protocol):
    def ask(self, question: str) -> AgentReply: ...


class GradesAnswers(Protocol):
    def grade(
        self, case: AgentCase, answer: str, tool_outputs: Sequence[str]
    ) -> JudgeVerdict: ...


@dataclass(frozen=True)
class RunSummary:
    attempts_run: int
    errors: int
    cases: int
    pass_rate: float | None
    ci_half_width: float | None
    metric_means: Mapping[str, float]


@dataclass(frozen=True)
class _Outcome:
    row: dict[str, Any] | None = None
    trace: list[dict[str, Any]] | None = None
    error: dict[str, Any] | None = None


def run_agent_eval(
    cases: Iterable[AgentCase],
    *,
    agent_factory: Callable[[AgentCase], AsksQuestions],
    judge: GradesAnswers,
    variant_dir: Path,
    reps: int,
    workers: int,
    expected_model: str,
    clock: Callable[[], float] = time.monotonic,
) -> RunSummary:
    (variant_dir / TRACES_DIR).mkdir(parents=True, exist_ok=True)
    results_path = variant_dir / RESULTS_FILE
    done = {(row["prompt_id"], row["rep"]) for row in read_jsonl(results_path)}
    todo = [(c, rep) for c in cases for rep in range(reps) if (c.id, rep) not in done]

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_attempt, c, rep, agent_factory, judge, expected_model, clock
            )
            for c, rep in todo
        ]
        # Results are written from this thread only, as attempts complete.
        for future in as_completed(futures):
            outcome = future.result()
            if outcome.error is not None:
                errors += 1
                _append(variant_dir / ERRORS_FILE, outcome.error)
                continue
            row = outcome.row
            trace_path = (
                variant_dir / TRACES_DIR / f"{row['prompt_id']}_rep{row['rep']}.json"
            )
            trace_path.write_text(
                json.dumps(outcome.trace, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _append(results_path, row)
    return summarize(read_jsonl(results_path), attempts_run=len(todo), errors=errors)


def _run_attempt(
    case: AgentCase,
    rep: int,
    agent_factory: Callable[[AgentCase], AsksQuestions],
    judge: GradesAnswers,
    expected_model: str,
    clock: Callable[[], float],
) -> _Outcome:
    """Never raises: every failure becomes an error record with a failure class."""
    base = {"prompt_id": case.id, "rep": rep}
    try:
        start = clock()
        reply = agent_factory(case).ask(case.question)
        latency = clock() - start
    except AgentError as exc:
        return _Outcome(
            error={**base, "failure_class": "serving_error", "message": str(exc)}
        )
    except Exception as exc:  # noqa: BLE001 - recorded, never scored as a model failure
        return _Outcome(
            error={**base, "failure_class": "harness_error", "message": repr(exc)}
        )

    usage = asdict(reply.usage)
    unexpected = [m for m in reply.models if not _same_model(m, expected_model)]
    if unexpected:
        return _Outcome(
            error={
                **base,
                "failure_class": "served_model_mismatch",
                "message": f"expected {expected_model}, served {unexpected}",
                "model": unexpected[0],
                "usage": usage,
            }
        )

    tool_outputs = _tool_outputs(reply.messages)
    search_count = _search_count(reply.messages)
    row: dict[str, Any] = {
        **base,
        "prompt": case.question,
        "tags": list(case.tags),
        "stop_reason": _stop_reason(reply),
        "model": reply.models[-1] if reply.models else expected_model,
        "usage": usage,
        "latency_s": round(latency, 2),
        "tool_calls": search_count,
        "answer_lines": answer_lines(reply.text),
        "meta": {"player": case.player().describe(), "answer": reply.text},
    }
    trace = to_trace(case, reply)
    if reply.truncated:
        return _Outcome(row={**row, "status": "truncated", "grade": {}}, trace=trace)
    if reply.refused:
        grade = {"pass": 0.0}
        return _Outcome(
            row={
                **row,
                "status": "ok",
                "grade": grade,
                "explanation": {"pass": "refused"},
            },
            trace=trace,
        )

    scores = programmatic_grades(case, reply.text, search_count, tool_outputs)
    try:
        verdict = judge.grade(case, reply.text, tool_outputs)
    except JudgeError as exc:
        return _Outcome(
            error={
                **base,
                "failure_class": "grader_error",
                "message": str(exc),
                "usage": usage,
            }
        )
    scores = {**scores, **verdict.scores}
    gating = [value for name, value in scores.items() if name not in DIAGNOSTIC_METRICS]
    return _Outcome(
        row={
            **row,
            "status": "ok",
            "grade": {"pass": float(all(v == 1.0 for v in gating)), **scores},
            "explanation": dict(verdict.explanations),
            "judge_model": verdict.model,
            "judge_usage": asdict(verdict.usage),
        },
        trace=trace,
    )


# --- transcript ----------------------------------------------------------------


def to_trace(case: AgentCase, reply: AgentReply) -> list[dict[str, Any]]:
    """Convert the API messages of one turn into the report's Turn[] format."""
    turns: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(case.player())}
    ]
    for message in reply.messages:
        content = message["content"]
        if message["role"] == "user":
            if isinstance(content, str):
                turns.append({"role": "user", "content": content})
            else:
                turns.extend(
                    {
                        "role": "tool_result",
                        "name": "search_knowledge_base",
                        "content": str(r["content"]),
                    }
                    for r in content
                )
            continue
        thinking = ""
        for block in content:
            kind = getattr(block, "type", None)
            if kind == "thinking":
                thinking += getattr(block, "thinking", "") or ""
            elif kind in ("text", "tool_use"):
                turn = (
                    {"role": "assistant", "content": block.text}
                    if kind == "text"
                    else {
                        "role": "tool_call",
                        "name": block.name,
                        "content": json.dumps(
                            block.input, ensure_ascii=False, indent=2
                        ),
                    }
                )
                turns.append({**turn, "thinking": thinking} if thinking else turn)
                thinking = ""
    return turns


def _tool_outputs(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(result["content"])
        for message in messages
        if message["role"] == "user" and isinstance(message["content"], list)
        for result in message["content"]
    ]


def _search_count(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        if message["role"] == "assistant"
        for block in message["content"]
        if getattr(block, "type", None) == "tool_use"
    )


def _stop_reason(reply: AgentReply) -> str:
    if reply.refused:
        return "refusal"
    return "max_tokens" if reply.truncated else "end_turn"


def _same_model(served: str, expected: str) -> bool:
    # Allow documented alias -> dated snapshot resolution (e.g. "<id>-2026...").
    return served == expected or served.startswith(f"{expected}-2")


# --- summary and state ------------------------------------------------------------


def summarize(
    rows: Sequence[Mapping[str, Any]], *, attempts_run: int, errors: int
) -> RunSummary:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("grade")]
    per_case: dict[str, list[float]] = {}
    for row in ok_rows:
        per_case.setdefault(row["prompt_id"], []).append(row["grade"]["pass"])
    case_means = [sum(v) / len(v) for v in per_case.values()]
    pass_rate = sum(case_means) / len(case_means) if case_means else None
    ci = None
    if len(case_means) > 1 and pass_rate is not None:
        variance = sum((m - pass_rate) ** 2 for m in case_means) / (len(case_means) - 1)
        ci = CI_Z * math.sqrt(variance / len(case_means))
    metric_values: dict[str, list[float]] = {}
    for row in ok_rows:
        for name, value in row["grade"].items():
            metric_values.setdefault(name, []).append(value)
    return RunSummary(
        attempts_run=attempts_run,
        errors=errors,
        cases=len(per_case),
        pass_rate=pass_rate,
        ci_half_width=ci,
        metric_means={name: sum(v) / len(v) for name, v in metric_values.items()},
    )


def row_cost_usd(row: Mapping[str, Any]) -> float:
    """Agent + judge cost of one row, derived from recorded usage and model."""
    return _usage_cost(row.get("model"), row.get("usage")) + _usage_cost(
        row.get("judge_model"), row.get("judge_usage")
    )


def _usage_cost(model: str | None, usage: Mapping[str, int] | None) -> float:
    price = PRICES.get(model or "")
    if not price or not usage:
        return 0.0
    input_equivalent = (
        usage.get("input_tokens", 0)
        + 1.25 * usage.get("cache_creation_input_tokens", 0)
        + 0.1 * usage.get("cache_read_input_tokens", 0)
    )
    return (
        input_equivalent * price["in"] + usage.get("output_tokens", 0) * price["out"]
    ) / 1e6


def initial_state() -> dict[str, Any]:
    checks = [*GATING_CHECKS, *DIAGNOSTIC_METRICS]
    judged = [name for name, _ in JUDGE_CRITERIA]
    return {
        "metrics": [
            {"id": "pass", "label": "pass", "kind": "binary"},
            *(
                {"id": name, "label": name[:14], "kind": "binary"}
                for name in (*checks, *judged)
            ),
        ],
        "perf_fields": [
            {"id": "latency_s", "label": "latency", "unit": "s"},
            {"id": "tool_calls", "label": "searches"},
            {"id": "answer_lines", "label": "lines"},
        ],
        "prices": {model: dict(p) for model, p in PRICES.items()},
    }


def ensure_state(flow_dir: Path) -> Path:
    """Write the current metric definitions, keeping the harness approval."""
    path = flow_dir / STATE_FILE
    flow_dir.mkdir(parents=True, exist_ok=True)
    existing = (
        json.loads(path.read_text(encoding="utf-8") or "{}") if path.exists() else {}
    )
    path.write_text(
        json.dumps({**existing, **initial_state()}, indent=2), encoding="utf-8"
    )
    return path


def check_harness(
    state_path: Path, harness_paths: Sequence[Path], *, approve: bool
) -> None:
    """Refuse to run if the grading code changed since the user approved it.

    Scores are only comparable across runs graded by the same code; this makes
    any change to the harness an explicit, user-approved decision.
    """
    digest = hashlib.sha256()
    for path in sorted(harness_paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    sha = digest.hexdigest()
    state = json.loads(state_path.read_text(encoding="utf-8") or "{}")
    if approve:
        state_path.write_text(
            json.dumps({**state, "harness_sha": sha}, indent=2), encoding="utf-8"
        )
        return
    if state.get("harness_sha") != sha:
        raise HarnessChangedError(
            "The grading harness is new or changed since it was last approved. "
            "Review it, then re-run with --approve-harness."
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
