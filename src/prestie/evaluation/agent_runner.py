"""Runs the real agent over the evaluation cases, grades each answer, and writes
the results in the layout the eval report builder reads:

    <flow>/_state.json                  metric definitions, prices, harness sha
    <flow>/<variant>/results.jsonl      one graded row per (case, rep)
    <flow>/<variant>/results.stale.jsonl rows graded by an older harness or case
    <flow>/<variant>/errors.jsonl       attempts that produced nothing gradable
    <flow>/<variant>/traces/<id>_rep<k>.json   full transcript per attempt

Rows are written as attempts finish, and (case, rep) pairs already present in
results.jsonl are skipped, so a crashed or interrupted run can simply resume.
Each row records the harness and the case version that graded it: a resumed
run only keeps rows graded by the current ones (the others are moved to
results.stale.jsonl and graded again), so a variant never mixes two graders.

A truncated answer is a failure. An attempt that could not be graded (serving
or judge error) is not scored, but a case left with no graded rep makes the
run incomplete: its pass rate is not a verdict.

A multi-turn case asks every question to the same agent and grades the last
answer: tool calls are checked on that turn. Earlier tool results are given to
the checks and the judge as the agent had them when answering, compacted
(sources and URLs of earlier searches still count as retrieved).
"""

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from prestie.agent.agent import AgentError, AgentReply, ToolCall, Usage
from prestie.agent.context import compact_history
from prestie.agent.prompts import build_system_prompt
from prestie.agent.tools import SEARCH_TOOL_NAME
from prestie.evaluation.agent_cases import (
    AgentCase,
    TurnCharacterSource,
    case_digest,
)
from prestie.evaluation.agent_checks import (
    GATING_CHECKS,
    answer_lines,
    count_calls,
    programmatic_grades,
)
from prestie.evaluation.agent_judge import (
    CONTEXT_METRIC,
    JUDGE_CRITERIA,
    Exchange,
    JudgeError,
    JudgeVerdict,
)
from prestie.pricing import PRICES, usage_cost_usd

RESULTS_FILE = "results.jsonl"
ERRORS_FILE = "errors.jsonl"
STALE_RESULTS_FILE = "results.stale.jsonl"
TRACES_DIR = "traces"
STATE_FILE = "_state.json"
CI_Z = 1.96  # 95% normal-approximation interval
# Reported but never gating: they tell a retrieval failure apart from a
# writing failure (an unanswerable case is expected to have invalid context).
DIAGNOSTIC_METRICS = ("retrieved", CONTEXT_METRIC)
TRUNCATED_STOP_REASON = "max_tokens"


class HarnessChangedError(Exception):
    """The grading code changed since the user last approved it."""


class AsksQuestions(Protocol):
    def ask(self, question: str) -> AgentReply: ...


class GradesAnswers(Protocol):
    def grade(
        self,
        case: AgentCase,
        answer: str,
        tool_outputs: Sequence[str],
        prior_exchanges: Sequence[Exchange] = (),
    ) -> JudgeVerdict: ...


# Builds the agent for one attempt; the source is None in manual mode.
AgentFactory = Callable[[AgentCase, TurnCharacterSource | None], AsksQuestions]


@dataclass(frozen=True)
class RunSummary:
    attempts_run: int
    errors: int
    cases: int
    pass_rate: float | None
    ci_half_width: float | None
    metric_means: Mapping[str, float]
    graded_attempts: int
    expected_attempts: int
    truncated: int
    ungraded_cases: tuple[str, ...]  # no graded rep: the pass rate is no verdict

    @property
    def complete(self) -> bool:
        return not self.ungraded_cases


@dataclass(frozen=True)
class _Conversation:
    replies: tuple[AgentReply, ...]
    latencies: tuple[float, ...]


class _SetupTurnFailed(Exception):
    """An earlier turn was refused or truncated: the graded turn is meaningless."""


@dataclass(frozen=True)
class _Outcome:
    row: dict[str, Any] | None = None
    trace: list[dict[str, Any]] | None = None
    error: dict[str, Any] | None = None


def run_agent_eval(
    cases: Iterable[AgentCase],
    *,
    agent_factory: AgentFactory,
    judge: GradesAnswers,
    variant_dir: Path,
    reps: int,
    workers: int,
    expected_model: str,
    harness_sha: str,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RunSummary:
    cases = tuple(cases)
    (variant_dir / TRACES_DIR).mkdir(parents=True, exist_ok=True)
    results_path = variant_dir / RESULTS_FILE
    versions = {
        c.id: {"harness_sha": harness_sha, "case_sha": case_digest(c)} for c in cases
    }
    kept = _keep_current_rows(variant_dir, versions, harness_sha)
    done = {(row["prompt_id"], row["rep"]) for row in kept}
    todo = [(c, rep) for c in cases for rep in range(reps) if (c.id, rep) not in done]

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_attempt,
                c,
                rep,
                agent_factory,
                judge,
                expected_model,
                versions[c.id],
                clock,
                now,
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
    return summarize(
        read_jsonl(results_path),
        case_ids=tuple(c.id for c in cases),
        reps=reps,
        attempts_run=len(todo),
        errors=errors,
    )


def _keep_current_rows(
    variant_dir: Path, versions: Mapping[str, Mapping[str, str]], harness_sha: str
) -> list[dict[str, Any]]:
    """Move rows graded by another harness or case version out of results.jsonl.

    Rows of cases not selected in this run are kept if the harness matches:
    their case version cannot be checked without the case.
    """
    results_path = variant_dir / RESULTS_FILE
    rows = read_jsonl(results_path)

    def current(row: Mapping[str, Any]) -> bool:
        expected = versions.get(row["prompt_id"], {"harness_sha": harness_sha})
        return all(row.get(key) == value for key, value in expected.items())

    kept = [row for row in rows if current(row)]
    stale = [row for row in rows if not current(row)]
    if stale:
        for row in stale:
            _append(variant_dir / STALE_RESULTS_FILE, row)
        _write_jsonl(results_path, kept)
    return kept


def _run_attempt(
    case: AgentCase,
    rep: int,
    agent_factory: AgentFactory,
    judge: GradesAnswers,
    expected_model: str,
    versions: Mapping[str, str],
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> _Outcome:
    """Never raises: every failure becomes an error record with a failure class."""
    base = {"prompt_id": case.id, "rep": rep}
    try:
        conversation = _play(case, agent_factory, clock, now)
    except _SetupTurnFailed as exc:
        return _Outcome(
            error={**base, "failure_class": "setup_turn_failed", "message": str(exc)}
        )
    except AgentError as exc:
        return _Outcome(
            error={**base, "failure_class": "serving_error", "message": str(exc)}
        )
    except Exception as exc:  # noqa: BLE001 - recorded, never scored as a model failure
        return _Outcome(
            error={**base, "failure_class": "harness_error", "message": repr(exc)}
        )

    replies = conversation.replies
    reply = replies[-1]
    usage = asdict(sum((r.usage for r in replies), Usage()))
    served = [m for r in replies for m in r.models]
    unexpected = [m for m in served if not _same_model(m, expected_model)]
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

    tool_outputs = _graded_turn_context(replies)
    tool_calls = _tool_calls(reply.messages)
    exchanges = [Exchange(q, r.text) for q, r in zip(_questions(case), replies[:-1])]
    row: dict[str, Any] = {
        **base,
        **versions,
        "prompt": case.question,
        "tags": list(case.tags),
        "stop_reason": _stop_reason(reply),
        "model": reply.models[-1] if reply.models else expected_model,
        "usage": usage,
        "latency_s": round(conversation.latencies[-1], 2),
        "tool_calls": count_calls(tool_calls, SEARCH_TOOL_NAME),
        "answer_lines": answer_lines(reply.text),
        "turns": _turn_records(case, conversation),
        "meta": {
            "player": case.describe_player(),
            "answer": reply.text,
            "conversation": [asdict(e) for e in exchanges],
        },
    }
    trace = to_trace(case, replies)
    if reply.truncated or reply.refused:
        # A failure, not a missing grade: leaving it out would raise the pass
        # rate by dropping the hardest cases.
        reason = "truncated" if reply.truncated else "refused"
        return _Outcome(
            row={
                **row,
                "status": "ok",
                "grade": {"pass": 0.0},
                "explanation": {"pass": reason},
            },
            trace=trace,
        )

    scores = programmatic_grades(case, reply.text, tool_calls, tool_outputs)
    try:
        verdict = judge.grade(case, reply.text, tool_outputs, exchanges)
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


def _play(
    case: AgentCase,
    agent_factory: AgentFactory,
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> _Conversation:
    """Ask every question of the case to one agent, switching the state per turn."""
    source = case.character_source(now()) if case.addon_mode else None
    agent = agent_factory(case, source)
    questions = _questions(case)
    replies: list[AgentReply] = []
    latencies: list[float] = []
    for index, question in enumerate(questions):
        if source is not None:
            source.select_turn(index)
        start = clock()
        reply = agent.ask(question)
        latencies.append(clock() - start)
        replies.append(reply)
        is_setup = index < len(questions) - 1
        if is_setup and (reply.refused or reply.truncated):
            reason = "refused" if reply.refused else "truncated"
            raise _SetupTurnFailed(f"turn {index} ({question!r}) was {reason}")
    return _Conversation(tuple(replies), tuple(latencies))


def _graded_turn_context(replies: Sequence[AgentReply]) -> list[str]:
    """The tool results the agent had when it wrote the last answer.

    Earlier turns are compacted before each new question (search results
    reduced to their sources, character state and equipment removed), so the
    judge must not credit the answer with passages the agent no longer had.
    """
    earlier = [m for r in replies[:-1] for m in r.messages]
    return [
        *_tool_outputs(compact_history(earlier)),
        *_tool_outputs(replies[-1].messages),
    ]


def _questions(case: AgentCase) -> tuple[str, ...]:
    return (*(turn.question for turn in case.prior_turns), case.question)


def _turn_records(case: AgentCase, conversation: _Conversation) -> list[dict[str, Any]]:
    """Per-turn usage and latency: how the cost grows along the conversation."""
    return [
        {
            "question": question,
            "usage": asdict(reply.usage),
            "latency_s": round(latency, 2),
            "tool_calls": len(_tool_uses(reply.messages)),
        }
        for question, reply, latency in zip(
            _questions(case), conversation.replies, conversation.latencies
        )
    ]


# --- transcript ----------------------------------------------------------------


def to_trace(case: AgentCase, replies: Sequence[AgentReply]) -> list[dict[str, Any]]:
    """Convert the API messages of every turn into the report's Turn[] format."""
    turns: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(case.player())}
    ]
    messages = [message for reply in replies for message in reply.messages]
    tool_names = _tool_names_by_id(messages)
    for message in messages:
        content = message["content"]
        if message["role"] == "user":
            if isinstance(content, str):
                turns.append({"role": "user", "content": content})
            else:
                turns.extend(
                    {
                        "role": "tool_result",
                        "name": tool_names.get(r["tool_use_id"], "?"),
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


def _tool_uses(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [
        block
        for message in messages
        if message["role"] == "assistant"
        for block in message["content"]
        if getattr(block, "type", None) == "tool_use"
    ]


def _tool_calls(messages: Sequence[Mapping[str, Any]]) -> tuple[ToolCall, ...]:
    return tuple(ToolCall(block.name, block.input) for block in _tool_uses(messages))


def _tool_names_by_id(messages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {block.id: block.name for block in _tool_uses(messages)}


def _stop_reason(reply: AgentReply) -> str:
    if reply.refused:
        return "refusal"
    return "max_tokens" if reply.truncated else "end_turn"


def _same_model(served: str, expected: str) -> bool:
    # Allow documented alias -> dated snapshot resolution (e.g. "<id>-2026...").
    return served == expected or served.startswith(f"{expected}-2")


# --- summary and state ------------------------------------------------------------


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    reps: int,
    attempts_run: int,
    errors: int,
) -> RunSummary:
    """Pass rate over the given cases: mean over cases of the mean over reps."""
    selected = set(case_ids)
    ok_rows = [
        r
        for r in rows
        if r["prompt_id"] in selected and r.get("status") == "ok" and r.get("grade")
    ]
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
        graded_attempts=len(ok_rows),
        expected_attempts=len(selected) * reps,
        truncated=sum(r.get("stop_reason") == TRUNCATED_STOP_REASON for r in ok_rows),
        ungraded_cases=tuple(c for c in dict.fromkeys(case_ids) if c not in per_case),
    )


def row_cost_usd(row: Mapping[str, Any]) -> float:
    """Agent + judge cost of one row, derived from recorded usage and model.

    A model missing from the price table counts 0 (see `prestie.pricing`).
    """
    return _usage_cost(row.get("model"), row.get("usage")) + _usage_cost(
        row.get("judge_model"), row.get("judge_usage")
    )


def _usage_cost(model: str | None, usage: Mapping[str, int] | None) -> float:
    if not model or not usage:
        return 0.0
    return usage_cost_usd(model, Usage(**usage)) or 0.0


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
        "prices": {
            model: {"in": input_price, "out": output_price}
            for model, (input_price, output_price) in PRICES.items()
        },
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


def harness_digest(harness_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(harness_paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_harness(
    state_path: Path, harness_paths: Sequence[Path], *, approve: bool
) -> str:
    """Refuse to run if the grading code changed since the user approved it.

    Scores are only comparable across runs graded by the same code; this makes
    any change to the harness an explicit, user-approved decision. Returns the
    approved digest, which every result row records.
    """
    sha = harness_digest(harness_paths)
    state = json.loads(state_path.read_text(encoding="utf-8") or "{}")
    if approve:
        state_path.write_text(
            json.dumps({**state, "harness_sha": sha}, indent=2), encoding="utf-8"
        )
        return sha
    if state.get("harness_sha") != sha:
        raise HarnessChangedError(
            "The grading harness is new or changed since it was last approved. "
            "Review it, then re-run with --approve-harness."
        )
    return sha


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Replace the file atomically: a crash leaves the old or the new rows."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    tmp.replace(path)


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
