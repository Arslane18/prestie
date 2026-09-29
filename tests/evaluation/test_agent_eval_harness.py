"""Review lot C: grading biases that could skew eval conclusions.

- a truncated answer is a failure, and a case with no graded rep blocks a verdict;
- every row records the harness and case version that graded it, and a resumed
  run only keeps rows graded by the current ones;
- the judge sees earlier turns as the agent had them (compacted);
- every cited URL must have been retrieved, whatever its domain;
- draft cases (promoted traces) cannot be run;
- the eval prices rows with the shared pricing table.
"""

import json

import pytest

from prestie.agent.agent import AgentError
from prestie.evaluation.agent_cases import case_digest
from prestie.evaluation.agent_checks import cited_urls, programmatic_grades
from prestie.evaluation.agent_judge import JUDGE_SYSTEM, JudgeError
from prestie.evaluation.agent_runner import (
    STALE_RESULTS_FILE,
    check_harness,
    initial_state,
    row_cost_usd,
    summarize,
)
from prestie.evaluation.retrieval import EvalCaseError
from tests.evaluation.test_agent_eval import (
    ANSWER,
    HARNESS_SHA,
    STAT_URL,
    FakeJudge,
    case,
    load_one,
    passages,
    reply,
    rows,
    run,
    searches,
)

WOWHEAD_URL = "https://www.wowhead.com/spell=49028/dancing-rune-weapon"


# --- truncation and coverage ----------------------------------------------------


class FlakyJudge(FakeJudge):
    """Fails on the case called "broken", like a judge hitting max_tokens."""

    def grade(self, c, answer, tool_outputs, prior_exchanges=()):
        if c.id == "broken":
            raise JudgeError("Judge stopped with stop_reason=max_tokens")
        return super().grade(c, answer, tool_outputs)


def test_a_case_with_no_graded_rep_blocks_the_verdict(tmp_path):
    cases = [case(id="graded"), case(id="broken")]

    summary = run(tmp_path, reply(), cases=cases, reps=2, judge=FlakyJudge())

    # Judge errors are an infrastructure failure: not scored as failures...
    assert summary.pass_rate == 1.0
    assert summary.errors == 2
    # ...but the summary says it cannot conclude.
    assert summary.ungraded_cases == ("broken",)
    assert (summary.graded_attempts, summary.expected_attempts) == (2, 4)
    assert not summary.complete


def test_a_full_run_is_complete(tmp_path):
    summary = run(tmp_path, reply(), reps=2)

    assert summary.complete
    assert (summary.graded_attempts, summary.expected_attempts) == (2, 2)


def test_serving_errors_leave_the_case_ungraded(tmp_path):
    summary = run(tmp_path, AgentError("API down"))

    assert summary.ungraded_cases == ("stats",)
    assert summary.pass_rate is None


def test_summary_counts_truncations_from_the_rows():
    row = {
        "prompt_id": "a",
        "rep": 0,
        "status": "ok",
        "stop_reason": "max_tokens",
        "grade": {"pass": 0.0},
    }

    summary = summarize([row], case_ids=("a",), reps=1, attempts_run=1, errors=0)

    assert summary.truncated == 1
    assert summary.pass_rate == 0.0


# --- harness and case versions --------------------------------------------------


def test_rows_record_the_harness_and_case_versions(tmp_path):
    run(tmp_path, reply())

    [row] = rows(tmp_path)
    assert row["harness_sha"] == HARNESS_SHA
    assert row["case_sha"] == case_digest(case())


def test_case_digest_changes_when_the_case_changes():
    assert case_digest(case()) == case_digest(case())
    assert case_digest(case()) != case_digest(case(judge_notes="Haste first"))


def test_resume_regrades_rows_from_another_harness_and_archives_them(tmp_path):
    run(tmp_path, reply())
    [old] = rows(tmp_path)
    path = tmp_path / "baseline" / "results.jsonl"
    path.write_text(json.dumps({**old, "harness_sha": "older"}) + "\n")
    judge = FakeJudge()

    run(tmp_path, reply(), judge=judge)

    assert judge.calls == 1  # the old row was not reused
    [row] = rows(tmp_path)
    assert row["harness_sha"] == HARNESS_SHA
    [stale] = rows(tmp_path, STALE_RESULTS_FILE)
    assert stale["harness_sha"] == "older"


def test_resume_regrades_rows_of_an_edited_case(tmp_path):
    run(tmp_path, reply())
    judge = FakeJudge()

    run(tmp_path, reply(), cases=[case(judge_notes="edited")], judge=judge)

    assert judge.calls == 1
    [row] = rows(tmp_path)
    assert row["case_sha"] == case_digest(case(judge_notes="edited"))


def test_resume_keeps_rows_of_cases_not_selected_this_time(tmp_path):
    run(tmp_path, reply(), cases=[case(id="a"), case(id="b")])

    run(tmp_path, reply(), cases=[case(id="a")])  # like --only a

    assert sorted(r["prompt_id"] for r in rows(tmp_path)) == ["a", "b"]
    assert rows(tmp_path, STALE_RESULTS_FILE) == []


def test_check_harness_returns_the_approved_sha(tmp_path):
    harness = tmp_path / "grader.py"
    harness.write_text("v1")
    state = tmp_path / "_state.json"
    state.write_text("{}")

    sha = check_harness(state, [harness], approve=True)

    assert sha == check_harness(state, [harness], approve=False)
    assert json.loads(state.read_text())["harness_sha"] == sha


# --- judge context in conversations ---------------------------------------------


def test_judge_system_prompt_describes_compacted_earlier_turns():
    assert "still had in its context" not in JUDGE_SYSTEM
    assert "sections and URLs" in JUDGE_SYSTEM


# --- citations ------------------------------------------------------------------


def test_citations_from_any_domain_are_extracted():
    answer = f"{ANSWER}\n- Wowhead — {WOWHEAD_URL}"

    assert cited_urls(answer) == (STAT_URL, WOWHEAD_URL)


def test_an_invented_link_to_another_site_fails_citations_valid():
    answer = f"{ANSWER}\n- Wowhead — {WOWHEAD_URL}"

    grades = programmatic_grades(case(), answer, searches(1), [passages(STAT_URL)])

    assert grades["citations_valid"] == 0.0


# --- drafts -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tags": ["real", "draft"]},
        {"judge_notes": "TODO: what a good answer must contain"},
    ],
)
def test_draft_cases_are_refused(tmp_path, overrides):
    entry = {"id": "real-1", "question": "q", "player": {"level": 80}, **overrides}

    with pytest.raises(EvalCaseError, match="draft"):
        load_one(tmp_path, entry)


# --- cost ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["claude-opus-5", "claude-opus-5-20260901", "claude-opus-4-8"]
)
def test_rows_are_priced_with_the_shared_table(model):
    row = {"model": model, "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}

    assert row_cost_usd(row) == pytest.approx(5.0)


def test_state_prices_come_from_the_shared_table():
    assert initial_state()["prices"]["claude-sonnet-5"] == {"in": 2.0, "out": 10.0}
