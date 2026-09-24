import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prestie.agent.agent import AgentError, AgentReply, Usage
from prestie.evaluation.agent_cases import AgentCase, load_agent_cases
from prestie.evaluation.agent_checks import (
    answer_lines,
    cited_urls,
    programmatic_grades,
    retrieved_urls,
)
from prestie.evaluation.agent_judge import Judge, JudgeError, JudgeVerdict
from prestie.evaluation.agent_runner import (
    HarnessChangedError,
    check_harness,
    ensure_state,
    run_agent_eval,
)
from prestie.evaluation.retrieval import EvalCaseError

SHIPPED_CASES = Path(__file__).parents[2] / "evals" / "agent_cases.json"
BASE = "https://www.icy-veins.com/wow"
MODEL = "claude-opus-5"
STAT_URL = (
    f"{BASE}/blood-death-knight-pve-tank-stat-priority#wowsanlaynht-stat-priority"
)


def case(**overrides) -> AgentCase:
    fields = {
        "id": "stats",
        "question": "Quelle stat ?",
        "level": 80,
        "hero_talent": "San'layn",
        "tags": ("sourced",),
        "should_search": True,
        "answerable": True,
        "expected_sources": ("blood-death-knight-pve-tank-stat-priority",),
        "must_include": (),
        "judge_notes": "",
        "detail_requested": False,
    }
    return AgentCase(**{**fields, **overrides})


def passages(*urls: str) -> str:
    results = "".join(
        f'<result index="1" section="S" source_url="{u}">x</result>' for u in urls
    )
    return f'<search_results query="q">{results}</search_results>'


ANSWER = (
    f"Haste d'abord.\n\nSources (contenu copié d'Icy Veins) :\n- San'layn — {STAT_URL}"
)


# --- cases -------------------------------------------------------------------


def test_shipped_agent_cases_are_valid_and_cover_both_directions():
    cases = load_agent_cases(SHIPPED_CASES)

    assert len(cases) >= 25
    assert len({c.id for c in cases}) == len(cases)
    assert any(not c.answerable for c in cases)
    assert any(c.should_search is False for c in cases)


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"id": "x", "question": "q", "player": {"level": 500, "hero_talent": None}},
        {"id": "x", "question": "", "player": {"level": 80, "hero_talent": None}},
        {"question": "q", "player": {"level": 80, "hero_talent": None}},
    ],
)
def test_malformed_agent_cases_are_rejected(tmp_path, bad_entry):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([bad_entry]), encoding="utf-8")

    with pytest.raises(EvalCaseError):
        load_agent_cases(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    entry = {"id": "x", "question": "q", "player": {"level": 80, "hero_talent": None}}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")

    with pytest.raises(EvalCaseError, match="duplicate"):
        load_agent_cases(path)


# --- programmatic checks -------------------------------------------------------


def test_cited_urls_ignore_trailing_punctuation():
    assert cited_urls(f"voir ({STAT_URL}).") == (STAT_URL,)


def test_retrieved_urls_are_read_from_tool_outputs():
    assert retrieved_urls([passages(STAT_URL)]) == (STAT_URL,)


def test_well_behaved_answer_passes_every_applicable_check():
    grades = programmatic_grades(case(), ANSWER, 1, [passages(STAT_URL)])

    assert grades == {
        "search_ok": 1.0,
        "sources_cited": 1.0,
        "citations_valid": 1.0,
        "concise": 1.0,
        "retrieved": 1.0,
    }


def test_invented_citation_fails_citations_valid():
    grades = programmatic_grades(case(), ANSWER, 1, [passages(f"{BASE}/other")])

    assert grades["citations_valid"] == 0.0
    assert grades["retrieved"] == 0.0


def test_search_expectations_in_both_directions():
    small_talk = case(should_search=False, answerable=True, expected_sources=())

    assert programmatic_grades(case(), "x", 0, [])["search_ok"] == 0.0
    assert programmatic_grades(small_talk, "Merci !", 0, [])["search_ok"] == 1.0
    assert programmatic_grades(small_talk, "Merci !", 1, [])["search_ok"] == 0.0
    assert "search_ok" not in programmatic_grades(case(should_search=None), "x", 0, [])


def test_sources_only_required_for_answerable_searched_cases():
    out_of_scope = case(answerable=False, expected_sources=())

    assert programmatic_grades(case(), "no sources", 1, [])["sources_cited"] == 0.0
    assert "sources_cited" not in programmatic_grades(
        out_of_scope, "je ne sais pas", 1, []
    )


def test_exact_copy_requires_every_expected_string():
    macro = case(must_include=("/cast [@focus] Mind Freeze",))

    assert (
        programmatic_grades(macro, "/cast [@focus] Mind Freeze", 1, [])["exact_copy"]
        == 1.0
    )
    assert programmatic_grades(macro, "/cast Mind Freeze", 1, [])["exact_copy"] == 0.0


def test_answer_lines_ignore_sources_section_and_blank_lines():
    body = "Ligne 1\n\nLigne 2\n"
    sources = "\nSources (contenu copié d’Icy Veins) :\n- a\n- b\n- c"

    assert answer_lines(body + sources) == 2


def test_long_lines_count_as_several_lines():
    assert answer_lines("x" * 250) == 3  # 100 characters per line


def test_concise_passes_up_to_eight_lines_and_fails_beyond():
    eight = "\n".join(f"ligne {i}" for i in range(8))
    nine = "\n".join(f"ligne {i}" for i in range(9))

    assert programmatic_grades(case(), eight, 1, [])["concise"] == 1.0
    assert programmatic_grades(case(), nine, 1, [])["concise"] == 0.0


def test_concise_does_not_apply_when_detail_is_requested():
    long_answer = "\n".join(f"ligne {i}" for i in range(30))

    grades = programmatic_grades(case(detail_requested=True), long_answer, 1, [])

    assert "concise" not in grades


def test_detail_requested_is_read_from_the_cases_file(tmp_path):
    entry = {
        "id": "x",
        "question": "Explique en détail",
        "player": {"level": 80, "hero_talent": None},
        "detail_requested": True,
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")

    assert load_agent_cases(path)[0].detail_requested is True


# --- judge ---------------------------------------------------------------------


class FakeJudgeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
            model="claude-sonnet-5",
            usage=SimpleNamespace(input_tokens=900, output_tokens=200),
        )


def verdicts(**overrides):
    base = {
        "context": {"reason": "ok", "verdict": "valid"},
        **{
            name: {"reason": "ok", "verdict": "pass"}
            for name in (
                "grounded",
                "admits_gap",
                "fits_player",
                "helpful",
                "french",
            )
        },
    }
    return {
        **base,
        **{k: {"reason": "because", "verdict": v} for k, v in overrides.items()},
    }


def test_judge_maps_verdicts_to_scores_and_skips_na():
    client = FakeJudgeClient(verdicts(admits_gap="na", helpful="fail"))

    verdict = Judge(client).grade(
        case(judge_notes="San'layn: Haste first"), ANSWER, [passages(STAT_URL)]
    )

    assert verdict.scores["grounded"] == 1.0
    assert verdict.scores["helpful"] == 0.0
    assert "admits_gap" not in verdict.scores
    assert verdict.explanations["helpful"] == "because"
    assert verdict.model == "claude-sonnet-5"
    request = client.requests[0]
    assert request["model"] == "claude-sonnet-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    prompt = request["messages"][0]["content"]
    for expected in ("Quelle stat ?", "San'layn: Haste first", STAT_URL, "Level 80"):
        assert expected in prompt


@pytest.mark.parametrize(
    ("label", "score"), [("valid", 1.0), ("invalid", 0.0)]
)
def test_judge_labels_the_retrieved_context_valid_or_invalid(label, score):
    client = FakeJudgeClient(verdicts(context=label))

    verdict = Judge(client).grade(case(), ANSWER, [passages(STAT_URL)])

    assert verdict.scores["context_valid"] == score
    assert verdict.explanations["context_valid"] == "because"
    schema = client.requests[0]["output_config"]["format"]["schema"]
    # Assessed first, before the answer criteria, so the answer cannot sway it.
    assert next(iter(schema["properties"])) == "context"
    assert schema["properties"]["context"]["properties"]["verdict"]["enum"] == [
        "valid",
        "invalid",
        "na",
    ]


def test_context_is_not_assessed_when_the_agent_did_not_search():
    verdict = Judge(FakeJudgeClient(verdicts(context="na"))).grade(case(), "x", [])

    assert "context_valid" not in verdict.scores


def test_judge_rejects_unusable_responses():
    with pytest.raises(JudgeError):
        Judge(FakeJudgeClient("not json")).grade(case(), "x", [])
    with pytest.raises(JudgeError):
        Judge(FakeJudgeClient(verdicts(), stop_reason="max_tokens")).grade(
            case(), "x", []
        )


# --- runner --------------------------------------------------------------------


def reply(text=ANSWER, *, model=MODEL, truncated=False, tool_output=None):
    tool_output = tool_output if tool_output is not None else passages(STAT_URL)
    tool_use = SimpleNamespace(
        type="tool_use", id="t1", name="search_knowledge_base", input={"query": "stats"}
    )
    messages = (
        {"role": "user", "content": "Quelle stat ?"},
        {"role": "assistant", "content": [tool_use]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": tool_output}
            ],
        },
        {"role": "assistant", "content": [SimpleNamespace(type="text", text=text)]},
    )
    return AgentReply(
        text=text,
        tool_calls=(),
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=100),
        truncated=truncated,
        messages=messages,
        models=(model, model),
    )


class FakeAgent:
    def __init__(self, outcome):
        self.outcome = outcome

    def ask(self, question):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def grade(self, case, answer, tool_outputs):
        self.calls += 1
        return JudgeVerdict(
            scores={"grounded": 1.0, "helpful": 1.0},
            explanations={"grounded": "ok", "helpful": "ok"},
            model="claude-sonnet-5",
            usage=Usage(input_tokens=900, output_tokens=200),
        )


def run(tmp_path, outcome, cases=None, reps=1, judge=None):
    return run_agent_eval(
        cases or [case()],
        agent_factory=lambda c: FakeAgent(outcome),
        judge=judge or FakeJudge(),
        variant_dir=tmp_path / "baseline",
        reps=reps,
        workers=2,
        expected_model=MODEL,
    )


def rows(tmp_path, name="results.jsonl"):
    path = tmp_path / "baseline" / name
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_runner_writes_graded_rows_and_traces(tmp_path):
    summary = run(tmp_path, reply(), reps=2)

    [row0, row1] = sorted(rows(tmp_path), key=lambda r: r["rep"])
    assert (row0["prompt_id"], row0["rep"], row1["rep"]) == ("stats", 0, 1)
    assert row0["status"] == "ok"
    assert row0["grade"]["pass"] == 1.0
    assert row0["grade"]["citations_valid"] == 1.0
    assert row0["grade"]["grounded"] == 1.0
    assert row0["model"] == MODEL
    assert row0["judge_model"] == "claude-sonnet-5"
    assert row0["usage"]["cache_read_input_tokens"] == 100
    assert row0["tool_calls"] == 1
    assert row0["answer_lines"] == 1
    trace = json.loads(
        (tmp_path / "baseline" / "traces" / "stats_rep0.json").read_text()
    )
    assert [turn["role"] for turn in trace] == [
        "system",
        "user",
        "tool_call",
        "tool_result",
        "assistant",
    ]
    assert summary.pass_rate == 1.0


def test_invalid_context_is_diagnostic_and_does_not_fail_the_case(tmp_path):
    class InvalidContextJudge(FakeJudge):
        def grade(self, case, answer, tool_outputs):
            verdict = super().grade(case, answer, tool_outputs)
            return JudgeVerdict(
                scores={**verdict.scores, "context_valid": 0.0},
                explanations={**verdict.explanations, "context_valid": "off-topic"},
                model=verdict.model,
                usage=verdict.usage,
            )

    run(tmp_path, reply(), judge=InvalidContextJudge())

    [row] = rows(tmp_path)
    assert row["grade"]["context_valid"] == 0.0
    assert row["grade"]["pass"] == 1.0


def test_state_file_picks_up_new_metrics_and_keeps_the_harness_approval(tmp_path):
    state_path = tmp_path / "_state.json"
    state_path.write_text(
        json.dumps({"metrics": [], "harness_sha": "abc"}), encoding="utf-8"
    )

    ensure_state(tmp_path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "context_valid" in {metric["id"] for metric in state["metrics"]}
    assert state["harness_sha"] == "abc"


def test_one_failed_check_fails_the_case(tmp_path):
    run(tmp_path, reply(tool_output=passages(f"{BASE}/elsewhere")))

    [row] = rows(tmp_path)
    assert row["grade"]["citations_valid"] == 0.0
    assert row["grade"]["pass"] == 0.0


def test_resume_skips_completed_attempts(tmp_path):
    run(tmp_path, reply())
    judge = FakeJudge()

    run(tmp_path, reply(), reps=2, judge=judge)

    assert judge.calls == 1  # only rep 1 was run
    assert sorted(r["rep"] for r in rows(tmp_path)) == [0, 1]


def test_truncated_answer_is_recorded_but_not_graded(tmp_path):
    judge = FakeJudge()

    run(tmp_path, reply(truncated=True), judge=judge)

    [row] = rows(tmp_path)
    assert row["status"] == "truncated"
    assert row["grade"] == {}
    assert judge.calls == 0


def test_serving_errors_go_to_the_sidecar_not_the_results(tmp_path):
    summary = run(tmp_path, AgentError("API down"))

    assert rows(tmp_path) == []
    [error] = rows(tmp_path, "errors.jsonl")
    assert (error["prompt_id"], error["failure_class"]) == ("stats", "serving_error")
    assert summary.errors == 1


def test_unexpected_served_model_is_an_error(tmp_path):
    run(tmp_path, reply(model="claude-opus-4-8"))

    assert rows(tmp_path) == []
    [error] = rows(tmp_path, "errors.jsonl")
    assert error["failure_class"] == "served_model_mismatch"


# --- harness gate ----------------------------------------------------------------


def test_harness_gate_requires_approval_then_detects_changes(tmp_path):
    harness = tmp_path / "grader.py"
    harness.write_text("v1")
    state = tmp_path / "_state.json"
    state.write_text("{}")

    with pytest.raises(HarnessChangedError):
        check_harness(state, [harness], approve=False)
    check_harness(state, [harness], approve=True)
    check_harness(state, [harness], approve=False)  # unchanged: fine

    harness.write_text("v2")
    with pytest.raises(HarnessChangedError):
        check_harness(state, [harness], approve=False)
