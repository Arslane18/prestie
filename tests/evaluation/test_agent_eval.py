import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prestie.agent.agent import AgentError, AgentReply, ToolCall, Usage
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


def searches(count: int) -> tuple[ToolCall, ...]:
    return tuple(ToolCall("search_knowledge_base", {"query": "q"}) for _ in range(count))


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
    grades = programmatic_grades(case(), ANSWER, searches(1), [passages(STAT_URL)])

    assert grades == {
        "search_ok": 1.0,
        "sources_cited": 1.0,
        "citations_valid": 1.0,
        "concise": 1.0,
        "retrieved": 1.0,
    }


def test_invented_citation_fails_citations_valid():
    grades = programmatic_grades(case(), ANSWER, searches(1), [passages(f"{BASE}/other")])

    assert grades["citations_valid"] == 0.0
    assert grades["retrieved"] == 0.0


def test_search_expectations_in_both_directions():
    small_talk = case(should_search=False, answerable=True, expected_sources=())

    assert programmatic_grades(case(), "x", searches(0), [])["search_ok"] == 0.0
    assert programmatic_grades(small_talk, "Merci !", searches(0), [])["search_ok"] == 1.0
    assert programmatic_grades(small_talk, "Merci !", searches(1), [])["search_ok"] == 0.0
    assert "search_ok" not in programmatic_grades(case(should_search=None), "x", searches(0), [])


def test_only_knowledge_base_searches_count_as_searches():
    small_talk = case(should_search=False, answerable=True, expected_sources=())
    state_read = (ToolCall("get_character_state", {}),)

    assert programmatic_grades(small_talk, "Merci !", state_read, [])["search_ok"] == 1.0


def addon_case(**overrides) -> AgentCase:
    return case(
        **{
            "addon_mode": True,
            "should_search": None,
            "expected_sources": (),
            **overrides,
        }
    )


def test_state_read_expectations_in_both_directions():
    state_read = (ToolCall("get_character_state", {}),)
    must_read = addon_case(should_read_state=True)
    must_not = addon_case(should_read_state=False)

    assert programmatic_grades(must_read, "x", state_read, [])["state_read"] == 1.0
    assert programmatic_grades(must_read, "x", (), [])["state_read"] == 0.0
    assert programmatic_grades(must_not, "x", (), [])["state_read"] == 1.0
    assert programmatic_grades(must_not, "x", state_read, [])["state_read"] == 0.0
    assert "state_read" not in programmatic_grades(addon_case(), "x", (), [])


def test_quest_lookup_requires_every_expected_quest():
    quest_case = addon_case(expected_quest_ids=(55881, 55763))

    def lookups(*ids):
        return tuple(ToolCall("get_quest_details", {"quest_id": i}) for i in ids)

    grades = programmatic_grades(quest_case, "x", lookups(55881, 55763), [])
    partial = programmatic_grades(quest_case, "x", lookups(55881, 1), [])

    assert grades["quest_lookup"] == 1.0
    assert partial["quest_lookup"] == 0.0
    assert "quest_lookup" not in programmatic_grades(addon_case(), "x", (), [])


def test_sources_only_required_for_answerable_searched_cases():
    out_of_scope = case(answerable=False, expected_sources=())

    assert programmatic_grades(case(), "no sources", searches(1), [])["sources_cited"] == 0.0
    assert "sources_cited" not in programmatic_grades(
        out_of_scope, "je ne sais pas", searches(1), []
    )


def test_exact_copy_requires_every_expected_string():
    macro = case(must_include=("/cast [@focus] Mind Freeze",))

    assert (
        programmatic_grades(macro, "/cast [@focus] Mind Freeze", searches(1), [])["exact_copy"]
        == 1.0
    )
    assert programmatic_grades(macro, "/cast Mind Freeze", searches(1), [])["exact_copy"] == 0.0


def test_answer_lines_ignore_sources_section_and_blank_lines():
    body = "Ligne 1\n\nLigne 2\n"
    sources = "\nSources (contenu copié d’Icy Veins) :\n- a\n- b\n- c"

    assert answer_lines(body + sources) == 2


def test_long_lines_count_as_several_lines():
    assert answer_lines("x" * 250) == 3  # 100 characters per line


def test_concise_passes_up_to_eight_lines_and_fails_beyond():
    eight = "\n".join(f"ligne {i}" for i in range(8))
    nine = "\n".join(f"ligne {i}" for i in range(9))

    assert programmatic_grades(case(), eight, searches(1), [])["concise"] == 1.0
    assert programmatic_grades(case(), nine, searches(1), [])["concise"] == 0.0


def test_concise_does_not_apply_when_detail_is_requested():
    long_answer = "\n".join(f"ligne {i}" for i in range(30))

    grades = programmatic_grades(case(detail_requested=True), long_answer, searches(1), [])

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
                "uses_state",
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
    assert trace[3]["name"] == "search_knowledge_base"


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


# --- addon-mode cases ------------------------------------------------------------

SNAPSHOT = {
    "character": "Testeur",
    "realm": "Hyjal",
    "level": 90,
    "class": {"name": "Chevalier de la mort", "file": "DEATHKNIGHT"},
    "spec": {"id": 250, "name": "Sang", "role": "TANK"},
    "heroTalent": "San'layn",
    "quests": [{"id": 55881, "title": "Purge totémique", "level": 6}],
}


def addon_entry(**overrides):
    entry = {
        "id": "state-stats",
        "question": "Quelle stat ?",
        "character": SNAPSHOT,
        "should_read_state": True,
        "expected_quest_ids": [55881],
    }
    return {**entry, **overrides}


def load_one(tmp_path, entry):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return load_agent_cases(path)[0]


def test_addon_case_carries_a_validated_character(tmp_path):
    loaded = load_one(tmp_path, addon_entry())

    assert loaded.addon_mode
    assert loaded.player() is None
    assert loaded.character.hero_talent == "San'layn"
    assert loaded.should_read_state is True
    assert loaded.expected_quest_ids == (55881,)
    assert "get_character_state" in loaded.describe_player()


def test_addon_case_state_is_fresh_at_run_time(tmp_path):
    from datetime import UTC, datetime, timedelta

    loaded = load_one(tmp_path, addon_entry(character_age_minutes=30))
    now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)

    state = loaded.character_source(now).latest()

    assert state.captured_at == now - timedelta(minutes=30)
    assert state.level == 90


def test_null_character_means_the_state_is_unavailable(tmp_path):
    from datetime import UTC, datetime

    from prestie.character.state import CharacterStateError

    loaded = load_one(tmp_path, addon_entry(character=None))

    assert loaded.addon_mode and loaded.character is None
    assert "unavailable" in loaded.describe_player()
    with pytest.raises(CharacterStateError):
        loaded.character_source(datetime.now(UTC)).latest()


@pytest.mark.parametrize(
    "overrides",
    [
        {"player": {"level": 80}},  # both player and character
        {"character": {"level": 90}},  # invalid snapshot
        {"character_age_minutes": -1},
        {"expected_quest_ids": ["55881"]},
    ],
)
def test_malformed_addon_cases_are_rejected(tmp_path, overrides):
    with pytest.raises(EvalCaseError):
        load_one(tmp_path, addon_entry(**overrides))


def test_state_fields_are_rejected_in_manual_cases(tmp_path):
    entry = {
        "id": "x",
        "question": "q",
        "player": {"level": 80},
        "should_read_state": True,
    }

    with pytest.raises(EvalCaseError, match="character"):
        load_one(tmp_path, entry)


def test_shipped_addon_cases_are_valid():
    cases = load_agent_cases(SHIPPED_CASES.with_name("agent_cases_addon.json"))

    assert all(c.addon_mode for c in cases)
    assert {c.should_read_state for c in cases} >= {True, False}
    assert any(c.expected_quest_ids for c in cases)
    assert any(c.character is None for c in cases)


def test_judge_grades_use_of_the_character_state():
    from prestie.evaluation.agent_judge import JUDGE_CRITERIA, build_judge_prompt

    definition = dict(JUDGE_CRITERIA)["uses_state"]
    prompt = build_judge_prompt(addon_case(), "answer", ["<character_state>"])

    assert "does not ask" in definition
    assert "Addon mode" in prompt
    assert "<character_state>" in prompt


def test_runner_grades_state_reads_in_addon_cases(tmp_path):
    state_call = SimpleNamespace(
        type="tool_use", id="s1", name="get_character_state", input={}
    )
    messages = (
        {"role": "user", "content": "Quelle stat ?"},
        {"role": "assistant", "content": [state_call]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "s1", "content": "STATE"}
            ],
        },
        {"role": "assistant", "content": [SimpleNamespace(type="text", text="ok")]},
    )
    outcome = AgentReply(text="ok", messages=messages, models=(MODEL,))

    run(tmp_path, outcome, cases=[addon_case(should_read_state=True)])

    [row] = rows(tmp_path)
    assert row["grade"]["state_read"] == 1.0
    assert row["tool_calls"] == 0  # searches only
    assert row["meta"]["player"].startswith("Addon mode")
    trace = json.loads(
        (tmp_path / "baseline" / "traces" / "stats_rep0.json").read_text()
    )
    assert trace[3] == {
        "role": "tool_result",
        "name": "get_character_state",
        "content": "STATE",
    }
