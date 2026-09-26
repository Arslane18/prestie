"""Multi-turn agent cases: earlier questions are played for real, the last is graded."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from prestie.agent.agent import AgentReply, Usage
from prestie.evaluation.agent_cases import load_agent_cases
from prestie.evaluation.agent_judge import JUDGE_CRITERIA, build_judge_prompt
from prestie.evaluation.agent_runner import run_agent_eval
from prestie.evaluation.retrieval import EvalCaseError
from tests.evaluation.test_agent_eval import (
    ANSWER,
    MODEL,
    SHIPPED_CASES,
    SNAPSHOT,
    STAT_URL,
    FakeJudge,
    addon_case,
    case,
    load_one,
    passages,
    rows,
)

SHADOW_SNAPSHOT = {
    **SNAPSHOT,
    "class": {"name": "Prêtre", "file": "PRIEST"},
    "spec": {"id": 258, "name": "Ombre", "role": "DAMAGER"},
    "heroTalent": None,
    "quests": [],
}


def multi_turn_entry(**overrides):
    entry = {
        "id": "follow-up",
        "turns": [
            {"question": "Quelle stat ?"},
            {"question": "Et en mythique+ ?"},
        ],
        "character": SNAPSHOT,
    }
    return {**entry, **overrides}


# --- cases -------------------------------------------------------------------


def test_multi_turn_case_grades_the_last_question(tmp_path):
    loaded = load_one(tmp_path, multi_turn_entry())

    assert loaded.question == "Et en mythique+ ?"
    assert [turn.question for turn in loaded.prior_turns] == ["Quelle stat ?"]
    assert loaded.addon_mode


def test_single_question_cases_have_no_prior_turns(tmp_path):
    loaded = load_one(tmp_path, {"id": "x", "question": "q", "player": {"level": 80}})

    assert loaded.prior_turns == ()


def test_a_turn_can_bring_a_new_character_state_after_a_reload(tmp_path):
    entry = multi_turn_entry(
        turns=[
            {"question": "Quelle stat ?"},
            {"question": "J'ai changé de spé, et maintenant ?"},
            {"question": "Et ma rotation ?", "character": SHADOW_SNAPSHOT},
        ]
    )
    loaded = load_one(tmp_path, entry)
    source = loaded.character_source(datetime.now(UTC))

    served = []
    for index in range(3):
        source.select_turn(index)
        served.append(source.latest().spec.name)

    assert served == ["Sang", "Sang", "Ombre"]
    assert loaded.character.spec.name == "Ombre"  # the graded turn's state


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": "q"},  # both question and turns
        {"turns": []},
        {"turns": "Quelle stat ?"},
        {"turns": [{"question": ""}]},
        {"turns": [{"question": "q", "extra": 1}]},
        {"turns": [{"question": "q"}, {"question": "r", "character": {"level": 9}}]},
    ],
)
def test_malformed_turns_are_rejected(tmp_path, overrides):
    with pytest.raises(EvalCaseError):
        load_one(tmp_path, multi_turn_entry(**overrides))


def test_turn_characters_need_addon_mode(tmp_path):
    entry = {
        "id": "x",
        "player": {"level": 80},
        "turns": [{"question": "q"}, {"question": "r", "character": SNAPSHOT}],
    }

    with pytest.raises(EvalCaseError, match="addon"):
        load_one(tmp_path, entry)


def test_shipped_multi_turn_cases_are_valid():
    cases = load_agent_cases(SHIPPED_CASES.with_name("agent_cases_multiturn.json"))

    assert len(cases) >= 8
    assert all(c.prior_turns for c in cases)
    assert any(
        turn.character != c.prior_turns[0].character
        for c in cases
        for turn in (*c.prior_turns[1:], c)
    ), "at least one case changes the character state between turns"


# --- runner ------------------------------------------------------------------


def turn_reply(question, text, *tool_uses, usage=None):
    """One turn's AgentReply; tool_uses are (name, input, output) triples."""
    blocks = [
        SimpleNamespace(type="tool_use", id=f"{question}-{i}", name=name, input=args)
        for i, (name, args, _) in enumerate(tool_uses)
    ]
    messages = [{"role": "user", "content": question}]
    if blocks:
        messages += [
            {"role": "assistant", "content": blocks},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": output}
                    for b, (_, _, output) in zip(blocks, tool_uses)
                ],
            },
        ]
    messages.append(
        {"role": "assistant", "content": [SimpleNamespace(type="text", text=text)]}
    )
    return AgentReply(
        text=text,
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        messages=tuple(messages),
        models=(MODEL,),
    )


class ScriptedAgent:
    """Answers each question with the next scripted reply, recording what it saw."""

    def __init__(self, replies, source=None):
        self.replies = list(replies)
        self.source = source
        self.asked = []
        self.states = []

    def ask(self, question):
        self.asked.append(question)
        if self.source is not None:
            self.states.append(self.source.latest().spec.name)
        return self.replies[len(self.asked) - 1]


def run_turns(tmp_path, cases, agents, judge=None):
    def factory(c, source):
        agent = ScriptedAgent(agents[c.id], source)
        agents[c.id] = agent
        return agent

    return run_agent_eval(
        cases,
        agent_factory=factory,
        judge=judge or FakeJudge(),
        variant_dir=tmp_path / "baseline",
        reps=1,
        workers=1,
        expected_model=MODEL,
    )


def search(url):
    return ("search_knowledge_base", {"query": "stats"}, passages(url))


def two_turn_case(**overrides):
    from prestie.evaluation.agent_cases import Turn

    return case(
        id="follow-up",
        question="Et en mythique+ ?",
        prior_turns=(Turn("Quelle stat ?"),),
        **{"should_search": None, **overrides},
    )


def test_runner_plays_every_turn_and_grades_the_last(tmp_path):
    first = turn_reply(
        "Quelle stat ?", "Hâte.", search(STAT_URL), usage=Usage(input_tokens=100)
    )
    # The follow-up cites a passage retrieved in the first turn, without searching.
    second = turn_reply("Et en mythique+ ?", ANSWER, usage=Usage(input_tokens=300))
    agents = {"follow-up": [first, second]}

    summary = run_turns(tmp_path, [two_turn_case()], agents)

    assert agents["follow-up"].asked == ["Quelle stat ?", "Et en mythique+ ?"]
    [row] = rows(tmp_path)
    assert row["prompt"] == "Et en mythique+ ?"
    assert row["grade"]["citations_valid"] == 1.0
    assert row["tool_calls"] == 0  # searches of the graded turn only
    assert row["usage"]["input_tokens"] == 400
    assert [t["usage"]["input_tokens"] for t in row["turns"]] == [100, 300]
    assert row["meta"]["conversation"] == [
        {"question": "Quelle stat ?", "answer": "Hâte."}
    ]
    assert summary.pass_rate == 1.0
    trace = json.loads(
        (tmp_path / "baseline" / "traces" / "follow-up_rep0.json").read_text()
    )
    assert [t["role"] for t in trace] == [
        "system",
        "user",
        "tool_call",
        "tool_result",
        "assistant",
        "user",
        "assistant",
    ]


def test_search_expectations_apply_to_the_graded_turn_only(tmp_path):
    first = turn_reply("Quelle stat ?", "Hâte.", search(STAT_URL))
    second = turn_reply("Et en mythique+ ?", ANSWER)
    agents = {"follow-up": [first, second]}

    run_turns(tmp_path, [two_turn_case(should_search=True)], agents)

    [row] = rows(tmp_path)
    assert row["grade"]["search_ok"] == 0.0


def test_judge_sees_earlier_exchanges_and_every_turn_passages(tmp_path):
    seen = {}

    class RecordingJudge(FakeJudge):
        def grade(self, case, answer, tool_outputs, prior_exchanges=()):
            seen["outputs"] = list(tool_outputs)
            seen["exchanges"] = list(prior_exchanges)
            return super().grade(case, answer, tool_outputs)

    first = turn_reply("Quelle stat ?", "Hâte.", search(STAT_URL))
    second = turn_reply("Et en mythique+ ?", ANSWER)

    run_turns(
        tmp_path,
        [two_turn_case()],
        {"follow-up": [first, second]},
        judge=RecordingJudge(),
    )

    assert seen["outputs"] == [passages(STAT_URL)]
    assert [(e.question, e.answer) for e in seen["exchanges"]] == [
        ("Quelle stat ?", "Hâte.")
    ]


def test_runner_switches_the_character_state_between_turns(tmp_path):
    entry = multi_turn_entry(
        turns=[
            {"question": "Quelle stat ?"},
            {"question": "Et maintenant ?", "character": SHADOW_SNAPSHOT},
        ]
    )
    loaded = load_one(tmp_path, entry)
    replies = [turn_reply("Quelle stat ?", "a"), turn_reply("Et maintenant ?", "b")]
    agents = {"follow-up": replies}

    run_turns(tmp_path, [loaded], agents)

    assert agents["follow-up"].states == ["Sang", "Ombre"]


@pytest.mark.parametrize("failure", ["refused", "truncated"])
def test_a_failed_setup_turn_is_an_error_not_a_grade(tmp_path, failure):
    first = AgentReply(text="", models=(MODEL,), **{failure: True})
    second = turn_reply("Et en mythique+ ?", ANSWER)
    agents = {"follow-up": [first, second]}

    run_turns(tmp_path, [two_turn_case()], agents)

    assert rows(tmp_path) == []
    [error] = rows(tmp_path, "errors.jsonl")
    assert error["failure_class"] == "setup_turn_failed"
    assert agents["follow-up"].asked == ["Quelle stat ?"]


# --- judge -------------------------------------------------------------------


def test_judge_prompt_lists_earlier_exchanges_before_the_question():
    exchanges = [SimpleNamespace(question="Quelle stat ?", answer="Hâte.")]

    prompt = build_judge_prompt(addon_case(), "x", [], exchanges)

    assert "<previous_exchanges>" in prompt
    assert prompt.index("Hâte.") < prompt.index("<question>")


def test_single_question_judge_prompt_has_no_conversation_block():
    assert "<previous_exchanges>" not in build_judge_prompt(addon_case(), "x", [])


def test_judge_grades_how_the_answer_follows_the_conversation():
    definition = dict(JUDGE_CRITERIA)["follows_conversation"]

    assert "earlier" in definition
    assert "na" in definition
