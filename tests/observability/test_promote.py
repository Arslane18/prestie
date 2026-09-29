import json

import pytest

from prestie.evaluation.agent_cases import load_agent_cases
from prestie.observability.promote import PromoteError, draft_case, find_trace

SNAPSHOT = {
    "character": "Lumina",
    "realm": "Khaz Modan",
    "level": 90,
    "class": {"name": "Prêtresse", "file": "PRIEST"},
    "spec": {"id": 258, "name": "Ombre", "role": "DAMAGER"},
    "quests": [],
}


def trace(**overrides):
    return {
        "trace_id": "4f3a9c0e" + "0" * 24,
        "started_at": "2026-09-29T21:14:00+00:00",
        "question": "Je peux prendre l'aggro ?",
        "answer": "Non, utilise Fade.",
        "outcome": "answered",
        "character": {"spec_id": 258, "level": 90, "hero_talent": None},
        "character_snapshot": SNAPSHOT,
        "tools": [
            {
                "name": "search_knowledge_base",
                "input": {"query": "threat", "spec": "shadow-priest"},
                "details": {"results": [{"source_url": "https://iv/a#fade"}]},
            }
        ],
        **overrides,
    }


def test_a_trace_is_found_by_an_id_prefix():
    traces = [trace(), trace(trace_id="ab" + "0" * 30)]

    assert find_trace(traces, "4f3a")["question"] == "Je peux prendre l'aggro ?"


@pytest.mark.parametrize("prefix", ["zz", ""])
def test_an_unknown_id_is_an_error(prefix):
    with pytest.raises(PromoteError, match="no trace"):
        find_trace([trace()], prefix or "missing")


def test_an_ambiguous_prefix_is_an_error():
    traces = [trace(trace_id="aa" + "0" * 30), trace(trace_id="ab" + "0" * 30)]

    with pytest.raises(PromoteError, match="several"):
        find_trace(traces, "a")


def test_a_draft_case_replays_the_question_with_the_character_of_the_time():
    case = draft_case(trace())

    assert case["id"] == "real-20260929-4f3a9c0e"
    assert case["question"] == "Je peux prendre l'aggro ?"
    assert case["character"] == SNAPSHOT
    assert case["expected_spec"] == "shadow-priest"
    assert case["tags"] == ["addon", "real", "draft"]
    assert "TODO" in case["judge_notes"]
    assert "Non, utilise Fade." in case["judge_notes"]  # the answer given then


def test_the_draft_is_a_valid_eval_case(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([draft_case(trace())]), encoding="utf-8")

    [case] = load_agent_cases(path)

    assert case.addon_mode and case.character.spec.name == "Ombre"


def test_manual_mode_traces_become_manual_cases():
    manual = trace(
        character_snapshot=None,
        character={"class": "DEATHKNIGHT", "spec": "Blood", "level": 80, "hero_talent": "San'layn"},
    )

    case = draft_case(manual)

    assert case["player"] == {"level": 80, "hero_talent": "San'layn"}
    assert "character" not in case


def test_a_trace_without_character_cannot_become_a_case():
    with pytest.raises(PromoteError, match="character"):
        draft_case(trace(character_snapshot=None, character=None))
