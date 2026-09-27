"""Gear cases: the agent must read the equipment when (and only when) needed."""

import pytest

from prestie.agent.agent import ToolCall
from prestie.evaluation.agent_cases import load_agent_cases
from prestie.evaluation.agent_checks import GATING_CHECKS, programmatic_grades
from prestie.evaluation.agent_judge import JUDGE_CRITERIA, Judge
from prestie.evaluation.retrieval import EvalCaseError
from tests.evaluation.test_agent_eval import (
    SHIPPED_CASES,
    FakeJudgeClient,
    addon_case,
    addon_entry,
    load_one,
    verdicts,
)

GEAR_READ = (ToolCall("get_equipment", {}),)


def test_should_read_gear_is_read_from_addon_cases(tmp_path):
    assert load_one(tmp_path, addon_entry(should_read_gear=True)).should_read_gear


def test_should_read_gear_needs_addon_mode(tmp_path):
    entry = {"id": "x", "question": "q", "player": {"level": 80}}

    with pytest.raises(EvalCaseError, match="character"):
        load_one(tmp_path, {**entry, "should_read_gear": True})


def test_should_read_gear_must_be_a_boolean(tmp_path):
    with pytest.raises(EvalCaseError, match="should_read_gear"):
        load_one(tmp_path, addon_entry(should_read_gear="yes"))


def test_gear_read_expectations_in_both_directions():
    must = addon_case(should_read_gear=True)
    must_not = addon_case(should_read_gear=False)

    assert programmatic_grades(must, "ok", GEAR_READ, [])["gear_read"] == 1.0
    assert programmatic_grades(must, "ok", (), [])["gear_read"] == 0.0
    assert programmatic_grades(must_not, "ok", GEAR_READ, [])["gear_read"] == 0.0
    assert "gear_read" not in programmatic_grades(addon_case(), "ok", GEAR_READ, [])
    assert "gear_read" in GATING_CHECKS


def test_judge_counts_the_equipment_as_a_source():
    judge = Judge(FakeJudgeClient(verdicts()))

    assert "equipment" in judge._system
    assert "equipment" in dict(JUDGE_CRITERIA)["uses_state"]


def test_shipped_gear_cases_are_valid():
    cases = load_agent_cases(SHIPPED_CASES.with_name("agent_cases_gear.json"))

    assert len(cases) >= 8
    assert {c.should_read_gear for c in cases} >= {True, False}
    assert any(c.character and c.character.equipment is None for c in cases)
