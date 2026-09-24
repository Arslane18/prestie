from datetime import UTC, datetime

import pytest

from prestie.character.state import (
    CharacterStateError,
    character_state_from_saved_variables,
)

CAPTURED_AT = 1790278000


def snapshot(**overrides):
    base = {
        "capturedAt": CAPTURED_AT,
        "character": "Testeur",
        "realm": "Hyjal",
        "level": 83,
        "class": {"name": "Chevalier de la mort", "file": "DEATHKNIGHT"},
        "spec": {"id": 250, "name": "Sang", "role": "TANK"},
        "heroTalent": "San'layn",
        "activeQuest": {
            "id": 1234,
            "title": "Une quête",
            "objectives": [
                {"text": "0/8 loups", "finished": False, "fulfilled": 0, "required": 8}
            ],
        },
        "quests": [
            {"id": 1234, "title": "Une quête", "level": 83, "complete": False},
            {"id": 99, "title": "Autre", "level": 80, "complete": True},
        ],
    }
    return {"PrestieDB": {"schema": 1, "snapshot": {**base, **overrides}}}


def test_builds_the_full_state():
    state = character_state_from_saved_variables(snapshot())

    assert state.character == "Testeur"
    assert state.level == 83
    assert state.class_token == "DEATHKNIGHT"
    assert state.spec.id == 250 and state.spec.role == "TANK"
    assert state.hero_talent == "San'layn"
    assert state.active_quest.title == "Une quête"
    assert state.active_quest.objectives[0].required == 8
    assert [quest.id for quest in state.quests] == [1234, 99]
    assert state.captured_at == datetime.fromtimestamp(CAPTURED_AT, UTC)


def test_blood_death_knight_is_covered_by_the_knowledge_base():
    assert character_state_from_saved_variables(snapshot()).covered_by_knowledge_base


def test_other_specs_are_not_covered():
    state = character_state_from_saved_variables(
        snapshot(
            **{
                "class": {"name": "Prêtresse", "file": "PRIEST"},
                "spec": {"id": 256, "name": "Discipline", "role": "HEALER"},
            }
        )
    )

    assert not state.covered_by_knowledge_base


def test_spec_id_zero_means_no_specialization_yet():
    # Below level 10 the game reports spec id 0 with no name.
    state = character_state_from_saved_variables(snapshot(spec={"id": 0}))

    assert state.spec is None
    assert not state.covered_by_knowledge_base


def test_optional_parts_may_be_missing():
    data = snapshot()
    for key in ("heroTalent", "activeQuest", "spec"):
        del data["PrestieDB"]["snapshot"][key]
    data["PrestieDB"]["snapshot"]["quests"] = []

    state = character_state_from_saved_variables(data)

    assert state.hero_talent is None
    assert state.active_quest is None
    assert state.spec is None
    assert state.quests == ()


def test_empty_objectives_table_is_accepted():
    # An empty Lua table parses as [], whatever it was meant to be.
    data = snapshot(activeQuest={"id": 5, "title": "Sans objectif", "objectives": []})

    assert character_state_from_saved_variables(data).active_quest.objectives == ()


@pytest.mark.parametrize(
    "data, message",
    [
        ({}, "PrestieDB"),
        ({"PrestieDB": {"schema": 1}}, "snapshot"),
        ({"PrestieDB": {"schema": 99, "snapshot": {}}}, "schema"),
        (snapshot(level="83"), "level"),
        (snapshot(level=0), "level"),
        (snapshot(capturedAt=None), "capturedAt"),
        (snapshot(quests=[{"title": "no id"}]), "id"),
        (snapshot(**{"class": "DK"}), "class"),
    ],
)
def test_rejects_malformed_data(data, message):
    with pytest.raises(CharacterStateError, match=message):
        character_state_from_saved_variables(data)
