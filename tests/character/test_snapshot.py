"""state_to_snapshot is the parser's inverse: a validated state goes back to the
addon's raw format, e.g. to turn a real traced turn into an eval case."""

from dataclasses import replace

import pytest

from prestie.character.snapshot import state_to_snapshot
from prestie.character.state import character_state_from_saved_variables
from tests.character.test_equipment import raw
from tests.character.test_state import CAPTURED_AT, snapshot


def parse(raw_snapshot):
    wrapped = {"PrestieDB": {"schema": 1, "snapshot": raw_snapshot}}
    return character_state_from_saved_variables(wrapped)


def roundtrip(state):
    return parse({"capturedAt": CAPTURED_AT, **state_to_snapshot(state)})


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # spec, hero talent, tracked quest, quest log
        {"equipment": raw()},  # worn items, bag candidates, character stats
        {"equipment": raw(bags=[], stats=[])},
        {"spec": None, "heroTalent": None, "activeQuest": None, "quests": []},
        {"equipmentError": "attempt to index a nil value"},
    ],
)
def test_parsing_the_snapshot_gives_back_the_same_state(overrides):
    state = character_state_from_saved_variables(snapshot(**overrides))

    assert roundtrip(state) == state


def test_the_snapshot_has_no_capture_time():
    # An eval case gets its capture time at run time, so it never ages.
    state = character_state_from_saved_variables(snapshot())

    assert "capturedAt" not in state_to_snapshot(state)


def test_the_snapshot_is_plain_json():
    import json

    state = character_state_from_saved_variables(snapshot(equipment=raw()))

    assert json.loads(json.dumps(state_to_snapshot(state))) == state_to_snapshot(state)


def test_a_two_handed_weapon_keeps_its_location():
    staff = {"slot": 16, "id": 1, "name": "Bâton", "stats": {}, "equipLoc": "INVTYPE_2HWEAPON"}
    state = character_state_from_saved_variables(
        snapshot(equipment=raw(equipped=[staff]))
    )

    assert replace(roundtrip(state)).equipment.equipped[0].two_handed
