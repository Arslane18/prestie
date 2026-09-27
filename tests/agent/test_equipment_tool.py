from datetime import timedelta

from prestie.agent.equipment_tool import EQUIPMENT_TOOL, EquipmentTool
from prestie.character.equipment import equipment_from_snapshot
from prestie.character.state import CharacterStateError
from tests.agent.test_character_tool import CAPTURED_AT, make_state
from tests.character.test_equipment import BAG_RING, raw


class FixedSource:
    def __init__(self, state=None, error=None):
        self.state, self.error = state, error

    def latest(self):
        if self.error:
            raise self.error
        return self.state


def run(state=None, error=None):
    tool = EquipmentTool(
        FixedSource(state, error), now=lambda: CAPTURED_AT + timedelta(minutes=12)
    )
    return tool.run({})


def geared(**overrides):
    return make_state(equipment=equipment_from_snapshot(raw(**overrides)))


def test_definition_takes_no_input():
    assert EQUIPMENT_TOOL["name"] == "get_equipment"
    assert EQUIPMENT_TOOL["input_schema"] == {"type": "object", "properties": {}}


def test_output_lists_worn_items_with_their_stats():
    outcome = run(geared())

    assert not outcome.is_error
    text = outcome.content
    assert text.startswith("<equipment ")
    assert 'age_minutes="12"' in text
    assert 'search with spec="blood-death-knight"' in text
    assert "Item level: 619.2 equipped (620.5 owned)" in text
    assert "Critical Strike 1234 (15.2%)" in text
    assert (
        "- head: Heaume du rempart, item level 623; Strength 1200, Stamina 5000, "
        "Critical Strike 400, Haste 300; enchanted, 1 gem, 1 empty socket"
    ) in text
    assert "- finger1: Anneau, item level 610; no stats\n" in text


def test_empty_slots_are_listed():
    text = run(geared()).content

    assert "Empty slots: neck, shoulder, chest," in text


def test_bag_items_are_compared_with_what_they_would_replace():
    text = run(geared()).content

    assert "Equippable items in bags:" in text
    assert (
        "- Anneau trouvé (Divers), item level 626; Mastery 500, Versatility 200; "
        "finger1: +16 item levels vs Anneau (610); "
        "finger2: slot empty"
    ) in text


def test_no_bag_items_is_said_explicitly():
    assert "Equippable items in bags: none" in run(geared(bags=[])).content


def test_bag_items_are_capped_best_first():
    rings = [{**BAG_RING, "name": f"R{i}", "itemLevel": 600 + i} for i in range(30)]

    text = run(geared(bags=rings)).content

    assert "- R29 " in text
    assert "- R0 " not in text
    assert "(10 more not listed)" in text


def test_missing_export_is_an_error_asking_to_update_the_addon():
    outcome = run(make_state())

    assert outcome.is_error
    assert "update the Prestie addon" in outcome.content


def test_addon_failure_is_reported():
    outcome = run(make_state(equipment_error="boom"))

    assert outcome.is_error
    assert "boom" in outcome.content


def test_unavailable_state_is_an_error():
    outcome = run(error=CharacterStateError("Prestie.lua not found"))

    assert outcome.is_error
    assert "Prestie.lua not found" in outcome.content


def test_equipment_calls_have_a_player_facing_label():
    from prestie.agent.agent import ToolCall
    from prestie.agent.tool_labels import tool_call_label

    assert tool_call_label(ToolCall("get_equipment", {})) == (
        "[équipement] lecture du stuff exporté par l'addon"
    )


def test_missing_enchants_are_not_flagged_by_the_tool():
    # Which slots take an enchant depends on the patch: the guides say, not the tool.
    assert "not enchanted" not in run(geared()).content


def two_hander(**overrides):
    staff = {"slot": 16, "id": 1, "name": "Bâton", "itemLevel": 620, "stats": {}}
    return geared(equipped=[{**staff, **overrides}])


def test_an_empty_off_hand_is_expected_with_a_two_handed_weapon():
    text = run(two_hander(equipLoc="INVTYPE_2HWEAPON")).content

    assert "- main_hand: Bâton, item level 620; no stats; two-handed" in text
    assert "off_hand" not in text.split("Empty slots:")[1].split("\n")[0]
    assert "Off hand: none needed (two-handed weapon)" in text


def test_an_empty_off_hand_is_listed_with_a_one_handed_weapon():
    text = run(two_hander(equipLoc="INVTYPE_WEAPON")).content

    assert "off_hand" in text.split("Empty slots:")[1].split("\n")[0]
    assert "two-handed" not in text
