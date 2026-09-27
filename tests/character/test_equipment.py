import pytest

from prestie.character.equipment import equipment_from_snapshot
from prestie.character.state import (
    CharacterStateError,
    character_state_from_saved_variables,
)
from tests.character.test_state import snapshot

HELM = {
    "slot": 1,
    "id": 212001,
    "name": "Heaume du rempart",
    "itemLevel": 623,
    "enchant": 7534,
    "gems": 1,
    "stats": {
        "ITEM_MOD_STRENGTH_SHORT": 1200,
        "ITEM_MOD_STAMINA_SHORT": 5000,
        "ITEM_MOD_CRIT_RATING_SHORT": 400,
        "ITEM_MOD_HASTE_RATING_SHORT": 300,
        "EMPTY_SOCKET_PRISMATIC": 1,
        "RESISTANCE0_NAME": 900,  # armor: not a stat the guides rank
    },
}
RING = {"slot": 11, "id": 212002, "name": "Anneau", "itemLevel": 610, "stats": {}}
BAG_RING = {
    "id": 212003,
    "name": "Anneau trouvé",
    "itemLevel": 626,
    "equipLoc": "INVTYPE_FINGER",
    "subType": "Divers",
    "stats": {"ITEM_MOD_MASTERY_RATING_SHORT": 500, "ITEM_MOD_VERSATILITY": 200},
}
STATS = {
    "itemLevel": 620.5,
    "equippedItemLevel": 619.25,
    "crit": {"rating": 1234, "percent": 15.2},
    "haste": {"rating": 800, "percent": 9},
    "mastery": {"rating": 950, "percent": 40.1},
    "versatility": {"rating": 400, "percent": 3.5},
}


def raw(**overrides):
    return {"equipped": [HELM, RING], "bags": [BAG_RING], "stats": STATS, **overrides}


def test_equipped_items_are_named_by_slot_with_readable_stats():
    equipment = equipment_from_snapshot(raw())

    helm, ring = equipment.equipped
    assert (helm.slot, helm.name, helm.item_level) == ("head", "Heaume du rempart", 623)
    assert dict(helm.stats) == {
        "strength": 1200,
        "stamina": 5000,
        "critical_strike": 400,
        "haste": 300,
    }
    assert (helm.enchant_id, helm.gems, helm.empty_sockets) == (7534, 1, 1)
    assert ring.slot == "finger1"
    assert (ring.enchant_id, ring.gems, ring.empty_sockets) == (None, 0, 0)


def test_bag_items_know_which_slots_they_fit():
    [ring] = equipment_from_snapshot(raw()).bags

    assert ring.slots == ("finger1", "finger2")
    assert dict(ring.stats) == {"mastery": 500, "versatility": 200}
    assert ring.sub_type == "Divers"


@pytest.mark.parametrize(
    "equip_loc, slots",
    [
        ("INVTYPE_ROBE", ("chest",)),
        ("INVTYPE_2HWEAPON", ("main_hand",)),
        ("INVTYPE_WEAPON", ("main_hand", "off_hand")),
        ("INVTYPE_SHIELD", ("off_hand",)),
        ("INVTYPE_CLOAK", ("back",)),
        ("INVTYPE_TRINKET", ("trinket1", "trinket2")),
    ],
)
def test_equip_locations_map_to_slots(equip_loc, slots):
    bag_item = {**BAG_RING, "equipLoc": equip_loc}

    [item] = equipment_from_snapshot(raw(bags=[bag_item])).bags

    assert item.slots == slots


def test_bag_items_that_fit_no_slot_are_left_out():
    tabard = {**BAG_RING, "equipLoc": "INVTYPE_TABARD"}

    assert equipment_from_snapshot(raw(bags=[tabard])).bags == ()


def test_character_stats_are_parsed():
    stats = equipment_from_snapshot(raw()).stats

    assert stats.item_level == 620.5
    assert stats.equipped_item_level == 619.25
    assert (stats.crit.rating, stats.crit.percent) == (1234, 15.2)
    assert stats.haste.percent == 9


def test_empty_lua_tables_mean_nothing_exported():
    # An empty Lua table parses as [], whatever it was meant to hold.
    equipment = equipment_from_snapshot({"equipped": [], "bags": {}, "stats": []})

    assert equipment.equipped == ()
    assert equipment.bags == ()
    assert equipment.stats is None


@pytest.mark.parametrize(
    "bad",
    [
        raw(equipped=[{**HELM, "slot": 4}]),  # shirt: never exported
        raw(equipped=[{**HELM, "itemLevel": "623"}]),
        raw(equipped=[{**HELM, "stats": {"ITEM_MOD_CRIT_RATING_SHORT": "x"}}]),
        raw(bags=[{**BAG_RING, "equipLoc": None}]),
        raw(stats={**STATS, "crit": 12}),
        raw(equipped="helm"),
    ],
)
def test_malformed_equipment_is_rejected(bad):
    with pytest.raises(CharacterStateError):
        equipment_from_snapshot(bad)


def test_state_carries_the_equipment_when_exported():
    state = character_state_from_saved_variables(snapshot(equipment=raw()))

    assert state.equipment.equipped[0].slot == "head"
    assert state.equipment_error is None


def test_older_addons_export_no_equipment():
    state = character_state_from_saved_variables(snapshot())

    assert state.equipment is None
    assert state.equipment_error is None


def test_addon_errors_while_reading_equipment_are_reported():
    state = character_state_from_saved_variables(
        snapshot(equipmentError="attempt to index a nil value")
    )

    assert state.equipment is None
    assert state.equipment_error == "attempt to index a nil value"


def test_worn_items_carry_their_equip_location_when_exported():
    staff = {**HELM, "slot": 16, "equipLoc": "INVTYPE_2HWEAPON"}

    worn = equipment_from_snapshot(raw(equipped=[staff, RING])).equipped

    assert worn[0].two_handed is True
    assert worn[1].two_handed is False  # older exports: no equipLoc
