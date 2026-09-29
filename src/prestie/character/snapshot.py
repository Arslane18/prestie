"""The parser's inverse: a validated character state back in the addon's format.

A trace keeps the character as the addon exported it, so that a real turn can
become an eval case (cases hold raw addon snapshots). Round-trip guarantee,
covered by tests: parsing `state_to_snapshot(state)` gives back `state`, the
capture time aside (a case gets its capture time when it runs).
"""

from collections.abc import Mapping
from typing import Any

from prestie.character.equipment import (
    EMPTY_SOCKET_PREFIX,
    EQUIP_LOCATION_SLOTS,
    SLOT_NAMES,
    STAT_NAMES,
    BagItem,
    CharacterStats,
    Equipment,
    EquippedItem,
    RatingStat,
)
from prestie.character.state import CharacterState

SLOT_IDS = {name: slot_id for slot_id, name in SLOT_NAMES.items()}
STAT_KEYS = {name: key for key, name in STAT_NAMES.items()}
# Several locations map to the same slots (robe and chest): the first one
# listed wins (reversed, so it is written last), and parses back to the same slots.
LOCATIONS_BY_SLOTS: Mapping[tuple[str, ...], str] = {
    slots: location for location, slots in reversed(EQUIP_LOCATION_SLOTS.items())
}
EMPTY_SOCKET_KEY = f"{EMPTY_SOCKET_PREFIX}PRISMATIC"


def state_to_snapshot(state: CharacterState) -> dict[str, Any]:
    """`PrestieDB.snapshot` without `capturedAt`, as plain JSON-ready data."""
    spec = state.spec
    return _without_none(
        {
            "character": state.character,
            "realm": state.realm,
            "level": state.level,
            "class": {"name": state.class_name, "file": state.class_token},
            "spec": (
                _without_none({"id": spec.id, "name": spec.name, "role": spec.role})
                if spec
                else None
            ),
            "heroTalent": state.hero_talent,
            "activeQuest": _active_quest(state),
            "quests": [
                _without_none(
                    {
                        "id": quest.id,
                        "title": quest.title,
                        "level": quest.level,
                        "complete": quest.complete,
                    }
                )
                for quest in state.quests
            ],
            "equipment": _equipment(state.equipment) if state.equipment else None,
            "equipmentError": state.equipment_error,
        }
    )


def _active_quest(state: CharacterState) -> dict[str, Any] | None:
    quest = state.active_quest
    if quest is None:
        return None
    objectives = [
        _without_none(
            {
                "text": objective.text,
                "finished": objective.finished,
                "fulfilled": objective.fulfilled,
                "required": objective.required,
            }
        )
        for objective in quest.objectives
    ]
    return _without_none(
        {"id": quest.id, "title": quest.title, "objectives": objectives}
    )


def _equipment(equipment: Equipment) -> dict[str, Any]:
    return _without_none(
        {
            "equipped": [_worn(item) for item in equipment.equipped],
            "bags": [_bag(item) for item in equipment.bags],
            "stats": _character_stats(equipment.stats) if equipment.stats else None,
        }
    )


def _worn(item: EquippedItem) -> dict[str, Any]:
    return _without_none(
        {"slot": SLOT_IDS[item.slot], "equipLoc": item.equip_location, **_item(item)}
    )


def _bag(item: BagItem) -> dict[str, Any]:
    return _without_none(
        {
            "equipLoc": LOCATIONS_BY_SLOTS[item.slots],
            "subType": item.sub_type,
            **_item(item),
        }
    )


def _item(item: EquippedItem | BagItem) -> dict[str, Any]:
    stats = {STAT_KEYS[name]: amount for name, amount in item.stats}
    sockets = {EMPTY_SOCKET_KEY: item.empty_sockets} if item.empty_sockets else {}
    return {
        "id": item.item_id,
        "name": item.name,
        "itemLevel": item.item_level,
        "enchant": item.enchant_id,
        "gems": item.gems,
        "stats": {**stats, **sockets},
    }


def _character_stats(stats: CharacterStats) -> dict[str, Any]:
    ratings = {
        key: _rating(rating)
        for key, rating in (
            ("crit", stats.crit),
            ("haste", stats.haste),
            ("mastery", stats.mastery),
            ("versatility", stats.versatility),
        )
        if rating is not None
    }
    return _without_none(
        {
            "itemLevel": stats.item_level,
            "equippedItemLevel": stats.equipped_item_level,
            **ratings,
        }
    )


def _rating(rating: RatingStat) -> dict[str, Any]:
    return _without_none({"rating": rating.rating, "percent": rating.percent})


def _without_none(data: Mapping[str, Any]) -> dict[str, Any]:
    # The addon omits absent fields (Lua nil) rather than writing nulls.
    return {key: value for key, value in data.items() if value is not None}
