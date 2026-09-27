"""Equipment exported by the Prestie addon: worn items, bag candidates, stats.

The addon exports what the game API returns, untranslated: inventory slot ids,
`INVTYPE_*` equip locations and `GetItemStats` keys (e.g.
"ITEM_MOD_CRIT_RATING_SHORT"). This module turns them into the names the
guides use (slot "head", stat "critical_strike") and drops what the guides
never rank (armor, resistances).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from prestie.character.fields import (
    CharacterStateError,
    items,
    mapping,
    optional,
    require,
)

# Inventory slot ids (INVSLOT_*); shirt (4) and tabard (19) are cosmetic.
SLOT_NAMES: Mapping[int, str] = MappingProxyType(
    {
        1: "head",
        2: "neck",
        3: "shoulder",
        5: "chest",
        6: "waist",
        7: "legs",
        8: "feet",
        9: "wrist",
        10: "hands",
        11: "finger1",
        12: "finger2",
        13: "trinket1",
        14: "trinket2",
        15: "back",
        16: "main_hand",
        17: "off_hand",
    }
)
EQUIP_LOCATION_SLOTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "INVTYPE_HEAD": ("head",),
        "INVTYPE_NECK": ("neck",),
        "INVTYPE_SHOULDER": ("shoulder",),
        "INVTYPE_CHEST": ("chest",),
        "INVTYPE_ROBE": ("chest",),
        "INVTYPE_WAIST": ("waist",),
        "INVTYPE_LEGS": ("legs",),
        "INVTYPE_FEET": ("feet",),
        "INVTYPE_WRIST": ("wrist",),
        "INVTYPE_HAND": ("hands",),
        "INVTYPE_FINGER": ("finger1", "finger2"),
        "INVTYPE_TRINKET": ("trinket1", "trinket2"),
        "INVTYPE_CLOAK": ("back",),
        "INVTYPE_WEAPON": ("main_hand", "off_hand"),
        "INVTYPE_2HWEAPON": ("main_hand",),
        "INVTYPE_WEAPONMAINHAND": ("main_hand",),
        "INVTYPE_RANGED": ("main_hand",),
        "INVTYPE_RANGEDRIGHT": ("main_hand",),
        "INVTYPE_WEAPONOFFHAND": ("off_hand",),
        "INVTYPE_SHIELD": ("off_hand",),
        "INVTYPE_HOLDABLE": ("off_hand",),
    }
)
STAT_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "ITEM_MOD_STRENGTH_SHORT": "strength",
        "ITEM_MOD_AGILITY_SHORT": "agility",
        "ITEM_MOD_INTELLECT_SHORT": "intellect",
        "ITEM_MOD_STAMINA_SHORT": "stamina",
        "ITEM_MOD_CRIT_RATING_SHORT": "critical_strike",
        "ITEM_MOD_HASTE_RATING_SHORT": "haste",
        "ITEM_MOD_MASTERY_RATING_SHORT": "mastery",
        "ITEM_MOD_VERSATILITY": "versatility",
    }
)
EMPTY_SOCKET_PREFIX = "EMPTY_SOCKET_"
SECONDARY_STATS = ("crit", "haste", "mastery", "versatility")


@dataclass(frozen=True)
class EquippedItem:
    slot: str
    item_id: int
    name: str | None
    item_level: int | None
    stats: Mapping[str, int]  # guide stat name -> amount on the item
    enchant_id: int | None
    gems: int
    empty_sockets: int


@dataclass(frozen=True)
class BagItem:
    item_id: int
    name: str | None
    item_level: int | None
    slots: tuple[str, ...]  # where it could be equipped
    sub_type: str | None  # armor or weapon type, in the client's language
    stats: Mapping[str, int]
    enchant_id: int | None
    gems: int
    empty_sockets: int


@dataclass(frozen=True)
class RatingStat:
    rating: int | None
    percent: float | None


@dataclass(frozen=True)
class CharacterStats:
    item_level: float | None  # average over the best items owned
    equipped_item_level: float | None
    crit: RatingStat | None
    haste: RatingStat | None
    mastery: RatingStat | None
    versatility: RatingStat | None


@dataclass(frozen=True)
class Equipment:
    equipped: tuple[EquippedItem, ...]
    bags: tuple[BagItem, ...]
    stats: CharacterStats | None


def equipment_from_snapshot(raw: Mapping[str, Any]) -> Equipment:
    """Validate the addon's `equipment` table."""
    bag_items = (_bag_item(_entry(entry, "bag item")) for entry in items(raw, "bags"))
    return Equipment(
        equipped=tuple(
            _equipped(_entry(entry, "equipped item"))
            for entry in items(raw, "equipped")
        ),
        bags=tuple(item for item in bag_items if item.slots),
        stats=_character_stats(mapping(raw, "stats")),
    )


def _equipped(entry: Mapping[str, Any]) -> EquippedItem:
    slot_id = require(entry, "slot", int)
    if slot_id not in SLOT_NAMES:
        raise CharacterStateError(f"unknown equipment slot {slot_id}")
    stats, empty_sockets = _stats(mapping(entry, "stats"))
    return EquippedItem(
        slot=SLOT_NAMES[slot_id],
        item_id=require(entry, "id", int),
        name=optional(entry, "name", str),
        item_level=optional(entry, "itemLevel", int),
        stats=stats,
        enchant_id=optional(entry, "enchant", int),
        gems=optional(entry, "gems", int) or 0,
        empty_sockets=empty_sockets,
    )


def _bag_item(entry: Mapping[str, Any]) -> BagItem:
    stats, empty_sockets = _stats(mapping(entry, "stats"))
    return BagItem(
        item_id=require(entry, "id", int),
        name=optional(entry, "name", str),
        item_level=optional(entry, "itemLevel", int),
        slots=EQUIP_LOCATION_SLOTS.get(require(entry, "equipLoc", str), ()),
        sub_type=optional(entry, "subType", str),
        stats=stats,
        enchant_id=optional(entry, "enchant", int),
        gems=optional(entry, "gems", int) or 0,
        empty_sockets=empty_sockets,
    )


def _stats(raw: Mapping[str, Any] | None) -> tuple[Mapping[str, int], int]:
    """The ranked stats of an item, and its number of empty sockets."""
    raw = raw or {}
    stats = {
        STAT_NAMES[key]: require(raw, key, int) for key in raw if key in STAT_NAMES
    }
    empty_sockets = sum(
        require(raw, key, int) for key in raw if key.startswith(EMPTY_SOCKET_PREFIX)
    )
    return MappingProxyType(stats), empty_sockets


def _character_stats(raw: Mapping[str, Any] | None) -> CharacterStats | None:
    if raw is None:
        return None
    ratings = {name: _rating(mapping(raw, name)) for name in SECONDARY_STATS}
    return CharacterStats(
        item_level=_number(raw, "itemLevel"),
        equipped_item_level=_number(raw, "equippedItemLevel"),
        **ratings,
    )


def _rating(raw: Mapping[str, Any] | None) -> RatingStat | None:
    if raw is None:
        return None
    return RatingStat(
        rating=optional(raw, "rating", int), percent=_number(raw, "percent")
    )


def _number(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CharacterStateError(f"'{key}' must be a number, got {value!r}")
    return value


def _entry(entry: Any, kind: str) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise CharacterStateError(f"invalid {kind}: {entry!r}")
    return entry
