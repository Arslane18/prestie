"""The agent's `get_equipment` tool: worn gear, bag candidates, secondary stats.

Like the character state, it takes no input and reads the addon's last export.
It is a separate tool so that gear, the largest part of the export, is only
read when a question is about gear. Comparisons that need no judgment (item
level gained per slot, empty slots) are computed here, so the model spends its
reasoning on what the guides say instead of on arithmetic.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from prestie.agent.character_tool import (
    SECONDS_PER_MINUTE,
    ProvidesCharacterState,
    knowledge_base_coverage,
    served_state_details,
)
from prestie.agent.tools import ToolOutcome
from prestie.character.equipment import (
    SLOT_NAMES,
    BagItem,
    CharacterStats,
    Equipment,
    EquippedItem,
    RatingStat,
)
from prestie.character.state import CharacterState, CharacterStateError

EQUIPMENT_TOOL_NAME = "get_equipment"
MAX_BAG_ITEMS = 20
STAT_LABELS: Mapping[str, str] = {
    "strength": "Strength",
    "agility": "Agility",
    "intellect": "Intellect",
    "stamina": "Stamina",
    "critical_strike": "Critical Strike",
    "haste": "Haste",
    "mastery": "Mastery",
    "versatility": "Versatility",
}
NOT_EXPORTED_MESSAGE = (
    "Equipment not exported: update the Prestie addon, then /reload in game."
)

EQUIPMENT_TOOL: dict[str, Any] = {
    "name": EQUIPMENT_TOOL_NAME,
    "description": (
        "Returns the player's gear as exported by the Prestie in-game addon: "
        "average item level, secondary stat ratings and percentages, each worn "
        "item (slot, item level, stats, enchant, gems) and the equippable items "
        "in the bags with the item level they would gain over what they "
        "replace. Call it for any question about gear, upgrades or stats on "
        "items, instead of asking the player. Like the character state, it is "
        "only as fresh as the last /reload (age_minutes). Item names are in the "
        "game client's language; stat names use the guides' English names."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


class EquipmentTool:
    definition = EQUIPMENT_TOOL

    def __init__(
        self,
        source: ProvidesCharacterState,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._source = source
        self._now = now

    def run(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        """Never raises: a missing export goes back as an error result."""
        try:
            state = self._source.latest()
        except CharacterStateError as exc:
            return ToolOutcome(f"Character state unavailable: {exc}", is_error=True)
        if state.equipment is None:
            if state.equipment_error:
                return ToolOutcome(
                    f"The addon failed to read the equipment: {state.equipment_error}",
                    is_error=True,
                )
            return ToolOutcome(NOT_EXPORTED_MESSAGE, is_error=True)
        return ToolOutcome(
            format_equipment(state, state.equipment, self._now()),
            details=served_state_details(state),
        )


def format_equipment(state: CharacterState, gear: Equipment, now: datetime) -> str:
    age_seconds = max(0, int((now - state.captured_at).total_seconds()))
    captured = state.captured_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    worn = {item.slot: item for item in gear.equipped}
    main_hand = worn.get("main_hand")
    two_handed = main_hand is not None and main_hand.two_handed
    empty = [
        slot
        for slot in SLOT_NAMES.values()
        if slot not in worn and not (slot == "off_hand" and two_handed)
    ]
    spec = state.spec.name if state.spec and state.spec.name else "no spec"
    return "\n".join(
        [
            f'<equipment captured_at="{captured}" '
            f'age_minutes="{age_seconds // SECONDS_PER_MINUTE}">',
            f"Character: level {state.level} {state.class_name}, {spec} "
            f"({knowledge_base_coverage(state)})",
            *_character_stats(gear.stats),
            "Equipped items:",
            *(_worn_line(item) for item in gear.equipped),
            f"Empty slots: {', '.join(empty) if empty else 'none'}",
            *(
                ["Off hand: none needed (two-handed weapon)"]
                if two_handed and "off_hand" not in worn
                else []
            ),
            *_bag_lines(gear.bags, worn),
            "</equipment>",
        ]
    )


def _character_stats(stats: CharacterStats | None) -> list[str]:
    if stats is None:
        return ["Item level and secondary stats: not exported"]
    ratings = [
        _rating(label, rating)
        for label, rating in (
            ("Critical Strike", stats.crit),
            ("Haste", stats.haste),
            ("Mastery", stats.mastery),
            ("Versatility", stats.versatility),
        )
        if rating is not None
    ]
    return [
        f"Item level: {_level(stats.equipped_item_level)} equipped "
        f"({_level(stats.item_level)} owned)",
        f"Secondary stats: {', '.join(ratings) if ratings else 'not exported'}",
    ]


def _rating(label: str, stat: RatingStat) -> str:
    rating = "?" if stat.rating is None else str(stat.rating)
    percent = "" if stat.percent is None else f" ({stat.percent:.1f}%)"
    return f"{label} {rating}{percent}"


def _level(value: float | None) -> str:
    # Truncated like the game's character sheet, which never rounds up.
    return "?" if value is None else f"{int(value * 10) / 10:.1f}"


def _worn_line(item: EquippedItem) -> str:
    return f"- {item.slot}: {_describe(item)}"


def _describe(item: EquippedItem | BagItem) -> str:
    stats = ", ".join(
        f"{STAT_LABELS[name]} {amount}" for name, amount in item.stats
    )
    # Only a present enchant is reported: which slots take one depends on the
    # patch, and the guides (gems/enchants pages) say it, not the tool.
    extras = ["enchanted"] if item.enchant_id else []
    if isinstance(item, EquippedItem) and item.two_handed:
        extras.append("two-handed")
    if item.gems:
        extras.append(f"{item.gems} gem{'s' if item.gems > 1 else ''}")
    if item.empty_sockets:
        plural = "s" if item.empty_sockets > 1 else ""
        extras.append(f"{item.empty_sockets} empty socket{plural}")
    level = "?" if item.item_level is None else item.item_level
    sub_type = item.sub_type if isinstance(item, BagItem) else None
    parts = [
        f"{item.name or f'item {item.item_id}'}{f' ({sub_type})' if sub_type else ''}"
        f", item level {level}",
        stats or "no stats",
        *([", ".join(extras)] if extras else []),
    ]
    return "; ".join(parts)


def _bag_lines(
    bags: tuple[BagItem, ...], worn: Mapping[str, EquippedItem]
) -> list[str]:
    if not bags:
        return ["Equippable items in bags: none"]
    ranked = sorted(bags, key=lambda item: item.item_level or 0, reverse=True)
    shown = ranked[:MAX_BAG_ITEMS]
    lines = [
        "Equippable items in bags:",
        *(f"- {_describe(item)}; {_comparisons(item, worn)}" for item in shown),
    ]
    if len(ranked) > len(shown):
        lines.append(f"({len(ranked) - len(shown)} more not listed)")
    return lines


def _comparisons(item: BagItem, worn: Mapping[str, EquippedItem]) -> str:
    return "; ".join(_compare(slot, item, worn.get(slot)) for slot in item.slots)


def _compare(slot: str, item: BagItem, current: EquippedItem | None) -> str:
    if current is None:
        return f"{slot}: slot empty"
    if item.item_level is None or current.item_level is None:
        return f"{slot}: vs {current.name or current.item_id} (item level unknown)"
    delta = item.item_level - current.item_level
    return (
        f"{slot}: {delta:+d} item levels vs {current.name or current.item_id} "
        f"({current.item_level})"
    )
