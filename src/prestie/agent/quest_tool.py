"""The agent's `get_quest_details` tool: official quest facts by id.

Claude gets quest ids from `get_character_state` (tracked quest, quest log)
and chains this tool to learn what a quest is about. The Blizzard API only
has static facts, so the output says explicitly what it does not include.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from prestie.agent.tools import ToolOutcome
from prestie.blizzard.client import BlizzardApiError, BlizzardNotFoundError
from prestie.blizzard.quests import Localized, QuestDetails

QUEST_DETAILS_TOOL_NAME = "get_quest_details"
MAX_QUEST_ID = 2**31 - 1  # quest ids are 32-bit signed integers in the game
COPPER_PER_SILVER = 100
COPPER_PER_GOLD = 10_000

QUEST_DETAILS_TOOL: dict[str, Any] = {
    "name": QUEST_DETAILS_TOOL_NAME,
    "description": (
        "Returns official facts about one quest from the Blizzard Game Data API: "
        "title and zone (French and English), level range, class restriction, the "
        "quest giver's text and rewards. Use the quest ids given by "
        "get_character_state. It has no objectives, NPC locations or walkthrough: "
        "the tracked quest's objectives are in get_character_state."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "quest_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Quest id, e.g. 55763 (shown as [55763] in the "
                "character state).",
            }
        },
        "required": ["quest_id"],
    },
}


class ProvidesQuests(Protocol):
    def get(self, quest_id: int) -> QuestDetails: ...


class QuestDetailsTool:
    definition = QUEST_DETAILS_TOOL

    def __init__(self, repository: ProvidesQuests):
        self._repository = repository

    def run(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        """Never raises: API failures go back to Claude as an error result."""
        quest_id = tool_input.get("quest_id")
        if not _valid_quest_id(quest_id):
            return ToolOutcome(
                "Invalid input: 'quest_id' must be a positive integer.", is_error=True
            )
        try:
            quest = self._repository.get(quest_id)
        except BlizzardNotFoundError:
            return ToolOutcome(
                f"Quest {quest_id} not found in the Blizzard API (hidden, removed "
                "or not a quest id).",
                is_error=True,
            )
        except BlizzardApiError as exc:
            return ToolOutcome(f"Quest details unavailable: {exc}", is_error=True)
        return ToolOutcome(format_quest(quest))


def _valid_quest_id(value: Any) -> bool:
    # bool is an int subclass: reject True/False explicitly.
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_QUEST_ID
    )


def format_quest(quest: QuestDetails) -> str:
    classes = ", ".join(_fr(name) for name in quest.classes)
    lines = [
        f'<quest_details id="{quest.id}" source="Blizzard Game Data API">',
        f"Title: {_both(quest.title)}",
        *([f"Zone: {_both(quest.area)}"] if quest.area else []),
        *_level_range(quest),
        *([f"Classes: {classes}"] if classes else []),
        f"Rewards: {_rewards(quest)}",
        *_description(quest.description),
        "Not included: objectives, NPC and map locations, walkthrough.",
        "</quest_details>",
    ]
    return "\n".join(lines)


def _both(text: Localized) -> str:
    if text.fr and text.en and text.fr != text.en:
        return f"{text.fr} (English: {text.en})"
    return text.fr or text.en or "?"


def _fr(text: Localized) -> str:
    return text.fr or text.en or "?"


def _level_range(quest: QuestDetails) -> list[str]:
    if quest.min_level is None and quest.max_level is None:
        return []
    low = quest.min_level if quest.min_level is not None else "?"
    high = quest.max_level if quest.max_level is not None else "?"
    return [f"Level range: {low}-{high}"]


def _rewards(quest: QuestDetails) -> str:
    items = " / ".join(_fr(item) for item in quest.item_choices)
    parts = [
        *([f"{quest.experience} XP"] if quest.experience else []),
        *([_money(quest.money_copper)] if quest.money_copper else []),
        *([f"spell: {_fr(quest.spell)}"] if quest.spell else []),
        *([f"choice of: {items}"] if items else []),
    ]
    return ", ".join(parts) or "none listed"


def _money(total_copper: int) -> str:
    gold, rest = divmod(total_copper, COPPER_PER_GOLD)
    silver, copper = divmod(rest, COPPER_PER_SILVER)
    amounts = ((gold, "g"), (silver, "s"), (copper, "c"))
    return " ".join(f"{amount} {unit}" for amount, unit in amounts if amount)


def _description(description: Localized | None) -> list[str]:
    text = _fr(description) if description else ""
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("{name}", "[character name]")
    return ["Quest giver's text:", normalized.strip()]
