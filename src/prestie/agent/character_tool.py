"""The agent's `get_character_state` tool: what the addon last exported.

The tool takes no input: Claude calls it to learn about the character instead
of asking the player. Its output is plain text wrapped in a tag carrying the
snapshot's age, because the state is only as fresh as the last /reload.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from prestie.agent.tools import ToolOutcome
from prestie.character.state import CharacterState, CharacterStateError, Quest

CHARACTER_STATE_TOOL_NAME = "get_character_state"
MAX_QUESTS_LISTED = 25
SECONDS_PER_MINUTE = 60

CHARACTER_STATE_TOOL: dict[str, Any] = {
    "name": CHARACTER_STATE_TOOL_NAME,
    "description": (
        "Returns the player's character as exported by the Prestie in-game addon: "
        "name, level, class, specialization, hero talent, the tracked quest with "
        "its objectives, and the quest log. Call it before answering whenever the "
        "answer depends on the character, instead of asking the player. The game "
        "only saves this snapshot on /reload or logout: age_minutes tells how old "
        "it is. Class, spec and quest names are in the game client's language."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


class ProvidesCharacterState(Protocol):
    def latest(self) -> CharacterState: ...


class CharacterStateTool:
    definition = CHARACTER_STATE_TOOL

    def __init__(
        self,
        source: ProvidesCharacterState,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._source = source
        self._now = now

    def run(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        """Never raises: a missing or invalid export goes back as an error result."""
        try:
            state = self._source.latest()
        except CharacterStateError as exc:
            return ToolOutcome(f"Character state unavailable: {exc}", is_error=True)
        return ToolOutcome(format_state(state, self._now()))


def format_state(state: CharacterState, now: datetime) -> str:
    age_seconds = max(0, int((now - state.captured_at).total_seconds()))
    captured = state.captured_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f'<character_state captured_at="{captured}" '
        f'age_minutes="{age_seconds // SECONDS_PER_MINUTE}">',
        f"Character: {state.character} ({state.realm})",
        f"Level: {state.level}",
        f"Class: {state.class_name} ({state.class_token})",
        f"Specialization: {_spec(state)}",
        f"Hero talent: {state.hero_talent or 'none'}",
        f"Knowledge base: {_coverage(state)}",
        *_tracked_quest(state),
        *_quest_log(state.quests),
        "</character_state>",
    ]
    return "\n".join(lines)


def _spec(state: CharacterState) -> str:
    if state.spec is None:
        return "none reported by the game"
    details = ", ".join(
        part for part in (f"spec id {state.spec.id}", state.spec.role) if part
    )
    return f"{state.spec.name or '?'} ({details})"


def _coverage(state: CharacterState) -> str:
    if state.covered_by_knowledge_base:
        return "covered (Blood Death Knight)"
    return "not covered (the guides only cover Blood Death Knight)"


def _tracked_quest(state: CharacterState) -> list[str]:
    quest = state.active_quest
    if quest is None:
        return ["Tracked quest: none"]
    objectives = [
        f"  - {objective.text}{' (done)' if objective.finished else ''}"
        for objective in quest.objectives
    ]
    return [f"Tracked quest: [{quest.id}] {quest.title or '?'}", *objectives]


def _quest_log(quests: tuple[Quest, ...]) -> list[str]:
    if not quests:
        return ["Quest log: empty"]
    shown = quests[:MAX_QUESTS_LISTED]
    header = f"Quest log ({len(quests)} quests"
    header += f", first {len(shown)} shown):" if len(shown) < len(quests) else "):"
    return [
        header,
        *(
            f"  - [{quest.id}] {quest.title or '?'}"
            f"{' (complete)' if quest.complete else ''}"
            for quest in shown
        ),
    ]
