"""Test cases for the agent evaluation (evals/agent_cases*.json).

Unlike retrieval cases, an agent case carries the player context the agent is
built with, and what a good answer must do: search or not, cite sources or
admit the gap, copy some strings verbatim.

A case runs in one of two modes, like `prestie chat`:
- manual: `"player": {"level": ..., "hero_talent": ...}` is written in the
  system prompt;
- addon: `"character": {...}` is an addon snapshot (same schema as
  PrestieDB.snapshot) served by the get_character_state tool, or null when the
  export is unavailable. These cases can also expect the agent to read the
  state (`should_read_state`) and to look up quests (`expected_quest_ids`).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from prestie.agent.prompts import PlayerContext
from prestie.catalog import spec_by_key
from prestie.character.state import (
    SAVED_VARIABLE,
    SUPPORTED_SCHEMA,
    CharacterState,
    CharacterStateError,
    character_state_from_saved_variables,
)
from prestie.evaluation.retrieval import EvalCaseError

DEFAULT_CHARACTER_AGE_MINUTES = 5
ADDON_ONLY_KEYS = ("should_read_state", "expected_quest_ids", "character_age_minutes")
UNAVAILABLE_STATE_MESSAGE = "Prestie.lua not found: install the addon, then /reload"


@dataclass(frozen=True)
class AgentCase:
    id: str
    question: str
    level: int | None
    hero_talent: str | None
    tags: tuple[str, ...]
    should_search: bool | None  # None: either is acceptable
    answerable: bool  # False: the sources do not cover it, the agent must say so
    expected_sources: tuple[str, ...]  # "page_slug#anchor" or "page_slug"
    must_include: tuple[str, ...]  # strings to copy verbatim (codes, macros)
    judge_notes: str
    detail_requested: bool = False  # True: the length cap does not apply
    addon_mode: bool = False
    character: CharacterState | None = None  # addon mode; None: unavailable
    character_age_minutes: int = DEFAULT_CHARACTER_AGE_MINUTES
    should_read_state: bool | None = None  # None: either is acceptable
    expected_quest_ids: tuple[int, ...] = ()
    # Catalog key every search must filter on (e.g. "shadow-priest"), or None.
    expected_spec: str | None = None

    def player(self) -> PlayerContext | None:
        """The manual-mode context; None in addon mode (the agent reads the state)."""
        if self.addon_mode:
            return None
        return PlayerContext(level=self.level or 0, hero_talent=self.hero_talent)

    def describe_player(self) -> str:
        player = self.player()
        if player is not None:
            return player.describe()
        if self.character is None:
            return "Addon mode: the character state is unavailable (the tool fails)."
        return (
            "Addon mode: the assistant can read the character through "
            "get_character_state (see its output in the tool results)."
        )

    def character_source(self, now: datetime) -> "FixedCharacterSource":
        """Serves the case's character as if exported `character_age_minutes` ago."""
        if self.character is None:
            return FixedCharacterSource(None)
        captured_at = now - timedelta(minutes=self.character_age_minutes)
        return FixedCharacterSource(replace(self.character, captured_at=captured_at))


@dataclass(frozen=True)
class FixedCharacterSource:
    state: CharacterState | None

    def latest(self) -> CharacterState:
        if self.state is None:
            raise CharacterStateError(UNAVAILABLE_STATE_MESSAGE)
        return self.state


def load_agent_cases(path: Path) -> tuple[AgentCase, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalCaseError(f"Cannot read agent cases from {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise EvalCaseError(f"{path}: expected a JSON list of cases")
    cases = tuple(_parse(entry, f"{path} case #{i}") for i, entry in enumerate(payload))
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise EvalCaseError(f"{path}: duplicate case ids: {', '.join(duplicates)}")
    return cases


def _parse(entry: Any, where: str) -> AgentCase:
    if not isinstance(entry, dict):
        raise EvalCaseError(f"{where}: expected an object")
    for key in ("id", "question"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise EvalCaseError(f"{where}: '{key}' must be a non-empty string")
    if "player" in entry and "character" in entry:
        raise EvalCaseError(f"{where}: give either 'player' or 'character', not both")
    case = AgentCase(
        id=entry["id"],
        question=entry["question"],
        level=None,
        hero_talent=None,
        tags=_strings(entry, "tags", where),
        should_search=entry.get("should_search"),
        answerable=bool(entry.get("answerable", True)),
        expected_sources=_strings(entry, "expected_sources", where),
        must_include=_strings(entry, "must_include", where),
        judge_notes=str(entry.get("judge_notes", "")),
        detail_requested=bool(entry.get("detail_requested", False)),
        expected_spec=_expected_spec(entry, where),
    )
    if "character" in entry:
        return _with_character(case, entry, where)
    return _with_player(case, entry, where)


def _with_player(case: AgentCase, entry: dict[str, Any], where: str) -> AgentCase:
    addon_keys = [key for key in ADDON_ONLY_KEYS if key in entry]
    if addon_keys:
        raise EvalCaseError(
            f"{where}: {', '.join(addon_keys)} need a 'character' (addon mode)"
        )
    player = entry.get("player")
    if not isinstance(player, dict) or not isinstance(player.get("level"), int):
        raise EvalCaseError(f"{where}: 'player' must be an object with an int 'level'")
    case = replace(case, level=player["level"], hero_talent=player.get("hero_talent"))
    try:
        case.player()
    except ValueError as exc:
        raise EvalCaseError(f"{where}: {exc}") from exc
    return case


def _with_character(case: AgentCase, entry: dict[str, Any], where: str) -> AgentCase:
    age = entry.get("character_age_minutes", DEFAULT_CHARACTER_AGE_MINUTES)
    if not isinstance(age, int) or isinstance(age, bool) or age < 0:
        raise EvalCaseError(f"{where}: 'character_age_minutes' must be an int >= 0")
    should_read = entry.get("should_read_state")
    if should_read is not None and not isinstance(should_read, bool):
        raise EvalCaseError(f"{where}: 'should_read_state' must be true, false or null")
    quest_ids = entry.get("expected_quest_ids", [])
    if not isinstance(quest_ids, list) or not all(
        isinstance(q, int) and not isinstance(q, bool) for q in quest_ids
    ):
        raise EvalCaseError(f"{where}: 'expected_quest_ids' must be a list of ints")
    return replace(
        case,
        addon_mode=True,
        character=_character(entry["character"], where),
        character_age_minutes=age,
        should_read_state=should_read,
        expected_quest_ids=tuple(quest_ids),
    )


def _character(snapshot: Any, where: str) -> CharacterState | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise EvalCaseError(f"{where}: 'character' must be an object or null")
    # capturedAt is set at run time (character_source), so cases do not age.
    data = {"schema": SUPPORTED_SCHEMA, "snapshot": {"capturedAt": 0, **snapshot}}
    try:
        return character_state_from_saved_variables({SAVED_VARIABLE: data})
    except CharacterStateError as exc:
        raise EvalCaseError(f"{where}: invalid 'character': {exc}") from exc


def _expected_spec(entry: dict[str, Any], where: str) -> str | None:
    key = entry.get("expected_spec")
    if key is not None and (not isinstance(key, str) or spec_by_key(key) is None):
        raise EvalCaseError(f"{where}: 'expected_spec' {key!r} is not a covered spec")
    return key


def _strings(entry: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise EvalCaseError(f"{where}: '{key}' must be a list of strings")
    return tuple(value)
