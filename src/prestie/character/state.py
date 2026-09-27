"""Character state exported by the Prestie addon, validated.

The addon writes `PrestieDB.snapshot` (see addon/Prestie/Prestie.lua). Names
(class, spec, quest titles) are in the game client's language; ids and the
class token (e.g. "DEATHKNIGHT") are language-independent.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from prestie.catalog import SpecGuide, spec_by_id, specs_for_class
from prestie.character.equipment import Equipment, equipment_from_snapshot
from prestie.character.fields import CharacterStateError
from prestie.character.fields import items as _items
from prestie.character.fields import mapping as _mapping
from prestie.character.fields import optional as _optional
from prestie.character.fields import require as _require

SAVED_VARIABLE = "PrestieDB"
SUPPORTED_SCHEMA = 1
NO_SPEC_ID = 0  # reported below level 10, before a specialization is chosen


@dataclass(frozen=True)
class Spec:
    id: int
    name: str | None
    role: str | None


@dataclass(frozen=True)
class QuestObjective:
    text: str
    finished: bool
    fulfilled: int | None
    required: int | None


@dataclass(frozen=True)
class ActiveQuest:
    id: int
    title: str | None
    objectives: tuple[QuestObjective, ...]


@dataclass(frozen=True)
class Quest:
    id: int
    title: str | None
    level: int | None
    complete: bool


@dataclass(frozen=True)
class CharacterState:
    character: str
    realm: str
    level: int
    class_name: str
    class_token: str
    spec: Spec | None
    hero_talent: str | None
    active_quest: ActiveQuest | None
    quests: tuple[Quest, ...]
    captured_at: datetime
    # None: not exported (addon older than the equipment export, or it failed).
    equipment: Equipment | None = None
    equipment_error: str | None = None  # the addon's error while reading it

    @property
    def guide(self) -> SpecGuide | None:
        """The knowledge base's guides for this character's spec, if covered."""
        return spec_by_id(self.spec.id) if self.spec else None

    @property
    def class_guides(self) -> tuple[SpecGuide, ...]:
        """Every covered spec of the character's class (useful before level 10)."""
        return specs_for_class(self.class_token)

    @property
    def covered_by_knowledge_base(self) -> bool:
        return self.guide is not None


def character_state_from_saved_variables(
    variables: Mapping[str, Any],
) -> CharacterState:
    """Build the state from parsed SavedVariables ({variable name: value})."""
    db = variables.get(SAVED_VARIABLE)
    if not isinstance(db, Mapping):
        raise CharacterStateError(f"{SAVED_VARIABLE} not found in the file")
    if db.get("schema") != SUPPORTED_SCHEMA:
        raise CharacterStateError(
            f"unsupported {SAVED_VARIABLE} schema {db.get('schema')!r} "
            f"(expected {SUPPORTED_SCHEMA}): update the addon or the backend"
        )
    snap = db.get("snapshot")
    if not isinstance(snap, Mapping):
        raise CharacterStateError(
            f"{SAVED_VARIABLE} has no snapshot yet: log in once, then /reload"
        )
    return _state(snap)


def _state(snap: Mapping[str, Any]) -> CharacterState:
    level = _require(snap, "level", int)
    if level < 1:
        raise CharacterStateError(f"invalid level {level}")
    wow_class = _mapping(snap, "class", required=True)
    return CharacterState(
        character=_require(snap, "character", str),
        realm=_require(snap, "realm", str),
        level=level,
        class_name=_require(wow_class, "name", str),
        class_token=_require(wow_class, "file", str),
        spec=_spec(_mapping(snap, "spec")),
        hero_talent=_optional(snap, "heroTalent", str),
        active_quest=_active_quest(_mapping(snap, "activeQuest")),
        quests=tuple(_quest(entry) for entry in _items(snap, "quests")),
        captured_at=datetime.fromtimestamp(_require(snap, "capturedAt", int), UTC),
        equipment=_equipment(_mapping(snap, "equipment")),
        equipment_error=_optional(snap, "equipmentError", str),
    )


def _equipment(raw: Mapping[str, Any] | None) -> Equipment | None:
    return None if raw is None else equipment_from_snapshot(raw)


def _spec(spec: Mapping[str, Any] | None) -> Spec | None:
    if not spec or spec.get("id", NO_SPEC_ID) == NO_SPEC_ID:
        return None
    return Spec(
        id=_require(spec, "id", int),
        name=_optional(spec, "name", str),
        role=_optional(spec, "role", str),
    )


def _active_quest(quest: Mapping[str, Any] | None) -> ActiveQuest | None:
    if not quest:
        return None
    return ActiveQuest(
        id=_require(quest, "id", int),
        title=_optional(quest, "title", str),
        objectives=tuple(_objective(entry) for entry in _items(quest, "objectives")),
    )


def _objective(entry: Any) -> QuestObjective:
    if not isinstance(entry, Mapping):
        raise CharacterStateError(f"invalid objective entry: {entry!r}")
    return QuestObjective(
        text=_optional(entry, "text", str) or "",
        finished=bool(entry.get("finished")),
        fulfilled=_optional(entry, "fulfilled", int),
        required=_optional(entry, "required", int),
    )


def _quest(entry: Any) -> Quest:
    if not isinstance(entry, Mapping):
        raise CharacterStateError(f"invalid quest entry: {entry!r}")
    return Quest(
        id=_require(entry, "id", int),
        title=_optional(entry, "title", str),
        level=_optional(entry, "level", int),
        complete=bool(entry.get("complete")),
    )
