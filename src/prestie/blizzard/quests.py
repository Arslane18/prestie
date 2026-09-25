"""Quest details from the Game Data API, cached on disk.

The quest endpoint gives official, static facts: title, zone, description,
level range, class restriction and rewards. It has no objectives, NPCs or
walkthrough. Quest data only changes with game patches, so each raw payload is
cached on disk (like the Icy Veins HTML): refresh by deleting the cache dir.

We keep English (the knowledge base's language) and French (the player's
client) out of the twelve locales the API returns.
"""

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from prestie.blizzard.client import BlizzardApiError

QUEST_PATH = "/data/wow/quest/{quest_id}"


@dataclass(frozen=True)
class Localized:
    en: str | None
    fr: str | None


@dataclass(frozen=True)
class QuestDetails:
    id: int
    title: Localized
    area: Localized | None
    description: Localized | None
    min_level: int | None
    max_level: int | None
    classes: tuple[Localized, ...]
    experience: int | None
    money_copper: int | None
    item_choices: tuple[Localized, ...]
    spell: Localized | None


def quest_details_from_api(payload: Mapping[str, Any]) -> QuestDetails:
    quest_id = _int(payload.get("id"))
    if quest_id is None:
        raise BlizzardApiError(
            f"quest payload without a numeric id: {payload.get('id')!r}"
        )
    requirements = _mapping(payload.get("requirements"))
    rewards = _mapping(payload.get("rewards"))
    return QuestDetails(
        id=quest_id,
        title=_localized(payload.get("title")) or Localized(None, None),
        area=_localized(_mapping(payload.get("area")).get("name")),
        description=_localized(payload.get("description")),
        min_level=_int(requirements.get("min_character_level")),
        max_level=_int(requirements.get("max_character_level")),
        classes=_names(requirements.get("classes")),
        experience=_int(rewards.get("experience")),
        money_copper=_int(_mapping(rewards.get("money")).get("value")),
        item_choices=_names(
            [
                _mapping(entry).get("item")
                for entry in _list(_mapping(rewards.get("items")).get("choice_of"))
            ]
        ),
        spell=_localized(_mapping(rewards.get("spell")).get("name")),
    )


def _localized(value: Any) -> Localized | None:
    if isinstance(value, str):
        # Single-locale form (?locale=fr_FR); we only ever ask for French that way.
        return Localized(en=None, fr=value)
    en, fr = _mapping(value).get("en_US"), _mapping(value).get("fr_FR")
    if not isinstance(en, str) and not isinstance(fr, str):
        return None
    return Localized(
        en=en if isinstance(en, str) else None,
        fr=fr if isinstance(fr, str) else None,
    )


def _names(entries: Any) -> tuple[Localized, ...]:
    names = (_localized(_mapping(entry).get("name")) for entry in _list(entries))
    return tuple(name for name in names if name is not None)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class QuestCache:
    """Raw API payloads: <root>/<region>/<quest id>.json with the fetch date."""

    def __init__(self, root: Path):
        self._root = root

    def get(self, region: str, quest_id: int) -> dict[str, Any] | None:
        try:
            entry = json.loads(self._path(region, quest_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # absent or corrupted: fetch again
        payload = entry.get("payload") if isinstance(entry, dict) else None
        return payload if isinstance(payload, dict) else None

    def put(
        self,
        region: str,
        quest_id: int,
        payload: Mapping[str, Any],
        *,
        fetched_at: datetime,
    ) -> None:
        path = self._path(region, quest_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"fetched_at": fetched_at.isoformat(), "payload": payload}
        # Write then rename: a crash never leaves a half-written entry behind.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(temporary, path)

    def _path(self, region: str, quest_id: int) -> Path:
        return self._root / region / f"{quest_id}.json"


class FetchesStaticData(Protocol):
    @property
    def region(self) -> str: ...

    def get_static(self, path: str) -> dict[str, Any]: ...


class QuestRepository:
    """Cache first, then the API; failures are not cached."""

    def __init__(
        self,
        client: FetchesStaticData,
        cache: QuestCache,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._client = client
        self._cache = cache
        self._now = now

    def get(self, quest_id: int) -> QuestDetails:
        region = self._client.region
        payload = self._cache.get(region, quest_id)
        if payload is None:
            payload = self._client.get_static(QUEST_PATH.format(quest_id=quest_id))
            self._cache.put(region, quest_id, payload, fetched_at=self._now())
        return quest_details_from_api(payload)
