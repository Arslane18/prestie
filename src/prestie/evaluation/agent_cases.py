"""Test cases for the agent evaluation (evals/agent_cases.json).

Unlike retrieval cases, an agent case carries the player context the agent is
built with, and what a good answer must do: search or not, cite sources or
admit the gap, copy some strings verbatim.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prestie.agent.prompts import PlayerContext
from prestie.evaluation.retrieval import EvalCaseError


@dataclass(frozen=True)
class AgentCase:
    id: str
    question: str
    level: int
    hero_talent: str | None
    tags: tuple[str, ...]
    should_search: bool | None  # None: either is acceptable
    answerable: bool  # False: the sources do not cover it, the agent must say so
    expected_sources: tuple[str, ...]  # "page_slug#anchor" or "page_slug"
    must_include: tuple[str, ...]  # strings to copy verbatim (codes, macros)
    judge_notes: str
    detail_requested: bool = False  # True: the length cap does not apply

    def player(self) -> PlayerContext:
        return PlayerContext(level=self.level, hero_talent=self.hero_talent)


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
    player = entry.get("player")
    if not isinstance(player, dict) or not isinstance(player.get("level"), int):
        raise EvalCaseError(f"{where}: 'player' must be an object with an int 'level'")
    case = AgentCase(
        id=entry["id"],
        question=entry["question"],
        level=player["level"],
        hero_talent=player.get("hero_talent"),
        tags=_strings(entry, "tags", where),
        should_search=entry.get("should_search"),
        answerable=bool(entry.get("answerable", True)),
        expected_sources=_strings(entry, "expected_sources", where),
        must_include=_strings(entry, "must_include", where),
        judge_notes=str(entry.get("judge_notes", "")),
        detail_requested=bool(entry.get("detail_requested", False)),
    )
    try:
        case.player()
    except ValueError as exc:
        raise EvalCaseError(f"{where}: {exc}") from exc
    return case


def _strings(entry: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise EvalCaseError(f"{where}: '{key}' must be a list of strings")
    return tuple(value)
