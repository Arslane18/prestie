"""Turn a real traced question into a draft eval case.

The eval so far holds questions its author imagined; real players ask others,
in other words. A promoted trace keeps the player's exact question and the
character as exported at that moment (addon format), so the case replays the
same situation. What a good answer must contain is left to the author: the
draft carries a TODO, and the answer given at the time sits in
`previous_answer`, a starting point the judge never sees (it treats
`judge_notes` as ground truth). The eval refuses drafts until they are written.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from prestie.catalog import spec_by_id

DEFAULT_REAL_CASES = Path("evals/agent_cases_real.json")
SHORT_ID_CHARS = 8


class PromoteError(Exception):
    """The trace cannot be found or cannot become a case."""


def find_trace(traces: Sequence[Mapping[str, Any]], prefix: str) -> Mapping[str, Any]:
    matches = [t for t in traces if str(t.get("trace_id", "")).startswith(prefix)]
    if not matches:
        raise PromoteError(f"no trace id starts with {prefix!r}")
    if len(matches) > 1:
        raise PromoteError(f"several traces start with {prefix!r}: give more characters")
    return matches[0]


def draft_case(trace: Mapping[str, Any]) -> dict[str, Any]:
    started = datetime.fromisoformat(trace["started_at"])
    base = {
        "id": f"real-{started:%Y%m%d}-{trace['trace_id'][:SHORT_ID_CHARS]}",
        "question": trace["question"],
        "answerable": True,
        "expected_sources": [],
        "judge_notes": "TODO: what a good answer must contain, from the guides.",
        "previous_answer": trace.get("answer", ""),
        "asked_on": f"{started:%Y-%m-%d}",
    }
    snapshot = trace.get("character_snapshot")
    if snapshot:
        guide = spec_by_id(snapshot.get("spec", {}).get("id", 0))
        return {
            **base,
            "character": snapshot,
            "tags": ["addon", "real", "draft"],
            **({"expected_spec": guide.key} if guide else {}),
        }
    summary = trace.get("character")
    if summary and summary.get("level"):
        return {
            **base,
            "player": {"level": summary["level"], "hero_talent": summary.get("hero_talent")},
            "tags": ["manual", "real", "draft"],
        }
    raise PromoteError("the trace has no character to replay the question with")


def append_draft(path: Path, case: Mapping[str, Any]) -> None:
    cases = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if any(existing.get("id") == case["id"] for existing in cases):
        raise PromoteError(f"{case['id']} is already in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([*cases, case], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
