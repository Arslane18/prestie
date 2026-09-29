"""Local trace storage: one JSON line per turn, one file per day.

JSONL is append-only (a crash never corrupts earlier lines), greppable, and
loads line by line. The directory is under data/, which git ignores: traces
hold the player's questions and stay on their machine.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_TRACE_DIR = Path("data/traces")
TRACE_FILE_SUFFIX = ".jsonl"


class Serializable(Protocol):
    @property
    def started_at(self) -> Any: ...

    def to_json(self) -> Mapping[str, Any]: ...


class JsonlTraceStore:
    def __init__(self, directory: Path = DEFAULT_TRACE_DIR):
        self._directory = directory

    def write(self, trace: Serializable) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{trace.started_at:%Y-%m-%d}{TRACE_FILE_SUFFIX}"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_json(), ensure_ascii=False) + "\n")

    def load(self) -> "LoadedTraces":
        """Every trace, oldest day first (file names sort by date).

        A line cut by a crash mid-write is skipped and counted, never fatal.
        """
        traces: list[dict[str, Any]] = []
        damaged = 0
        if self._directory.exists():
            for path in sorted(self._directory.glob(f"*{TRACE_FILE_SUFFIX}")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        damaged += 1
        return LoadedTraces(tuple(traces), damaged)

    def read_all(self) -> Iterator[dict[str, Any]]:
        yield from self.load().traces


@dataclass(frozen=True)
class LoadedTraces:
    traces: tuple[dict[str, Any], ...]
    damaged_lines: int
