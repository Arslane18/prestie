"""Watches the addon's SavedVariables file and keeps the latest parsed state.

The game only writes the file on /reload, logout or exit, so the state is as
fresh as the last of those. Change detection is by polling (mtime + size):
inotify does not see writes made by Windows on /mnt/c when running in WSL,
and one `stat` per check is negligible.
"""

import time
from collections.abc import Callable, Iterator
from pathlib import Path

from prestie.character.lua_table import LuaParseError, parse_saved_variables
from prestie.character.state import (
    CharacterState,
    CharacterStateError,
    character_state_from_saved_variables,
)

DEFAULT_POLL_INTERVAL_S = 1.0

_FileVersion = tuple[int, int]  # (mtime_ns, size)


class SavedVariablesWatcher:
    def __init__(self, path: Path):
        self._path = path
        self._version: _FileVersion | None = None
        self._state: CharacterState | None = None
        self._failed_version: _FileVersion | None = None

    @property
    def path(self) -> Path:
        return self._path

    def latest(self) -> CharacterState:
        """The current state, re-parsed only if the file changed since last read."""
        version = self._current_version()
        if version is None:
            raise CharacterStateError(
                f"{self._path} not found: install the Prestie addon, log in, "
                "then /reload once"
            )
        if version != self._version or self._state is None:
            try:
                self._state = self._read()
            except CharacterStateError:
                self._failed_version = version
                raise
            # Only remember the version once parsed: a half-written file is retried.
            self._version = version
        return self._state

    def poll(self) -> CharacterState | None:
        """The new state if the file changed since the last read, else None.

        A version that failed to parse is not retried until the file changes.
        """
        version = self._current_version()
        if version in (None, self._version, self._failed_version):
            return None
        return self.latest()

    def watch(
        self,
        interval: float = DEFAULT_POLL_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
        on_error: Callable[[CharacterStateError], None] | None = None,
    ) -> Iterator[CharacterState]:
        """Yield the state at start and after every change; never returns.

        Without `on_error`, an unreadable file ends the iteration with the error.
        """
        while True:
            try:
                state = self.poll()
            except CharacterStateError as exc:
                if on_error is None:
                    raise
                on_error(exc)
                state = None
            if state is not None:
                yield state
            sleep(interval)

    def _current_version(self) -> _FileVersion | None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _read(self) -> CharacterState:
        try:
            text = self._path.read_text(encoding="utf-8")
            variables = parse_saved_variables(text)
        except (OSError, UnicodeDecodeError) as exc:
            raise CharacterStateError(f"cannot read {self._path}: {exc}") from exc
        except LuaParseError as exc:
            raise CharacterStateError(f"cannot parse {self._path}: {exc}") from exc
        return character_state_from_saved_variables(variables)
