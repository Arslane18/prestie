"""Typed field readers for data parsed from SavedVariables (a Lua table).

Every reader raises CharacterStateError with the offending key, so a malformed
export fails with a message that says what to fix.
"""

from collections.abc import Mapping
from typing import Any


class CharacterStateError(ValueError):
    """The exported data is missing or does not match the expected schema."""


def require(data: Mapping[str, Any], key: str, kind: type) -> Any:
    value = data.get(key)
    # bool is a subclass of int: reject it where a number is expected.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise CharacterStateError(f"'{key}' must be a {kind.__name__}, got {value!r}")
    return value


def optional(data: Mapping[str, Any], key: str, kind: type) -> Any:
    return None if data.get(key) is None else require(data, key, kind)


def mapping(
    data: Mapping[str, Any], key: str, *, required: bool = False
) -> Mapping[str, Any] | None:
    value = data.get(key)
    if value is None or value == []:  # an empty Lua table parses as []
        if required:
            raise CharacterStateError(f"'{key}' is missing")
        return None
    if not isinstance(value, Mapping):
        raise CharacterStateError(f"'{key}' must be a table of fields, got {value!r}")
    return value


def items(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if value is None or value == {}:
        return []
    if not isinstance(value, list):
        raise CharacterStateError(f"'{key}' must be a list, got {value!r}")
    return value
