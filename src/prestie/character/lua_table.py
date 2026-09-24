"""Parser for WoW SavedVariables files (Lua data literals -> Python values).

The game writes each saved variable as a Lua assignment:

    PrestieDB = {
        ["snapshot"] = {
            ["level"] = 83,
            ["quests"] = { { ["id"] = 1234 }, },
        },
    }

We parse this data subset ourselves instead of running the file in a Lua
interpreter: the file is external input, and a parser that only knows
literals cannot execute anything. Mapping: string/number/boolean map to the
Python equivalent, `nil` drops the entry (as in Lua), and a table becomes a
list when its keys are exactly 1..n, a dict otherwise.
"""

import re
from typing import Any

_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DECIMAL_ESCAPE = re.compile(r"\d{1,3}")
_KEYWORDS = {"true": True, "false": False, "nil": None}
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "\n": "\n",
}


class LuaParseError(ValueError):
    """The text is not a valid SavedVariables data file."""


def parse_saved_variables(text: str) -> dict[str, Any]:
    """Parse `Name = value` assignments into {name: value}."""
    return _Parser(text).parse_file()


class _Parser:
    def __init__(self, text: str):
        self._text = text
        self._pos = 0

    def parse_file(self) -> dict[str, Any]:
        variables: dict[str, Any] = {}
        self._skip_blank()
        while self._pos < len(self._text):
            name = self._expect_name()
            self._expect("=")
            value = self._value()
            if value is not None:
                variables = {**variables, name: value}
            self._skip_blank()
        return variables

    def _value(self) -> Any:
        self._skip_blank()
        char = self._peek()
        if char == "{":
            return self._table()
        if char in ('"', "'"):
            return self._string()
        number = _NUMBER.match(self._text, self._pos)
        if number:
            self._pos = number.end()
            literal = number.group()
            is_int = not any(c in literal for c in ".eE")
            return int(literal) if is_int else float(literal)
        name = _NAME.match(self._text, self._pos)
        if name and name.group() in _KEYWORDS:
            self._pos = name.end()
            return _KEYWORDS[name.group()]
        raise self._error("expected a value")

    def _table(self) -> list[Any] | dict[Any, Any]:
        self._expect("{")
        entries: dict[Any, Any] = {}
        next_index = 1
        while True:
            self._skip_blank()
            if self._peek() == "}":
                self._pos += 1
                return _to_python(entries)
            key, value = self._field()
            if key is None:
                key, next_index = next_index, next_index + 1
            if value is not None:
                entries = {**entries, key: value}
            self._skip_blank()
            if self._peek() in (",", ";"):
                self._pos += 1
            elif self._peek() != "}":
                raise self._error("expected ',' or '}'")

    def _field(self) -> tuple[Any, Any]:
        """Return (key, value); key is None for a positional entry."""
        if self._peek() == "[":
            self._pos += 1
            key = self._value()
            self._expect("]")
            self._expect("=")
            return key, self._value()
        name = _NAME.match(self._text, self._pos)
        if name and name.group() not in _KEYWORDS:
            self._pos = name.end()
            self._expect("=")
            return name.group(), self._value()
        return None, self._value()

    def _string(self) -> str:
        quote = self._text[self._pos]
        self._pos += 1
        parts: list[str] = []
        while True:
            char = self._peek()
            if char in ("", "\n"):
                raise self._error("unterminated string")
            self._pos += 1
            if char == quote:
                return "".join(parts)
            parts.append(self._escape() if char == "\\" else char)

    def _escape(self) -> str:
        digits = _DECIMAL_ESCAPE.match(self._text, self._pos)
        if digits:
            self._pos = digits.end()
            return chr(int(digits.group()))
        if self._text.startswith("\r\n", self._pos):
            self._pos += 2
            return "\n"
        char = self._peek()
        if char not in _ESCAPES:
            raise self._error(f"invalid escape '\\{char}'")
        self._pos += 1
        return _ESCAPES[char]

    def _expect_name(self) -> str:
        name = _NAME.match(self._text, self._pos)
        if not name:
            raise self._error("expected a variable name")
        self._pos = name.end()
        return name.group()

    def _expect(self, token: str) -> None:
        self._skip_blank()
        if not self._text.startswith(token, self._pos):
            raise self._error(f"expected '{token}'")
        self._pos += len(token)

    def _skip_blank(self) -> None:
        """Skip whitespace (CRLF included) and `--` line comments."""
        while self._pos < len(self._text):
            if self._text[self._pos].isspace():
                self._pos += 1
            elif self._text.startswith("--", self._pos):
                end = self._text.find("\n", self._pos)
                self._pos = len(self._text) if end == -1 else end
            else:
                return

    def _peek(self) -> str:
        return self._text[self._pos] if self._pos < len(self._text) else ""

    def _error(self, message: str) -> LuaParseError:
        line = self._text.count("\n", 0, self._pos) + 1
        return LuaParseError(f"{message} at line {line}")


def _to_python(entries: dict[Any, Any]) -> list[Any] | dict[Any, Any]:
    if set(entries) == set(range(1, len(entries) + 1)):
        return [entries[index] for index in range(1, len(entries) + 1)]
    return entries
