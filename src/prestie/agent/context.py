"""Context management: what earlier turns keep in the conversation sent to Claude.

Every request resends the whole conversation, so each tool result stays in the
context, and in the bill, for the rest of the conversation. Most of it stops
being useful once the turn is answered: the answer already holds what mattered.
Before each new question, earlier turns are compacted:

- search results (the largest part, ~5 passages each) keep only the query and
  the sections and URLs they came from; Claude searches again if it needs a
  passage's content;
- the character state is dropped: it may have changed since (a /reload between
  two questions), so Claude must read it again instead of trusting an old copy;
- everything else (questions, answers, tool calls, quest facts, errors) stays.

The current turn is never compacted: Claude needs its fresh results. Compaction
is a pure function of the stored history, so an already compacted turn yields
the same bytes on every request and the cached prefix stays valid; only the
turn that just ended is rewritten, once.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from prestie.agent.character_tool import CHARACTER_STATE_TOOL_NAME
from prestie.agent.tools import SEARCH_TOOL_NAME

Message = Mapping[str, Any]

SEARCH_OPENING = re.compile(r"<search_results [^>]*>")
SEARCH_SOURCE = re.compile(
    r'<result index="[^"]*" (section="[^"]*" source_url="[^"]*")>'
)
COMPACTED_SEARCH_NOTE = (
    "Passages removed from the context after the turn that used them. "
    "Sections returned:"
)
SEARCH_AGAIN_NOTE = (
    "Search again to read them before relying on details your earlier answers "
    "do not state."
)
COMPACTED_STATE = (
    "<character_state_removed>The character state read in an earlier turn was "
    "removed: the player may have changed it since (level, spec, hero talent, "
    "quest, then /reload). Call get_character_state again for the current "
    "state.</character_state_removed>"
)


def compact_history(history: Sequence[Message]) -> tuple[Message, ...]:
    """The finished turns as they are sent to Claude, without stale tool results."""
    names = _tool_names_by_id(history)
    return tuple(_compact_message(message, names) for message in history)


def _compact_message(message: Message, names: Mapping[str, str]) -> Message:
    content = message["content"]
    if message["role"] != "user" or isinstance(content, str):
        return message
    return {
        **message,
        "content": [_compact_block(block, names) for block in content],
    }


def _compact_block(block: Mapping[str, Any], names: Mapping[str, str]) -> Any:
    if block.get("type") != "tool_result" or block.get("is_error"):
        return block
    compact = COMPACTORS.get(names.get(block["tool_use_id"], ""))
    content = block["content"]
    if compact is None or not isinstance(content, str):
        return block
    return {**block, "content": compact(content)}


def _compact_search(content: str) -> str:
    opening = SEARCH_OPENING.match(content)
    if opening is None:  # "No results ..." is already short
        return content
    if COMPACTED_SEARCH_NOTE in content:  # already compacted
        return content
    sources = [f"<source {attrs}/>" for attrs in SEARCH_SOURCE.findall(content)]
    return "\n".join(
        [
            opening.group(0),
            COMPACTED_SEARCH_NOTE,
            *sources,
            SEARCH_AGAIN_NOTE,
            "</search_results>",
        ]
    )


def _compact_state(content: str) -> str:
    return COMPACTED_STATE


COMPACTORS: Mapping[str, Callable[[str], str]] = {
    SEARCH_TOOL_NAME: _compact_search,
    CHARACTER_STATE_TOOL_NAME: _compact_state,
}


def _tool_names_by_id(history: Sequence[Message]) -> dict[str, str]:
    return {
        block.id: block.name
        for message in history
        if message["role"] == "assistant"
        for block in message["content"]
        if getattr(block, "type", None) == "tool_use"
    }
