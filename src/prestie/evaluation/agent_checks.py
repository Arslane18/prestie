"""Deterministic checks on an agent answer: free, reproducible, no judge needed.

Each check returns 1.0 (pass) or 0.0 (fail), or is left out of the result
when it does not apply to the case (reported as n/a, excluded from means).
"""

import math
import re
from collections.abc import Iterable, Sequence
from html import unescape

from prestie.agent.agent import ToolCall
from prestie.agent.character_tool import CHARACTER_STATE_TOOL_NAME
from prestie.agent.prompts import MAX_ANSWER_LINES, MAX_LINE_CHARS
from prestie.agent.quest_tool import QUEST_DETAILS_TOOL_NAME
from prestie.agent.tools import SEARCH_TOOL_NAME
from prestie.evaluation.agent_cases import AgentCase

SOURCES_HEADING = "sources (contenu copié d'icy veins)"
CITED_URL = re.compile(r"https?://www\.icy-veins\.com/wow/[^\s)\]>\"'`<]+")
RETRIEVED_URL = re.compile(r'source_url="([^"]*)"')
TRAILING_PUNCTUATION = ".,;:!?"
WOW_PATH_MARKER = "/wow/"
# A simple question must be answered within the line budget the system prompt
# states (sources excluded). Lines longer than LINE_WIDTH count as several, so
# one endless paragraph cannot dodge the cap.
MAX_SIMPLE_ANSWER_LINES = MAX_ANSWER_LINES
LINE_WIDTH = MAX_LINE_CHARS

# Checks that decide whether a case passes ("retrieved" is diagnostic only).
GATING_CHECKS = (
    "search_ok",
    "sources_cited",
    "citations_valid",
    "exact_copy",
    "concise",
    "state_read",
    "quest_lookup",
    "spec_filter",
)


def cited_urls(answer: str) -> tuple[str, ...]:
    urls = (match.rstrip(TRAILING_PUNCTUATION) for match in CITED_URL.findall(answer))
    return tuple(dict.fromkeys(urls))


def retrieved_urls(tool_outputs: Iterable[str]) -> tuple[str, ...]:
    urls = (
        unescape(url)
        for output in tool_outputs
        for url in RETRIEVED_URL.findall(output)
    )
    return tuple(dict.fromkeys(urls))


def answer_lines(answer: str) -> int:
    """Visual line count of the answer body, without the sources section."""
    body = answer[: _sources_start(answer)]
    return sum(
        math.ceil(len(line.strip()) / LINE_WIDTH)
        for line in body.splitlines()
        if line.strip()
    )


def _sources_start(answer: str) -> int:
    index = _normalize(answer).find(SOURCES_HEADING)
    return index if index >= 0 else len(answer)


def _normalize(text: str) -> str:
    # Same length as the input, so indexes stay valid.
    return text.replace("\u2019", "'").lower()


def source_key(url: str) -> str:
    """'https://.../wow/<slug>#<anchor>' -> '<slug>#<anchor>'."""
    return url.partition(WOW_PATH_MARKER)[2]


def count_calls(tool_calls: Sequence[ToolCall], name: str) -> int:
    return sum(1 for call in tool_calls if call.name == name)


def programmatic_grades(
    case: AgentCase,
    answer: str,
    tool_calls: Sequence[ToolCall],
    tool_outputs: Iterable[str],
) -> dict[str, float]:
    retrieved = retrieved_urls(tool_outputs)
    cited = cited_urls(answer)
    checks: dict[str, bool | None] = {
        "search_ok": _search_ok(case, count_calls(tool_calls, SEARCH_TOOL_NAME)),
        "sources_cited": _sources_cited(case, answer, cited),
        "citations_valid": all(url in retrieved for url in cited) if cited else None,
        "concise": (
            None
            if case.detail_requested
            else answer_lines(answer) <= MAX_SIMPLE_ANSWER_LINES
        ),
        "exact_copy": (
            all(text in answer for text in case.must_include)
            if case.must_include
            else None
        ),
        "retrieved": (
            any(_matches(url, case.expected_sources) for url in retrieved)
            if case.expected_sources
            else None
        ),
        "state_read": _state_read(case, tool_calls),
        "quest_lookup": _quest_lookup(case, tool_calls),
        "spec_filter": _spec_filter(case, tool_calls),
    }
    return {name: float(ok) for name, ok in checks.items() if ok is not None}


def _search_ok(case: AgentCase, search_count: int) -> bool | None:
    if case.should_search is None:
        return None
    return (search_count > 0) == case.should_search


def _state_read(case: AgentCase, tool_calls: Sequence[ToolCall]) -> bool | None:
    if case.should_read_state is None:
        return None
    read = count_calls(tool_calls, CHARACTER_STATE_TOOL_NAME) > 0
    return read == case.should_read_state


def _quest_lookup(case: AgentCase, tool_calls: Sequence[ToolCall]) -> bool | None:
    if not case.expected_quest_ids:
        return None
    looked_up = {
        call.input.get("quest_id")
        for call in tool_calls
        if call.name == QUEST_DETAILS_TOOL_NAME
    }
    return set(case.expected_quest_ids) <= looked_up


def _spec_filter(case: AgentCase, tool_calls: Sequence[ToolCall]) -> bool | None:
    """Every search targets the expected spec (unfiltered searches mix specs)."""
    searches = [call for call in tool_calls if call.name == SEARCH_TOOL_NAME]
    if case.expected_spec is None or not searches:
        return None
    return all(call.input.get("spec") == case.expected_spec for call in searches)


def _sources_cited(case: AgentCase, answer: str, cited: tuple[str, ...]) -> bool | None:
    if not (case.answerable and case.should_search):
        return None
    normalized = answer.replace("’", "'").lower()
    return SOURCES_HEADING in normalized and bool(cited)


def _matches(url: str, expected: tuple[str, ...]) -> bool:
    key = source_key(url)
    return key in expected or key.partition("#")[0] in expected
