"""Online checks: the evaluation's deterministic checks, on real traffic.

Real questions come without expectations (nobody wrote whether the agent
should have read the state), so only the checks that need none apply. They
are computed when a report is built, never stored in the trace: a new or
fixed check applies to the whole history at once.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from prestie.agent.tools import SEARCH_TOOL_NAME
from prestie.evaluation.agent_checks import SOURCES_HEADING, cited_urls

ANSWERED = "answered"

INVALID_CITATION = "invalid_citation"  # a cited URL no search returned
MISSING_SOURCES = "missing_sources"  # searched, answered, no sources section
SOURCES_ADDED = "sources_added"  # the model forgot them, the agent appended them
UNFILTERED_SEARCH = "unfiltered_search"  # spec known, search without it
TOOL_ERROR = "tool_error"
NOT_ANSWERED = "not_answered"  # error, refusal, truncation or interruption


def online_flags(trace: Mapping[str, Any]) -> tuple[str, ...]:
    """The suspicious signals of one traced turn, in a stable order."""
    tools = trace.get("tools", ())
    searches = [t for t in tools if t.get("name") == SEARCH_TOOL_NAME]
    answer = trace.get("answer", "")
    answered = trace.get("outcome") == ANSWERED
    spec_known = bool((trace.get("character") or {}).get("spec_id"))
    checks = (
        (INVALID_CITATION, not _citations_retrieved(answer, searches)),
        (MISSING_SOURCES, answered and bool(searches) and not _has_sources(answer)),
        (SOURCES_ADDED, bool(trace.get("sources_added"))),
        (
            UNFILTERED_SEARCH,
            spec_known and any(not s["input"].get("spec") for s in searches),
        ),
        (TOOL_ERROR, any(t.get("is_error") for t in tools)),
        (NOT_ANSWERED, not answered),
    )
    return tuple(name for name, raised in checks if raised)


def _citations_retrieved(answer: str, searches: Sequence[Mapping[str, Any]]) -> bool:
    """Each cited URL was retrieved, as is or as the page of a retrieved passage
    (same rule as the evaluation's citations_valid)."""
    retrieved = {
        result.get("source_url", "")
        for search in searches
        for result in search.get("details", {}).get("results", ())
    }
    pages = {url.partition("#")[0] for url in retrieved}
    return all(url in retrieved or url in pages for url in cited_urls(answer))


def _has_sources(answer: str) -> bool:
    return SOURCES_HEADING in answer.replace("’", "'").lower()
