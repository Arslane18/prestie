"""Icy Veins attribution, guaranteed by code rather than only asked for.

Reusing Icy Veins content requires crediting it with a visible link to the
source article. The system prompt asks the model to end each sourced answer
with a sources section; when it forgets, the answer would ship uncredited.
After a turn that searched the knowledge base, the agent appends the missing
section itself, listing each guide page the searches returned.

The pages returned are not necessarily the ones the answer used (only the
model knows that), so the model's own section, which lists the passages it
used, is always preferred; this is the fallback.
"""

from collections.abc import Iterable, Mapping
from typing import Any

# Same heading as the prompt asks for; the evaluation's `sources_cited` and
# the online `missing_sources` check look for it (tested).
SOURCES_TITLE = "Sources (contenu copié d'Icy Veins) :"
_HEADING = SOURCES_TITLE.removesuffix(" :").lower()


def has_sources_section(answer: str) -> bool:
    return _HEADING in answer.replace("’", "'").lower()


def missing_sources_section(
    answer: str, search_details: Iterable[Mapping[str, Any]]
) -> str | None:
    """The sources section to append to `answer`, or None if none is needed.

    `search_details` are the `details` of this turn's successful searches.
    """
    if has_sources_section(answer):
        return None
    pages = _consulted_pages(search_details)
    if not pages:
        return None
    lines = [f"- {name} — {url}" for url, name in pages.items()]
    return "\n".join(["", "", SOURCES_TITLE, *lines])


def _consulted_pages(search_details: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Page URL (without the section anchor) -> guide title, in first-seen order."""
    pages: dict[str, str] = {}
    for details in search_details:
        for result in details.get("results", ()):
            url = str(result.get("source_url") or "").partition("#")[0]
            if url and url not in pages:
                pages[url] = str(result.get("title") or result.get("section") or url)
    return pages
