"""Parse a cached Icy Veins guide into logical sections with provenance metadata.

A section is the text between two headings (h2-h4). Sections are the unit the
chunker will embed: they follow the author's own structure (rotation, stat
priority, a given cooldown...), which keeps each chunk about one topic.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from bs4 import BeautifulSoup

from prestie.ingestion.icy_veins.blocks import (
    Block,
    Heading,
    normalize_space,
    walk_content,
)
from prestie.ingestion.icy_veins.cache import CachedPage
from prestie.ingestion.icy_veins.conditions import ConditionLabels
from prestie.ingestion.icy_veins.pages import GuidePage

CONTENT_SELECTOR = "div.guide-page-content"
INTRODUCTION = "Introduction"
PATCH_IN_TITLE = re.compile(r" - (\d+\.\d+(?:\.\d+)?) - ")
UPDATED_AT_FORMAT = "%b %d, %Y - %I:%M %p"  # e.g. "Aug 10, 2026 - 7:20 PM"


class ParseError(Exception):
    """The HTML does not look like an Icy Veins guide page."""


@dataclass(frozen=True)
class GuideSection:
    heading_path: tuple[str, ...]
    anchor: str | None
    text: str
    spell_ids: tuple[int, ...]
    item_ids: tuple[int, ...]


@dataclass(frozen=True)
class ParsedGuide:
    page: GuidePage
    url: str
    title: str
    patch: str | None
    authors: tuple[str, ...]
    source_updated_at: datetime | None  # naive: Icy Veins does not state a timezone
    fetched_at: datetime
    sections: tuple[GuideSection, ...]


def parse_guide(page: GuidePage, cached: CachedPage) -> ParsedGuide:
    soup = BeautifulSoup(cached.html, "lxml")
    content = soup.select_one(CONTENT_SELECTOR)
    if content is None:
        raise ParseError(f"{page.slug}: no '{CONTENT_SELECTOR}' element in cached HTML")

    labels = ConditionLabels.from_content(content)
    return ParsedGuide(
        page=page,
        url=cached.url,
        title=_title(soup),
        patch=_patch(soup),
        authors=_authors(soup),
        source_updated_at=_updated_at(soup),
        fetched_at=cached.fetched_at,
        sections=_build_sections(walk_content(content, labels)),
    )


def _build_sections(events: Iterable[Heading | Block]) -> tuple[GuideSection, ...]:
    sections: list[GuideSection] = []
    path: tuple[tuple[int, str], ...] = ()
    anchor: str | None = None
    blocks: list[Block] = []

    for event in events:
        if isinstance(event, Block):
            blocks.append(event)
            continue
        sections.extend(_make_section(path, anchor, blocks))
        path = tuple(entry for entry in path if entry[0] < event.level)
        path = (*path, (event.level, event.text))
        anchor = event.anchor
        blocks = []

    sections.extend(_make_section(path, anchor, blocks))
    return tuple(sections)


def _make_section(
    path: tuple[tuple[int, str], ...], anchor: str | None, blocks: list[Block]
) -> tuple[GuideSection, ...]:
    if not blocks:
        return ()
    heading_path = tuple(text for _, text in path) or (INTRODUCTION,)
    # Identical blocks show up when the page repeats content per rotation preset.
    unique_texts = dict.fromkeys(block.text for block in blocks)
    return (
        GuideSection(
            heading_path=heading_path,
            anchor=anchor,
            text="\n\n".join(unique_texts),
            spell_ids=_unique(i for block in blocks for i in block.spell_ids),
            item_ids=_unique(i for block in blocks for i in block.item_ids),
        ),
    )


def _unique(ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(ids))


def _title(soup: BeautifulSoup) -> str:
    tag = soup.find("h1") or soup.find("title")
    return normalize_space(tag.get_text(" ")) if tag else ""


def _patch(soup: BeautifulSoup) -> str | None:
    title = soup.find("title")
    match = PATCH_IN_TITLE.search(title.get_text()) if title else None
    return match[1] if match else None


def _authors(soup: BeautifulSoup) -> tuple[str, ...]:
    names = (
        normalize_space(a.get_text(" ")) for a in soup.select(".guide-header__author")
    )
    return tuple(dict.fromkeys(name for name in names if name))


def _updated_at(soup: BeautifulSoup) -> datetime | None:
    tag = soup.select_one(".guide-header__updated-date")
    if tag is None:
        return None
    raw = normalize_space(tag.get_text(" "))
    try:
        # Naive on purpose: the page gives no timezone and we will not guess one.
        return datetime.strptime(raw, UPDATED_AT_FORMAT)  # noqa: DTZ007
    except ValueError:
        return None
