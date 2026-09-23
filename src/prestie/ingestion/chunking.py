"""Turn parsed guide sections into chunks: the units we embed and retrieve.

Each chunk is one section (or part of an oversized one) prefixed with a small
context header (guide title + heading path). A bare "Haste > Mastery" means
nothing on its own; with its header the embedding knows it is about Blood DK
stat priority, which is what makes it retrievable for "quelles stats en DK sang ?".

Sections are only split when they exceed MAX_CHUNK_CHARS, and always on
natural boundaries (paragraphs, then lines, then sentences). Oversized tables
repeat their header row in every part so each part stays self-explanatory.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from prestie.ingestion.icy_veins.parser import GuideSection, ParsedGuide

SOURCE = "icy-veins"
# ~500 tokens: large enough for a whole rotation priority list, small enough
# that a retrieved chunk stays focused on one topic.
MAX_CHUNK_CHARS = 2000
PARAGRAPH_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
TABLE_HEADER_LINES = 2


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: Mapping[str, Any]


def chunk_guide(guide: ParsedGuide) -> tuple[Chunk, ...]:
    return tuple(
        chunk
        for index, section in enumerate(guide.sections)
        for chunk in _chunk_section(guide, index, section)
    )


def _chunk_section(
    guide: ParsedGuide, index: int, section: GuideSection
) -> Iterable[Chunk]:
    header = f"{guide.title}\nSection: {' > '.join(section.heading_path)}\n\n"
    bodies = split_text(section.text, MAX_CHUNK_CHARS - len(header))
    for part, body in enumerate(bodies):
        yield Chunk(
            id=f"{SOURCE}:{guide.page.slug}:{index:03d}:{part}",
            text=header + body,
            metadata=_metadata(guide, section, part, len(bodies)),
        )


def _metadata(
    guide: ParsedGuide, section: GuideSection, part: int, part_count: int
) -> dict[str, Any]:
    source_url = f"{guide.url}#{section.anchor}" if section.anchor else guide.url
    byline = f" (by {', '.join(guide.authors)})" if guide.authors else ""
    optional = {
        "patch": guide.patch,
        "authors": list(guide.authors),
        "source_updated_at": (
            guide.source_updated_at.isoformat() if guide.source_updated_at else None
        ),
        "spell_ids": list(section.spell_ids),
        "item_ids": list(section.item_ids),
    }
    return {
        "source": SOURCE,
        "source_url": source_url,
        "page_slug": guide.page.slug,
        "content_type": guide.page.content_type,
        "wow_class": guide.page.wow_class,
        "spec": guide.page.spec,
        "title": guide.title,
        "section": " > ".join(section.heading_path),
        # Required by Icy Veins' reuse terms: explicit credit + visible link.
        "attribution": f"Copied from Icy Veins{byline}: {source_url}",
        "fetched_at": guide.fetched_at.isoformat(),
        "part": part,
        "part_count": part_count,
        # Chroma cannot store empty lists; absent is clearer than empty anyway.
        **{key: value for key, value in optional.items() if value},
    }


# --- splitting ---------------------------------------------------------------


def split_text(text: str, budget: int) -> list[str]:
    units = [
        unit
        for paragraph in text.split(PARAGRAPH_SEPARATOR)
        for unit in _fit_paragraph(paragraph, budget)
    ]
    return _pack(units, budget, PARAGRAPH_SEPARATOR)


def _fit_paragraph(paragraph: str, budget: int) -> list[str]:
    if len(paragraph) <= budget:
        return [paragraph]
    lines = paragraph.split(LINE_SEPARATOR)
    if _is_table(lines):
        return _split_table(lines, budget)
    pieces = [piece for line in lines for piece in _fit_line(line, budget)]
    return _pack(pieces, budget, LINE_SEPARATOR)


def _split_table(lines: list[str], budget: int) -> list[str]:
    header = LINE_SEPARATOR.join(lines[:TABLE_HEADER_LINES])
    row_budget = budget - len(header) - len(LINE_SEPARATOR)
    rows = [
        piece
        for row in lines[TABLE_HEADER_LINES:]
        for piece in _fit_line(row, row_budget)
    ]
    return [
        header + LINE_SEPARATOR + body
        for body in _pack(rows, row_budget, LINE_SEPARATOR)
    ]


def _fit_line(line: str, budget: int) -> list[str]:
    if len(line) <= budget:
        return [line]
    sentences = [
        sentence[start : start + budget]  # last resort: hard cut a huge sentence
        for sentence in SENTENCE_END.split(line)
        for start in range(0, len(sentence), budget)
    ]
    return _pack(sentences, budget, " ")


def _is_table(lines: list[str]) -> bool:
    return (
        len(lines) > TABLE_HEADER_LINES
        and lines[0].startswith("|")
        and lines[1].startswith("| ---")
    )


def _pack(units: Iterable[str], budget: int, separator: str) -> list[str]:
    """Greedily merge consecutive units while they fit in the budget."""
    packed: list[str] = []
    for unit in units:
        candidate = f"{packed[-1]}{separator}{unit}" if packed else unit
        if packed and len(candidate) <= budget:
            packed[-1] = candidate
        else:
            packed.append(unit)
    return packed
