"""Walk an Icy Veins guide body and emit a flat stream of headings and text blocks.

Headings sit at arbitrary depths (inside image blocks, rotation widgets, even
`<details>` nested in a `<p>`), so the walk is a depth-first traversal in
document order rather than a scan of the container's direct children. Each
leaf block (paragraph, list, table...) is rendered to Markdown-flavoured plain
text, which is what will be embedded and later shown to the LLM.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby

from bs4 import NavigableString, PageElement, Tag
from bs4.element import PreformattedString

from prestie.ingestion.icy_veins.conditions import ConditionLabels

SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "img", "input", "button", "iframe", "form"}
)
SKIPPED_CLASSES = frozenset(
    {
        # Site chrome and navigation
        "table-of-contents",
        "content-toc",
        "changelog_wrapper",
        "internal-links",
        "cta-block__wrapper-new",
        "app-banner",
        "inline-author-block",
        "back-to-top",
        "raider-io-links",
        "image_block_header_buttons",
        # Interactive widgets whose meaning is captured elsewhere (or not at all)
        "leveling_slider_container",
        "leveling-slider-intro",
        "midnight-skill-builder-embed",
        "rotation_switches",
        "talent-calculator-description",
        "talent-calculator-filters-inner",
    }
)
INLINE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "br",
        "code",
        "em",
        "i",
        "label",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "u",
    }
)
BLOCK_LEVEL_TAGS = ("div", "p", "ul", "ol", "table", "details", "section", "blockquote")
TEXT_BLOCK_TAGS = frozenset(
    {"p", "summary", "blockquote", "pre", "figcaption", "dt", "dd"}
)
HEADING_TAGS = ("h2", "h3", "h4", "h5")
WOWHEAD_ID = re.compile(r"^(spell|item)=(\d+)")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?)%])")
SPACE_AFTER_OPEN_PAREN = re.compile(r"\(\s+")
DEFAULT_MAX_SCORE = 5


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    anchor: str | None


@dataclass(frozen=True)
class Block:
    text: str
    spell_ids: tuple[int, ...] = ()
    item_ids: tuple[int, ...] = ()


def walk_content(content: Tag, labels: ConditionLabels) -> Iterator[Heading | Block]:
    yield from _walk_children(content, labels, ())


def normalize_space(text: str) -> str:
    text = " ".join(text.split())
    text = SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return SPACE_AFTER_OPEN_PAREN.sub("(", text)


# --- traversal ---------------------------------------------------------------


def _walk_children(
    node: Tag, labels: ConditionLabels, conditions: tuple[str, ...]
) -> Iterator[Heading | Block]:
    """Group consecutive inline nodes into one block; recurse into the rest."""
    for is_inline, group in groupby(node.children, key=_is_inline):
        nodes = list(group)
        if is_inline:
            yield from _block(_inline_text(nodes), nodes, conditions)
            continue
        for child in nodes:
            if isinstance(child, Tag):
                yield from _walk_element(child, labels, conditions)


def _walk_element(
    element: Tag, labels: ConditionLabels, conditions: tuple[str, ...]
) -> Iterator[Heading | Block]:
    if _is_skipped(element):
        return
    own_condition = labels.describe(element)
    conditions = conditions + ((own_condition,) if own_condition else ())
    classes = _classes(element)

    if "heading_container" in classes:
        yield from _heading(element, labels)
    elif "performance-overview" in classes:
        yield from _block(_render_performance(element), [element], conditions)
    elif "faq-block__dropdown" in classes:
        yield from _block(_render_faq(element, labels), [element], conditions)
    elif "export-string" in classes:
        yield from _block(_render_export_string(element), [element], conditions)
    elif element.name in ("ul", "ol"):
        yield from _block(
            "\n".join(_render_list(element, labels)), [element], conditions
        )
    elif element.name == "table":
        yield from _block(_render_table(element), [element], conditions)
    elif element.name in TEXT_BLOCK_TAGS and not _contains_blocks(element):
        yield from _block(_inline_text(element.children), [element], conditions)
    else:
        yield from _walk_children(element, labels, conditions)


def _heading(container: Tag, labels: ConditionLabels) -> Iterator[Heading]:
    tag = container.find(HEADING_TAGS)
    if tag is None:
        return
    text = normalize_space(tag.get_text(" "))
    condition = labels.describe(tag) or labels.describe(container)
    if text:
        yield Heading(
            level=int(tag.name[1]),
            text=f"{text} {condition}" if condition else text,
            anchor=tag.get("id"),
        )


def _block(
    text: str, sources: Iterable[PageElement], conditions: tuple[str, ...]
) -> Iterator[Block]:
    text = text.strip()
    if not text:
        return
    spell_ids, item_ids = _wowhead_ids(sources)
    prefix = " ".join(conditions)
    yield Block(
        text=f"{prefix} {text}" if prefix else text,
        spell_ids=spell_ids,
        item_ids=item_ids,
    )


# --- predicates ----------------------------------------------------------------


def _classes(element: Tag) -> list[str]:
    return element.get("class") or []


def _is_skipped(element: Tag) -> bool:
    return element.name in SKIPPED_TAGS or not SKIPPED_CLASSES.isdisjoint(
        _classes(element)
    )


def _is_text(node: PageElement) -> bool:
    # PreformattedString covers comments, CDATA, doctype...
    return isinstance(node, NavigableString) and not isinstance(
        node, PreformattedString
    )


def _is_inline(node: PageElement) -> bool:
    if _is_text(node):
        return True
    return (
        isinstance(node, Tag)
        and node.name in INLINE_TAGS
        and "heading_container" not in _classes(node)
        and not _contains_blocks(node)
    )


def _contains_blocks(element: Tag) -> bool:
    return element.find(BLOCK_LEVEL_TAGS) is not None


# --- rendering -----------------------------------------------------------------


def _raw_inline(nodes: Iterable[PageElement]) -> str:
    parts = []
    for node in nodes:
        if _is_text(node):
            parts.append(str(node))
        elif isinstance(node, Tag) and not _is_skipped(node):
            parts.append(" " if node.name == "br" else _raw_inline(node.children))
    return "".join(parts)


def _inline_text(nodes: Iterable[PageElement]) -> str:
    return normalize_space(_raw_inline(nodes))


def _render_list(list_tag: Tag, labels: ConditionLabels, depth: int = 0) -> list[str]:
    ordered = list_tag.name == "ol"
    lines = []
    for index, item in enumerate(list_tag.find_all("li", recursive=False), start=1):
        if _is_skipped(item):
            continue
        nested = [
            c for c in item.children if isinstance(c, Tag) and c.name in ("ul", "ol")
        ]
        text = _inline_text(c for c in item.children if c not in nested)
        condition = labels.describe(item)
        marker = f"{index}." if ordered else "-"
        body = " ".join(part for part in (condition, text) if part)
        lines.append(f"{'  ' * depth}{marker} {body}")
        for sub_list in nested:
            lines.extend(_render_list(sub_list, labels, depth + 1))
    return lines


def _render_table(table: Tag) -> str:
    rows = [
        [
            _inline_text(cell.children).replace("|", "\\|")
            for cell in row.find_all(["th", "td"])
        ]
        for row in table.find_all("tr")
    ]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header, *body = padded
    lines = [header, ["---"] * width, *body]
    return "\n".join(f"| {' | '.join(cells)} |" for cells in lines)


def _render_faq(details: Tag, labels: ConditionLabels) -> str:
    question = details.select_one(".faq-block__question") or details.find("summary")
    answer = details.select_one(".faq-block__answer") or details.select_one(
        ".faq-block__details"
    )
    question_text = _inline_text(question.children) if question else ""
    answer_blocks = _walk_children(answer, labels, ()) if answer else iter(())
    answer_text = "\n".join(e.text for e in answer_blocks if isinstance(e, Block))
    return f"Q: {question_text}\nA: {answer_text}"


def _render_export_string(details: Tag) -> str:
    title = details.select_one(".export-string__title")
    code = details.select_one(".export-string__code")
    if code is None:
        return ""
    title_text = _inline_text(title.children) if title else "talent build"
    return f"Talent import string ({title_text}): {code.get_text(strip=True)}"


def _render_performance(overview: Tag) -> str:
    max_label = f"{float(overview.get('data-max', DEFAULT_MAX_SCORE)):g}"
    lines = []
    for tab in overview.select(".performance-overview__tab"):
        label = tab.select_one(".performance-overview__label")
        panel = overview.find(id=tab.get("data-target"))
        if label is None or panel is None:
            continue
        rating = panel.select_one(".performance-header")
        rating_text = f" ({_inline_text(rating.children)})" if rating else ""
        comment = " ".join(_inline_text(p.children) for p in panel.find_all("p"))
        lines.append(
            f"{_inline_text(label.children)}: {tab.get('data-score', '?')}/{max_label}"
            f"{rating_text} — {comment}"
        )
    return "\n".join(lines)


def _wowhead_ids(
    sources: Iterable[PageElement],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    found: dict[str, dict[int, None]] = {"spell": {}, "item": {}}
    for source in sources:
        if not isinstance(source, Tag):
            continue
        for tag in [source, *source.find_all(attrs={"data-wowhead": True})]:
            match = WOWHEAD_ID.match(str(tag.get("data-wowhead", "")))
            if match:
                found[match[1]][int(match[2])] = None
    return tuple(found["spell"]), tuple(found["item"])
