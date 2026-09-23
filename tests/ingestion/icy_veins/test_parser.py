from datetime import UTC, datetime

import pytest

from prestie.ingestion.icy_veins.cache import CachedPage
from prestie.ingestion.icy_veins.pages import GuidePage
from prestie.ingestion.icy_veins.parser import ParseError, parse_guide
from tests.ingestion.icy_veins.html_fixtures import heading, page, spell

GUIDE = GuidePage("blood-death-knight-pve-tank-guide", "overview")
FETCHED_AT = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)


def parse(content: str, **page_kwargs):
    cached = CachedPage(
        GUIDE.slug, GUIDE.url, page(content, **page_kwargs), FETCHED_AT, 200
    )
    return parse_guide(GUIDE, cached)


def test_page_metadata_is_extracted():
    guide = parse("<p>x</p>")

    assert guide.title == "Blood Death Knight Tank Guide — 12.1"
    assert guide.patch == "12.1"
    assert guide.authors == ("Mandl", "Panthea")
    assert guide.source_updated_at == datetime(2026, 8, 10, 19, 20)  # noqa: DTZ001
    assert guide.fetched_at == FETCHED_AT
    assert guide.url == GUIDE.url
    assert guide.page == GUIDE


def test_sections_follow_heading_hierarchy():
    guide = parse(
        heading(2, "1.", "Rotation", "rotation")
        + "<p>Rotation intro</p>"
        + heading(3, "1.1.", "Single Target", "single-target")
        + "<p>ST text</p>"
        + heading(3, "1.2.", "AoE", "aoe")
        + "<p>AoE text</p>"
        + heading(2, "2.", "Stats", "stats")
        + "<p>Stat text</p>"
    )

    assert [(s.heading_path, s.anchor, s.text) for s in guide.sections] == [
        (("Rotation",), "rotation", "Rotation intro"),
        (("Rotation", "Single Target"), "single-target", "ST text"),
        (("Rotation", "AoE"), "aoe", "AoE text"),
        (("Stats",), "stats", "Stat text"),
    ]


def test_heading_without_content_is_dropped_but_kept_in_child_paths():
    guide = parse(
        heading(2, "1.", "Mechanics", "mechanics")
        + heading(3, "1.1.", "Runes", "runes")
        + "<p>Runes text</p>"
    )

    assert [s.heading_path for s in guide.sections] == [("Mechanics", "Runes")]


def test_h4_nests_under_h3():
    guide = parse(
        heading(2, "1.", "A")
        + heading(3, "1.1.", "B")
        + heading(4, "1.1.1.", "C")
        + "<p>deep</p>"
        + heading(3, "1.2.", "D")
        + "<p>sibling</p>"
    )

    assert [s.heading_path for s in guide.sections] == [("A", "B", "C"), ("A", "D")]


def test_content_before_first_heading_becomes_introduction():
    guide = parse(
        '<div class="guide-intro"><p>Welcome!</p></div>' + heading(2, "1.", "A")
    )

    [intro] = guide.sections
    assert intro.heading_path == ("Introduction",)
    assert intro.anchor is None
    assert intro.text == "Welcome!"


def test_section_text_joins_blocks_and_drops_exact_duplicates():
    guide = parse(
        heading(2, "1.", "Rules")
        + "<p>Exterminate rules</p><ul><li>a</li></ul><p>Exterminate rules</p>"
    )

    assert guide.sections[0].text == "Exterminate rules\n\n- a"


def test_section_collects_spell_ids_of_its_blocks_only():
    guide = parse(
        heading(2, "1.", "A")
        + f"<p>{spell(49998, 'Death Strike')} {spell(50842, 'Blood Boil')}</p>"
        + f"<p>{spell(49998, 'Death Strike')}</p>"
        + heading(2, "2.", "B")
        + f"<p>{spell(195181, 'Bone Shield')}</p>"
    )

    assert [s.spell_ids for s in guide.sections] == [(49998, 50842), (195181,)]


def test_boilerplate_never_reaches_sections():
    guide = parse(heading(2, "1.", "A") + "<p>Real content</p>")

    all_text = " ".join(s.text for s in guide.sections)
    assert "Changelog" not in all_text
    assert "Adjusted things" not in all_text
    assert "Patreon" not in all_text
    assert "Table of contents" not in all_text


def test_missing_content_container_raises_parse_error():
    cached = CachedPage(
        GUIDE.slug, GUIDE.url, "<html><body>maintenance</body></html>", FETCHED_AT, 200
    )

    with pytest.raises(ParseError, match=GUIDE.slug):
        parse_guide(GUIDE, cached)


def test_missing_optional_header_metadata_yields_none():
    cached = CachedPage(
        GUIDE.slug,
        GUIDE.url,
        '<html><body><h1>Guide</h1><div class="guide-page-content"><p>x</p></div></body></html>',
        FETCHED_AT,
        200,
    )

    guide = parse_guide(GUIDE, cached)

    assert (guide.patch, guide.source_updated_at, guide.authors) == (None, None, ())


def test_unreadable_update_date_yields_none():
    html = page("<p>x</p>").replace("Aug 10, 2026 - 7:20 PM", "last Tuesday")
    cached = CachedPage(GUIDE.slug, GUIDE.url, html, FETCHED_AT, 200)

    assert parse_guide(GUIDE, cached).source_updated_at is None
