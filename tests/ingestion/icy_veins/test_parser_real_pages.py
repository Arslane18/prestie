"""Smoke-test the parser against the real cached pages (run `prestie scrape` first).

Real HTML is not committed (copyrighted content), so these tests skip when the
local cache is absent.
"""

from pathlib import Path

import pytest

from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES
from prestie.ingestion.icy_veins.parser import parse_guide

CACHE_DIR = Path(__file__).parents[3] / "data" / "raw" / "icy-veins"
MIN_SECTIONS_PER_PAGE = 5
BOILERPLATE_MARKERS = ("In The Same Category", "Support Our Writers", "Changelog")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def guides():
    cache = HtmlCache(CACHE_DIR)
    cached = [(page, cache.get(page.slug)) for page in BLOOD_DK_PAGES]
    if any(entry is None for _, entry in cached):
        pytest.skip("Icy Veins pages not cached locally; run `prestie scrape`")
    return [parse_guide(page, entry) for page, entry in cached]


def test_every_page_yields_sections_and_metadata(guides):
    for guide in guides:
        assert len(guide.sections) >= MIN_SECTIONS_PER_PAGE, guide.page.slug
        assert guide.patch is not None, guide.page.slug
        assert guide.source_updated_at is not None, guide.page.slug
        assert guide.authors, guide.page.slug


def test_no_section_is_empty_or_boilerplate(guides):
    for guide in guides:
        for section in guide.sections:
            assert section.text.strip(), section.heading_path
            assert not any(marker in section.text for marker in BOILERPLATE_MARKERS)


def test_rotation_variants_are_labelled_by_hero_talent(guides):
    rotation = next(g for g in guides if g.page.content_type == "rotation")
    text = "\n".join(section.text for section in rotation.sections)

    assert "[Deathbringer only]" in text
    assert "[San'layn only]" in text


def test_leveling_content_carries_level_ranges(guides):
    leveling = next(g for g in guides if g.page.content_type == "leveling")

    assert any("[Levels " in section.text for section in leveling.sections)
