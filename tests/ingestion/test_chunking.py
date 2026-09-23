from datetime import UTC, datetime

from prestie.ingestion.chunking import MAX_CHUNK_CHARS, chunk_guide
from prestie.ingestion.icy_veins.pages import GuidePage
from prestie.ingestion.icy_veins.parser import GuideSection, ParsedGuide

PAGE = GuidePage("blood-death-knight-pve-tank-stat-priority", "stat_priority")


def section(
    text: str,
    path: tuple[str, ...] = ("Stat Priority",),
    anchor: str | None = "stat-priority",
    spell_ids: tuple[int, ...] = (),
) -> GuideSection:
    return GuideSection(path, anchor, text, spell_ids, ())


def guide(*sections: GuideSection) -> ParsedGuide:
    return ParsedGuide(
        page=PAGE,
        url=PAGE.url,
        title="Blood Death Knight Stat Priority — 12.1",
        patch="12.1",
        authors=("Mandl", "Panthea"),
        source_updated_at=datetime(2026, 8, 10, 19, 20),  # noqa: DTZ001
        fetched_at=datetime(2026, 9, 23, 10, 0, tzinfo=UTC),
        sections=sections,
    )


def test_short_section_becomes_one_chunk_with_context_header():
    [chunk] = chunk_guide(
        guide(section("Haste > Mastery", path=("Stats", "San'layn Stat Priority")))
    )

    assert chunk.text == (
        "Blood Death Knight Stat Priority — 12.1\n"
        "Section: Stats > San'layn Stat Priority\n\n"
        "Haste > Mastery"
    )


def test_chunk_ids_are_stable_and_unique():
    parsed = guide(section("a"), section("b", anchor=None))

    ids = [chunk.id for chunk in chunk_guide(parsed)]

    assert ids == [
        "icy-veins:blood-death-knight-pve-tank-stat-priority:000:0",
        "icy-veins:blood-death-knight-pve-tank-stat-priority:001:0",
    ]
    assert ids == [chunk.id for chunk in chunk_guide(parsed)]


def test_metadata_carries_provenance_and_attribution():
    [chunk] = chunk_guide(guide(section("x", spell_ids=(49998, 50842))))

    assert chunk.metadata == {
        "source": "icy-veins",
        "source_url": f"{PAGE.url}#stat-priority",
        "page_slug": PAGE.slug,
        "content_type": "stat_priority",
        "wow_class": "death-knight",
        "spec": "blood",
        "title": "Blood Death Knight Stat Priority — 12.1",
        "section": "Stat Priority",
        "patch": "12.1",
        "authors": ["Mandl", "Panthea"],
        "attribution": (
            f"Copied from Icy Veins (by Mandl, Panthea): {PAGE.url}#stat-priority"
        ),
        "fetched_at": "2026-09-23T10:00:00+00:00",
        "source_updated_at": "2026-08-10T19:20:00",
        "spell_ids": [49998, 50842],
        "part": 0,
        "part_count": 1,
    }


def test_optional_metadata_is_omitted_rather_than_empty():
    parsed = guide(section("x", anchor=None))
    parsed = ParsedGuide(
        **{**parsed.__dict__, "patch": None, "source_updated_at": None, "authors": ()}
    )

    [chunk] = chunk_guide(parsed)

    assert chunk.metadata["source_url"] == PAGE.url
    assert chunk.metadata["attribution"] == f"Copied from Icy Veins: {PAGE.url}"
    for key in ("patch", "source_updated_at", "authors", "spell_ids", "item_ids"):
        assert key not in chunk.metadata


def test_long_section_is_split_on_paragraph_boundaries():
    paragraph = "word " * 200  # ~1000 chars
    text = "\n\n".join(f"P{i} {paragraph.strip()}" for i in range(5))

    chunks = chunk_guide(guide(section(text)))

    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert [c.metadata["part"] for c in chunks] == list(range(len(chunks)))
    assert {c.metadata["part_count"] for c in chunks} == {len(chunks)}
    bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
    assert all(body.startswith("P") for body in bodies)  # never cut mid-paragraph
    assert all(c.text.startswith("Blood Death Knight Stat Priority") for c in chunks)


def test_oversized_table_is_split_by_rows_repeating_its_header():
    header = "| Ability | Description |\n| --- | --- |"
    rows = "\n".join(f"| Spell {i} | {'effect ' * 20}|" for i in range(40))

    chunks = chunk_guide(guide(section(f"{header}\n{rows}")))

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        assert body.startswith(header)
        assert len(chunk.text) <= MAX_CHUNK_CHARS


def test_oversized_list_is_split_by_lines():
    items = "\n".join(f"{i}. {'do the thing ' * 15}" for i in range(1, 30))

    chunks = chunk_guide(guide(section(items)))

    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
