from bs4 import BeautifulSoup

from prestie.ingestion.icy_veins.blocks import Block, Heading, walk_content
from prestie.ingestion.icy_veins.conditions import ConditionLabels
from tests.ingestion.icy_veins.html_fixtures import heading, item, spell


def walk(html: str) -> list[Block | Heading]:
    content = BeautifulSoup(f'<div class="c">{html}</div>', "lxml").select_one("div.c")
    return list(walk_content(content, ConditionLabels.from_content(content)))


def texts(html: str) -> list[str]:
    return [event.text for event in walk(html) if isinstance(event, Block)]


def test_paragraph_renders_spell_names_with_clean_spacing():
    html = f"<p>Use {spell(49998, 'Death Strike')}, then\n   {spell(50842, 'Blood Boil')}.</p>"

    assert texts(html) == ["Use Death Strike, then Blood Boil."]


def test_block_collects_spell_and_item_ids_once():
    html = (
        f"<p>{spell(49998, 'Death Strike')} {spell(49998, 'Death Strike')} "
        f"{item(122263, 'Helm')}</p>"
    )

    [block] = walk(html)

    assert block.spell_ids == (49998,)
    assert block.item_ids == (122263,)


def test_heading_event_has_level_text_and_anchor_without_number():
    [event] = walk(heading(3, "1.2.", "Stat Priority", anchor="stat-priority"))

    assert event == Heading(level=3, text="Stat Priority", anchor="stat-priority")


def test_headings_nested_in_layout_blocks_are_found_in_document_order():
    html = (
        '<div class="image_block"><div class="image_block_header">'
        f"{heading(2, '2.', 'Mechanics', 'mechanics')}</div>"
        f'<div class="image_block_content"><p>Intro</p>{heading(3, "2.1.", "Runes")}'
        "<p>Runes text</p></div></div>"
    )

    kinds = [(type(event).__name__, getattr(event, "text", "")) for event in walk(html)]

    assert kinds == [
        ("Heading", "Mechanics"),
        ("Block", "Intro"),
        ("Heading", "Runes"),
        ("Block", "Runes text"),
    ]


def test_unordered_and_nested_ordered_lists_render_as_markdown():
    html = "<ul><li>First<ol><li>Sub one</li><li>Sub two</li></ol></li><li>Second</li></ul>"

    assert texts(html) == ["- First\n  1. Sub one\n  2. Sub two\n- Second"]


def test_table_renders_as_markdown_table():
    html = (
        "<table><thead><tr><th>Stat</th><th>10% DR</th></tr></thead>"
        "<tbody><tr><td>Haste</td><td>1320</td></tr></tbody></table>"
    )

    assert texts(html) == ["| Stat | 10% DR |\n| --- | --- |\n| Haste | 1320 |"]


def test_table_without_header_row_uses_first_row_as_header():
    html = "<table><tr><td>Slot</td><td>Item</td></tr><tr><td>Helm</td><td>Hat</td></tr></table>"

    assert texts(html) == ["| Slot | Item |\n| --- | --- |\n| Helm | Hat |"]


def test_conditional_list_items_are_prefixed_with_their_condition():
    html = (
        '<ol><li>Always</li><li class="bdk_level_71_90">Late game</li></ol>'
        '<p class="bdk_level_1_10">Early game</p>'
    )

    assert texts(html) == [
        "1. Always\n2. [Levels 71-90] Late game",
        "[Levels 1-10] Early game",
    ]


def test_conditional_heading_carries_its_condition():
    html = (
        '<div class="heading_container heading_number_3">'
        '<h3 class="bdk_level_10_20" id="early">Early</h3></div>'
    )

    [event] = walk(html)

    assert event.text == "Early [Levels 10-20]"


def test_condition_on_wrapper_applies_to_inner_blocks():
    html = '<div class="bdk_level_50_60"><p>One</p><p>Two</p></div>'

    assert texts(html) == ["[Levels 50-60] One", "[Levels 50-60] Two"]


def test_boilerplate_and_interactive_widgets_are_skipped():
    html = (
        '<div class="changelog_wrapper"><p>Changelog</p></div>'
        '<section class="internal-links"><h3>In The Same Category</h3></section>'
        '<div class="leveling_slider_container"><span>Level: 80</span></div>'
        '<div class="rotation_switches"><div>Deathbringer</div></div>'
        '<div class="midnight-skill-builder-embed"></div>'
        "<script>var x = 1;</script>"
        "<p>Kept</p>"
    )

    assert texts(html) == ["Kept"]


def test_bare_inline_text_inside_layout_div_becomes_a_block():
    html = f'<div class="image_block_content">Press {spell(1, "Marrowrend")} often<p>After</p></div>'

    assert texts(html) == ["Press Marrowrend often", "After"]


def test_faq_dropdown_renders_as_question_and_answer():
    html = (
        '<details class="faq-block__dropdown"><summary><span class="faq-block__badge">FAQ</span>'
        '<span class="faq-block__question">Should I sim?</span></summary>'
        '<div class="faq-block__answer"><p>Yes, mostly.</p><p>Not for survival.</p></div></details>'
    )

    assert texts(html) == ["Q: Should I sim?\nA: Yes, mostly.\nNot for survival."]


def test_talent_export_string_renders_title_and_code():
    html = (
        '<details class="export-string"><summary><span class="export-string__title">'
        "Blood Raid - San'layn</span></summary>"
        '<span class="export-string__code">CoPAAAAA</span></details>'
    )

    assert texts(html) == ["Talent import string (Blood Raid - San'layn): CoPAAAAA"]


def test_generic_details_keeps_summary_and_body():
    html = "<details><summary>Exterminate rules</summary><p>Not copied.</p></details>"

    assert texts(html) == ["Exterminate rules", "Not copied."]


def test_performance_overview_renders_one_line_per_rating():
    html = """
    <div class="performance-overview">
      <div class="performance-overview__tabs">
        <button class="performance-overview__tab" data-score="5.0" data-target="p0">
          <span class="performance-overview__label">Damage</span></button>
        <button class="performance-overview__tab" data-score="4.5" data-target="p1">
          <span class="performance-overview__label">Raid</span></button>
      </div>
      <div class="performance-overview__panel" id="p0"><summary class="performance-header">
        Overperforming</summary><p>Overtuned in AoE.</p></div>
      <div class="performance-overview__panel" id="p1"><summary class="performance-header">
        Great</summary><p>Solid choice.</p></div>
    </div>"""

    assert texts(html) == [
        (
            "Damage: 5.0/5 (Overperforming) — Overtuned in AoE.\n"
            "Raid: 4.5/5 (Great) — Solid choice."
        )
    ]


def test_empty_blocks_are_not_emitted():
    assert texts("<p>   </p><div><svg></svg></div>") == []


def stat(name: str) -> str:
    return (
        '<div class="stat-container"><div class="stat-icon-name">'
        f'<div class="stat-icon"></div><div class="stat-name">{name}</div></div></div>'
    )


def separator(kind: str) -> str:
    return (
        f'<div class="stat-separator"><div class="separator-icon {kind}">'
        '<i class="ri-arrow-drop-right-line"></i></div></div>'
    )


def test_stat_priority_widget_keeps_comparison_operators():
    html = (
        '<div class="stat-priority-widget"><div class="stat-priority-widget-inner">'
        + stat("Strength")
        + separator("false")
        + stat("Haste")
        + separator("greater-equal")
        + stat("Critical Strike")
        + separator("equal")
        + stat("Mastery")
        + "</div></div>"
    )

    assert texts(html) == ["Strength > Haste >= Critical Strike = Mastery"]


def test_macro_export_string_is_rendered_as_macro():
    html = (
        '<details class="export-string"><summary><span class="export-string__title">'
        "Mind Freeze Macro</span></summary>"
        '<div class="export-string__code_large_macro">/cast [@focus] Mind Freeze</div>'
        "</details>"
    )

    assert texts(html) == ["Macro (Mind Freeze Macro):\n/cast [@focus] Mind Freeze"]
