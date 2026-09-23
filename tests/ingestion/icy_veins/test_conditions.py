from bs4 import BeautifulSoup

from prestie.ingestion.icy_veins.conditions import ConditionLabels

ROTATION_SWITCHES = """
<div class="rotation_switches">
  <div class="rotation_presets">
    <div class="rotation_presets_header">Deathbringer</div>
    <div class="rotation_preset_item" data-preset="preset-1 talent-4">Single Target</div>
  </div>
  <div class="rotation_presets">
    <div class="rotation_presets_header">San'layn</div>
    <div class="rotation_preset_item" data-preset="preset-2 ">Single Target</div>
  </div>
  <div class="rotation_switch" id="rotation_switch_talent-2"><label>Consumption</label></div>
</div>
"""


def labels_from(html: str) -> ConditionLabels:
    return ConditionLabels.from_content(BeautifulSoup(html, "lxml"))


def element(classes: str):
    return BeautifulSoup(f'<p class="{classes}">x</p>', "lxml").p


def test_level_range_class_becomes_level_label():
    assert labels_from("").describe(element("bdk_level_71_90")) == "[Levels 71-90]"


def test_open_ended_level_class_becomes_minimum_level_label():
    assert labels_from("").describe(element("bdk_level_27")) == "[Level 27+]"


def test_preset_on_uses_hero_talent_name_from_page():
    labels = labels_from(ROTATION_SWITCHES)

    assert (
        labels.describe(element("rotation_line_preset-1_on")) == "[Deathbringer only]"
    )


def test_preset_off_alone_is_negated():
    labels = labels_from(ROTATION_SWITCHES)

    assert labels.describe(element("rotation_line_preset-2_off")) == "[Not San'layn]"


def test_on_condition_makes_redundant_off_condition_implicit():
    labels = labels_from(ROTATION_SWITCHES)
    both = element("rotation_line_preset-2_on rotation_line_preset-1_off")

    assert labels.describe(both) == "[San'layn only]"


def test_talent_switch_uses_talent_name():
    labels = labels_from(ROTATION_SWITCHES)

    assert labels.describe(element("rotation_line_talent-2_on")) == "[With Consumption]"


def test_unknown_key_falls_back_to_raw_key():
    assert labels_from("").describe(element("rotation_line_preset-9_on")) == (
        "[preset-9 only]"
    )


def test_element_without_condition_has_no_label():
    assert labels_from(ROTATION_SWITCHES).describe(element("some-style")) is None
