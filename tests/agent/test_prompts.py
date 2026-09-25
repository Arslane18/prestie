import pytest

from prestie.agent.prompts import (
    MAX_ANSWER_LINES,
    MAX_LINE_CHARS,
    MAX_PLAYER_LEVEL,
    PlayerContext,
    build_system_prompt,
)


def test_player_context_describes_level_class_spec_and_hero_talent():
    player = PlayerContext(level=80, hero_talent="San'layn")

    assert player.describe() == (
        "Level 80 Death Knight, Blood specialization, San'layn hero talent"
    )


def test_unknown_hero_talent_is_stated_explicitly():
    assert "hero talent not specified" in PlayerContext(level=45).describe()


@pytest.mark.parametrize("level", [0, MAX_PLAYER_LEVEL + 1])
def test_out_of_range_level_is_rejected(level):
    with pytest.raises(ValueError, match="level"):
        PlayerContext(level=level)


def test_unknown_hero_talent_name_is_rejected():
    with pytest.raises(ValueError, match="hero talent"):
        PlayerContext(level=80, hero_talent="Rider of the Apocalypse")


def test_system_prompt_embeds_player_context_and_ground_rules():
    prompt = build_system_prompt(PlayerContext(level=80, hero_talent="Deathbringer"))

    assert "Level 80 Death Knight, Blood specialization, Deathbringer" in prompt
    assert "search_knowledge_base" in prompt
    assert "Icy Veins" in prompt
    assert "French" in prompt


def test_system_prompt_is_deterministic_for_prompt_caching():
    player = PlayerContext(level=80)

    assert build_system_prompt(player) == build_system_prompt(player)


def test_system_prompt_restricts_general_knowledge_to_stable_game_concepts():
    prompt = build_system_prompt(PlayerContext(level=80))

    assert "general knowledge" in prompt
    assert "patch" in prompt


def test_system_prompt_asks_to_match_answer_depth_to_the_question():
    prompt = build_system_prompt(PlayerContext(level=80))

    assert "beginner" in prompt
    assert "short" in prompt


def test_system_prompt_states_the_line_budget_the_evaluation_enforces():
    prompt = build_system_prompt(PlayerContext(level=80))

    # Same numbers as the `concise` check: 8 displayed lines of 100 characters.
    assert f"at most {MAX_ANSWER_LINES} lines" in prompt
    assert f"{MAX_LINE_CHARS} characters" in prompt
    assert "counts as several" in prompt


def test_system_prompt_forbids_unsourced_remarks_about_the_player():
    prompt = build_system_prompt(PlayerContext(level=80))

    assert "Do not add remarks about the player's level" in prompt


def test_addon_mode_prompt_points_to_the_character_state_tool():
    prompt = build_system_prompt(None)

    assert "get_character_state" in prompt
    assert "Do not ask the player" in prompt
    assert "/reload" in prompt
    assert "only cover the specs listed" in prompt


def test_manual_mode_prompt_does_not_mention_the_character_state_tool():
    # The agent evaluation runs in manual mode: its prompt must not change.
    assert "get_character_state" not in build_system_prompt(PlayerContext(level=80))


def test_addon_mode_prompt_explains_quest_details():
    prompt = build_system_prompt(None)

    assert "get_quest_details" in prompt
    assert "walkthrough" in prompt


def test_manual_mode_prompt_does_not_mention_quest_details():
    assert "get_quest_details" not in build_system_prompt(PlayerContext(level=80))


def test_addon_mode_prompt_counts_state_remarks_in_the_line_budget():
    prompt = build_system_prompt(None)

    assert "counts toward the line budget" in prompt


def test_addon_mode_prompt_forbids_guessing_quest_locations():
    prompt = build_system_prompt(None)

    assert "do not infer where" in prompt
    assert "interface tips" in prompt


def test_prompt_lists_every_covered_spec_with_its_search_filter():
    from prestie.catalog import COVERED_SPECS

    prompt = build_system_prompt(PlayerContext(level=80))

    for spec in COVERED_SPECS:
        assert f"{spec.name} (spec={spec.key})" in prompt
    assert "Set the spec filter to the player's spec" in prompt


def test_class_wide_questions_keep_the_player_spec_filter():
    prompt = build_system_prompt(PlayerContext(level=80))

    assert "question about the class as a whole still uses the player's spec" in prompt


def test_addon_mode_asks_the_spec_when_the_state_is_unavailable():
    prompt = build_system_prompt(None)

    assert "ask which class and spec the player plays" in prompt
    assert "instead of answering for every covered spec" in prompt
