import pytest

from prestie.agent.prompts import MAX_PLAYER_LEVEL, PlayerContext, build_system_prompt


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
