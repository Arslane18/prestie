from pathlib import Path

import pytest

from prestie.config import ConfigError, Settings, load_settings


def test_defaults_apply_when_only_the_key_is_set():
    settings = load_settings({"VOYAGE_API_KEY": "secret"})

    assert settings == Settings(
        voyage_api_key="secret",
        voyage_model="voyage-4-large",
        chroma_dir=Path("data/chroma"),
    )


def test_overrides_are_read_from_environment():
    settings = load_settings(
        {
            "VOYAGE_API_KEY": "secret",
            "VOYAGE_MODEL": "voyage-4-lite",
            "PRESTIE_CHROMA_DIR": "/tmp/kb",
        }
    )

    assert settings.voyage_model == "voyage-4-lite"
    assert settings.chroma_dir == Path("/tmp/kb")


def test_missing_key_only_fails_when_the_key_is_needed():
    settings = load_settings({})

    with pytest.raises(ConfigError, match="VOYAGE_API_KEY"):
        settings.require_voyage_api_key()


def test_blank_key_counts_as_missing():
    with pytest.raises(ConfigError):
        load_settings({"VOYAGE_API_KEY": "   "}).require_voyage_api_key()


def test_repr_never_leaks_the_key():
    assert "secret" not in repr(load_settings({"VOYAGE_API_KEY": "secret"}))


def test_claude_settings_default_to_opus_and_optional_key():
    settings = load_settings({})

    assert settings.claude_model == "claude-opus-5"
    assert settings.anthropic_api_key is None


def test_claude_settings_are_read_from_environment():
    settings = load_settings(
        {"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_MODEL": "claude-sonnet-5"}
    )

    assert settings.anthropic_api_key == "sk-ant-x"
    assert settings.claude_model == "claude-sonnet-5"
    assert "sk-ant-x" not in repr(settings)


def test_saved_variables_path_is_read_from_environment():
    settings = load_settings({"PRESTIE_SAVEDVARIABLES": "/wow/Prestie.lua"})

    assert settings.require_saved_variables_path() == Path("/wow/Prestie.lua")


def test_missing_saved_variables_path_fails_only_when_needed():
    settings = load_settings({})

    with pytest.raises(ConfigError, match="PRESTIE_SAVEDVARIABLES"):
        settings.require_saved_variables_path()
