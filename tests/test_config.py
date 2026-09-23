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
