"""Runtime settings, read from the environment and an optional `.env` file.

Real environment variables win over `.env` values. The API key is only
required by commands that call Voyage, so `prestie scrape` works without it.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_VOYAGE_MODEL = "voyage-4-large"
DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_CLAUDE_MODEL = "claude-opus-5"


class ConfigError(Exception):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    voyage_api_key: str | None = field(repr=False)
    voyage_model: str = DEFAULT_VOYAGE_MODEL
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    # None lets the Anthropic SDK resolve credentials itself (env var, `ant` profile).
    anthropic_api_key: str | None = field(default=None, repr=False)
    claude_model: str = DEFAULT_CLAUDE_MODEL
    # The addon's SavedVariables file (WTF/Account/<ACCOUNT>/SavedVariables/Prestie.lua).
    saved_variables_path: Path | None = None

    def require_saved_variables_path(self) -> Path:
        if self.saved_variables_path is None:
            raise ConfigError(
                "PRESTIE_SAVEDVARIABLES is not set: point it to "
                "WTF/Account/<ACCOUNT>/SavedVariables/Prestie.lua in .env"
            )
        return self.saved_variables_path

    def require_voyage_api_key(self) -> str:
        if not self.voyage_api_key:
            raise ConfigError(
                "VOYAGE_API_KEY is not set: copy .env.example to .env and fill it in"
            )
        return self.voyage_api_key


def load_settings(env: Mapping[str, str | None] | None = None) -> Settings:
    if env is None:
        env = {**dotenv_values(".env"), **os.environ}
    return Settings(
        voyage_api_key=_secret(env, "VOYAGE_API_KEY"),
        voyage_model=env.get("VOYAGE_MODEL") or DEFAULT_VOYAGE_MODEL,
        chroma_dir=Path(env.get("PRESTIE_CHROMA_DIR") or DEFAULT_CHROMA_DIR),
        anthropic_api_key=_secret(env, "ANTHROPIC_API_KEY"),
        claude_model=env.get("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL,
        saved_variables_path=_path(env, "PRESTIE_SAVEDVARIABLES"),
    )


def _path(env: Mapping[str, str | None], name: str) -> Path | None:
    value = (env.get(name) or "").strip()
    return Path(value) if value else None


def _secret(env: Mapping[str, str | None], name: str) -> str | None:
    return (env.get(name) or "").strip() or None
