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


class ConfigError(Exception):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    voyage_api_key: str | None = field(repr=False)
    voyage_model: str = DEFAULT_VOYAGE_MODEL
    chroma_dir: Path = DEFAULT_CHROMA_DIR

    def require_voyage_api_key(self) -> str:
        if not self.voyage_api_key:
            raise ConfigError(
                "VOYAGE_API_KEY is not set: copy .env.example to .env and fill it in"
            )
        return self.voyage_api_key


def load_settings(env: Mapping[str, str | None] | None = None) -> Settings:
    if env is None:
        env = {**dotenv_values(".env"), **os.environ}
    api_key = (env.get("VOYAGE_API_KEY") or "").strip()
    return Settings(
        voyage_api_key=api_key or None,
        voyage_model=env.get("VOYAGE_MODEL") or DEFAULT_VOYAGE_MODEL,
        chroma_dir=Path(env.get("PRESTIE_CHROMA_DIR") or DEFAULT_CHROMA_DIR),
    )
