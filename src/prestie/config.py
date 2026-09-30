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
# Traces of real chat turns (questions included): under data/, never committed.
DEFAULT_TRACE_DIR = Path("data/traces")
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
# The agent's model: Claude through the API, or a local llama-server.
LLM_PROVIDERS = ("anthropic", "local")
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_LOCAL_LLM_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_LLM_MODEL = "qwen3.5-9b"
TRUE_VALUES = ("1", "true", "yes", "on")
FALSE_VALUES = ("", "0", "false", "no", "off")
DEFAULT_BLIZZARD_REGION = "eu"
BLIZZARD_REGIONS = ("us", "eu", "kr", "tw")


class ConfigError(Exception):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    voyage_api_key: str | None = field(repr=False)
    voyage_model: str = DEFAULT_VOYAGE_MODEL
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    trace_dir: Path = DEFAULT_TRACE_DIR
    # None lets the Anthropic SDK resolve credentials itself (env var, `ant` profile).
    anthropic_api_key: str | None = field(default=None, repr=False)
    claude_model: str = DEFAULT_CLAUDE_MODEL
    llm_provider: str = DEFAULT_LLM_PROVIDER
    local_llm_url: str = DEFAULT_LOCAL_LLM_URL
    # A label sent to llama-server (which serves the one model it loaded) and
    # recorded in traces and eval rows.
    local_llm_model: str = DEFAULT_LOCAL_LLM_MODEL
    local_llm_api_key: str | None = field(default=None, repr=False)
    local_llm_thinking: bool = False
    # The addon's SavedVariables file (WTF/Account/<ACCOUNT>/SavedVariables/Prestie.lua).
    saved_variables_path: Path | None = None
    # Battle.net app credentials (https://develop.battle.net/access/clients).
    blizzard_client_id: str | None = field(default=None, repr=False)
    blizzard_client_secret: str | None = field(default=None, repr=False)
    blizzard_region: str = DEFAULT_BLIZZARD_REGION

    @property
    def agent_model(self) -> str:
        """The model the agent runs on, whichever the provider."""
        return self.local_llm_model if self.llm_provider == "local" else self.claude_model

    def require_blizzard_credentials(self) -> tuple[str, str]:
        missing = [
            name
            for name, value in (
                ("BLIZZARD_CLIENT_ID", self.blizzard_client_id),
                ("BLIZZARD_CLIENT_SECRET", self.blizzard_client_secret),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                f"{' and '.join(missing)} not set: create a client on "
                "https://develop.battle.net/access/clients and fill in .env"
            )
        return self.blizzard_client_id, self.blizzard_client_secret  # type: ignore[return-value]

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
        trace_dir=Path(env.get("PRESTIE_TRACE_DIR") or DEFAULT_TRACE_DIR),
        anthropic_api_key=_secret(env, "ANTHROPIC_API_KEY"),
        claude_model=env.get("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL,
        llm_provider=_choice(env, "PRESTIE_LLM", LLM_PROVIDERS, DEFAULT_LLM_PROVIDER),
        local_llm_url=env.get("PRESTIE_LOCAL_LLM_URL") or DEFAULT_LOCAL_LLM_URL,
        local_llm_model=env.get("PRESTIE_LOCAL_LLM_MODEL") or DEFAULT_LOCAL_LLM_MODEL,
        local_llm_api_key=_secret(env, "PRESTIE_LOCAL_LLM_API_KEY"),
        local_llm_thinking=_flag(env, "PRESTIE_LOCAL_LLM_THINKING"),
        saved_variables_path=_path(env, "PRESTIE_SAVEDVARIABLES"),
        blizzard_client_id=_secret(env, "BLIZZARD_CLIENT_ID"),
        blizzard_client_secret=_secret(env, "BLIZZARD_CLIENT_SECRET"),
        blizzard_region=_region(env),
    )


def _region(env: Mapping[str, str | None]) -> str:
    region = (env.get("BLIZZARD_REGION") or DEFAULT_BLIZZARD_REGION).strip().lower()
    if region not in BLIZZARD_REGIONS:
        raise ConfigError(
            f"BLIZZARD_REGION must be one of {', '.join(BLIZZARD_REGIONS)}, "
            f"got '{region}'"
        )
    return region


def _choice(
    env: Mapping[str, str | None], name: str, choices: tuple[str, ...], default: str
) -> str:
    value = (env.get(name) or default).strip().lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, got '{value}'")
    return value


def _flag(env: Mapping[str, str | None], name: str) -> bool:
    value = (env.get(name) or "").strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be true or false, got '{value}'")


def _path(env: Mapping[str, str | None], name: str) -> Path | None:
    value = (env.get(name) or "").strip()
    return Path(value) if value else None


def _secret(env: Mapping[str, str | None], name: str) -> str | None:
    return (env.get(name) or "").strip() or None
