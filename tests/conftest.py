"""Isolation from the developer's own configuration.

`load_settings()` reads `.env`, where real environment variables win. Tests
that go through it would otherwise follow whatever the developer set for
daily use (e.g. PRESTIE_LLM=local on the local-model branch); pinning the
model choice here keeps them on the defaults. Tests of the local model pass
their settings explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def default_model_settings(monkeypatch):
    monkeypatch.setenv("PRESTIE_LLM", "anthropic")
    monkeypatch.setenv("PRESTIE_LOCAL_LLM_THINKING", "false")
