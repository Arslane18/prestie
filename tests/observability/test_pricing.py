import pytest

from prestie.agent.agent import Usage
from prestie.pricing import usage_cost_usd


def test_cost_counts_cache_writes_at_1_25x_and_reads_at_0_1x_the_input_price():
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )

    # Opus 5: $5 in, $25 out -> 5 + 25 + 0.5 + 6.25
    assert usage_cost_usd("claude-opus-5", usage) == pytest.approx(36.75)


def test_dated_snapshots_use_their_alias_price():
    usage = Usage(output_tokens=1_000_000)

    assert usage_cost_usd("claude-sonnet-5-20260801", usage) == pytest.approx(10.0)


def test_unknown_models_have_no_cost_rather_than_a_wrong_one():
    assert usage_cost_usd("some-local-model", Usage(input_tokens=10)) is None
