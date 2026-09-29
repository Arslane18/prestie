"""What a Claude request costs, from the usage the API reports.

Prompt caching changes the input price: tokens written to the cache cost
1.25x the input price (5-minute cache), tokens read from it 0.1x. A model
missing from the table has no cost rather than a wrong one.
"""

from collections.abc import Mapping

from prestie.agent.agent import Usage

TOKENS_PER_PRICE_UNIT = 1_000_000
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1
# USD per million tokens: (input, output).
PRICES: Mapping[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),  # server-side fallback target
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def usage_cost_usd(model: str, usage: Usage) -> float | None:
    price = _price(model)
    if price is None:
        return None
    input_price, output_price = price
    input_equivalent = (
        usage.input_tokens
        + CACHE_WRITE_MULTIPLIER * usage.cache_creation_input_tokens
        + CACHE_READ_MULTIPLIER * usage.cache_read_input_tokens
    )
    return (
        input_equivalent * input_price + usage.output_tokens * output_price
    ) / TOKENS_PER_PRICE_UNIT


def _price(model: str) -> tuple[float, float] | None:
    # A dated snapshot ("<alias>-2026...") is priced like its alias.
    return next(
        (
            price
            for alias, price in PRICES.items()
            if model == alias or model.startswith(f"{alias}-2")
        ),
        None,
    )
