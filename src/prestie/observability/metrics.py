"""Aggregate metrics over traces: cost, cache, latency, tool use, corpus gaps.

Pure functions over trace dicts (as stored in data/traces/*.jsonl), so the
report is testable without any real traffic.

Corpus gaps are ranked, not thresholded: a search whose best passage is far
from the query (large cosine distance) probably asks about something the
guides do not cover, but what "far" means depends on the embedding model and
has not been calibrated yet. The worst-covered searches are listed first; a
search that returned nothing at all ranks above every other.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from prestie.agent.tools import SEARCH_TOOL_NAME
from prestie.observability.checks import online_flags

DEFAULT_GAPS = 10
THUMBS_DOWN = "thumbs_down"  # the player's own verdict, next to online checks
DOWN = "down"
CACHE_READ = "cache_read_input_tokens"
INPUT_KEYS = ("input_tokens", CACHE_READ, "cache_creation_input_tokens")

Trace = Mapping[str, Any]


@dataclass(frozen=True)
class CorpusGap:
    question: str
    queries: tuple[str, ...]
    spec: str | None
    best_distance: float | None  # None: the search returned no passage
    trace_id: str


@dataclass(frozen=True)
class FlaggedTurn:
    question: str
    flags: tuple[str, ...]
    trace_id: str


@dataclass(frozen=True)
class TraceSummary:
    turns: int
    sessions: int
    outcomes: Mapping[str, int]
    cost_usd: float
    cost_per_turn_usd: float | None
    cost_per_day_usd: Mapping[date, float]
    unpriced_turns: int
    cache_read_share: float | None
    duration_p50_s: float | None
    duration_p95_s: float | None
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    mean_request_s: float | None
    mean_tool_s: float | None
    mean_other_s: float | None
    searches: int
    searches_per_turn: float | None
    alternative_query_share: float | None
    unfiltered_searches: int
    tool_errors: int
    tool_calls: Mapping[str, int]
    corpus_gaps: tuple[CorpusGap, ...]
    flagged_turns: tuple[FlaggedTurn, ...]
    feedback: Mapping[str, int]  # vote -> number of voted turns


def percentile(values: Sequence[float], rank: float) -> float | None:
    """Nearest-rank percentile: always one of the observed values."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(rank / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize_traces(
    traces: Iterable[Trace],
    *,
    since: date | None = None,
    gaps: int = DEFAULT_GAPS,
    feedback: Mapping[str, str] | None = None,
) -> TraceSummary:
    votes = feedback or {}
    selected = [t for t in traces if since is None or _day(t) >= since]
    tools = [tool for t in selected for tool in t.get("tools", ())]
    searches = [tool for tool in tools if tool.get("name") == SEARCH_TOOL_NAME]
    priced = [t["cost_usd"] for t in selected if t.get("cost_usd") is not None]
    durations = [t["duration_s"] for t in selected]
    first_tokens = [
        t["time_to_first_token_s"]
        for t in selected
        if t.get("time_to_first_token_s") is not None
    ]
    return TraceSummary(
        turns=len(selected),
        sessions=len({t.get("session_id") for t in selected}),
        outcomes=dict(Counter(t.get("outcome") for t in selected)),
        cost_usd=sum(priced),
        cost_per_turn_usd=_ratio(sum(priced), len(priced)),
        cost_per_day_usd=_cost_per_day(selected),
        unpriced_turns=len(selected) - len(priced),
        cache_read_share=_cache_read_share(selected),
        duration_p50_s=percentile(durations, 50),
        duration_p95_s=percentile(durations, 95),
        ttft_p50_s=percentile(first_tokens, 50),
        ttft_p95_s=percentile(first_tokens, 95),
        **_time_split(selected),
        searches=len(searches),
        searches_per_turn=_ratio(len(searches), len(selected)),
        alternative_query_share=_ratio(
            sum(1 for s in searches if s["input"].get("alternative_queries")),
            len(searches),
        ),
        unfiltered_searches=sum(1 for s in searches if not s["input"].get("spec")),
        tool_errors=sum(1 for tool in tools if tool.get("is_error")),
        tool_calls=dict(Counter(tool.get("name") for tool in tools)),
        corpus_gaps=_worst_covered(selected, gaps),
        flagged_turns=tuple(
            FlaggedTurn(t.get("question", ""), flags, t.get("trace_id", ""))
            for t in selected
            if (flags := _flags(t, votes))
        ),
        feedback=dict(
            Counter(votes[t["trace_id"]] for t in selected if t.get("trace_id") in votes)
        ),
    )


def _flags(trace: Trace, votes: Mapping[str, str]) -> tuple[str, ...]:
    voted_down = votes.get(trace.get("trace_id", "")) == DOWN
    return (*online_flags(trace), *((THUMBS_DOWN,) if voted_down else ()))


def _day(trace: Trace) -> date:
    return datetime.fromisoformat(trace["started_at"]).date()


def _ratio(part: float, whole: float) -> float | None:
    return part / whole if whole else None


def _cost_per_day(traces: Sequence[Trace]) -> dict[date, float]:
    days: dict[date, float] = {}
    for trace in traces:
        if trace.get("cost_usd") is not None:
            day = _day(trace)
            days = {**days, day: days.get(day, 0.0) + trace["cost_usd"]}
    return dict(sorted(days.items()))


def _cache_read_share(traces: Sequence[Trace]) -> float | None:
    usages = [r["usage"] for t in traces for r in t.get("requests", ())]
    total = sum(u.get(key, 0) for u in usages for key in INPUT_KEYS)
    return _ratio(sum(u.get(CACHE_READ, 0) for u in usages), total)


def _time_split(traces: Sequence[Trace]) -> dict[str, float | None]:
    """Mean seconds per turn spent in API requests, in tools, and elsewhere
    (streaming to the consumer, the agent's own work)."""
    if not traces:
        return {"mean_request_s": None, "mean_tool_s": None, "mean_other_s": None}
    count = len(traces)
    request = sum(r["duration_s"] for t in traces for r in t.get("requests", ()))
    tool = sum(x["duration_s"] for t in traces for x in t.get("tools", ()))
    total = sum(t["duration_s"] for t in traces)
    return {
        "mean_request_s": request / count,
        "mean_tool_s": tool / count,
        "mean_other_s": (total - request - tool) / count,
    }


def _worst_covered(traces: Sequence[Trace], limit: int) -> tuple[CorpusGap, ...]:
    gaps = [
        CorpusGap(
            question=trace.get("question", ""),
            queries=tuple(tool["details"].get("queries", ())),
            spec=tool["input"].get("spec"),
            best_distance=min(
                (r["distance"] for r in tool["details"].get("results", ())),
                default=None,
            ),
            trace_id=trace.get("trace_id", ""),
        )
        for trace in traces
        for tool in trace.get("tools", ())
        if tool.get("name") == SEARCH_TOOL_NAME
        and not tool.get("is_error")
        and "details" in tool
    ]
    # No passage at all first, then the largest best distance.
    ranked = sorted(
        gaps,
        key=lambda g: (g.best_distance is not None, -(g.best_distance or 0.0)),
    )
    return tuple(ranked[:limit])
