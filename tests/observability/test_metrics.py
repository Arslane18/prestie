from datetime import date

import pytest

from prestie.observability.metrics import percentile, summarize_traces


def usage(inp=0, out=0, read=0, write=0):
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
    }


def search(query, *distances, spec="shadow-priest", alternatives=(), error=False):
    tool_input = {"query": query, **({"spec": spec} if spec else {})}
    if alternatives:
        tool_input["alternative_queries"] = list(alternatives)
    return {
        "name": "search_knowledge_base",
        "input": tool_input,
        "is_error": error,
        "duration_s": 0.5,
        "details": {
            "queries": [query, *alternatives],
            "results": [{"section": f"S{d}", "distance": d} for d in distances],
        },
    }


def trace(
    *,
    day="2026-09-29",
    session="s1",
    outcome="answered",
    duration=10.0,
    ttft=4.0,
    cost=0.05,
    requests=None,
    tools=(),
    question="q",
):
    return {
        "trace_id": f"{session}-{question}",
        "session_id": session,
        "started_at": f"{day}T20:00:00+00:00",
        "question": question,
        "outcome": outcome,
        "duration_s": duration,
        "time_to_first_token_s": ttft,
        "cost_usd": cost,
        "requests": requests
        or [{"model": "claude-opus-5", "duration_s": 6.0, "usage": usage(inp=10)}],
        "tools": list(tools),
    }


def test_percentile_uses_the_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None


def test_volume_outcomes_and_sessions():
    summary = summarize_traces(
        [
            trace(session="a"),
            trace(session="a", outcome="error"),
            trace(session="b", outcome="refused"),
        ]
    )

    assert summary.turns == 3
    assert summary.sessions == 2
    assert summary.outcomes == {"answered": 1, "error": 1, "refused": 1}


def test_cost_total_per_turn_and_per_day():
    summary = summarize_traces(
        [
            trace(day="2026-09-28", cost=0.10),
            trace(day="2026-09-29", cost=0.05),
            trace(day="2026-09-29", cost=0.03),
        ]
    )

    assert summary.cost_usd == pytest.approx(0.18)
    assert summary.cost_per_turn_usd == pytest.approx(0.06)
    assert summary.cost_per_day_usd == {
        date(2026, 9, 28): pytest.approx(0.10),
        date(2026, 9, 29): pytest.approx(0.08),
    }


def test_unpriced_turns_are_counted_apart_from_the_cost():
    summary = summarize_traces([trace(cost=0.05), trace(cost=None)])

    assert summary.cost_usd == pytest.approx(0.05)
    assert summary.unpriced_turns == 1


def test_cache_read_share_is_over_every_input_token():
    requests = [
        {"model": "m", "duration_s": 1.0, "usage": usage(inp=100, write=300)},
        {"model": "m", "duration_s": 1.0, "usage": usage(inp=100, read=500)},
    ]

    summary = summarize_traces([trace(requests=requests)])

    assert summary.cache_read_share == pytest.approx(500 / 1000)


def test_latency_percentiles_and_where_the_time_goes():
    requests = [{"model": "m", "duration_s": 6.0, "usage": usage()}]
    turns = [
        trace(duration=10.0, ttft=4.0, requests=requests, tools=[search("a", 0.4)]),
        trace(duration=20.0, ttft=8.0, requests=requests, tools=[]),
    ]

    summary = summarize_traces(turns)

    assert summary.duration_p50_s == 10.0
    assert summary.duration_p95_s == 20.0
    assert summary.ttft_p50_s == 4.0
    assert summary.mean_request_s == 6.0
    assert summary.mean_tool_s == pytest.approx(0.25)  # 0.5 s over two turns
    assert summary.mean_other_s == pytest.approx(15.0 - 6.0 - 0.25)


def test_search_habits():
    summary = summarize_traces(
        [
            trace(tools=[search("a", 0.4), search("b", 0.5, alternatives=["c"])]),
            trace(tools=[search("d", 0.4, spec=None), search("e", error=True)]),
        ]
    )

    assert summary.searches == 4
    assert summary.searches_per_turn == 2.0
    assert summary.alternative_query_share == 0.25
    assert summary.unfiltered_searches == 1
    assert summary.tool_errors == 1
    assert summary.tool_calls == {"search_knowledge_base": 4}


def test_worst_covered_searches_come_first():
    summary = summarize_traces(
        [
            trace(question="clear", tools=[search("stats", 0.35, 0.5)]),
            trace(question="vague", tools=[search("ranged tank", 0.71, 0.74)]),
            trace(question="nothing", tools=[search("pvp arena")]),
        ],
        gaps=2,
    )

    assert [(g.question, g.best_distance) for g in summary.corpus_gaps] == [
        ("nothing", None),  # no passage at all: the clearest gap
        ("vague", 0.71),
    ]
    assert summary.corpus_gaps[1].queries == ("ranged tank",)
    assert summary.corpus_gaps[1].spec == "shadow-priest"


def test_turns_can_be_limited_to_recent_days():
    summary = summarize_traces(
        [trace(day="2026-09-20"), trace(day="2026-09-29")], since=date(2026, 9, 25)
    )

    assert summary.turns == 1


def test_no_traces_gives_an_empty_summary():
    summary = summarize_traces([])

    assert summary.turns == 0
    assert summary.cost_per_turn_usd is None
    assert summary.duration_p50_s is None
    assert summary.corpus_gaps == ()
