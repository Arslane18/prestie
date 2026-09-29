from datetime import UTC, datetime
from itertools import count

import pytest

from prestie.agent.agent import (
    AgentError,
    AgentReply,
    RequestFinished,
    TextDelta,
    ToolCall,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    Usage,
)
from prestie.observability.trace import TurnRecorder

NOW = datetime(2026, 9, 29, 20, 0, tzinfo=UTC)
SEARCH = ToolCall("search_knowledge_base", {"query": "stats", "spec": "shadow-priest"})
HITS = {"queries": ["stats"], "results": [{"section": "S", "distance": 0.31}]}


class MemorySink:
    def __init__(self):
        self.traces = []

    def write(self, trace):
        self.traces.append(trace)


def recorder(sink, **overrides):
    ticks = count()
    options = {
        "session_id": "s1",
        "character": lambda: {"class": "PRIEST", "spec": "Ombre", "level": 90},
        "clock": lambda: float(next(ticks)),
        "now": lambda: NOW,
        "new_id": lambda: "t1",
        **overrides,
    }
    return TurnRecorder(sink, **options)


def answered_turn():
    yield ToolCallStarted(SEARCH)
    yield ToolCallFinished(SEARCH, is_error=False, duration_s=0.4, details=HITS)
    yield RequestFinished("claude-opus-5", Usage(input_tokens=100), "tool_use", 1.5)
    yield TextDelta("Hâte ")
    yield TextDelta("d'abord.")
    yield RequestFinished(
        "claude-opus-5",
        Usage(output_tokens=50, cache_read_input_tokens=4000),
        "end_turn",
        2.0,
    )
    yield TurnFinished(AgentReply("Hâte d'abord."))


def test_events_pass_through_unchanged():
    sink = MemorySink()

    events = list(recorder(sink).observe("Stats ?", answered_turn()))

    assert events == list(answered_turn())


def test_a_turn_becomes_one_trace():
    sink = MemorySink()

    list(recorder(sink).observe("Stats ?", answered_turn()))

    [trace] = sink.traces
    assert (trace.trace_id, trace.session_id, trace.turn) == ("t1", "s1", 1)
    assert trace.started_at == NOW
    assert trace.question == "Stats ?"
    assert trace.character == {"class": "PRIEST", "spec": "Ombre", "level": 90}
    assert trace.outcome == "answered"
    assert trace.answer == "Hâte d'abord."
    assert [t.name for t in trace.tools] == ["search_knowledge_base"]
    assert trace.tools[0].input == SEARCH.input
    assert trace.tools[0].details == HITS
    assert [r.stop_reason for r in trace.requests] == ["tool_use", "end_turn"]


def test_time_to_first_token_and_total_duration_come_from_the_clock():
    sink = MemorySink()

    list(recorder(sink).observe("q", answered_turn()))

    [trace] = sink.traces
    # clock: 0 at start, then one tick per event; the first text is event 4.
    assert trace.time_to_first_token_s == 4.0
    assert trace.duration_s > trace.time_to_first_token_s


def test_cost_sums_every_request():
    sink = MemorySink()

    list(recorder(sink).observe("q", answered_turn()))

    # 100 in * $5 + 50 out * $25 + 4000 cache reads * $0.5, per million tokens
    assert sink.traces[0].cost_usd == pytest.approx((500 + 1250 + 2000) / 1e6)


def test_turns_are_numbered_within_the_session():
    sink = MemorySink()
    rec = recorder(sink, new_id=iter(["a", "b"]).__next__)

    list(rec.observe("q1", answered_turn()))
    list(rec.observe("q2", answered_turn()))

    assert [(t.trace_id, t.turn) for t in sink.traces] == [("a", 1), ("b", 2)]


@pytest.mark.parametrize(
    "reply, outcome",
    [
        (AgentReply("no", refused=True), "refused"),
        (AgentReply("cut", truncated=True), "truncated"),
    ],
)
def test_refusals_and_truncations_are_recorded(reply, outcome):
    sink = MemorySink()

    list(recorder(sink).observe("q", iter([TurnFinished(reply)])))

    assert sink.traces[0].outcome == outcome


def test_a_failed_turn_is_still_traced_and_the_error_propagates():
    sink = MemorySink()

    def failing():
        yield TextDelta("x")
        raise AgentError("API down")

    with pytest.raises(AgentError):
        list(recorder(sink).observe("q", failing()))

    [trace] = sink.traces
    assert trace.outcome == "error"
    assert trace.error == "API down"


def test_a_turn_abandoned_by_the_client_is_traced_as_interrupted():
    sink = MemorySink()
    stream = recorder(sink).observe("q", answered_turn())

    next(stream)
    stream.close()  # e.g. the window closed mid-answer

    assert sink.traces[0].outcome == "interrupted"


def test_an_unavailable_character_is_recorded_as_none():
    sink = MemorySink()

    list(recorder(sink, character=lambda: None).observe("q", answered_turn()))

    assert sink.traces[0].character is None


def test_a_failing_sink_never_breaks_the_answer(caplog):
    class BrokenSink:
        def write(self, trace):
            raise OSError("disk full")

    events = list(recorder(BrokenSink()).observe("q", answered_turn()))

    assert isinstance(events[-1], TurnFinished)
    assert "disk full" in caplog.text


def test_trace_serializes_to_json():
    import json

    sink = MemorySink()
    list(recorder(sink).observe("q", answered_turn()))

    data = json.loads(json.dumps(sink.traces[0].to_json()))

    assert data["started_at"] == "2026-09-29T20:00:00+00:00"
    assert data["requests"][1]["usage"]["cache_read_input_tokens"] == 4000
    assert data["tools"][0]["details"]["results"][0]["distance"] == 0.31
