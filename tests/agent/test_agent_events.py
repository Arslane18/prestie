"""Observability events: what each API request and each tool call cost."""

from itertools import count

from prestie.agent.agent import (
    RequestFinished,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from prestie.agent.tools import ToolOutcome
from tests.agent.test_agent import (
    MODEL,
    FakeClient,
    FakeTool,
    make_agent,
    text_response,
    tool_response,
)


def ticking_clock(step=1.0):
    """Each call advances by `step` seconds: durations become predictable."""
    ticks = count()
    return lambda: next(ticks) * step


def events_of(agent, question="q"):
    return list(agent.ask_stream(question))


def test_each_api_request_reports_its_model_usage_and_duration():
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "stats"})),
        text_response("ok", cache_read=900, cache_write=50),
    )

    events = events_of(make_agent(client, clock=ticking_clock()))

    requests = [e for e in events if isinstance(e, RequestFinished)]
    assert [r.stop_reason for r in requests] == ["tool_use", "end_turn"]
    assert all(r.model == MODEL for r in requests)
    assert requests[1].usage.cache_read_input_tokens == 900
    assert requests[1].usage.cache_creation_input_tokens == 50
    assert all(r.duration_s > 0 for r in requests)


def test_tool_calls_report_duration_error_flag_and_details():
    tool = FakeTool(
        ToolOutcome("PASSAGES", details={"results": [{"key": "p#a", "distance": 0.3}]})
    )
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "stats"})),
        text_response("ok"),
    )

    events = events_of(make_agent(client, tool, clock=ticking_clock()))

    [finished] = [e for e in events if isinstance(e, ToolCallFinished)]
    assert finished.call.input == {"query": "stats"}
    assert finished.is_error is False
    assert finished.duration_s > 0
    assert finished.details == {"results": [{"key": "p#a", "distance": 0.3}]}


def test_event_order_brackets_each_tool_call():
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "a"})), text_response("ok")
    )

    kinds = [type(e) for e in events_of(make_agent(client))]

    started = kinds.index(ToolCallStarted)
    assert kinds[started + 1] is ToolCallFinished
    assert kinds[-1] is TurnFinished
    assert kinds[-2] is RequestFinished  # the last request ends just before the turn


def test_failed_tools_are_flagged():
    client = FakeClient(
        tool_response(("search_knowledge_base", {})), text_response("désolé")
    )

    tool = FakeTool(ToolOutcome("bad input", is_error=True))

    events = events_of(make_agent(client, tool))

    [finished] = [e for e in events if isinstance(e, ToolCallFinished)]
    assert finished.is_error is True


def test_text_still_streams_before_the_request_finishes():
    client = FakeClient(text_response("Bonjour à toi"))

    kinds = [type(e) for e in events_of(make_agent(client))]

    assert kinds.index(TextDelta) < kinds.index(RequestFinished)


# --- lot A: a tool crash must not kill the turn -----------------------------------


class CrashingTool(FakeTool):
    def run(self, tool_input):
        raise RuntimeError("chroma collection does not exist")


def test_an_unexpected_tool_crash_becomes_an_error_result(caplog):
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "stats"})),
        text_response("Recherche indisponible."),
    )

    reply = make_agent(client, CrashingTool()).ask("q")

    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    assert "RuntimeError" in result["content"]
    assert reply.text == "Recherche indisponible."  # the turn went on
    assert "chroma collection does not exist" in caplog.text  # and it was logged
