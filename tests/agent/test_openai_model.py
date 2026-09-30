"""The OpenAI-compatible adapter (llama-server): translation and streaming.

The server is replaced by an httpx MockTransport that replays Server-Sent
Events, so these tests pin the wire format both ways without a GPU.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from prestie.agent.agent import Agent, AgentError, TextDelta, Usage
from prestie.agent.model import ModelResponse
from prestie.agent.openai_model import OpenAICompatibleChatModel
from prestie.agent.tools import SEARCH_TOOL, ToolOutcome
from tests.agent.test_agent import FakeTool

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "qwen3.5-9b"


def chunk(delta=None, finish=None, usage=None, model=MODEL):
    data = {"model": model, "choices": []}
    if delta is not None or finish is not None:
        data["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        data["usage"] = usage
    return data


def sse(*chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"}
    )


USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 20,
    "prompt_tokens_details": {"cached_tokens": 800},
}


class FakeServer:
    """Answers each request with the next scripted response, keeping the bodies."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.responses.pop(0)

    @property
    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def adapter(server, **options):
    http = httpx.Client(transport=httpx.MockTransport(server))
    return OpenAICompatibleChatModel(BASE_URL, MODEL, http=http, **options)


def run(model, messages=(), allow_tools=True):
    """Drain one request: (events, response)."""
    stream = model.stream(
        system="Tu es Prestie.",
        tools=[SEARCH_TOOL],
        messages=list(messages) or [{"role": "user", "content": "Quelle stat ?"}],
        max_tokens=4096,
        allow_tools=allow_tools,
    )
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as done:
            return events, done.value


# --- the answer ------------------------------------------------------------------


def test_text_is_streamed_and_the_response_describes_the_end():
    server = FakeServer(
        sse(
            chunk({"role": "assistant", "content": "Hâte"}),
            chunk({"content": " d'abord."}),
            chunk(finish="stop"),
            chunk(usage=USAGE),
        )
    )

    events, response = run(adapter(server))

    assert events == [TextDelta("Hâte"), TextDelta(" d'abord.")]
    assert isinstance(response, ModelResponse)
    assert response.text == "Hâte d'abord."
    assert response.stop_reason == "end_turn"
    assert response.model == MODEL
    [block] = response.content
    assert (block.type, block.text) == ("text", "Hâte d'abord.")


def test_cached_prompt_tokens_are_reported_like_prompt_caching():
    server = FakeServer(sse(chunk({"content": "ok"}, finish="stop"), chunk(usage=USAGE)))

    _, response = run(adapter(server))

    # Same meaning as Claude's usage: input_tokens counts the uncached part.
    assert response.usage == Usage(
        input_tokens=200, output_tokens=20, cache_read_input_tokens=800
    )


def test_tool_call_fragments_are_assembled():
    server = FakeServer(
        sse(
            chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                   "function": {"name": "search_knowledge_base", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": "stat'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 's", "spec": "shadow-priest"}'}}]}),
            chunk(finish="tool_calls"),
        )
    )

    events, response = run(adapter(server))

    assert events == []
    assert response.stop_reason == "tool_use"
    [call] = response.content
    assert (call.type, call.id, call.name) == ("tool_use", "call_a", "search_knowledge_base")
    assert call.input == {"query": "stats", "spec": "shadow-priest"}


def test_parallel_tool_calls_keep_their_order():
    server = FakeServer(
        sse(
            chunk({"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "get_character_state", "arguments": "{}"}},
                {"index": 1, "id": "b", "function": {"name": "get_equipment", "arguments": "{}"}},
            ]}),
            chunk(finish="tool_calls"),
        )
    )

    _, response = run(adapter(server))

    assert [block.id for block in response.content] == ["a", "b"]


def test_unparsable_tool_arguments_reach_the_tool_as_an_invalid_input():
    server = FakeServer(
        sse(
            chunk({"tool_calls": [{"index": 0, "id": "a",
                                   "function": {"name": "search_knowledge_base", "arguments": "{query: stats"}}]}),
            chunk(finish="tool_calls"),
        )
    )

    _, response = run(adapter(server))

    # The tool rejects it (no 'query'), the model reads the error and retries.
    [call] = response.content
    assert call.input == {"_invalid_arguments": "{query: stats"}


@pytest.mark.parametrize(
    ("finish", "stop_reason"),
    [("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "refusal")],
)
def test_finish_reasons_map_to_the_agent_vocabulary(finish, stop_reason):
    server = FakeServer(sse(chunk({"content": "x"}, finish=finish)))

    _, response = run(adapter(server))

    assert response.stop_reason == stop_reason


def test_reasoning_is_not_part_of_the_answer():
    server = FakeServer(
        sse(
            chunk({"reasoning_content": "The player asks about stats..."}),
            chunk({"content": "Hâte."}, finish="stop"),
        )
    )

    events, response = run(adapter(server))

    assert events == [TextDelta("Hâte.")]
    assert response.text == "Hâte."


# --- the request -------------------------------------------------------------------


def test_the_request_is_translated_to_the_openai_format():
    server = FakeServer(sse(chunk({"content": "ok"}, finish="stop")))
    history = [
        {"role": "user", "content": "Quelle stat ?"},
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text="Je cherche."),
                SimpleNamespace(type="tool_use", id="t1", name="search_knowledge_base",
                                input={"query": "stats"}),
                SimpleNamespace(type="tool_use", id="t2", name="get_equipment", input={}),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "PASSAGES"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "absent", "is_error": True},
            ],
        },
    ]

    run(adapter(server), history, allow_tools=False)

    [body] = server.bodies
    assert body["model"] == MODEL
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 4096
    assert body["tool_choice"] == "none"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": SEARCH_TOOL["name"],
                "description": SEARCH_TOOL["description"],
                "parameters": SEARCH_TOOL["input_schema"],
            },
        }
    ]
    assert body["messages"] == [
        {"role": "system", "content": "Tu es Prestie."},
        {"role": "user", "content": "Quelle stat ?"},
        {
            "role": "assistant",
            "content": "Je cherche.",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "search_knowledge_base", "arguments": '{"query": "stats"}'}},
                {"id": "t2", "type": "function",
                 "function": {"name": "get_equipment", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "PASSAGES"},
        {"role": "tool", "tool_call_id": "t2", "content": "Error: absent"},
    ]


def test_tools_are_offered_while_rounds_remain():
    server = FakeServer(sse(chunk({"content": "ok"}, finish="stop")))

    run(adapter(server))

    assert "tool_choice" not in server.bodies[0]


def test_thinking_and_the_api_key_are_options():
    server = FakeServer(sse(chunk({"content": "ok"}, finish="stop")))

    run(adapter(server, thinking=True, api_key="local-secret"))

    assert server.bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert server.requests[0].headers["authorization"] == "Bearer local-secret"
    assert str(server.requests[0].url) == f"{BASE_URL}/chat/completions"


# --- failures ---------------------------------------------------------------------


def test_an_http_error_becomes_an_agent_error():
    server = FakeServer(httpx.Response(400, json={"error": {"message": "context too long"}}))

    with pytest.raises(AgentError, match="400.*context too long"):
        run(adapter(server))


def test_an_unreachable_server_becomes_an_agent_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    model = OpenAICompatibleChatModel(
        BASE_URL, MODEL, http=httpx.Client(transport=httpx.MockTransport(refuse))
    )

    with pytest.raises(AgentError, match="llama-server"):
        run(model)


def test_an_error_event_in_the_stream_becomes_an_agent_error():
    server = FakeServer(
        httpx.Response(
            200,
            text='data: {"error": {"message": "slot unavailable"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )

    with pytest.raises(AgentError, match="slot unavailable"):
        run(adapter(server))


# --- in the agent loop ----------------------------------------------------------


def test_the_agent_runs_a_tool_round_on_the_local_model():
    server = FakeServer(
        sse(
            chunk({"tool_calls": [{"index": 0, "id": "call_a",
                                   "function": {"name": "search_knowledge_base",
                                                "arguments": '{"query": "stats"}'}}]}),
            chunk(finish="tool_calls"),
        ),
        sse(chunk({"content": "Hâte d'abord."}, finish="stop")),
    )
    tool = FakeTool(ToolOutcome("PASSAGES"))
    agent = Agent(adapter(server), system_prompt="Tu es Prestie.", tools=[tool])

    reply = agent.ask("Quelle stat ?")

    assert reply.text == "Hâte d'abord."
    assert tool.inputs == [{"query": "stats"}]
    assert server.bodies[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_a",
        "content": "PASSAGES",
    }
    assert reply.models == (MODEL, MODEL)
