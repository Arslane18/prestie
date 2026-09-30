"""The agent loop runs on any ChatModel, not only on the Anthropic SDK.

The fake model below returns plain blocks (no SDK object): if the loop still
works, the provider-specific code lives in the adapters, where it belongs.
"""

from types import SimpleNamespace

from prestie.agent.agent import (
    Agent,
    RequestFinished,
    TextDelta,
    ToolCallFinished,
    Usage,
)
from prestie.agent.model import ModelResponse
from prestie.agent.tools import ToolOutcome
from tests.agent.test_agent import FakeTool

LOCAL_MODEL = "qwen3.5-8b-q4_k_m"


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_input, block_id="call_0"):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(*blocks, stop_reason="end_turn", text=None):
    answer = text if text is not None else "".join(
        b.text for b in blocks if b.type == "text"
    )
    return ModelResponse(
        content=tuple(blocks),
        text=answer,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=100, output_tokens=20),
        model=LOCAL_MODEL,
    )


class ScriptedModel:
    """Replays responses, streaming their text, and records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def stream(self, **request):
        self.requests.append(request)
        reply = self.responses.pop(0)
        for block in reply.content:
            if block.type == "text":
                yield TextDelta(block.text)
        return reply


def agent_on(model, tool=None, **kwargs):
    return Agent(model, system_prompt="Tu es Prestie.", tools=[tool or FakeTool()], **kwargs)


def test_a_tool_round_then_an_answer_on_a_plain_model():
    model = ScriptedModel(
        response(tool_block("search_knowledge_base", {"query": "stats"}), stop_reason="tool_use"),
        response(text_block("Hâte d'abord.")),
    )
    tool = FakeTool(ToolOutcome("PASSAGES"))

    events = list(agent_on(model, tool).ask_stream("Quelle stat ?"))

    reply = events[-1].reply
    assert reply.text == "Hâte d'abord."
    assert tool.inputs == [{"query": "stats"}]
    [first, second] = model.requests
    assert first["system"] == "Tu es Prestie."
    assert first["tools"][0]["name"] == "search_knowledge_base"
    assert first["allow_tools"] is True
    [result] = second["messages"][-1]["content"]
    assert (result["tool_use_id"], result["content"]) == ("call_0", "PASSAGES")
    assert [e.model for e in events if isinstance(e, RequestFinished)] == [
        LOCAL_MODEL,
        LOCAL_MODEL,
    ]
    assert reply.models == (LOCAL_MODEL, LOCAL_MODEL)
    assert any(isinstance(e, ToolCallFinished) for e in events)


def test_the_answer_text_comes_from_the_model_response():
    # An adapter decides what the answer is (e.g. text after a fallback marker).
    model = ScriptedModel(response(text_block("brouillon"), text="réponse"))

    assert agent_on(model).ask("?").text == "réponse"


def test_a_refusal_ends_the_turn():
    model = ScriptedModel(response(stop_reason="refusal"))

    reply = agent_on(model).ask("?")

    assert reply.refused


def test_tools_are_withheld_after_the_last_allowed_round():
    model = ScriptedModel(
        response(tool_block("search_knowledge_base", {"query": "a"}), stop_reason="tool_use"),
        response(text_block("fini")),
    )

    agent_on(model, max_tool_rounds=1).ask("?")

    assert [r["allow_tools"] for r in model.requests] == [True, False]


def test_usage_adds_up_over_the_turn():
    model = ScriptedModel(
        response(tool_block("search_knowledge_base", {"query": "a"}), stop_reason="tool_use"),
        response(text_block("fini")),
    )

    reply = agent_on(model).ask("?")

    assert reply.usage == Usage(input_tokens=200, output_tokens=40)
