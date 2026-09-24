from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from prestie.agent.agent import Agent, AgentError
from prestie.agent.tools import SEARCH_TOOL, ToolOutcome

MODEL = "claude-opus-5"
SYSTEM = "system prompt"


def usage(input_tokens=100, output_tokens=20, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def text_response(text: str, stop_reason: str = "end_turn", **usage_kwargs):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=usage(**usage_kwargs),
        model=MODEL,
    )


def tool_response(*calls: tuple[str, dict]):
    blocks = [SimpleNamespace(type="text", text="Je cherche.")] + [
        SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=tool_input)
        for i, (name, tool_input) in enumerate(calls)
    ]
    return SimpleNamespace(
        content=blocks, stop_reason="tool_use", usage=usage(), model=MODEL
    )


class FakeClient:
    """Stands in for anthropic.Anthropic: replays scripted responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeTool:
    def __init__(self, outcome: ToolOutcome | None = None, definition=None):
        self.definition = definition or SEARCH_TOOL
        self.inputs: list[dict] = []
        self.outcome = outcome or ToolOutcome("<search_results>...</search_results>")

    def run(self, tool_input):
        self.inputs.append(tool_input)
        return self.outcome


def make_agent(client, tool=None, **kwargs) -> Agent:
    tools = kwargs.pop("tools", None) or [tool or FakeTool()]
    return Agent(client, model=MODEL, system_prompt=SYSTEM, tools=tools, **kwargs)


def test_direct_answer_needs_a_single_request():
    client = FakeClient(text_response("Bonjour !"))

    reply = make_agent(client).ask("Salut")

    assert reply.text == "Bonjour !"
    assert reply.tool_calls == ()
    [request] = client.requests
    assert request["model"] == MODEL
    assert request["system"] == SYSTEM
    assert request["tools"][0]["name"] == "search_knowledge_base"
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["fallbacks"] == "default"
    assert request["messages"] == [{"role": "user", "content": "Salut"}]


def test_tool_call_is_executed_and_its_result_sent_back():
    tool = FakeTool(ToolOutcome("RESULTS"))
    seen = []
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "stat priority"})),
        text_response("Hâte d'abord."),
    )

    reply = make_agent(client, tool, on_tool_call=seen.append).ask("Quelles stats ?")

    assert reply.text == "Hâte d'abord."
    assert tool.inputs == [{"query": "stat priority"}]
    assert [call.input for call in reply.tool_calls] == [{"query": "stat priority"}]
    assert seen == list(reply.tool_calls)
    tool_results = client.requests[1]["messages"][-1]
    assert tool_results["role"] == "user"
    assert tool_results["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_0", "content": "RESULTS"}
    ]


def test_parallel_tool_calls_are_answered_in_one_message():
    client = FakeClient(
        tool_response(
            ("search_knowledge_base", {"query": "a"}),
            ("search_knowledge_base", {"query": "b"}),
        ),
        text_response("ok"),
    )

    make_agent(client).ask("q")

    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_0", "toolu_1"]


def test_tool_errors_are_flagged_to_the_model():
    client = FakeClient(
        tool_response(("search_knowledge_base", {})), text_response("désolé")
    )

    make_agent(client, FakeTool(ToolOutcome("bad input", is_error=True))).ask("q")

    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True


def test_unknown_tool_name_is_answered_with_an_error_result():
    client = FakeClient(tool_response(("delete_everything", {})), text_response("ok"))

    make_agent(client).ask("q")

    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    assert "Unknown tool" in result["content"]


def test_each_tool_call_is_routed_to_the_tool_with_that_name():
    search = FakeTool(ToolOutcome("PASSAGES"))
    state = FakeTool(
        ToolOutcome("STATE"),
        definition={"name": "get_character_state", "input_schema": {}},
    )
    client = FakeClient(
        tool_response(("get_character_state", {}), ("search_knowledge_base", {})),
        text_response("ok"),
    )

    make_agent(client, tools=[search, state]).ask("q")

    assert [t["name"] for t in client.requests[0]["tools"]] == [
        "search_knowledge_base",
        "get_character_state",
    ]
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["content"] for r in results] == ["STATE", "PASSAGES"]


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError, match="search_knowledge_base"):
        make_agent(FakeClient(), tools=[FakeTool(), FakeTool()])


def test_conversation_history_is_kept_between_questions():
    client = FakeClient(text_response("Réponse 1"), text_response("Réponse 2"))
    agent = make_agent(client)

    agent.ask("Question 1")
    agent.ask("Question 2")

    history = client.requests[1]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[-1]["content"] == "Question 2"


def test_tool_rounds_are_capped_then_the_model_must_answer():
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "1"})),
        tool_response(("search_knowledge_base", {"query": "2"})),
        text_response("final"),
    )

    reply = make_agent(client, max_tool_rounds=2).ask("q")

    assert reply.text == "final"
    assert "tool_choice" not in client.requests[0]
    assert client.requests[2]["tool_choice"] == {"type": "none"}


def test_refusal_is_reported_and_not_kept_in_history():
    client = FakeClient(text_response("", stop_reason="refusal"), text_response("ok"))
    agent = make_agent(client)

    reply = agent.ask("something refused")
    agent.ask("next")

    assert reply.refused
    assert client.requests[1]["messages"] == [{"role": "user", "content": "next"}]


def test_truncated_answer_is_flagged():
    client = FakeClient(text_response("début…", stop_reason="max_tokens"))

    assert make_agent(client).ask("q").truncated


def test_usage_is_summed_over_every_request_of_the_turn():
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "x"})),
        text_response("ok", input_tokens=300, output_tokens=50, cache_read=200),
    )

    reply = make_agent(client).ask("q")

    assert reply.usage.input_tokens == 400
    assert reply.usage.output_tokens == 70
    assert reply.usage.cache_read_input_tokens == 200


def test_api_errors_raise_agent_error_and_leave_history_untouched():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(
        anthropic.APIConnectionError(request=request), text_response("ok")
    )
    agent = make_agent(client)

    with pytest.raises(AgentError, match="Anthropic"):
        agent.ask("q1")
    agent.ask("q2")

    assert client.requests[1]["messages"] == [{"role": "user", "content": "q2"}]


def test_total_input_tokens_include_cached_tokens():
    from prestie.agent.agent import Usage

    usage = Usage(
        input_tokens=4,
        output_tokens=700,
        cache_read_input_tokens=1228,
        cache_creation_input_tokens=5781,
    )

    assert usage.total_input_tokens == 4 + 1228 + 5781


def test_reply_exposes_the_turn_messages_and_served_models():
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "x"})),
        text_response("final"),
    )

    reply = make_agent(client, FakeTool(ToolOutcome("PASSAGES"))).ask("q")

    assert reply.models == (MODEL, MODEL)
    assert [m["role"] for m in reply.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert reply.messages[2]["content"][0]["content"] == "PASSAGES"
