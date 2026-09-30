"""Hand-written agentic loop over a chat model with tools (no agent framework).

One question = one "turn", which may take several API requests:

    user question ──► Claude ──stop_reason="tool_use"──► we run the tool(s)
         ▲                                                     │
         └──────────── tool_result (passages, state) ◄─────────┘
                 ... until stop_reason="end_turn" (final answer)

The API is stateless: every request resends the whole conversation (system
prompt + tool definitions + history). Prompt caching makes that cheap: the
unchanged prefix is read from cache at ~10% of the input price. Earlier turns
are sent compacted (see `context.py`): their passages and character state are
replaced by short stubs, so the history grows by little more than the answers.

Every request is streamed: `ask_stream` yields the answer's text as Claude
writes it and announces each tool call, so a UI can show progress instead of a
spinner. `ask` runs the same loop and only keeps the final reply.

The loop talks to a `ChatModel` (`model.py`), not to a provider's SDK: the
Claude specifics (caching, fallbacks, SDK errors) live in `anthropic_model.py`,
so the same loop can run on a local model.
"""

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from prestie.agent.attribution import missing_sources_section
from prestie.agent.context import compact_history
from prestie.agent.model import (  # re-exported: callers import them from here
    AgentError,
    ChatModel,
    FallbackRestart,
    TextDelta,
    Usage,
)
from prestie.agent.tools import SEARCH_TOOL_NAME, ToolOutcome

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "AgentReply",
    "FallbackRestart",
    "RequestFinished",
    "TextDelta",
    "ToolCall",
    "ToolCallFinished",
    "ToolCallStarted",
    "TurnFinished",
    "Usage",
]

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16000
# A focused Q&A rarely needs more than 2-3 searches; the cap bounds cost/latency.
DEFAULT_MAX_TOOL_ROUNDS = 5
REFUSAL_MESSAGE = "Désolé, je ne peux pas répondre à cette demande."


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: Mapping[str, Any]


@dataclass(frozen=True)
class AgentReply:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    refused: bool = False
    truncated: bool = False
    # This turn's messages (question, assistant blocks, tool results), for traces.
    messages: tuple[dict[str, Any], ...] = ()
    # Model that actually served each request (may differ after a fallback).
    models: tuple[str, ...] = ()
    # The model forgot the Icy Veins sources section and the agent added it
    # (`text` includes it; the history keeps what the model wrote).
    sources_added: bool = False


@dataclass(frozen=True)
class ToolCallStarted:
    call: ToolCall


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool returned (for traces; a UI can ignore it)."""

    call: ToolCall
    is_error: bool
    duration_s: float
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestFinished:
    """One API request of the turn ended (for traces; a UI can ignore it)."""

    model: str
    usage: Usage
    stop_reason: str
    duration_s: float


@dataclass(frozen=True)
class TurnFinished:
    reply: AgentReply


AgentEvent = (
    TextDelta
    | ToolCallStarted
    | ToolCallFinished
    | RequestFinished
    | FallbackRestart
    | TurnFinished
)


class Tool(Protocol):
    """A tool the agent can offer Claude: its definition and its executor."""

    @property
    def definition(self) -> Mapping[str, Any]: ...  # name, description, input_schema

    def run(self, tool_input: Mapping[str, Any]) -> ToolOutcome: ...


class Agent:
    def __init__(
        self,
        model: ChatModel,
        system_prompt: str,
        tools: Sequence[Tool],
        *,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._model = model
        self._system_prompt = system_prompt
        self._tools = _index_by_name(tools)
        self._on_tool_call = on_tool_call or (lambda call: None)
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._clock = clock
        self._history: tuple[dict[str, Any], ...] = ()

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def ask(self, question: str) -> AgentReply:
        """Run one turn and return the final reply (the streamed events are dropped)."""
        for event in self.ask_stream(question):
            if isinstance(event, TurnFinished):
                return event.reply
        raise AssertionError("ask_stream ended without TurnFinished")  # unreachable

    def ask_stream(self, question: str) -> Iterator[AgentEvent]:
        """Run one turn, yielding text as it is generated and each tool call.

        History is only updated when the turn completes cleanly. The last event
        is always TurnFinished; API failures raise AgentError mid-iteration.
        """
        # The full history is kept; earlier turns are compacted when sent.
        history = compact_history(self._history)
        turn: tuple[dict[str, Any], ...] = ({"role": "user", "content": question},)
        tool_calls: tuple[ToolCall, ...] = ()
        usage = Usage()
        models: tuple[str, ...] = ()
        tool_rounds = 0
        searches: tuple[Mapping[str, Any], ...] = ()  # details, for attribution

        while True:
            started = self._clock()
            response = yield from self._model.stream(
                system=self._system_prompt,
                tools=[tool.definition for tool in self._tools.values()],
                messages=[*history, *turn],
                max_tokens=self._max_tokens,
                allow_tools=tool_rounds < self._max_tool_rounds,
            )
            usage = usage + response.usage
            models = (*models, response.model)
            yield RequestFinished(
                response.model,
                response.usage,
                response.stop_reason,
                self._clock() - started,
            )
            if response.stop_reason == "refusal":
                yield TurnFinished(
                    AgentReply(
                        REFUSAL_MESSAGE,
                        tool_calls,
                        usage,
                        refused=True,
                        messages=turn,
                        models=models,
                    )
                )
                return

            content = list(response.content)
            turn = (*turn, {"role": "assistant", "content": content})
            if response.stop_reason != "tool_use":
                break

            tool_rounds += 1
            tool_uses = [block for block in content if block.type == "tool_use"]
            calls = tuple(ToolCall(block.name, block.input) for block in tool_uses)
            tool_calls = (*tool_calls, *calls)
            results = []
            for block, call in zip(tool_uses, calls):
                yield ToolCallStarted(call)
                started = self._clock()
                outcome = self._run_tool(call)
                yield ToolCallFinished(
                    call, outcome.is_error, self._clock() - started, outcome.details
                )
                if call.name == SEARCH_TOOL_NAME and not outcome.is_error:
                    searches = (*searches, outcome.details)
                results.append(_tool_result(block, outcome))
            # All results of one round go back in a single user message.
            turn = (*turn, {"role": "user", "content": results})

        truncated = response.stop_reason == "max_tokens"
        # A truncated tool_use without its tool_result would make the next request invalid.
        if not (truncated and _has_tool_use(content)):
            self._history = (*self._history, *turn)
        text = response.text
        # A cut-off answer is not completed: it is shown as interrupted.
        sources = None if truncated else missing_sources_section(text, searches)
        if sources:
            yield TextDelta(sources)
        yield TurnFinished(
            AgentReply(
                text + (sources or ""),
                tool_calls,
                usage,
                truncated=truncated,
                messages=turn,
                models=models,
                sources_added=bool(sources),
            )
        )

    def _run_tool(self, call: ToolCall) -> ToolOutcome:
        self._on_tool_call(call)
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolOutcome(f"Unknown tool: {call.name}", is_error=True)
        try:
            return tool.run(call.input)
        except Exception as exc:  # logged, and the turn goes on
            # Tools report their expected failures themselves; anything else
            # (a stale index, a full disk) must not end the conversation.
            logger.exception("tool %s failed", call.name)
            return ToolOutcome(
                f"Tool failed ({type(exc).__name__}): it is unavailable right now.",
                is_error=True,
            )


def _tool_result(block: Any, outcome: ToolOutcome) -> dict[str, Any]:
    result = {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": outcome.content,
    }
    return {**result, "is_error": True} if outcome.is_error else result


def _index_by_name(tools: Sequence[Tool]) -> dict[str, Tool]:
    names = [tool.definition["name"] for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}")
    return dict(zip(names, tools))


def _has_tool_use(content: Sequence[Any]) -> bool:
    return any(block.type == "tool_use" for block in content)
