"""Hand-written agentic loop over the Claude Messages API (no agent framework).

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
"""

import logging
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from prestie.agent.attribution import missing_sources_section
from prestie.agent.context import compact_history
from prestie.agent.tools import SEARCH_TOOL_NAME, ToolOutcome

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16000
# A focused Q&A rarely needs more than 2-3 searches; the cap bounds cost/latency.
DEFAULT_MAX_TOOL_ROUNDS = 5
# Server-side fallback: if Claude's safety classifiers decline a request, the API
# retries it on the recommended fallback model instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_MESSAGE = "Désolé, je ne peux pas répondre à cette demande."


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: Mapping[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> "Usage":
        return cls(
            **{name: getattr(usage, name, 0) or 0 for name in cls.__dataclass_fields__}
        )

    @property
    def total_input_tokens(self) -> int:
        """`input_tokens` only counts uncached tokens; this is what Claude read."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class AgentReply:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()
    refused: bool = False
    truncated: bool = False
    # This turn's messages (question, assistant blocks, tool results), for traces.
    messages: tuple[dict[str, Any], ...] = ()
    # Model that actually served each request (may differ after a fallback).
    models: tuple[str, ...] = ()
    # The model forgot the Icy Veins sources section and the agent added it
    # (`text` includes it; the history keeps what the model wrote).
    sources_added: bool = False


class AgentError(Exception):
    """The Claude API call failed; the message is safe to show to the user."""


@dataclass(frozen=True)
class TextDelta:
    """A piece of the answer, as generated."""

    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    call: ToolCall


@dataclass(frozen=True)
class FallbackRestart:
    """The model was declined mid-answer and a fallback model takes over: text
    streamed since the start of this request is superseded and should be
    cleared from the display."""


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
        client: Any,
        model: str,
        system_prompt: str,
        tools: Sequence[Tool],
        *,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client = client
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
            response = yield from self._stream_request(
                (*history, *turn), allow_tools=tool_rounds < self._max_tool_rounds
            )
            request_usage = Usage.from_api(response.usage)
            usage = usage + request_usage
            model = str(getattr(response, "model", self._model))
            models = (*models, model)
            yield RequestFinished(
                model,
                request_usage,
                str(response.stop_reason),
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

            content = _echo_content(response.content)
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
        text = _text(response.content)
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

    def _stream_request(
        self, messages: Sequence[dict[str, Any]], *, allow_tools: bool
    ) -> Generator[AgentEvent, None, Any]:
        """Stream one API request; yields text deltas, returns the final message."""
        extra = {} if allow_tools else {"tool_choice": {"type": "none"}}
        try:
            with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                # Same tools in the same order every time: they are part of the
                # cached prefix.
                tools=[dict(tool.definition) for tool in self._tools.values()],
                messages=list(messages),
                # Automatic caching: caches the longest reusable prefix, so each
                # request of the loop re-reads system + tools + history from cache.
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
                **extra,
            ) as stream:
                for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                    elif _is_fallback_start(event):
                        yield FallbackRestart()
                return stream.get_final_message()
        except anthropic.APIError as exc:
            raise _agent_error(exc) from exc

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


def _agent_error(exc: anthropic.APIError) -> AgentError:
    if isinstance(exc, anthropic.AuthenticationError):
        return AgentError("Clé API Anthropic invalide ou absente (ANTHROPIC_API_KEY).")
    if isinstance(exc, anthropic.RateLimitError):
        return AgentError(
            "Limite de débit de l'API Anthropic atteinte, réessaie dans un instant."
        )
    if isinstance(exc, anthropic.APIStatusError):
        return AgentError(
            f"Erreur de l'API Anthropic ({exc.status_code}) : {exc.message}"
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return AgentError(
            "Impossible de joindre l'API Anthropic (réseau ou délai dépassé)."
        )
    return AgentError(f"Erreur de l'API Anthropic : {exc}")


def _is_fallback_start(event: Any) -> bool:
    block = getattr(event, "content_block", None)
    return event.type == "content_block_start" and block.type == "fallback"


def _echo_content(content: Sequence[Any]) -> list[Any]:
    """The assistant blocks to keep, per the server-side fallback rules.

    After a mid-stream fallback, only the declined partial's text blocks are
    kept before the last `fallback` marker (tool calls and thinking from the
    declined model must not be sent back); everything after it is kept.
    """
    markers = [i for i, block in enumerate(content) if block.type == "fallback"]
    if not markers:
        return list(content)
    last = markers[-1]
    return [block for block in content[:last] if block.type == "text"] + list(
        content[last + 1 :]
    )


def _text(content: Sequence[Any]) -> str:
    """The answer: text written after the last fallback marker, if any."""
    markers = [i for i, block in enumerate(content) if block.type == "fallback"]
    answer = content[markers[-1] + 1 :] if markers else content
    return "\n".join(block.text for block in answer if block.type == "text").strip()


def _has_tool_use(content: Sequence[Any]) -> bool:
    return any(block.type == "tool_use" for block in content)
