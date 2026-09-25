"""Hand-written agentic loop over the Claude Messages API (no agent framework).

One question = one "turn", which may take several API requests:

    user question ──► Claude ──stop_reason="tool_use"──► we run the tool(s)
         ▲                                                     │
         └──────────── tool_result (passages, state) ◄─────────┘
                 ... until stop_reason="end_turn" (final answer)

The API is stateless: every request resends the whole conversation (system
prompt + tool definitions + history). Prompt caching makes that cheap: the
unchanged prefix is read from cache at ~10% of the input price.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from prestie.agent.tools import ToolOutcome

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


class AgentError(Exception):
    """The Claude API call failed; the message is safe to show to the user."""


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
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._tools = _index_by_name(tools)
        self._on_tool_call = on_tool_call or (lambda call: None)
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._history: tuple[dict[str, Any], ...] = ()

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def ask(self, question: str) -> AgentReply:
        """Run one turn. History is only updated when the turn completes cleanly."""
        turn_start = len(self._history)
        messages = (*self._history, {"role": "user", "content": question})
        tool_calls: tuple[ToolCall, ...] = ()
        usage = Usage()
        models: tuple[str, ...] = ()
        tool_rounds = 0

        while True:
            response = self._request(
                messages, allow_tools=tool_rounds < self._max_tool_rounds
            )
            usage = usage + Usage.from_api(response.usage)
            models = (*models, str(getattr(response, "model", self._model)))
            if response.stop_reason == "refusal":
                return AgentReply(
                    REFUSAL_MESSAGE,
                    tool_calls,
                    usage,
                    refused=True,
                    messages=messages[turn_start:],
                    models=models,
                )

            messages = (*messages, {"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break

            tool_rounds += 1
            tool_uses = [
                block for block in response.content if block.type == "tool_use"
            ]
            calls = tuple(ToolCall(block.name, block.input) for block in tool_uses)
            tool_calls = (*tool_calls, *calls)
            # All results of one round go back in a single user message.
            results = [
                self._run_tool(block, call) for block, call in zip(tool_uses, calls)
            ]
            messages = (*messages, {"role": "user", "content": results})

        truncated = response.stop_reason == "max_tokens"
        # A truncated tool_use without its tool_result would make the next request invalid.
        if not (truncated and _has_tool_use(response.content)):
            self._history = messages
        return AgentReply(
            _text(response.content),
            tool_calls,
            usage,
            truncated=truncated,
            messages=messages[turn_start:],
            models=models,
        )

    def _request(self, messages: Sequence[dict[str, Any]], *, allow_tools: bool) -> Any:
        extra = {} if allow_tools else {"tool_choice": {"type": "none"}}
        try:
            return self._client.beta.messages.create(
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
            )
        except anthropic.AuthenticationError as exc:
            raise AgentError(
                "Clé API Anthropic invalide ou absente (ANTHROPIC_API_KEY)."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise AgentError(
                "Limite de débit de l'API Anthropic atteinte, réessaie dans un instant."
            ) from exc
        except anthropic.APIStatusError as exc:
            raise AgentError(
                f"Erreur de l'API Anthropic ({exc.status_code}) : {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError(
                "Impossible de joindre l'API Anthropic (réseau ou délai dépassé)."
            ) from exc

    def _run_tool(self, block: Any, call: ToolCall) -> dict[str, Any]:
        self._on_tool_call(call)
        tool = self._tools.get(call.name)
        if tool is None:
            outcome = ToolOutcome(f"Unknown tool: {call.name}", is_error=True)
        else:
            outcome = tool.run(call.input)
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


def _text(content: Sequence[Any]) -> str:
    return "\n".join(block.text for block in content if block.type == "text").strip()


def _has_tool_use(content: Sequence[Any]) -> bool:
    return any(block.type == "tool_use" for block in content)
