"""Hand-written agentic loop over the Claude Messages API (no agent framework).

One question = one "turn", which may take several API requests:

    user question ──► Claude ──stop_reason="tool_use"──► we run the search
         ▲                                                     │
         └──────────── tool_result (passages) ◄────────────────┘
                 ... until stop_reason="end_turn" (final answer)

The API is stateless: every request resends the whole conversation (system
prompt + tool definitions + history). Prompt caching makes that cheap: the
unchanged prefix is read from cache at ~10% of the input price.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from prestie.agent.tools import SEARCH_TOOL, SEARCH_TOOL_NAME, ToolOutcome

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


class AgentError(Exception):
    """The Claude API call failed; the message is safe to show to the user."""


class RunsTool(Protocol):
    def run(self, tool_input: Mapping[str, Any]) -> ToolOutcome: ...


class Agent:
    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        tool: RunsTool,
        *,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._tool = tool
        self._on_tool_call = on_tool_call or (lambda call: None)
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._history: tuple[dict[str, Any], ...] = ()

    def ask(self, question: str) -> AgentReply:
        """Run one turn. History is only updated when the turn completes cleanly."""
        messages = (*self._history, {"role": "user", "content": question})
        tool_calls: tuple[ToolCall, ...] = ()
        usage = Usage()
        tool_rounds = 0

        while True:
            response = self._request(
                messages, allow_tools=tool_rounds < self._max_tool_rounds
            )
            usage = usage + Usage.from_api(response.usage)
            if response.stop_reason == "refusal":
                return AgentReply(REFUSAL_MESSAGE, tool_calls, usage, refused=True)

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
            _text(response.content), tool_calls, usage, truncated=truncated
        )

    def _request(self, messages: Sequence[dict[str, Any]], *, allow_tools: bool) -> Any:
        extra = {} if allow_tools else {"tool_choice": {"type": "none"}}
        try:
            return self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                tools=[SEARCH_TOOL],
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
        if call.name != SEARCH_TOOL_NAME:
            outcome = ToolOutcome(f"Unknown tool: {call.name}", is_error=True)
        else:
            outcome = self._tool.run(call.input)
        result = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": outcome.content,
        }
        return {**result, "is_error": True} if outcome.is_error else result


def _text(content: Sequence[Any]) -> str:
    return "\n".join(block.text for block in content if block.type == "text").strip()


def _has_tool_use(content: Sequence[Any]) -> bool:
    return any(block.type == "tool_use" for block in content)
