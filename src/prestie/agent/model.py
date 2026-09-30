"""The port between the agent loop and a language model provider.

The loop (`agent.py`) only needs one thing from a model: "given this system
prompt, these tools and this conversation, stream the answer and tell me how
it ended". `ChatModel` is that contract; each provider gets an adapter:

- `anthropic_model.py`: the Claude Messages API (server-side fallbacks,
  prompt caching);
- an OpenAI-compatible adapter for a local server (llama.cpp), next.

The conversation keeps the Anthropic shape throughout (content blocks with a
`type`: "text", "tool_use"; tool results as "tool_result" blocks): it is what
the history, the compaction and the traces are written against. An adapter
for another API translates at its boundary, both ways. Stop reasons use the
same vocabulary: "end_turn", "tool_use", "max_tokens", "refusal".
"""

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


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
        """`input_tokens` only counts uncached tokens; this is what the model read."""
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


class AgentError(Exception):
    """The model call failed; the message is safe to show to the user."""


@dataclass(frozen=True)
class TextDelta:
    """A piece of the answer, as generated."""

    text: str


@dataclass(frozen=True)
class FallbackRestart:
    """The model was declined mid-answer and a fallback model takes over: text
    streamed since the start of this request is superseded and should be
    cleared from the display."""


@dataclass(frozen=True)
class TextBlock:
    """An assistant text block, for adapters whose API has no block objects."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    """An assistant tool call, in the Anthropic block shape the history uses."""

    id: str
    name: str
    input: Mapping[str, Any]
    type: str = "tool_use"


@dataclass(frozen=True)
class ModelResponse:
    """How one request ended."""

    content: tuple[Any, ...]  # assistant blocks to append to the conversation
    text: str  # the answer shown to the player
    stop_reason: str  # "end_turn", "tool_use", "max_tokens" or "refusal"
    usage: Usage
    model: str  # the model that actually served the request


# What a model streams while it answers.
StreamEvent = TextDelta | FallbackRestart


class ChatModel(Protocol):
    def stream(
        self,
        *,
        system: str,
        tools: Sequence[Mapping[str, Any]],
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        allow_tools: bool,
    ) -> Generator[StreamEvent, None, ModelResponse]:
        """Stream one request: yield text as it comes, return how it ended.

        `allow_tools=False` asks for a final answer (tool rounds exhausted);
        the tool definitions are still sent, since they are part of the
        cached prefix. Failures raise AgentError.
        """
        ...
