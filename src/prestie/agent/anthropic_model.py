"""`ChatModel` adapter for the Claude Messages API.

Everything specific to Claude lives here: the SDK's streaming events, prompt
caching (`cache_control`), server-side fallbacks when a safety classifier
declines (`fallbacks: "default"`, which may insert `fallback` markers in the
content), and the SDK's error types.
"""

from collections.abc import Generator, Mapping, Sequence
from typing import Any

import anthropic

from prestie.agent.model import (
    AgentError,
    FallbackRestart,
    ModelResponse,
    StreamEvent,
    TextDelta,
    Usage,
)

# Server-side fallback: if Claude's safety classifiers decline a request, the API
# retries it on the recommended fallback model instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicChatModel:
    def __init__(self, client: Any, model: str):
        self._client = client
        self.model = model

    def stream(
        self,
        *,
        system: str,
        tools: Sequence[Mapping[str, Any]],
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        allow_tools: bool,
    ) -> Generator[StreamEvent, None, ModelResponse]:
        extra = {} if allow_tools else {"tool_choice": {"type": "none"}}
        try:
            with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                # Same tools in the same order every time: they are part of the
                # cached prefix.
                tools=[dict(tool) for tool in tools],
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
                message = stream.get_final_message()
        except anthropic.APIError as exc:
            raise _agent_error(exc) from exc
        return ModelResponse(
            content=tuple(_echo_content(message.content)),
            text=_text(message.content),
            stop_reason=str(message.stop_reason),
            usage=Usage.from_api(message.usage),
            model=str(getattr(message, "model", self.model)),
        )


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
