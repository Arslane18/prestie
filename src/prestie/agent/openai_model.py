"""`ChatModel` adapter for OpenAI-compatible servers, written for llama-server.

llama.cpp's server speaks the OpenAI Chat Completions format: `POST
/v1/chat/completions` with `stream: true` returns Server-Sent Events, one
`data: {json}` line per chunk, ended by `data: [DONE]`. This adapter
translates at the boundary, both ways, so the rest of Prestie keeps its
Anthropic-shaped history:

    Anthropic history                    OpenAI messages
    system prompt (separate)        ->   {"role": "system"}
    assistant text + tool_use       ->   {"role": "assistant", "content",
                                          "tool_calls": [{id, function:
                                          {name, arguments: JSON string}}]}
    user tool_result blocks         ->   one {"role": "tool", "tool_call_id"}
                                          per result ("Error: " if is_error)
    tools {name, input_schema}      ->   {"type": "function", "function":
                                          {name, parameters}}

In the stream, text arrives in `delta.content`; a tool call arrives in
pieces (`delta.tool_calls[i]`: first its id and name, then fragments of
its JSON arguments), assembled here. The reasoning of a thinking model
arrives apart (`delta.reasoning_content`) and is not part of the answer.

Written by hand with httpx rather than the OpenAI SDK, to see the protocol.
"""

import json
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from prestie.agent.model import (
    AgentError,
    ModelResponse,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)

DEFAULT_TIMEOUT_S = 300.0  # a long prompt on a busy GPU can take a while
SSE_DATA = "data: "
SSE_DONE = "[DONE]"
ERROR_PREFIX = "Error: "  # OpenAI tool messages have no error flag
INVALID_ARGUMENTS_KEY = "_invalid_arguments"
STOP_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAICompatibleChatModel:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        thinking: bool = False,
        http: httpx.Client | None = None,
    ):
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self._api_key = api_key
        self._thinking = thinking
        self._http = http or httpx.Client(timeout=DEFAULT_TIMEOUT_S)

    def stream(
        self,
        *,
        system: str,
        tools: Sequence[Mapping[str, Any]],
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        allow_tools: bool,
    ) -> Generator[StreamEvent, None, ModelResponse]:
        body = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "tools": [_to_openai_tool(tool) for tool in tools],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            # llama.cpp passes these to the chat template (Qwen: thinking on/off).
            "chat_template_kwargs": {"enable_thinking": self._thinking},
            **({} if allow_tools else {"tool_choice": "none"}),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        answer = _Answer()
        try:
            with self._http.stream("POST", self._url, json=body, headers=headers) as http:
                if http.status_code >= 400:
                    http.read()
                    raise AgentError(_http_error_message(http))
                for chunk in _sse_chunks(http.iter_lines()):
                    text = answer.add(chunk)
                    if text:
                        yield TextDelta(text)
        except httpx.HTTPError as exc:
            raise AgentError(
                f"Impossible de joindre le modèle local (llama-server, {self._url}) : {exc}"
            ) from exc
        return answer.response(default_model=self.model)


# --- request ---------------------------------------------------------------


def to_openai_messages(
    system: str, messages: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] == "assistant":
            converted.append(_assistant_message(message["content"]))
        else:
            converted.extend(_user_messages(message["content"]))
    return converted


def _assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    text = "".join(
        _field(block, "text") for block in content if _field(block, "type") == "text"
    )
    calls = [
        {
            "id": _field(block, "id"),
            "type": "function",
            "function": {
                "name": _field(block, "name"),
                "arguments": json.dumps(_field(block, "input"), ensure_ascii=False),
            },
        }
        for block in content
        if _field(block, "type") == "tool_use"
    ]
    # Thinking blocks are not sent back: the local model starts fresh each time.
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    return {**message, "tool_calls": calls} if calls else message


def _user_messages(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    messages = []
    for block in content:
        kind = _field(block, "type")
        if kind == "tool_result":
            result = str(_field(block, "content"))
            error = _field(block, "is_error")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _field(block, "tool_use_id"),
                    "content": f"{ERROR_PREFIX}{result}" if error else result,
                }
            )
        elif kind == "text":
            messages.append({"role": "user", "content": _field(block, "text")})
    return messages


def _to_openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def _field(block: Any, name: str) -> Any:
    """Blocks are SDK objects, our dataclasses, or plain dicts (tool results)."""
    if isinstance(block, Mapping):
        return block.get(name)
    return getattr(block, name, None)


# --- response ----------------------------------------------------------------


def _sse_chunks(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith(SSE_DATA):
            continue  # blank separators, comments, keep-alives
        data = line[len(SSE_DATA) :].strip()
        if data == SSE_DONE:
            return
        chunk = json.loads(data)
        if "error" in chunk:
            error = chunk["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise AgentError(f"Erreur du modèle local : {message}")
        yield chunk


@dataclass
class _ToolCallParts:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _Answer:
    """Accumulates the chunks of one streamed response."""

    text: str = ""
    calls: dict[int, _ToolCallParts] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None

    def add(self, chunk: Mapping[str, Any]) -> str:
        """Record one chunk; return its answer text, to stream."""
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = _usage(chunk["usage"])
        text = ""
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            for part in delta.get("tool_calls") or ():
                call = self.calls.setdefault(part.get("index", 0), _ToolCallParts())
                call.id = part.get("id") or call.id
                function = part.get("function") or {}
                call.name = function.get("name") or call.name
                call.arguments += function.get("arguments") or ""
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
        self.text += text
        return text

    def response(self, *, default_model: str) -> ModelResponse:
        tool_uses = [
            ToolUseBlock(call.id, call.name, _arguments(call.arguments))
            for _, call in sorted(self.calls.items())
        ]
        text = self.text.strip()
        blocks = ([TextBlock(text)] if text else []) + tool_uses
        # Some servers end a tool call with "stop": the calls are what matter.
        stop_reason = (
            "tool_use"
            if tool_uses
            else STOP_REASONS.get(self.finish_reason or "stop", "end_turn")
        )
        return ModelResponse(
            content=tuple(blocks),
            text=text,
            stop_reason=stop_reason,
            usage=self.usage,
            model=self.model or default_model,
        )


def _arguments(raw: str) -> Mapping[str, Any]:
    """The call's input; unparsable JSON goes to the tool as an invalid input,
    which it rejects, so the model reads the error and can try again."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {INVALID_ARGUMENTS_KEY: raw}
    return parsed if isinstance(parsed, dict) else {INVALID_ARGUMENTS_KEY: raw}


def _usage(raw: Mapping[str, Any]) -> Usage:
    prompt = raw.get("prompt_tokens") or 0
    cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    # Same meaning as the Anthropic usage: input_tokens is the uncached part.
    return Usage(
        input_tokens=prompt - cached,
        output_tokens=raw.get("completion_tokens") or 0,
        cache_read_input_tokens=cached,
    )


def _http_error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        detail = error.get("message", error) if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        detail = response.text[:200]
    return f"Erreur du modèle local ({response.status_code}) : {detail}"
