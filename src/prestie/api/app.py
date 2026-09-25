"""Local HTTP API behind the companion window.

Server-Sent Events (SSE) in a nutshell: the response stays open with
`Content-Type: text/event-stream` and the server writes `event:`/`data:`
blocks as things happen. It only flows server -> browser, which is all a chat
answer or a character update needs (WebSocket would add a return channel we
do not use).

    POST /api/chat              SSE: tool, text, restart, done | error
    POST /api/reset             new conversation
    GET  /api/character         current character state (JSON envelope)
    GET  /api/character/stream  SSE: state | error, pushed on every file write

The API listens on 127.0.0.1 only, but any web page open in the player's
browser can still send requests to localhost. Two guards keep other sites
from spending the player's API credits: POSTs need a custom header, which a
cross-site page cannot add without a CORS preflight we never grant, and only
local Host headers are accepted, which blocks DNS rebinding.
"""

import asyncio
import logging
import threading
from pathlib import Path
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from prestie.agent.agent import (
    AgentError,
    AgentEvent,
    AgentReply,
    FallbackRestart,
    TextDelta,
    ToolCallStarted,
    TurnFinished,
)
from prestie.agent.tool_labels import tool_call_label
from prestie.character.state import CharacterState, CharacterStateError

logger = logging.getLogger(__name__)

CLIENT_HEADER = "X-Prestie-Client"
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost")
MAX_QUESTION_CHARS = 2000
CHARACTER_POLL_INTERVAL_S = 1.0
SECONDS_PER_MINUTE = 60
BUSY_MESSAGE = "Une réponse est déjà en cours, attends qu'elle se termine."
UNEXPECTED_ERROR_MESSAGE = "Erreur inattendue du serveur Prestie (voir ses logs)."
STATIC_DIR = Path(__file__).parent / "static"
# Scripts and styles only from our own files: even if model or game text ever
# reached the page as HTML, no inline script could run.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS
    ),
]


class ChatRequest(BaseModel):
    question: Question


class StreamsAnswers(Protocol):
    def ask_stream(self, question: str) -> Iterator[AgentEvent]: ...


class WatchesCharacter(Protocol):
    def latest(self) -> CharacterState: ...

    def poll(self) -> CharacterState | None: ...


class ChatSession:
    """The single conversation of this local, single-player app."""

    def __init__(self, agent_factory: Callable[[], StreamsAnswers]):
        self._agent_factory = agent_factory
        self._agent: StreamsAnswers | None = None
        # One turn at a time: the agent's history is not safe to interleave.
        self.turn_lock = threading.Lock()

    @property
    def agent(self) -> StreamsAnswers:
        if self._agent is None:
            self._agent = self._agent_factory()
        return self._agent

    def reset(self) -> None:
        self._agent = None


def create_app(
    *,
    agent_factory: Callable[[], StreamsAnswers],
    watcher_factory: Callable[[], WatchesCharacter],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
) -> FastAPI:
    app = FastAPI(title="Prestie", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    _add_error_handlers(app)
    session = ChatSession(agent_factory)
    watcher = watcher_factory()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={
                "Content-Security-Policy": CONTENT_SECURITY_POLICY,
                "Cache-Control": "no-store",
            },
        )

    @app.post(
        "/api/chat",
        response_class=EventSourceResponse,
        dependencies=[Depends(require_client_header)],
    )
    def chat(request: ChatRequest) -> Iterator[ServerSentEvent]:
        # A sync generator: FastAPI iterates it in a worker thread, so the
        # blocking Claude calls never stall the event loop.
        if not session.turn_lock.acquire(blocking=False):
            yield _error_event(BUSY_MESSAGE)
            return
        try:
            yield from _chat_events(session.agent, request.question)
        finally:
            session.turn_lock.release()

    @app.post("/api/reset", dependencies=[Depends(require_client_header)])
    def reset() -> dict[str, Any]:
        if not session.turn_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        try:
            session.reset()
        finally:
            session.turn_lock.release()
        return _envelope(None)

    @app.get("/api/character")
    def character() -> dict[str, Any]:
        try:
            state = watcher.latest()
        except CharacterStateError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _envelope(character_json(state, now()))

    @app.get("/api/character/stream", response_class=EventSourceResponse)
    async def character_stream() -> AsyncIterator[ServerSentEvent]:
        # One watcher per connection: each tracks what it already pushed.
        async for event in character_updates(watcher_factory(), now=now):
            yield event

    return app


def require_client_header(
    client: Annotated[str | None, Header(alias=CLIENT_HEADER)] = None,
) -> None:
    if not client:
        raise HTTPException(status_code=403, detail=f"missing {CLIENT_HEADER} header")


def _chat_events(agent: StreamsAnswers, question: str) -> Iterator[ServerSentEvent]:
    try:
        for event in agent.ask_stream(question):
            yield _to_sse(event)
    except AgentError as exc:
        yield _error_event(str(exc))
    except Exception:  # noqa: BLE001 - logged here, never leaked to the page
        logger.exception("chat turn failed")
        yield _error_event(UNEXPECTED_ERROR_MESSAGE)


def _to_sse(event: AgentEvent) -> ServerSentEvent:
    if isinstance(event, TextDelta):
        return ServerSentEvent(event="text", data={"text": event.text})
    if isinstance(event, ToolCallStarted):
        return ServerSentEvent(
            event="tool",
            data={"name": event.call.name, "label": tool_call_label(event.call)},
        )
    if isinstance(event, FallbackRestart):
        return ServerSentEvent(event="restart", data={})
    if isinstance(event, TurnFinished):
        return ServerSentEvent(event="done", data=_reply_json(event.reply))
    raise TypeError(f"unknown agent event: {event!r}")


def _reply_json(reply: AgentReply) -> dict[str, Any]:
    return {
        "text": reply.text,
        "refused": reply.refused,
        "truncated": reply.truncated,
        "tool_calls": len(reply.tool_calls),
        "usage": asdict(reply.usage),
    }


def _error_event(message: str) -> ServerSentEvent:
    return ServerSentEvent(event="error", data={"message": message})


def character_json(state: CharacterState, now: datetime) -> dict[str, Any]:
    age_seconds = max(0, int((now - state.captured_at).total_seconds()))
    return {
        **asdict(state),
        "captured_at": state.captured_at.isoformat(),
        "age_minutes": age_seconds // SECONDS_PER_MINUTE,
        "covered_by_knowledge_base": state.covered_by_knowledge_base,
        "guide_spec": state.guide.key if state.guide else None,
        "class_guides": [spec.key for spec in state.class_guides],
    }


async def character_updates(
    watcher: WatchesCharacter,
    *,
    now: Callable[[], datetime],
    interval: float = CHARACTER_POLL_INTERVAL_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[ServerSentEvent]:
    """Push the state on every change, and each distinct error once."""
    last_error: str | None = None
    while True:
        try:
            # stat() and parsing block: keep them off the event loop.
            state = await asyncio.to_thread(watcher.poll)
        except CharacterStateError as exc:
            state = None
            if str(exc) != last_error:
                last_error = str(exc)
                yield _error_event(last_error)
        if state is not None:
            last_error = None
            yield ServerSentEvent(event="state", data=character_json(state, now()))
        await sleep(interval)


def _envelope(data: Any, error: str | None = None) -> dict[str, Any]:
    return {"success": error is None, "data": data, "error": error}


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            _envelope(None, str(exc.detail)), status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages = "; ".join(error["msg"] for error in exc.errors())
        return JSONResponse(_envelope(None, messages), status_code=422)
