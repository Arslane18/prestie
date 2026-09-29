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
import json
import logging
import threading
import uuid
from pathlib import Path
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Sequence,
)
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent, format_sse_event
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from prestie.agent.agent import (
    AgentError,
    AgentEvent,
    AgentReply,
    FallbackRestart,
    RequestFinished,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from prestie.agent.tool_labels import tool_call_label
from prestie.api.streaming import ThreadedStream
from prestie.character.snapshot import state_to_snapshot
from prestie.character.state import CharacterState, CharacterStateError
from prestie.observability.trace import TurnRecorder, WritesTraces

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
# reached the page as HTML, no inline script could run. 'unsafe-eval' is only
# there for pywebview's Qt backend, which builds window.pywebview.api with
# `new Function`; our own code never evaluates strings.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self'; "
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


# Trace ids are uuid4 hex strings: nothing else reaches the votes file.
TraceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
TRACING_OFF_MESSAGE = "Les traces sont désactivées : pas de vote possible."


class FeedbackRequest(BaseModel):
    trace_id: TraceId
    rating: Literal["up", "down"]


class RecordsTraces(WritesTraces, Protocol):
    def write_feedback(self, trace_id: str, rating: str, at: datetime) -> None: ...


class StreamsAnswers(Protocol):
    def ask_stream(self, question: str) -> Iterator[AgentEvent]: ...


class WatchesCharacter(Protocol):
    def latest(self) -> CharacterState: ...

    def poll(self) -> CharacterState | None: ...


class ChatSession:
    """The single conversation of this local, single-player app.

    With a trace sink, each turn is recorded; a reset starts a new session id,
    since the agent forgets the conversation.
    """

    def __init__(
        self,
        agent_factory: Callable[[], StreamsAnswers],
        recorder_factory: Callable[[], TurnRecorder] | None = None,
    ):
        self._agent_factory = agent_factory
        self._recorder_factory = recorder_factory
        self._agent: StreamsAnswers | None = None
        self._recorder: TurnRecorder | None = None
        # One turn at a time: the agent's history is not safe to interleave.
        self.turn_lock = threading.Lock()

    @property
    def agent(self) -> StreamsAnswers:
        if self._agent is None:
            self._agent = self._agent_factory()
            self._recorder = (
                self._recorder_factory() if self._recorder_factory else None
            )
        return self._agent

    def ask_stream(self, question: str) -> tuple[str | None, Iterator[AgentEvent]]:
        """The turn's trace id (None when not traced) and its events."""
        events = self.agent.ask_stream(question)
        if self._recorder is None:
            return None, events
        trace_id = uuid.uuid4().hex
        return trace_id, self._recorder.observe(question, events, trace_id=trace_id)

    def reset(self) -> None:
        self._agent = None
        self._recorder = None


def create_app(
    *,
    agent_factory: Callable[[], StreamsAnswers],
    watcher_factory: Callable[[], WatchesCharacter],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    trace_sink: RecordsTraces | None = None,
) -> FastAPI:
    # No docs nor schema: a local app has no API consumers to describe itself to.
    app = FastAPI(title="Prestie", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    _add_error_handlers(app)
    watcher = watcher_factory()
    recorder_factory = (
        (
            lambda: TurnRecorder(
                trace_sink,
                session_id=uuid.uuid4().hex,
                character=lambda: character_summary(watcher, now()),
                character_snapshot=lambda: character_snapshot(watcher),
            )
        )
        if trace_sink is not None
        else None
    )
    session = ChatSession(agent_factory, recorder_factory)
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

    @app.post("/api/chat", dependencies=[Depends(require_client_header)])
    async def chat(request: ChatRequest) -> EventSourceResponse:
        # Checked before the response starts, so a busy turn is a real 409.
        if not session.turn_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        try:
            # The blocking Claude calls run in a worker that owns the turn and
            # releases the lock only once the turn is fully closed.
            stream = ThreadedStream(
                lambda: _chat_events(lambda: session.ask_stream(request.question)),
                on_done=session.turn_lock.release,
            )
        except BaseException:
            session.turn_lock.release()
            raise
        return EventSourceResponse(_sse_bytes(stream))

    @app.post("/api/feedback", dependencies=[Depends(require_client_header)])
    def feedback(request: FeedbackRequest) -> dict[str, Any]:
        if trace_sink is None:
            raise HTTPException(status_code=409, detail=TRACING_OFF_MESSAGE)
        trace_sink.write_feedback(request.trace_id, request.rating, now())
        return _envelope(None)

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


# Timings and usage for traces: nothing the page shows.
TRACE_ONLY_EVENTS = (ToolCallFinished, RequestFinished)


def _chat_events(
    start: Callable[[], tuple[str | None, Iterator[AgentEvent]]],
) -> Iterator[ServerSentEvent]:
    """SSE events of one turn. `start` builds the agent and its event stream
    inside the error handling, so a failure there still reaches the page."""
    try:
        trace_id, events = start()
        for event in events:
            if not isinstance(event, TRACE_ONLY_EVENTS):
                yield _to_sse(event, trace_id)
    except AgentError as exc:
        yield _error_event(str(exc))
    except Exception:  # noqa: BLE001 - logged here, never leaked to the page
        logger.exception("chat turn failed")
        yield _error_event(UNEXPECTED_ERROR_MESSAGE)


async def _sse_bytes(events: AsyncIterable[ServerSentEvent]) -> AsyncIterator[bytes]:
    async for event in events:
        yield format_sse_event(
            event=event.event, data_str=json.dumps(event.data, ensure_ascii=False)
        )


def _to_sse(event: AgentEvent, trace_id: str | None = None) -> ServerSentEvent:
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
        return ServerSentEvent(event="done", data=_reply_json(event.reply, trace_id))
    raise TypeError(f"unknown agent event: {event!r}")


def _reply_json(reply: AgentReply, trace_id: str | None = None) -> dict[str, Any]:
    return {
        "trace_id": trace_id,  # lets the page attach the player's vote
        "text": reply.text,
        "refused": reply.refused,
        "truncated": reply.truncated,
        "tool_calls": len(reply.tool_calls),
        "usage": asdict(reply.usage),
    }


def _error_event(message: str) -> ServerSentEvent:
    return ServerSentEvent(event="error", data={"message": message})


# Gear is for the agent (get_equipment), not the card: kept out of every push.
CARD_EXCLUDED_FIELDS = ("equipment", "equipment_error")


def character_json(state: CharacterState, now: datetime) -> dict[str, Any]:
    age_seconds = max(0, int((now - state.captured_at).total_seconds()))
    card = replace(state, equipment=None, equipment_error=None)
    fields = {k: v for k, v in asdict(card).items() if k not in CARD_EXCLUDED_FIELDS}
    return {
        **fields,
        "captured_at": state.captured_at.isoformat(),
        "age_minutes": age_seconds // SECONDS_PER_MINUTE,
        "covered_by_knowledge_base": state.covered_by_knowledge_base,
        "guide_spec": state.guide.key if state.guide else None,
        "class_guides": [spec.key for spec in state.class_guides],
    }


def character_summary(
    watcher: WatchesCharacter, now: datetime
) -> dict[str, Any] | None:
    """What a trace keeps of the character: enough to replay the question."""
    try:
        state = watcher.latest()
    except CharacterStateError:
        return None
    age_seconds = max(0, int((now - state.captured_at).total_seconds()))
    return {
        "class": state.class_token,
        "spec_id": state.spec.id if state.spec else None,
        "spec": state.spec.name if state.spec else None,
        "level": state.level,
        "hero_talent": state.hero_talent,
        "age_minutes": age_seconds // SECONDS_PER_MINUTE,
    }


def character_snapshot(watcher: WatchesCharacter) -> dict[str, Any] | None:
    """The character in the addon's format, so a traced turn can be replayed."""
    try:
        return state_to_snapshot(watcher.latest())
    except CharacterStateError:
        return None


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
    # Starlette's own class also covers its 404/405, raised before any route.
    @app.exception_handler(StarletteHTTPException)
    async def http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            _envelope(None, str(exc.detail)), status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages = "; ".join(error["msg"] for error in exc.errors())
        return JSONResponse(_envelope(None, messages), status_code=422)
