"""One trace per turn: what the player asked, what the agent did, what it cost.

`TurnRecorder.observe` wraps the agent's event stream and passes every event
through unchanged, so the UI streams exactly as before; it only watches. The
agent does not know it is traced: timings of each API request and tool call
come from its own events (RequestFinished, ToolCallFinished), and the
recorder adds what only the consumer sees, like the time before the first
word reaches the player.

The fields follow the OpenTelemetry GenAI conventions in spirit (model,
input/output/cache tokens per request, one entry per tool call), so an
export to a tracing backend stays a mapping away.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from prestie.agent.agent import (
    AgentError,
    AgentEvent,
    RequestFinished,
    TextDelta,
    ToolCallFinished,
    TurnFinished,
    Usage,
)
from prestie.pricing import usage_cost_usd

logger = logging.getLogger(__name__)

# How a turn ended.
ANSWERED = "answered"
REFUSED = "refused"
TRUNCATED = "truncated"
ERROR = "error"
INTERRUPTED = "interrupted"  # the consumer stopped reading (window closed)


class WritesTraces(Protocol):
    def write(self, trace: "TurnTrace") -> None: ...


@dataclass(frozen=True)
class RequestTrace:
    model: str
    stop_reason: str
    duration_s: float
    usage: Usage


@dataclass(frozen=True)
class ToolTrace:
    name: str
    input: Mapping[str, Any]
    is_error: bool
    duration_s: float
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnTrace:
    trace_id: str
    session_id: str
    turn: int  # 1-based, within the session
    started_at: datetime
    question: str
    character: Mapping[str, Any] | None  # summary at question time
    # The character in the addon's raw format, to replay the turn as an eval case.
    character_snapshot: Mapping[str, Any] | None
    outcome: str
    error: str | None
    duration_s: float
    time_to_first_token_s: float | None
    requests: tuple[RequestTrace, ...]
    tools: tuple[ToolTrace, ...]
    answer: str
    cost_usd: float | None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return {**data, "started_at": self.started_at.isoformat()}


class TurnRecorder:
    """Traces every turn of one conversation (one per window, reset = new)."""

    def __init__(
        self,
        sink: WritesTraces,
        *,
        session_id: str,
        character: Callable[[], Mapping[str, Any] | None] = lambda: None,
        character_snapshot: Callable[[], Mapping[str, Any] | None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ):
        self._sink = sink
        self.session_id = session_id
        self._character = character
        self._character_snapshot = character_snapshot
        self._clock = clock
        self._now = now
        self._new_id = new_id
        self._turns = 0

    def observe(
        self,
        question: str,
        events: Iterator[AgentEvent],
        trace_id: str | None = None,
    ) -> Iterator[AgentEvent]:
        """Pass `events` through, then write the turn's trace. `trace_id` lets
        the caller know the id upfront (e.g. to attach the player's vote)."""
        trace_id = trace_id or self._new_id()
        self._turns += 1
        turn = _TurnState(
            started=self._clock(), started_at=self._now(), turn=self._turns
        )
        character = self._character()
        snapshot = self._character_snapshot()
        try:
            for event in events:
                turn = turn.record(event, self._clock())
                yield event
        except AgentError as exc:
            turn = turn.failed(str(exc))
            raise
        except Exception as exc:
            turn = turn.failed(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            trace = turn.trace(
                trace_id=trace_id,
                session_id=self.session_id,
                question=question,
                character=character,
                character_snapshot=snapshot,
                ended=self._clock(),
            )
            self._write(trace)

    def _write(self, trace: TurnTrace) -> None:
        try:
            self._sink.write(trace)
        except OSError:
            # Tracing must never break the answer, but a lost trace is logged.
            logger.exception("could not write trace %s", trace.trace_id)


@dataclass(frozen=True)
class _TurnState:
    """What the recorder has seen so far of one turn (immutable, replaced)."""

    started: float
    started_at: datetime
    turn: int
    first_token_at: float | None = None
    requests: tuple[RequestTrace, ...] = ()
    tools: tuple[ToolTrace, ...] = ()
    outcome: str = INTERRUPTED
    error: str | None = None
    answer: str = ""

    def record(self, event: AgentEvent, at: float) -> "_TurnState":
        if isinstance(event, TextDelta) and self.first_token_at is None:
            return replace(self, first_token_at=at)
        if isinstance(event, RequestFinished):
            request = RequestTrace(
                event.model, event.stop_reason, event.duration_s, event.usage
            )
            return replace(self, requests=(*self.requests, request))
        if isinstance(event, ToolCallFinished):
            tool = ToolTrace(
                event.call.name,
                dict(event.call.input),
                event.is_error,
                event.duration_s,
                dict(event.details),
            )
            return replace(self, tools=(*self.tools, tool))
        if isinstance(event, TurnFinished):
            reply = event.reply
            outcome = (
                REFUSED if reply.refused else TRUNCATED if reply.truncated else ANSWERED
            )
            return replace(self, outcome=outcome, answer=reply.text)
        return self

    def failed(self, message: str) -> "_TurnState":
        return replace(self, outcome=ERROR, error=message)

    def trace(
        self,
        *,
        trace_id: str,
        session_id: str,
        question: str,
        character: Mapping[str, Any] | None,
        character_snapshot: Mapping[str, Any] | None,
        ended: float,
    ) -> TurnTrace:
        return TurnTrace(
            trace_id=trace_id,
            session_id=session_id,
            turn=self.turn,
            started_at=self.started_at,
            question=question,
            character=character,
            character_snapshot=character_snapshot,
            outcome=self.outcome,
            error=self.error,
            duration_s=ended - self.started,
            time_to_first_token_s=(
                None
                if self.first_token_at is None
                else self.first_token_at - self.started
            ),
            requests=self.requests,
            tools=self.tools,
            answer=self.answer,
            cost_usd=_cost(self.requests),
        )


def _cost(requests: tuple[RequestTrace, ...]) -> float | None:
    costs = [usage_cost_usd(r.model, r.usage) for r in requests]
    if any(cost is None for cost in costs):
        return None  # a request on an unpriced model: no partial total
    return sum(costs)
