"""Run one chat turn's blocking event stream in a thread that owns it.

FastAPI iterates a sync generator by calling `next()` in a thread pool. When
the client disconnects, that iteration is cancelled but nobody calls
`close()` on the generator: its `finally` blocks (closing the Claude stream,
writing the trace, releasing the turn lock) wait for the garbage collector,
and then run on the event loop's thread.

Here a dedicated worker thread builds the stream, pulls every event from it
and always closes it itself. The consumer (the HTTP response) reads events
from a queue; when it goes away, it raises a cancel flag and the worker stops
at the next event, closes the stream in its own thread, then calls `on_done`
(which releases the turn lock). Until then, a new turn cannot start: the
agent's history is not meant for two turns at once.

Limit: a worker blocked inside `next()` (Claude thinking before its first
word) only sees the flag at the next event, so the delay is bounded by that
event, but deterministic.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

_ITEM, _ERROR, _DONE = "item", "error", "done"


class ThreadedStream:
    """An async view of a blocking iterator that one worker thread owns."""

    def __init__(
        self,
        start: Callable[[], Iterator[Any]],
        *,
        on_done: Callable[[], None] = lambda: None,
        name: str = "prestie-turn",
    ):
        self._start = start
        self._on_done = on_done
        self._cancelled = threading.Event()
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        # Started now, not on first read: `on_done` must run even if the
        # response body is never iterated.
        self._worker = threading.Thread(target=self._work, name=name, daemon=True)
        self._worker.start()

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            while True:
                kind, value = await self._queue.get()
                if kind == _DONE:
                    return
                if kind == _ERROR:
                    raise value
                yield value
        finally:
            # Normal end, error or client gone: the worker stops at its next event.
            self._cancelled.set()

    def _work(self) -> None:
        iterator: Iterator[Any] | None = None
        try:
            iterator = self._start()
            for item in iterator:
                if self._cancelled.is_set():
                    break
                self._post(_ITEM, item)
        except BaseException as exc:  # noqa: BLE001 - re-raised by the consumer
            self._post(_ERROR, exc)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()  # runs the stream's cleanup here, in this thread
            except Exception:
                logger.exception("closing the turn's stream failed")
            finally:
                self._on_done()
                self._post(_DONE, None)

    def _post(self, kind: str, value: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, value))
        except RuntimeError:
            pass  # the event loop is closed (server shutting down): nobody reads
