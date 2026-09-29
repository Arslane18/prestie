import asyncio
import threading

import pytest

from prestie.api.streaming import ThreadedStream


def collect(start, on_done):
    async def run():
        return [item async for item in ThreadedStream(start, on_done=on_done)]

    return asyncio.run(run())


def test_items_arrive_in_order_and_on_done_runs_in_the_worker():
    done_in = []

    items = collect(
        lambda: iter([1, 2, 3]),
        on_done=lambda: done_in.append(threading.current_thread().name),
    )

    assert items == [1, 2, 3]
    assert done_in == ["prestie-turn"]


def test_an_error_while_streaming_reaches_the_consumer_and_still_ends_the_turn():
    done = []

    def failing():
        yield 1
        raise RuntimeError("API down")

    with pytest.raises(RuntimeError, match="API down"):
        collect(failing, on_done=lambda: done.append(True))

    assert done == [True]


def test_an_error_while_starting_is_reported_too():
    done = []

    def cannot_start():
        raise RuntimeError("chroma is gone")

    with pytest.raises(RuntimeError, match="chroma is gone"):
        collect(cannot_start, on_done=lambda: done.append(True))

    assert done == [True]


def test_the_stream_is_closed_by_the_worker_when_it_ends():
    closed_in = []

    def stream():
        try:
            yield 1
        finally:
            closed_in.append(threading.current_thread().name)

    collect(stream, on_done=lambda: None)

    assert closed_in == ["prestie-turn"]
