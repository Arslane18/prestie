"""The window closing mid-answer: the turn must end promptly and cleanly.

These tests speak raw ASGI to simulate a client that goes away after the
first event, which TestClient cannot do.
"""

import asyncio
import json
import threading

from prestie.agent.agent import AgentReply, TextDelta, TurnFinished
from prestie.api.app import CLIENT_HEADER, create_app
from tests.api.test_app import NOW, FakeWatcher

WAIT_S = 2.0
POLL_S = 0.02


class BlockingAgent:
    """Streams one word, then waits for the test before going on."""

    def __init__(self):
        self.resume = threading.Event()
        self.closed_in: list[str] = []  # thread that ran the generator's cleanup
        self.finished = False

    def ask_stream(self, question):
        try:
            yield TextDelta("Hâte ")
            self.resume.wait(WAIT_S)
            yield TextDelta("d'abord.")
            self.finished = True
            yield TurnFinished(AgentReply("Hâte d'abord."))
        finally:
            self.closed_in.append(threading.current_thread().name)


def app_with(agent):
    return create_app(
        agent_factory=lambda: agent,
        watcher_factory=lambda: FakeWatcher(),
        now=lambda: NOW,
        allowed_hosts=["testserver"],
    )


def scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/chat",
        "raw_path": b"/api/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (CLIENT_HEADER.lower().encode(), b"1"),
        ],
        "client": ("127.0.0.1", 5000),
        "server": ("testserver", 80),
    }


async def post_chat(app, *, disconnect_after_first_event=False):
    """One /api/chat request over raw ASGI; returns (status, body bytes)."""
    disconnect = asyncio.Event()
    request_sent = False
    sent: list[dict] = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            body = json.dumps({"question": "Stats ?"}).encode()
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()  # never set: the client stays until the end
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        body = message.get("body", b"")
        if disconnect_after_first_event and b"event:" in body:
            disconnect.set()

    await app(scope(), receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent)


async def wait_until(condition):
    for _ in range(int(WAIT_S / POLL_S)):
        if condition():
            return
        await asyncio.sleep(POLL_S)


def test_a_disconnect_closes_the_turn_in_a_worker_and_frees_the_lock():
    agent = BlockingAgent()
    app = app_with(agent)

    async def scenario():
        status, _ = await post_chat(app, disconnect_after_first_event=True)
        agent.resume.set()  # the agent reaches its next event
        await wait_until(lambda: agent.closed_in)  # no garbage collector needed
        stopped_early = not agent.finished  # read before the next turn runs
        again, _ = await asyncio.wait_for(post_chat(app), timeout=WAIT_S * 2)
        return status, stopped_early, again

    status, stopped_early, again = asyncio.run(scenario())

    assert status == 200
    assert agent.closed_in, "the turn's generator was never closed"
    assert agent.closed_in[0] != threading.main_thread().name  # not the event loop
    assert stopped_early, "the turn ran to the end with nobody reading"
    assert again == 200, "the lock was still held after the disconnect"


def test_a_second_question_during_a_turn_gets_a_real_409():
    agent = BlockingAgent()
    app = app_with(agent)

    async def scenario():
        running = asyncio.create_task(post_chat(app))
        await asyncio.sleep(0.2)  # the first turn is streaming
        busy = await asyncio.wait_for(post_chat(app), timeout=WAIT_S)
        agent.resume.set()
        await asyncio.wait_for(running, timeout=WAIT_S * 2)
        return busy

    status, body = asyncio.run(scenario())

    payload = json.loads(body)
    assert status == 409
    assert payload["success"] is False
    assert "déjà en cours" in payload["error"]
