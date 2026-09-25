import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from prestie.agent.agent import (
    AgentError,
    AgentReply,
    FallbackRestart,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    TurnFinished,
    Usage,
)
from prestie.api.app import CLIENT_HEADER, character_updates, create_app
from prestie.character.state import CharacterState, CharacterStateError, Spec

NOW = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)
HEADERS = {CLIENT_HEADER: "1"}


def make_state(**overrides) -> CharacterState:
    base = dict(
        character="Lumina",
        realm="Khaz Modan",
        level=6,
        class_name="Prêtresse",
        class_token="PRIEST",
        spec=Spec(id=256, name="Discipline", role="HEALER"),
        hero_talent=None,
        active_quest=None,
        quests=(),
        captured_at=NOW - timedelta(minutes=12),
    )
    return CharacterState(**{**base, **overrides})


class ScriptedAgent:
    def __init__(self, events=None, error=None):
        self.events = events
        self.error = error
        self.questions: list[str] = []

    def ask_stream(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        yield from self.events or [
            ToolCallStarted(ToolCall("search_knowledge_base", {"query": "stats"})),
            TextDelta("Hâte "),
            TextDelta("d'abord."),
            TurnFinished(
                AgentReply(
                    "Hâte d'abord.",
                    tool_calls=(ToolCall("search_knowledge_base", {}),),
                    usage=Usage(input_tokens=10, output_tokens=5),
                )
            ),
        ]


class FakeWatcher:
    def __init__(self, state=None, error=None):
        self.state = state or make_state()
        self.error = error

    def latest(self):
        if self.error:
            raise self.error
        return self.state


@pytest.fixture
def agents():
    return []


def make_client(agents, *, agent=None, watcher=None) -> TestClient:
    def agent_factory():
        created = agent or ScriptedAgent()
        agents.append(created)
        return created

    app = create_app(
        agent_factory=agent_factory,
        watcher_factory=lambda: watcher or FakeWatcher(),
        now=lambda: NOW,
        allowed_hosts=["testserver"],
    )
    return TestClient(app)


def sse_events(response) -> list[tuple[str, dict]]:
    events = []
    for chunk in response.text.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in chunk.splitlines() if ": " in line
        )
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


# --- chat ------------------------------------------------------------------------


def test_chat_streams_tool_calls_text_and_the_final_reply(agents):
    client = make_client(agents)

    response = client.post("/api/chat", json={"question": "Stats ?"}, headers=HEADERS)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert [name for name, _ in events] == ["tool", "text", "text", "done"]
    assert events[0][1] == {
        "name": "search_knowledge_base",
        "label": "[recherche] stats",
    }
    assert events[1][1] == {"text": "Hâte "}
    done = events[-1][1]
    assert done["text"] == "Hâte d'abord."
    assert done["refused"] is False and done["truncated"] is False
    assert done["usage"]["output_tokens"] == 5
    assert agents[0].questions == ["Stats ?"]


def test_chat_relays_fallback_restarts(agents):
    agent = ScriptedAgent(
        events=[
            TextDelta("partial"),
            FallbackRestart(),
            TextDelta("final"),
            TurnFinished(AgentReply("final")),
        ]
    )
    client = make_client(agents, agent=agent)

    events = sse_events(
        client.post("/api/chat", json={"question": "q"}, headers=HEADERS)
    )

    assert [name for name, _ in events] == ["text", "restart", "text", "done"]


def test_agent_errors_become_an_error_event(agents):
    client = make_client(agents, agent=ScriptedAgent(error=AgentError("API en panne")))

    events = sse_events(
        client.post("/api/chat", json={"question": "q"}, headers=HEADERS)
    )

    assert events == [("error", {"message": "API en panne"})]


def test_unexpected_errors_are_not_leaked(agents):
    client = make_client(agents, agent=ScriptedAgent(error=RuntimeError("secret")))

    [(name, payload)] = sse_events(
        client.post("/api/chat", json={"question": "q"}, headers=HEADERS)
    )

    assert name == "error"
    assert "secret" not in payload["message"]


def test_the_conversation_is_kept_until_reset(agents):
    client = make_client(agents)

    client.post("/api/chat", json={"question": "q1"}, headers=HEADERS)
    client.post("/api/chat", json={"question": "q2"}, headers=HEADERS)
    reset = client.post("/api/reset", headers=HEADERS)
    client.post("/api/chat", json={"question": "q3"}, headers=HEADERS)

    assert reset.json() == {"success": True, "data": None, "error": None}
    assert [agent.questions for agent in agents] == [["q1", "q2"], ["q3"]]


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_invalid_questions_are_rejected(agents, question):
    client = make_client(agents)

    response = client.post("/api/chat", json={"question": question}, headers=HEADERS)

    assert response.status_code == 422
    assert all(agent.questions == [] for agent in agents)


@pytest.mark.parametrize("path", ["/api/chat", "/api/reset"])
def test_posts_require_the_client_header(agents, path):
    client = make_client(agents)

    response = client.post(path, json={"question": "q"})

    assert response.status_code == 403
    assert response.json()["success"] is False


def test_unknown_hosts_are_rejected(agents):
    # A page on another domain resolving to 127.0.0.1 (DNS rebinding) is refused.
    client = make_client(agents)

    response = client.get("/api/character", headers={"host": "evil.example"})

    assert response.status_code == 400


# --- character -------------------------------------------------------------------


def test_character_returns_the_state_with_its_age(agents):
    client = make_client(agents)

    body = client.get("/api/character").json()

    assert body["success"] is True and body["error"] is None
    data = body["data"]
    assert data["character"] == "Lumina"
    assert data["spec"] == {"id": 256, "name": "Discipline", "role": "HEALER"}
    assert data["age_minutes"] == 12
    assert data["captured_at"] == "2026-09-25T19:48:00+00:00"
    assert data["covered_by_knowledge_base"] is False


def test_character_error_uses_the_envelope(agents):
    watcher = FakeWatcher(error=CharacterStateError("Prestie.lua not found"))
    client = make_client(agents, watcher=watcher)

    response = client.get("/api/character")

    assert response.status_code == 503
    assert response.json() == {
        "success": False,
        "data": None,
        "error": "Prestie.lua not found",
    }


class PollingWatcher:
    """Returns scripted poll results: a state, None (unchanged) or an error."""

    def __init__(self, results):
        self.results = list(results)

    def poll(self):
        result = self.results.pop(0) if self.results else None
        if isinstance(result, Exception):
            raise result
        return result


async def _no_sleep(_seconds):
    return None


def collect_updates(watcher, count):
    async def run():
        events = []
        stream = character_updates(watcher, interval=0, now=lambda: NOW, sleep=_no_sleep)
        async for event in stream:
            events.append(event)
            if len(events) == count:
                break
        return events

    return asyncio.run(run())


def test_character_updates_push_each_new_state_and_each_new_error():
    broken = CharacterStateError("cannot parse")
    watcher = PollingWatcher(
        [make_state(level=6), None, broken, None, make_state(level=7)]
    )

    events = collect_updates(watcher, 3)

    assert [event.event for event in events] == ["state", "error", "state"]
    assert events[0].data["level"] == 6
    assert events[1].data == {"message": "cannot parse"}
    assert events[2].data["level"] == 7


def test_character_updates_report_a_repeated_error_once():
    broken = CharacterStateError("not found")
    watcher = PollingWatcher([broken, broken, broken, make_state(level=8)])

    events = collect_updates(watcher, 2)

    assert [event.event for event in events] == ["error", "state"]


# --- page ------------------------------------------------------------------------


def test_index_is_served_with_a_strict_content_security_policy(agents):
    client = make_client(agents)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<script type="module" src="/static/app.js">' in response.text
    policy = response.headers["content-security-policy"]
    assert "default-src 'self'" in policy
    assert "script-src 'self'" in policy
    assert "unsafe-inline" not in policy


@pytest.mark.parametrize(
    "path, media", [("/static/app.js", "javascript"), ("/static/style.css", "css")]
)
def test_static_assets_are_served(agents, path, media):
    response = make_client(agents).get(path)

    assert response.status_code == 200
    assert media in response.headers["content-type"]


def test_ui_helpers_pass_their_node_tests():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    test_file = Path(__file__).parent / "ui" / "format.test.mjs"

    result = subprocess.run(
        [node, "--test", str(test_file)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
