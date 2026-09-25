import json
from datetime import UTC, datetime

import httpx
import pytest

from prestie import cli
from prestie.ingestion.icy_veins.cache import CachedPage, HtmlCache
from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES
from tests.ingestion.icy_veins.html_fixtures import heading
from tests.ingestion.icy_veins.html_fixtures import page as html_page

ROBOTS_TXT = "User-agent: *\nAllow: /\n"
FETCHED_AT = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)


def fake_client(status_code: int = 200) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_TXT)
        return httpx.Response(status_code, text="<html></html>")

    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture(autouse=True)
def no_real_network_or_sleep(monkeypatch):
    monkeypatch.setattr(cli, "build_client", fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)


def test_scrape_caches_every_mvp_page(tmp_path, capsys):
    exit_code = cli.main(["scrape", "--cache-dir", str(tmp_path)])

    cache = HtmlCache(tmp_path)
    assert exit_code == 0
    assert all(cache.get(page.slug) is not None for page in BLOOD_DK_PAGES)
    assert "downloaded" in capsys.readouterr().out


def test_second_scrape_is_served_from_cache(tmp_path, capsys):
    cli.main(["scrape", "--cache-dir", str(tmp_path)])
    capsys.readouterr()

    cli.main(["scrape", "--cache-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "downloaded" not in out
    assert out.count("cached") == len(BLOOD_DK_PAGES)


def test_scrape_reports_failure_with_non_zero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_client", lambda: fake_client(status_code=503))

    exit_code = cli.main(["scrape", "--cache-dir", str(tmp_path)])

    assert exit_code == 1
    assert "503" in capsys.readouterr().err


class FakeEmbedder:
    model = "fake-model"

    def embed_documents(self, texts):
        return [[1.0, float(len(text) % 7)] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def knowledge_env(tmp_path, monkeypatch):
    """Cache every MVP page with synthetic HTML and point the store at tmp_path."""
    cache = HtmlCache(tmp_path / "raw")
    for page in BLOOD_DK_PAGES:
        html = html_page(heading(2, "1.", "Stats", "stats") + f"<p>{page.slug}</p>")
        cache.put(CachedPage(page.slug, page.url, html, FETCHED_AT, 200))
    monkeypatch.setenv("PRESTIE_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("VOYAGE_MODEL", FakeEmbedder.model)
    monkeypatch.setattr(cli, "build_embedder", lambda settings: FakeEmbedder())
    return tmp_path / "raw"


def test_ingest_indexes_every_cached_page(knowledge_env, capsys):
    exit_code = cli.main(["ingest", "--cache-dir", str(knowledge_env)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.count("1 sections -> 1 chunks") == len(BLOOD_DK_PAGES)
    assert f"{len(BLOOD_DK_PAGES)} chunks indexed" in out


def test_search_prints_ranked_hits_with_source(knowledge_env, capsys):
    cli.main(["ingest", "--cache-dir", str(knowledge_env)])
    capsys.readouterr()

    exit_code = cli.main(["search", "stat priority", "-k", "2"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.count("https://www.icy-veins.com/wow/") == 2
    assert "Stats" in out


def test_search_can_filter_by_content_type(knowledge_env, capsys):
    cli.main(["ingest", "--cache-dir", str(knowledge_env)])
    capsys.readouterr()

    cli.main(["search", "anything", "--content-type", "leveling"])

    out = capsys.readouterr().out
    assert "blood-death-knight-leveling-guide" in out
    assert "stat-priority" not in out


def test_ingest_without_api_key_fails_cleanly(knowledge_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_embedder", cli.build_voyage_embedder)
    monkeypatch.setenv("VOYAGE_API_KEY", "")

    exit_code = cli.main(["ingest", "--cache-dir", str(knowledge_env)])

    assert exit_code == 1
    assert "VOYAGE_API_KEY" in capsys.readouterr().err


def test_eval_prints_per_question_ranks_and_summary(knowledge_env, tmp_path, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"question": "leveling?", "expected": ["blood-death-knight-leveling-guide"]},'
        ' {"question": "out of scope?", "expected": []}]',
        encoding="utf-8",
    )
    cli.main(["ingest", "--cache-dir", str(knowledge_env)])
    capsys.readouterr()

    exit_code = cli.main(
        ["eval", "--cases", str(cases), "-k", str(len(BLOOD_DK_PAGES))]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "leveling?" in out
    assert "n/a" in out
    assert "hit@1 =" in out
    assert "MRR =" in out
    assert "(1 answerable questions" in out


def test_eval_with_malformed_cases_fails_cleanly(knowledge_env, tmp_path, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text('{"not": "a list"}', encoding="utf-8")

    exit_code = cli.main(["eval", "--cases", str(cases)])

    assert exit_code == 1
    assert "expected a JSON list" in capsys.readouterr().err


class FakeAgent:
    def __init__(self, on_tool_call):
        self.on_tool_call = on_tool_call
        self.questions: list[str] = []

    def ask_stream(self, question):
        from prestie.agent.agent import (
            AgentReply,
            TextDelta,
            ToolCall,
            ToolCallStarted,
            TurnFinished,
            Usage,
        )

        self.questions.append(question)
        call = ToolCall(name="search_knowledge_base", input={"query": "stat priority"})
        yield ToolCallStarted(call)
        yield TextDelta("Priorité : ")
        yield TextDelta("Hâte.")
        yield TurnFinished(
            AgentReply(
                text="Priorité : Hâte.",
                tool_calls=(call,),
                usage=Usage(
                    input_tokens=1200, output_tokens=80, cache_read_input_tokens=900
                ),
            )
        )


@pytest.fixture
def fake_agent(monkeypatch):
    created = {}

    def build(settings, player, on_tool_call):
        created["player"] = player
        created["agent"] = FakeAgent(on_tool_call)
        return created["agent"]

    monkeypatch.setattr(cli, "build_agent", build)
    return created


def test_chat_one_shot_prints_searches_and_answer(fake_agent, capsys):
    exit_code = cli.main(
        ["chat", "--level", "80", "--hero-talent", "San'layn", "-q", "Quelles stats ?"]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert fake_agent["agent"].questions == ["Quelles stats ?"]
    assert fake_agent["player"].hero_talent == "San'layn"
    assert "stat priority" in out
    assert "Priorité : Hâte." in out


def test_chat_verbose_prints_token_usage(fake_agent, capsys):
    cli.main(["chat", "--level", "80", "-q", "q", "--verbose"])

    out = capsys.readouterr().out
    assert "2100" in out  # total input = 1200 uncached + 900 read from cache
    assert "900" in out


def test_chat_interactive_loop_until_exit(fake_agent, monkeypatch, capsys):
    answers = iter(["Question 1", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    exit_code = cli.main(["chat", "--level", "80"])

    assert exit_code == 0
    assert fake_agent["agent"].questions == ["Question 1"]


def test_chat_rejects_invalid_level(fake_agent, capsys):
    exit_code = cli.main(["chat", "--level", "500", "-q", "q"])

    assert exit_code == 1
    assert "level" in capsys.readouterr().err


@pytest.fixture
def agent_eval_env(tmp_path, monkeypatch):
    """Fake agent + judge so eval-agent runs without any API call."""
    from prestie.agent.agent import AgentReply
    from prestie.evaluation.agent_judge import JudgeVerdict

    class EvalAgent:
        def ask(self, question):
            return AgentReply(text="Merci à toi !", models=("claude-opus-5",))

    class EvalJudge:
        def __init__(self, client, model):
            self.model = model

        def grade(self, case, answer, tool_outputs):
            return JudgeVerdict(
                {"helpful": 1.0}, {"helpful": "ok"}, self.model, Usage()
            )

    from prestie.agent.agent import Usage

    monkeypatch.setattr(cli, "build_agent", lambda *args, **kwargs: EvalAgent())
    monkeypatch.setattr(cli, "Judge", EvalJudge)
    monkeypatch.setattr(cli, "build_eval_client", lambda settings: object())
    monkeypatch.setattr(cli, "open_retriever", lambda settings: object())
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    return tmp_path / "flow"


def test_eval_agent_refuses_to_run_an_unapproved_harness(agent_eval_env, capsys):
    exit_code = cli.main(
        ["eval-agent", "--flow-dir", str(agent_eval_env), "--only", "thanks"]
    )

    assert exit_code == 1
    assert "--approve-harness" in capsys.readouterr().err


def test_eval_agent_pilot_on_selected_cases(agent_eval_env, capsys):
    exit_code = cli.main(
        [
            "eval-agent",
            "--flow-dir",
            str(agent_eval_env),
            "--only",
            "thanks,insult",
            "--reps",
            "1",
            "--approve-harness",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "2 attempts run, 0 errors" in out
    assert "pass = 1.00" in out
    assert (agent_eval_env / "baseline" / "results.jsonl").exists()
    assert (agent_eval_env / "_state.json").exists()


def test_eval_agent_rejects_unknown_case_ids(agent_eval_env, capsys):
    exit_code = cli.main(
        [
            "eval-agent",
            "--flow-dir",
            str(agent_eval_env),
            "--only",
            "nope",
            "--approve-harness",
        ]
    )

    assert exit_code == 1
    assert "nope" in capsys.readouterr().err


def test_chat_without_level_runs_in_addon_mode(fake_agent, capsys):
    exit_code = cli.main(["chat", "-q", "Quelle quête ?"])

    assert exit_code == 0
    assert fake_agent["player"] is None
    assert fake_agent["agent"].questions == ["Quelle quête ?"]


def test_chat_rejects_hero_talent_without_level(fake_agent, capsys):
    exit_code = cli.main(["chat", "--hero-talent", "San'layn", "-q", "q"])

    assert exit_code == 1
    assert "--level" in capsys.readouterr().err


def test_character_state_tool_calls_are_announced():
    from prestie.agent.agent import ToolCall

    assert "personnage" in cli.format_tool_call(ToolCall("get_character_state", {}))


SAVED_VARIABLES = """\
PrestieDB = {
["schema"] = 1,
["snapshot"] = {
["capturedAt"] = 1790278000,
["character"] = "Testeur",
["realm"] = "Hyjal",
["level"] = 83,
["class"] = {
["name"] = "Chevalier de la mort",
["file"] = "DEATHKNIGHT",
},
},
}
"""


def test_watch_prints_the_character_state(tmp_path, monkeypatch, capsys):
    saved = tmp_path / "Prestie.lua"
    saved.write_text(SAVED_VARIABLES, encoding="utf-8")
    monkeypatch.setenv("PRESTIE_SAVEDVARIABLES", str(saved))

    def stop(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", stop)

    exit_code = cli.main(["watch"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Level: 83" in out
    assert "Testeur (Hyjal)" in out


def test_watch_without_configured_path_fails_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("PRESTIE_SAVEDVARIABLES", "")

    assert cli.main(["watch"]) == 1
    assert "PRESTIE_SAVEDVARIABLES" in capsys.readouterr().err


def test_quest_tool_calls_are_announced():
    from prestie.agent.agent import ToolCall

    line = cli.format_tool_call(ToolCall("get_quest_details", {"quest_id": 55763}))

    assert "55763" in line


def test_addon_mode_agent_gets_search_character_and_quest_tools(tmp_path):
    from prestie.config import load_settings

    settings = load_settings(
        {
            "PRESTIE_SAVEDVARIABLES": str(tmp_path / "Prestie.lua"),
            "BLIZZARD_CLIENT_ID": "id",
            "BLIZZARD_CLIENT_SECRET": "secret",
        }
    )

    agent = cli.build_agent(
        settings, None, lambda call: None, retriever=object(), client=object()
    )

    assert agent.tool_names == (
        "search_knowledge_base",
        "get_character_state",
        "get_quest_details",
    )


def test_manual_mode_agent_only_searches(tmp_path):
    from prestie.agent.prompts import PlayerContext
    from prestie.config import load_settings

    agent = cli.build_agent(
        load_settings({}),
        PlayerContext(level=80),
        lambda call: None,
        retriever=object(),
        client=object(),
    )

    assert agent.tool_names == ("search_knowledge_base",)


def test_addon_mode_without_blizzard_credentials_fails_cleanly(tmp_path):
    from prestie.config import ConfigError, load_settings

    settings = load_settings({"PRESTIE_SAVEDVARIABLES": str(tmp_path / "P.lua")})

    with pytest.raises(ConfigError, match="BLIZZARD_CLIENT_ID"):
        cli.build_agent(
            settings, None, lambda call: None, retriever=object(), client=object()
        )


def test_eval_agent_serves_the_case_character_in_addon_mode(
    agent_eval_env, monkeypatch, tmp_path, capsys
):
    from prestie.agent.agent import AgentReply

    built = []

    class EvalAgent:
        def ask(self, question):
            return AgentReply(text="ok", models=("claude-opus-5",))

    def build(settings, player, on_tool_call, **kwargs):
        built.append((player, kwargs.get("character_source")))
        return EvalAgent()

    monkeypatch.setattr(cli, "build_agent", build)
    cases = tmp_path / "addon_cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "state",
                    "question": "q",
                    "character": {
                        "character": "T",
                        "realm": "R",
                        "level": 90,
                        "class": {"name": "DK", "file": "DEATHKNIGHT"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "eval-agent",
            "--cases",
            str(cases),
            "--flow-dir",
            str(agent_eval_env),
            "--reps",
            "1",
            "--approve-harness",
        ]
    )

    assert exit_code == 0, capsys.readouterr().err
    [(player, source)] = built
    assert player is None
    assert source.latest().level == 90


def test_build_agent_uses_an_injected_character_source(tmp_path):
    from prestie.config import load_settings

    settings = load_settings(
        {"BLIZZARD_CLIENT_ID": "id", "BLIZZARD_CLIENT_SECRET": "secret"}
    )

    agent = cli.build_agent(
        settings,
        None,
        lambda call: None,
        retriever=object(),
        client=object(),
        character_source=object(),
    )

    assert "get_character_state" in agent.tool_names


def test_serve_starts_the_api_on_localhost_only(monkeypatch, tmp_path):
    started = {}
    monkeypatch.setenv("PRESTIE_SAVEDVARIABLES", str(tmp_path / "Prestie.lua"))
    monkeypatch.setenv("BLIZZARD_CLIENT_ID", "id")
    monkeypatch.setenv("BLIZZARD_CLIENT_SECRET", "secret")
    monkeypatch.setattr(cli, "open_retriever", lambda settings: object())
    monkeypatch.setattr(cli, "build_anthropic_client", lambda settings: object())
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs)
    )

    assert cli.main(["serve", "--port", "8123"]) == 0
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 8123
    assert any(route.path == "/api/chat" for route in started["app"].routes)


def test_serve_fails_fast_without_addon_config(monkeypatch, capsys):
    monkeypatch.setenv("PRESTIE_SAVEDVARIABLES", "")

    assert cli.main(["serve"]) == 1
    assert "PRESTIE_SAVEDVARIABLES" in capsys.readouterr().err
