"""Icy Veins attribution: an answer built on searches always ends with its sources."""

from prestie.agent.agent import TextDelta, TurnFinished
from prestie.agent.attribution import SOURCES_TITLE, missing_sources_section
from prestie.agent.tools import ToolOutcome
from prestie.evaluation.agent_checks import SOURCES_HEADING
from tests.agent.test_agent import (
    FakeClient,
    FakeTool,
    make_agent,
    text_response,
    tool_response,
)

ROTATION = "https://www.icy-veins.com/wow/frost-mage-pve-dps-rotation-cooldowns-abilities"
TALENTS = "https://www.icy-veins.com/wow/frost-mage-pve-dps-spec-builds-talents"


def result(url, title, section="S"):
    return {"section": section, "source_url": url, "title": title, "distance": 0.3}


SEARCH_DETAILS = (
    {
        "queries": ["rotation"],
        "results": [
            result(f"{ROTATION}#single-target", "Frost Mage Rotation Guide"),
            result(f"{ROTATION}#aoe", "Frost Mage Rotation Guide"),
            result(f"{TALENTS}#hero-talents", "Frost Mage Talents"),
        ],
    },
)


def test_the_title_matches_what_the_evaluation_looks_for():
    assert SOURCES_TITLE.lower().startswith(SOURCES_HEADING)


def test_a_missing_section_lists_each_consulted_page_once_in_order():
    section = missing_sources_section("Frostbolt en boucle.", SEARCH_DETAILS)

    assert section == (
        f"\n\n{SOURCES_TITLE}\n"
        f"- Frost Mage Rotation Guide — {ROTATION}\n"
        f"- Frost Mage Talents — {TALENTS}"
    )


def test_an_answer_with_its_sources_is_left_alone():
    answer = f"Frostbolt.\n\nSources (contenu copié d’Icy Veins) :\n- R — {ROTATION}"

    assert missing_sources_section(answer, SEARCH_DETAILS) is None


def test_nothing_to_credit_without_retrieved_passages():
    assert missing_sources_section("Bonjour !", ()) is None
    assert missing_sources_section("?", ({"queries": ["x"], "results": []},)) is None


def test_the_section_name_stands_in_for_a_missing_title():
    details = ({"results": [result(f"{ROTATION}#aoe", None, section="AoE")]},)

    section = missing_sources_section("x", details)

    assert f"- AoE — {ROTATION}" in section


# --- in the agent ------------------------------------------------------------


def searching_agent(answer):
    tool = FakeTool(ToolOutcome("<search_results/>", details=SEARCH_DETAILS[0]))
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "rotation"})),
        text_response(answer),
    )
    return make_agent(client, tool)


def test_the_agent_appends_forgotten_sources_and_says_so():
    agent = searching_agent("Frostbolt en boucle.")

    events = list(agent.ask_stream("Ma rotation ?"))

    [finished] = [e for e in events if isinstance(e, TurnFinished)]
    reply = finished.reply
    assert reply.sources_added
    assert reply.text.startswith("Frostbolt en boucle.\n\nSources")
    streamed = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert streamed.endswith(reply.text[len("Frostbolt en boucle.") :])


def test_the_history_keeps_what_the_model_wrote():
    agent = searching_agent("Frostbolt en boucle.")

    reply = agent.ask("Ma rotation ?")

    [last_block] = reply.messages[-1]["content"]
    assert last_block.text == "Frostbolt en boucle."


def test_an_answer_that_cites_its_sources_is_unchanged():
    answer = f"Frostbolt.\n\n{SOURCES_TITLE}\n- R — {ROTATION}#aoe"

    reply = searching_agent(answer).ask("Ma rotation ?")

    assert reply.text == answer
    assert not reply.sources_added


def test_no_sources_are_added_without_a_search():
    reply = make_agent(FakeClient(text_response("Bonjour !"))).ask("Salut")

    assert reply.text == "Bonjour !"
    assert not reply.sources_added


def test_a_truncated_answer_gets_no_sources():
    tool = FakeTool(ToolOutcome("<search_results/>", details=SEARCH_DETAILS[0]))
    client = FakeClient(
        tool_response(("search_knowledge_base", {"query": "rotation"})),
        text_response("Frostbolt en bou", stop_reason="max_tokens"),
    )

    reply = make_agent(client, tool).ask("Ma rotation ?")

    assert reply.text == "Frostbolt en bou"
    assert not reply.sources_added
