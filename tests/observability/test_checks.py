from prestie.observability.checks import online_flags

URL = "https://www.icy-veins.com/wow/shadow-priest-pve-dps-rotation-cooldowns-abilities"
SOURCED = f"Fade.\n\nSources (contenu copié d'Icy Veins) :\n- Fade Pulling — {URL}#fade"


def search(*urls, spec="shadow-priest", error=False):
    return {
        "name": "search_knowledge_base",
        "input": {"query": "q", **({"spec": spec} if spec else {})},
        "is_error": error,
        "duration_s": 0.3,
        "details": {"queries": ["q"], "results": [{"source_url": u} for u in urls]},
    }


def turn(answer=SOURCED, tools=(), outcome="answered", spec_id=258):
    return {
        "outcome": outcome,
        "answer": answer,
        "character": {"spec_id": spec_id} if spec_id else None,
        "tools": list(tools),
    }


def test_a_clean_sourced_answer_raises_no_flag():
    assert online_flags(turn(tools=[search(f"{URL}#fade")])) == ()


def test_citing_the_page_of_a_retrieved_passage_is_fine():
    page_only = f"Fade.\n\nSources (contenu copié d'Icy Veins) :\n- Rotation — {URL}"

    assert online_flags(turn(answer=page_only, tools=[search(f"{URL}#fade")])) == ()


def test_citing_another_section_than_the_retrieved_one_is_flagged():
    flags = online_flags(turn(tools=[search(f"{URL}#other-anchor")]))

    assert flags == ("invalid_citation",)


def test_a_citation_that_no_search_returned_is_flagged():
    tools = [search("https://www.icy-veins.com/wow/holy-priest-pve-healing-guide")]

    assert online_flags(turn(tools=tools)) == ("invalid_citation",)


def test_searching_then_answering_without_sources_is_flagged():
    flags = online_flags(turn(answer="Fade.", tools=[search(f"{URL}#fade")]))

    assert flags == ("missing_sources",)


def test_answers_without_a_search_need_no_sources():
    assert online_flags(turn(answer="Avec plaisir !")) == ()


def test_an_unfiltered_search_is_flagged_when_the_spec_was_known():
    unfiltered = search(f"{URL}#fade", spec=None)

    assert online_flags(turn(tools=[unfiltered])) == ("unfiltered_search",)
    assert online_flags(turn(tools=[unfiltered], spec_id=None)) == ()


def test_tool_errors_and_unanswered_turns_are_flagged():
    flags = online_flags(turn(answer="", tools=[search(error=True)], outcome="error"))

    assert flags == ("tool_error", "not_answered")


def test_sources_added_by_the_backend_are_flagged():
    # The model forgot them: the answer is attributed, but the prompt missed.
    flags = online_flags({**turn(tools=[search(f"{URL}#fade")]), "sources_added": True})

    assert flags == ("sources_added",)
