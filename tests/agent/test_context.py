from types import SimpleNamespace

from prestie.agent.context import compact_history

SEARCH_OUTPUT = (
    '<search_results query="stat priority">\n'
    '<result index="1" section="Stat Priority &gt; Deathbringer"'
    ' source_url="https://www.icy-veins.com/wow/blood#deathbringer">\n'
    "Strength > Critical Strike > Mastery\n</result>\n"
    '<result index="2" section="Easy Mode"'
    ' source_url="https://www.icy-veins.com/wow/blood-easy">\n'
    "Long passage text\n</result>\n"
    "</search_results>"
)
STATE_OUTPUT = (
    '<character_state captured_at="2026-09-26T10:00:00Z" age_minutes="5">\n'
    "Hero talent: San'layn\n</character_state>"
)
QUEST_OUTPUT = '<quest_details id="55763">\nTitle: Le sauvetage\n</quest_details>'


def tool_use(block_id, name):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input={})


def result(block_id, content, **extra):
    return {"type": "tool_result", "tool_use_id": block_id, "content": content, **extra}


def turn(*calls):
    """One finished turn: question, tool calls and results, final answer."""
    uses = [tool_use(f"t{i}", name) for i, (name, _) in enumerate(calls)]
    results = [result(f"t{i}", output) for i, (_, output) in enumerate(calls)]
    answer = {"role": "assistant", "content": [SimpleNamespace(type="text", text="A")]}
    return (
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": uses},
        {"role": "user", "content": results},
        answer,
    )


def compacted_results(history):
    return [
        block
        for message in compact_history(history)
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
    ]


def test_search_passages_are_replaced_by_the_sections_they_came_from():
    [compacted] = compacted_results(turn(("search_knowledge_base", SEARCH_OUTPUT)))

    content = compacted["content"]
    assert compacted["tool_use_id"] == "t0"
    assert "Long passage text" not in content
    assert 'query="stat priority"' in content
    assert "Stat Priority &gt; Deathbringer" in content
    assert "https://www.icy-veins.com/wow/blood#deathbringer" in content
    assert "https://www.icy-veins.com/wow/blood-easy" in content
    assert "search again" in content.lower()


def test_character_state_is_replaced_by_a_reminder_to_read_it_again():
    [compacted] = compacted_results(turn(("get_character_state", STATE_OUTPUT)))

    assert "San'layn" not in compacted["content"]
    assert "get_character_state" in compacted["content"]
    assert "/reload" in compacted["content"]


def test_small_static_results_and_errors_are_kept():
    history = turn(
        ("get_quest_details", QUEST_OUTPUT),
        ("search_knowledge_base", "No results in the knowledge base for: x"),
    )
    errored = (
        *history[:2],
        {
            "role": "user",
            "content": [
                result("t0", QUEST_OUTPUT),
                result("t1", "Search failed: boom", is_error=True),
            ],
        },
        history[3],
    )

    assert [r["content"] for r in compacted_results(history)] == [
        QUEST_OUTPUT,
        "No results in the knowledge base for: x",
    ]
    assert compacted_results(errored)[1] == result(
        "t1", "Search failed: boom", is_error=True
    )


def test_questions_and_assistant_blocks_are_kept_as_they_are():
    history = turn(("search_knowledge_base", SEARCH_OUTPUT))

    compacted = compact_history(history)

    assert compacted[0] == history[0]
    assert compacted[1] is history[1]
    assert compacted[3] is history[3]


def test_compaction_does_not_mutate_the_history():
    history = turn(("search_knowledge_base", SEARCH_OUTPUT))

    compact_history(history)

    assert history[2]["content"][0]["content"] == SEARCH_OUTPUT


def test_compaction_is_deterministic_and_idempotent():
    # The cached prefix must stay byte-identical from one request to the next.
    history = turn(
        ("search_knowledge_base", SEARCH_OUTPUT),
        ("get_character_state", STATE_OUTPUT),
    )

    once = compact_history(history)

    assert compact_history(history) == once
    assert compact_history(once) == once
