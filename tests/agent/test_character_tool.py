from datetime import UTC, datetime, timedelta

from prestie.agent.character_tool import CHARACTER_STATE_TOOL, CharacterStateTool
from prestie.character.state import (
    ActiveQuest,
    CharacterState,
    CharacterStateError,
    Quest,
    QuestObjective,
    Spec,
)

CAPTURED_AT = datetime(2026, 9, 24, 19, 30, tzinfo=UTC)


def make_state(**overrides) -> CharacterState:
    base = dict(
        character="Testeur",
        realm="Hyjal",
        level=83,
        class_name="Chevalier de la mort",
        class_token="DEATHKNIGHT",
        spec=Spec(id=250, name="Sang", role="TANK"),
        hero_talent="San'layn",
        active_quest=ActiveQuest(
            id=1234,
            title="Une quête",
            objectives=(
                QuestObjective("Loups tués : 3/8", False, 3, 8),
                QuestObjective("Parler à Bob", True, 1, 1),
            ),
        ),
        quests=(
            Quest(1234, "Une quête", 83, False),
            Quest(99, "Autre", 80, True),
        ),
        captured_at=CAPTURED_AT,
    )
    return CharacterState(**{**base, **overrides})


class FakeSource:
    def __init__(self, state=None, error=None):
        self.state = state or make_state()
        self.error = error

    def latest(self):
        if self.error:
            raise self.error
        return self.state


def run(source, now=CAPTURED_AT + timedelta(minutes=12)):
    return CharacterStateTool(source, now=lambda: now).run({})


def test_definition_takes_no_input():
    assert CHARACTER_STATE_TOOL["name"] == "get_character_state"
    assert CHARACTER_STATE_TOOL["input_schema"] == {
        "type": "object",
        "properties": {},
    }
    assert CharacterStateTool.definition is CHARACTER_STATE_TOOL


def test_description_tells_claude_to_use_it_instead_of_asking():
    description = CHARACTER_STATE_TOOL["description"]

    assert "instead of asking" in description
    assert "/reload" in description


def test_output_lists_character_spec_and_tracked_quest():
    outcome = run(FakeSource())

    assert not outcome.is_error
    for expected in (
        "Testeur (Hyjal)",
        "Level: 83",
        "Chevalier de la mort (DEATHKNIGHT)",
        "Sang (spec id 250, TANK)",
        "Hero talent: San'layn",
        "Tracked quest: [1234] Une quête",
        "- Loups tués : 3/8",
        "- Parler à Bob (done)",
        "[99] Autre (complete)",
        "covered",
    ):
        assert expected in outcome.content, expected


def test_output_states_the_snapshot_age():
    content = run(FakeSource()).content

    assert 'captured_at="2026-09-24T19:30:00Z"' in content
    assert 'age_minutes="12"' in content


def test_output_flags_characters_the_knowledge_base_does_not_cover():
    state = make_state(
        class_name="Prêtresse",
        class_token="PRIEST",
        spec=None,
        hero_talent=None,
        active_quest=None,
        quests=(),
    )

    content = run(FakeSource(state)).content

    assert "Specialization: none" in content
    assert "Hero talent: none" in content
    assert "Tracked quest: none" in content
    assert "not covered" in content
    assert "Blood Death Knight" in content


def test_quest_log_is_capped():
    quests = tuple(Quest(i, f"Q{i}", 80, False) for i in range(1, 60))

    content = run(FakeSource(make_state(quests=quests))).content

    assert "Quest log (59 quests" in content
    assert "[25] Q25" in content
    assert "[26] Q26" not in content


def test_missing_state_is_reported_as_a_tool_error():
    outcome = run(FakeSource(error=CharacterStateError("file not found")))

    assert outcome.is_error
    assert "file not found" in outcome.content
