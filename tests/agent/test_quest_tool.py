import pytest

from prestie.agent.quest_tool import QUEST_DETAILS_TOOL, QuestDetailsTool
from prestie.blizzard.client import BlizzardApiError, BlizzardNotFoundError
from prestie.blizzard.quests import Localized, QuestDetails


def make_quest(**overrides) -> QuestDetails:
    base = dict(
        id=58953,
        title=Localized("A Priest's End", "La fin d’un prêtre"),
        area=Localized("Exile's Reach", "Confins de l’Exil"),
        description=Localized(None, "Vous êtes un prêtre {name} ?\r\n\r\nPriez."),
        min_level=1,
        max_level=10,
        classes=(Localized("Priest", "Prêtre"),),
        experience=1750,
        money_copper=41_020,
        item_choices=(Localized("Plate Girdle", "Ceinturon"),),
        spell=Localized("Learn Resurrection", "Apprendre Résurrection"),
    )
    return QuestDetails(**{**base, **overrides})


class FakeRepository:
    def __init__(self, quest=None, error=None):
        self.quest = quest or make_quest()
        self.error = error
        self.ids: list[int] = []

    def get(self, quest_id):
        self.ids.append(quest_id)
        if self.error:
            raise self.error
        return self.quest


def test_definition_requires_a_quest_id():
    schema = QUEST_DETAILS_TOOL["input_schema"]

    assert QUEST_DETAILS_TOOL["name"] == "get_quest_details"
    assert schema["required"] == ["quest_id"]
    assert schema["properties"]["quest_id"]["type"] == "integer"
    assert QuestDetailsTool.definition is QUEST_DETAILS_TOOL


def test_description_says_what_the_api_does_not_cover():
    assert "walkthrough" in QUEST_DETAILS_TOOL["description"]


def test_output_lists_facts_in_both_languages():
    repository = FakeRepository()

    outcome = QuestDetailsTool(repository).run({"quest_id": 58953})

    assert repository.ids == [58953]
    assert not outcome.is_error
    for expected in (
        '<quest_details id="58953"',
        "Title: La fin d’un prêtre (English: A Priest's End)",
        "Zone: Confins de l’Exil (English: Exile's Reach)",
        "Level range: 1-10",
        "Classes: Prêtre",
        "1750 XP",
        "4 g 10 s 20 c",
        "Apprendre Résurrection",
        "choice of: Ceinturon",
        "Not included:",
    ):
        assert expected in outcome.content, expected


def test_description_is_normalized():
    content = QuestDetailsTool(FakeRepository()).run({"quest_id": 1}).content

    assert "Vous êtes un prêtre [character name] ?\n\nPriez." in content
    assert "\r" not in content


def test_sparse_quest_omits_missing_lines():
    quest = make_quest(
        area=None,
        description=None,
        min_level=None,
        max_level=None,
        classes=(),
        experience=None,
        money_copper=None,
        item_choices=(),
        spell=None,
    )

    content = QuestDetailsTool(FakeRepository(quest)).run({"quest_id": 1}).content

    assert "Zone" not in content
    assert "Level range" not in content
    assert "Rewards: none listed" in content


@pytest.mark.parametrize("quest_id", [None, "55763", 0, -3, True, 2**40])
def test_invalid_ids_are_rejected_without_calling_the_api(quest_id):
    repository = FakeRepository()

    outcome = QuestDetailsTool(repository).run({"quest_id": quest_id})

    assert outcome.is_error
    assert "quest_id" in outcome.content
    assert repository.ids == []


def test_unknown_quest_is_an_error_result():
    tool = QuestDetailsTool(FakeRepository(error=BlizzardNotFoundError("404")))

    outcome = tool.run({"quest_id": 999})

    assert outcome.is_error
    assert "999 not found" in outcome.content


def test_api_failures_are_error_results():
    tool = QuestDetailsTool(FakeRepository(error=BlizzardApiError("rate limit")))

    outcome = tool.run({"quest_id": 1})

    assert outcome.is_error
    assert "rate limit" in outcome.content
