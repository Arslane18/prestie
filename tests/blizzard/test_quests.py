import json
from datetime import UTC, datetime

import pytest

from prestie.blizzard.client import BlizzardApiError, BlizzardNotFoundError
from prestie.blizzard.quests import (
    Localized,
    QuestCache,
    QuestRepository,
    quest_details_from_api,
)

NOW = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)


def names(en, fr):
    return {"en_US": en, "fr_FR": fr, "de_DE": "ignored"}


# Shape of /data/wow/quest/{id} without a locale parameter (all locales).
PRIEST_QUEST = {
    "_links": {"self": {"href": "https://eu.api.blizzard.com/..."}},
    "id": 58953,
    "title": names("A Priest's End", "La fin d’un prêtre"),
    "area": {"key": {"href": "..."}, "name": names("Exile's Reach", "Confins"), "id": 10424},
    "description": names("Say a few prayers, {name}.", "Priez, {name}."),
    "requirements": {
        "min_character_level": 1,
        "max_character_level": 10,
        "classes": [{"key": {"href": "..."}, "name": names("Priest", "Prêtre"), "id": 5}],
    },
    "rewards": {
        "experience": 1750,
        "money": {"value": 4000, "units": {"gold": 0, "silver": 40, "copper": 0}},
        "spell": {"key": {"href": "..."}, "name": names("Learn Resurrection", "Apprendre Résurrection"), "id": 1},
    },
}

ITEM_QUEST = {
    "id": 55881,
    "title": names("Purge the Totems", "Purge totémique"),
    "requirements": {"min_character_level": 6, "max_character_level": 9},
    "rewards": {
        "experience": 1300,
        "items": {
            "choice_of": [
                {"item": {"name": names("Plate Girdle", "Ceinturon"), "id": 175231}},
                {"item": {"name": names("Cloth Sash", "Écharpe"), "id": 175232}},
            ]
        },
    },
}


def test_parses_titles_zone_description_and_requirements():
    quest = quest_details_from_api(PRIEST_QUEST)

    assert quest.id == 58953
    assert quest.title == Localized(en="A Priest's End", fr="La fin d’un prêtre")
    assert quest.area == Localized(en="Exile's Reach", fr="Confins")
    assert quest.description.fr == "Priez, {name}."
    assert (quest.min_level, quest.max_level) == (1, 10)
    assert quest.classes == (Localized("Priest", "Prêtre"),)


def test_parses_rewards():
    priest = quest_details_from_api(PRIEST_QUEST)
    items = quest_details_from_api(ITEM_QUEST)

    assert priest.experience == 1750
    assert priest.money_copper == 4000
    assert priest.spell == Localized("Learn Resurrection", "Apprendre Résurrection")
    assert items.item_choices == (
        Localized("Plate Girdle", "Ceinturon"),
        Localized("Cloth Sash", "Écharpe"),
    )


def test_minimal_quest_leaves_optional_fields_empty():
    quest = quest_details_from_api({"id": 1, "title": names("T", "T")})

    assert quest.area is None and quest.description is None
    assert quest.min_level is None and quest.classes == ()
    assert quest.experience is None and quest.item_choices == ()


def test_single_locale_strings_are_read_as_french():
    # With ?locale=fr_FR the API returns plain strings instead of locale maps.
    quest = quest_details_from_api({"id": 1, "title": "Titre"})

    assert quest.title == Localized(en=None, fr="Titre")


def test_payload_without_id_is_rejected():
    with pytest.raises(BlizzardApiError, match="id"):
        quest_details_from_api({"title": "x"})


def test_cache_round_trips_the_raw_payload(tmp_path):
    cache = QuestCache(tmp_path)

    assert cache.get("eu", 58953) is None
    cache.put("eu", 58953, PRIEST_QUEST, fetched_at=NOW)

    assert cache.get("eu", 58953) == PRIEST_QUEST
    stored = json.loads((tmp_path / "eu" / "58953.json").read_text(encoding="utf-8"))
    assert stored["fetched_at"] == "2026-09-25T10:00:00+00:00"
    assert cache.get("us", 58953) is None


def test_corrupted_cache_entry_counts_as_missing(tmp_path):
    (tmp_path / "eu").mkdir()
    (tmp_path / "eu" / "1.json").write_text("{not json", encoding="utf-8")

    assert QuestCache(tmp_path).get("eu", 1) is None


class FakeClient:
    region = "eu"

    def __init__(self, payloads=None, error=None):
        self.payloads = payloads or {}
        self.error = error
        self.paths: list[str] = []

    def get_static(self, path):
        self.paths.append(path)
        if self.error:
            raise self.error
        return self.payloads[path]


def test_repository_fetches_once_then_serves_from_cache(tmp_path):
    client = FakeClient({"/data/wow/quest/58953": PRIEST_QUEST})
    repository = QuestRepository(client, QuestCache(tmp_path), now=lambda: NOW)

    first = repository.get(58953)
    second = QuestRepository(client, QuestCache(tmp_path), now=lambda: NOW).get(58953)

    assert first == second
    assert client.paths == ["/data/wow/quest/58953"]


def test_repository_does_not_cache_failures(tmp_path):
    client = FakeClient(error=BlizzardNotFoundError("not found"))
    repository = QuestRepository(client, QuestCache(tmp_path), now=lambda: NOW)

    for _ in range(2):
        with pytest.raises(BlizzardNotFoundError):
            repository.get(404)

    assert len(client.paths) == 2
