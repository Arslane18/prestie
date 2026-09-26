"""The specializations the knowledge base covers: the single source of truth.

Adding a spec here makes the ingestion scrape and index its guide pages,
lets the agent filter searches on it, and marks characters playing it as
covered. `key` ("discipline-priest") is the slug prefix Icy Veins uses and
the value of the search tool's `spec` filter; `spec_id` and `class_token`
("PRIEST") are what the addon exports.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecGuide:
    wow_class: str  # Icy Veins slug form: "death-knight", "priest"
    class_token: str  # the game's class file token: "DEATHKNIGHT", "PRIEST"
    spec: str  # "blood", "discipline"
    role: str  # Icy Veins page role: "tank", "healing", "dps"
    spec_id: int  # in-game specialization id
    name: str  # English name, as in the guides

    @property
    def key(self) -> str:
        return f"{self.spec}-{self.wow_class}"


# Every playable spec (Midnight), grouped by class. Ids and roles checked
# against the Blizzard Game Data API (/data/wow/playable-specialization).
COVERED_SPECS: tuple[SpecGuide, ...] = (
    SpecGuide(
        "death-knight", "DEATHKNIGHT", "blood", "tank", 250, "Blood Death Knight"
    ),
    SpecGuide("death-knight", "DEATHKNIGHT", "frost", "dps", 251, "Frost Death Knight"),
    SpecGuide(
        "death-knight", "DEATHKNIGHT", "unholy", "dps", 252, "Unholy Death Knight"
    ),
    SpecGuide("demon-hunter", "DEMONHUNTER", "havoc", "dps", 577, "Havoc Demon Hunter"),
    SpecGuide(
        "demon-hunter", "DEMONHUNTER", "vengeance", "tank", 581, "Vengeance Demon Hunter"
    ),
    SpecGuide(
        "demon-hunter", "DEMONHUNTER", "devourer", "dps", 1480, "Devourer Demon Hunter"
    ),
    SpecGuide("druid", "DRUID", "balance", "dps", 102, "Balance Druid"),
    SpecGuide("druid", "DRUID", "feral", "dps", 103, "Feral Druid"),
    SpecGuide("druid", "DRUID", "guardian", "tank", 104, "Guardian Druid"),
    SpecGuide("druid", "DRUID", "restoration", "healing", 105, "Restoration Druid"),
    SpecGuide("evoker", "EVOKER", "devastation", "dps", 1467, "Devastation Evoker"),
    SpecGuide(
        "evoker", "EVOKER", "preservation", "healing", 1468, "Preservation Evoker"
    ),
    SpecGuide("evoker", "EVOKER", "augmentation", "dps", 1473, "Augmentation Evoker"),
    SpecGuide("hunter", "HUNTER", "beast-mastery", "dps", 253, "Beast Mastery Hunter"),
    SpecGuide("hunter", "HUNTER", "marksmanship", "dps", 254, "Marksmanship Hunter"),
    SpecGuide("hunter", "HUNTER", "survival", "dps", 255, "Survival Hunter"),
    SpecGuide("mage", "MAGE", "arcane", "dps", 62, "Arcane Mage"),
    SpecGuide("mage", "MAGE", "fire", "dps", 63, "Fire Mage"),
    SpecGuide("mage", "MAGE", "frost", "dps", 64, "Frost Mage"),
    SpecGuide("monk", "MONK", "brewmaster", "tank", 268, "Brewmaster Monk"),
    SpecGuide("monk", "MONK", "windwalker", "dps", 269, "Windwalker Monk"),
    SpecGuide("monk", "MONK", "mistweaver", "healing", 270, "Mistweaver Monk"),
    SpecGuide("paladin", "PALADIN", "holy", "healing", 65, "Holy Paladin"),
    SpecGuide("paladin", "PALADIN", "protection", "tank", 66, "Protection Paladin"),
    SpecGuide("paladin", "PALADIN", "retribution", "dps", 70, "Retribution Paladin"),
    SpecGuide("priest", "PRIEST", "discipline", "healing", 256, "Discipline Priest"),
    SpecGuide("priest", "PRIEST", "holy", "healing", 257, "Holy Priest"),
    SpecGuide("priest", "PRIEST", "shadow", "dps", 258, "Shadow Priest"),
    SpecGuide("rogue", "ROGUE", "assassination", "dps", 259, "Assassination Rogue"),
    SpecGuide("rogue", "ROGUE", "outlaw", "dps", 260, "Outlaw Rogue"),
    SpecGuide("rogue", "ROGUE", "subtlety", "dps", 261, "Subtlety Rogue"),
    SpecGuide("shaman", "SHAMAN", "elemental", "dps", 262, "Elemental Shaman"),
    SpecGuide("shaman", "SHAMAN", "enhancement", "dps", 263, "Enhancement Shaman"),
    SpecGuide("shaman", "SHAMAN", "restoration", "healing", 264, "Restoration Shaman"),
    SpecGuide("warlock", "WARLOCK", "affliction", "dps", 265, "Affliction Warlock"),
    SpecGuide("warlock", "WARLOCK", "demonology", "dps", 266, "Demonology Warlock"),
    SpecGuide("warlock", "WARLOCK", "destruction", "dps", 267, "Destruction Warlock"),
    SpecGuide("warrior", "WARRIOR", "arms", "dps", 71, "Arms Warrior"),
    SpecGuide("warrior", "WARRIOR", "fury", "dps", 72, "Fury Warrior"),
    SpecGuide("warrior", "WARRIOR", "protection", "tank", 73, "Protection Warrior"),
)

_BY_ID = {spec.spec_id: spec for spec in COVERED_SPECS}
_BY_KEY = {spec.key: spec for spec in COVERED_SPECS}


def spec_by_id(spec_id: int) -> SpecGuide | None:
    return _BY_ID.get(spec_id)


def spec_by_key(key: str) -> SpecGuide | None:
    return _BY_KEY.get(key)


def specs_for_class(class_token: str) -> tuple[SpecGuide, ...]:
    return tuple(spec for spec in COVERED_SPECS if spec.class_token == class_token)
