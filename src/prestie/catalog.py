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


COVERED_SPECS: tuple[SpecGuide, ...] = (
    SpecGuide("death-knight", "DEATHKNIGHT", "blood", "tank", 250, "Blood Death Knight"),
    SpecGuide("priest", "PRIEST", "discipline", "healing", 256, "Discipline Priest"),
    SpecGuide("priest", "PRIEST", "holy", "healing", 257, "Holy Priest"),
    SpecGuide("priest", "PRIEST", "shadow", "dps", 258, "Shadow Priest"),
)

_BY_ID = {spec.spec_id: spec for spec in COVERED_SPECS}
_BY_KEY = {spec.key: spec for spec in COVERED_SPECS}


def spec_by_id(spec_id: int) -> SpecGuide | None:
    return _BY_ID.get(spec_id)


def spec_by_key(key: str) -> SpecGuide | None:
    return _BY_KEY.get(key)


def specs_for_class(class_token: str) -> tuple[SpecGuide, ...]:
    return tuple(spec for spec in COVERED_SPECS if spec.class_token == class_token)
