"""Catalog of Icy Veins guide pages to ingest.

`content_type` is the page-level default; the parser may refine it per section
(e.g. the rotation page also contains a mechanics section).
"""

from dataclasses import dataclass

BASE_URL = "https://www.icy-veins.com/wow"


@dataclass(frozen=True)
class GuidePage:
    slug: str
    content_type: str
    wow_class: str = "death-knight"
    spec: str = "blood"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.slug}"


BLOOD_DK_PAGES: tuple[GuidePage, ...] = (
    GuidePage("blood-death-knight-pve-tank-guide", "overview"),
    GuidePage("blood-death-knight-leveling-guide", "leveling"),
    GuidePage("blood-death-knight-pve-tank-rotation-cooldowns-abilities", "rotation"),
    GuidePage("blood-death-knight-pve-tank-stat-priority", "stat_priority"),
    GuidePage("blood-death-knight-pve-tank-spec-builds-talents", "talents"),
    GuidePage("blood-death-knight-pve-tank-spell-summary", "mechanics"),
    GuidePage("blood-death-knight-pve-tank-easy-mode", "beginner"),
    GuidePage("blood-death-knight-pve-tank-mythic-plus-tips", "mythic_plus"),
)
