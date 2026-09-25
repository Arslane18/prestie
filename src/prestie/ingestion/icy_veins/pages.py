"""Catalog of Icy Veins guide pages to ingest, generated from the covered specs.

Every spec guide on Icy Veins uses the same slugs:
  {spec}-{class}-pve-{role}-{page}   e.g. shadow-priest-pve-dps-stat-priority
  {spec}-{class}-leveling-guide

`content_type` is the page-level default; the parser may refine it per section
(e.g. the rotation page also contains a mechanics section).
"""

from dataclasses import dataclass

from prestie.catalog import COVERED_SPECS, SpecGuide, spec_by_key

BASE_URL = "https://www.icy-veins.com/wow"
# (slug suffix after "-pve-{role}-", content type)
PVE_PAGES: tuple[tuple[str, str], ...] = (
    ("guide", "overview"),
    ("rotation-cooldowns-abilities", "rotation"),
    ("stat-priority", "stat_priority"),
    ("spec-builds-talents", "talents"),
    ("spell-summary", "mechanics"),
    ("easy-mode", "beginner"),
    ("mythic-plus-tips", "mythic_plus"),
)


@dataclass(frozen=True)
class GuidePage:
    slug: str
    content_type: str
    wow_class: str = "death-knight"
    spec: str = "blood"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.slug}"


def guide_pages(spec: SpecGuide) -> tuple[GuidePage, ...]:
    def page(slug: str, content_type: str) -> GuidePage:
        return GuidePage(slug, content_type, spec.wow_class, spec.spec)

    return (
        page(f"{spec.key}-leveling-guide", "leveling"),
        *(
            page(f"{spec.key}-pve-{spec.role}-{suffix}", content_type)
            for suffix, content_type in PVE_PAGES
        ),
    )


BLOOD_DK_PAGES = guide_pages(spec_by_key("blood-death-knight"))  # type: ignore[arg-type]
ALL_PAGES: tuple[GuidePage, ...] = tuple(
    page for spec in COVERED_SPECS for page in guide_pages(spec)
)
