"""Turn Icy Veins' conditional CSS classes into human-readable labels.

Some guide content is shown or hidden by JavaScript depending on reader input:
  - leveling slider:  `bdk_level_71_90` (levels 71 to 90), `bdk_level_27` (27+)
  - rotation presets: `rotation_line_preset-1_on` / `_off` (hero talent choice)
  - talent switches:  `rotation_line_talent-2_on` (optional talent picked)

Static HTML contains every variant at once, so without these labels a chunk
would silently mix, say, the Deathbringer and San'layn rotations. The preset
and talent names are read from the page's own selector widgets.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from bs4 import Tag

LEVEL_CLASS = re.compile(r"^[a-z]+_level_(\d+)(?:_(\d+))?$")
ROTATION_CLASS = re.compile(r"^rotation_line_(.+)_(on|off)$")
TALENT_KEY_PREFIX = "talent-"
SWITCH_ID_PREFIX = "rotation_switch_"


def _clean(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class ConditionLabels:
    names: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_content(cls, content: Tag) -> "ConditionLabels":
        return cls(
            MappingProxyType({**_preset_names(content), **_switch_names(content)})
        )

    def describe(self, element: Tag) -> str | None:
        classes = element.get("class") or []
        rotation = [m for c in classes if (m := ROTATION_CLASS.match(c))]
        enabled = [m[1] for m in rotation if m[2] == "on"]
        # Presets are radio choices: "only A" already implies "not B".
        disabled = [] if enabled else [m[1] for m in rotation if m[2] == "off"]

        labels = [
            *(_level_label(m) for c in classes if (m := LEVEL_CLASS.match(c))),
            *(self._enabled_label(key) for key in enabled),
            *(self._disabled_label(key) for key in disabled),
        ]
        return f"[{', '.join(labels)}]" if labels else None

    def _enabled_label(self, key: str) -> str:
        name = self.names.get(key, key)
        return f"With {name}" if key.startswith(TALENT_KEY_PREFIX) else f"{name} only"

    def _disabled_label(self, key: str) -> str:
        name = self.names.get(key, key)
        return f"Without {name}" if key.startswith(TALENT_KEY_PREFIX) else f"Not {name}"


def _level_label(match: re.Match[str]) -> str:
    low, high = match.groups()
    return f"Levels {low}-{high}" if high else f"Level {low}+"


def _preset_names(content: Tag) -> dict[str, str]:
    return {
        tokens[0]: _clean(header.get_text(" "))
        for group in content.select(".rotation_presets")
        if (header := group.select_one(".rotation_presets_header")) is not None
        for item in group.select(".rotation_preset_item")
        if (tokens := str(item.get("data-preset", "")).split())
    }


def _switch_names(content: Tag) -> dict[str, str]:
    return {
        switch["id"].removeprefix(SWITCH_ID_PREFIX): _clean(switch.get_text(" "))
        for switch in content.select(f".rotation_switch[id^={SWITCH_ID_PREFIX}]")
    }
