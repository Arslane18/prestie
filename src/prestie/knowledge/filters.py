"""Chroma metadata filters shared by the search tool and the retrieval eval."""

from collections.abc import Mapping
from typing import Any

from prestie.catalog import COVERED_SPECS, SpecGuide, spec_by_key


def metadata_filter(
    spec_key: str | None = None, content_type: str | None = None
) -> dict[str, Any] | None:
    """Filter on a covered spec and/or a page type; several conditions need $and."""
    guide = spec_by_key(spec_key) if spec_key else None
    conditions = [
        *([{"wow_class": guide.wow_class}, {"spec": guide.spec}] if guide else []),
        *([{"content_type": content_type}] if content_type else []),
    ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def filtered_spec(where: Mapping[str, Any] | None) -> SpecGuide | None:
    """The spec a filter built by `metadata_filter` restricts to, if any."""
    if not where:
        return None
    conditions = where.get("$and", [where])
    fields = {key: value for condition in conditions for key, value in condition.items()}
    return next(
        (
            spec
            for spec in COVERED_SPECS
            if (spec.wow_class, spec.spec) == (fields.get("wow_class"), fields.get("spec"))
        ),
        None,
    )
