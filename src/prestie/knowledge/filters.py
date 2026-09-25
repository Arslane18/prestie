"""Chroma metadata filters shared by the search tool and the retrieval eval."""

from typing import Any

from prestie.catalog import spec_by_key


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
