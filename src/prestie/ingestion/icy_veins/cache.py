"""Local cache of raw HTML pages.

Each page is stored as two files so the raw HTML stays easy to open and diff:
  <slug>.html       raw HTML, exactly as served
  <slug>.meta.json  provenance: url, fetch timestamp (ISO-8601 UTC), HTTP status

The metadata file is written last and acts as the "commit" marker: an HTML file
without its sidecar is treated as absent.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CachedPage:
    slug: str
    url: str
    html: str
    fetched_at: datetime
    status_code: int


class HtmlCache:
    def __init__(self, root: Path):
        self._root = root

    def get(self, slug: str) -> CachedPage | None:
        html_path, meta_path = self._paths(slug)
        if not (html_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return CachedPage(
            slug=meta["slug"],
            url=meta["url"],
            html=html_path.read_text(encoding="utf-8"),
            fetched_at=datetime.fromisoformat(meta["fetched_at"]),
            status_code=meta["status_code"],
        )

    def put(self, page: CachedPage) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        html_path, meta_path = self._paths(page.slug)
        meta = {
            "slug": page.slug,
            "url": page.url,
            "fetched_at": page.fetched_at.isoformat(),
            "status_code": page.status_code,
        }
        html_path.write_text(page.html, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _paths(self, slug: str) -> tuple[Path, Path]:
        return self._root / f"{slug}.html", self._root / f"{slug}.meta.json"
