from datetime import UTC, datetime

from prestie.ingestion.icy_veins.cache import CachedPage, HtmlCache


def make_page(slug: str = "some-guide") -> CachedPage:
    return CachedPage(
        slug=slug,
        url=f"https://www.icy-veins.com/wow/{slug}",
        html="<html><body>Death Strike</body></html>",
        fetched_at=datetime(2026, 9, 23, 12, 0, tzinfo=UTC),
        status_code=200,
    )


def test_get_returns_none_when_page_was_never_cached(tmp_path):
    cache = HtmlCache(tmp_path)

    assert cache.get("unknown") is None


def test_put_then_get_round_trips_html_and_metadata(tmp_path):
    cache = HtmlCache(tmp_path)
    page = make_page()

    cache.put(page)

    assert cache.get("some-guide") == page


def test_cache_persists_across_instances(tmp_path):
    HtmlCache(tmp_path).put(make_page())

    assert HtmlCache(tmp_path).get("some-guide") == make_page()


def test_put_writes_raw_html_and_json_sidecar(tmp_path):
    HtmlCache(tmp_path).put(make_page())

    assert (
        (tmp_path / "some-guide.html").read_text(encoding="utf-8").startswith("<html>")
    )
    assert (tmp_path / "some-guide.meta.json").exists()


def test_get_ignores_html_without_metadata(tmp_path):
    (tmp_path / "orphan.html").write_text("<html></html>", encoding="utf-8")

    assert HtmlCache(tmp_path).get("orphan") is None
