from datetime import UTC, datetime

import httpx
import pytest

from prestie.ingestion.icy_veins.cache import CachedPage, HtmlCache
from prestie.ingestion.icy_veins.pages import GuidePage
from prestie.ingestion.icy_veins.scraper import (
    IcyVeinsScraper,
    RobotsDisallowedError,
    ScrapeError,
    load_robots,
)

ROBOTS_TXT = "User-agent: *\nAllow: /\nDisallow: /wow/*/modules/\n"
FIXED_NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
PAGE_A = GuidePage(slug="page-a", content_type="overview")
PAGE_B = GuidePage(slug="page-b", content_type="leveling")


class FakeSite:
    """Minimal stand-in for icy-veins.com that records every request."""

    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.requested_paths: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_TXT)
        return httpx.Response(self.status_code, text=f"<html>{request.url.path}</html>")


@pytest.fixture
def site():
    return FakeSite()


@pytest.fixture
def client(site):
    return httpx.Client(transport=httpx.MockTransport(site.handle))


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def make_scraper(client, tmp_path, sleeps):
    def _make(delay_seconds: float = 4.0) -> IcyVeinsScraper:
        return IcyVeinsScraper(
            client=client,
            cache=HtmlCache(tmp_path),
            robots=load_robots(client),
            delay_seconds=delay_seconds,
            sleep=sleeps.append,
            clock=lambda: FIXED_NOW,
        )

    return _make


def test_fetch_downloads_page_and_stores_it_in_cache(make_scraper, tmp_path):
    result = make_scraper().fetch(PAGE_A)

    assert result.from_cache is False
    assert result.cached.html == "<html>/wow/page-a</html>"
    assert result.cached.fetched_at == FIXED_NOW
    assert HtmlCache(tmp_path).get("page-a") == result.cached


def test_fetch_serves_cached_page_without_network(make_scraper, site, tmp_path):
    cached = CachedPage("page-a", PAGE_A.url, "<html>old</html>", FIXED_NOW, 200)
    HtmlCache(tmp_path).put(cached)

    result = make_scraper().fetch(PAGE_A)

    assert result.from_cache is True
    assert result.cached == cached
    assert "/wow/page-a" not in site.requested_paths


def test_force_refetches_even_when_cached(make_scraper, site, tmp_path):
    HtmlCache(tmp_path).put(
        CachedPage("page-a", PAGE_A.url, "<html>old</html>", FIXED_NOW, 200)
    )

    result = make_scraper().fetch(PAGE_A, force=True)

    assert result.from_cache is False
    assert "/wow/page-a" in site.requested_paths


def test_waits_between_network_requests_but_not_before_the_first(make_scraper, sleeps):
    scraper = make_scraper(delay_seconds=4.0)
    scraper.fetch(PAGE_A)
    scraper.fetch(PAGE_B)

    assert sleeps == [4.0]


def test_cache_hits_do_not_trigger_delay(make_scraper, sleeps, tmp_path):
    cache = HtmlCache(tmp_path)
    cache.put(CachedPage("page-a", PAGE_A.url, "<html></html>", FIXED_NOW, 200))
    cache.put(CachedPage("page-b", PAGE_B.url, "<html></html>", FIXED_NOW, 200))

    scraper = make_scraper()
    scraper.fetch(PAGE_A)
    scraper.fetch(PAGE_B)

    assert sleeps == []


def test_refuses_url_disallowed_by_robots(make_scraper, site):
    forbidden = GuidePage(slug="blood-death-knight/modules/x", content_type="overview")

    with pytest.raises(RobotsDisallowedError):
        make_scraper().fetch(forbidden)

    assert not any("modules" in path for path in site.requested_paths)


def test_http_error_raises_and_leaves_cache_untouched(tmp_path, sleeps):
    failing_site = FakeSite(status_code=503)
    client = httpx.Client(transport=httpx.MockTransport(failing_site.handle))
    scraper = IcyVeinsScraper(
        client=client,
        cache=HtmlCache(tmp_path),
        robots=load_robots(client),
        sleep=sleeps.append,
        clock=lambda: FIXED_NOW,
    )

    with pytest.raises(ScrapeError, match="503"):
        scraper.fetch(PAGE_A)

    assert HtmlCache(tmp_path).get("page-a") is None


def test_unreachable_robots_txt_raises_instead_of_scraping_blindly():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )

    with pytest.raises(ScrapeError, match="robots.txt"):
        load_robots(client)


def test_network_failure_is_wrapped_in_scrape_error(tmp_path):
    def boom(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_TXT)
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(boom))
    scraper = IcyVeinsScraper(
        client=client, cache=HtmlCache(tmp_path), robots=load_robots(client)
    )

    with pytest.raises(ScrapeError, match="page-a"):
        scraper.fetch(PAGE_A)
