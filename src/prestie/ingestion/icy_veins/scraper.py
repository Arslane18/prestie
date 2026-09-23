"""Polite Icy Veins scraper: cache-first, robots.txt-aware, rate-limited.

Icy Veins allows non-commercial reuse with attribution, which
does not exempt us from being a good citizen: every page is fetched at most once
unless explicitly forced, and consecutive network requests are spaced out.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.robotparser import RobotFileParser

import httpx

from prestie.ingestion.icy_veins.cache import CachedPage, HtmlCache
from prestie.ingestion.icy_veins.pages import GuidePage

ROBOTS_URL = "https://www.icy-veins.com/robots.txt"
USER_AGENT = "prestie/0.1 (personal non-commercial WoW helper project)"
DEFAULT_DELAY_SECONDS = 4.0
DEFAULT_TIMEOUT_SECONDS = 30.0


class ScrapeError(Exception):
    """A page could not be downloaded."""


class RobotsDisallowedError(ScrapeError):
    """robots.txt forbids fetching this URL."""


@dataclass(frozen=True)
class ScrapeResult:
    page: GuidePage
    cached: CachedPage
    from_cache: bool


def build_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


def load_robots(client: httpx.Client, robots_url: str = ROBOTS_URL) -> RobotFileParser:
    try:
        response = client.get(robots_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ScrapeError(
            f"Could not load robots.txt from {robots_url}: {exc}"
        ) from exc
    robots = RobotFileParser(robots_url)
    robots.parse(response.text.splitlines())
    return robots


class IcyVeinsScraper:
    def __init__(
        self,
        client: httpx.Client,
        cache: HtmlCache,
        robots: RobotFileParser,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._client = client
        self._cache = cache
        self._robots = robots
        self._delay_seconds = delay_seconds
        self._sleep = sleep
        self._clock = clock
        self._has_requested = False

    def fetch(self, page: GuidePage, *, force: bool = False) -> ScrapeResult:
        if not force:
            cached = self._cache.get(page.slug)
            if cached is not None:
                return ScrapeResult(page=page, cached=cached, from_cache=True)

        cached = self._download(page)
        self._cache.put(cached)
        return ScrapeResult(page=page, cached=cached, from_cache=False)

    def _download(self, page: GuidePage) -> CachedPage:
        if not self._robots.can_fetch(USER_AGENT, page.url):
            raise RobotsDisallowedError(f"robots.txt disallows {page.url}")

        self._wait_before_request()
        try:
            response = self._client.get(page.url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ScrapeError(
                f"{page.slug}: HTTP {exc.response.status_code} for {page.url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ScrapeError(
                f"{page.slug}: request to {page.url} failed: {exc}"
            ) from exc

        return CachedPage(
            slug=page.slug,
            url=str(response.url),
            html=response.text,
            fetched_at=self._clock(),
            status_code=response.status_code,
        )

    def _wait_before_request(self) -> None:
        if self._has_requested:
            self._sleep(self._delay_seconds)
        self._has_requested = True
