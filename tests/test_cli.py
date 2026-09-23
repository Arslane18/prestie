import httpx
import pytest

from prestie import cli
from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES

ROBOTS_TXT = "User-agent: *\nAllow: /\n"


def fake_client(status_code: int = 200) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_TXT)
        return httpx.Response(status_code, text="<html></html>")

    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture(autouse=True)
def no_real_network_or_sleep(monkeypatch):
    monkeypatch.setattr(cli, "build_client", fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)


def test_scrape_caches_every_mvp_page(tmp_path, capsys):
    exit_code = cli.main(["scrape", "--cache-dir", str(tmp_path)])

    cache = HtmlCache(tmp_path)
    assert exit_code == 0
    assert all(cache.get(page.slug) is not None for page in BLOOD_DK_PAGES)
    assert "downloaded" in capsys.readouterr().out


def test_second_scrape_is_served_from_cache(tmp_path, capsys):
    cli.main(["scrape", "--cache-dir", str(tmp_path)])
    capsys.readouterr()

    cli.main(["scrape", "--cache-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "downloaded" not in out
    assert out.count("cached") == len(BLOOD_DK_PAGES)


def test_scrape_reports_failure_with_non_zero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_client", lambda: fake_client(status_code=503))

    exit_code = cli.main(["scrape", "--cache-dir", str(tmp_path)])

    assert exit_code == 1
    assert "503" in capsys.readouterr().err
