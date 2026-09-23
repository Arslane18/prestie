"""Command-line entry point: `prestie <command>`."""

import argparse
import sys
import time
from pathlib import Path

from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES
from prestie.ingestion.icy_veins.scraper import (
    IcyVeinsScraper,
    ScrapeError,
    build_client,
    load_robots,
)

DEFAULT_CACHE_DIR = Path("data/raw/icy-veins")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prestie")
    commands = parser.add_subparsers(required=True)

    scrape = commands.add_parser(
        "scrape", help="Download Icy Veins guides into the local cache"
    )
    scrape.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    scrape.add_argument(
        "--force", action="store_true", help="Re-download pages already cached"
    )
    scrape.set_defaults(handler=_scrape)
    return parser


def _scrape(args: argparse.Namespace) -> int:
    with build_client() as client:
        try:
            scraper = IcyVeinsScraper(
                client=client,
                cache=HtmlCache(args.cache_dir),
                robots=load_robots(client),
                sleep=time.sleep,
            )
            for page in BLOOD_DK_PAGES:
                result = scraper.fetch(page, force=args.force)
                status = "cached" if result.from_cache else "downloaded"
                print(f"[{status:>10}] {page.slug}")
        except ScrapeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0
