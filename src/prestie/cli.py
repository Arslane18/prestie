"""Command-line entry point: `prestie <command>`."""

import argparse
import sys
import time
from pathlib import Path

import chromadb

from prestie.config import ConfigError, Settings, load_settings
from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES
from prestie.ingestion.icy_veins.parser import ParseError
from prestie.ingestion.icy_veins.scraper import (
    IcyVeinsScraper,
    ScrapeError,
    build_client,
    load_robots,
)
from prestie.ingestion.pipeline import IngestError, ingest_pages
from prestie.knowledge.embeddings import Embedder, EmbeddingError, VoyageEmbedder
from prestie.knowledge.store import (
    DEFAULT_RESULTS,
    EmbeddingModelMismatchError,
    KnowledgeStore,
    SearchHit,
)

DEFAULT_CACHE_DIR = Path("data/raw/icy-veins")
SNIPPET_CHARS = 300
KNOWLEDGE_ERRORS = (
    ConfigError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    IngestError,
    ParseError,
)


def build_voyage_embedder(settings: Settings) -> Embedder:
    return VoyageEmbedder.from_api_key(
        settings.require_voyage_api_key(), settings.voyage_model
    )


build_embedder = build_voyage_embedder  # seam replaced by a fake in tests


def open_store(settings: Settings, *, reset: bool = False) -> KnowledgeStore:
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return KnowledgeStore.open(client, settings.voyage_model, reset=reset)


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

    ingest = commands.add_parser(
        "ingest", help="Parse, chunk, embed and index the cached guides"
    )
    ingest.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ingest.add_argument(
        "--reset", action="store_true", help="Drop the index and rebuild it"
    )
    ingest.set_defaults(handler=_ingest)

    search = commands.add_parser(
        "search", help="Query the knowledge base (retrieval only, no LLM)"
    )
    search.add_argument("query")
    search.add_argument("-k", type=int, default=DEFAULT_RESULTS, dest="n_results")
    search.add_argument("--content-type", help="e.g. rotation, stat_priority")
    search.set_defaults(handler=_search)
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


def _ingest(args: argparse.Namespace) -> int:
    try:
        settings = load_settings()
        embedder = build_embedder(settings)
        store = open_store(settings, reset=args.reset)
        reports = ingest_pages(
            BLOOD_DK_PAGES, HtmlCache(args.cache_dir), embedder, store
        )
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for report in reports:
        print(f"{report.slug}: {report.sections} sections -> {report.chunks} chunks")
    print(f"{store.count()} chunks indexed with {settings.voyage_model}")
    return 0


def _search(args: argparse.Namespace) -> int:
    where = {"content_type": args.content_type} if args.content_type else None
    try:
        settings = load_settings()
        store = open_store(settings)
        query_vector = build_embedder(settings).embed_query(args.query)
        hits = store.search(query_vector, n_results=args.n_results, where=where)
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not hits:
        print("No results (is the index empty? run `prestie ingest`).")
    for rank, hit in enumerate(hits, start=1):
        print(_format_hit(rank, hit))
    return 0


def _format_hit(rank: int, hit: SearchHit) -> str:
    body = " ".join(hit.text.split("\n\n", 1)[-1].split())
    snippet = body[:SNIPPET_CHARS] + ("…" if len(body) > SNIPPET_CHARS else "")
    part_count = hit.metadata.get("part_count", 1)
    part = f" (part {hit.metadata['part'] + 1}/{part_count})" if part_count > 1 else ""
    return (
        f"[{rank}] distance={hit.distance:.3f} | {hit.metadata['section']}{part}\n"
        f"    {hit.metadata['source_url']}\n"
        f"    {snippet}\n"
    )
