"""Offline ingestion: cached HTML → sections → chunks → vectors → vector store.

Runs from the local cache only; scraping is a separate step (`prestie scrape`)
so re-indexing never hits Icy Veins again.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from prestie.ingestion.chunking import chunk_guide
from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.ingestion.icy_veins.pages import GuidePage
from prestie.ingestion.icy_veins.parser import parse_guide
from prestie.knowledge.embeddings import Embedder
from prestie.knowledge.store import KnowledgeStore


class IngestError(Exception):
    """A page could not be ingested."""


@dataclass(frozen=True)
class PageReport:
    slug: str
    sections: int
    chunks: int


def ingest_pages(
    pages: Iterable[GuidePage],
    cache: HtmlCache,
    embedder: Embedder,
    store: KnowledgeStore,
) -> tuple[PageReport, ...]:
    return tuple(_ingest_page(page, cache, embedder, store) for page in pages)


def _ingest_page(
    page: GuidePage, cache: HtmlCache, embedder: Embedder, store: KnowledgeStore
) -> PageReport:
    cached = cache.get(page.slug)
    if cached is None:
        raise IngestError(f"{page.slug} is not cached yet: run `prestie scrape` first")

    guide = parse_guide(page, cached)
    chunks = chunk_guide(guide)
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    store.replace_page(
        page.slug,
        ids=[chunk.id for chunk in chunks],
        texts=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return PageReport(slug=page.slug, sections=len(guide.sections), chunks=len(chunks))
