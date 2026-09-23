from datetime import UTC, datetime

import chromadb
import pytest

from prestie.ingestion.icy_veins.cache import CachedPage, HtmlCache
from prestie.ingestion.icy_veins.pages import GuidePage
from prestie.ingestion.pipeline import IngestError, ingest_pages
from prestie.knowledge.store import KnowledgeStore
from tests.ingestion.icy_veins.html_fixtures import heading, page

PAGE = GuidePage("blood-death-knight-pve-tank-stat-priority", "stat_priority")
FETCHED_AT = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)


class FakeEmbedder:
    model = "fake-model"

    def __init__(self):
        self.document_batches: list[list[str]] = []

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


@pytest.fixture
def cache(tmp_path):
    cache = HtmlCache(tmp_path / "raw")
    html = page(
        heading(2, "1.", "Stat Priority", "stat-priority")
        + "<p>Haste over Mastery.</p>"
        + heading(2, "2.", "Breakdown", "breakdown")
        + "<p>Versatility is fine.</p>"
    )
    cache.put(CachedPage(PAGE.slug, PAGE.url, html, FETCHED_AT, 200))
    return cache


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return KnowledgeStore.open(client, embedding_model=FakeEmbedder.model)


def test_ingest_parses_chunks_embeds_and_stores_each_page(cache, store):
    embedder = FakeEmbedder()

    [report] = ingest_pages([PAGE], cache, embedder, store)

    assert (report.slug, report.sections, report.chunks) == (PAGE.slug, 2, 2)
    assert store.count() == 2
    assert len(embedder.document_batches) == 1
    assert embedder.document_batches[0][0].startswith("Blood Death Knight")


def test_reingesting_is_idempotent(cache, store):
    ingest_pages([PAGE], cache, FakeEmbedder(), store)
    ingest_pages([PAGE], cache, FakeEmbedder(), store)

    assert store.count() == 2


def test_stored_chunks_keep_attribution_metadata(cache, store):
    ingest_pages([PAGE], cache, FakeEmbedder(), store)

    [hit] = store.search([1.0, 1.0], n_results=1, where={"section": "Breakdown"})

    assert hit.metadata["attribution"].startswith("Copied from Icy Veins")
    assert hit.metadata["source_url"] == f"{PAGE.url}#breakdown"


def test_uncached_page_raises_with_hint_to_scrape(tmp_path, store):
    with pytest.raises(IngestError, match="prestie scrape"):
        ingest_pages([PAGE], HtmlCache(tmp_path / "empty"), FakeEmbedder(), store)
