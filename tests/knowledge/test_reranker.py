from types import SimpleNamespace

import pytest
import voyageai.error

from prestie.knowledge.embeddings import EmbeddingError
from prestie.knowledge.reranker import VoyageReranker
from prestie.knowledge.store import SearchHit


def hit(id_: str) -> SearchHit:
    return SearchHit(id=id_, text=f"text of {id_}", metadata={}, distance=0.4)


class FakeVoyage:
    def __init__(self, order, error=None):
        self.order = order  # indexes of documents, best first
        self.error = error
        self.calls: list[dict] = []

    def rerank(self, query, documents, model, top_k=None, truncation=True):
        self.calls.append(
            {"query": query, "documents": documents, "model": model, "top_k": top_k}
        )
        if self.error:
            raise self.error
        results = [
            SimpleNamespace(index=i, relevance_score=1 - rank / 10)
            for rank, i in enumerate(self.order[:top_k])
        ]
        return SimpleNamespace(results=results, total_tokens=42)


def test_reranking_reorders_hits_by_relevance_and_keeps_top_k():
    client = FakeVoyage(order=[2, 0, 1])
    reranker = VoyageReranker(client, model="rerank-2.5")

    ranked = reranker.rerank("stats?", [hit("a"), hit("b"), hit("c")], top_k=2)

    assert [h.id for h in ranked] == ["c", "a"]
    assert client.calls[0]["documents"] == ["text of a", "text of b", "text of c"]
    assert client.calls[0]["model"] == "rerank-2.5"


def test_no_candidates_means_no_call():
    client = FakeVoyage(order=[])

    assert VoyageReranker(client, model="m").rerank("q", [], top_k=3) == []
    assert client.calls == []


def test_voyage_failures_are_reported_like_embedding_failures():
    client = FakeVoyage(order=[], error=voyageai.error.RateLimitError("slow down"))

    with pytest.raises(EmbeddingError, match="rerank"):
        VoyageReranker(client, model="m").rerank("q", [hit("a")], top_k=1)
