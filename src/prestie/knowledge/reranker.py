"""Second-stage ranking: re-score retrieved candidates with a cross-encoder.

Vector search is a "bi-encoder": the question and each chunk are embedded
separately, then compared. It is fast and scales to the whole corpus, but it
never sees the question and a chunk together. A reranker is a
"cross-encoder": it reads (question, chunk) pairs and scores how well each
chunk answers the question. Too slow for the whole corpus, so the usual
pattern is two stages: vector search fetches ~30 candidates, the reranker
picks the best few.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import voyageai
import voyageai.error

from prestie.knowledge.embeddings import MAX_RETRIES, EmbeddingError
from prestie.knowledge.store import SearchHit


class Reranks(Protocol):
    def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int
    ) -> list[SearchHit]: ...


class VoyageReranker:
    def __init__(self, client: Any, model: str):
        self._client = client
        self.model = model

    @classmethod
    def from_api_key(cls, api_key: str, model: str) -> "VoyageReranker":
        return cls(voyageai.Client(api_key=api_key, max_retries=MAX_RETRIES), model)

    def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int
    ) -> list[SearchHit]:
        """The `top_k` hits most relevant to `query`, best first."""
        if not hits:
            return []
        try:
            result = self._client.rerank(
                query, [hit.text for hit in hits], model=self.model, top_k=top_k
            )
        except voyageai.error.VoyageError as exc:
            raise EmbeddingError(
                f"Voyage rerank failed (model={self.model}, {len(hits)} candidates): "
                f"{exc}"
            ) from exc
        return [hits[item.index] for item in result.results]
