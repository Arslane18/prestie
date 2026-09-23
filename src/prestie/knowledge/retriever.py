"""Question → relevant chunks. The single entry point for retrieval, shared by
`prestie search`, the retrieval evaluation and (next) the agent's tool."""

from collections.abc import Mapping
from typing import Any

from prestie.knowledge.embeddings import Embedder
from prestie.knowledge.store import DEFAULT_RESULTS, KnowledgeStore, SearchHit


class Retriever:
    def __init__(self, embedder: Embedder, store: KnowledgeStore):
        self._embedder = embedder
        self._store = store

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_RESULTS,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        query_vector = self._embedder.embed_query(query)
        return self._store.search(query_vector, n_results=n_results, where=where)
