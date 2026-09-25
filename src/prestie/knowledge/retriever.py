"""Question → relevant chunks. The single entry point for retrieval, shared by
`prestie search`, the retrieval evaluation and the agent's tool.

Without a reranker: one vector search for `n_results` chunks. With one: the
vector search fetches `candidates` chunks and the reranker keeps the best
`n_results` (see reranker.py for why two stages).
"""

from collections.abc import Mapping
from typing import Any

from prestie.knowledge.embeddings import Embedder
from prestie.knowledge.reranker import Reranks
from prestie.knowledge.store import DEFAULT_RESULTS, KnowledgeStore, SearchHit

DEFAULT_CANDIDATES = 30


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: KnowledgeStore,
        *,
        reranker: Reranks | None = None,
        candidates: int = DEFAULT_CANDIDATES,
    ):
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._candidates = candidates

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_RESULTS,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        query_vector = self._embedder.embed_query(query)
        if self._reranker is None:
            return self._store.search(query_vector, n_results=n_results, where=where)
        candidates = self._store.search(
            query_vector, n_results=max(self._candidates, n_results), where=where
        )
        return self._reranker.rerank(query, candidates, top_k=n_results)
