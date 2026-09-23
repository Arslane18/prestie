"""Vector store: Chroma collection holding chunk texts, vectors and metadata.

We compute embeddings ourselves (Voyage) and hand them to Chroma, so Chroma is
created with `embedding_function=None`: it only stores vectors and runs the
nearest-neighbour search (HNSW index, cosine distance = 1 - cosine similarity).

Vectors from different models live in unrelated spaces, so comparing them is
meaningless. The collection records the model it was built with and refuses to
be opened with another one; `reset=True` rebuilds it from scratch.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import chromadb.errors
from chromadb.api import ClientAPI

COLLECTION_NAME = "wow_guides"
MODEL_METADATA_KEY = "embedding_model"
DEFAULT_RESULTS = 5


class EmbeddingModelMismatchError(Exception):
    """The collection was built with a different embedding model."""


@dataclass(frozen=True)
class SearchHit:
    id: str
    text: str
    metadata: Mapping[str, Any]
    distance: float  # cosine distance: 0 = same direction, lower is closer


class KnowledgeStore:
    def __init__(self, collection: Any):
        self._collection = collection

    @classmethod
    def open(
        cls,
        client: ClientAPI,
        embedding_model: str,
        *,
        reset: bool = False,
        name: str = COLLECTION_NAME,
    ) -> "KnowledgeStore":
        if reset:
            _delete_if_exists(client, name)
        collection = client.get_or_create_collection(
            name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={MODEL_METADATA_KEY: embedding_model},
            embedding_function=None,
        )
        built_with = (collection.metadata or {}).get(MODEL_METADATA_KEY)
        if built_with != embedding_model:
            raise EmbeddingModelMismatchError(
                f"Collection '{name}' was built with '{built_with}', not "
                f"'{embedding_model}'. Re-index with `prestie ingest --reset`."
            )
        return cls(collection)

    def count(self) -> int:
        return self._collection.count()

    def replace_page(
        self,
        page_slug: str,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Swap all chunks of one page, so chunks removed upstream do not linger."""
        if not len(ids) == len(texts) == len(embeddings) == len(metadatas):
            raise ValueError(
                "ids, texts, embeddings and metadatas must have the same length"
            )
        self._collection.delete(where={"page_slug": page_slug})
        if ids:
            self._collection.upsert(
                ids=list(ids),
                documents=list(texts),
                embeddings=[list(vector) for vector in embeddings],
                metadatas=[dict(metadata) for metadata in metadatas],
            )

    def search(
        self,
        query_embedding: Sequence[float],
        n_results: int = DEFAULT_RESULTS,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=dict(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchHit(id=id_, text=text, metadata=metadata, distance=distance)
            for id_, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


def _delete_if_exists(client: ClientAPI, name: str) -> None:
    try:
        client.delete_collection(name)
    except chromadb.errors.NotFoundError:
        pass
