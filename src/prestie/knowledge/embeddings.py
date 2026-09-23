"""Text → vector. An embedding maps text to a point in a high-dimensional space
where semantically close texts are close together; retrieval = nearest points.

Voyage models are asymmetric: documents and queries are embedded with a
different `input_type`, because a short question ("quelle stat monter ?") and
the passage answering it do not look alike, yet must land close to each other.
Always use `embed_documents` when indexing and `embed_query` when searching.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import voyageai
import voyageai.error

DEFAULT_BATCH_SIZE = 64  # far below Voyage's 1000 texts / 120k tokens per call
MAX_RETRIES = 3


class EmbeddingError(Exception):
    """The embedding provider failed to embed the given texts."""


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    def __init__(self, client: Any, model: str, batch_size: int = DEFAULT_BATCH_SIZE):
        self._client = client
        self.model = model
        self._batch_size = batch_size

    @classmethod
    def from_api_key(cls, api_key: str, model: str) -> "VoyageEmbedder":
        return cls(voyageai.Client(api_key=api_key, max_retries=MAX_RETRIES), model)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            vector
            for start in range(0, len(texts), self._batch_size)
            for vector in self._embed(
                texts[start : start + self._batch_size], "document"
            )
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        try:
            result = self._client.embed(
                list(texts), model=self.model, input_type=input_type
            )
        except voyageai.error.VoyageError as exc:
            raise EmbeddingError(
                f"Voyage embedding failed (model={self.model}, "
                f"{len(texts)} {input_type} text(s)): {exc}"
            ) from exc
        return [list(vector) for vector in result.embeddings]
