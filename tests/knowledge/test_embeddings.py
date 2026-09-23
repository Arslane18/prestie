import pytest
import voyageai.error

from prestie.knowledge.embeddings import EmbeddingError, VoyageEmbedder


class FakeResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.total_tokens = 7


class FakeVoyageClient:
    def __init__(self, error: Exception | None = None):
        self.calls: list[dict] = []
        self.error = error

    def embed(self, texts, model=None, input_type=None, **kwargs):
        self.calls.append(
            {"texts": list(texts), "model": model, "input_type": input_type}
        )
        if self.error:
            raise self.error
        return FakeResult([[float(len(t)), 0.0] for t in texts])


def test_documents_are_embedded_with_document_input_type_and_model():
    client = FakeVoyageClient()
    embedder = VoyageEmbedder(client, model="voyage-4-large")

    vectors = embedder.embed_documents(["ab", "abc"])

    assert vectors == [[2.0, 0.0], [3.0, 0.0]]
    assert client.calls == [
        {"texts": ["ab", "abc"], "model": "voyage-4-large", "input_type": "document"}
    ]


def test_query_uses_query_input_type():
    client = FakeVoyageClient()

    vector = VoyageEmbedder(client, model="m").embed_query("hello")

    assert vector == [5.0, 0.0]
    assert client.calls[0]["input_type"] == "query"


def test_documents_are_sent_in_batches_preserving_order():
    client = FakeVoyageClient()
    embedder = VoyageEmbedder(client, model="m", batch_size=2)

    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(call["texts"]) for call in client.calls] == [2, 2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_no_documents_means_no_api_call():
    client = FakeVoyageClient()

    assert VoyageEmbedder(client, model="m").embed_documents([]) == []
    assert client.calls == []


def test_voyage_errors_are_wrapped_with_context():
    client = FakeVoyageClient(error=voyageai.error.AuthenticationError("bad key"))

    with pytest.raises(EmbeddingError, match="voyage-4-large"):
        VoyageEmbedder(client, model="voyage-4-large").embed_query("x")
