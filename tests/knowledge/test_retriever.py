import chromadb

from prestie.knowledge.retriever import Retriever
from prestie.knowledge.store import KnowledgeStore


class FakeEmbedder:
    model = "fake"

    def __init__(self):
        self.queries: list[str] = []

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0] if "rune" in text else [0.0, 1.0]


def test_search_embeds_the_question_as_a_query_and_returns_hits(tmp_path):
    store = KnowledgeStore.open(
        chromadb.PersistentClient(path=str(tmp_path)), embedding_model="fake"
    )
    store.replace_page(
        "p",
        ids=["p:0", "p:1"],
        texts=["runes", "stats"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"page_slug": "p", "content_type": "rotation"},
            {"page_slug": "p", "content_type": "stat_priority"},
        ],
    )
    embedder = FakeEmbedder()
    retriever = Retriever(embedder, store)

    hits = retriever.search("how do runes work", n_results=1)

    assert embedder.queries == ["how do runes work"]
    assert [hit.id for hit in hits] == ["p:0"]
    assert [
        hit.id for hit in retriever.search("x", where={"content_type": "rotation"})
    ] == ["p:0"]
