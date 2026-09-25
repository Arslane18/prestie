import chromadb
import pytest

from prestie.knowledge.store import (
    EmbeddingModelMismatchError,
    KnowledgeStore,
    SearchHit,
)

MODEL = "voyage-4-large"


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path))


def meta(slug: str, content_type: str = "rotation") -> dict:
    return {"page_slug": slug, "content_type": content_type, "spell_ids": [49998]}


def test_search_returns_nearest_chunks_first_with_cosine_distance(client):
    store = KnowledgeStore.open(client, embedding_model=MODEL)
    store.replace_page(
        "page-a",
        ids=["a:0", "a:1"],
        texts=["about runes", "about stats"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[meta("page-a"), meta("page-a", "stat_priority")],
    )

    hits = store.search([0.9, 0.1], n_results=2)

    assert [hit.id for hit in hits] == ["a:0", "a:1"]
    assert isinstance(hits[0], SearchHit)
    assert hits[0].text == "about runes"
    assert hits[0].metadata["content_type"] == "rotation"
    assert hits[0].distance < hits[1].distance


def test_search_can_filter_on_metadata(client):
    store = KnowledgeStore.open(client, embedding_model=MODEL)
    store.replace_page(
        "page-a",
        ids=["a:0", "a:1"],
        texts=["rotation", "stats"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[meta("page-a"), meta("page-a", "stat_priority")],
    )

    hits = store.search([1.0, 0.0], where={"content_type": "stat_priority"})

    assert [hit.id for hit in hits] == ["a:1"]


def test_replace_page_removes_stale_chunks_of_that_page_only(client):
    store = KnowledgeStore.open(client, embedding_model=MODEL)
    store.replace_page(
        "page-a", ["a:0", "a:1"], ["x", "y"], [[1, 0], [0, 1]], [meta("page-a")] * 2
    )
    store.replace_page("page-b", ["b:0"], ["z"], [[1, 1]], [meta("page-b")])

    store.replace_page("page-a", ["a:0"], ["x2"], [[1, 0]], [meta("page-a")])

    assert store.count() == 2
    assert {hit.id for hit in store.search([1, 0], n_results=5)} == {"a:0", "b:0"}


def test_reopening_with_another_embedding_model_is_refused(client):
    KnowledgeStore.open(client, embedding_model=MODEL)

    with pytest.raises(EmbeddingModelMismatchError, match="voyage-4-lite"):
        KnowledgeStore.open(client, embedding_model="voyage-4-lite")


def test_reset_drops_existing_collection_and_allows_new_model(client):
    store = KnowledgeStore.open(client, embedding_model=MODEL)
    store.replace_page("page-a", ["a:0"], ["x"], [[1, 0]], [meta("page-a")])

    fresh = KnowledgeStore.open(client, embedding_model="voyage-4-lite", reset=True)

    assert fresh.count() == 0


def test_mismatched_lengths_are_rejected(client):
    store = KnowledgeStore.open(client, embedding_model=MODEL)

    with pytest.raises(ValueError, match="same length"):
        store.replace_page("page-a", ["a:0", "a:1"], ["x"], [[1, 0]], [meta("page-a")])


def test_search_on_empty_store_returns_nothing(client):
    assert KnowledgeStore.open(client, embedding_model=MODEL).search([1.0, 0.0]) == []


def test_search_can_combine_metadata_filters(client):
    # The search tool filters on class + spec (+ content type) with Chroma's $and.
    store = KnowledgeStore.open(client, embedding_model=MODEL)
    store.replace_page(
        "page-a",
        ids=["dk", "shadow", "disc"],
        texts=["blood", "shadow", "disc"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        metadatas=[
            {**meta("page-a"), "wow_class": "death-knight", "spec": "blood"},
            {**meta("page-a"), "wow_class": "priest", "spec": "shadow"},
            {**meta("page-a"), "wow_class": "priest", "spec": "discipline"},
        ],
    )

    hits = store.search(
        [1.0, 0.0],
        where={"$and": [{"wow_class": "priest"}, {"spec": "shadow"}]},
    )

    assert [hit.id for hit in hits] == ["shadow"]
