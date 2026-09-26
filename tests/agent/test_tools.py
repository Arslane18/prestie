from prestie.agent.tools import SEARCH_TOOL, KnowledgeBaseTool
from prestie.catalog import COVERED_SPECS
from prestie.ingestion.icy_veins.pages import ALL_PAGES
from prestie.knowledge.embeddings import EmbeddingError
from prestie.knowledge.store import SearchHit


def make_hit(
    section: str, url: str, text: str = "Title\nSection: x\n\nBody"
) -> SearchHit:
    return SearchHit(
        id=section,
        text=text,
        metadata={"section": section, "source_url": url},
        distance=0.4,
    )


class FakeRetriever:
    def __init__(self, hits=None, error: Exception | None = None):
        self.hits = hits or []
        self.error = error
        self.calls: list[dict] = []

    def search(self, query, n_results=5, where=None):
        self.calls.append({"query": query, "n_results": n_results, "where": where})
        if self.error:
            raise self.error
        return self.hits


def test_tool_schema_exposes_every_content_type_as_enum():
    properties = SEARCH_TOOL["input_schema"]["properties"]

    assert SEARCH_TOOL["name"] == "search_knowledge_base"
    assert set(properties["content_type"]["enum"]) == {
        page.content_type for page in ALL_PAGES
    }
    assert SEARCH_TOOL["input_schema"]["required"] == ["query"]


def test_run_formats_hits_with_their_source_url():
    retriever = FakeRetriever(
        [make_hit("Stat Priority", "https://iv/stat#san", "T\nSection: S\n\nHaste")]
    )

    outcome = KnowledgeBaseTool(retriever).run({"query": "secondary stats"})

    assert not outcome.is_error
    assert 'source_url="https://iv/stat#san"' in outcome.content
    assert 'section="Stat Priority"' in outcome.content
    assert "Haste" in outcome.content
    assert retriever.calls == [
        {"query": "secondary stats", "n_results": 5, "where": None}
    ]


def test_content_type_becomes_a_metadata_filter():
    retriever = FakeRetriever([make_hit("Rotation", "https://iv/rot")])

    KnowledgeBaseTool(retriever).run({"query": "aoe", "content_type": "rotation"})

    assert retriever.calls[0]["where"] == {"content_type": "rotation"}


def test_no_hits_is_reported_plainly():
    outcome = KnowledgeBaseTool(FakeRetriever([])).run({"query": "trinkets"})

    assert not outcome.is_error
    assert "No results" in outcome.content


def test_invalid_input_returns_an_error_result_instead_of_raising():
    tool = KnowledgeBaseTool(FakeRetriever())

    for bad_input in (
        {},
        {"query": "   "},
        {"query": 42},
        {"query": "x" * 1000},
        {"query": "ok", "content_type": "pvp"},
        {"query": "ok", "spec": "frost-paladin"},
        {"query": "ok", "spec": 3},
    ):
        assert tool.run(bad_input).is_error, bad_input


def test_retrieval_failure_is_returned_as_error_result():
    tool = KnowledgeBaseTool(FakeRetriever(error=EmbeddingError("voyage down")))

    outcome = tool.run({"query": "runes"})

    assert outcome.is_error
    assert "voyage down" in outcome.content


def test_every_content_type_is_described_to_the_model():
    description = SEARCH_TOOL["input_schema"]["properties"]["content_type"][
        "description"
    ]

    for content_type in {page.content_type for page in ALL_PAGES}:
        assert f"{content_type}:" in description


def test_content_type_filter_is_presented_as_a_second_attempt():
    description = SEARCH_TOOL["input_schema"]["properties"]["content_type"][
        "description"
    ]

    assert "without" in description.lower()


def test_tool_schema_exposes_every_covered_spec_as_enum():
    spec = SEARCH_TOOL["input_schema"]["properties"]["spec"]

    assert spec["enum"] == [guide.key for guide in COVERED_SPECS]
    assert "every class and specialization" in SEARCH_TOOL["description"]
    assert "Shadow Priest" not in SEARCH_TOOL["description"]


def test_spec_becomes_a_class_and_spec_filter():
    retriever = FakeRetriever([make_hit("Rotation", "https://iv/rot")])

    KnowledgeBaseTool(retriever).run({"query": "aoe", "spec": "shadow-priest"})

    assert retriever.calls[0]["where"] == {
        "$and": [{"wow_class": "priest"}, {"spec": "shadow"}]
    }


def test_spec_and_content_type_filters_combine():
    retriever = FakeRetriever([make_hit("Rotation", "https://iv/rot")])

    KnowledgeBaseTool(retriever).run(
        {"query": "aoe", "spec": "blood-death-knight", "content_type": "rotation"}
    )

    assert retriever.calls[0]["where"] == {
        "$and": [
            {"wow_class": "death-knight"},
            {"spec": "blood"},
            {"content_type": "rotation"},
        ]
    }
