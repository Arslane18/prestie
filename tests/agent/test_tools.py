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


# --- alternative queries (multi-query, fused in code) ------------------------------


class PerQueryRetriever(FakeRetriever):
    def __init__(self, hits_by_query):
        super().__init__()
        self.hits_by_query = hits_by_query

    def search(self, query, n_results=5, where=None):
        super().search(query, n_results, where)
        return self.hits_by_query.get(query, [])


def test_alternative_queries_are_searched_and_fused():
    fade = make_hit("Fade Pulling", "https://iv/rot#fade-pulling")
    defensives = make_hit("Defensives", "https://iv/easy#defensives")
    retriever = PerQueryRetriever(
        {
            "reduce threat": [defensives, fade],
            "can a priest tank": [fade],
            "pulling with the tank": [fade],
        }
    )

    outcome = KnowledgeBaseTool(retriever).run(
        {
            "query": "reduce threat",
            "alternative_queries": ["can a priest tank", "pulling with the tank"],
            "spec": "shadow-priest",
        }
    )

    assert [call["query"] for call in retriever.calls] == [
        "reduce threat",
        "can a priest tank",
        "pulling with the tank",
    ]
    assert all(call["where"] == retriever.calls[0]["where"] for call in retriever.calls)
    # Found by all three queries, the Fade passage now comes first.
    assert outcome.content.index("Fade Pulling") < outcome.content.index("Defensives")
    assert 'alternative_queries="can a priest tank | pulling with the tank"' in (
        outcome.content
    )


def test_blank_or_duplicate_alternatives_are_ignored():
    retriever = PerQueryRetriever({})

    KnowledgeBaseTool(retriever).run(
        {"query": "stats", "alternative_queries": [" ", "Stats"]}
    )

    assert [call["query"] for call in retriever.calls] == ["stats"]


def test_alternative_queries_are_validated():
    tool = KnowledgeBaseTool(FakeRetriever())

    for bad in (
        {"query": "ok", "alternative_queries": "not a list"},
        {"query": "ok", "alternative_queries": ["a", "b", "c"]},
        {"query": "ok", "alternative_queries": [42]},
        {"query": "ok", "alternative_queries": ["x" * 1000]},
    ):
        assert tool.run(bad).is_error, bad


def test_schema_offers_up_to_two_alternative_queries():
    alternatives = SEARCH_TOOL["input_schema"]["properties"]["alternative_queries"]

    assert alternatives["type"] == "array"
    assert alternatives["maxItems"] == 2
    assert "ambiguous" in alternatives["description"]


def test_label_mentions_alternative_phrasings():
    from prestie.agent.agent import ToolCall
    from prestie.agent.tool_labels import tool_call_label

    call = ToolCall(
        "search_knowledge_base",
        {"query": "threat", "alternative_queries": ["tanking", "pulling"]},
    )

    assert tool_call_label(call) == "[recherche] threat (+2 formulations)"


# --- details for traces ------------------------------------------------------------


def test_search_details_list_the_queries_and_each_returned_passage():
    hit = SearchHit(
        id="c1",
        text="T",
        metadata={"section": "Fade Pulling", "source_url": "https://iv/rot#fade"},
        distance=0.42,
    )
    tool = KnowledgeBaseTool(PerQueryRetriever({"threat": [hit], "tanking": [hit]}))

    outcome = tool.run(
        {"query": "threat", "alternative_queries": ["tanking"], "spec": "shadow-priest"}
    )

    assert outcome.details == {
        "queries": ["threat", "tanking"],
        "results": [
            {
                "section": "Fade Pulling",
                "source_url": "https://iv/rot#fade",
                "title": None,
                "distance": 0.42,
            }
        ],
    }


def test_search_without_results_has_empty_details_results():
    outcome = KnowledgeBaseTool(FakeRetriever([])).run({"query": "x"})

    assert outcome.details == {"queries": ["x"], "results": []}



# --- review lot D: descriptions for every class, gear included -----------------


def test_rotation_page_type_is_not_described_with_one_class_resource():
    help_text = SEARCH_TOOL["input_schema"]["properties"]["content_type"]["description"]
    [rotation] = [line for line in help_text.splitlines() if line.startswith("- rotation:")]

    assert "rune" not in rotation.lower()


def test_search_description_covers_gear_questions():
    assert "gear" in SEARCH_TOOL["description"]


def test_spec_parameter_does_not_point_to_a_tool_manual_mode_lacks():
    spec = SEARCH_TOOL["input_schema"]["properties"]["spec"]["description"]

    assert "get_character_state" not in spec


def test_search_details_carry_the_guide_title_for_attribution():
    hit = SearchHit(
        id="c1",
        text="T",
        metadata={
            "section": "Rotation > Single Target",
            "source_url": "https://iv/frost-mage-rotation#single",
            "title": "Frost Mage Rotation Guide",
        },
        distance=0.3,
    )

    outcome = KnowledgeBaseTool(FakeRetriever([hit])).run({"query": "rotation"})

    assert outcome.details["results"][0]["title"] == "Frost Mage Rotation Guide"
