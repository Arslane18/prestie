import json
from types import SimpleNamespace

import anthropic
import pytest

from prestie.knowledge.filters import metadata_filter
from prestie.knowledge.rewriting import (
    CachedQueryWriter,
    ClaudeQueryWriter,
    QueryRewriteError,
    RewritingRetriever,
    reciprocal_rank_fusion,
)
from prestie.knowledge.store import SearchHit


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(chunk_id, f"text {chunk_id}", {}, 0.1)


def ids(hits):
    return [h.id for h in hits]


# --- fusion ----------------------------------------------------------------------


def test_fusion_ranks_passages_found_by_several_queries_first():
    fused = reciprocal_rank_fusion(
        [[hit("a"), hit("b"), hit("c")], [hit("b"), hit("d")], [hit("b"), hit("a")]],
        n_results=3,
    )

    assert ids(fused) == ["b", "a", "d"]


def test_fusion_keeps_each_passage_once_and_truncates():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("a")]], n_results=1)

    assert ids(fused) == ["a"]


def test_fusion_breaks_ties_by_first_appearance():
    assert ids(reciprocal_rank_fusion([[hit("x")], [hit("y")]], n_results=2)) == [
        "x",
        "y",
    ]


# --- strategies ------------------------------------------------------------------


class FakeRetriever:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def search(self, query, n_results=5, where=None):
        self.calls.append((query, n_results, where))
        return self.results.get(query, [hit(query)])


class FakeWriter:
    def __init__(self):
        self.calls = []

    def queries(self, question, spec_name, count):
        self.calls.append(("queries", question, spec_name, count))
        return tuple(f"q{i}" for i in range(count))

    def passage(self, question, spec_name):
        self.calls.append(("passage", question, spec_name))
        return "Fade drops threat when pulling with the tank."


WHERE = metadata_filter("shadow-priest")


def test_single_strategy_searches_with_one_rewritten_query():
    retriever, writer = FakeRetriever(), FakeWriter()

    hits = RewritingRetriever(retriever, writer, "single").search("aggro ?", 5, WHERE)

    assert writer.calls == [("queries", "aggro ?", "Shadow Priest", 1)]
    assert retriever.calls == [("q0", 5, WHERE)]
    assert ids(hits) == ["q0"]


def test_multi_strategy_fuses_the_question_and_its_rewrites():
    retriever = FakeRetriever(
        {"aggro ?": [hit("a")], "q0": [hit("b")], "q1": [hit("b")], "q2": [hit("c")]}
    )

    hits = RewritingRetriever(retriever, FakeWriter(), "multi").search(
        "aggro ?", 2, WHERE
    )

    assert [call[0] for call in retriever.calls] == ["aggro ?", "q0", "q1", "q2"]
    assert ids(hits) == ["b", "a"]


def test_hyde_strategy_searches_with_a_hypothetical_passage():
    retriever, writer = FakeRetriever(), FakeWriter()

    RewritingRetriever(retriever, writer, "hyde").search("aggro ?", 5, WHERE)

    assert writer.calls == [("passage", "aggro ?", "Shadow Priest")]
    assert retriever.calls[0][0].startswith("Fade drops threat")


def test_without_a_spec_filter_the_writer_gets_no_spec():
    writer = FakeWriter()

    RewritingRetriever(FakeRetriever(), writer, "single").search("q", 5, None)

    assert writer.calls[0][2] is None


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        RewritingRetriever(FakeRetriever(), FakeWriter(), "magic")


# --- Claude writer ---------------------------------------------------------------


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.requests = []
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_claude_writer_asks_for_english_queries_with_structured_output():
    client = FakeClient({"queries": ["Fade threat", "tanking as Shadow"]})

    queries = ClaudeQueryWriter(client, model="m").queries("aggro ?", "Shadow Priest", 2)

    assert queries == ("Fade threat", "tanking as Shadow")
    [request] = client.requests
    assert request["model"] == "m"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "Shadow Priest" in request["messages"][0]["content"]
    assert "aggro ?" in request["messages"][0]["content"]


def test_claude_writer_keeps_only_the_requested_number_of_queries():
    client = FakeClient({"queries": ["a", "b", "c"]})

    assert ClaudeQueryWriter(client, model="m").queries("q", None, 1) == ("a",)


def test_claude_writer_writes_a_hypothetical_passage():
    client = FakeClient({"passage": "Use Fade when you pull."})

    assert ClaudeQueryWriter(client, model="m").passage("q", None) == (
        "Use Fade when you pull."
    )


@pytest.mark.parametrize(
    "client",
    [
        FakeClient("not json"),
        FakeClient({"queries": []}),
        FakeClient({"queries": ["a"]}, stop_reason="max_tokens"),
    ],
)
def test_claude_writer_rejects_unusable_responses(client):
    with pytest.raises(QueryRewriteError):
        ClaudeQueryWriter(client, model="m").queries("q", None, 1)


def test_claude_writer_wraps_api_errors():
    client = FakeClient({"queries": ["a"]})
    client.response = anthropic.APIConnectionError(request=None)

    with pytest.raises(QueryRewriteError):
        ClaudeQueryWriter(client, model="m").queries("q", None, 1)


# --- cache -----------------------------------------------------------------------


def test_cache_reuses_rewrites_across_runs(tmp_path):
    path = tmp_path / "rewrites.json"
    writer = FakeWriter()

    first = CachedQueryWriter(writer, path, model="m")
    first.queries("q", "Shadow Priest", 3)
    first.passage("q", "Shadow Priest")
    again = CachedQueryWriter(writer, path, model="m")

    assert again.queries("q", "Shadow Priest", 3) == ("q0", "q1", "q2")
    assert again.passage("q", "Shadow Priest").startswith("Fade")
    assert len(writer.calls) == 2  # one per distinct rewrite, never repeated


def test_cache_is_keyed_by_model_and_spec(tmp_path):
    path = tmp_path / "rewrites.json"
    writer = FakeWriter()

    CachedQueryWriter(writer, path, model="m").queries("q", "A", 1)
    CachedQueryWriter(writer, path, model="m").queries("q", "B", 1)
    CachedQueryWriter(writer, path, model="other").queries("q", "A", 1)

    assert len(writer.calls) == 3
