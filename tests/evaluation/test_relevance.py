import json
from types import SimpleNamespace

import pytest

from prestie.evaluation.relevance import (
    RelevanceJudge,
    RelevanceJudgeError,
    pool_passages,
    record_judgments,
)
from prestie.evaluation.retrieval import RetrievalCase, evaluate, load_cases
from prestie.knowledge.store import SearchHit

BASE = "https://www.icy-veins.com/wow"
CASE = RetrievalCase("stats?", expected=("stat#san",), spec="shadow-priest")


def hit(id_: str, slug: str, anchor: str) -> SearchHit:
    return SearchHit(
        id=id_,
        text=f"passage {id_}",
        metadata={
            "page_slug": slug,
            "source_url": f"{BASE}/{slug}#{anchor}",
            "section": anchor,
        },
        distance=0.3,
    )


def test_pool_unions_the_top_hits_of_every_system_without_duplicates():
    vector = [hit("1", "stat", "san"), hit("2", "guide", "intro"), hit("3", "gear", "x")]
    rerank = [hit("2", "guide", "intro"), hit("4", "easy", "stats"), hit("5", "m", "z")]

    pooled = pool_passages(CASE, [vector, rerank], depth=2)

    # "stat#san" is already a hand label; hits beyond depth 2 are left out.
    assert [p.key for p in pooled] == ["guide#intro", "easy#stats"]


def test_pool_skips_passages_already_judged():
    case = RetrievalCase(
        "stats?",
        expected=(),
        judged_relevant=("guide#intro",),
        judged_irrelevant=("easy#stats",),
    )
    hits = [hit("2", "guide", "intro"), hit("4", "easy", "stats"), hit("6", "new", "y")]

    assert [p.key for p in pool_passages(case, [hits], depth=3)] == ["new#y"]


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
            model="claude-sonnet-5",
        )


def verdicts(*labels):
    return {
        "judgments": [
            {"passage": i, "reason": "because", "label": label}
            for i, label in enumerate(labels, start=1)
        ]
    }


def two_passages():
    return pool_passages(
        CASE, [[hit("2", "guide", "intro"), hit("4", "easy", "stats")]], depth=2
    )


def test_judge_labels_every_passage_in_one_call():
    client = FakeClient(verdicts("answers", "partial"))

    judgments = RelevanceJudge(client).judge(CASE, two_passages())

    assert [(j.key, j.label) for j in judgments] == [
        ("guide#intro", "answers"),
        ("easy#stats", "partial"),
    ]
    [request] = client.requests
    assert request["model"] == "claude-sonnet-5"
    prompt = request["messages"][0]["content"]
    assert "stats?" in prompt and "Shadow Priest" in prompt and "passage 4" in prompt


def test_judge_rejects_answers_that_do_not_cover_every_passage():
    with pytest.raises(RelevanceJudgeError):
        RelevanceJudge(FakeClient(verdicts("answers"))).judge(CASE, two_passages())
    with pytest.raises(RelevanceJudgeError):
        RelevanceJudge(FakeClient("{not json")).judge(CASE, two_passages())


def test_nothing_to_judge_means_no_call():
    client = FakeClient(verdicts())

    assert RelevanceJudge(client).judge(CASE, ()) == ()
    assert client.requests == []


def test_judgments_are_recorded_next_to_the_hand_labels(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [{"question": "stats?", "expected": ["stat#san"], "spec": "shadow-priest"}]
        ),
        encoding="utf-8",
    )
    hits = [hit("2", "guide", "intro"), hit("4", "easy", "stats"), hit("5", "m", "z")]
    pooled = pool_passages(CASE, [hits], depth=3)
    client = FakeClient(verdicts("answers", "partial", "irrelevant"))
    judgments = RelevanceJudge(client).judge(CASE, pooled)

    record_judgments(path, {0: judgments}, judge_model="claude-sonnet-5")

    [case] = load_cases(path)
    assert case.expected == ("stat#san",)  # hand labels untouched
    assert case.judged_relevant == ("guide#intro",)
    # "partial" is not relevant: only passages that answer count.
    assert case.judged_irrelevant == ("easy#stats", "m#z")
    entry = json.loads(path.read_text(encoding="utf-8"))[0]
    assert entry["judge_model"] == "claude-sonnet-5"


def test_judged_relevant_passages_count_as_hits_unless_disabled():
    case = RetrievalCase(
        "stats?", expected=("stat#san",), judged_relevant=("guide#intro",)
    )

    def search(question, n_results, where=None):
        return [hit("2", "guide", "intro")]

    assert evaluate([case], search, k=5).results[0].rank == 1
    assert evaluate([case], search, k=5, judged=False).results[0].rank is None


def test_an_anchorless_passage_is_judged_alone_not_as_its_whole_page():
    intro = SearchHit(
        id="i",
        text="intro",
        metadata={"page_slug": "talents", "source_url": f"{BASE}/talents"},
        distance=0.3,
    )
    case = RetrievalCase("build?", expected=(), judged_irrelevant=("talents#",))

    # A bare "talents" key would mean "any passage of the page" and hide the
    # page's other sections from later judging runs (or count them all as
    # hits if the intro were relevant).
    assert [p.key for p in pool_passages(CASE, [[intro]], depth=1)] == ["talents#"]
    new_section = hit("7", "talents", "raid-build")
    assert [p.key for p in pool_passages(case, [[intro, new_section]], 2)] == [
        "talents#raid-build"
    ]
