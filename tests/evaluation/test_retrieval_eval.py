import json
from pathlib import Path

import pytest

from prestie.evaluation.retrieval import (
    EvalCaseError,
    RetrievalCase,
    evaluate,
    load_cases,
)
from prestie.knowledge.store import SearchHit

BASE = "https://www.icy-veins.com/wow"
SHIPPED_CASES = Path(__file__).parents[2] / "evals" / "retrieval_cases.json"


def hit(slug: str, anchor: str | None, distance: float = 0.3) -> SearchHit:
    url = f"{BASE}/{slug}" + (f"#{anchor}" if anchor else "")
    return SearchHit(
        id=f"{slug}:{anchor}",
        text="...",
        metadata={"page_slug": slug, "source_url": url, "section": anchor or "Intro"},
        distance=distance,
    )


def fake_search(results: dict[str, list[SearchHit]]):
    def search(question: str, n_results: int) -> list[SearchHit]:
        return results[question][:n_results]

    return search


def test_rank_is_position_of_first_expected_source():
    case = RetrievalCase("stats?", expected=("stat-page#san",))
    search = fake_search({"stats?": [hit("other", "x"), hit("stat-page", "san")]})

    [result] = evaluate([case], search, k=5).results

    assert result.rank == 2
    assert result.found


def test_expected_page_without_anchor_matches_any_section_of_that_page():
    case = RetrievalCase("m+?", expected=("mplus-page",))
    search = fake_search({"m+?": [hit("mplus-page", "affixes")]})

    assert evaluate([case], search, k=5).results[0].rank == 1


def test_missing_source_has_no_rank():
    case = RetrievalCase("q", expected=("wanted#a",))
    search = fake_search({"q": [hit("other", "x")]})

    [result] = evaluate([case], search, k=5).results

    assert result.rank is None
    assert not result.found


def test_metrics_only_count_answerable_cases():
    cases = [
        RetrievalCase("a", expected=("p#1",)),
        RetrievalCase("b", expected=("p#2",)),
        RetrievalCase("c", expected=()),  # not covered by the corpus
    ]
    search = fake_search(
        {
            "a": [hit("p", "1")],
            "b": [hit("p", "x"), hit("p", "y"), hit("p", "2")],
            "c": [hit("p", "z", distance=0.55)],
        }
    )

    report = evaluate(cases, search, k=5)

    assert report.answerable_count == 2
    assert report.hit_rate(1) == 0.5
    assert report.hit_rate(3) == 1.0
    assert report.mrr == pytest.approx((1 + 1 / 3) / 2)
    unanswerable = report.results[2]
    assert not unanswerable.case.answerable
    assert unanswerable.top_distance == 0.55


def test_load_cases_reads_json_file(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {"question": "q1", "expected": ["p#a", "p"]},
                {"question": "q2", "expected": [], "note": "not in corpus"},
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases == (
        RetrievalCase("q1", expected=("p#a", "p")),
        RetrievalCase("q2", expected=(), note="not in corpus"),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "not a list"},
        [{"expected": ["p"]}],
        [{"question": "", "expected": []}],
        [{"question": "q", "expected": "p#a"}],
    ],
)
def test_load_cases_rejects_malformed_files(tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvalCaseError):
        load_cases(path)


def test_shipped_cases_file_is_valid():
    cases = load_cases(SHIPPED_CASES)

    assert len(cases) >= 10
    assert any(not case.answerable for case in cases)
