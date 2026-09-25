"""Retrieval evaluation: does the right source show up in the top-k results?

Each case is a question plus the sections that answer it ("page_slug#anchor",
or just "page_slug" when any section of that page is acceptable). Cases with
no expected source are questions the corpus cannot answer: they are excluded
from the metrics but their best distance is reported, to see how "close" an
irrelevant match can look.

Metrics, over answerable cases:
  - hit@k: share of questions whose first relevant chunk ranks <= k
  - MRR (mean reciprocal rank): average of 1/rank (0 when not found);
    1.0 means the right source always comes first.
And over cases tagged with a spec:
  - spec_precision@k: share of the top-k chunks that come from that spec. The
    rank metrics only look at the first relevant chunk; this one sees the
    other specs' passages that still crowd the context sent to the model.

With `spec_filter=True`, each case's spec is passed as a metadata filter, the
way the agent searches; without it, every spec competes.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prestie.catalog import spec_by_key
from prestie.knowledge.filters import metadata_filter
from prestie.knowledge.store import SearchHit

# (question, number of results, metadata filter or None) -> hits
SearchFn = Callable[[str, int, Mapping[str, Any] | None], Sequence[SearchHit]]


class EvalCaseError(Exception):
    """The evaluation cases file is malformed."""


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected: tuple[str, ...]
    note: str = ""
    spec: str | None = None  # catalog key of the spec the question is about

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


@dataclass(frozen=True)
class CaseResult:
    case: RetrievalCase
    rank: int | None  # 1-based rank of the first expected source, if retrieved
    hits: tuple[SearchHit, ...] = field(repr=False)

    @property
    def found(self) -> bool:
        return self.rank is not None

    @property
    def top_distance(self) -> float | None:
        return self.hits[0].distance if self.hits else None


@dataclass(frozen=True)
class EvalReport:
    k: int
    results: tuple[CaseResult, ...]

    @property
    def _answerable(self) -> list[CaseResult]:
        return [result for result in self.results if result.case.answerable]

    @property
    def answerable_count(self) -> int:
        return len(self._answerable)

    def hit_rate(self, at: int) -> float:
        answerable = self._answerable
        if not answerable:
            return 0.0
        hits = sum(1 for r in answerable if r.rank is not None and r.rank <= at)
        return hits / len(answerable)

    def spec_precision(self, at: int) -> float:
        with_spec = [r for r in self.results if r.case.spec is not None]
        if not with_spec:
            return 0.0
        return sum(_spec_share(r, at) for r in with_spec) / len(with_spec)

    @property
    def mrr(self) -> float:
        answerable = self._answerable
        if not answerable:
            return 0.0
        return sum(1 / r.rank if r.rank else 0.0 for r in answerable) / len(answerable)


def evaluate(
    cases: Iterable[RetrievalCase],
    search: SearchFn,
    k: int,
    *,
    spec_filter: bool = False,
) -> EvalReport:
    return EvalReport(
        k=k,
        results=tuple(_evaluate_case(c, search, k, spec_filter) for c in cases),
    )


def _evaluate_case(
    case: RetrievalCase, search: SearchFn, k: int, spec_filter: bool
) -> CaseResult:
    where = metadata_filter(case.spec) if spec_filter else None
    hits = tuple(search(case.question, k, where))
    rank = next(
        (i for i, hit in enumerate(hits, start=1) if _matches(hit, case.expected)),
        None,
    )
    return CaseResult(case=case, rank=rank, hits=hits)


def _spec_share(result: CaseResult, at: int) -> float:
    guide = spec_by_key(result.case.spec or "")
    top = result.hits[:at]
    if guide is None or not top:
        return 0.0
    same = sum(
        1
        for hit in top
        if (hit.metadata.get("wow_class"), hit.metadata.get("spec"))
        == (guide.wow_class, guide.spec)
    )
    return same / len(top)


def source_key(hit: SearchHit) -> str:
    """'page_slug#anchor' (or 'page_slug' for anchor-less sections)."""
    anchor = str(hit.metadata.get("source_url", "")).partition("#")[2]
    slug = str(hit.metadata.get("page_slug", ""))
    return f"{slug}#{anchor}" if anchor else slug


def _matches(hit: SearchHit, expected: tuple[str, ...]) -> bool:
    key = source_key(hit)
    return key in expected or key.partition("#")[0] in expected


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalCaseError(f"Cannot read evaluation cases from {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise EvalCaseError(f"{path}: expected a JSON list of cases")
    return tuple(_parse_case(entry, index, path) for index, entry in enumerate(payload))


def _parse_case(entry: Any, index: int, path: Path) -> RetrievalCase:
    where = f"{path} case #{index}"
    if not isinstance(entry, dict):
        raise EvalCaseError(f"{where}: expected an object")
    question = entry.get("question")
    expected = entry.get("expected")
    if not isinstance(question, str) or not question.strip():
        raise EvalCaseError(f"{where}: 'question' must be a non-empty string")
    if not isinstance(expected, list) or not all(isinstance(e, str) for e in expected):
        raise EvalCaseError(f"{where}: 'expected' must be a list of strings")
    spec = entry.get("spec")
    if spec is not None and (not isinstance(spec, str) or spec_by_key(spec) is None):
        raise EvalCaseError(f"{where}: 'spec' {spec!r} is not a covered spec")
    return RetrievalCase(
        question=question,
        expected=tuple(expected),
        note=str(entry.get("note", "")),
        spec=spec,
    )
