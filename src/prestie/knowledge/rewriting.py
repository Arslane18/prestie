"""Query rewriting: search with better queries than the player's own words.

The player asks in French, often vaguely ("je peux prendre l'aggro ?"), while
the guides are English and use precise terms ("Fade", "threat"). A vector
search only finds passages close to the query it gets, so the query decides
what the model will ever read. Three strategies are compared:

- single: an LLM writes one English search query (what the agent itself does
  when it calls the search tool);
- multi: the LLM writes several queries covering different readings of the
  question; each is searched and the lists are merged by Reciprocal Rank
  Fusion, so a passage found by several queries rises to the top;
- hyde (Hypothetical Document Embeddings): the LLM writes a short fake guide
  passage answering the question, and that passage is the query. A passage
  resembles other passages more than a question does; its facts may be wrong,
  only its vocabulary matters.

`RewritingRetriever` has the same `search` signature as `Retriever`, so the
retrieval eval can compare the strategies directly.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import anthropic

from prestie.knowledge.filters import filtered_spec
from prestie.knowledge.store import DEFAULT_RESULTS, SearchHit

STRATEGIES = ("single", "multi", "hyde")
MULTI_QUERY_COUNT = 3
# Standard RRF constant: dampens the gap between the first ranks, so agreement
# between lists matters more than a single list's top position.
RRF_K = 60
WRITER_MAX_TOKENS = 4000

WRITER_SYSTEM = """\
You write search queries for a World of Warcraft assistant. The knowledge \
base holds English Icy Veins class guides (rotation, talents, stat priority, \
gear, Mythic+, spell descriptions), one set per specialization, and searches \
are already restricted to the player's specialization. The player writes in \
French, often briefly or vaguely. Use the English names of spells, talents \
and game terms (e.g. "Sang vampirique" -> "Vampiric Blood", "aggro" -> \
"threat"). When the question can be read several ways, cover the plausible \
readings."""

QUERIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}
PASSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"passage": {"type": "string"}},
    "required": ["passage"],
    "additionalProperties": False,
}


class QueryRewriteError(Exception):
    """The rewriting call failed or returned nothing usable."""


class Searches(Protocol):
    def search(
        self,
        query: str,
        n_results: int = DEFAULT_RESULTS,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]: ...


class WritesQueries(Protocol):
    def queries(
        self, question: str, spec_name: str | None, count: int
    ) -> tuple[str, ...]: ...

    def passage(self, question: str, spec_name: str | None) -> str: ...


def reciprocal_rank_fusion(
    hit_lists: Sequence[Sequence[SearchHit]], n_results: int, k: int = RRF_K
) -> list[SearchHit]:
    """Merge ranked lists: each passage scores the sum of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (k + rank)
            first_seen.setdefault(hit.id, hit)
    # sorted() is stable: ties keep their first-appearance order.
    ranked = sorted(first_seen, key=lambda chunk_id: -scores[chunk_id])
    return [first_seen[chunk_id] for chunk_id in ranked[:n_results]]


class RewritingRetriever:
    def __init__(self, retriever: Searches, writer: WritesQueries, strategy: str):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown rewriting strategy {strategy!r}; one of {STRATEGIES}"
            )
        self._retriever = retriever
        self._writer = writer
        self._strategy = strategy

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_RESULTS,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        spec = filtered_spec(where)
        spec_name = spec.name if spec else None
        if self._strategy == "hyde":
            passage = self._writer.passage(query, spec_name)
            return self._retriever.search(passage, n_results, where)
        if self._strategy == "single":
            [rewritten] = self._writer.queries(query, spec_name, 1)
            return self._retriever.search(rewritten, n_results, where)
        rewrites = self._writer.queries(query, spec_name, MULTI_QUERY_COUNT)
        hit_lists = [
            self._retriever.search(text, n_results, where)
            for text in (query, *rewrites)
        ]
        return reciprocal_rank_fusion(hit_lists, n_results)


class ClaudeQueryWriter:
    def __init__(self, client: Any, model: str):
        self._client = client
        self.model = model

    def queries(
        self, question: str, spec_name: str | None, count: int
    ) -> tuple[str, ...]:
        instruction = (
            "Write one search query for this question."
            if count == 1
            else f"Write {count} different search queries for this question, "
            "each covering a different reading or aspect of it."
        )
        payload = self._ask(_prompt(question, spec_name, instruction), QUERIES_SCHEMA)
        queries = tuple(q.strip() for q in payload.get("queries", []) if q.strip())
        if not queries:
            raise QueryRewriteError("the rewriter returned no query")
        return queries[:count]

    def passage(self, question: str, spec_name: str | None) -> str:
        instruction = (
            "Write a short passage (60 to 100 words) as it could appear in an "
            "Icy Veins guide for this specialization, answering the question. "
            "It is only used to find similar passages: use the guides' usual "
            "vocabulary; exact numbers do not matter."
        )
        payload = self._ask(_prompt(question, spec_name, instruction), PASSAGE_SCHEMA)
        passage = str(payload.get("passage", "")).strip()
        if not passage:
            raise QueryRewriteError("the rewriter returned an empty passage")
        return passage

    def _ask(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=WRITER_MAX_TOKENS,
                system=WRITER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIError as exc:
            raise QueryRewriteError(f"rewrite request failed: {exc}") from exc
        if response.stop_reason != "end_turn":
            raise QueryRewriteError(f"rewriter stopped with {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QueryRewriteError(f"unparseable rewrite: {text[:100]!r}") from exc
        if not isinstance(payload, dict):
            raise QueryRewriteError(f"unexpected rewrite: {text[:100]!r}")
        return payload


def _prompt(question: str, spec_name: str | None, instruction: str) -> str:
    spec = spec_name or "not specified"
    return (
        f"<specialization>{spec}</specialization>\n"
        f"<question>{question}</question>\n{instruction}"
    )


class CachedQueryWriter:
    """Stores every rewrite on disk, so the relevance judge and the eval runs
    see exactly the same queries, and each rewrite is paid for once."""

    def __init__(self, writer: WritesQueries, path: Path, *, model: str):
        self._writer = writer
        self._path = path
        self._model = model
        self._entries: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    def queries(
        self, question: str, spec_name: str | None, count: int
    ) -> tuple[str, ...]:
        key = self._key("queries", question, spec_name, count)
        if key not in self._entries:
            self._store(key, list(self._writer.queries(question, spec_name, count)))
        return tuple(self._entries[key])

    def passage(self, question: str, spec_name: str | None) -> str:
        key = self._key("passage", question, spec_name, 1)
        if key not in self._entries:
            self._store(key, self._writer.passage(question, spec_name))
        return self._entries[key]

    def _key(self, kind: str, question: str, spec: str | None, count: int) -> str:
        raw = json.dumps([self._model, kind, question, spec, count], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _store(self, key: str, value: Any) -> None:
        self._entries = {**self._entries, key: value}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8"
        )
