"""The agent's `search_knowledge_base` tool: definition sent to Claude + executor.

Tool use in a nutshell: we describe the tool (name, description, JSON Schema of
its input) in every request. When Claude decides it needs it, the response
stops with `stop_reason == "tool_use"` and a `tool_use` block holding the
arguments. We run the search ourselves and send the passages back in a
`tool_result` block; Claude then continues with that information in context.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any, Protocol

from prestie.catalog import COVERED_SPECS
from prestie.ingestion.icy_veins.pages import ALL_PAGES
from prestie.knowledge.embeddings import EmbeddingError
from prestie.knowledge.filters import metadata_filter
from prestie.knowledge.store import SearchHit

SEARCH_TOOL_NAME = "search_knowledge_base"
RESULTS_PER_SEARCH = 5
MAX_QUERY_CHARS = 500
CONTENT_TYPES = tuple(sorted({page.content_type for page in ALL_PAGES}))
SPEC_KEYS = [spec.key for spec in COVERED_SPECS]
# What each guide page holds, so the model can pick a filter knowingly.
# A page type missing here fails at import time, on purpose.
CONTENT_TYPE_DESCRIPTIONS = {
    "beginner": "simplified 'easy mode' guide: basic rotation, beginner talents, "
    "basic stat priority",
    "leveling": "leveling to 90: leveling rotation by level, heirlooms, "
    "leveling talents",
    "mechanics": "spell glossary: what every ability and talent does",
    "mythic_plus": "Mythic+ dungeons: group utility (interrupts, stuns), macros, "
    "affixes",
    "overview": "spec overview, performance ratings, patch changes",
    "rotation": "full expert rotation per hero talent, opener, cooldowns, threat, "
    "runes and Runic Power",
    "stat_priority": "secondary stat priority per hero talent, stat breakdown, "
    "diminishing returns",
    "talents": "talent builds for raid, Mythic+ and delves, import strings, hero "
    "talents, PvP talents",
}
CONTENT_TYPE_HELP = (
    "Optional filter on the guide page type. Search without it first: semantic "
    "search already ranks passages across pages, and a wrong filter hides the "
    "answer (e.g. the simplified rotation is in 'beginner', not 'rotation'). "
    "Use it on a follow-up search when the first results missed. Page types:\n"
    + "\n".join(
        f"- {name}: {CONTENT_TYPE_DESCRIPTIONS[name]}" for name in CONTENT_TYPES
    )
)

SEARCH_TOOL: dict[str, Any] = {
    "name": SEARCH_TOOL_NAME,
    "description": (
        "Semantic search over Icy Veins guides (English, patch 12.1) for every "
        "class and specialization. Returns the most relevant guide passages with "
        "their section name and source URL. Use it for any question about "
        "rotation, talents, stats, cooldowns, mechanics, leveling or Mythic+. "
        "Phrase the query in English with English spell names. Call it several "
        "times with different queries when a question covers several topics."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, in English, e.g. 'secondary stat "
                "priority San'layn' or 'when to use Vampiric Blood'.",
            },
            "spec": {
                "type": "string",
                "enum": SPEC_KEYS,
                "description": "Only search this spec's guides. Set it to the "
                "player's spec (from the player context or get_character_state) "
                "unless the question is about another spec; without it, passages "
                "from every spec compete.",
            },
            "content_type": {
                "type": "string",
                "enum": list(CONTENT_TYPES),
                "description": CONTENT_TYPE_HELP,
            },
        },
        "required": ["query"],
    },
}


class Retrieves(Protocol):
    def search(
        self,
        query: str,
        n_results: int = ...,
        where: Mapping[str, Any] | None = ...,
    ) -> Sequence[SearchHit]: ...


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False


class KnowledgeBaseTool:
    definition = SEARCH_TOOL

    def __init__(self, retriever: Retrieves, n_results: int = RESULTS_PER_SEARCH):
        self._retriever = retriever
        self._n_results = n_results

    def run(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        """Never raises: failures go back to Claude as an error tool_result."""
        error = _validate(tool_input)
        if error:
            return ToolOutcome(error, is_error=True)
        query = tool_input["query"].strip()
        where = metadata_filter(tool_input.get("spec"), tool_input.get("content_type"))
        try:
            hits = self._retriever.search(query, n_results=self._n_results, where=where)
        except EmbeddingError as exc:
            return ToolOutcome(f"Search failed: {exc}", is_error=True)
        return ToolOutcome(format_results(query, hits))


def _validate(tool_input: Mapping[str, Any]) -> str | None:
    query = tool_input.get("query")
    if not isinstance(query, str) or not query.strip():
        return "Invalid input: 'query' must be a non-empty string."
    if len(query) > MAX_QUERY_CHARS:
        return f"Invalid input: 'query' must be at most {MAX_QUERY_CHARS} characters."
    spec = tool_input.get("spec")
    if spec is not None and spec not in SPEC_KEYS:
        return f"Invalid input: unknown spec {spec!r}. Valid values: {', '.join(SPEC_KEYS)}."
    content_type = tool_input.get("content_type")
    if content_type is not None and content_type not in CONTENT_TYPES:
        return (
            f"Invalid input: unknown content_type '{content_type}'. "
            f"Valid values: {', '.join(CONTENT_TYPES)}."
        )
    return None


def format_results(query: str, hits: Sequence[SearchHit]) -> str:
    if not hits:
        return f"No results in the knowledge base for: {query}"
    passages = "\n".join(
        f'<result index="{index}" section="{_attr(hit.metadata.get("section"))}"'
        f' source_url="{_attr(hit.metadata.get("source_url"))}">\n'
        f"{hit.text}\n</result>"
        for index, hit in enumerate(hits, start=1)
    )
    return f'<search_results query="{_attr(query)}">\n{passages}\n</search_results>'


def _attr(value: object) -> str:
    """Escape for a double-quoted attribute; keep apostrophes (San'layn) readable."""
    return escape(str(value or ""), quote=False).replace('"', "&quot;")
