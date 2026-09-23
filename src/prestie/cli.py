"""Command-line entry point: `prestie <command>`."""

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import anthropic
import chromadb

from prestie.agent.agent import Agent, AgentError, AgentReply, ToolCall
from prestie.agent.prompts import HERO_TALENTS, PlayerContext, build_system_prompt
from prestie.agent.tools import KnowledgeBaseTool
from prestie.config import ConfigError, Settings, load_settings
from prestie.evaluation.retrieval import (
    CaseResult,
    EvalCaseError,
    EvalReport,
    evaluate,
    load_cases,
)
from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES
from prestie.ingestion.icy_veins.parser import ParseError
from prestie.ingestion.icy_veins.scraper import (
    IcyVeinsScraper,
    ScrapeError,
    build_client,
    load_robots,
)
from prestie.ingestion.pipeline import IngestError, ingest_pages
from prestie.knowledge.embeddings import Embedder, EmbeddingError, VoyageEmbedder
from prestie.knowledge.retriever import Retriever
from prestie.knowledge.store import (
    DEFAULT_RESULTS,
    EmbeddingModelMismatchError,
    KnowledgeStore,
    SearchHit,
)

DEFAULT_CACHE_DIR = Path("data/raw/icy-veins")
DEFAULT_EVAL_CASES = Path("evals/retrieval_cases.json")
SNIPPET_CHARS = 300
EVAL_QUESTION_CHARS = 60
EVAL_SECTION_CHARS = 55
EXIT_COMMANDS = frozenset({"exit", "quit", "q"})
KNOWLEDGE_ERRORS = (
    ConfigError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    EvalCaseError,
    IngestError,
    ParseError,
)


def build_voyage_embedder(settings: Settings) -> Embedder:
    return VoyageEmbedder.from_api_key(
        settings.require_voyage_api_key(), settings.voyage_model
    )


build_embedder = build_voyage_embedder  # seam replaced by a fake in tests


def open_store(settings: Settings, *, reset: bool = False) -> KnowledgeStore:
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return KnowledgeStore.open(client, settings.voyage_model, reset=reset)


def open_retriever(settings: Settings) -> Retriever:
    return Retriever(build_embedder(settings), open_store(settings))


def build_agent(
    settings: Settings,
    player: PlayerContext,
    on_tool_call: Callable[[ToolCall], None],
) -> Agent:
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    except anthropic.AnthropicError as exc:
        raise ConfigError(
            "No Anthropic credentials: set ANTHROPIC_API_KEY in .env"
        ) from exc
    return Agent(
        client,
        model=settings.claude_model,
        system_prompt=build_system_prompt(player),
        tool=KnowledgeBaseTool(open_retriever(settings)),
        on_tool_call=on_tool_call,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prestie")
    commands = parser.add_subparsers(required=True)

    scrape = commands.add_parser(
        "scrape", help="Download Icy Veins guides into the local cache"
    )
    scrape.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    scrape.add_argument(
        "--force", action="store_true", help="Re-download pages already cached"
    )
    scrape.set_defaults(handler=_scrape)

    ingest = commands.add_parser(
        "ingest", help="Parse, chunk, embed and index the cached guides"
    )
    ingest.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ingest.add_argument(
        "--reset", action="store_true", help="Drop the index and rebuild it"
    )
    ingest.set_defaults(handler=_ingest)

    search = commands.add_parser(
        "search", help="Query the knowledge base (retrieval only, no LLM)"
    )
    search.add_argument("query")
    search.add_argument("-k", type=int, default=DEFAULT_RESULTS, dest="n_results")
    search.add_argument("--content-type", help="e.g. rotation, stat_priority")
    search.set_defaults(handler=_search)

    evaluation = commands.add_parser(
        "eval", help="Measure retrieval quality on a set of reference questions"
    )
    evaluation.add_argument("--cases", type=Path, default=DEFAULT_EVAL_CASES)
    evaluation.add_argument("-k", type=int, default=DEFAULT_RESULTS)
    evaluation.set_defaults(handler=_eval)

    chat = commands.add_parser("chat", help="Ask the Blood DK assistant (Claude + RAG)")
    chat.add_argument("--level", type=int, required=True, help="Character level")
    chat.add_argument("--hero-talent", choices=HERO_TALENTS)
    chat.add_argument("-q", "--question", help="Ask one question and exit")
    chat.add_argument("--verbose", action="store_true", help="Show token usage")
    chat.set_defaults(handler=_chat)
    return parser


def _scrape(args: argparse.Namespace) -> int:
    with build_client() as client:
        try:
            scraper = IcyVeinsScraper(
                client=client,
                cache=HtmlCache(args.cache_dir),
                robots=load_robots(client),
                sleep=time.sleep,
            )
            for page in BLOOD_DK_PAGES:
                result = scraper.fetch(page, force=args.force)
                status = "cached" if result.from_cache else "downloaded"
                print(f"[{status:>10}] {page.slug}")
        except ScrapeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


def _ingest(args: argparse.Namespace) -> int:
    try:
        settings = load_settings()
        embedder = build_embedder(settings)
        store = open_store(settings, reset=args.reset)
        reports = ingest_pages(
            BLOOD_DK_PAGES, HtmlCache(args.cache_dir), embedder, store
        )
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for report in reports:
        print(f"{report.slug}: {report.sections} sections -> {report.chunks} chunks")
    print(f"{store.count()} chunks indexed with {settings.voyage_model}")
    return 0


def _search(args: argparse.Namespace) -> int:
    where = {"content_type": args.content_type} if args.content_type else None
    try:
        retriever = open_retriever(load_settings())
        hits = retriever.search(args.query, n_results=args.n_results, where=where)
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not hits:
        print("No results (is the index empty? run `prestie ingest`).")
    for rank, hit in enumerate(hits, start=1):
        print(_format_hit(rank, hit))
    return 0


def _eval(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.cases)
        retriever = open_retriever(load_settings())
        report = evaluate(cases, retriever.search, k=args.k)
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_format_report(report))
    return 0


def _format_report(report: EvalReport) -> str:
    lines = [f"{'rank':>4}  {'question':<{EVAL_QUESTION_CHARS}}  top result (distance)"]
    lines.extend(_format_case(result) for result in report.results)
    cutoffs = sorted({1, 3, report.k})
    rates = " | ".join(f"hit@{at} = {report.hit_rate(at):.2f}" for at in cutoffs)
    lines.append(
        f"\n{rates} | MRR = {report.mrr:.2f}  "
        f"({report.answerable_count} answerable questions, k={report.k})"
    )
    return "\n".join(lines)


def _format_case(result: CaseResult) -> str:
    if not result.case.answerable:
        status = "n/a"
    else:
        status = str(result.rank) if result.found else "MISS"
    question = _truncate(result.case.question, EVAL_QUESTION_CHARS)
    top = result.hits[0] if result.hits else None
    top_text = (
        f"{_truncate(str(top.metadata.get('section', '?')), EVAL_SECTION_CHARS)}"
        f" ({top.distance:.3f})"
        if top
        else "-"
    )
    return f"{status:>4}  {question:<{EVAL_QUESTION_CHARS}}  {top_text}"


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _chat(args: argparse.Namespace) -> int:
    try:
        player = PlayerContext(level=args.level, hero_talent=args.hero_talent)
        agent = build_agent(load_settings(), player, on_tool_call=_print_tool_call)
    except (ValueError, *KNOWLEDGE_ERRORS) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    one_shot = args.question is not None
    if not one_shot:
        print(f"Prestie — {player.describe()}. Tape 'exit' pour quitter.")
    for question in [args.question] if one_shot else _read_questions():
        try:
            reply = agent.ask(question)
        except AgentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if one_shot:
                return 1
            continue
        print(_format_reply(reply, verbose=args.verbose))
    return 0


def _read_questions() -> Iterator[str]:
    while True:
        try:
            line = input("\ntoi> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in EXIT_COMMANDS:
            return
        if line:
            yield line


def _print_tool_call(call: ToolCall) -> None:
    content_type = call.input.get("content_type")
    scope = f" [{content_type}]" if content_type else ""
    print(f"  [recherche] {call.input.get('query', '?')}{scope}")


def _format_reply(reply: AgentReply, *, verbose: bool) -> str:
    lines = [f"\nprestie> {reply.text}"]
    if reply.truncated:
        lines.append("[réponse tronquée : limite de tokens atteinte]")
    if verbose:
        usage = reply.usage
        lines.append(
            f"[tokens] entrée: {usage.input_tokens} "
            f"(cache lu: {usage.cache_read_input_tokens}, "
            f"cache écrit: {usage.cache_creation_input_tokens}) "
            f"· sortie: {usage.output_tokens} · recherches: {len(reply.tool_calls)}"
        )
    return "\n".join(lines)


def _format_hit(rank: int, hit: SearchHit) -> str:
    body = " ".join(hit.text.split("\n\n", 1)[-1].split())
    snippet = body[:SNIPPET_CHARS] + ("…" if len(body) > SNIPPET_CHARS else "")
    part_count = hit.metadata.get("part_count", 1)
    part = f" (part {hit.metadata['part'] + 1}/{part_count})" if part_count > 1 else ""
    return (
        f"[{rank}] distance={hit.distance:.3f} | {hit.metadata['section']}{part}\n"
        f"    {hit.metadata['source_url']}\n"
        f"    {snippet}\n"
    )
