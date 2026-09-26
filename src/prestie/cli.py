"""Command-line entry point: `prestie <command>`."""

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import chromadb
import uvicorn

from prestie.agent.agent import (
    Agent,
    AgentError,
    AgentReply,
    FallbackRestart,
    TextDelta,
    Tool,
    ToolCall,
    ToolCallStarted,
    TurnFinished,
)
from prestie.agent.character_tool import (
    CharacterStateTool,
    ProvidesCharacterState,
    format_state,
)
from prestie.agent.prompts import HERO_TALENTS, PlayerContext, build_system_prompt
from prestie.agent.quest_tool import QuestDetailsTool
from prestie.agent.tool_labels import tool_call_label
from prestie.agent.tools import KnowledgeBaseTool
from prestie.api.app import create_app
from prestie.blizzard.client import GameDataClient, build_http_client
from prestie.blizzard.quests import QuestCache, QuestRepository
from prestie.character.state import CharacterStateError
from prestie.character.watcher import SavedVariablesWatcher
from prestie.config import ConfigError, Settings, load_settings
from prestie.evaluation import agent_cases, agent_checks, agent_judge, agent_runner
from prestie.evaluation.agent_cases import AgentCase, load_agent_cases
from prestie.evaluation.agent_judge import DEFAULT_JUDGE_MODEL, Judge
from prestie.evaluation.relevance import (
    RelevanceJudge,
    RelevanceJudgeError,
    pool_passages,
    record_judgments,
)
from prestie.evaluation.agent_runner import (
    RESULTS_FILE,
    HarnessChangedError,
    RunSummary,
    check_harness,
    ensure_state,
    read_jsonl,
    row_cost_usd,
    run_agent_eval,
)
from prestie.evaluation.retrieval import (
    CaseResult,
    EvalCaseError,
    EvalReport,
    evaluate,
    load_cases,
)
from prestie.ingestion.icy_veins.cache import HtmlCache
from prestie.catalog import COVERED_SPECS, spec_by_key
from prestie.ingestion.icy_veins.pages import ALL_PAGES, GuidePage, guide_pages
from prestie.ingestion.icy_veins.parser import ParseError
from prestie.ingestion.icy_veins.scraper import (
    IcyVeinsScraper,
    ScrapeError,
    build_client,
    load_robots,
)
from prestie.ingestion.pipeline import IngestError, ingest_pages
from prestie.knowledge.embeddings import Embedder, EmbeddingError, VoyageEmbedder
from prestie.knowledge.filters import metadata_filter
from prestie.knowledge.reranker import VoyageReranker
from prestie.knowledge.retriever import Retriever
from prestie.knowledge.store import (
    DEFAULT_RESULTS,
    EmbeddingModelMismatchError,
    KnowledgeStore,
    SearchHit,
)

DEFAULT_CACHE_DIR = Path("data/raw/icy-veins")
DEFAULT_EVAL_CASES = Path("evals/retrieval_cases.json")
DEFAULT_QUEST_CACHE_DIR = Path("data/blizzard/quests")
DEFAULT_API_PORT = 8000
LOCALHOST = "127.0.0.1"  # never 0.0.0.0: the API spends the player's API credits
# Native Windows window (pywebview), run with Windows' own uv.
WINDOW_LAUNCHER = Path(__file__).resolve().parents[2] / "desktop" / "prestie_window.py"
WINDOW_OPEN_DELAY_S = 1.5  # let uvicorn start before the page loads
SNIPPET_CHARS = 300
EVAL_QUESTION_CHARS = 60
EVAL_SECTION_CHARS = 55
EXIT_COMMANDS = frozenset({"exit", "quit", "q"})
DEFAULT_AGENT_CASES = Path("evals/agent_cases.json")
DEFAULT_AGENT_FLOW_DIR = Path(".claude/hillclimb/agent-qa")
BASELINE_VARIANT = "baseline"
VARIANT_PATTERN = re.compile(r"v\d+")
DEFAULT_REPS = 2
DEFAULT_EVAL_WORKERS = 4
EVAL_REQUEST_TIMEOUT_S = 180.0
EVAL_MAX_RETRIES = 4
# Grading code covered by the harness-approval gate.
HARNESS_PATHS = tuple(
    Path(module.__file__)
    for module in (agent_cases, agent_checks, agent_judge, agent_runner)
)
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


def build_reranker(settings: Settings, model: str) -> VoyageReranker:
    return VoyageReranker.from_api_key(settings.require_voyage_api_key(), model)


# Systems whose top passages are pooled for relevance judging: every retrieval
# variant compared in the eval, so no system is favored by the labels.
# (label, rerank model or None, spec filter)
JUDGE_POOL_SYSTEMS = (
    ("vector", None, False),
    ("vector+filter", None, True),
    ("rerank-lite+filter", "rerank-2.5-lite", True),
    ("rerank", "rerank-2.5", False),
    ("rerank+filter", "rerank-2.5", True),
)
DEFAULT_POOL_DEPTH = 3


def build_anthropic_client(settings: Settings, **options: Any) -> anthropic.Anthropic:
    try:
        return anthropic.Anthropic(api_key=settings.anthropic_api_key, **options)
    except anthropic.AnthropicError as exc:
        raise ConfigError(
            "No Anthropic credentials: set ANTHROPIC_API_KEY in .env"
        ) from exc


def build_agent(
    settings: Settings,
    player: PlayerContext | None,
    on_tool_call: Callable[[ToolCall], None],
    *,
    retriever: Retriever | None = None,
    client: anthropic.Anthropic | None = None,
    character_source: ProvidesCharacterState | None = None,
) -> Agent:
    """The single place the agent is wired, shared by `chat` and `eval-agent`.

    With a `player`, the context is written in the system prompt (manual mode).
    Without one, the agent reads it from the addon export (or from
    `character_source`, the evaluation's fixed state) and looks quests up in
    the Blizzard API (addon mode).
    """
    tools: list[Tool] = [KnowledgeBaseTool(retriever or open_retriever(settings))]
    if player is None:
        source = character_source or SavedVariablesWatcher(
            settings.require_saved_variables_path()
        )
        tools.extend([CharacterStateTool(source), build_quest_tool(settings)])
    return Agent(
        client or build_anthropic_client(settings),
        model=settings.claude_model,
        system_prompt=build_system_prompt(player),
        tools=tools,
        on_tool_call=on_tool_call,
    )


def build_quest_tool(settings: Settings) -> QuestDetailsTool:
    client_id, client_secret = settings.require_blizzard_credentials()
    client = GameDataClient(
        build_http_client(),
        client_id,
        client_secret,
        region=settings.blizzard_region,
    )
    return QuestDetailsTool(QuestRepository(client, QuestCache(DEFAULT_QUEST_CACHE_DIR)))


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
    _add_spec_argument(scrape)
    scrape.set_defaults(handler=_scrape)

    ingest = commands.add_parser(
        "ingest", help="Parse, chunk, embed and index the cached guides"
    )
    ingest.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ingest.add_argument(
        "--reset", action="store_true", help="Drop the index and rebuild it"
    )
    _add_spec_argument(ingest)
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
    evaluation.add_argument(
        "--spec-filter",
        action="store_true",
        help="Filter each search on the case's spec, as the agent does",
    )
    evaluation.add_argument(
        "--hand-labels-only",
        action="store_true",
        help="Ignore the LLM-judged relevant passages (judge-retrieval)",
    )
    evaluation.add_argument(
        "--rerank",
        metavar="MODEL",
        help="Rerank the vector candidates with this Voyage model "
        "(e.g. rerank-2.5, rerank-2.5-lite)",
    )
    evaluation.set_defaults(handler=_eval)

    judge = commands.add_parser(
        "judge-retrieval",
        help="Pool the top passages of every retrieval variant and have an LLM "
        "judge their relevance (completes the eval labels)",
    )
    judge.add_argument("--cases", type=Path, default=DEFAULT_EVAL_CASES)
    judge.add_argument("--depth", type=int, default=DEFAULT_POOL_DEPTH)
    judge.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    judge.set_defaults(handler=_judge_retrieval)

    chat = commands.add_parser("chat", help="Ask the Blood DK assistant (Claude + RAG)")
    chat.add_argument(
        "--level",
        type=int,
        help="Character level (manual mode); omit to read the character "
        "from the addon export",
    )
    chat.add_argument("--hero-talent", choices=HERO_TALENTS)
    chat.add_argument("-q", "--question", help="Ask one question and exit")
    chat.add_argument("--verbose", action="store_true", help="Show token usage")
    chat.set_defaults(handler=_chat)

    serve = commands.add_parser(
        "serve", help="Run the local API for the companion window (addon mode)"
    )
    serve.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    serve.add_argument(
        "--open",
        action="store_true",
        help="Open the native companion window (Windows, via WSL)",
    )
    serve.set_defaults(handler=_serve)

    watch = commands.add_parser(
        "watch", help="Print the character state each time the addon export changes"
    )
    watch.set_defaults(handler=_watch)

    agent_eval = commands.add_parser(
        "eval-agent", help="Run the agent on reference questions and grade the answers"
    )
    agent_eval.add_argument("--cases", type=Path, default=DEFAULT_AGENT_CASES)
    agent_eval.add_argument("--flow-dir", type=Path, default=DEFAULT_AGENT_FLOW_DIR)
    agent_eval.add_argument("--variant", default=BASELINE_VARIANT, type=_variant_name)
    agent_eval.add_argument("--reps", type=int, default=DEFAULT_REPS)
    agent_eval.add_argument("--only", help="Comma-separated case ids (pilot runs)")
    agent_eval.add_argument("--workers", type=int, default=DEFAULT_EVAL_WORKERS)
    agent_eval.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    agent_eval.add_argument(
        "--approve-harness",
        action="store_true",
        help="Record the current grading code as approved (after reviewing it)",
    )
    agent_eval.set_defaults(handler=_eval_agent)
    return parser


def _add_spec_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spec",
        choices=[spec.key for spec in COVERED_SPECS],
        help="Only this spec's guide pages (default: every covered spec)",
    )


def _selected_pages(spec_key: str | None) -> tuple[GuidePage, ...]:
    spec = spec_by_key(spec_key) if spec_key else None
    return guide_pages(spec) if spec else ALL_PAGES


def _variant_name(value: str) -> str:
    if value != BASELINE_VARIANT and not VARIANT_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "variant must be 'baseline' or v<N> (v1, v2...)"
        )
    return value


def _scrape(args: argparse.Namespace) -> int:
    with build_client() as client:
        try:
            scraper = IcyVeinsScraper(
                client=client,
                cache=HtmlCache(args.cache_dir),
                robots=load_robots(client),
                sleep=time.sleep,
            )
            for page in _selected_pages(args.spec):
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
            _selected_pages(args.spec), HtmlCache(args.cache_dir), embedder, store
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
        settings = load_settings()
        reranker = build_reranker(settings, args.rerank) if args.rerank else None
        retriever = Retriever(
            build_embedder(settings), open_store(settings), reranker=reranker
        )
        report = evaluate(
            cases,
            retriever.search,
            k=args.k,
            spec_filter=args.spec_filter,
            judged=not args.hand_labels_only,
        )
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_format_report(report, spec_filter=args.spec_filter, rerank=args.rerank))
    return 0


def _judge_retrieval(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.cases)
        settings = load_settings()
        embedder, store = build_embedder(settings), open_store(settings)
        systems = [
            (
                Retriever(
                    embedder,
                    store,
                    reranker=build_reranker(settings, model) if model else None,
                ),
                spec_filter,
            )
            for _, model, spec_filter in JUDGE_POOL_SYSTEMS
        ]
        judge = RelevanceJudge(build_eval_client(settings), model=args.judge_model)
        judgments = {}
        for index, case in enumerate(cases):
            hit_lists = [
                retriever.search(
                    case.question,
                    n_results=args.depth,
                    where=metadata_filter(case.spec) if spec_filter else None,
                )
                for retriever, spec_filter in systems
            ]
            passages = pool_passages(case, hit_lists, depth=args.depth)
            try:
                judgments[index] = judge.judge(case, passages)
            except RelevanceJudgeError as exc:
                print(f"error on {case.question[:50]!r}: {exc}", file=sys.stderr)
                continue
            relevant = sum(j.relevant for j in judgments[index])
            print(
                f"{len(passages):2} judged, {relevant} relevant  "
                f"{_truncate(case.question, EVAL_QUESTION_CHARS)}"
            )
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    record_judgments(args.cases, judgments, judge_model=judge.model)
    return 0


def _format_report(
    report: EvalReport, *, spec_filter: bool = False, rerank: str | None = None
) -> str:
    lines = [f"{'rank':>4}  {'question':<{EVAL_QUESTION_CHARS}}  top result (distance)"]
    lines.extend(_format_case(result) for result in report.results)
    cutoffs = sorted({1, 3, report.k})
    rates = " | ".join(f"hit@{at} = {report.hit_rate(at):.2f}" for at in cutoffs)
    lines.append(
        f"\n{rates} | MRR = {report.mrr:.2f}  "
        f"({report.answerable_count} answerable questions, k={report.k})"
    )
    lines.append(
        f"spec_precision@{report.k} = {report.spec_precision(report.k):.2f} "
        f"(spec filter {'on' if spec_filter else 'off'}, "
        f"rerank {rerank or 'off'})"
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
    if args.level is None and args.hero_talent:
        print("error: --hero-talent requires --level", file=sys.stderr)
        return 1
    try:
        player = (
            PlayerContext(level=args.level, hero_talent=args.hero_talent)
            if args.level is not None
            else None
        )
        # Tool calls are shown from the event stream (see _stream_reply).
        agent = build_agent(load_settings(), player, on_tool_call=_ignore_tool_call)
    except (ValueError, *KNOWLEDGE_ERRORS) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    one_shot = args.question is not None
    if not one_shot:
        who = player.describe() if player else "personnage lu depuis l'addon"
        print(f"Prestie — {who}. Tape 'exit' pour quitter.")
    for question in [args.question] if one_shot else _read_questions():
        try:
            _stream_reply(agent, question, verbose=args.verbose)
        except AgentError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            if one_shot:
                return 1
    return 0


def _stream_reply(agent: Agent, question: str, *, verbose: bool) -> None:
    """Print the answer as it is generated, with tool calls on their own lines."""
    print("\nprestie> ", end="", flush=True)
    for event in agent.ask_stream(question):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallStarted):
            print(f"\n{format_tool_call(event.call)}", flush=True)
        elif isinstance(event, FallbackRestart):
            print("\n[autre modèle, la réponse reprend]", flush=True)
        elif isinstance(event, TurnFinished):
            print(_format_footer(event.reply, verbose=verbose))


def _serve(args: argparse.Namespace) -> int:
    try:
        settings = load_settings()
        # Fail now rather than on the first question.
        saved_variables = settings.require_saved_variables_path()
        settings.require_blizzard_credentials()
        retriever = open_retriever(settings)
        client = build_anthropic_client(settings)
    except KNOWLEDGE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    app = create_app(
        agent_factory=lambda: build_agent(
            settings, None, _ignore_tool_call, retriever=retriever, client=client
        ),
        watcher_factory=lambda: SavedVariablesWatcher(saved_variables),
    )
    print(f"Prestie sur http://{LOCALHOST}:{args.port} (Ctrl+C pour arrêter)")
    if args.open:
        open_companion_window(args.port)
    uvicorn.run(app, host=LOCALHOST, port=args.port)
    return 0


def open_companion_window(port: int) -> None:
    """Launch the window once the server is up (uvicorn.run blocks this thread)."""
    timer = threading.Timer(WINDOW_OPEN_DELAY_S, _open_or_explain, args=(port,))
    timer.daemon = True
    timer.start()


def _open_or_explain(port: int) -> None:
    if not launch_native_window(port):
        print(f"Ouvre http://localhost:{port} dans ton navigateur.")


def launch_native_window(port: int) -> bool:
    """Run desktop/prestie_window.py with Windows' uv; False when not on WSL."""
    powershell = shutil.which("powershell.exe")
    wslpath = shutil.which("wslpath")
    if powershell is None or wslpath is None:
        return False
    converted = subprocess.run(
        [wslpath, "-w", str(WINDOW_LAUNCHER)], capture_output=True, text=True
    )
    script = converted.stdout.strip()
    if converted.returncode != 0 or not script:
        return False
    arguments = f"'run','{script}','--url','http://localhost:{port}'"
    subprocess.Popen(
        [
            powershell,
            "-NoProfile",
            "-Command",
            f"Start-Process uv -WindowStyle Hidden -ArgumentList {arguments}",
        ]
    )
    return True


def _watch(args: argparse.Namespace) -> int:
    try:
        watcher = SavedVariablesWatcher(load_settings().require_saved_variables_path())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Surveillance de {watcher.path} (Ctrl+C pour arrêter)")
    try:
        for state in watcher.watch(sleep=time.sleep, on_error=_print_state_error):
            print(f"\n[{time.strftime('%H:%M:%S')}] nouvel état")
            print(format_state(state, datetime.now(UTC)))
    except KeyboardInterrupt:
        pass
    return 0


def _print_state_error(error: CharacterStateError) -> None:
    print(f"error: {error}", file=sys.stderr)


def _eval_agent(args: argparse.Namespace) -> int:
    try:
        cases = _select_cases(load_agent_cases(args.cases), args.only)
        state_path = ensure_state(args.flow_dir)
        check_harness(state_path, HARNESS_PATHS, approve=args.approve_harness)
        settings = load_settings()
        client = build_eval_client(settings)
        retriever = open_retriever(settings)
        summary = run_agent_eval(
            cases,
            agent_factory=lambda case, source: build_agent(
                settings,
                case.player(),
                _ignore_tool_call,
                retriever=retriever,
                client=client,
                character_source=source,
            ),
            judge=Judge(client, model=args.judge_model),
            variant_dir=args.flow_dir / args.variant,
            reps=args.reps,
            workers=args.workers,
            expected_model=settings.claude_model,
        )
    except (HarnessChangedError, *KNOWLEDGE_ERRORS) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rows = read_jsonl(args.flow_dir / args.variant / RESULTS_FILE)
    print(_format_eval_summary(summary, rows, args.flow_dir))
    return 0


def build_eval_client(settings: Settings) -> anthropic.Anthropic:
    return build_anthropic_client(
        settings, timeout=EVAL_REQUEST_TIMEOUT_S, max_retries=EVAL_MAX_RETRIES
    )


def _select_cases(
    cases: tuple[AgentCase, ...], only: str | None
) -> tuple[AgentCase, ...]:
    if not only:
        return cases
    wanted = [case_id.strip() for case_id in only.split(",") if case_id.strip()]
    known = {case.id for case in cases}
    unknown = [case_id for case_id in wanted if case_id not in known]
    if unknown:
        raise EvalCaseError(f"Unknown case ids: {', '.join(unknown)}")
    return tuple(case for case in cases if case.id in wanted)


def _ignore_tool_call(call: ToolCall) -> None:
    return None


def _format_eval_summary(
    summary: RunSummary, rows: list[dict[str, Any]], flow_dir: Path
) -> str:
    if summary.pass_rate is None:
        headline = "no graded answers"
    else:
        ci = (
            f" ± {summary.ci_half_width:.2f}"
            if summary.ci_half_width is not None
            else ""
        )
        headline = f"pass = {summary.pass_rate:.2f}{ci} over {summary.cases} cases"
    metrics = "  ".join(
        f"{name}={value:.2f}"
        for name, value in summary.metric_means.items()
        if name != "pass"
    )
    cost = sum(row_cost_usd(row) for row in rows)
    return "\n".join(
        [
            f"{summary.attempts_run} attempts run, {summary.errors} errors (see errors.jsonl)",
            headline,
            metrics,
            f"measured cost of all rows: ${cost:.2f}",
            f"report: node <claude-api skill>/shared/evals/report/build-report-lite.mjs {flow_dir}",
        ]
    )


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


def format_tool_call(call: ToolCall) -> str:
    return f"  {tool_call_label(call)}"


def _format_footer(reply: AgentReply, *, verbose: bool) -> str:
    lines = [""]
    if reply.refused:
        lines.append(reply.text)
    if reply.truncated:
        lines.append("[réponse tronquée : limite de tokens atteinte]")
    if verbose:
        usage = reply.usage
        lines.append(
            f"[tokens] entrée totale: {usage.total_input_tokens} "
            f"(lus depuis le cache: {usage.cache_read_input_tokens}, "
            f"écrits en cache: {usage.cache_creation_input_tokens}, "
            f"hors cache: {usage.input_tokens}) "
            f"· sortie: {usage.output_tokens} · appels d'outils: {len(reply.tool_calls)}"
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
