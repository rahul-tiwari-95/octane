"""octane ask command."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from octane.cli._shared import console, err_console, _get_synapse, _try_daemon_route, _print_dag_trace


def register(app: typer.Typer) -> None:
    app.command()(ask)


def ask(
    query: str = typer.Argument(..., help="Your question or instruction"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DAG trace after response"),
    deep: str = typer.Option(
        None,
        "--deep",
        help="Deep mode with optional N rounds (e.g. --deep, --deep 5, --deep 20). Default: 3 rounds.",
    ),
    monitor: bool = typer.Option(False, "--monitor", help="Show live RAM/CPU/model metrics during query"),
    recall: bool = typer.Option(False, "--recall", "-r", help="Infer exclusively on stored Postgres/Redis data — no web search"),
):
    """🧠 Ask Octane anything — routed through OSA."""
    if recall:
        asyncio.run(_ask_recall(query))
        return
    from octane.cli._shared import parse_deep_flag
    deep_val = parse_deep_flag(deep)
    asyncio.run(_ask(query, verbose=verbose, deep_rounds=deep_val, monitor=monitor))


async def _ask(query: str, verbose: bool = False, deep_rounds: int | None = None, monitor: bool = False):
    if verbose:
        from octane.utils import setup_logging
        setup_logging(level_override="debug")

    from octane.daemon.client import is_daemon_running

    # ── Try daemon routing first ──────────────────────────────────────────────
    deep = deep_rounds is not None
    if is_daemon_running() and not deep and not monitor:
        console.print("[dim]📡 Routing through daemon...[/]")
        chunks_received = False
        try:
            async for chunk in _try_daemon_route("ask", {"query": query}):
                if not chunks_received:
                    console.print("[bold green]🔥 Octane:[/] ", end="")
                    chunks_received = True
                console.print(chunk, end="")
            if chunks_received:
                console.print("\n")
                return
        except Exception as exc:
            console.print(f"[yellow]⚠ Daemon error, falling back to direct: {exc}[/]")

    from octane.osa.orchestrator import Orchestrator
    from octane.tools.topology import ModelTier, detect_topology, get_topology

    synapse = _get_synapse()
    osa = Orchestrator(synapse)

    with console.status("[dim]Checking inference engine...[/]", spinner="dots"):
        status = await osa.pre_flight(wait_for_bodega=True)

    try:
        topo_name = detect_topology()
        topo = get_topology(topo_name)
        fast_model = topo.resolve(ModelTier.FAST)
        mid_model  = topo.resolve(ModelTier.MID)
        reason_model = topo.resolve(ModelTier.REASON)
    except Exception:
        topo_name = "?"
        fast_model = mid_model = reason_model = "?"

    if status["bodega_reachable"] and status["model_loaded"]:
        deep_tag = f" | [bold cyan]⬇ deep mode ({deep_rounds} rounds)[/]" if deep else ""
        monitor_tag = " | [bold yellow]📊 monitor[/]" if monitor else ""
        console.print(
            f"[dim]🧠 topology:[bold]{topo_name}[/bold] "
            f"FAST=[cyan]{fast_model}[/cyan] "
            f"MID=[cyan]{mid_model}[/cyan] "
            f"REASON=[cyan]{reason_model}[/cyan]"
            f"{deep_tag}{monitor_tag}[/]"
        )
    elif status["bodega_reachable"]:
        console.print(f"[yellow]⚠ Bodega reachable but no model loaded — using keyword fallback[/]")
    else:
        console.print(f"[yellow]⚠ Bodega offline — using keyword fallback[/]")

    console.print(f"\n[dim]Query: {query}[/]\n")

    # ── MSR clarification hook ─────────────────────────────────────────────
    async def clarification_hook(questions) -> str | None:
        """Interactive MCQ prompt for Multi-Shot Refinement."""
        n_total = len(questions)
        answers: list[str] = []
        console.print()
        console.print(Panel(
            f"[bold cyan]🎯 Octane wants to focus your deep search[/]\n"
            f"[dim]Answer {n_total} quick question{'s' if n_total != 1 else ''} to steer the research "
            f"(press Enter to skip any)[/]",
            border_style="cyan",
            padding=(0, 2),
        ))
        loop = asyncio.get_event_loop()
        for i, q in enumerate(questions, 1):
            console.print(f"\n[bold]({i}/{n_total})[/] [cyan]{q.text}[/]")
            option_letters = "ABCDEFGH"
            for j, opt in enumerate(q.options):
                console.print(f"  [bold]{option_letters[j]}[/]  {opt}")
            console.print("  [dim]↵  Skip[/]")
            raw_answer = await loop.run_in_executor(
                None, lambda: input("  Your choice: ").strip().upper()
            )
            if raw_answer:
                idx = ord(raw_answer[0]) - ord('A')
                if 0 <= idx < len(q.options):
                    answers.append(f"{q.text}: {q.options[idx]}")
                    console.print(f"  [green]✓[/] [dim]{q.options[idx]}[/]")
                else:
                    console.print("  [dim](skipped — unrecognised choice)[/]")
            else:
                console.print("  [dim](skipped)[/]")
        if answers:
            ctx = "; ".join(answers)
            console.print(f"\n[dim]🔍 Deep search steering: {ctx}[/]\n")
            return ctx
        console.print("\n[dim]No steering applied — running full-breadth deep search[/]\n")
        return None

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    _STAGES = [
        "🔀  Routing query…",
        "🌐  Fetching data…",
        "🧠  Synthesizing…",
    ]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20, complete_style="bold magenta", pulse_style="magenta"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    stage_task = progress.add_task(_STAGES[0], total=len(_STAGES), completed=0)
    progress.start()
    _stage_idx = 0

    def _advance_stage(label: str) -> None:
        nonlocal _stage_idx
        _stage_idx += 1
        progress.update(stage_task, description=label, completed=_stage_idx)

    full_output_parts = []
    first_token = True
    extra_meta = {"deep": True, "deep_rounds": deep_rounds} if deep else {}
    hook = clarification_hook if deep else None

    # ── --monitor: live metrics task ──────────────────────────────────────────
    _monitor_stop = asyncio.Event()

    async def _monitor_loop() -> None:
        """Poll system metrics every 2 s and print a live one-liner."""
        import psutil  # type: ignore[import]
        import time

        t0 = time.monotonic()
        try:
            while not _monitor_stop.is_set():
                try:
                    vm = psutil.virtual_memory()
                    cpu = psutil.cpu_percent(interval=None)
                    ram_used = vm.used / (1024 ** 3)
                    ram_total = vm.total / (1024 ** 3)
                    ram_pct = vm.percent
                    elapsed = time.monotonic() - t0
                    ram_col = "red" if ram_pct > 85 else "yellow" if ram_pct > 70 else "green"
                    cpu_col = "red" if cpu > 85 else "yellow" if cpu > 60 else "green"
                    line = (
                        f"[dim]📊 {elapsed:5.1f}s[/]  "
                        f"RAM [{ram_col}]{ram_used:.1f}/{ram_total:.0f} GB ({ram_pct:.0f}%)[/{ram_col}]  "
                        f"CPU [{cpu_col}]{cpu:.0f}%[/{cpu_col}]"
                    )
                    console.print(line)
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass

    if monitor:
        monitor_task: asyncio.Task | None = asyncio.ensure_future(_monitor_loop())
        try:
            import psutil as _psutil  # type: ignore[import]
            _psutil.cpu_percent(interval=None)
        except Exception:
            pass
    else:
        monitor_task = None

    _advance_stage(_STAGES[1])
    async for chunk in osa.run_stream(query, extra_metadata=extra_meta, clarification_hook=hook):
        if first_token:
            _advance_stage(_STAGES[2])
            progress.stop()
            if monitor_task:
                _monitor_stop.set()
                await asyncio.sleep(0)
            console.print("[bold green]🔥 Octane:[/] ", end="")
            first_token = False
        console.print(chunk, end="")
        full_output_parts.append(chunk)
    if first_token:
        progress.stop()
    if monitor_task:
        _monitor_stop.set()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    console.print("\n")

    # ── Auto-store deep findings ──────────────────────────────────────────────
    # When --deep, persist the synthesis to research_findings (fire-and-forget).
    # Web pages are stored automatically by WebAgent._store_pages() now that
    # Orchestrator._connect_memory_pg() also connects Router.pg.
    if deep and full_output_parts:
        output_text = "".join(full_output_parts)
        asyncio.ensure_future(_store_deep_finding(query, output_text, osa))

    recent = synapse.get_recent_traces(limit=1)
    if recent:
        t = recent[0]
        real_events = [e for e in t.events if e.correlation_id != "preflight"]
        egress = next((e for e in real_events if e.event_type == "egress"), None)
        dag_nodes = egress.payload.get("dag_nodes", "?") if egress and egress.payload else "?"
        dag_reason = egress.payload.get("dag_reasoning", "") if egress and egress.payload else ""

        _print_ask_footer(
            agents_used=t.agents_used,
            event_count=len(real_events),
            duration_ms=t.total_duration_ms,
            correlation_id=t.correlation_id,
        )

        if verbose:
            _print_dag_trace(t, real_events, dag_nodes, dag_reason)


async def _store_deep_finding(query: str, content: str, osa) -> None:
    """Persist a --deep synthesis result to research_findings (non-blocking)."""
    try:
        from octane.research.store import ResearchStore
        store = ResearchStore()
        await store.add_finding(
            task_id="ask_deep",
            topic=query,
            content=content,
            agents_used=list(osa.router.list_agents()),
            sources=[],
        )
    except Exception:
        pass  # never crash the pipeline over storage


def _print_ask_footer(
    agents_used: list[str],
    event_count: int,
    duration_ms: float,
    correlation_id: str,
) -> None:
    """Print a styled Rich footer after every octane ask response."""
    _AGENT_COLOURS = {
        "web": "cyan",
        "code": "yellow",
        "memory": "blue",
        "sysstat": "green",
        "pnl": "magenta",
        "osa": "dim",
        "user": "dim",
        "osa.decomposer": "dim",
        "osa.evaluator": "dim",
    }

    tags = Text()
    visible = [a for a in agents_used if a not in ("user", "osa", "osa.decomposer", "osa.evaluator")]
    for i, a in enumerate(visible):
        colour = _AGENT_COLOURS.get(a, "white")
        tags.append(f" {a} ", style=f"bold {colour} on grey15")
        if i < len(visible) - 1:
            tags.append("  ", style="")

    rule_text = Text()
    rule_text.append("  ")
    rule_text.append_text(tags)
    rule_text.append(f"   {duration_ms:.0f}ms", style="dim")
    rule_text.append(f"   trace: ", style="dim")
    rule_text.append(correlation_id[:16], style="bold dim")
    rule_text.append("  ", style="")

    console.print()
    console.print(Rule(rule_text, style="dim"))
    console.print(
        f"[dim]  Run [bold]octane trace {correlation_id[:8]}…[/bold] to inspect · "
        f"[bold]octane dag \"…\"[/bold] to preview routing[/]"
    )
    console.print()


# ── --recall: exclusive stored-data inference ─────────────────────────────────
# Pulls context from Postgres (research_findings + web_pages) and Redis
# (research task metadata) matching the query keywords, then streams synthesis
# directly through the 37b REASON-tier model.  Zero live web requests.

_RECALL_SYSTEM = """\
You are a research analyst with access to pre-stored research findings and web \
page summaries collected in previous sessions.

Your task:
1. Answer the question using ONLY the provided stored context below.
2. Cite which sources (research findings or web page titles) support each claim.
3. If the stored context is insufficient to answer confidently, state that clearly \
   rather than speculating.
4. Be analytical and concise. Use markdown headers where helpful.
"""

_MAX_RECALL_CHARS = 12_000  # chars of stored content to include in a single prompt

_RECALL_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those", "what", "which",
    "who", "when", "where", "how", "why", "all", "every", "any", "some",
    "no", "not", "only", "just", "give", "tell", "show", "summarize",
    "summarise", "everything", "stored", "based", "using", "find", "get",
    "me", "us", "about", "please",
})


def _extract_recall_keywords(query: str) -> list[str]:
    """Extract meaningful search tokens from a natural-language recall query.

    Strips stopwords, punctuation, and short words so that a sentence like
    'Summarize everything stored about NVDA' → ['nvda'] and the Postgres
    ILIKE search actually finds matching rows.
    """
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in query.split():
        clean = tok.strip(".,!?:;\"'()[]{}").lower()
        if len(clean) > 2 and clean not in _RECALL_STOPWORDS and clean not in seen:
            seen.add(clean)
            keywords.append(clean)
    return keywords if keywords else [query.lower()]


async def _ask_recall(query: str) -> None:
    """Stream an answer synthesized exclusively from stored Postgres/Redis data."""
    from octane.tools.pg_client import PgClient
    from octane.tools.bodega_router import BodegaRouter
    from octane.tools.topology import ModelTier, detect_topology, get_topology
    from octane.config import settings

    err_console.print(f"\n[bold cyan]📚 Recall mode[/] — querying stored data only\n")
    err_console.print(f"[dim]Query: {query}[/]\n")

    # ── 1. Pull context from Postgres ─────────────────────────────────────────
    pg = PgClient()
    await pg.connect()

    findings_chunks: list[str] = []
    pages_chunks: list[str] = []

    if pg.available:
        # Extract meaningful keywords — avoids sending the whole sentence as ILIKE
        # e.g. "Summarize everything stored about NVDA" → ["nvda"]
        keywords = _extract_recall_keywords(query)
        patterns = [f"%{kw}%" for kw in keywords]

        # research_findings
        findings = await pg.fetch(
            """
            SELECT topic, content, cycle_num, created_at
            FROM   research_findings
            WHERE  topic ILIKE ANY($1) OR content ILIKE ANY($1)
            ORDER  BY created_at DESC
            LIMIT  12
            """,
            patterns,
        )
        for row in findings:
            ts = str(row["created_at"])[:10]
            findings_chunks.append(
                f"[Research finding — {row['topic']} — cycle {row['cycle_num']} — {ts}]\n"
                + str(row["content"])
            )

        # web_pages
        pages = await pg.fetch(
            """
            SELECT title, url, content, fetched_at
            FROM   web_pages
            WHERE  content ILIKE ANY($1) OR title ILIKE ANY($1) OR url ILIKE ANY($1)
            ORDER  BY fetched_at DESC
            LIMIT  8
            """,
            patterns,
        )
        for row in pages:
            ts = str(row["fetched_at"])[:10]
            title = row["title"] or row["url"]
            pages_chunks.append(
                f"[Web page — {title} — {ts}]\n"
                + str(row["content"] or "")[:2000]
            )
    else:
        err_console.print("[yellow]⚠ Postgres unavailable — recall context will be empty[/]")

    await pg.close()

    # ── 2. Pull task list from Redis ──────────────────────────────────────────
    redis_tasks: list[str] = []
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        ids = await r.smembers("research:active")
        for tid in ids:
            raw = await r.get(f"research:task:{tid}")
            if raw:
                import json as _json
                meta = _json.loads(raw)
                t_topic = meta.get("topic", "")
                if query.lower() in t_topic.lower() or t_topic.lower() in query.lower():
                    redis_tasks.append(
                        f"[Active research task — {t_topic} — id: {tid} — cycles: {meta.get('cycle_count', 0)}]"
                    )
        await r.aclose()
    except Exception:
        pass  # Redis offline — not fatal, Postgres findings cover the gap

    # ── 3. Build context block ────────────────────────────────────────────────
    total_sources = len(findings_chunks) + len(pages_chunks)
    if total_sources == 0 and not redis_tasks:
        err_console.print(
            "[yellow]No stored data matches your query.[/]\n"
            "[dim]  Run [bold]octane research start \"<topic>\"[/bold] to build a research corpus.\n"
            "  Run [bold]octane ask \"<query>\"[/bold] to run a live query (stores pages automatically).[/]"
        )
        return

    context_parts: list[str] = []
    char_budget = _MAX_RECALL_CHARS

    if redis_tasks:
        block = "\n".join(redis_tasks)
        context_parts.append("## Active Research Tasks\n" + block)
        char_budget -= len(block)

    for chunk in findings_chunks:
        if char_budget <= 0:
            break
        context_parts.append(chunk[:char_budget])
        char_budget -= len(chunk)

    for chunk in pages_chunks:
        if char_budget <= 0:
            break
        context_parts.append(chunk[:char_budget])
        char_budget -= len(chunk)

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"## Stored Context\n\n{context}\n\n---\n\n## Question\n\n{query}"

    # ── 4. Print recall summary ───────────────────────────────────────────────
    err_console.print(
        f"[dim]Sources: [bold]{len(findings_chunks)}[/bold] research findings  ·  "
        f"[bold]{len(pages_chunks)}[/bold] web pages  ·  "
        f"[bold]{len(redis_tasks)}[/bold] active tasks  ·  "
        f"[bold]{len(context):,}[/bold] chars of context[/]\n"
    )

    # ── 5. Stream synthesis from REASON tier (37b) ────────────────────────────
    topo = get_topology(detect_topology())
    router = BodegaRouter(topology=topo)

    with err_console.status("[dim]Connecting to inference engine...[/]", spinner="dots"):
        health = await router._client.health()
        if health.get("status") != "ok":
            if health.get("status") == "unhealthy":
                err_console.print("[red]Bodega is online but no models are loaded. Run 'octane model load' first.[/]")
            else:
                err_console.print("[red]Bodega inference engine is offline. Start it first.[/]")
            return

    from octane.utils.response_templates import apply_template
    _recall_sys = apply_template(_RECALL_SYSTEM, "ask")
    console.print("[bold green]🔥 Octane (recall):[/] ", end="")
    word_count = 0
    try:
        async for chunk in router.chat_stream(
            prompt=prompt,
            system=_recall_sys,
            tier=ModelTier.REASON,
            max_tokens=2048,
        ):
            console.print(chunk, end="")
            word_count += chunk.count(" ")
    except Exception as exc:
        console.print(f"\n[red]Inference error: {exc}[/]")
        return

    console.print("\n")
    err_console.print(
        Rule(
            Text(
                f"  recall   {len(findings_chunks)}f·{len(pages_chunks)}p   ~{word_count}w  ",
                style="dim",
            ),
            style="dim",
        )
    )
    err_console.print(
        "[dim]  Run [bold]octane store findings[/bold] to browse research findings\n"
        "  Run [bold]octane store pages[/bold] to browse stored web pages\n"
        "  Run [bold]octane store stats[/bold] to see all Postgres/Redis data[/]"
    )
    console.print()
