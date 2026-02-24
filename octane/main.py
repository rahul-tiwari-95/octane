"""Octane CLI — the user interface.

Commands:
    octane health    — System status (SysStat Agent)
    octane ask       — Ask a question (routed through OSA)
    octane chat      — Interactive multi-turn chat session
    octane session   — Chat until END, then print full annotated session replay
    octane trace     — View Synapse trace for a query
    octane agents    — List registered agents
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from octane.utils import setup_logging

# Initialize logging on import
setup_logging()

app = typer.Typer(
    name="octane",
    help="🔥 Octane — Local-first agentic OS for Apple Silicon",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _get_synapse():
    """Get or create the global SynapseEventBus."""
    from octane.models.synapse import SynapseEventBus
    if not hasattr(_get_synapse, "_bus"):
        _get_synapse._bus = SynapseEventBus()
    return _get_synapse._bus


# ── octane health ─────────────────────────────────────────────


@app.command()
def health():
    """🩺 System health — RAM, CPU, loaded model, server status."""
    asyncio.run(_health())


async def _health():
    from octane.agents.sysstat.agent import SysStatAgent
    from octane.models.schemas import AgentRequest
    from octane.tools.bodega_inference import BodegaInferenceClient

    synapse = _get_synapse()
    bodega = BodegaInferenceClient()

    try:
        agent = SysStatAgent(synapse, bodega)
        request = AgentRequest(query="health check", source="cli")
        response = await agent.run(request)

        if response.success:
            # Build rich output
            data = response.data
            system = data.get("system", {})
            model = data.get("model", {})
            server_health = data.get("server_health", {})

            # System metrics table
            sys_table = Table(show_header=False, box=None, padding=(0, 2))
            sys_table.add_column("Metric", style="cyan")
            sys_table.add_column("Value", style="white")

            ram_used = system.get("ram_used_gb", 0)
            ram_total = system.get("ram_total_gb", 0)
            ram_pct = system.get("ram_percent", 0)
            ram_color = "green" if ram_pct < 70 else "yellow" if ram_pct < 90 else "red"

            sys_table.add_row("RAM", f"[{ram_color}]{ram_used:.1f} / {ram_total:.1f} GB ({ram_pct}%)[/]")
            sys_table.add_row("CPU", f"{system.get('cpu_percent', '?')}% ({system.get('cpu_count', '?')} cores)")
            sys_table.add_row("RAM Available", f"{system.get('ram_available_gb', 0):.1f} GB")

            console.print(Panel(sys_table, title="[bold cyan]⚙ System Resources[/]", border_style="cyan"))

            # Bodega status
            bodega_table = Table(show_header=False, box=None, padding=(0, 2))
            bodega_table.add_column("Metric", style="magenta")
            bodega_table.add_column("Value", style="white")

            server_status = server_health.get("status", "unknown")
            status_emoji = "✅" if server_status == "ok" else "⚠️"
            bodega_table.add_row("Server", f"{status_emoji} {server_status}")

            if "error" in model:
                bodega_table.add_row("Model", f"[red]⚠ {model['error']}[/]")
            else:
                model_name = model.get("model", model.get("model_path", "unknown"))
                bodega_table.add_row("Model", f"[green]{model_name}[/]")

            console.print(Panel(bodega_table, title="[bold magenta]🧠 Bodega Inference Engine[/]", border_style="magenta"))

            # Model topology recommendation
            topology = data.get("topology", {})
            if topology:
                topo_table = Table(show_header=False, box=None, padding=(0, 2))
                topo_table.add_column("Role", style="yellow")
                topo_table.add_column("Model", style="white")
                topo_table.add_row(
                    "Tier",
                    f"[bold]{topology.get('tier', '?')}[/] — {topology.get('description', '')}",
                )
                topo_table.add_row("RAM Available", f"{topology.get('ram_gb', '?')} GB")
                for role, model_id in topology.get("models", {}).items():
                    topo_table.add_row(role.capitalize(), f"[dim]{model_id}[/]")
                console.print(Panel(topo_table, title="[bold yellow]⚡ Recommended Topology[/]", border_style="yellow"))

        else:
            console.print(f"[red]Health check failed: {response.error}[/]")

        console.print(f"\n[dim]Duration: {response.duration_ms}ms | Correlation: {response.correlation_id}[/]")

    finally:
        await bodega.close()


# ── octane ask ────────────────────────────────────────────────


def _print_dag_trace(trace, events, dag_nodes, dag_reason: str) -> None:
    """Print a compact DAG execution trace table for --verbose mode."""
    from rich.table import Table

    # DAG summary
    console.print()
    console.print(Panel(
        f"[bold]DAG nodes:[/] {dag_nodes}  ·  [bold]Reasoning:[/] {dag_reason[:120] or 'keyword fallback'}",
        title="[bold dim]⚙ DAG Execution Trace[/]",
        border_style="dim",
    ))

    # Dispatch + result events
    dispatch_events = [e for e in events if e.event_type in ("dispatch", "egress")]
    if dispatch_events:
        tbl = Table(show_header=True, box=None, padding=(0, 2))
        tbl.add_column("Δt", style="dim", width=9, justify="right")
        tbl.add_column("Step", style="cyan", width=20)
        tbl.add_column("Agent", style="green", width=14)
        tbl.add_column("Detail", style="white")

        t0 = trace.started_at
        for e in dispatch_events:
            offset = f"+{(e.timestamp - t0).total_seconds() * 1000:.0f}ms" if t0 else "—"
            if e.event_type == "dispatch":
                agent = e.target or "—"
                detail = (e.payload or {}).get("instruction", "")[:80]
            else:
                agent = "evaluator"
                agents_used = (e.payload or {}).get("agents_used", [])
                ok = (e.payload or {}).get("tasks_succeeded", "?")
                total = (e.payload or {}).get("tasks_total", "?")
                detail = f"[dim]agents={agents_used} {ok}/{total} succeeded[/]"

            tbl.add_row(offset, e.event_type, agent, detail)
        console.print(tbl)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question or instruction"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DAG trace after response"),
):
    """🧠 Ask Octane anything — routed through OSA."""
    asyncio.run(_ask(query, verbose=verbose))


async def _ask(query: str, verbose: bool = False):
    from octane.osa.orchestrator import Orchestrator

    synapse = _get_synapse()
    osa = Orchestrator(synapse)

    # Pre-flight check — show Bodega status once
    with console.status("[dim]Checking inference engine...[/]", spinner="dots"):
        status = await osa.pre_flight()

    if status["bodega_reachable"] and status["model_loaded"]:
        model_display = status.get("model") or "unknown"
        # Trim long model paths for display
        if model_display and "/" in model_display:
            model_display = model_display.split("/")[-1]
        console.print(f"[dim]🧠 {model_display} | LLM decomposition + synthesis active[/]")
    elif status["bodega_reachable"]:
        console.print(f"[yellow]⚠ Bodega reachable but no model loaded — using keyword fallback[/]")
    else:
        console.print(f"[yellow]⚠ Bodega offline — using keyword fallback[/]")

    console.print(f"\n[dim]Processing: {query}[/]\n")

    console.print("[bold green]🔥 Octane:[/] ", end="")
    full_output_parts = []
    async for chunk in osa.run_stream(query):
        console.print(chunk, end="")
        full_output_parts.append(chunk)
    console.print("\n")

    # Show trace summary with the correlation ID for octane trace
    recent = synapse.get_recent_traces(limit=1)
    if recent:
        t = recent[0]
        # Filter out preflight
        real_events = [e for e in t.events if e.correlation_id != "preflight"]
        # Extract DAG info from egress event
        egress = next((e for e in real_events if e.event_type == "egress"), None)
        dag_nodes = egress.payload.get("dag_nodes", "?") if egress and egress.payload else "?"
        dag_reason = egress.payload.get("dag_reasoning", "") if egress and egress.payload else ""

        console.print(
            f"\n[dim]Agents: {', '.join(t.agents_used)} | "
            f"Events: {len(real_events)} | "
            f"Duration: {t.total_duration_ms}ms | "
            f"Trace ID: [bold]{t.correlation_id}[/][/]"
        )

        if verbose:
            _print_dag_trace(t, real_events, dag_nodes, dag_reason)


# ── octane trace ──────────────────────────────────────────────


@app.command()
def trace(
    correlation_id: str = typer.Argument(
        None,
        help="Correlation ID to trace. If omitted, shows recent traces from disk.",
    ),
):
    """🔍 View Synapse trace for a query lifecycle."""
    synapse = _get_synapse()

    if correlation_id:
        t = synapse.get_trace(correlation_id)
        if not t.events:
            console.print(f"[yellow]No trace found for: {correlation_id}[/]")
            return

        started = t.started_at.strftime("%H:%M:%S") if t.started_at else "?"
        console.print(Panel(
            f"[bold]Correlation ID:[/] {t.correlation_id}\n"
            f"[bold]Started:[/] {started}\n"
            f"[bold]Duration:[/] {t.total_duration_ms}ms\n"
            f"[bold]Success:[/] {'✅' if t.success else '❌'}\n"
            f"[bold]Agents:[/] {', '.join(t.agents_used)}",
            title="[bold blue]🔍 Synapse Trace[/]",
            border_style="blue",
        ))

        table = Table(title="Events", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Timestamp", style="dim", width=12)
        table.add_column("Type", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Target", style="magenta")
        table.add_column("Info", style="white")

        t0 = t.started_at
        for i, event in enumerate(t.events, 1):
            if t0:
                offset_ms = (event.timestamp - t0).total_seconds() * 1000
                ts_display = f"+{offset_ms:.0f}ms"
            else:
                ts_display = "—"

            # Summarise the payload into one short line
            info = ""
            if event.error:
                info = f"[red]ERR: {event.error[:60]}[/]"
            elif event.payload:
                # Pick the most meaningful payload field
                for key in ("reasoning", "template", "output_preview", "query", "error"):
                    if key in event.payload:
                        val = str(event.payload[key])[:80]
                        info = f"[dim]{key}:[/] {val}"
                        break

            table.add_row(
                str(i),
                ts_display,
                event.event_type,
                event.source,
                event.target or "—",
                info,
            )

        console.print(table)
    else:
        # Show recent traces from disk (cross-process)
        trace_ids = synapse.list_traces(limit=15)
        if not trace_ids:
            console.print("[yellow]No traces found. Run 'octane ask' first.[/]")
            return

        table = Table(title="Recent Traces  (from ~/.octane/traces/)")
        table.add_column("Correlation ID", style="cyan")
        table.add_column("Events", justify="right")
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Status", justify="center")

        for cid in trace_ids:
            if cid == "preflight":
                continue
            t = synapse.get_trace(cid)
            table.add_row(
                t.correlation_id,
                str(len(t.events)),
                f"{t.total_duration_ms:.0f}ms",
                "✅" if t.success else "❌",
            )

        console.print(table)
        console.print("[dim]Run: octane trace <correlation_id>  for full event log[/]")


# ── octane chat ───────────────────────────────────────────────


@app.command()
def chat():
    """💬 Interactive multi-turn chat session with Octane."""
    asyncio.run(_chat())


async def _chat():
    from octane.osa.orchestrator import Orchestrator

    synapse = _get_synapse()
    osa = Orchestrator(synapse)
    session_id = f"chat_{int(__import__('time').time())}"

    # Rolling conversation buffer — last 6 turns injected into every Evaluator call
    conversation_history: list[dict[str, str]] = []

    console.print(Panel(
        "[bold green]Octane Chat[/]\n"
        "[dim]Type your message and press Enter. "
        "Type [bold]exit[/bold] or [bold]quit[/bold] to end the session.[/]",
        border_style="green",
    ))

    # Pre-flight once at session start
    with console.status("[dim]Starting up...[/]", spinner="dots"):
        status = await osa.pre_flight()

    if status["bodega_reachable"] and status["model_loaded"]:
        model_display = (status.get("model") or "").split("/")[-1] or "model loaded"
        console.print(f"[dim]🧠 {model_display} ready[/]\n")
    else:
        note = status.get("note", "Bodega offline")
        console.print(f"[yellow]⚠ {note}[/]\n")

    turn = 0
    while True:
        try:
            query = console.input("[bold cyan]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/]")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "bye", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        turn += 1
        # Add user turn to history before processing
        conversation_history.append({"role": "user", "content": query})

        console.print(f"\n[bold green]Octane:[/] ", end="")
        response_parts: list[str] = []
        async for chunk in osa.run_stream(
            query,
            session_id=session_id,
            conversation_history=conversation_history,
        ):
            console.print(chunk, end="")
            response_parts.append(chunk)
        console.print()  # newline after stream ends
        console.print()

        # Add assistant turn to history
        assistant_reply = "".join(response_parts).strip()
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Keep history bounded to last 12 entries (6 turns)
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

    console.print(f"[dim]Session {session_id} — {turn} turn(s)[/]")


# ── octane feedback ───────────────────────────────────────────


@app.command()
def feedback(
    signal: str = typer.Argument(..., help="thumbs_up or thumbs_down"),
    trace_id: str = typer.Argument(None, help="Correlation ID from a previous response"),
    user_id: str = typer.Option("default", "--user", "-u", help="User ID"),
):
    """👍 Record feedback on a response to improve future answers.

    Examples:
        octane feedback thumbs_up
        octane feedback thumbs_down <trace_id>
    """
    asyncio.run(_feedback(signal, trace_id, user_id))


async def _feedback(signal: str, trace_id: str | None, user_id: str):
    from octane.osa.router import Router

    valid_signals = {"thumbs_up", "thumbs_down"}
    if signal not in valid_signals:
        console.print(f"[red]Unknown signal '{signal}'. Use: thumbs_up or thumbs_down[/]")
        raise typer.Exit(1)

    synapse = _get_synapse()
    router = Router(synapse)
    pnl_agent = router.get_agent("pnl")

    if not pnl_agent:
        console.print("[red]PnL agent not available.[/]")
        raise typer.Exit(1)

    from octane.models.schemas import AgentRequest
    query = f"feedback {signal}"
    metadata: dict = {"user_id": user_id}
    if trace_id:
        metadata["correlation_id"] = trace_id

    response = await pnl_agent.execute(
        AgentRequest(query=query, source="cli", metadata=metadata)
    )

    score_display = response.data.get("score", "?") if response.data else "?"
    emoji = "👍" if signal == "thumbs_up" else "👎"
    console.print(f"{emoji} [green]{response.output}[/]")

    # Show if a preference nudge happened (score reset to 0 = nudge fired)
    if isinstance(score_display, int) and score_display == 0:
        console.print("[dim]Preference nudge applied — verbosity updated.[/]")


# ── octane session ────────────────────────────────────────────


@app.command()
def session():
    """🧬 Chat session with full annotated replay when you type END.

    Ask questions back and forth. Type END to stop and see the complete
    trace of every agent decision, Redis write, and event for the session.
    """
    asyncio.run(_session())


async def _session():
    from octane.osa.orchestrator import Orchestrator
    from octane.models.synapse import SynapseEventBus

    import time

    # Fresh synapse bus so we capture only this session's events
    synapse = SynapseEventBus()
    osa = Orchestrator(synapse)
    session_id = f"session_{int(time.time())}"

    console.print(Panel(
        "[bold green]Octane Session[/]\n"
        "[dim]Ask anything. Type [bold]END[/bold] when done to see the full replay.[/]",
        border_style="green",
    ))

    with console.status("[dim]Starting up...[/]", spinner="dots"):
        status = await osa.pre_flight()

    model_display = (status.get("model") or "").split("/")[-1] or "model loaded"
    if status["bodega_reachable"] and status["model_loaded"]:
        console.print(f"[dim]🧠 {model_display} · session [bold]{session_id}[/bold][/]\n")
    else:
        console.print(f"[yellow]⚠ {status.get('note', 'Bodega offline')}[/]\n")

    # ── Turn loop ──────────────────────────────────────────────
    turns: list[dict] = []   # {turn, query, correlation_id, output}

    turn = 0
    while True:
        try:
            query = console.input("[bold cyan]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session interrupted.[/]")
            break

        if not query:
            continue

        if query.strip().upper() == "END":
            break

        turn += 1
        cids_before = set(synapse.get_all_correlation_ids())

        console.print(f"\n[bold green]Octane:[/] ", end="")
        chunks = []
        async for chunk in osa.run_stream(query, session_id=session_id):
            console.print(chunk, end="")
            chunks.append(chunk)
        console.print("\n")

        # Find the new correlation_id emitted during this turn
        cids_after = set(synapse.get_all_correlation_ids())
        new_cids = cids_after - cids_before - {"preflight"}
        cid = next(iter(new_cids)) if new_cids else "unknown"

        turns.append({
            "turn": turn,
            "query": query,
            "correlation_id": cid,
            "output": "".join(chunks).strip(),
        })

    if not turns:
        console.print("[dim]No turns to replay.[/]")
        return

    # ── Full session replay ────────────────────────────────────
    console.print("\n")
    console.rule("[bold yellow]🧬 SESSION REPLAY[/]", style="yellow")
    console.print(f"[dim]Session ID: {session_id} · {len(turns)} turn(s)[/]\n")

    # Collect all Redis memory keys written during this session
    redis_writes: list[str] = []
    try:
        from octane.tools.redis_client import RedisClient
        redis = RedisClient()
        redis._use_fallback = False  # try real Redis
        pattern = f"memory:{session_id}:*"
        redis_keys = await redis.keys_matching(pattern)
        redis_writes = sorted(redis_keys)
    except Exception:
        pass

    for t in turns:
        trace = synapse.get_trace(t["correlation_id"])

        # ── Turn header ───────────────────────────────────────
        console.print(Panel(
            f"[bold]Turn {t['turn']}[/]  ·  [cyan]{t['query']}[/]\n"
            f"[dim]Trace: {t['correlation_id']}  ·  {trace.total_duration_ms:.0f}ms  ·  "
            f"Agents: {', '.join(a for a in trace.agents_used if a not in ('user', 'osa', ''))}[/]",
            border_style="cyan",
            title=f"[bold cyan]Turn {t['turn']}[/]",
        ))

        # ── Event timeline ────────────────────────────────────
        event_table = Table(show_lines=True, box=None, padding=(0, 1))
        event_table.add_column("Δt", style="dim", width=9, justify="right")
        event_table.add_column("Event", style="cyan", width=24)
        event_table.add_column("Source → Target", style="green", width=28)
        event_table.add_column("Detail", style="white")

        t0 = trace.started_at
        for event in trace.events:
            if event.correlation_id == "preflight":
                continue

            offset = f"+{(event.timestamp - t0).total_seconds() * 1000:.0f}ms" if t0 else "—"
            src_tgt = f"{event.source}"
            if event.target:
                src_tgt += f" → {event.target}"

            # Build a human-readable detail line from the payload
            detail = ""
            if event.error:
                detail = f"[red]ERR: {event.error[:80]}[/]"
            elif event.payload:
                # Pick the most informative field in priority order
                for key in ("template", "reasoning", "output_preview", "query",
                            "approach", "agents_used", "tasks_succeeded"):
                    if key in event.payload:
                        val = str(event.payload[key])
                        label = {
                            "template":        "→ template",
                            "reasoning":       "reasoning",
                            "output_preview":  "output",
                            "query":           "query",
                            "approach":        "plan",
                            "agents_used":     "agents",
                            "tasks_succeeded": "succeeded",
                        }.get(key, key)
                        detail = f"[dim]{label}:[/] {val[:90]}"
                        break

            event_table.add_row(offset, event.event_type, src_tgt, detail)

        console.print(event_table)

        # ── Final answer ──────────────────────────────────────
        console.print(f"\n[bold green]Answer:[/] {t['output']}\n")

    # ── Redis memory written this session ─────────────────────
    console.rule("[bold magenta]🧠 Redis Memory Written[/]", style="magenta")
    if redis_writes:
        mem_table = Table(show_header=False, box=None, padding=(0, 2))
        mem_table.add_column("Key", style="magenta")
        mem_table.add_column("Value", style="white")

        redis2 = RedisClient()
        for key in redis_writes:
            try:
                raw = await redis2.get(key)
                import json
                val = json.loads(raw) if raw else {}
                answer = val.get("answer", raw or "—")
                if len(answer) > 100:
                    answer = answer[:97] + "..."
                mem_table.add_row(key, answer)
            except Exception:
                mem_table.add_row(key, "[dim](unreadable)[/]")

        console.print(mem_table)
    else:
        console.print("[dim]No memory keys found for this session (Redis may be offline).[/]")

    console.print(f"\n[dim]Session {session_id} complete.[/]")


# ── octane agents ─────────────────────────────────────────────


@app.command()
def agents():
    """📋 List all registered agents."""
    from octane.osa.router import Router

    synapse = _get_synapse()
    router = Router(synapse)

    table = Table(title="Registered Agents")
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")

    for name in router.list_agents():
        table.add_row(name, "✅ registered")

    console.print(table)


# ── octane version ────────────────────────────────────────────


@app.command()
def version():
    """📦 Show Octane version."""
    from octane import __version__
    console.print(f"[bold cyan]🔥 Octane[/] v{__version__}")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    app()
