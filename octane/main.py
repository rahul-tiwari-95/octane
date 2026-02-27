"""Octane CLI — the user interface.

Commands:
    octane health    — System status (SysStat Agent)
    octane ask       — Ask a question (routed through OSA)
    octane chat      — Interactive multi-turn chat session
    octane session   — Chat until END, then print full annotated session replay
    octane trace     — View Synapse trace for a query (visual timeline)
    octane dag       — Dry-run decomposition — show task DAG without executing
    octane pref      — Manage user preferences (show / set / reset)
    octane agents    — List registered agents
    octane version   — Show version info
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
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

    console.print(f"\n[dim]Query: {query}[/]\n")

    # Spinner runs while guard → decompose → dispatch → first Evaluator token
    # Stops automatically the moment the first streamed chunk arrives.
    status = console.status("[dim]⚙  Routing and dispatching...[/]", spinner="dots")
    status.start()
    full_output_parts = []
    first_token = True
    async for chunk in osa.run_stream(query):
        if first_token:
            status.stop()
            console.print("[bold green]🔥 Octane:[/] ", end="")
            first_token = False
        console.print(chunk, end="")
        full_output_parts.append(chunk)
    if first_token:
        status.stop()  # pipeline ran but no tokens yielded (guard block, etc.)
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

        _print_ask_footer(
            agents_used=t.agents_used,
            event_count=len(real_events),
            duration_ms=t.total_duration_ms,
            correlation_id=t.correlation_id,
        )

        if verbose:
            _print_dag_trace(t, real_events, dag_nodes, dag_reason)


def _print_ask_footer(
    agents_used: list[str],
    event_count: int,
    duration_ms: float,
    correlation_id: str,
) -> None:
    """Print a styled Rich footer after every octane ask response.

    Replaces the old dim one-liner. Shows:
      • Trace ID — copy-paste ready for octane trace <id>
      • Agent tags (colour-coded by type)
      • Duration
      • Hint to inspect with octane trace / octane dag
    """
    from rich.text import Text

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

    # Build agent tag line
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


# ── octane trace ──────────────────────────────────────────────

# Colour mapping for each event type in the visual timeline
_EVENT_COLOURS: dict[str, str] = {
    "ingress":              "bold white",
    "guard":                "bold yellow",
    "decomposition":        "bold cyan",
    "decomposition_complete": "cyan",
    "dispatch":             "bold green",
    "agent_complete":       "green",
    "memory_read":          "blue",
    "memory_write":         "blue",
    "egress":               "bold magenta",
    "preflight":            "dim",
}

_EVENT_ICONS: dict[str, str] = {
    "ingress":              "→",
    "guard":                "🛡",
    "decomposition":        "🔀",
    "decomposition_complete": "✔",
    "dispatch":             "⚡",
    "agent_complete":       "✅",
    "memory_read":          "🧠",
    "memory_write":         "💾",
    "egress":               "←",
    "preflight":            "·",
}


@app.command()
def trace(
    correlation_id: str = typer.Argument(
        None,
        help="Correlation ID to trace. Partial IDs are accepted. "
             "Omit to list recent traces.",
    ),
):
    """🔍 Visual timeline of a query lifecycle — events, agents, DAG, duration."""
    asyncio.run(_trace(correlation_id))


async def _trace(correlation_id: str | None):
    synapse = _get_synapse()

    if correlation_id:
        # Resolve partial IDs — user can type first 8 chars
        resolved = _resolve_trace_id(synapse, correlation_id)
        if resolved is None:
            console.print(f"[yellow]No trace found matching: [bold]{correlation_id}[/bold][/]")
            console.print("[dim]Run 'octane trace' (no args) to list recent traces.[/]")
            return

        t = synapse.get_trace(resolved)
        real_events = [e for e in t.events if e.correlation_id != "preflight"]

        if not real_events:
            console.print(f"[yellow]Trace [bold]{resolved}[/bold] exists but has no events.[/]")
            return

        # ── Header panel ──────────────────────────────────────
        started_str = t.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if t.started_at else "?"
        agent_tags = _format_agent_tags(t.agents_used)

        egress = next((e for e in real_events if e.event_type == "egress"), None)
        dag_nodes_raw = egress.payload.get("dag_nodes_json", "") if egress and egress.payload else ""
        dag_reason = egress.payload.get("dag_reasoning", "") if egress and egress.payload else ""

        header_lines = [
            f"[bold]Trace ID:[/]  {t.correlation_id}",
            f"[bold]Started:[/]   {started_str}",
            f"[bold]Duration:[/]  {t.total_duration_ms:.0f} ms",
            f"[bold]Status:[/]    {'[green]✅ success[/]' if t.success else '[red]❌ failed[/]'}",
            f"[bold]Agents:[/]    {agent_tags}",
        ]
        if dag_reason:
            header_lines.append(f"[bold]Routing:[/]   [dim]{dag_reason[:100]}[/]")

        console.print(Panel(
            "\n".join(header_lines),
            title="[bold blue]🔍 Synapse Trace[/]",
            border_style="blue",
        ))

        # ── DAG section (if dag_nodes_json present) ───────────
        if dag_nodes_raw:
            import json as _json
            try:
                dag_nodes = _json.loads(dag_nodes_raw) if isinstance(dag_nodes_raw, str) else dag_nodes_raw
                if dag_nodes:
                    dag_table = Table(title="Task DAG", show_lines=False, box=None, padding=(0, 2))
                    dag_table.add_column("Node", style="dim", width=4, justify="right")
                    dag_table.add_column("Agent", style="cyan", width=12)
                    dag_table.add_column("Sub-agent", style="green", width=14)
                    dag_table.add_column("Instruction", style="white")
                    for i, node in enumerate(dag_nodes, 1):
                        dag_table.add_row(
                            str(i),
                            node.get("agent", "?"),
                            node.get("metadata", {}).get("sub_agent", "—"),
                            (node.get("instruction") or "")[:80],
                        )
                    console.print(dag_table)
            except Exception:
                pass  # malformed dag_nodes_json — skip silently

        # ── Visual event timeline ─────────────────────────────
        tl_table = Table(
            title="Event Timeline",
            show_lines=False,
            box=None,
            padding=(0, 2),
        )
        tl_table.add_column("", style="dim", width=2)   # icon
        tl_table.add_column("Δt", style="dim", width=9, justify="right")
        tl_table.add_column("Event", width=26)
        tl_table.add_column("Source → Target", style="dim", width=32)
        tl_table.add_column("Detail", style="white")

        t0 = t.started_at
        for event in real_events:
            offset = (
                f"+{(event.timestamp - t0).total_seconds() * 1000:.0f}ms"
                if t0 else "—"
            )
            icon = _EVENT_ICONS.get(event.event_type, "·")
            colour = _EVENT_COLOURS.get(event.event_type, "white")
            type_str = f"[{colour}]{event.event_type}[/]"

            src_tgt = event.source
            if event.target:
                src_tgt += f" → {event.target}"

            # Compact detail: pick the most informative payload field
            detail = ""
            if event.error:
                detail = f"[red]✗ {event.error[:80]}[/]"
            elif event.payload:
                for key in ("template", "reasoning", "output_preview", "query",
                            "approach", "agents_used", "tasks_succeeded", "agent"):
                    if key in event.payload:
                        val = str(event.payload[key])
                        label = {
                            "template": "→",
                            "reasoning": "reason:",
                            "output_preview": "output:",
                            "query": "q:",
                            "approach": "plan:",
                            "agents_used": "agents:",
                            "tasks_succeeded": "ok:",
                            "agent": "agent:",
                        }.get(key, f"{key}:")
                        detail = f"[dim]{label}[/] {val[:90]}"
                        break

            tl_table.add_row(icon, offset, type_str, src_tgt, detail)

        console.print(tl_table)
        console.print(
            f"[dim]  {len(real_events)} events · "
            f"Run [bold]octane dag \"<query>\"[/bold] to preview routing before executing[/]"
        )

    else:
        # ── Recent traces list ────────────────────────────────
        trace_ids = synapse.list_traces(limit=15)
        if not trace_ids:
            console.print("[yellow]No traces found. Run 'octane ask' first.[/]")
            return

        table = Table(title="Recent Traces  (~/.octane/traces/)", show_lines=False)
        table.add_column("Correlation ID", style="cyan")
        table.add_column("Started", style="dim", width=20)
        table.add_column("Events", justify="right", style="dim")
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Agents", style="green")
        table.add_column("", justify="center", width=3)  # status

        for cid in trace_ids:
            if cid == "preflight":
                continue
            t = synapse.get_trace(cid)
            started_str = t.started_at.strftime("%m-%d %H:%M:%S") if t.started_at else "?"
            visible_agents = [a for a in t.agents_used if a not in ("user", "osa", "osa.decomposer", "osa.evaluator")]
            table.add_row(
                t.correlation_id,
                started_str,
                str(len(t.events)),
                f"{t.total_duration_ms:.0f}ms",
                ", ".join(visible_agents) or "—",
                "✅" if t.success else "❌",
            )

        console.print(table)
        console.print("[dim]  octane trace <id>   — full event timeline[/]")
        console.print("[dim]  octane trace <id>   — partial IDs accepted (first 8 chars)[/]")


def _resolve_trace_id(synapse, partial_id: str) -> str | None:
    """Return a full correlation ID that starts with partial_id, or None."""
    # Try exact match first
    t = synapse.get_trace(partial_id)
    if t.events:
        return partial_id
    # Try prefix match across stored traces
    all_ids = synapse.list_traces(limit=50)
    for cid in all_ids:
        if cid.startswith(partial_id):
            return cid
    return None


def _format_agent_tags(agents_used: list[str]) -> str:
    """Return a Rich markup string of coloured agent tags."""
    _AGENT_COLOURS = {
        "web": "cyan", "code": "yellow", "memory": "blue",
        "sysstat": "green", "pnl": "magenta",
    }
    parts = []
    for a in agents_used:
        if a in ("user", "osa", "osa.decomposer", "osa.evaluator"):
            continue
        c = _AGENT_COLOURS.get(a, "white")
        parts.append(f"[bold {c}]{a}[/]")
    return "  ".join(parts) if parts else "[dim]osa[/]"


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

        # Spinner covers guard → decompose → dispatch → first Evaluator token
        _status = console.status("[dim]⚙  Working...[/]", spinner="dots")
        _status.start()
        response_parts: list[str] = []
        _first = True
        async for chunk in osa.run_stream(
            query,
            session_id=session_id,
            conversation_history=conversation_history,
        ):
            if _first:
                _status.stop()
                console.print(f"\n[bold green]Octane:[/] ", end="")
                _first = False
            console.print(chunk, end="")
            response_parts.append(chunk)
        if _first:
            _status.stop()  # no tokens (guard blocked, etc.)
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


# ── octane watch ─────────────────────────────────────────────

watch_app = typer.Typer(
    name="watch",
    help="📡 Background stock / asset monitors (powered by Shadows).",
    no_args_is_help=True,
)
app.add_typer(watch_app, name="watch")


def _get_watch_shadow():
    """Return (shadow_name, redis_url) from settings."""
    from octane.config import settings
    return "octane", settings.redis_url


@watch_app.command("start")
def watch_start(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL or BTC"),
    interval_hours: float = typer.Option(1.0, "--every", "-e", help="Poll interval in hours"),
):
    """📈 Start a perpetual background monitor for a ticker.

    Schedules ``monitor_ticker`` as a Shadows perpetual task, then
    launches the Octane background worker subprocess if it is not
    already running.

    Examples::

        octane watch start AAPL
        octane watch start BTC --every 0.5
    """
    asyncio.run(_watch_start(ticker.upper(), interval_hours))


async def _watch_start(ticker: str, interval_hours: float):
    import subprocess
    from datetime import timedelta
    from shadows import Shadow
    from octane.tasks.monitor import monitor_ticker, POLL_INTERVAL
    from octane.tasks.worker_process import read_pid, PID_FILE
    from octane.config import settings

    shadow_name, redis_url = _get_watch_shadow()

    # ── 1. Schedule the perpetual task via Shadow ──────────────
    console.print(f"[dim]Connecting to Redis at {redis_url}...[/]")
    try:
        every = timedelta(hours=interval_hours)
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            shadow.register(monitor_ticker)
            # Use ticker symbol as the stable task key — guarantees exactly one
            # perpetual loop per symbol regardless of how many times the user
            # runs this command.
            await shadow.add(monitor_ticker, key=ticker)(ticker=ticker)
        console.print(
            f"[green]✅ Scheduled monitor for [bold]{ticker}[/bold] "
            f"(every {every}, key={ticker})[/]"
        )
    except Exception as exc:
        console.print(f"[red]Failed to schedule task: {exc}[/]")
        raise typer.Exit(1)

    # ── 2. Ensure the worker subprocess is running ─────────────
    existing_pid = read_pid()
    if existing_pid:
        console.print(f"[dim]Worker already running (PID {existing_pid}) — task picked up automatically.[/]")
        return

    console.print("[dim]Starting Octane background worker...[/]")
    import sys
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "octane.tasks.worker_process",
            "--shadows-name", shadow_name,
            "--redis-url", redis_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,   # detach from terminal
    )
    # Give it a moment to write the PID file
    import time
    time.sleep(0.8)
    pid = read_pid()
    if pid:
        console.print(f"[green]🚀 Worker started (PID {pid})[/]")
    else:
        # Process launched but PID file not yet visible — report the Popen PID
        console.print(f"[green]🚀 Worker launched (PID ~{proc.pid})[/]")

    console.print(
        f"\n[dim]Run [bold]octane watch status[/bold] to see running monitors.\n"
        f"Run [bold]octane watch latest {ticker}[/bold] to see the latest quote.[/]"
    )


@watch_app.command("stop")
def watch_stop():
    """🛑 Stop the Octane background worker."""
    from octane.tasks.worker_process import read_pid, _remove_pid
    import os
    import signal as _signal

    pid = read_pid()
    if pid is None:
        console.print("[yellow]No worker is running.[/]")
        return

    try:
        os.kill(pid, _signal.SIGTERM)
        _remove_pid()
        console.print(f"[green]✅ Worker (PID {pid}) stopped.[/]")
    except ProcessLookupError:
        _remove_pid()
        console.print(f"[yellow]Worker PID {pid} was already gone — cleaned up.[/]")
    except PermissionError:
        console.print(f"[red]Permission denied stopping PID {pid}.[/]")


@watch_app.command("status")
def watch_status():
    """📊 Show running monitors and worker status."""
    asyncio.run(_watch_status())


async def _watch_status():
    from shadows import Shadow
    from octane.tasks.worker_process import read_pid
    from octane.config import settings

    shadow_name, redis_url = _get_watch_shadow()

    # Worker process status
    pid = read_pid()
    worker_line = (
        f"[green]🟢 Running (PID {pid})[/]" if pid else "[red]🔴 Not running[/]"
    )
    console.print(Panel(worker_line, title="[bold]Octane Worker[/]", border_style="cyan"))

    # Scheduled tasks snapshot
    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            from octane.tasks.monitor import monitor_ticker
            shadow.register(monitor_ticker)
            snapshot = await shadow.snapshot()

        table = Table(title="Scheduled / Running Tasks", show_lines=False)
        table.add_column("Key", style="cyan")
        table.add_column("Function", style="green")
        table.add_column("When (UTC)", style="yellow")
        table.add_column("State", style="white")

        for exe in snapshot.future:
            table.add_row(
                exe.key,
                exe.function.__name__,
                exe.when.strftime("%Y-%m-%d %H:%M:%S"),
                "[dim]scheduled[/]",
            )
        for exe in snapshot.running:
            table.add_row(
                exe.key,
                exe.function.__name__,
                exe.when.strftime("%Y-%m-%d %H:%M:%S"),
                f"[bold green]running[/] on {exe.worker}",
            )

        total = snapshot.total_tasks
        if total == 0:
            console.print("[dim]No tasks scheduled. Run: octane watch start <ticker>[/]")
        else:
            console.print(table)
            console.print(f"[dim]Total: {total} task(s)  ·  Workers active: {len(snapshot.workers)}[/]")

    except Exception as exc:
        console.print(f"[yellow]Could not reach Redis: {exc}[/]")
        console.print("[dim]Is Redis running? Check: redis-cli ping[/]")


@watch_app.command("latest")
def watch_latest(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
):
    """💹 Show the latest stored quote for a ticker."""
    asyncio.run(_watch_latest(ticker.upper()))


async def _watch_latest(ticker: str):
    from octane.tools.redis_client import RedisClient
    import json

    redis = RedisClient()
    key = f"watch:{ticker}:latest"
    raw = await redis.get(key)
    await redis.close()

    if not raw:
        console.print(
            f"[yellow]No data yet for [bold]{ticker}[/bold]. "
            f"Run: octane watch start {ticker}[/]"
        )
        return

    try:
        data = json.loads(raw)
    except Exception:
        data = {"raw": raw}

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(Panel(table, title=f"[bold green]📈 {ticker} — Latest Quote[/]", border_style="green"))


@watch_app.command("cancel")
def watch_cancel(
    ticker: str = typer.Argument(..., help="Ticker symbol to stop monitoring"),
):
    """❌ Cancel the perpetual monitor for a ticker."""
    asyncio.run(_watch_cancel(ticker.upper()))


async def _watch_cancel(ticker: str):
    from shadows import Shadow
    from octane.config import settings

    shadow_name, redis_url = _get_watch_shadow()

    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            await shadow.cancel(ticker)
        console.print(f"[green]✅ Monitor for [bold]{ticker}[/bold] cancelled.[/]")
    except Exception as exc:
        console.print(f"[red]Failed to cancel: {exc}[/]")
        raise typer.Exit(1)


# ── octane dag ────────────────────────────────────────────────


@app.command()
def dag(
    query: str = typer.Argument(..., help="Query to dry-run through the Decomposer"),
):
    """🔀 Preview how Octane would decompose a query — no agents run.

    Shows the task DAG that would be built for QUERY: which agent(s) would
    handle it, what sub-agent hint is selected, and the routing reasoning.
    Useful for understanding routing before committing to a full 'octane ask'.

    Examples::

        octane dag "what is AAPL trading at?"
        octane dag "write a python script to sort a list"
        octane dag "compare NVDA and AMD earnings"
    """
    asyncio.run(_dag(query))


async def _dag(query: str):
    from octane.osa.orchestrator import Orchestrator
    from octane.osa.decomposer import PIPELINE_TEMPLATES

    synapse = _get_synapse()
    osa = Orchestrator(synapse)

    # Run pre-flight silently — Bodega status affects LLM vs keyword routing
    with console.status("[dim]Connecting to inference engine...[/]", spinner="dots"):
        status = await osa.pre_flight()

    routing_mode = (
        "[green]LLM[/]" if (status["bodega_reachable"] and status["model_loaded"])
        else "[yellow]keyword fallback[/]"
    )

    # Dry-run the Decomposer — does NOT dispatch any agents
    with console.status("[dim]Decomposing...[/]", spinner="dots"):
        task_dag = await osa.decomposer.decompose(query)

    # ── Header ────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Query:[/]   {query}\n"
        f"[bold]Routing:[/] {routing_mode}  ·  "
        f"[bold]Nodes:[/] {len(task_dag.nodes)}  ·  "
        f"[bold]Waves:[/] {len(task_dag.execution_order())}\n"
        f"[bold]Reason:[/]  [dim]{task_dag.reasoning[:120] or '—'}[/]",
        title="[bold cyan]🔀 DAG Dry-Run[/]",
        border_style="cyan",
    ))

    # ── Node table ────────────────────────────────────────────
    node_table = Table(show_lines=False, box=None, padding=(0, 2))
    node_table.add_column("#", style="dim", width=3, justify="right")
    node_table.add_column("Wave", style="dim", width=5, justify="center")
    node_table.add_column("Agent", style="cyan", width=12)
    node_table.add_column("Sub-agent", style="green", width=16)
    node_table.add_column("Template", style="yellow", width=20)
    node_table.add_column("Instruction", style="white")

    # Build wave → node_id mapping
    wave_map: dict[str, int] = {}
    for wave_idx, wave in enumerate(task_dag.execution_order(), 1):
        for node_id in wave:
            wave_map[node_id] = wave_idx

    for i, node in enumerate(task_dag.nodes, 1):
        wave_num = wave_map.get(node.id, "?")
        template = node.metadata.get("template", "—")
        sub_agent = node.metadata.get("sub_agent", "—")

        # Colour the agent cell by type
        _AGENT_COLOURS = {
            "web": "cyan", "code": "yellow", "memory": "blue",
            "sysstat": "green", "pnl": "magenta",
        }
        ac = _AGENT_COLOURS.get(node.agent, "white")
        agent_str = f"[bold {ac}]{node.agent}[/]"

        node_table.add_row(
            str(i),
            str(wave_num),
            agent_str,
            sub_agent,
            template,
            (node.instruction or "")[:80],
        )

    console.print(node_table)

    # ── Template description hint ─────────────────────────────
    if task_dag.nodes:
        first_template = task_dag.nodes[0].metadata.get("template", "")
        tmpl_info = PIPELINE_TEMPLATES.get(first_template, {})
        desc = tmpl_info.get("description", "")
        if desc:
            console.print(f"\n[dim]  Template description: {desc}[/]")

    console.print(
        f"\n[dim]  This DAG would dispatch {len(task_dag.nodes)} task(s). "
        f"Run [bold]octane ask \"{query}\"[/bold] to execute.[/]"
    )


# ── octane pref ───────────────────────────────────────────────

pref_app = typer.Typer(
    name="pref",
    help="⚙  Manage user preferences — controls verbosity, expertise, response style.",
    no_args_is_help=True,
)
app.add_typer(pref_app, name="pref")

# Valid values for each preference key — used for validation + autocomplete hints
_PREF_CHOICES: dict[str, list[str]] = {
    "verbosity":      ["concise", "detailed"],
    "expertise":      ["beginner", "intermediate", "advanced"],
    "response_style": ["prose", "bullets", "code-first"],
    "domains":        [],   # free text — comma-separated list
}


@pref_app.command("show")
def pref_show(
    user_id: str = typer.Option("default", "--user", "-u", help="User ID"),
):
    """📋 Show all current preference values for a user.

    Example::

        octane pref show
        octane pref show --user alice
    """
    asyncio.run(_pref_show(user_id))


async def _pref_show(user_id: str):
    from octane.agents.pnl.preference_manager import PreferenceManager, DEFAULTS

    pm = PreferenceManager()
    profile = await pm.get_all(user_id)

    table = Table(title=f"Preferences — user: [bold]{user_id}[/]", show_lines=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="bold white", width=18)
    table.add_column("Default", style="dim", width=18)
    table.add_column("Choices", style="dim")

    for key, default in DEFAULTS.items():
        current = profile.get(key, default)
        is_custom = current != default
        value_str = f"[bold green]{current}[/]" if is_custom else current
        choices = ", ".join(_PREF_CHOICES.get(key, [])) or "free text"
        table.add_row(key, value_str, default, choices)

    console.print(table)
    console.print("[dim]  Green = customised · Run: octane pref set <key> <value>[/]")


@pref_app.command("set")
def pref_set(
    key: str = typer.Argument(..., help="Preference key (e.g. verbosity)"),
    value: str = typer.Argument(..., help="New value"),
    user_id: str = typer.Option("default", "--user", "-u", help="User ID"),
):
    """✏  Set a preference value.

    Examples::

        octane pref set verbosity detailed
        octane pref set expertise beginner
        octane pref set response_style bullets
        octane pref set domains "technology,finance,science"
    """
    asyncio.run(_pref_set(user_id, key, value))


async def _pref_set(user_id: str, key: str, value: str):
    from octane.agents.pnl.preference_manager import PreferenceManager, DEFAULTS

    if key not in DEFAULTS:
        console.print(f"[red]Unknown preference key: [bold]{key}[/bold][/]")
        console.print(f"[dim]Valid keys: {', '.join(DEFAULTS)}[/]")
        raise typer.Exit(1)

    choices = _PREF_CHOICES.get(key, [])
    if choices and value not in choices:
        console.print(f"[red]Invalid value [bold]{value!r}[/bold] for [bold]{key}[/bold][/]")
        console.print(f"[dim]Valid values: {', '.join(choices)}[/]")
        raise typer.Exit(1)

    pm = PreferenceManager()
    await pm.set(user_id, key, value)
    console.print(f"[green]✅ {key}[/] = [bold]{value}[/]  [dim](user: {user_id})[/]")


@pref_app.command("reset")
def pref_reset(
    key: str = typer.Argument(None, help="Key to reset. Omit to reset ALL preferences."),
    user_id: str = typer.Option("default", "--user", "-u", help="User ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """🔄 Reset a preference (or all preferences) to default values.

    Examples::

        octane pref reset verbosity
        octane pref reset --yes           # reset all without prompting
    """
    asyncio.run(_pref_reset(user_id, key, yes))


async def _pref_reset(user_id: str, key: str | None, skip_confirm: bool):
    from octane.agents.pnl.preference_manager import PreferenceManager, DEFAULTS

    pm = PreferenceManager()

    if key:
        if key not in DEFAULTS:
            console.print(f"[red]Unknown preference key: {key}[/]")
            raise typer.Exit(1)
        await pm.delete(user_id, key)
        console.print(f"[green]✅ {key}[/] reset to default: [bold]{DEFAULTS[key]}[/]  [dim](user: {user_id})[/]")
    else:
        if not skip_confirm:
            confirm = console.input(
                f"[yellow]Reset ALL preferences for user [bold]{user_id}[/bold]? [y/N]: [/]"
            ).strip().lower()
            if confirm not in ("y", "yes"):
                console.print("[dim]Cancelled.[/]")
                return
        for k in DEFAULTS:
            await pm.delete(user_id, k)
        console.print(f"[green]✅ All preferences reset to defaults[/]  [dim](user: {user_id})[/]")


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
    """📦 Show Octane version, stack, and registered agents."""
    import sys
    import platform
    from octane import __version__
    from octane.osa.router import Router

    synapse = _get_synapse()
    router = Router(synapse)
    agent_names = router.list_agents()

    try:
        import shadows as _shadows
        shadows_ver = getattr(_shadows, "__version__", "unknown")
    except ImportError:
        shadows_ver = "not installed"

    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    arch = platform.machine()   # arm64 on Apple Silicon

    # Build the agent tag line
    _AGENT_COLOURS = {
        "web": "cyan", "code": "yellow", "memory": "blue",
        "sysstat": "green", "pnl": "magenta",
    }
    agent_parts = []
    for name in agent_names:
        c = _AGENT_COLOURS.get(name, "white")
        agent_parts.append(f"[bold {c}]{name}[/]")
    agents_line = "  ".join(agent_parts)

    body = (
        f"[bold cyan]🔥 Octane[/]  v[bold]{__version__}[/]\n\n"
        f"[bold]Python:[/]  {python_ver}  ({arch})\n"
        f"[bold]Shadows:[/] {shadows_ver}\n\n"
        f"[bold]Agents:[/]  {agents_line}\n\n"
        f"[dim]Run [bold]octane --help[/bold] for all commands.[/]"
    )

    console.print(Panel(body, border_style="cyan", padding=(1, 4)))


# ── octane workflow ───────────────────────────────────────────

workflow_app = typer.Typer(
    name="workflow",
    help="🗂  Save, list, and replay Octane pipeline templates.",
    no_args_is_help=True,
)
app.add_typer(workflow_app, name="workflow")


@workflow_app.command("export")
def workflow_export(
    correlation_id: str = typer.Argument(..., help="Trace ID to export (from octane ask footer)"),
    name: str = typer.Option(None, "--name", "-n", help="Template name (default: first 8 chars of trace ID)"),
    description: str = typer.Option("", "--desc", "-d", help="Human-readable description"),
    out: str = typer.Option(None, "--out", "-o", help="Output file path (default: ~/.octane/workflows/<name>.workflow.json)"),
):
    """📤 Export a previous query as a reusable workflow template.

    Reads the Synapse trace for CORRELATION_ID and saves the DAG structure
    as a parameterised ``.workflow.json`` file.

    Example::

        octane workflow export abc12345 --name stock-check
    """
    from octane.workflow import export_from_trace
    from pathlib import Path

    try:
        template = export_from_trace(correlation_id, name=name, description=description)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(1)

    save_path = Path(out) if out else None
    saved_to = template.save(save_path)

    placeholders = template.list_placeholders()
    console.print(Panel(
        f"[bold]Name:[/]        {template.name}\n"
        f"[bold]Nodes:[/]       {len(template.nodes)}\n"
        f"[bold]Reasoning:[/]   {template.reasoning[:80] or '—'}\n"
        f"[bold]Placeholders:[/] {', '.join(placeholders) or 'none'}\n"
        f"[bold]Saved to:[/]    [cyan]{saved_to}[/]",
        title="[bold green]✅ Workflow Exported[/]",
        border_style="green",
    ))


@workflow_app.command("list")
def workflow_list():
    """📋 List saved workflow templates."""
    from octane.workflow import list_workflows, WorkflowTemplate
    import os

    files = list_workflows()
    if not files:
        console.print("[yellow]No workflows saved yet. Run: octane workflow export <trace_id>[/]")
        return

    table = Table(title="Saved Workflows  (~/.octane/workflows/)", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Nodes", justify="right", style="yellow")
    table.add_column("Variables", style="green")
    table.add_column("Modified", style="dim", width=18)
    table.add_column("Description", style="white")

    for f in files:
        try:
            t = WorkflowTemplate.load(f)
            # Last-modified timestamp
            mtime = os.path.getmtime(f)
            from datetime import datetime as _dt
            mod_str = _dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            # Variable display: show keys + default values, highlight placeholders
            var_parts = []
            for k, v in t.variables.items():
                if v:
                    var_parts.append(f"[dim]{k}[/]=[bold]{v}[/]")
                else:
                    var_parts.append(f"[bold yellow]{{{{[/][dim]{k}[/][bold yellow]}}}}[/]")
            var_str = "  ".join(var_parts) if var_parts else "[dim]—[/]"
            table.add_row(
                t.name,
                str(len(t.nodes)),
                var_str,
                mod_str,
                t.description[:55] or "—",
            )
        except Exception:
            table.add_row(f.stem, "?", "?", "?", "[red]parse error[/]")

    console.print(table)
    console.print(
        f"[dim]  {len(files)} template(s) · "
        f"octane workflow run <name>.workflow.json [--var key=value][/]"
    )


@workflow_app.command("run")
def workflow_run(
    file: str = typer.Argument(..., help="Path to .workflow.json file"),
    var: list[str] = typer.Option([], "--var", "-v", help="Variable override: key=value (repeatable)"),
    query: str = typer.Option("", "--query", "-q", help="Override the {{query}} variable"),
    verbose: bool = typer.Option(False, "--verbose", help="Show DAG trace after response"),
):
    """▶  Run a saved workflow template.

    Fills ``{{variables}}``, builds the TaskDAG, and streams the result —
    bypassing the Decomposer (the pipeline shape is fixed by the template).

    Examples::

        octane workflow run ~/.octane/workflows/stock-check.workflow.json --var ticker=MSFT
        octane workflow run stock-check.workflow.json --query "Compare NVDA and AMD"
    """
    asyncio.run(_workflow_run(file, var, query, verbose))


async def _workflow_run(file_str: str, var_list: list[str], query_override: str, verbose: bool):
    from pathlib import Path
    from octane.workflow.runner import load_workflow
    from octane.osa.orchestrator import Orchestrator

    path = Path(file_str)
    # Try ~/.octane/workflows/ as fallback if not an absolute/relative path
    if not path.exists():
        from octane.workflow import WORKFLOW_DIR
        fallback = WORKFLOW_DIR / file_str
        if fallback.exists():
            path = fallback
        else:
            # Also try adding .workflow.json suffix
            fallback2 = WORKFLOW_DIR / f"{file_str}.workflow.json"
            if fallback2.exists():
                path = fallback2

    try:
        template = load_workflow(path)
    except FileNotFoundError:
        console.print(f"[red]Workflow file not found: {file_str}[/]")
        console.print("[dim]Run 'octane workflow list' to see available templates.[/]")
        raise typer.Exit(1)

    # Parse --var key=value overrides
    overrides: dict[str, str] = {}
    for item in var_list:
        if "=" in item:
            k, _, v = item.partition("=")
            overrides[k.strip()] = v.strip()
        else:
            console.print(f"[yellow]Ignoring malformed --var '{item}' (expected key=value)[/]")

    if query_override:
        overrides["query"] = query_override

    # Build the DAG
    try:
        dag = template.to_dag(overrides)
    except Exception as exc:
        console.print(f"[red]Failed to build DAG from template: {exc}[/]")
        raise typer.Exit(1)

    # Run
    synapse = _get_synapse()
    osa = Orchestrator(synapse)

    effective_query = overrides.get("query") or template.variables.get("query") or template.description
    console.print(Panel(
        f"[bold]Template:[/] {template.name}  ·  [bold]Nodes:[/] {len(dag.nodes)}  ·  "
        f"[bold]Query:[/] {effective_query[:80]}",
        title="[bold cyan]▶ Running Workflow[/]",
        border_style="cyan",
    ))

    status = console.status("[dim]⚙  Running pipeline...[/]", spinner="dots")
    status.start()
    first_token = True
    full_parts: list[str] = []

    async for chunk in osa.run_from_dag(dag, query=effective_query):
        if first_token:
            status.stop()
            console.print("[bold green]🔥 Octane:[/] ", end="")
            first_token = False
        console.print(chunk, end="")
        full_parts.append(chunk)

    if first_token:
        status.stop()  # pipeline ran but produced no output

    console.print("\n")

    recent = synapse.get_recent_traces(limit=1)
    if recent:
        t = recent[0]
        real_events = [e for e in t.events if e.correlation_id != "preflight"]
        egress = next((e for e in real_events if e.event_type == "egress"), None)
        dag_nodes = egress.payload.get("dag_nodes", "?") if egress and egress.payload else "?"
        console.print(
            f"[dim]Agents: {', '.join(t.agents_used)} | "
            f"DAG nodes: {dag_nodes} | "
            f"Duration: {t.total_duration_ms}ms | "
            f"Trace: [bold]{t.correlation_id}[/][/]"
        )
        if verbose:
            dag_reason = egress.payload.get("dag_reasoning", "") if egress and egress.payload else ""
            _print_dag_trace(t, real_events, dag_nodes, dag_reason)


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    app()
