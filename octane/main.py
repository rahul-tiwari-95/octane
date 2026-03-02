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
            elif not model.get("loaded"):
                bodega_table.add_row("Model", "[yellow]no model loaded[/]")
            else:
                model_name = model.get("model_path", model.get("model", "unknown"))
                total = model.get("total_loaded", 1)
                suffix = f" [dim]({total} loaded)[/]" if total and total > 1 else ""
                bodega_table.add_row("Model", f"[green]{model_name}[/]{suffix}")

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


# ── octane sysstat ───────────────────────────────────────────


@app.command()
def sysstat():
    """📊 Live system snapshot — RAM, CPU, loaded model (no Bodega required)."""
    asyncio.run(_sysstat())


async def _sysstat():
    from octane.agents.sysstat.agent import SysStatAgent
    from octane.models.schemas import AgentRequest
    from octane.tools.bodega_inference import BodegaInferenceClient
    from rich.table import Table

    synapse = _get_synapse()
    bodega = BodegaInferenceClient()

    try:
        agent = SysStatAgent(synapse, bodega)
        request = AgentRequest(query="sysstat", source="cli")
        response = await agent.run(request)

        if not response.success:
            console.print(f"[red]sysstat failed: {response.error}[/]")
            return

        data = response.data
        system = data.get("system", {})
        model = data.get("model", {})

        # ── System table ──────────────────────────────────────────────
        sys_tbl = Table(show_header=False, box=None, padding=(0, 2))
        sys_tbl.add_column("Metric", style="cyan")
        sys_tbl.add_column("Value", style="white")

        ram_used = system.get("ram_used_gb", 0)
        ram_total = system.get("ram_total_gb", 0)
        ram_pct = system.get("ram_percent", 0)
        ram_color = "green" if ram_pct < 70 else "yellow" if ram_pct < 90 else "red"

        sys_tbl.add_row("RAM", f"[{ram_color}]{ram_used:.1f} / {ram_total:.1f} GB ({ram_pct}%)[/]")
        sys_tbl.add_row("CPU", f"{system.get('cpu_percent', '?')}% ({system.get('cpu_count', '?')} cores)")
        sys_tbl.add_row("Available", f"{system.get('ram_available_gb', 0):.1f} GB free")
        sys_tbl.add_row("Uptime", f"{system.get('uptime_hours', 0):.1f} h")

        # ── Model table ───────────────────────────────────────────────
        mod_tbl = Table(show_header=False, box=None, padding=(0, 2))
        mod_tbl.add_column("Key", style="magenta")
        mod_tbl.add_column("Value", style="white")

        if "error" in model:
            mod_tbl.add_row("Status", f"[yellow]⚠ {model['error']}[/]")
        else:
            model_name = model.get("model", model.get("model_path", "—"))
            mod_tbl.add_row("Model", f"[green]{model_name}[/]")
            if model.get("loaded"):
                mod_tbl.add_row("Status", "[green]✓ loaded[/]")
            if model.get("context_length"):
                mod_tbl.add_row("Context", f"{model['context_length']:,} tokens")

        console.print(Panel(sys_tbl, title="[bold cyan]💻 System[/]", border_style="cyan"))
        console.print(Panel(mod_tbl, title="[bold magenta]🧠 Model[/]", border_style="magenta"))
        console.print(f"[dim]Duration: {response.duration_ms}ms[/]")

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
    deep: bool = typer.Option(False, "--deep", help="Deep mode: multi-round search with iterative query expansion"),
):
    """🧠 Ask Octane anything — routed through OSA."""
    asyncio.run(_ask(query, verbose=verbose, deep=deep))


async def _ask(query: str, verbose: bool = False, deep: bool = False):
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
        deep_tag = " | [bold cyan]⬇ deep mode[/]" if deep else ""
        console.print(f"[dim]🧠 {model_display} | LLM decomposition + synthesis active{deep_tag}[/]")
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
    extra_meta = {"deep": True} if deep else {}
    async for chunk in osa.run_stream(query, extra_metadata=extra_meta):
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


# Slash commands available inside the chat REPL
_CHAT_HELP = """[bold]Slash commands:[/]
  [cyan]/help[/]          — show this message
  [cyan]/trace [id][/]    — show Synapse trace for last response (or a specific id)
  [cyan]/history[/]       — print current conversation history
  [cyan]/clear[/]         — clear conversation history and start fresh
  [cyan]/exit[/]          — end the session (also: exit, quit, q)
"""


async def _chat():
    from octane.osa.orchestrator import Orchestrator

    synapse = _get_synapse()
    # HIL interactive=True in chat — high-risk decisions get presented to user
    osa = Orchestrator(synapse, hil_interactive=True)
    session_id = f"chat_{int(__import__('time').time())}"

    # Rolling conversation buffer — last 6 turns (12 entries) injected into Evaluator
    conversation_history: list[dict[str, str]] = []
    # Track correlation IDs for /trace
    last_correlation_id: str | None = None

    console.print(Panel(
        "[bold green]Octane Chat[/]\n"
        "[dim]Type your message and press Enter. "
        "Use [bold cyan]/help[/bold cyan] for slash commands, "
        "[bold]exit[/bold] or [bold]quit[/bold] to end the session.[/]",
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

        # ── Slash commands ────────────────────────────────────────────────
        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/help":
                console.print(_CHAT_HELP)
                continue

            elif cmd == "/clear":
                conversation_history.clear()
                console.print("[dim]✓ Conversation history cleared.[/]\n")
                continue

            elif cmd == "/history":
                if not conversation_history:
                    console.print("[dim]No history yet.[/]\n")
                else:
                    for i, msg in enumerate(conversation_history, 1):
                        role_tag = "[bold cyan]You:[/]" if msg["role"] == "user" else "[bold green]Octane:[/]"
                        preview = msg["content"][:120].replace("\n", " ")
                        console.print(f"  {i}. {role_tag} {preview}")
                    console.print()
                continue

            elif cmd == "/trace":
                cid = parts[1].strip() if len(parts) > 1 else last_correlation_id
                if cid:
                    asyncio.get_event_loop().run_until_complete(
                        _trace(cid)
                    ) if False else None
                    # Use the synapse we already have
                    _print_synapse_trace(synapse, cid)
                else:
                    console.print("[yellow]No trace available yet — ask something first.[/]\n")
                continue

            else:
                console.print(f"[yellow]Unknown command '{cmd}'. Type /help for options.[/]\n")
                continue

        # ── Normal query ──────────────────────────────────────────────────
        if query.lower() in ("exit", "quit", "bye", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        turn += 1
        conversation_history.append({"role": "user", "content": query})

        _status = console.status("[dim]⚙  Working...[/]", spinner="dots")
        _status.start()
        response_parts: list[str] = []
        _first = True

        # Capture correlation_id from the egress event after streaming
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
            _status.stop()
        console.print()

        assistant_reply = "".join(response_parts).strip()
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Extract correlation_id from most recent egress event
        egress_events = [
            e for e in synapse._events
            if e.event_type == "egress"
        ]
        if egress_events:
            last_correlation_id = egress_events[-1].correlation_id
            console.print(
                f"[dim]  ↳ trace: {last_correlation_id[:16]}…  "
                f"(/trace to inspect)[/]\n"
            )
        else:
            console.print()

        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

    console.print(f"[dim]Session {session_id} — {turn} turn(s)[/]")


def _print_synapse_trace(synapse, correlation_id: str) -> None:
    """Print a compact inline Synapse trace for use inside /trace slash command."""
    from rich.table import Table
    events = [e for e in synapse._events if e.correlation_id == correlation_id]
    if not events:
        console.print(f"[yellow]No events found for {correlation_id}[/]\n")
        return
    table = Table(title=f"Trace: {correlation_id[:24]}…", show_header=True, header_style="bold")
    table.add_column("Event", style="cyan", no_wrap=True, width=26)
    table.add_column("Source → Target", width=28)
    table.add_column("Details", overflow="fold")
    for ev in events:
        target = getattr(ev, "target", "—") or "—"
        details = ""
        payload = getattr(ev, "payload", None) or {}
        if isinstance(payload, dict):
            if "agents_used" in payload:
                details = f"agents: {', '.join(payload['agents_used'])}"
            elif "template" in payload:
                details = f"template: {payload['template']}"
            elif "output_preview" in payload:
                details = payload["output_preview"][:60]
        table.add_row(ev.event_type, f"{ev.source} → {target}", details)
    console.print(table)
    console.print()



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


async def _ensure_shadow_group(shadow_name: str, redis_url: str) -> None:
    """Pre-create the Shadows consumer group, silently ignoring BUSYGROUP.

    The installed version of Shadows checks ``"BUSYGROUP" not in repr(e)`` to
    swallow duplicate-group errors, but in recent redis-py builds
    ``repr(ResponseError)`` returns ``'server:ResponseError'`` (no message
    text), so the guard never fires and the error propagates.  Creating the
    group here before Shadow.__aenter__ runs works around the bug.
    """
    import redis.asyncio as aioredis
    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.xgroup_create(
            name=f"{shadow_name}:stream",
            groupname="shadows-workers",
            id="0-0",
            mkstream=True,
        )
    except Exception:
        pass  # BUSYGROUP or any other error — group already exists, carry on
    finally:
        await client.aclose()


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
    await _ensure_shadow_group(shadow_name, redis_url)

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


# ── octane model ──────────────────────────────────────────────

model_app = typer.Typer(
    name="model",
    help="🧠 Manage the loaded LLM (reload, switch, inspect).",
    no_args_is_help=True,
)
app.add_typer(model_app, name="model")


@model_app.command("reload-parser")
def model_reload_parser(
    parser: str = typer.Option("qwen3", "--parser", "-p", help="Reasoning parser name (qwen3, harmony, etc.)"),
):
    """🔄 Reload the current model with reasoning_parser enabled.

    Unloads the running model and reloads the exact same model with the
    reasoning parser activated.  After this, Octane receives native
    reasoning_content from the API — no manual <think> token stripping needed.

    Examples::

        octane model reload-parser
        octane model reload-parser --parser qwen3
    """
    asyncio.run(_model_reload_parser(parser))


async def _model_reload_parser(parser: str) -> None:
    from octane.tools.bodega_inference import BodegaInferenceClient
    from octane.agents.sysstat.model_manager import ModelManager

    bodega = BodegaInferenceClient()
    manager = ModelManager(bodega)

    # Show current model
    info = await bodega.current_model()
    if not info.get("loaded"):
        console.print("[red]No model currently loaded.[/]")
        return

    model_path = info["model_info"]["model_path"]
    console.print(f"[dim]Current model: [bold]{model_path}[/bold][/]")
    console.print(f"[dim]Reloading with reasoning_parser=[bold]{parser}[/bold]...[/]")
    console.print("[yellow]⚠  Model will be unloaded briefly — Octane will be offline for ~5–15s.[/]")

    try:
        result = await manager.reload_with_reasoning_parser(reasoning_parser=parser)
        console.print(f"[green]✅ Model reloaded with reasoning_parser={parser!r}[/]")
        console.print(f"[dim]Bodega response: {result}[/]")
    except Exception as exc:
        console.print(f"[red]Reload failed: {exc}[/]")


@model_app.command("info")
def model_info():
    """🔍 Show the currently loaded model and its configuration."""
    asyncio.run(_model_info())


async def _model_info() -> None:
    from octane.tools.bodega_inference import BodegaInferenceClient
    bodega = BodegaInferenceClient()
    info = await bodega.current_model()
    if not info.get("loaded"):
        console.print("[yellow]No model currently loaded.[/]")
        return
    mi = info["model_info"]
    console.print(Panel(
        f"[bold]Model:[/]    {mi.get('model_path')}\n"
        f"[bold]Type:[/]     {mi.get('model_type', 'lm')}\n"
        f"[bold]Context:[/]  {mi.get('context_length', '?')} tokens\n"
        f"[bold]Parser:[/]   {mi.get('reasoning_parser', '[dim]none[/dim]')}",
        title="🧠 Loaded Model",
        border_style="cyan",
        padding=(0, 2),
    ))


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


# ── octane research ───────────────────────────────────────────

research_app = typer.Typer(
    name="research",
    help="🔬 Long-running background research workflows (powered by Shadows).",
    no_args_is_help=True,
)
app.add_typer(research_app, name="research")


# ══════════════════════════════════════════════════════════════
# octane db  — Schema migrations
# ══════════════════════════════════════════════════════════════

db_app = typer.Typer(
    name="db",
    help="🗄 Postgres schema migrations.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate():
    """📦 Apply pending schema migrations (idempotent, safe to run repeatedly)."""
    asyncio.run(_db_migrate())


async def _db_migrate():
    from octane.tools.migrations import MigrationRunner
    console.print("[dim]Running migrations…[/]")
    runner = MigrationRunner()
    result = await runner.migrate()
    if result.error:
        console.print(f"[red]❌ Migration failed: {result.error}[/]")
        raise typer.Exit(1)
    if result.applied:
        console.print(
            f"[green]✅ Migration [bold]{result.version}[/bold] applied.[/]\n"
            f"[dim]Tables: {', '.join(sorted(result.tables))}[/]"
        )
    else:
        console.print(
            f"[green]✅ Schema already current (version [bold]{result.version}[/bold]).[/]\n"
            f"[dim]Tables: {', '.join(sorted(result.tables))}[/]"
        )


@db_app.command("status")
def db_status():
    """📊 Show migration versions and per-table row counts."""
    asyncio.run(_db_status())


async def _db_status():
    from octane.tools.migrations import MigrationRunner
    runner = MigrationRunner()
    status = await runner.status()

    if not status.pg_available:
        console.print("[red]❌ Postgres unavailable.[/]")
        raise typer.Exit(1)

    # Version panel
    if status.applied_versions:
        ver_str = "[green]" + ", ".join(status.applied_versions) + "[/]"
    else:
        ver_str = "[yellow]none applied[/]"
    pending_str = (
        "[yellow]pending: " + ", ".join(status.pending_versions) + "[/]"
        if status.pending_versions
        else "[dim]up to date[/]"
    )
    console.print(Panel(
        f"Applied: {ver_str}\n{pending_str}",
        title="[bold]Schema Migrations[/]",
        border_style="cyan",
    ))

    if not status.table_counts:
        console.print("[dim]No tables found.[/]")
        return

    tbl = Table(title="Tables", show_lines=False)
    tbl.add_column("Table", style="cyan")
    tbl.add_column("Rows", justify="right", style="green")
    for name, count in sorted(status.table_counts.items()):
        tbl.add_row(name, str(count) if count >= 0 else "?")
    console.print(tbl)


@db_app.command("reset")
def db_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """💥 Drop all tables and re-apply schema.  [red bold]DEV ONLY.[/]"""
    if not yes:
        confirm = typer.confirm(
            "⚠️  This will DROP all Octane tables. Continue?", default=False
        )
        if not confirm:
            console.print("[dim]Aborted.[/]")
            raise typer.Exit(0)
    asyncio.run(_db_reset())


async def _db_reset():
    from octane.tools.migrations import MigrationRunner
    console.print("[yellow]Dropping all tables and re-applying schema…[/]")
    runner = MigrationRunner()
    ok = await runner.reset()
    if ok:
        console.print("[green]✅ Schema reset complete.[/]")
    else:
        console.print("[red]❌ Reset failed — check logs.[/]")
        raise typer.Exit(1)


def _get_research_shadow():
    """Return (shadow_name, redis_url) for the research namespace."""
    from octane.config import settings
    return "octane", settings.redis_url


@research_app.command("start")
def research_start(
    topic: str = typer.Argument(..., help="Research topic or question"),
    every: float = typer.Option(6.0, "--every", "-e", help="Cycle interval in hours (default: 6)"),
    depth: str = typer.Option("deep", "--depth", "-d",
                              help="Research depth: shallow (2 angles), deep (4, default), exhaustive (8)"),
):
    """🔬 Start a background research workflow.

    Schedules a perpetual Shadows task that runs the full OSA pipeline for
    TOPIC every EVERY hours — extracting content, synthesising findings,
    and storing them in Postgres for review via ``octane research report``.

    The ``--depth`` flag controls how many parallel search angles are used
    per cycle:

    \\b
        shallow    — 2 angles (fastest, good for quick market checks)
        deep       — 4 angles (default, balanced coverage)
        exhaustive — 8 angles (most thorough, use for deep-dive research)

    Examples::

        octane research start "NVDA earnings outlook"
        octane research start "Apple Vision Pro market reception" --every 12
        octane research start "Fed rate decision impact" --depth exhaustive --every 3
    """
    _valid_depths = {"shallow", "deep", "exhaustive"}
    if depth not in _valid_depths:
        console.print(f"[red]Invalid depth '{depth}'. Must be one of: {', '.join(sorted(_valid_depths))}[/]")
        raise typer.Exit(1)
    asyncio.run(_research_start(topic, every, depth))


async def _research_start(topic: str, interval_hours: float, depth: str = "deep"):
    import subprocess
    import sys
    import time
    from shadows import Shadow
    from octane.research.models import ResearchTask
    from octane.research.store import ResearchStore
    from octane.tasks.research import research_cycle
    from octane.tasks.worker_process import read_pid, PID_FILE
    from octane.config import settings

    shadow_name, redis_url = _get_research_shadow()
    await _ensure_shadow_group(shadow_name, redis_url)

    # ── 1. Create and register the task metadata ───────────────────────────
    task = ResearchTask(topic=topic, interval_hours=interval_hours)
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    await store.register_task(task)
    await store.log_entry(task.id, f"🔬 Research task created: {topic}")

    # ── 2. Schedule via Shadows ────────────────────────────────────────────
    console.print(f"[dim]Scheduling research task [bold]{task.id}[/bold]…[/]")
    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            shadow.register(research_cycle)
            await shadow.add(research_cycle, key=task.id)(
                task_id=task.id,
                topic=topic,
                interval_hours=interval_hours,
                depth=depth,
            )
        console.print(
            f"[green]✅ Research started[/]\n"
            f"  [bold]ID:[/]     [cyan]{task.id}[/]\n"
            f"  [bold]Topic:[/]  {topic}\n"
            f"  [bold]Every:[/]  {interval_hours}h  ·  "
            f"[bold]Depth:[/] {depth}\n"
        )
    except Exception as exc:
        console.print(f"[red]Failed to schedule task: {exc}[/]")
        await store.close()
        raise typer.Exit(1)

    # ── 3. Ensure the worker subprocess is running ─────────────────────────
    existing_pid = read_pid()
    if existing_pid:
        console.print(f"[dim]Worker already running (PID {existing_pid}) — task picked up automatically.[/]")
        await store.close()
        return

    console.print("[dim]Starting Octane background worker…[/]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "octane.tasks.worker_process",
         "--shadows-name", shadow_name, "--redis-url", redis_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.8)
    pid = read_pid()
    if pid:
        console.print(f"[green]🚀 Worker started (PID {pid})[/]")
    else:
        console.print(f"[green]🚀 Worker launched (PID ~{proc.pid})[/]")

    console.print(
        f"\n[dim]Run [bold]octane research log {task.id}[/bold] to follow progress.\n"
        f"Run [bold]octane research log {task.id} --follow[/bold] to stream live.[/]"
    )
    await store.close()


@research_app.command("status")
def research_status():
    """📊 List all active research tasks."""
    asyncio.run(_research_status())


async def _research_status():
    from octane.research.store import ResearchStore
    from octane.config import settings
    from octane.tasks.worker_process import read_pid

    _, redis_url = _get_research_shadow()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    tasks = await store.list_tasks()
    await store.close()

    pid = read_pid()
    worker_line = (
        f"[green]🟢 Running (PID {pid})[/]" if pid else "[red]🔴 Not running[/]"
    )
    console.print(Panel(worker_line, title="[bold]Octane Worker[/]", border_style="cyan"))

    if not tasks:
        console.print("[dim]No research tasks found. Run: octane research start \"<topic>\"[/]")
        return

    from rich.table import Table as _Table
    tbl = _Table(title="Active Research Tasks", show_lines=False)
    tbl.add_column("ID", style="cyan", width=10)
    tbl.add_column("Topic", style="white")
    tbl.add_column("Every", style="yellow", justify="right", width=7)
    tbl.add_column("Depth", style="blue", justify="right", width=10)
    tbl.add_column("Cycles", style="green", justify="right", width=7)
    tbl.add_column("Findings", style="magenta", justify="right", width=9)
    tbl.add_column("Age", style="dim", justify="right", width=8)
    tbl.add_column("Status", width=8)

    for t in tasks:
        age_str = f"{t.age_hours:.1f}h"
        status_str = "[green]active[/]" if t.is_active else "[red]stopped[/]"
        tbl.add_row(
            t.id, t.topic[:55], f"{t.interval_hours}h",
            getattr(t, "depth", "deep"),
            str(t.cycle_count), str(t.finding_count),
            age_str, status_str,
        )
    console.print(tbl)
    console.print(
        "[dim]  octane research log <id>          — view progress log\n"
        "  octane research report <id>       — synthesise all findings\n"
        "  octane research stop <id>         — cancel task[/]"
    )


@research_app.command("log")
def research_log(
    task_id: str = typer.Argument(..., help="Research task ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new entries in real time"),
    n: int = typer.Option(50, "--lines", "-n", help="Number of recent log lines to show"),
):
    """📋 Show the research progress log.

    Use ``--follow`` to stream live entries as the background task runs.

    Examples::

        octane research log abc12345
        octane research log abc12345 --follow
    """
    asyncio.run(_research_log(task_id, follow, n))


async def _research_log(task_id: str, follow: bool, n: int):
    from octane.research.store import ResearchStore
    from octane.config import settings
    import time as _time

    _, redis_url = _get_research_shadow()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    task = await store.get_task(task_id)
    if task is None:
        console.print(f"[yellow]No task found with ID: [bold]{task_id}[/][/]")
        console.print("[dim]Run 'octane research status' to see all tasks.[/]")
        await store.close()
        return

    console.print(Panel(
        f"[bold]ID:[/]    {task.id}\n"
        f"[bold]Topic:[/] {task.topic}\n"
        f"[bold]Every:[/] {task.interval_hours}h  ·  "
        f"[bold]Depth:[/] {getattr(task, 'depth', 'deep')}  ·  "
        f"[bold]Cycles:[/] {task.cycle_count}  ·  "
        f"[bold]Findings:[/] {task.finding_count}",
        title="[bold cyan]🔬 Research Task[/]",
        border_style="cyan",
    ))

    # ── Print existing entries ─────────────────────────────────────────────
    entries = await store.get_log(task_id, n=n)
    if not entries:
        console.print("[dim]No log entries yet — task may not have run yet.[/]")
    else:
        for line in entries:
            _print_log_line(line)

    if not follow:
        await store.close()
        return

    # ── Follow mode: poll for new entries ──────────────────────────────────
    console.print("[dim]  Following… (Ctrl+C to stop)[/]")
    seen = len(entries)
    last_cycle = task.cycle_count
    last_findings = task.finding_count
    try:
        while True:
            await asyncio.sleep(2.0)
            all_entries = await store.get_log(task_id, n=_LOG_FOLLOW_BUFFER)
            new = all_entries[seen:]
            for line in new:
                _print_log_line(line)
            seen = len(all_entries)
            # Re-fetch task state and print a compact refresh when counts change
            refreshed = await store.get_task(task_id)
            if refreshed and (
                refreshed.cycle_count != last_cycle
                or refreshed.finding_count != last_findings
            ):
                last_cycle = refreshed.cycle_count
                last_findings = refreshed.finding_count
                console.print(
                    f"[bold cyan]  ↺ Status:[/] Cycles={last_cycle}  "
                    f"Findings={last_findings}  "
                    f"(next cycle in ~{refreshed.interval_hours}h)"
                )
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[dim]Stopped following.[/]")
    finally:
        await store.close()


_LOG_FOLLOW_BUFFER = 200  # max entries fetched during --follow polling


def _print_log_line(line: str) -> None:
    """Colour-code a single research log line for terminal output."""
    if "✅" in line or "complete" in line.lower():
        console.print(f"[green]{line}[/]")
    elif "⚠" in line or "warn" in line.lower() or "error" in line.lower():
        console.print(f"[yellow]{line}[/]")
    elif "⚙" in line or "start" in line.lower():
        console.print(f"[cyan]{line}[/]")
    else:
        console.print(f"[dim]{line}[/]")


@research_app.command("report")
def research_report(
    task_id: str = typer.Argument(..., help="Research task ID"),
    raw: bool = typer.Option(False, "--raw", help="Print raw findings without LLM synthesis"),
    cycles: int = typer.Option(None, "--cycles", "-c", help="Use only the last N cycles"),
    since: str = typer.Option(None, "--since", help="Include findings on/after ISO date (e.g. 2026-01-01)"),
    export: str = typer.Option(None, "--export", "-o", help="Save report to this file path (.md)"),
):
    """📄 Synthesise all findings into a final research report.

    Pulls every stored finding for the task and passes them through the
    ResearchSynthesizer for a cohesive narrative.  Use ``--raw`` to skip
    synthesis and print findings sequentially.

    Examples::

        octane research report abc12345
        octane research report abc12345 --raw
        octane research report abc12345 --cycles 3
        octane research report abc12345 --since 2026-01-01
        octane research report abc12345 --export ~/reports/nvda.md
    """
    asyncio.run(_research_report(task_id, raw, cycles=cycles, since=since, export=export))


async def _research_report(
    task_id: str,
    raw_mode: bool,
    *,
    cycles: int | None = None,
    since: str | None = None,
    export: str | None = None,
):
    import pathlib
    from datetime import datetime, timezone
    from octane.research.store import ResearchStore
    from octane.research.synthesizer import ResearchSynthesizer
    from octane.config import settings

    _, redis_url = _get_research_shadow()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    task = await store.get_task(task_id)
    if task is None:
        console.print(f"[yellow]No task found: [bold]{task_id}[/][/]")
        await store.close()
        return

    console.print(Panel(
        f"[bold]Topic:[/]    {task.topic}\n"
        f"[bold]Cycles:[/]   {task.cycle_count}  ·  "
        f"[bold]Age:[/]      {task.age_hours:.1f}h",
        title=f"[bold magenta]📄 Research Report — {task_id}[/]",
        border_style="magenta",
    ))

    # Parse --since
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        except ValueError:
            console.print(f"[red]Invalid --since date: {since!r}. Use ISO format, e.g. 2026-01-01.[/]")
            await store.close()
            return

    if raw_mode:
        # ── Raw mode: fetch findings directly, print sequentially ──────────
        findings = await store.get_findings(task_id)
        await store.close()

        if not findings:
            console.print(
                f"[yellow]No findings stored yet for [bold]{task_id}[/].[/]"
            )
            return

        if since_dt:
            findings = [f for f in findings if (f.created_at or datetime.min.replace(tzinfo=timezone.utc)) >= since_dt]
        if cycles:
            findings = findings[-cycles:]

        if not findings:
            console.print("[yellow]No findings match the specified filters.[/]")
            return

        for f in findings:
            ts_str = f.created_at.strftime("%m-%d %H:%M UTC") if f.created_at else "unknown"
            console.print(Panel(
                f.content,
                title=f"[dim]Cycle {f.cycle_num}  ·  {ts_str}  ·  {f.word_count} words[/]",
                border_style="dim",
            ))
        return

    # ── Synthesis mode via ResearchSynthesizer ─────────────────────────────
    from octane.tools.bodega_inference import BodegaInferenceClient

    bodega = BodegaInferenceClient()
    try:
        with console.status("[dim]⚙  Synthesising findings…[/]", spinner="dots"):
            synth = ResearchSynthesizer(store, bodega=bodega)
            report_text = await synth.generate(
                task_id,
                cycles=cycles,
                since=since_dt,
            )
    except Exception as exc:
        console.print(f"[yellow]Synthesis unavailable ({exc}) — falling back to plain format.[/]\n")
        plain_synth = ResearchSynthesizer(store, bodega=None)
        report_text = await plain_synth.generate(task_id, cycles=cycles, since=since_dt)
    finally:
        await store.close()
        await bodega.close()

    console.print(Panel(
        report_text,
        title=f"[bold green]🔬 Synthesised Report: {task.topic}[/]",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Export to file ─────────────────────────────────────────────────────
    if export:
        out_path = pathlib.Path(export).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        console.print(f"[dim]  ✅ Report saved → {out_path}[/]")


@research_app.command("stop")
def research_stop(
    task_id: str = typer.Argument(..., help="Research task ID to cancel"),
):
    """🛑 Stop a background research task."""
    asyncio.run(_research_stop(task_id))


async def _research_stop(task_id: str):
    from shadows import Shadow
    from octane.research.store import ResearchStore
    from octane.config import settings

    shadow_name, redis_url = _get_research_shadow()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    task = await store.get_task(task_id)
    if task is None:
        console.print(f"[yellow]No task found: [bold]{task_id}[/][/]")
        await store.close()
        return

    # Cancel in Shadows
    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            await shadow.cancel(task_id)
        console.print(f"[green]✅ Shadows task [bold]{task_id}[/bold] cancelled.[/]")
    except Exception as exc:
        console.print(f"[yellow]Shadows cancel: {exc} (updating status anyway)[/]")

    # Mark stopped in Redis metadata
    await store.update_task_status(task_id, "stopped")
    await store.log_entry(task_id, "🛑 Task stopped by user")
    await store.close()

    console.print(
        f"[dim]Findings retained — run "
        f"[bold]octane research report {task_id}[/bold] to read the report.[/]"
    )


@research_app.command("list")
def research_list():
    """📋 List all research tasks with findings counts and last-run time."""
    asyncio.run(_research_list())


async def _research_list():
    from octane.research.store import ResearchStore
    from octane.config import settings

    _, redis_url = _get_research_shadow()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    tasks = await store.list_tasks()
    await store.close()

    if not tasks:
        console.print("[dim]No research tasks found.[/]")
        console.print("[dim]Start one with: octane research start \"<topic>\"[/]")
        return

    tbl = Table(
        title="Research Tasks",
        show_lines=True,
        border_style="cyan",
    )
    tbl.add_column("ID",       style="cyan",    width=10)
    tbl.add_column("Topic",    style="white",   min_width=30)
    tbl.add_column("Depth",    style="blue",    width=10, justify="center")
    tbl.add_column("Cycles",   style="green",   width=7,  justify="right")
    tbl.add_column("Findings", style="magenta", width=9,  justify="right")
    tbl.add_column("Every",    style="yellow",  width=7,  justify="right")
    tbl.add_column("Age",      style="dim",     width=8,  justify="right")
    tbl.add_column("Status",   width=9,         justify="center")

    for t in tasks:
        age_h = t.age_hours
        age_str = f"{age_h:.0f}h" if age_h >= 1 else f"{int(age_h * 60)}m"
        if t.is_active:
            status_str = "[green]● active[/]"
        else:
            status_str = "[red]◼ stopped[/]"
        depth_val = getattr(t, "depth", "deep")

        tbl.add_row(
            t.id,
            t.topic[:60],
            depth_val,
            str(t.cycle_count),
            str(t.finding_count),
            f"{t.interval_hours}h",
            age_str,
            status_str,
        )

    console.print(tbl)
    console.print(
        "[dim]  octane research log <id>       — view progress log\n"
        "  octane research report <id>    — synthesise findings\n"
        "  octane research stop <id>      — cancel task[/]"
    )


# ══════════════════════════════════════════════════════════════
# Session 18A — Files CLI
# ══════════════════════════════════════════════════════════════

files_app = typer.Typer(
    name="files",
    help="📁 Index and search local files (Session 18A).",
    no_args_is_help=True,
)
app.add_typer(files_app, name="files")


@files_app.command("add")
def files_add(
    path: str = typer.Argument(..., help="File or folder path to index"),
    project: str = typer.Option("", "--project", "-p", help="Project name (optional)"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into subdirectories"),
):
    """📄 Index a file or folder into Postgres."""
    asyncio.run(_files_add(path, project, recursive))


async def _files_add(path: str, project: str, recursive: bool):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import FileIndexer, ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable — cannot index files.[/]")
        return

    project_id = None
    if project:
        ps = ProjectStore(pg)
        row = await ps.create(project)
        project_id = row["id"] if row else None

    indexer = FileIndexer(pg, project_id=project_id)
    from pathlib import Path
    p = Path(path).expanduser().resolve()
    if p.is_file():
        row = await indexer.index_file(p)
        if row:
            console.print(f"[green]✅ Indexed:[/] {p.name}  ({row.get('word_count',0)} words)")
        else:
            console.print(f"[yellow]Skipped or failed: {p}[/]")
    elif p.is_dir():
        rows = await indexer.index_folder(p, recursive=recursive)
        console.print(f"[green]✅ Indexed {len(rows)} files from {p}[/]")
    else:
        console.print(f"[red]Path not found: {path}[/]")
    await pg.close()


@files_app.command("list")
def files_list(
    project: str = typer.Option("", "--project", "-p", help="Filter by project"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """📋 List indexed files."""
    asyncio.run(_files_list(project, limit))


async def _files_list(project: str, limit: int):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return

    project_id = None
    if project:
        ps = ProjectStore(pg)
        row = await ps.get(project)
        project_id = row["id"] if row else None

    if project_id is not None:
        rows = await pg.fetch(
            "SELECT id, filename, extension, word_count, indexed_at FROM user_files "
            "WHERE project_id=$1 ORDER BY indexed_at DESC LIMIT $2",
            project_id, limit,
        )
    else:
        rows = await pg.fetch(
            "SELECT id, filename, extension, word_count, indexed_at FROM user_files "
            "ORDER BY indexed_at DESC LIMIT $1", limit
        )

    if not rows:
        console.print("[dim]No files indexed yet. Run [bold]octane files add <path>[/bold].[/]")
        await pg.close(); return

    from rich.table import Table
    table = Table(title="Indexed Files")
    table.add_column("ID", style="dim")
    table.add_column("Filename")
    table.add_column("Ext", style="cyan")
    table.add_column("Words", justify="right")
    table.add_column("Indexed At", style="dim")
    for r in rows:
        table.add_row(
            str(r["id"]), r["filename"], r["extension"],
            str(r["word_count"]),
            str(r["indexed_at"])[:19],
        )
    console.print(table)
    await pg.close()


@files_app.command("search")
def files_search(
    query: str = typer.Argument(..., help="Semantic search query"),
    limit: int = typer.Option(5, "--limit", "-n"),
):
    """🔍 Semantic search across indexed files."""
    asyncio.run(_files_search(query, limit))


async def _files_search(query: str, limit: int):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import EmbeddingEngine
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return

    engine = EmbeddingEngine(pg)
    results = await engine.semantic_search(query, source_type="user_file", limit=limit)
    if not results:
        console.print("[dim]No results. Have you indexed any files? Run [bold]octane files add <path>[/bold].[/]")
        await pg.close(); return

    for i, r in enumerate(results, 1):
        dist = r.get("distance", 0)
        console.print(f"\n[bold cyan]#{i}[/] [dim](distance={dist:.3f})[/]")
        console.print(r.get("chunk_text", "")[:300])
    await pg.close()


@files_app.command("stats")
def files_stats(
    project: str = typer.Option("", "--project", "-p"),
):
    """📊 Show indexing statistics."""
    asyncio.run(_files_stats(project))


async def _files_stats(project: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import FileIndexer, ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return

    project_id = None
    if project:
        ps = ProjectStore(pg)
        row = await ps.get(project)
        project_id = row["id"] if row else None

    indexer = FileIndexer(pg, project_id=project_id)
    stats = await indexer.stats(project_id=project_id)
    console.print(f"[bold]Total files:[/] {stats['total_files']}")
    console.print(f"[bold]Total words:[/] {stats['total_words']:,}")
    if stats["by_extension"]:
        from rich.table import Table
        table = Table(title="By Extension")
        table.add_column("Extension"); table.add_column("Files", justify="right"); table.add_column("Words", justify="right")
        for row in stats["by_extension"]:
            table.add_row(row["extension"], str(row["n"]), str(int(row.get("words") or 0)))
        console.print(table)
    await pg.close()


@files_app.command("reindex")
def files_reindex(
    path: str = typer.Argument(..., help="File to force-reindex"),
):
    """🔄 Force re-index a file (ignores cached hash)."""
    asyncio.run(_files_reindex(path))


async def _files_reindex(path: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import FileIndexer
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return
    indexer = FileIndexer(pg)
    row = await indexer.reindex(path)
    if row:
        console.print(f"[green]✅ Re-indexed:[/] {row.get('filename')} ({row.get('word_count',0)} words)")
    else:
        console.print(f"[yellow]Reindex failed for: {path}[/]")
    await pg.close()


# ══════════════════════════════════════════════════════════════
# Session 18A — Project CLI
# ══════════════════════════════════════════════════════════════

project_app = typer.Typer(
    name="project",
    help="🗂  Manage research projects (Session 18A).",
    no_args_is_help=True,
)
app.add_typer(project_app, name="project")


@project_app.command("list")
def project_list(
    all_: bool = typer.Option(False, "--all", "-a", help="Include archived projects"),
):
    """📋 List all projects."""
    asyncio.run(_project_list(all_))


async def _project_list(include_archived: bool):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return

    ps = ProjectStore(pg)
    rows = await ps.list(include_archived=include_archived)
    if not rows:
        console.print("[dim]No projects. Run [bold]octane project create <name>[/bold].[/]")
        await pg.close(); return

    from rich.table import Table
    table = Table(title="Projects")
    table.add_column("ID", style="dim"); table.add_column("Name", style="bold")
    table.add_column("Status", style="cyan"); table.add_column("Created", style="dim")
    for r in rows:
        table.add_row(str(r["id"]), r["name"], r["status"], str(r["created_at"])[:10])
    console.print(table)
    await pg.close()


@project_app.command("create")
def project_create(
    name: str = typer.Argument(..., help="Project name"),
    description: str = typer.Option("", "--desc", "-d"),
):
    """➕ Create a new project."""
    asyncio.run(_project_create(name, description))


async def _project_create(name: str, description: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return
    ps = ProjectStore(pg)
    row = await ps.create(name, description)
    if row:
        console.print(f"[green]✅ Project created:[/] [bold]{name}[/] (ID {row['id']})")
    else:
        console.print(f"[red]Failed to create project: {name}[/]")
    await pg.close()


@project_app.command("show")
def project_show(
    name: str = typer.Argument(..., help="Project name"),
):
    """🔍 Show project details and content counts."""
    asyncio.run(_project_show(name))


async def _project_show(name: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return
    ps = ProjectStore(pg)
    row = await ps.get(name)
    if not row:
        console.print(f"[yellow]Project not found: {name}[/]"); await pg.close(); return

    pid = row["id"]
    pages = await pg.fetchval("SELECT COUNT(*) FROM web_pages WHERE project_id=$1", pid)
    files = await pg.fetchval("SELECT COUNT(*) FROM user_files WHERE project_id=$1", pid)
    artifacts = await pg.fetchval("SELECT COUNT(*) FROM generated_artifacts WHERE project_id=$1", pid)
    findings = await pg.fetchval("SELECT COUNT(*) FROM research_findings_v2 WHERE project_id=$1", pid)

    console.print(f"[bold cyan]Project:[/] {row['name']}  (ID {pid})")
    console.print(f"  Status:    {row['status']}")
    console.print(f"  Created:   {str(row['created_at'])[:19]}")
    if row.get("description"):
        console.print(f"  Desc:      {row['description']}")
    console.print(f"\n  [bold]Web pages:[/]   {pages or 0}")
    console.print(f"  [bold]Files:[/]       {files or 0}")
    console.print(f"  [bold]Artifacts:[/]   {artifacts or 0}")
    console.print(f"  [bold]Findings v2:[/] {findings or 0}")
    await pg.close()


@project_app.command("archive")
def project_archive(
    name: str = typer.Argument(..., help="Project name to archive"),
):
    """📦 Archive a project (soft-delete)."""
    asyncio.run(_project_archive(name))


async def _project_archive(name: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]"); return
    ps = ProjectStore(pg)
    ok = await ps.archive(name)
    if ok:
        console.print(f"[green]✅ Archived:[/] {name}")
    else:
        console.print(f"[yellow]Project not found: {name}[/]")
    await pg.close()


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    app()
