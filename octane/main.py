"""Octane CLI — the user interface.

Commands:
    octane health    — System status (SysStat Agent)
    octane ask       — Ask a question (routed through OSA)
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

        else:
            console.print(f"[red]Health check failed: {response.error}[/]")

        console.print(f"\n[dim]Duration: {response.duration_ms}ms | Correlation: {response.correlation_id}[/]")

    finally:
        await bodega.close()


# ── octane ask ────────────────────────────────────────────────


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question or instruction"),
):
    """🧠 Ask Octane anything — routed through OSA."""
    asyncio.run(_ask(query))


async def _ask(query: str):
    from octane.osa.orchestrator import Orchestrator

    synapse = _get_synapse()
    osa = Orchestrator(synapse)

    console.print(f"\n[dim]Processing: {query}[/]\n")

    output = await osa.run(query)

    console.print(Panel(
        output,
        title="[bold green]🔥 Octane[/]",
        border_style="green",
    ))

    # Show trace summary
    trace = synapse.get_recent_traces(limit=1)
    if trace:
        t = trace[0]
        console.print(f"\n[dim]Agents: {', '.join(t.agents_used)} | "
                      f"Events: {len(t.events)} | "
                      f"Duration: {t.total_duration_ms}ms | "
                      f"Trace: {t.correlation_id}[/]")


# ── octane trace ──────────────────────────────────────────────


@app.command()
def trace(
    correlation_id: str = typer.Argument(
        None,
        help="Correlation ID to trace. If omitted, shows recent traces.",
    ),
):
    """🔍 View Synapse trace for a query lifecycle."""
    synapse = _get_synapse()

    if correlation_id:
        t = synapse.get_trace(correlation_id)
        if not t.events:
            console.print(f"[yellow]No trace found for: {correlation_id}[/]")
            return

        console.print(Panel(
            f"[bold]Correlation ID:[/] {t.correlation_id}\n"
            f"[bold]Duration:[/] {t.total_duration_ms}ms\n"
            f"[bold]Success:[/] {'✅' if t.success else '❌'}\n"
            f"[bold]Agents:[/] {', '.join(t.agents_used)}",
            title="[bold blue]Synapse Trace[/]",
            border_style="blue",
        ))

        # Event table
        table = Table(title="Events", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Type", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Target", style="magenta")
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Error", style="red")

        for i, event in enumerate(t.events, 1):
            table.add_row(
                str(i),
                event.event_type,
                event.source,
                event.target or "—",
                f"{event.duration_ms}ms" if event.duration_ms else "—",
                event.error[:50] if event.error else "—",
            )

        console.print(table)
    else:
        traces = synapse.get_recent_traces()
        if not traces:
            console.print("[yellow]No traces recorded yet. Try running 'octane ask' first.[/]")
            return

        table = Table(title="Recent Traces")
        table.add_column("Correlation ID", style="cyan")
        table.add_column("Events", justify="right")
        table.add_column("Agents", style="green")
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Status", justify="center")

        for t in traces:
            table.add_row(
                t.correlation_id[:12] + "...",
                str(len(t.events)),
                ", ".join(t.agents_used),
                f"{t.total_duration_ms}ms",
                "✅" if t.success else "❌",
            )

        console.print(table)


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
