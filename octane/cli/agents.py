"""octane agents and version commands."""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_synapse


def register(app: typer.Typer) -> None:
    app.command()(agents)
    app.command()(version)


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
    arch = platform.machine()

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
