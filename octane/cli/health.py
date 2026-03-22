"""octane health / sysstat commands."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_synapse


def register(app: typer.Typer) -> None:
    app.command()(health)
    app.command()(sysstat)


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
            data = response.data
            system = data.get("system", {})
            model = data.get("model", {})
            server_health = data.get("server_health", {})

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


def sysstat():
    """📊 Live system snapshot — RAM, CPU, loaded model (no Bodega required)."""
    asyncio.run(_sysstat())


async def _sysstat():
    from octane.agents.sysstat.agent import SysStatAgent
    from octane.models.schemas import AgentRequest
    from octane.tools.bodega_inference import BodegaInferenceClient

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
