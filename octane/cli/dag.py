"""octane dag command."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_synapse


def register(app: typer.Typer) -> None:
    app.command()(dag)


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

    with console.status("[dim]Connecting to inference engine...[/]", spinner="dots"):
        status = await osa.pre_flight(wait_for_bodega=True)

    routing_mode = (
        "[green]LLM[/]" if (status["bodega_reachable"] and status["model_loaded"])
        else "[yellow]keyword fallback[/]"
    )

    with console.status("[dim]Decomposing...[/]", spinner="dots"):
        task_dag = await osa.decomposer.decompose(query)

    console.print(Panel(
        f"[bold]Query:[/]   {query}\n"
        f"[bold]Routing:[/] {routing_mode}  ·  "
        f"[bold]Nodes:[/] {len(task_dag.nodes)}  ·  "
        f"[bold]Waves:[/] {len(task_dag.execution_order())}\n"
        f"[bold]Reason:[/]  [dim]{task_dag.reasoning[:120] or '—'}[/]",
        title="[bold cyan]🔀 DAG Dry-Run[/]",
        border_style="cyan",
    ))

    node_table = Table(show_lines=False, box=None, padding=(0, 2))
    node_table.add_column("#", style="dim", width=3, justify="right")
    node_table.add_column("Wave", style="dim", width=5, justify="center")
    node_table.add_column("Agent", style="cyan", width=12)
    node_table.add_column("Sub-agent", style="green", width=16)
    node_table.add_column("Template", style="yellow", width=20)
    node_table.add_column("Instruction", style="white")

    wave_map: dict[str, int] = {}
    for wave_idx, wave in enumerate(task_dag.execution_order(), 1):
        for node in wave:
            wave_map[node.task_id] = wave_idx

    _AGENT_COLOURS = {
        "web": "cyan", "code": "yellow", "memory": "blue",
        "sysstat": "green", "pnl": "magenta",
    }

    for i, node in enumerate(task_dag.nodes, 1):
        wave_num = wave_map.get(node.task_id, "?")
        template = node.metadata.get("template", "—")
        sub_agent = node.metadata.get("sub_agent", "—")
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
