"""octane workflow sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_synapse, _print_dag_trace

workflow_app = typer.Typer(
    name="workflow",
    help="🗂  Save, list, and replay Octane pipeline templates.",
    no_args_is_help=True,
)


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
            mtime = os.path.getmtime(f)
            from datetime import datetime as _dt
            mod_str = _dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
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
    if not path.exists():
        from octane.workflow import WORKFLOW_DIR
        fallback = WORKFLOW_DIR / file_str
        if fallback.exists():
            path = fallback
        else:
            fallback2 = WORKFLOW_DIR / f"{file_str}.workflow.json"
            if fallback2.exists():
                path = fallback2

    try:
        template = load_workflow(path)
    except FileNotFoundError:
        console.print(f"[red]Workflow file not found: {file_str}[/]")
        console.print("[dim]Run 'octane workflow list' to see available templates.[/]")
        raise typer.Exit(1)

    overrides: dict[str, str] = {}
    for item in var_list:
        if "=" in item:
            k, _, v = item.partition("=")
            overrides[k.strip()] = v.strip()
        else:
            console.print(f"[yellow]Ignoring malformed --var '{item}' (expected key=value)[/]")

    if query_override:
        overrides["query"] = query_override

    try:
        dag = template.to_dag(overrides)
    except Exception as exc:
        console.print(f"[red]Failed to build DAG from template: {exc}[/]")
        raise typer.Exit(1)

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
        status.stop()

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
