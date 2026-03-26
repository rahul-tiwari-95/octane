"""octane power commands — investigate, compare, chain."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from octane.cli._shared import console


def register(app: typer.Typer) -> None:
    app.command()(investigate)
    app.command()(compare)
    app.command()(chain)


def investigate(
    query: str = typer.Argument(..., help="The topic to investigate deeply."),
    max_dimensions: int = typer.Option(
        None,
        "--max-dimensions",
        "-d",
        help="Maximum number of research dimensions (2-8).",
        min=2,
        max=8,
    ),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream findings as they arrive."),
):
    """🔍 Investigate a topic across multiple independent research dimensions.

    Decomposes your query into parallel research threads, runs each through
    the full web + memory agent stack, then synthesizes a structured report.

    Examples:
        octane investigate "impact of tariffs on semiconductor supply chains"
        octane investigate "longevity interventions" --max-dimensions 6
    """
    asyncio.run(_investigate(query, max_dimensions, stream))


async def _investigate(query: str, max_dimensions: int | None, stream: bool):
    from rich.markdown import Markdown
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from octane.osa.investigate import InvestigateOrchestrator
    from octane.models.synapse import SynapseEventBus
    from octane.agents.web.agent import WebAgent

    synapse = SynapseEventBus()
    web_agent = WebAgent(synapse)
    orchestrator = InvestigateOrchestrator(web_agent=web_agent)

    console.print(
        Panel(
            f"[bold white]{query}[/]",
            title="[bold cyan]🔍 Investigating[/]",
            border_style="cyan",
        )
    )
    console.print(
        "[dim]⚙  Planning dimensions (~5s) · Web research in parallel · "
        "Synthesis via 8B model[/]"
    )
    kwargs: dict = {}
    if max_dimensions is not None:
        kwargs["max_dimensions"] = max_dimensions
    report_text = ""
    total_ms = 0.0
    n_ok = 0
    n_total = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=False,
    ) as progress:
        plan_task = progress.add_task("[cyan]Planning dimensions (8B)…[/]", total=None)
        research_task = None

        async for event in orchestrator.run_stream(query, **kwargs):
            etype = event.get("type")

            if etype == "plan":
                data = event["data"]
                n = len(data.get("dimensions", []))
                progress.update(plan_task, description=f"[green]✅ {n} dimensions planned[/]", completed=1, total=1)
                if stream:
                    dims = data.get("dimensions", [])
                    tbl = Table(show_header=True, box=None, padding=(0, 2))
                    tbl.add_column("#", style="dim", width=3)
                    tbl.add_column("Dimension", style="cyan")
                    tbl.add_column("Priority", style="yellow", width=8)
                    tbl.add_column("Policy / Query", style="dim")
                    for d in dims:
                        policy = (
                            d.get("queries", [""])[0][:60] if d.get("queries")
                            else d.get("rationale", "")[:60]
                        )
                        tbl.add_row(d.get("id", ""), d.get("label", ""), str(d.get("priority", "")), policy)
                    console.print(tbl)
                research_task = progress.add_task("[cyan]Researching…[/]", total=n, completed=0)

            elif etype == "finding":
                data = event["data"]
                label = data.get("dimension_label", "?")
                success = data.get("success", False)
                latency = data.get("latency_ms", 0)
                icon = "✅" if success else "⚠️"
                if research_task is not None:
                    progress.advance(research_task)
                if stream:
                    console.print(f"  {icon} [cyan]{label}[/]  [dim]{latency:.0f}ms[/]")

            elif etype == "synthesis":
                data = event["data"]
                report_text = data.get("report", "")
                t = plan_task if research_task is None else research_task
                progress.update(t, description="[yellow]🧠 Synthesizing (8B)…[/]")

            elif etype == "done":
                data = event["data"]
                n_ok = data.get("dimensions_completed", 0)
                n_total = n_ok + data.get("dimensions_failed", 0)
                total_ms = data.get("total_ms", 0.0)

    console.print(
        Panel(
            Markdown(report_text) if report_text else "[dim]No report generated.[/]",
            title="[bold green]📋 Investigation Report[/]",
            border_style="green",
        )
    )
    console.print(f"[dim]{n_ok}/{n_total} dimensions successful · {total_ms / 1000:.1f}s total[/]")


def compare(
    query: str = typer.Argument(..., help="The comparison query, e.g. 'NVDA vs AMD vs INTC'."),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream cells as they complete."),
):
    """⚖️  Compare items across multiple dimensions in a structured matrix.

    Extracts the items to compare from your query, plans comparison dimensions,
    runs parallel research for every (item × dimension) cell, and synthesizes
    a side-by-side report with a verdict.

    Examples:
        octane compare "NVDA vs AMD vs INTC"
        octane compare "compare React, Vue, and Svelte for a startup"
        octane compare "Python or Go for a backend API service"
    """
    asyncio.run(_compare(query, stream))


async def _compare(query: str, stream: bool):
    from rich.markdown import Markdown
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from octane.osa.compare import CompareOrchestrator
    from octane.models.synapse import SynapseEventBus
    from octane.agents.web.agent import WebAgent

    synapse = SynapseEventBus()
    web_agent = WebAgent(synapse)
    orchestrator = CompareOrchestrator(web_agent=web_agent)

    console.print(
        Panel(
            f"[bold white]{query}[/]",
            title="[bold yellow]⚖️  Comparing[/]",
            border_style="yellow",
        )
    )
    console.print(
        "[dim]⚙  Planning comparison matrix (~5s) · Web research in parallel · "
        "Synthesis via 8B model[/]"
    )
    n_done = 0
    report_text = ""
    total_ms = 0.0
    n_ok = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=False,
    ) as progress:
        plan_task = progress.add_task("[cyan]Planning comparison matrix (8B)…[/]", total=None)
        cell_task = None

        async for event in orchestrator.run_stream(query):
            etype = event.get("type")

            if etype == "plan":
                data = event["data"]
                items = data.get("items", [])
                dims = data.get("dimensions", [])
                n_items = len(items)
                n_dims = len(dims)
                n_cells = n_items * n_dims
                progress.update(
                    plan_task,
                    description=f"[green]✅ {n_items} items × {n_dims} dims = {n_cells} cells[/]",
                    completed=1,
                    total=1,
                )
                if stream:
                    plan_tbl = Table(show_header=True, box=None, padding=(0, 2))
                    plan_tbl.add_column("Item", style="yellow")
                    plan_tbl.add_column("Dimensions", style="cyan")
                    plan_tbl.add_column("Policy / Angle", style="dim")
                    for item in items:
                        dim_labels = ", ".join(d.get("label", "") for d in dims[:3])
                        policy = item.get("canonical_query", "")[:55]
                        plan_tbl.add_row(item.get("label", "?"), dim_labels, policy)
                    console.print(plan_tbl)
                cell_task = progress.add_task("[yellow]Researching cells…[/]", total=n_cells, completed=0)

            elif etype == "cell":
                n_done += 1
                if cell_task is not None:
                    progress.advance(cell_task)
                if stream:
                    d = event["data"]
                    item = d.get("item_label", "?")
                    dim = d.get("dimension_label", "?")
                    ok = "✅" if d.get("success") else "⚠️"
                    ms = d.get("latency_ms", 0)
                    console.print(f"  {ok} [yellow]{item}[/] / [cyan]{dim}[/]  [dim]{ms:.0f}ms[/]")

            elif etype == "matrix":
                if cell_task is not None:
                    progress.update(cell_task, description="[green]✅ Matrix complete[/]")

            elif etype == "synthesis":
                data = event["data"]
                report_text = data.get("report", "")

            elif etype == "done":
                data = event["data"]
                total_ms = data.get("total_ms", 0.0)
                n_ok = data.get("n_successful", 0)

    console.print(
        Panel(
            Markdown(report_text) if report_text else "[dim]No report generated.[/]",
            title="[bold yellow]📊 Comparison Report[/]",
            border_style="yellow",
        )
    )
    console.print(
        f"[dim]{n_ok}/{n_cells} cells successful · {total_ms / 1000:.1f}s total[/]"
    )


def chain(
    steps: list[str] = typer.Argument(
        ...,
        help="Ordered steps, e.g. 'ask What is NVDA revenue' 'synthesize {prev}'",
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        "-v",
        help="Template variables as KEY=VALUE (can be repeated).  e.g. --var ticker=NVDA",
    ),
    save: str = typer.Option(
        None,
        "--save",
        "-s",
        help="Save chain definition to a JSON workflow file.",
    ),
):
    """⛓️  Execute a multi-step pipeline with variable interpolation.

    Steps are executed in order.  Use {prev} to reference the previous step's
    output, {step_name} to reference a named step, and {all} to join all
    prior outputs.  Use {{var}} for template variables supplied via --var.

    Examples:
        octane chain "ask What are NVDA's main products" "synthesize {prev}"
        octane chain "search NVDA earnings" "analyze {prev}" "synthesize {all}"
        octane chain "ask {{ticker}} revenue forecast" --var ticker=NVDA
        octane chain "search {{topic}} latest news" --var topic="AI chips" --save chain.json
    """
    asyncio.run(_chain(steps, var, save))


async def _chain(steps: list[str], var: list[str], save: str | None):
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from octane.osa.chain_parser import ChainParser, ChainValidationError
    from octane.osa.chain import ChainExecutor

    template_vars: dict[str, str] = {}
    for v in var:
        if "=" in v:
            k, _, val = v.partition("=")
            template_vars[k.strip()] = val.strip()
        else:
            console.print(f"[yellow]⚠️  Ignoring malformed --var: {v!r} (expected KEY=VALUE)[/]")

    parser = ChainParser()
    try:
        plan = parser.parse(list(steps), template_vars=template_vars)
    except ChainValidationError as exc:
        console.print(
            Panel(
                f"[red]Step {exc.step_index + 1}: {exc.reason}\n\n"
                f"Raw: [white]{exc.step_raw}[/]",
                title="[red]Chain Parse Error[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    n = len(plan.steps)
    console.print(
        Panel(
            "\n".join(
                f"  [dim]{i + 1}.[/] [cyan]{s.name}[/]: [white]{s.command}[/] {s.args}"
                for i, s in enumerate(plan.steps)
            ),
            title=f"[bold magenta]⛓️  Chain ({n} steps)[/]",
            border_style="magenta",
        )
    )

    executor = ChainExecutor(template_vars=template_vars)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        tasks: dict[str, object] = {}

        async for event in executor.run_stream(plan, save_path=save):
            etype = event.get("type")

            if etype == "step_start":
                d = event["data"]
                tid = progress.add_task(
                    f"[magenta]{d['name']}[/] → [white]{d['command']}[/]",
                    total=None,
                )
                tasks[d["name"]] = tid

            elif etype == "step_done":
                d = event["data"]
                tid = tasks.get(d["name"])
                if tid is not None:
                    progress.update(
                        tid,  # type: ignore[arg-type]
                        description=f"[green]✅ {d['name']}[/]  [dim]{d['latency_ms']:.0f}ms[/]",
                        completed=1,
                        total=1,
                    )
                output_preview = (d.get("output", "") or "")[:120].replace("\n", " ")
                console.print(f"  ✅ [green]{d['name']}[/]  [dim]{output_preview}…[/]")

            elif etype == "step_error":
                d = event["data"]
                console.print(f"  ❌ [red]{d['name']}[/]: {d.get('error', 'unknown error')}")

            elif etype == "done":
                d = event["data"]
                n_ok = d["n_successful"]
                n_total = d["n_steps"]
                total_ms = d["total_ms"]
                saved_to = d.get("saved_to")

                console.print(Rule(style="magenta"))
                console.print(
                    f"[bold]Chain complete:[/] {n_ok}/{n_total} steps "
                    f"successful · [dim]{total_ms / 1000:.1f}s[/]"
                )
                if saved_to:
                    console.print(f"[dim]Chain saved to: {saved_to}[/]")

                final = d.get("final_output", "")
                if final:
                    console.print(
                        Panel(
                            Markdown(final) if len(final) > 80 else final,
                            title="[bold green]🏁 Final Output[/]",
                            border_style="green",
                        )
                    )
