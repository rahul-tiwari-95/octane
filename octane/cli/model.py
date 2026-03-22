"""octane model sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel

from octane.cli._shared import console

model_app = typer.Typer(
    name="model",
    help="🧠 Manage the loaded LLM (reload, switch, inspect).",
    no_args_is_help=True,
)


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
