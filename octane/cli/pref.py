"""octane pref sub-app."""

from __future__ import annotations

import asyncio

import typer

from octane.cli._shared import console

pref_app = typer.Typer(
    name="pref",
    help="⚙  Manage user preferences — controls verbosity, expertise, response style.",
    no_args_is_help=True,
)

_PREF_CHOICES: dict[str, list[str]] = {
    "verbosity":              ["concise", "detailed"],
    "expertise":              ["beginner", "intermediate", "advanced"],
    "response_style":         ["prose", "bullets", "code-first"],
    "domains":                [],
    "assistant_name":         [],
    "assistant_personality":  [],
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

    from rich.table import Table as _Table
    table = _Table(title=f"Preferences — user: [bold]{user_id}[/]", show_lines=False, box=None, padding=(0, 2))
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


# ── Response Templates ────────────────────────────────────────────────────────

from pathlib import Path as _Path

_TEMPLATES_DIR = _Path.home() / ".octane" / "templates"

_VALID_TEMPLATE_NAMES = (
    "compare", "investigate", "news", "search", "ask", "evaluate", "chat",
)


@pref_app.command("template")
def pref_template(
    name: str = typer.Argument(
        None,
        help="Template name to view (e.g. compare, investigate, news, ask, chat).",
    ),
):
    """📝 List or view response templates.

    Templates let you customise how Octane formats its output for each command.

    Place a ``.txt`` file in ``~/.octane/templates/`` with your formatting
    instructions.  The file name maps to the command:

        compare.txt     → octane compare
        investigate.txt → octane investigate
        news.txt        → octane news / web news
        search.txt      → web search results
        ask.txt         → octane ask --recall
        evaluate.txt    → OSA evaluator
        chat.txt        → octane chat (command results)

    Examples::

        octane pref template              # list all templates
        octane pref template compare      # view compare template
    """
    if name is None:
        # List all templates
        if not _TEMPLATES_DIR.is_dir():
            console.print(f"[dim]No templates directory found at {_TEMPLATES_DIR}[/]")
            console.print("[dim]Create it with: octane pref template-set compare \"Your instructions here\"[/]")
            return

        found = False
        for tname in _VALID_TEMPLATE_NAMES:
            path = _TEMPLATES_DIR / f"{tname}.txt"
            if path.is_file():
                content = path.read_text(encoding="utf-8").strip()
                preview = (content[:80] + "…") if len(content) > 80 else content
                console.print(f"  [cyan]{tname}[/]  →  {preview}")
                found = True
            else:
                console.print(f"  [dim]{tname}[/]  →  [dim](default)[/]")
        if not found:
            console.print("\n[dim]No custom templates. Use: octane pref template-set <name> \"instructions\"[/]")
    else:
        if name not in _VALID_TEMPLATE_NAMES:
            console.print(f"[red]Unknown template: {name}[/]")
            console.print(f"[dim]Valid: {', '.join(_VALID_TEMPLATE_NAMES)}[/]")
            raise typer.Exit(1)
        path = _TEMPLATES_DIR / f"{name}.txt"
        if not path.is_file():
            console.print(f"[dim]{name}: using default (no custom template)[/]")
        else:
            console.print(f"[cyan]{name}.txt[/]:")
            console.print(path.read_text(encoding="utf-8").strip())


@pref_app.command("template-set")
def pref_template_set(
    name: str = typer.Argument(..., help="Template name (e.g. compare, investigate, news)."),
    instructions: str = typer.Argument(..., help="Formatting instructions in plain English."),
):
    """✏  Set a custom response template.

    Examples::

        octane pref template-set compare "Use bullet points. End with a haiku."
        octane pref template-set news "Respond as a formal BBC news anchor."
        octane pref template-set investigate "Write in the style of a detective report."
    """
    if name not in _VALID_TEMPLATE_NAMES:
        console.print(f"[red]Unknown template: {name}[/]")
        console.print(f"[dim]Valid: {', '.join(_VALID_TEMPLATE_NAMES)}[/]")
        raise typer.Exit(1)

    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    path = _TEMPLATES_DIR / f"{name}.txt"
    path.write_text(instructions.strip() + "\n", encoding="utf-8")

    # Clear the in-process cache so it picks up the new template immediately
    from octane.utils.response_templates import clear_cache
    clear_cache()

    console.print(f"[green]✅ Template saved:[/] {path}")
    console.print(f"[dim]  {instructions.strip()[:100]}[/]")


@pref_app.command("template-reset")
def pref_template_reset(
    name: str = typer.Argument(
        None,
        help="Template to reset. Omit to reset all templates.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """🔄 Remove a custom response template (reverts to default).

    Examples::

        octane pref template-reset compare
        octane pref template-reset --yes    # reset all
    """
    if name:
        if name not in _VALID_TEMPLATE_NAMES:
            console.print(f"[red]Unknown template: {name}[/]")
            raise typer.Exit(1)
        path = _TEMPLATES_DIR / f"{name}.txt"
        if path.is_file():
            path.unlink()
            from octane.utils.response_templates import clear_cache
            clear_cache()
            console.print(f"[green]✅ {name} template removed — using default[/]")
        else:
            console.print(f"[dim]{name}: already using default[/]")
    else:
        if not _TEMPLATES_DIR.is_dir():
            console.print("[dim]No templates to reset.[/]")
            return
        templates = list(_TEMPLATES_DIR.glob("*.txt"))
        if not templates:
            console.print("[dim]No custom templates found.[/]")
            return
        if not yes:
            confirm = console.input(
                f"[yellow]Remove {len(templates)} custom template(s)? [y/N]: [/]"
            ).strip().lower()
            if confirm not in ("y", "yes"):
                console.print("[dim]Cancelled.[/]")
                return
        for t in templates:
            t.unlink()
        from octane.utils.response_templates import clear_cache
        clear_cache()
        console.print(f"[green]✅ {len(templates)} template(s) removed — all using defaults[/]")
