"""octane db sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console

db_app = typer.Typer(
    name="db",
    help="🗄 Postgres schema migrations.",
    no_args_is_help=True,
)


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
