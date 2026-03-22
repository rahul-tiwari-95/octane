"""octane project sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.table import Table

from octane.cli._shared import console

project_app = typer.Typer(
    name="project",
    help="📂 Manage projects.",
    no_args_is_help=True,
)


@project_app.command("list")
def project_list():
    """📋 List all projects."""
    asyncio.run(_project_list())


async def _project_list():
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    ps = ProjectStore(pg)
    projects = await ps.list()
    if not projects:
        console.print("[dim]No projects yet. Create one with [bold]octane project create <name>[/bold].[/]")
        await pg.close()
        return

    table = Table(title="Projects")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Files", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Archived", justify="center")
    table.add_column("Created", style="dim")
    for p in projects:
        table.add_row(
            str(p["id"]),
            p["name"],
            str(p.get("file_count", 0)),
            str(p.get("word_count", 0)),
            "✓" if p.get("archived") else "",
            str(p.get("created_at", ""))[:19],
        )
    console.print(table)
    await pg.close()


@project_app.command("create")
def project_create(
    name: str = typer.Argument(..., help="Project name"),
    description: str = typer.Option("", "--description", "-d"),
):
    """✨ Create a new project."""
    asyncio.run(_project_create(name, description))


async def _project_create(name: str, description: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    ps = ProjectStore(pg)
    row = await ps.create(name, description=description)
    if row:
        console.print(f"[green]✅ Project created:[/] [bold]{row['name']}[/] (id={row['id']})")
    else:
        console.print(f"[yellow]Failed to create project '{name}'. Does it already exist?[/]")
    await pg.close()


@project_app.command("show")
def project_show(
    name: str = typer.Argument(..., help="Project name or ID"),
):
    """🔎 Show project details and file list."""
    asyncio.run(_project_show(name))


async def _project_show(name: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore, FileIndexer
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    ps = ProjectStore(pg)
    try:
        project_id = int(name)
        row = await ps.get_by_id(project_id)
    except ValueError:
        row = await ps.get(name)

    if not row:
        console.print(f"[red]Project not found: {name}[/]")
        await pg.close()
        return

    console.print(f"\n[bold]Project:[/] {row['name']}  (id={row['id']})")
    if row.get("description"):
        console.print(f"[dim]{row['description']}[/]")
    console.print(f"[dim]Created: {str(row.get('created_at',''))[:19]}[/]")
    if row.get("archived"):
        console.print("[yellow]⚠ Archived[/]")

    indexer = FileIndexer(pg, project_id=row["id"])
    files = await pg.fetch(
        "SELECT filename, extension, word_count, indexed_at FROM user_files "
        "WHERE project_id=$1 ORDER BY indexed_at DESC LIMIT 30",
        row["id"],
    )
    if files:
        table = Table(title=f"Files in '{row['name']}'")
        table.add_column("Filename")
        table.add_column("Ext", style="cyan")
        table.add_column("Words", justify="right")
        table.add_column("Indexed", style="dim")
        for f in files:
            table.add_row(f["filename"], f["extension"], str(f["word_count"]), str(f["indexed_at"])[:19])
        console.print(table)
    else:
        console.print("[dim]No files indexed yet for this project.[/]")
    await pg.close()


@project_app.command("archive")
def project_archive(
    name: str = typer.Argument(..., help="Project name or ID"),
    restore: bool = typer.Option(False, "--restore", help="Un-archive instead"),
):
    """📦 Archive (or restore) a project."""
    asyncio.run(_project_archive(name, restore))


async def _project_archive(name: str, restore: bool):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    ps = ProjectStore(pg)
    try:
        project_id = int(name)
        row = await ps.get_by_id(project_id)
    except ValueError:
        row = await ps.get(name)

    if not row:
        console.print(f"[red]Project not found: {name}[/]")
        await pg.close()
        return

    if restore:
        await ps.set_archived(row["id"], False)
        console.print(f"[green]✅ Restored project:[/] {row['name']}")
    else:
        await ps.set_archived(row["id"], True)
        console.print(f"[yellow]📦 Archived project:[/] {row['name']}")
    await pg.close()
