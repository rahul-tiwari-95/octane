"""octane files sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console

files_app = typer.Typer(
    name="files",
    help="📁 Index and search local files.",
    no_args_is_help=True,
)


@files_app.command("add")
def files_add(
    path: str = typer.Argument(..., help="File or folder path to index"),
    project: str = typer.Option("", "--project", "-p", help="Project name (optional)"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into subdirectories"),
):
    """📄 Index a file or folder into Postgres."""
    asyncio.run(_files_add(path, project, recursive))


async def _files_add(path: str, project: str, recursive: bool):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import FileIndexer, ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable — cannot index files.[/]")
        return

    project_id = None
    if project:
        ps = ProjectStore(pg)
        row = await ps.create(project)
        project_id = row["id"] if row else None

    indexer = FileIndexer(pg, project_id=project_id)
    from pathlib import Path
    p = Path(path).expanduser().resolve()
    if p.is_file():
        row = await indexer.index_file(p)
        if row:
            console.print(f"[green]✅ Indexed:[/] {p.name}  ({row.get('word_count',0)} words)")
        else:
            console.print(f"[yellow]Skipped or failed: {p}[/]")
    elif p.is_dir():
        rows = await indexer.index_folder(p, recursive=recursive)
        console.print(f"[green]✅ Indexed {len(rows)} files from {p}[/]")
    else:
        console.print(f"[red]Path not found: {path}[/]")
    await pg.close()


@files_app.command("list")
def files_list(
    project: str = typer.Option("", "--project", "-p", help="Filter by project"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """📋 List indexed files."""
    asyncio.run(_files_list(project, limit))


async def _files_list(project: str, limit: int):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    project_id = None
    if project:
        ps = ProjectStore(pg)
        row = await ps.get(project)
        project_id = row["id"] if row else None

    if project_id is not None:
        rows = await pg.fetch(
            "SELECT id, filename, extension, word_count, indexed_at FROM user_files "
            "WHERE project_id=$1 ORDER BY indexed_at DESC LIMIT $2",
            project_id, limit,
        )
    else:
        rows = await pg.fetch(
            "SELECT id, filename, extension, word_count, indexed_at FROM user_files "
            "ORDER BY indexed_at DESC LIMIT $1", limit
        )

    if not rows:
        console.print("[dim]No files indexed yet. Run [bold]octane files add <path>[/bold].[/]")
        await pg.close()
        return

    table = Table(title="Indexed Files")
    table.add_column("ID", style="dim")
    table.add_column("Filename")
    table.add_column("Ext", style="cyan")
    table.add_column("Words", justify="right")
    table.add_column("Indexed At", style="dim")
    for r in rows:
        table.add_row(
            str(r["id"]), r["filename"], r["extension"],
            str(r["word_count"]),
            str(r["indexed_at"])[:19],
        )
    console.print(table)
    await pg.close()


@files_app.command("search")
def files_search(
    query: str = typer.Argument(..., help="Semantic search query"),
    limit: int = typer.Option(5, "--limit", "-n"),
):
    """🔍 Semantic search across indexed files."""
    asyncio.run(_files_search(query, limit))


async def _files_search(query: str, limit: int):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import EmbeddingEngine
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    engine = EmbeddingEngine(pg)
    results = await engine.semantic_search(query, source_type="user_file", limit=limit)
    if not results:
        console.print("[dim]No results. Have you indexed any files? Run [bold]octane files add <path>[/bold].[/]")
        await pg.close()
        return

    for i, r in enumerate(results, 1):
        dist = r.get("distance", 0)
        console.print(f"\n[bold cyan]#{i}[/] [dim](distance={dist:.3f})[/]")
        console.print(r.get("chunk_text", "")[:300])
    await pg.close()


@files_app.command("stats")
def files_stats(
    project: str = typer.Option("", "--project", "-p"),
):
    """📊 Show indexing statistics."""
    asyncio.run(_files_stats(project))


async def _files_stats(project: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import FileIndexer, ProjectStore
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    project_id = None
    if project:
        ps = ProjectStore(pg)
        row = await ps.get(project)
        project_id = row["id"] if row else None

    indexer = FileIndexer(pg, project_id=project_id)
    stats = await indexer.stats(project_id=project_id)
    console.print(f"[bold]Total files:[/] {stats['total_files']}")
    console.print(f"[bold]Total words:[/] {stats['total_words']:,}")
    if stats["by_extension"]:
        table = Table(title="By Extension")
        table.add_column("Extension")
        table.add_column("Files", justify="right")
        table.add_column("Words", justify="right")
        for row in stats["by_extension"]:
            table.add_row(row["extension"], str(row["n"]), str(int(row.get("words") or 0)))
        console.print(table)
    await pg.close()


@files_app.command("reindex")
def files_reindex(
    path: str = typer.Argument(..., help="File to force-reindex"),
):
    """🔄 Force re-index a file (ignores cached hash)."""
    asyncio.run(_files_reindex(path))


async def _files_reindex(path: str):
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import FileIndexer
    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return
    indexer = FileIndexer(pg)
    row = await indexer.reindex(path)
    if row:
        console.print(f"[green]✅ Re-indexed:[/] {row.get('filename')} ({row.get('word_count',0)} words)")
    else:
        console.print(f"[yellow]Reindex failed for: {path}[/]")
    await pg.close()
