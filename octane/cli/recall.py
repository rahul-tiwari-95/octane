"""octane recall — unified knowledge search across ALL stored data.

Searches extracted documents, web pages, research findings, user files,
and generated artifacts in one command.

Usage:
    octane recall "attention mechanisms"
    octane recall "NVDA earnings" --type youtube
    octane recall "transformer architecture" --limit 20
"""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from octane.cli._shared import console

recall_app = typer.Typer(
    name="recall",
    help="🧠 Unified knowledge search across all stored data.",
    no_args_is_help=True,
)


@recall_app.command("search")
def recall_search(
    query: str = typer.Argument(..., help="Search query."),
    source_type: str | None = typer.Option(
        None, "--type", "-t",
        help="Filter by source: youtube, arxiv, pdf, epub, web, finding, file, artifact.",
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results per category."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show content preview."),
):
    """🔍 Search ALL accumulated knowledge — extractions, pages, findings, files.

    Examples::

        octane recall search "attention mechanisms"
        octane recall search "NVDA" --type web
        octane recall search "transformer" -v --limit 20
    """
    asyncio.run(_recall_search(query, source_type, limit, verbose))


@recall_app.command("stats")
def recall_stats():
    """📊 Show knowledge base summary — counts across all data sources."""
    asyncio.run(_recall_stats())


async def _recall_search(query: str, source_type: str | None, limit: int, verbose: bool):
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    console.print()
    console.print(Rule(f"🧠 Recall: '{query}'", style="bold blue"))
    console.print()

    total_hits = 0

    # ── 1. Extracted documents ───────────────────────────────────────
    if source_type is None or source_type in ("youtube", "arxiv", "pdf", "epub", "web"):
        type_filter = ""
        params: list = [f"%{query}%", limit]
        if source_type:
            type_filter = "AND source_type = $3"
            params.append(source_type)

        rows = await pg.fetch(
            f"""
            SELECT id, source_type, source_url, title, author,
                   total_words, reliability_score, extracted_at,
                   LEFT(raw_text, 300) AS preview
            FROM extracted_documents
            WHERE (raw_text ILIKE $1 OR title ILIKE $1 OR author ILIKE $1)
                  {type_filter}
            ORDER BY extracted_at DESC LIMIT $2
            """,
            *params,
        )
        if rows:
            total_hits += len(rows)
            console.print(f"[bold cyan]📄 Extracted Documents ({len(rows)})[/]")
            for r in rows:
                ts = str(r["extracted_at"])[:19]
                console.print(
                    f"  [{r['source_type']}] "
                    f"[bold]{r['title'][:70] or '(untitled)'}[/]  "
                    f"[dim]{r['total_words']:,}w  {ts}[/]"
                )
                if verbose and r.get("preview"):
                    console.print(f"    [dim]{str(r['preview']).replace(chr(10), ' ')[:200]}[/]")
            console.print()

    # ── 2. Web pages ─────────────────────────────────────────────────
    if source_type is None or source_type == "web":
        rows = await pg.fetch(
            """
            SELECT id, url, title, word_count, fetched_at,
                   LEFT(content, 300) AS preview
            FROM web_pages
            WHERE content ILIKE $1 OR url ILIKE $1 OR title ILIKE $1
            ORDER BY fetched_at DESC LIMIT $2
            """,
            f"%{query}%", limit,
        )
        if rows:
            total_hits += len(rows)
            console.print(f"[bold cyan]🌐 Web Pages ({len(rows)})[/]")
            for r in rows:
                ts = str(r["fetched_at"])[:19]
                console.print(
                    f"  [bold]{r['title'][:70] or '(untitled)'}[/]  "
                    f"[dim]{r['word_count']:,}w  {ts}[/]"
                )
                if verbose and r.get("preview"):
                    console.print(f"    [dim]{str(r['preview']).replace(chr(10), ' ')[:200]}[/]")
            console.print()

    # ── 3. Research findings ─────────────────────────────────────────
    if source_type is None or source_type == "finding":
        rows = await pg.fetch(
            """
            SELECT id, task_id, cycle_num, topic, word_count, created_at,
                   LEFT(content, 300) AS preview
            FROM research_findings
            WHERE content ILIKE $1 OR topic ILIKE $1
            ORDER BY created_at DESC LIMIT $2
            """,
            f"%{query}%", limit,
        )
        if rows:
            total_hits += len(rows)
            console.print(f"[bold cyan]🔬 Research Findings ({len(rows)})[/]")
            for r in rows:
                ts = str(r["created_at"])[:19]
                console.print(
                    f"  [bold]{r['topic'][:70]}[/]  "
                    f"[dim]cycle {r['cycle_num']}  {r['word_count']:,}w  {ts}[/]"
                )
                if verbose and r.get("preview"):
                    console.print(f"    [dim]{str(r['preview']).replace(chr(10), ' ')[:200]}[/]")
            console.print()

    # ── 4. User files ────────────────────────────────────────────────
    if source_type is None or source_type == "file":
        rows = await pg.fetch(
            """
            SELECT id, filename, extension, word_count, indexed_at,
                   LEFT(content, 300) AS preview
            FROM user_files
            WHERE content ILIKE $1 OR filename ILIKE $1
            ORDER BY indexed_at DESC LIMIT $2
            """,
            f"%{query}%", limit,
        )
        if rows:
            total_hits += len(rows)
            console.print(f"[bold cyan]📁 Indexed Files ({len(rows)})[/]")
            for r in rows:
                ts = str(r["indexed_at"])[:19]
                console.print(
                    f"  [bold]{r['filename']}[/] ({r['extension']})  "
                    f"[dim]{r['word_count']:,}w  {ts}[/]"
                )
            console.print()

    # ── 5. Generated artifacts ───────────────────────────────────────
    if source_type is None or source_type == "artifact":
        rows = await pg.fetch(
            """
            SELECT id, artifact_type, filename, description, created_at,
                   LEFT(content, 300) AS preview
            FROM generated_artifacts
            WHERE content ILIKE $1 OR filename ILIKE $1 OR description ILIKE $1
            ORDER BY created_at DESC LIMIT $2
            """,
            f"%{query}%", limit,
        )
        if rows:
            total_hits += len(rows)
            console.print(f"[bold cyan]🧪 Generated Artifacts ({len(rows)})[/]")
            for r in rows:
                ts = str(r["created_at"])[:19]
                label = r['filename'] or r['description'][:70] or '(untitled)'
                console.print(
                    f"  [{r['artifact_type']}] [bold]{label}[/]  "
                    f"[dim]{ts}[/]"
                )
            console.print()

    await pg.close()

    if total_hits == 0:
        console.print(f"[yellow]No results for '{query}'.[/]")
        console.print("[dim]Populate the knowledge base with:[/]")
        console.print("  [dim]octane extract run <url>     — extract a web page / video / paper[/]")
        console.print("  [dim]octane files add <path>      — index local files[/]")
        console.print("  [dim]octane ask \"<question>\"      — web research auto-stores pages[/]")
    else:
        console.print(f"[dim]Total: {total_hits} results across all sources.[/]")


async def _recall_stats():
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    tbl = Table(title="🧠 Knowledge Base", show_lines=False, border_style="blue")
    tbl.add_column("Source", style="cyan")
    tbl.add_column("Count", justify="right", style="yellow")
    tbl.add_column("Total Words", justify="right", style="dim")

    sources = [
        ("Extracted Documents", "extracted_documents"),
        ("Web Pages", "web_pages"),
        ("Research Findings", "research_findings"),
        ("Indexed Files", "user_files"),
        ("Generated Artifacts", "generated_artifacts"),
        ("Embeddings", "embeddings"),
    ]

    grand_total_rows = 0
    grand_total_words = 0

    for label, table_name in sources:
        count = int(await pg.fetchval(f"SELECT COUNT(*) FROM {table_name}") or 0)
        words = 0
        if table_name == "extracted_documents":
            words = int(await pg.fetchval("SELECT COALESCE(SUM(total_words), 0) FROM extracted_documents") or 0)
        elif table_name not in ("embeddings", "generated_artifacts"):
            words = int(await pg.fetchval(f"SELECT COALESCE(SUM(word_count), 0) FROM {table_name}") or 0)
        grand_total_rows += count
        grand_total_words += words
        tbl.add_row(label, f"{count:,}", f"{words:,}" if words else "-")

    console.print(tbl)
    console.print(f"\n  [bold]Total items:[/] {grand_total_rows:,}")
    console.print(f"  [bold]Total words:[/] {grand_total_words:,}")
    await pg.close()
