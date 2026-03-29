"""octane stats — personal analytics dashboard.

Aggregates data across all stored sources: extractions, web pages,
research findings, indexed files, embeddings, traces.

Usage:
    octane stats
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console


def register(app: typer.Typer) -> None:
    app.command("stats")(stats)


def stats():
    """📊 Personal analytics dashboard — aggregated knowledge base metrics."""
    asyncio.run(_stats())


async def _stats():
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()

    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    # ── Knowledge sources ────────────────────────────────────────────────
    tbl = Table(
        title="📊 Octane Knowledge Base",
        show_lines=False,
        border_style="cyan",
    )
    tbl.add_column("Source", style="cyan")
    tbl.add_column("Count", justify="right", style="bold yellow")
    tbl.add_column("Total Words", justify="right", style="dim")

    sources = [
        ("Extracted Documents", "extracted_documents", "total_words"),
        ("Web Pages", "web_pages", "word_count"),
        ("Research Findings", "research_findings", "word_count"),
        ("Indexed Files", "user_files", "word_count"),
        ("Generated Artifacts", "generated_artifacts", None),
        ("Embeddings (vectors)", "embeddings", None),
    ]

    grand_docs = 0
    grand_words = 0

    for label, table_name, word_col in sources:
        count = int(await pg.fetchval(f"SELECT COUNT(*) FROM {table_name}") or 0)  # noqa: S608
        words = 0
        if word_col:
            words = int(
                await pg.fetchval(f"SELECT COALESCE(SUM({word_col}), 0) FROM {table_name}") or 0  # noqa: S608
            )
        grand_docs += count
        grand_words += words
        tbl.add_row(label, f"{count:,}", f"{words:,}" if words else "-")

    console.print(tbl)
    console.print(f"\n  [bold]Total items:[/] {grand_docs:,}")
    console.print(f"  [bold]Total words:[/] {grand_words:,}")

    # ── Extraction breakdown by source type ──────────────────────────────
    rows = await pg.fetch(
        """
        SELECT source_type,
               COUNT(*)                       AS doc_count,
               COALESCE(SUM(total_words), 0)  AS total_words,
               COALESCE(SUM(total_chunks), 0) AS total_chunks
        FROM extracted_documents
        GROUP BY source_type
        ORDER BY doc_count DESC
        """
    )
    if rows:
        console.print()
        etbl = Table(
            title="📄 Extractions by Source Type",
            show_lines=False,
            border_style="green",
        )
        etbl.add_column("Type", style="green")
        etbl.add_column("Docs", justify="right", style="yellow")
        etbl.add_column("Words", justify="right", style="dim")
        etbl.add_column("Chunks", justify="right", style="dim")
        for r in rows:
            etbl.add_row(
                r["source_type"],
                str(r["doc_count"]),
                f"{r['total_words']:,}",
                str(r["total_chunks"]),
            )
        console.print(etbl)

    # ── Trace stats ──────────────────────────────────────────────────────
    traces_dir = Path.home() / ".octane" / "traces"
    if traces_dir.exists():
        trace_files = list(traces_dir.glob("*.jsonl"))
        console.print(
            f"\n  [bold]Traces:[/] {len(trace_files)} query executions recorded"
        )

    # ── Local extraction mirror ──────────────────────────────────────────
    extractions_dir = Path.home() / ".octane" / "extractions"
    if extractions_dir.exists():
        mirror_files = list(extractions_dir.glob("*.md"))
        total_size = sum(f.stat().st_size for f in mirror_files) / 1024 / 1024
        console.print(
            f"  [bold]Local mirror:[/] {len(mirror_files)} files  "
            f"({total_size:.1f} MB in ~/.octane/extractions/)"
        )

    await pg.close()
