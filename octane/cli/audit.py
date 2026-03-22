"""octane audit command — show provenance chain for stored findings."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console


def register(app: typer.Typer) -> None:
    app.command("audit")(_audit_cmd)


def _audit_cmd(
    finding_id: str = typer.Argument(..., help="Finding ID or web page ID to inspect"),
    table_hint: str = typer.Option(
        "auto", "--table", "-t",
        help="Table to look in: auto|web_pages|research_findings_v2|generated_artifacts",
    ),
):
    """🔍 Show the complete provenance chain for a stored finding."""
    asyncio.run(_audit(finding_id, table_hint))


async def _audit(finding_id: str, table_hint: str) -> None:
    from octane.tools.pg_client import PgClient
    from octane.security.provenance import ProvenanceRecord

    # Resolve ID — strip common prefixes like "wp-", "rf-", "art-"
    clean_id = finding_id.lstrip("wpartf-").strip()
    try:
        row_id = int(clean_id)
    except ValueError:
        console.print(f"[red]Invalid ID:[/] {finding_id!r} — must be numeric.")
        raise typer.Exit(1)

    pg = PgClient()
    await pg.connect()

    tables_to_check: list[str]
    if table_hint == "auto":
        tables_to_check = ["web_pages", "research_findings_v2", "generated_artifacts"]
    else:
        tables_to_check = [table_hint]

    found = False
    try:
        for tbl in tables_to_check:
            # Try to fetch the row — check if provenance column exists
            try:
                rows = await pg.pool.fetch(
                    f"SELECT id, provenance, created_at FROM {tbl} WHERE id = $1", row_id
                )
            except Exception:
                # Table may not have provenance column yet — skip
                continue

            if not rows:
                continue

            row = rows[0]
            found = True
            prov_raw = row.get("provenance") or {}
            created_at = row.get("created_at", "unknown")

            if not prov_raw:
                console.print(Panel(
                    f"Row [bold]{row_id}[/] found in [bold]{tbl}[/]\n\n"
                    f"[yellow]No provenance data recorded.[/]\n"
                    f"[dim](This row predates Session 28 provenance tracking.)[/]\n\n"
                    f"Created at: {created_at}",
                    title=f"Audit — {tbl} / ID {row_id}",
                    border_style="yellow",
                ))
                break

            record = ProvenanceRecord.from_dict(prov_raw)
            console.print(Panel(
                f"[green]✓ Provenance found[/]\n\n"
                f"{record.format()}\n\n"
                f"[dim]Stored at: {created_at}[/]",
                title=f"Audit — {tbl} / ID {row_id}",
                border_style="cyan",
            ))
            break

        if not found:
            console.print(
                f"[red]No record found[/] with ID {row_id} in "
                f"{', '.join(tables_to_check)}."
            )
    finally:
        await pg.close()
