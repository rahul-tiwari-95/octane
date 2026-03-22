"""octane store — browse Postgres tables and Redis data."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from octane.cli._shared import console

store_app = typer.Typer(
    name="store",
    help="Browse and search all data stored in Postgres and Redis.",
    no_args_is_help=True,
)


# ── stats ─────────────────────────────────────────────────────────────────────

@store_app.command("stats")
def store_stats():
    """Row counts for every Postgres table + Redis namespace counts and memory."""
    asyncio.run(_store_stats())


async def _store_stats():
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()

    # ── Postgres ─────────────────────────────────────────────────────────────
    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
    else:
        tables = await pg.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        )
        tbl = Table(title="Postgres Tables", show_lines=False, border_style="cyan")
        tbl.add_column("Table",     style="cyan")
        tbl.add_column("Rows",      justify="right", style="yellow")
        tbl.add_column("Size",      justify="right", style="dim")

        total_rows = 0
        for row in tables:
            name = row["tablename"]
            try:
                count = int(await pg.fetchval(f"SELECT COUNT(*) FROM {name}") or 0)
                size  = await pg.fetchval(
                    "SELECT pg_size_pretty(pg_total_relation_size($1))", name
                )
                total_rows += count
                tbl.add_row(name, f"{count:,}", str(size or "?"))
            except Exception:
                tbl.add_row(name, "?", "?")

        console.print(tbl)
        console.print(f"  [dim]Total rows: {total_rows:,}[/]\n")
        await pg.close()

    # ── Redis ────────────────────────────────────────────────────────────────
    try:
        from octane.config import settings
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        mem_info = await r.info("memory")
        used_mb  = int(mem_info.get("used_memory", 0)) / 1024 / 1024

        namespaces: dict[str, int] = {}
        async for key in r.scan_iter("*"):
            ns = key.split(":")[0]
            namespaces[ns] = namespaces.get(ns, 0) + 1

        total_keys = sum(namespaces.values())
        rtbl = Table(
            title=f"Redis Namespaces  ({used_mb:.1f} MB  ·  {total_keys} keys)",
            show_lines=False,
            border_style="magenta",
        )
        rtbl.add_column("Namespace", style="magenta")
        rtbl.add_column("Keys", justify="right", style="yellow")

        for ns, count in sorted(namespaces.items(), key=lambda x: -x[1]):
            rtbl.add_row(f"{ns}:*", str(count))

        console.print(rtbl)
        await r.aclose()
    except Exception as exc:
        console.print(f"  [dim]Redis: {exc}[/]")


# ── pages ─────────────────────────────────────────────────────────────────────

@store_app.command("pages")
def store_pages(
    query: str = typer.Argument(None, help="Search term (matches url, title, content)"),
    limit: int  = typer.Option(20, "--limit", "-n", help="Max results (default: 20)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show content preview"),
):
    """Browse web pages fetched and stored during all 'octane ask' runs.

    Pages are stored in Postgres automatically every time the web agent
    extracts content.  Search across URL, title, and full content.

    Examples::

        octane store pages
        octane store pages "NVDA"
        octane store pages "Iran war" --limit 50
        octane store pages "interest rates" -v
    """
    asyncio.run(_store_pages(query, limit, verbose))


async def _store_pages(query: str | None, limit: int, verbose: bool):
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()

    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    if query:
        rows = await pg.fetch(
            """
            SELECT id, url, title, word_count, fetch_status, fetched_at,
                   LEFT(content, 500) AS preview
            FROM   web_pages
            WHERE  content ILIKE $1 OR url ILIKE $1 OR title ILIKE $1
            ORDER  BY fetched_at DESC
            LIMIT  $2
            """,
            f"%{query}%", limit,
        )
        total = await pg.fetchval(
            "SELECT COUNT(*) FROM web_pages WHERE content ILIKE $1 OR url ILIKE $1 OR title ILIKE $1",
            f"%{query}%",
        )
    else:
        rows = await pg.fetch(
            """
            SELECT id, url, title, word_count, fetch_status, fetched_at,
                   LEFT(content, 500) AS preview
            FROM   web_pages
            ORDER  BY fetched_at DESC
            LIMIT  $1
            """,
            limit,
        )
        total = await pg.fetchval("SELECT COUNT(*) FROM web_pages")

    await pg.close()

    if not rows:
        msg = f"No pages matching '{query}'" if query else "No pages stored yet."
        console.print(f"[yellow]{msg}[/]")
        console.print("[dim]Pages are stored automatically when 'octane ask' runs a web query.[/]")
        return

    console.print()
    heading = f"Web Pages in Postgres"
    if query:
        heading += f"  —  '{query}'"
    heading += f"  ({len(rows)} of {total or len(rows)} results)"
    console.print(Rule(heading, style="cyan"))
    console.print()

    for r in rows:
        ts     = str(r["fetched_at"])[:19] + " UTC"
        title  = r["title"] or "(no title)"
        status = r["fetch_status"] or "ok"
        status_str = "[green]ok[/]" if status == "ok" else f"[yellow]{status}[/]"

        console.print(f"  [cyan]{r['url'][:110]}[/]")
        console.print(f"  [dim]  title:   {title[:80]}[/]")
        console.print(f"  [dim]  fetched: {ts}   words: {r['word_count']}   status: [/]{status_str}")

        if verbose and r["preview"]:
            preview = str(r["preview"]).replace("\n", " ")
            console.print(f"  [dim]  preview: {preview[:400]}[/]")

        console.print()

    console.print(
        "  [dim]octane store pages <query>           — keyword search\n"
        "  octane store pages <query> -v         — with content preview\n"
        "  octane store pages <query> --limit 50 — show more[/]"
    )


# ── findings ──────────────────────────────────────────────────────────────────

@store_app.command("findings")
def store_findings(
    query: str = typer.Argument(None, help="Search term (matches topic + content)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results (default: 20)"),
):
    """Search research findings stored by background research tasks.

    Examples::

        octane store findings
        octane store findings "NVDA earnings"
        octane store findings "Fed rate" --limit 50
    """
    asyncio.run(_store_findings(query, limit))


async def _store_findings(query: str | None, limit: int):
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()

    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    base_select = """
        SELECT id, task_id, cycle_num, topic, word_count, created_at,
               LEFT(content, 500) AS preview
        FROM   research_findings
    """
    if query:
        rows = await pg.fetch(
            base_select + " WHERE content ILIKE $1 OR topic ILIKE $1 ORDER BY created_at DESC LIMIT $2",
            f"%{query}%", limit,
        )
        total = await pg.fetchval(
            "SELECT COUNT(*) FROM research_findings WHERE content ILIKE $1 OR topic ILIKE $1",
            f"%{query}%",
        )
    else:
        rows = await pg.fetch(base_select + " ORDER BY created_at DESC LIMIT $1", limit)
        total = await pg.fetchval("SELECT COUNT(*) FROM research_findings")

    await pg.close()

    if not rows:
        msg = f"No findings matching '{query}'" if query else "No research findings stored yet."
        console.print(f"[yellow]{msg}[/]")
        console.print("[dim]Start a background task with: octane research start \"<topic>\"[/]")
        return

    console.print()
    heading = "Research Findings"
    if query:
        heading += f"  —  '{query}'"
    heading += f"  ({len(rows)} of {total or len(rows)} results)"
    console.print(Rule(heading, style="magenta"))
    console.print()

    for r in rows:
        ts      = str(r["created_at"])[:19] + " UTC"
        preview = str(r["preview"]).replace("\n", " ")
        console.print(Panel(
            preview + ("…" if r["word_count"] > 90 else ""),
            title=f"[dim]{r['topic'][:60]}  ·  cycle {r['cycle_num']}  ·  {ts}  ·  {r['word_count']}w  ·  task: {r['task_id']}[/]",
            border_style="dim",
        ))


# ── artifacts ────────────────────────────────────────────────────────────────

@store_app.command("artifacts")
def store_artifacts(
    query: str = typer.Argument(None, help="Search term (matches title + content)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results (default: 20)"),
):
    """Search generated artifacts stored in Postgres.

    Examples::

        octane store artifacts
        octane store artifacts "portfolio"
    """
    asyncio.run(_store_artifacts(query, limit))


async def _store_artifacts(query: str | None, limit: int):
    from octane.tools.pg_client import PgClient

    pg = PgClient()
    await pg.connect()

    if not pg.available:
        console.print("[red]Postgres unavailable.[/]")
        return

    base_select = """
        SELECT id, artifact_type, title, word_count, created_at,
               LEFT(content, 500) AS preview
        FROM   generated_artifacts
    """
    if query:
        rows = await pg.fetch(
            base_select + " WHERE content ILIKE $1 OR title ILIKE $1 ORDER BY created_at DESC LIMIT $2",
            f"%{query}%", limit,
        )
    else:
        rows = await pg.fetch(base_select + " ORDER BY created_at DESC LIMIT $1", limit)
        total = await pg.fetchval("SELECT COUNT(*) FROM generated_artifacts")

    await pg.close()

    if not rows:
        msg = f"No artifacts matching '{query}'" if query else "No artifacts stored yet."
        console.print(f"[yellow]{msg}[/]")
        return

    console.print()
    heading = "Generated Artifacts"
    if query:
        heading += f"  —  '{query}'"
    console.print(Rule(heading, style="yellow"))
    console.print()

    for r in rows:
        ts      = str(r["created_at"])[:19] + " UTC"
        preview = str(r["preview"]).replace("\n", " ")
        console.print(Panel(
            preview + ("…" if r["word_count"] > 90 else ""),
            title=f"[dim]{r['artifact_type']}  ·  {r['title'][:60]}  ·  {ts}  ·  {r['word_count']}w[/]",
            border_style="dim",
        ))


# ── redis ─────────────────────────────────────────────────────────────────────

@store_app.command("redis")
def store_redis():
    """Show all Redis key namespaces with key counts and sample keys."""
    asyncio.run(_store_redis())


async def _store_redis():
    try:
        from octane.config import settings
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        mem_info = await r.info("memory")
        used_mb  = int(mem_info.get("used_memory", 0)) / 1024 / 1024

        namespaces: dict[str, list[str]] = {}
        async for key in r.scan_iter("*"):
            ns = key.split(":")[0]
            if ns not in namespaces:
                namespaces[ns] = []
            namespaces[ns].append(key)

        if not namespaces:
            console.print("[dim]Redis is empty.[/]")
            await r.aclose()
            return

        total_keys = sum(len(v) for v in namespaces.values())
        tbl = Table(
            title=f"Redis  ({used_mb:.1f} MB  ·  {total_keys} keys)",
            show_lines=True,
            border_style="magenta",
        )
        tbl.add_column("Namespace", style="magenta")
        tbl.add_column("Keys", justify="right", style="yellow")
        tbl.add_column("Sample", style="dim")

        for ns, keys in sorted(namespaces.items(), key=lambda x: -len(x[1])):
            sample = "  ".join(sorted(keys)[:3])
            if len(keys) > 3:
                sample += f"  ... +{len(keys) - 3} more"
            tbl.add_row(f"{ns}:*", str(len(keys)), sample[:90])

        console.print(tbl)
        await r.aclose()

    except Exception as exc:
        console.print(f"[red]Redis error: {exc}[/]")
