"""octane search — Composable search commands for piping.

Usage:
    octane search web "NVDA earnings" --json --limit 10
    octane search news "AI chip export" --json --limit 5
    octane search youtube "transformer explained" --json --limit 5
    octane search arxiv "attention mechanism" --json --limit 10

Pipe examples:
    octane search web "NVDA" --json | octane extract --stdin --json
    octane search arxiv "RAG" --urls-only | octane extract --stdin --json
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer

from octane.cli._shared import console

search_app = typer.Typer(
    help="🔍 Search the web, news, YouTube, and arXiv.",
    no_args_is_help=True,
)


# ── Web Search ────────────────────────────────────────────────────────────────

@search_app.command("web")
def search_web(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results."),
    output_json: bool = typer.Option(False, "--json", help="Output structured JSON to stdout."),
    urls_only: bool = typer.Option(False, "--urls-only", help="Print only URLs (one per line)."),
):
    """Search the web via Brave/Beru API."""
    asyncio.run(_search_web(query, limit, output_json, urls_only))


async def _search_web(query: str, limit: int, output_json: bool, urls_only: bool):
    from octane.tools.bodega_intel import BodegaIntelClient

    intel = BodegaIntelClient()
    raw = await intel.web_search(query, count=limit)

    results = []
    for r in raw.get("web", {}).get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("description", ""),
            "age": r.get("age", ""),
        })

    if urls_only:
        for r in results:
            if r["url"]:
                print(r["url"])
        return

    if output_json:
        print(json.dumps({"query": query, "source": "web", "results": results}, indent=2))
        return

    # Rich table
    from rich.table import Table

    table = Table(title=f"Web: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Title", max_width=50)
    table.add_column("URL", max_width=50)
    table.add_column("Age", width=8)

    for i, r in enumerate(results[:limit], 1):
        table.add_row(str(i), r["title"][:50], r["url"][:50], r.get("age", ""))
    console.print(table)


# ── News Search ───────────────────────────────────────────────────────────────

@search_app.command("news")
def search_news(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results."),
    period: str = typer.Option("3d", "--period", help="Time period: 1d, 3d, 7d."),
    output_json: bool = typer.Option(False, "--json", help="Output structured JSON to stdout."),
    urls_only: bool = typer.Option(False, "--urls-only", help="Print only URLs (one per line)."),
):
    """Search news via Brave/Beru API."""
    asyncio.run(_search_news(query, limit, period, output_json, urls_only))


async def _search_news(query: str, limit: int, period: str, output_json: bool, urls_only: bool):
    from octane.tools.bodega_intel import BodegaIntelClient

    intel = BodegaIntelClient()
    raw = await intel.news_search(query, period=period, max_results=limit)

    results = []
    for r in raw.get("articles", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source": r.get("source", ""),
            "age": r.get("age", ""),
            "description": r.get("description", ""),
        })

    if urls_only:
        for r in results:
            if r["url"]:
                print(r["url"])
        return

    if output_json:
        print(json.dumps({"query": query, "source": "news", "results": results}, indent=2))
        return

    from rich.table import Table

    table = Table(title=f"News: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Title", max_width=45)
    table.add_column("Source", max_width=15)
    table.add_column("URL", max_width=45)
    table.add_column("Age", width=8)

    for i, r in enumerate(results[:limit], 1):
        table.add_row(str(i), r["title"][:45], r.get("source", "")[:15], r["url"][:45], r.get("age", ""))
    console.print(table)


# ── YouTube Search ────────────────────────────────────────────────────────────

@search_app.command("youtube")
def search_youtube(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results."),
    output_json: bool = typer.Option(False, "--json", help="Output structured JSON to stdout."),
    urls_only: bool = typer.Option(False, "--urls-only", help="Print only URLs (one per line)."),
):
    """Search YouTube for videos."""
    asyncio.run(_search_youtube(query, limit, output_json, urls_only))


async def _search_youtube(query: str, limit: int, output_json: bool, urls_only: bool):
    from octane.extractors.youtube.search import search_youtube as yt_search

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: yt_search(query, limit=limit))

    items = []
    for r in results:
        items.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "channel": r.get("channel", ""),
            "duration": r.get("duration", ""),
            "views": r.get("views", ""),
        })

    if urls_only:
        for r in items:
            if r["url"]:
                print(r["url"])
        return

    if output_json:
        print(json.dumps({"query": query, "source": "youtube", "results": items}, indent=2))
        return

    from rich.table import Table

    table = Table(title=f"YouTube: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Title", max_width=50)
    table.add_column("Channel", max_width=20)
    table.add_column("Duration")
    table.add_column("URL", max_width=45)

    for i, r in enumerate(items[:limit], 1):
        table.add_row(str(i), r["title"][:50], r["channel"][:20], r["duration"], r["url"])
    console.print(table)


# ── arXiv Search ──────────────────────────────────────────────────────────────

@search_app.command("arxiv")
def search_arxiv(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results."),
    output_json: bool = typer.Option(False, "--json", help="Output structured JSON to stdout."),
    urls_only: bool = typer.Option(False, "--urls-only", help="Print only arXiv IDs (one per line)."),
):
    """Search arXiv for academic papers."""
    asyncio.run(_search_arxiv(query, limit, output_json, urls_only))


async def _search_arxiv(query: str, limit: int, output_json: bool, urls_only: bool):
    from octane.extractors.academic.arxiv_search import search_arxiv as arxiv_search

    try:
        results = arxiv_search(query, max_results=limit)
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "503" in err_msg:
            print(json.dumps({"error": "arXiv rate-limited", "query": query}), file=sys.stderr)
        else:
            print(json.dumps({"error": str(exc), "query": query}), file=sys.stderr)
        raise typer.Exit(1)

    items = []
    for r in results:
        authors = r.get("authors", [])
        items.append({
            "arxiv_id": r.get("arxiv_id", ""),
            "title": r.get("title", ""),
            "authors": authors[:5],
            "published": r.get("published", ""),
            "summary": r.get("summary", "")[:300],
            "url": r.get("url", f"https://arxiv.org/abs/{r.get('arxiv_id', '')}"),
        })

    if urls_only:
        for r in items:
            print(r["arxiv_id"])
        return

    if output_json:
        print(json.dumps({"query": query, "source": "arxiv", "results": items}, indent=2))
        return

    from rich.table import Table

    table = Table(title=f"arXiv: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("ID", max_width=15)
    table.add_column("Title", max_width=45)
    table.add_column("Authors", max_width=25)
    table.add_column("Date", max_width=12)

    for i, r in enumerate(items[:limit], 1):
        authors_str = ", ".join(r["authors"][:3])
        if len(r["authors"]) > 3:
            authors_str += " et al."
        date = r["published"][:10] if r["published"] else ""
        table.add_row(str(i), r["arxiv_id"], r["title"][:45], authors_str[:25], date)
    console.print(table)
