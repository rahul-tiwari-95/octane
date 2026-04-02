"""octane extract — Content extraction from YouTube, arXiv, PDF, EPUB.

Usage:
    octane extract run "https://youtube.com/watch?v=..."
    octane extract run "2408.09869"                        # arXiv ID
    octane extract run ./path/to/paper.pdf --quality deep
    octane extract run ./path/to/book.epub
    octane extract search-youtube "attention mechanisms"
    octane extract search-arxiv "retrieval augmented generation"

Pipe usage:
    octane search web "NVDA" --json | octane extract stdin --json
    octane search arxiv "RAG" --urls-only | octane extract stdin --json
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer

from octane.cli._shared import console

extract_app = typer.Typer(help="📄 Extract content from YouTube, arXiv, PDF, EPUB.", no_args_is_help=True)


@extract_app.command("run")
def extract_run(
    source: str = typer.Argument(..., help="URL, arXiv ID, or file path to extract."),
    quality: str = typer.Option("auto", help="Extraction quality: fast, deep, or auto."),
    source_type: str | None = typer.Option(None, "--type", help="Override source type: youtube, arxiv, pdf, epub."),
    show_chunks: bool = typer.Option(False, "--chunks", help="Display individual chunks."),
    output: str | None = typer.Option(None, "--output", "-o", help="Save full text to file (default: ~/.octane/extracts/<title>.md)."),
    open_folder: bool = typer.Option(False, "--open", help="Open the output folder in Finder after saving."),
):
    """Extract content from a source and display results."""
    asyncio.run(_extract_run(source, quality, source_type, show_chunks, output, open_folder))


@extract_app.command("stdin")
def extract_stdin_cmd(
    output_json: bool = typer.Option(False, "--json", help="Output structured JSON to stdout."),
    quality: str = typer.Option("auto", help="Extraction quality: fast, deep, or auto."),
    top_n: int = typer.Option(10, "--top-n", help="Max pages to extract."),
):
    """Read URLs from stdin and extract content.

    Accepts two input formats:
      1. Plain text — one URL per line
      2. JSON — from ``octane search --json`` (reads .results[].url)

    Examples::

        octane search web "NVDA" --json | octane extract stdin --json
        octane search arxiv "RAG" --urls-only | octane extract stdin --json
        echo "https://example.com" | octane extract stdin
    """
    asyncio.run(_extract_stdin(output_json, quality, top_n))


async def _extract_stdin(output_json: bool, quality: str, top_n: int):
    from octane.agents.web.content_extractor import ContentExtractor

    if sys.stdin.isatty():
        print("Error: no input on stdin. Pipe data in.", file=sys.stderr)
        raise typer.Exit(1)

    raw_input = sys.stdin.read().strip()
    if not raw_input:
        print("Error: empty stdin.", file=sys.stderr)
        raise typer.Exit(1)

    urls: list[str] = []
    # Try JSON first (from octane search --json)
    try:
        data = json.loads(raw_input)
        if isinstance(data, dict) and "results" in data:
            for r in data["results"]:
                u = r.get("url", "")
                if u:
                    urls.append(u)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    u = item.get("url", "")
                    if u:
                        urls.append(u)
    except (json.JSONDecodeError, TypeError):
        # Plain text — one URL/ID per line
        for line in raw_input.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        print("Error: no URLs found in stdin.", file=sys.stderr)
        raise typer.Exit(1)

    extractor = ContentExtractor()
    extracted = await extractor.extract_batch(urls[:top_n], top_n=top_n)
    usable = [
        a for a in extracted
        if a.text and a.method not in ("unavailable", "failed")
    ]

    if output_json:
        items = []
        for a in usable:
            items.append({
                "url": a.url,
                "title": a.title,
                "text": a.text,
                "word_count": a.word_count,
                "method": a.method,
            })
        print(json.dumps({"extracted": items, "total": len(items)}, indent=2))
    else:
        from rich.panel import Panel
        console.print(f"[bold]Extracted {len(usable)}/{len(urls)} pages[/]\n")
        for a in usable:
            preview = a.text[:300] + "..." if len(a.text) > 300 else a.text
            console.print(Panel(
                f"[dim]{a.url}[/]\n[bold]{a.title}[/]\n\n{preview}",
                border_style="green",
                title=f"{a.word_count:,} words · {a.method}",
            ))


@extract_app.command("youtube-login")
def youtube_login_cmd():
    """🔑 Log in to YouTube — saves cookies for authenticated transcript extraction.

    Opens a browser window. Log in to your Google account, then press Enter.
    Future transcript fetches will use your session — no more IP blocks.
    """
    asyncio.run(_youtube_login())


async def _youtube_login():
    from octane.extractors.youtube.transcript import youtube_login, _YOUTUBE_COOKIE_FILE

    if _YOUTUBE_COOKIE_FILE.exists():
        console.print("[yellow]⚠  Existing YouTube cookies found. This will overwrite them.[/]")

    success = await youtube_login()
    if not success:
        console.print("[red]❌  YouTube login failed. Is Playwright installed?[/]")
        console.print("[dim]Run: playwright install chromium[/]")
        raise typer.Exit(1)


async def _extract_run(source: str, quality: str, source_type_str: str | None, show_chunks: bool,
                       output: str | None = None, open_folder: bool = False):
    import re as _re
    import subprocess
    from pathlib import Path
    from rich.panel import Panel
    from octane.extractors.pipeline import extract, detect_source_type
    from octane.extractors.models import SourceType

    st = None
    if source_type_str:
        try:
            st = SourceType(source_type_str.lower())
        except ValueError:
            console.print(f"[red]Unknown source type: {source_type_str}[/]")
            raise typer.Exit(1)

    detected, identifier = detect_source_type(source)
    console.print(
        f"[dim]📄 Extracting from [bold]{(st or detected).value}[/bold]: {source}[/]"
    )

    # ── Dedup check: skip if content already stored ──────────────────────
    doc = await extract(source, quality=quality, source_type=st)

    # Summary panel
    summary = (
        f"[bold]Title:[/] {doc.title}\n"
        f"[bold]Author:[/] {doc.author}\n"
        f"[bold]Source:[/] {doc.source_type.value}\n"
        f"[bold]Method:[/] {doc.extraction_method}\n"
        f"[bold]Words:[/] {doc.total_words:,}\n"
        f"[bold]Chunks:[/] {doc.total_chunks}\n"
        f"[bold]Reliability:[/] {doc.reliability_score:.2f}\n"
        f"[bold]Hash:[/] {doc.content_hash}"
    )
    console.print(Panel(summary, title="📋 Extraction Result", border_style="green"))

    # Show first 500 chars of text
    if doc.raw_text:
        preview = doc.raw_text[:500]
        if len(doc.raw_text) > 500:
            preview += "..."
        console.print(Panel(preview, title="📝 Text Preview", border_style="dim"))

    # Metadata
    if doc.metadata:
        meta_lines = "\n".join(f"  {k}: {v}" for k, v in doc.metadata.items() if k != "summary")
        console.print(f"\n[dim]{meta_lines}[/]")

    # Show chunks if requested
    if show_chunks and doc.chunks:
        console.print(f"\n[bold]Chunks ({len(doc.chunks)}):[/]")
        for chunk in doc.chunks[:10]:
            ts = ""
            if chunk.metadata.timestamp_start is not None:
                ts = f" [{chunk.metadata.timestamp_start:.0f}s-{chunk.metadata.timestamp_end:.0f}s]"
            ch = ""
            if chunk.metadata.chapter:
                ch = f" [{chunk.metadata.chapter}]"
            console.print(f"  [dim]#{chunk.index}{ts}{ch}[/] {chunk.text[:100]}...")
        if len(doc.chunks) > 10:
            console.print(f"  [dim]... and {len(doc.chunks) - 10} more chunks[/]")

    # ── Auto-persist to Postgres ─────────────────────────────────────────
    await _persist_extraction(doc)

    # Save full text to file
    if output or open_folder:
        extract_dir = Path.home() / ".octane" / "extracts"
        extract_dir.mkdir(parents=True, exist_ok=True)

        if output:
            out_path = Path(output)
        else:
            # Sanitise title for filename
            safe = _re.sub(r'[^\w\s-]', '', doc.title or 'untitled')[:80].strip()
            safe = _re.sub(r'\s+', '_', safe)
            out_path = extract_dir / f"{safe}.md"

        # Build markdown content
        lines = [
            f"# {doc.title}",
            f"",
            f"**Author:** {doc.author}",
            f"**Source:** {doc.source_type.value}  ",
            f"**Method:** {doc.extraction_method}  ",
            f"**Words:** {doc.total_words:,}  ",
            f"**Reliability:** {doc.reliability_score:.2f}  ",
            f"",
            f"---",
            f"",
        ]
        if doc.raw_text:
            lines.append(doc.raw_text)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"\n[green]💾 Saved to:[/] {out_path}")

        if open_folder:
            subprocess.Popen(["open", "-R", str(out_path)])


# ── Persistence helpers ───────────────────────────────────────────────────────

async def _persist_extraction(doc) -> dict | None:
    """Save an ExtractedDocument to Postgres (dedup by content_hash)."""
    from octane.tools.pg_client import PgClient
    from octane.tools.structured_store import ExtractionStore

    pg = PgClient()
    await pg.connect()
    if not pg.available:
        console.print("[dim]  (Postgres unavailable — extraction not persisted)[/]")
        return None

    store = ExtractionStore(pg)

    # Dedup: skip if already stored
    if doc.content_hash and await store.seen(doc.content_hash):
        console.print("[dim]  ♻️  Already stored (dedup by content hash)[/]")
        await pg.close()
        return None

    chunks_dicts = [
        {
            "index": c.index,
            "text": c.text,
            "word_count": c.word_count,
            "metadata": {
                "page": c.metadata.page,
                "chapter": c.metadata.chapter,
                "timestamp_start": c.metadata.timestamp_start,
                "timestamp_end": c.metadata.timestamp_end,
            },
        }
        for c in (doc.chunks or [])
    ]

    row = await store.store(
        source_type=doc.source_type.value,
        source_url=doc.source_url,
        title=doc.title,
        author=doc.author,
        raw_text=doc.raw_text,
        chunks=chunks_dicts,
        extraction_method=doc.extraction_method,
        reliability_score=doc.reliability_score,
        metadata=doc.metadata,
        content_hash=doc.content_hash,
    )
    if row:
        console.print(f"[green]  💾 Persisted to Postgres (id={row['id']})[/]")
        local = row.get("local_path")
        if local:
            console.print(f"[dim]  📁 Local mirror: {local}[/]")
    await pg.close()
    return row


# ── Batch extraction ──────────────────────────────────────────────────────────

@extract_app.command("batch")
def extract_batch_cmd(
    file: str = typer.Argument(..., help="Path to a text file with one URL per line."),
    quality: str = typer.Option("auto", help="Extraction quality: fast, deep, or auto."),
):
    """📦 Batch extract from a file of URLs (one per line).

    Each URL is extracted sequentially, persisted to Postgres, and a progress
    summary is printed at the end.

    Examples::

        octane extract batch urls.txt
        octane extract batch ~/research_urls.txt --quality deep
    """
    asyncio.run(_extract_batch(file, quality))


async def _extract_batch(file_path: str, quality: str):
    from pathlib import Path
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    from octane.extractors.pipeline import extract

    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    urls = [
        line.strip()
        for line in p.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        console.print("[yellow]No URLs found in file.[/]")
        return

    console.print(f"[bold]📦 Batch extraction: {len(urls)} URLs from {p.name}[/]\n")

    succeeded = 0
    failed = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(urls))
        for url in urls:
            progress.update(task, description=f"[dim]{url[:60]}[/]")
            try:
                doc = await extract(url, quality=quality)
                if doc.raw_text and len(doc.raw_text.strip()) > 50:
                    row = await _persist_extraction(doc)
                    if row:
                        succeeded += 1
                    else:
                        skipped += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            progress.advance(task)

    console.print(
        f"\n[bold]Batch complete:[/] "
        f"[green]{succeeded} stored[/]  "
        f"[yellow]{skipped} skipped (dedup)[/]  "
        f"[red]{failed} failed[/]"
    )


# ── Search with --extract-all ────────────────────────────────────────────────

@extract_app.command("search-youtube")
def search_youtube_cmd(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, help="Max results."),
    extract_all: bool = typer.Option(False, "--extract-all", help="Extract transcripts for all results."),
    urls_only: bool = typer.Option(False, "--urls-only", help="Print only URLs (one per line, for piping)."),
):
    """Search YouTube and display results.

    Examples::

        octane extract search-youtube "attention mechanisms"
        octane extract search-youtube "transformers" --extract-all
        octane extract search-youtube "RAG" --urls-only | xargs -I{} octane extract run {}
    """
    asyncio.run(_search_youtube(query, limit, extract_all, urls_only))


async def _search_youtube(query: str, limit: int, extract_all: bool, urls_only: bool):
    from octane.extractors.youtube.search import search_youtube
    from octane.extractors.pipeline import extract
    from rich.table import Table

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: search_youtube(query, limit=limit))

    if urls_only:
        for r in results:
            print(r["url"])  # stdout for piping — no Rich formatting
        return

    table = Table(title=f"YouTube: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Title", max_width=50)
    table.add_column("Channel", max_width=20)
    table.add_column("Duration")
    table.add_column("URL", max_width=45)

    for i, r in enumerate(results, 1):
        table.add_row(str(i), r["title"][:50], r["channel"][:20], r["duration"], r["url"])
    console.print(table)

    if extract_all and results:
        console.print(f"\n[bold]🔄 Extracting all {len(results)} results...[/]\n")
        for i, r in enumerate(results, 1):
            console.print(f"[cyan]  [{i}/{len(results)}][/] {r['title'][:60]}")
            try:
                doc = await extract(r["url"])
                if doc.raw_text:
                    await _persist_extraction(doc)
                    console.print(f"    [green]✅ {doc.total_words:,} words[/]")
                else:
                    console.print(f"    [yellow]⚠  No content extracted[/]")
            except Exception as exc:
                console.print(f"    [red]❌ {exc}[/]")


@extract_app.command("search-arxiv")
def search_arxiv_cmd(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, help="Max results."),
    extract_all: bool = typer.Option(False, "--extract-all", help="Extract all results."),
    urls_only: bool = typer.Option(False, "--urls-only", help="Print only arXiv IDs (for piping)."),
):
    """Search arXiv and display results.

    Examples::

        octane extract search-arxiv "retrieval augmented generation"
        octane extract search-arxiv "LLM agents" --extract-all
        octane extract search-arxiv "RAG" --urls-only
    """
    asyncio.run(_search_arxiv(query, limit, extract_all, urls_only))


async def _search_arxiv(query: str, limit: int, extract_all: bool, urls_only: bool):
    from octane.extractors.academic.arxiv_search import search_arxiv
    from octane.extractors.pipeline import extract
    from rich.table import Table

    try:
        results = search_arxiv(query, max_results=limit)
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "503" in err_msg:
            console.print(f"[red]arXiv rate-limited (HTTP {err_msg.split()[-1] if err_msg.split() else '429'}). Wait a few minutes and retry.[/]")
        else:
            console.print(f"[red]arXiv search failed: {exc}[/]")
        return

    if urls_only:
        for r in results:
            print(r["arxiv_id"])
        return

    table = Table(title=f"arXiv: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("ID", max_width=15)
    table.add_column("Title", max_width=45)
    table.add_column("Authors", max_width=25)
    table.add_column("Date", max_width=12)

    for i, r in enumerate(results, 1):
        authors = ", ".join(r["authors"][:3])
        if len(r["authors"]) > 3:
            authors += " et al."
        date = r["published"][:10] if r["published"] else ""
        table.add_row(str(i), r["arxiv_id"], r["title"][:45], authors[:25], date)
    console.print(table)

    if extract_all and results:
        console.print(f"\n[bold]🔄 Extracting all {len(results)} results...[/]\n")
        for i, r in enumerate(results, 1):
            console.print(f"[cyan]  [{i}/{len(results)}][/] {r['title'][:60]}")
            try:
                doc = await extract(r["arxiv_id"])
                if doc.raw_text:
                    await _persist_extraction(doc)
                    console.print(f"    [green]✅ {doc.total_words:,} words[/]")
                else:
                    console.print(f"    [yellow]⚠  No content extracted[/]")
            except Exception as exc:
                console.print(f"    [red]❌ {exc}[/]")
