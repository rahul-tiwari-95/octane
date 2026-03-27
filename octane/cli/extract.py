"""octane extract — Content extraction from YouTube, arXiv, PDF, EPUB.

Usage:
    octane extract "https://youtube.com/watch?v=..."
    octane extract "2408.09869"                        # arXiv ID
    octane extract ./path/to/paper.pdf --quality deep
    octane extract ./path/to/book.epub
    octane extract --search-youtube "attention mechanisms"
    octane extract --search-arxiv "retrieval augmented generation"
"""

from __future__ import annotations

import asyncio

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


@extract_app.command("search-youtube")
def search_youtube_cmd(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, help="Max results."),
):
    """Search YouTube and display results."""
    from octane.extractors.youtube.search import search_youtube
    from rich.table import Table

    results = search_youtube(query, limit=limit)

    table = Table(title=f"YouTube: {query}", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Title", max_width=50)
    table.add_column("Channel", max_width=20)
    table.add_column("Duration")
    table.add_column("URL", max_width=45)

    for i, r in enumerate(results, 1):
        table.add_row(str(i), r["title"][:50], r["channel"][:20], r["duration"], r["url"])

    console.print(table)


@extract_app.command("search-arxiv")
def search_arxiv_cmd(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, help="Max results."),
):
    """Search arXiv and display results."""
    from octane.extractors.academic.arxiv_search import search_arxiv
    from rich.table import Table

    results = search_arxiv(query, max_results=limit)

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
    from rich.markdown import Markdown
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
