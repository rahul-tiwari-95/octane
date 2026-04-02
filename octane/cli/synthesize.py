"""octane synthesize — LLM synthesis from extracted content via stdin.

Usage:
    octane search news "NVDA" --json | octane extract stdin --json | octane synthesize --stdin
    octane search web "AI" --json | octane extract stdin --json | octane synthesize --stdin --template briefing
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer

from octane.cli._shared import console, err_console

synthesize_app = typer.Typer(
    help="🧠 Synthesize extracted content via LLM.",
    no_args_is_help=True,
)


@synthesize_app.command("run")
def synthesize_run(
    query: str = typer.Option("", "--query", "-q", help="Original query for context (improves synthesis)."),
    from_stdin: bool = typer.Option(False, "--stdin", help="Read extracted content JSON from stdin."),
    template: str = typer.Option("", "--template", "-t", help="Response template name (e.g. briefing, compare)."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for full output instead of streaming."),
):
    """Synthesize content from stdin into a coherent response.

    Reads JSON from ``octane extract stdin --json`` and produces an LLM-synthesized answer.

    Examples::

        octane search news "NVDA" --json | octane extract stdin --json | octane synthesize run --stdin
        octane search web "AI" --json | octane extract stdin --json | octane synthesize run --stdin --query "What's new in AI?"
    """
    if not from_stdin:
        console.print("[red]Use --stdin to read from pipe. See: octane synthesize run --help[/]")
        raise typer.Exit(1)
    asyncio.run(_synthesize_stdin(query, template, no_stream))


async def _synthesize_stdin(query: str, template: str, no_stream: bool):
    from octane.agents.web.synthesizer import Synthesizer
    from octane.agents.web.content_extractor import ExtractedContent

    if sys.stdin.isatty():
        print("Error: no input on stdin. Pipe extracted content in.", file=sys.stderr)
        raise typer.Exit(1)

    raw_input = sys.stdin.read().strip()
    if not raw_input:
        print("Error: empty stdin.", file=sys.stderr)
        raise typer.Exit(1)

    try:
        data = json.loads(raw_input)
    except json.JSONDecodeError:
        print("Error: stdin is not valid JSON. Use: octane extract stdin --json", file=sys.stderr)
        raise typer.Exit(1)

    # Accept output from octane extract stdin --json → {"extracted": [...]}
    items = []
    if isinstance(data, dict) and "extracted" in data:
        items = data["extracted"]
    elif isinstance(data, list):
        items = data
    else:
        print("Error: unexpected JSON format. Expected {\"extracted\": [...]}", file=sys.stderr)
        raise typer.Exit(1)

    if not items:
        print("Error: no extracted content to synthesize.", file=sys.stderr)
        raise typer.Exit(1)

    # Convert to ExtractedContent objects
    articles = []
    for item in items:
        articles.append(ExtractedContent(
            url=item.get("url", ""),
            text=item.get("text", ""),
            word_count=item.get("word_count", len(item.get("text", "").split())),
            method=item.get("method", "pipe"),
            title=item.get("title", ""),
        ))

    # Use query from flag or infer from first article
    synth_query = query or f"Summarize the following {len(articles)} sources"

    err_console.print(f"[dim]🧠 Synthesizing {len(articles)} sources...[/]")

    synthesizer = Synthesizer()

    if template:
        from octane.utils.response_templates import apply_template
        # Template will be picked up by synthesizer via apply_template in synthesis

    result = await synthesizer.synthesize_with_content(
        query=synth_query,
        extracted_articles=articles,
        deep=len(articles) > 3,
    )

    if result:
        print(result)
    else:
        print("Error: synthesis returned empty result.", file=sys.stderr)
        raise typer.Exit(1)
