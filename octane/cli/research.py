"""octane research sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_shadow_config, _ensure_shadow_group

research_app = typer.Typer(
    name="research",
    help="🔬 Long-running background research workflows (powered by Shadows).",
    no_args_is_help=True,
)

_LOG_FOLLOW_BUFFER = 200


@research_app.command("start")
def research_start(
    topic: str = typer.Argument(..., help="Research topic or question"),
    every: float = typer.Option(6.0, "--every", "-e", help="Cycle interval in hours (default: 6)"),
    depth: str = typer.Option("deep", "--depth", "-d",
                              help="Research depth: shallow (2 angles), deep (4, default), exhaustive (8)"),
):
    """🔬 Start a background research workflow.

    Schedules a perpetual Shadows task that runs the full OSA pipeline for
    TOPIC every EVERY hours — extracting content, synthesising findings,
    and storing them in Postgres for review via ``octane research report``.

    The ``--depth`` flag controls how many parallel search angles are used
    per cycle:

    \\b
        shallow    — 2 angles (fastest, good for quick market checks)
        deep       — 4 angles (default, balanced coverage)
        exhaustive — 8 angles (most thorough, use for deep-dive research)

    Examples::

        octane research start "NVDA earnings outlook"
        octane research start "Apple Vision Pro market reception" --every 12
        octane research start "Fed rate decision impact" --depth exhaustive --every 3
    """
    _valid_depths = {"shallow", "deep", "exhaustive"}
    if depth not in _valid_depths:
        console.print(f"[red]Invalid depth '{depth}'. Must be one of: {', '.join(sorted(_valid_depths))}[/]")
        raise typer.Exit(1)
    asyncio.run(_research_start(topic, every, depth))


async def _research_start(topic: str, interval_hours: float, depth: str = "deep"):
    import subprocess
    import sys
    import time
    from shadows import Shadow
    from octane.research.models import ResearchTask
    from octane.research.store import ResearchStore
    from octane.tasks.research import research_cycle
    from octane.tasks.worker_process import read_pid
    from octane.config import settings

    shadow_name, redis_url = _get_shadow_config()
    await _ensure_shadow_group(shadow_name, redis_url)

    task = ResearchTask(topic=topic, interval_hours=interval_hours)
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    await store.register_task(task)
    await store.log_entry(task.id, f"🔬 Research task created: {topic}")

    console.print(f"[dim]Scheduling research task [bold]{task.id}[/bold]…[/]")
    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            shadow.register(research_cycle)
            await shadow.add(research_cycle, key=task.id)(
                task_id=task.id,
                topic=topic,
                interval_hours=interval_hours,
                depth=depth,
            )
        console.print(
            f"[green]✅ Research started[/]\n"
            f"  [bold]ID:[/]     [cyan]{task.id}[/]\n"
            f"  [bold]Topic:[/]  {topic}\n"
            f"  [bold]Every:[/]  {interval_hours}h  ·  "
            f"[bold]Depth:[/] {depth}\n"
        )
    except Exception as exc:
        console.print(f"[red]Failed to schedule task: {exc}[/]")
        await store.close()
        raise typer.Exit(1)

    existing_pid = read_pid()
    if existing_pid:
        console.print(f"[dim]Worker already running (PID {existing_pid}) — task picked up automatically.[/]")
        await store.close()
        return

    console.print("[dim]Starting Octane background worker…[/]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "octane.tasks.worker_process",
         "--shadows-name", shadow_name, "--redis-url", redis_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.8)
    pid = read_pid()
    if pid:
        console.print(f"[green]🚀 Worker started (PID {pid})[/]")
    else:
        console.print(f"[green]🚀 Worker launched (PID ~{proc.pid})[/]")

    console.print(
        f"\n[dim]Run [bold]octane research log {task.id}[/bold] to follow progress.\n"
        f"Run [bold]octane research log {task.id} --follow[/bold] to stream live.[/]"
    )
    await store.close()


@research_app.command("status")
def research_status():
    """📊 List all active research tasks."""
    asyncio.run(_research_status())


async def _research_status():
    from octane.research.store import ResearchStore
    from octane.config import settings
    from octane.tasks.worker_process import read_pid

    _, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    tasks = await store.list_tasks()
    await store.close()

    pid = read_pid()
    worker_line = (
        f"[green]🟢 Running (PID {pid})[/]" if pid else "[red]🔴 Not running[/]"
    )
    console.print(Panel(worker_line, title="[bold]Octane Worker[/]", border_style="cyan"))

    if not tasks:
        console.print("[dim]No research tasks found. Run: octane research start \"<topic>\"[/]")
        return

    tbl = Table(title="Active Research Tasks", show_lines=False)
    tbl.add_column("ID", style="cyan", width=10)
    tbl.add_column("Topic", style="white")
    tbl.add_column("Every", style="yellow", justify="right", width=7)
    tbl.add_column("Depth", style="blue", justify="right", width=10)
    tbl.add_column("Cycles", style="green", justify="right", width=7)
    tbl.add_column("Findings", style="magenta", justify="right", width=9)
    tbl.add_column("Age", style="dim", justify="right", width=8)
    tbl.add_column("Status", width=8)

    for t in tasks:
        age_str = f"{t.age_hours:.1f}h"
        status_str = "[green]active[/]" if t.is_active else "[red]stopped[/]"
        tbl.add_row(
            t.id, t.topic[:55], f"{t.interval_hours}h",
            getattr(t, "depth", "deep"),
            str(t.cycle_count), str(t.finding_count),
            age_str, status_str,
        )
    console.print(tbl)
    console.print(
        "[dim]  octane research log <id>          — view progress log\n"
        "  octane research report <id>       — synthesise all findings\n"
        "  octane research stop <id>         — cancel task[/]"
    )


@research_app.command("log")
def research_log(
    task_id: str = typer.Argument(..., help="Research task ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new entries in real time"),
    n: int = typer.Option(50, "--lines", "-n", help="Number of recent log lines to show"),
):
    """📋 Show the research progress log.

    Use ``--follow`` to stream live entries as the background task runs.

    Examples::

        octane research log abc12345
        octane research log abc12345 --follow
    """
    asyncio.run(_research_log(task_id, follow, n))


async def _research_log(task_id: str, follow: bool, n: int):
    from octane.research.store import ResearchStore
    from octane.config import settings

    _, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    task = await store.get_task(task_id)
    if task is None:
        console.print(f"[yellow]No task found with ID: [bold]{task_id}[/][/]")
        console.print("[dim]Run 'octane research status' to see all tasks.[/]")
        await store.close()
        return

    console.print(Panel(
        f"[bold]ID:[/]    {task.id}\n"
        f"[bold]Topic:[/] {task.topic}\n"
        f"[bold]Every:[/] {task.interval_hours}h  ·  "
        f"[bold]Depth:[/] {getattr(task, 'depth', 'deep')}  ·  "
        f"[bold]Cycles:[/] {task.cycle_count}  ·  "
        f"[bold]Findings:[/] {task.finding_count}",
        title="[bold cyan]🔬 Research Task[/]",
        border_style="cyan",
    ))

    entries = await store.get_log(task_id, n=n)
    if not entries:
        console.print("[dim]No log entries yet — task may not have run yet.[/]")
    else:
        for line in entries:
            _print_log_line(line)

    if not follow:
        await store.close()
        return

    console.print("[dim]  Following… (Ctrl+C to stop)[/]")
    seen = len(entries)
    last_cycle = task.cycle_count
    last_findings = task.finding_count
    try:
        while True:
            await asyncio.sleep(2.0)
            all_entries = await store.get_log(task_id, n=_LOG_FOLLOW_BUFFER)
            new = all_entries[seen:]
            for line in new:
                _print_log_line(line)
            seen = len(all_entries)
            refreshed = await store.get_task(task_id)
            if refreshed and (
                refreshed.cycle_count != last_cycle
                or refreshed.finding_count != last_findings
            ):
                last_cycle = refreshed.cycle_count
                last_findings = refreshed.finding_count
                console.print(
                    f"[bold cyan]  ↺ Status:[/] Cycles={last_cycle}  "
                    f"Findings={last_findings}  "
                    f"(next cycle in ~{refreshed.interval_hours}h)"
                )
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[dim]Stopped following.[/]")
    finally:
        await store.close()


def _print_log_line(line: str) -> None:
    """Colour-code a single research log line for terminal output."""
    if "✅" in line or "complete" in line.lower():
        console.print(f"[green]{line}[/]")
    elif "⚠" in line or "warn" in line.lower() or "error" in line.lower():
        console.print(f"[yellow]{line}[/]")
    elif "⚙" in line or "start" in line.lower():
        console.print(f"[cyan]{line}[/]")
    else:
        console.print(f"[dim]{line}[/]")


@research_app.command("report")
def research_report(
    task_id: str = typer.Argument(..., help="Research task ID"),
    raw: bool = typer.Option(False, "--raw", help="Print raw findings without LLM synthesis"),
    cycles: int = typer.Option(None, "--cycles", "-c", help="Use only the last N cycles"),
    since: str = typer.Option(None, "--since", help="Include findings on/after ISO date (e.g. 2026-01-01)"),
    export: str = typer.Option(None, "--export", "-o", help="Save report to this file path (.md)"),
):
    """📄 Synthesise all findings into a final research report.

    Pulls every stored finding for the task and passes them through the
    ResearchSynthesizer for a cohesive narrative.  Use ``--raw`` to skip
    synthesis and print findings sequentially.

    Examples::

        octane research report abc12345
        octane research report abc12345 --raw
        octane research report abc12345 --cycles 3
        octane research report abc12345 --since 2026-01-01
        octane research report abc12345 --export ~/reports/nvda.md
    """
    asyncio.run(_research_report(task_id, raw, cycles=cycles, since=since, export=export))


async def _research_report(
    task_id: str,
    raw_mode: bool,
    *,
    cycles: int | None = None,
    since: str | None = None,
    export: str | None = None,
):
    import pathlib
    from datetime import datetime, timezone
    from octane.research.store import ResearchStore
    from octane.research.synthesizer import ResearchSynthesizer
    from octane.config import settings

    _, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    task = await store.get_task(task_id)
    if task is None:
        console.print(f"[yellow]No task found: [bold]{task_id}[/][/]")
        await store.close()
        return

    console.print(Panel(
        f"[bold]Topic:[/]    {task.topic}\n"
        f"[bold]Cycles:[/]   {task.cycle_count}  ·  "
        f"[bold]Age:[/]      {task.age_hours:.1f}h",
        title=f"[bold magenta]📄 Research Report — {task_id}[/]",
        border_style="magenta",
    ))

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        except ValueError:
            console.print(f"[red]Invalid --since date: {since!r}. Use ISO format, e.g. 2026-01-01.[/]")
            await store.close()
            return

    if raw_mode:
        findings = await store.get_findings(task_id)
        await store.close()

        if not findings:
            console.print(f"[yellow]No findings stored yet for [bold]{task_id}[/].[/]")
            return

        if since_dt:
            findings = [f for f in findings if (f.created_at or datetime.min.replace(tzinfo=timezone.utc)) >= since_dt]
        if cycles:
            findings = findings[-cycles:]

        if not findings:
            console.print("[yellow]No findings match the specified filters.[/]")
            return

        for f in findings:
            ts_str = f.created_at.strftime("%m-%d %H:%M UTC") if f.created_at else "unknown"
            console.print(Panel(
                f.content,
                title=f"[dim]Cycle {f.cycle_num}  ·  {ts_str}  ·  {f.word_count} words[/]",
                border_style="dim",
            ))
        return

    from octane.tools.bodega_inference import BodegaInferenceClient

    bodega = BodegaInferenceClient()
    try:
        with console.status("[dim]⚙  Synthesising findings…[/]", spinner="dots"):
            synth = ResearchSynthesizer(store, bodega=bodega)
            report_text = await synth.generate(
                task_id,
                cycles=cycles,
                since=since_dt,
            )
    except Exception as exc:
        console.print(f"[yellow]Synthesis unavailable ({exc}) — falling back to plain format.[/]\n")
        plain_synth = ResearchSynthesizer(store, bodega=None)
        report_text = await plain_synth.generate(task_id, cycles=cycles, since=since_dt)
    finally:
        await store.close()
        await bodega.close()

    console.print(Panel(
        report_text,
        title=f"[bold green]🔬 Synthesised Report: {task.topic}[/]",
        border_style="green",
        padding=(1, 2),
    ))

    if export:
        out_path = pathlib.Path(export).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        console.print(f"[dim]  ✅ Report saved → {out_path}[/]")


@research_app.command("stop")
def research_stop(
    task_id: str = typer.Argument(..., help="Research task ID to cancel"),
):
    """🛑 Stop a background research task."""
    asyncio.run(_research_stop(task_id))


async def _research_stop(task_id: str):
    from shadows import Shadow
    from octane.research.store import ResearchStore
    from octane.config import settings

    shadow_name, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    task = await store.get_task(task_id)
    if task is None:
        console.print(f"[yellow]No task found: [bold]{task_id}[/][/]")
        await store.close()
        return

    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            await shadow.cancel(task_id)
        console.print(f"[green]✅ Shadows task [bold]{task_id}[/bold] cancelled.[/]")
    except Exception as exc:
        console.print(f"[yellow]Shadows cancel: {exc} (updating status anyway)[/]")

    await store.update_task_status(task_id, "stopped")
    await store.log_entry(task_id, "🛑 Task stopped by user")
    await store.close()

    console.print(
        f"[dim]Findings retained — run "
        f"[bold]octane research report {task_id}[/bold] to read the report.[/]"
    )


@research_app.command("list")
def research_list():
    """📋 List all research tasks with findings counts and last-run time."""
    asyncio.run(_research_list())


async def _research_list():
    from octane.research.store import ResearchStore
    from octane.config import settings

    _, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    tasks = await store.list_tasks()
    await store.close()

    if not tasks:
        console.print("[dim]No research tasks found.[/]")
        console.print("[dim]Start one with: octane research start \"<topic>\"[/]")
        return

    tbl = Table(
        title="Research Tasks",
        show_lines=True,
        border_style="cyan",
    )
    tbl.add_column("ID",       style="cyan",    width=10)
    tbl.add_column("Topic",    style="white",   min_width=30)
    tbl.add_column("Depth",    style="blue",    width=10, justify="center")
    tbl.add_column("Cycles",   style="green",   width=7,  justify="right")
    tbl.add_column("Findings", style="magenta", width=9,  justify="right")
    tbl.add_column("Every",    style="yellow",  width=7,  justify="right")
    tbl.add_column("Age",      style="dim",     width=8,  justify="right")
    tbl.add_column("Status",   width=9,         justify="center")

    for t in tasks:
        age_h = t.age_hours
        age_str = f"{age_h:.0f}h" if age_h >= 1 else f"{int(age_h * 60)}m"
        if t.is_active:
            status_str = "[green]● active[/]"
        else:
            status_str = "[red]◼ stopped[/]"
        depth_val = getattr(t, "depth", "deep")

        tbl.add_row(
            t.id,
            t.topic[:60],
            depth_val,
            str(t.cycle_count),
            str(t.finding_count),
            f"{t.interval_hours}h",
            age_str,
            status_str,
        )

    console.print(tbl)
    console.print(
        "[dim]  octane research log <id>       — view progress log\n"
        "  octane research report <id>    — synthesise findings\n"
        "  octane research stop <id>      — cancel task[/]"
    )


@research_app.command("library")
def research_library():
    """📚 Browse all stored research findings grouped by task."""
    asyncio.run(_research_library())


async def _research_library():
    from octane.research.store import ResearchStore
    from octane.config import settings

    _, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)
    tasks = await store.list_tasks()

    if not tasks:
        console.print("[dim]No research tasks found.[/]")
        console.print("[dim]Start one with: octane research start \"<topic>\"[/]")
        await store.close()
        return

    total_findings = 0
    for task in tasks:
        findings = await store.get_findings(task.id)
        total_findings += len(findings)
        status_str = "[green]● active[/]" if task.is_active else "[red]◼ stopped[/]"
        console.print(Panel(
            f"[bold]ID:[/]       {task.id}\n"
            f"[bold]Status:[/]   {status_str}   [bold]Depth:[/] {getattr(task, 'depth', 'deep')}\n"
            f"[bold]Cycles:[/]   {task.cycle_count}  ·  "
            f"[bold]Findings:[/] {len(findings)}  ·  "
            f"[bold]Every:[/]    {task.interval_hours}h",
            title=f"[bold cyan]📚 {task.topic[:70]}[/]",
            border_style="cyan",
        ))
        if findings:
            for f in findings[-2:]:
                ts_str = f.created_at.strftime("%m-%d %H:%M UTC") if f.created_at else "?"
                preview = f.content[:280].replace("\n", " ")
                if len(f.content) > 280:
                    preview += "…"
                console.print(f"  [dim]Cycle {f.cycle_num} · {ts_str} · {f.word_count}w[/]  {preview}")
        else:
            console.print("  [dim]No findings stored yet.[/]")
        console.print(
            f"[dim]  octane research report {task.id}        — synthesised report\n"
            f"  octane research report {task.id} --raw  — raw findings[/]\n"
        )

    console.print(f"[dim]Library: {len(tasks)} task(s) · {total_findings} total findings[/]")
    await store.close()


@research_app.command("recall")
def research_recall(
    query: str = typer.Argument(..., help="Keyword or phrase to search in stored findings"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results to show (default: 10)"),
):
    """🔎 Search stored research findings for a keyword or phrase.

    Performs a case-insensitive search across the content and topic of every
    stored finding and returns the most recent matches with the matching
    passage highlighted.

    Examples::

        octane research recall "NVDA"
        octane research recall "interest rates" --limit 5
        octane research recall "Fed" --limit 20
    """
    asyncio.run(_research_recall(query, limit))


async def _research_recall(query: str, limit: int):
    from octane.research.store import ResearchStore
    from octane.config import settings

    _, redis_url = _get_shadow_config()
    store = ResearchStore(redis_url=redis_url, postgres_url=settings.postgres_url)

    with console.status(f"[dim]Searching findings for '{query}'…[/]", spinner="dots"):
        findings = await store.search_findings(query, limit=limit)

    await store.close()

    # ── Primary: research task findings ─────────────────────────────────────
    if findings:
        console.print(Panel(
            f"[bold]{len(findings)}[/] finding(s) matching [bold cyan]{query}[/]",
            title="[bold]Research Recall[/]",
            border_style="cyan",
        ))

        for f in findings:
            ts_str  = f.created_at.strftime("%Y-%m-%d %H:%M UTC") if f.created_at else "?"
            content = f.content
            idx     = content.lower().find(query.lower())
            if idx >= 0:
                start   = max(0, idx - 100)
                end     = min(len(content), idx + len(query) + 200)
                snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
                snippet = snippet.replace(content[idx: idx + len(query)],
                                          f"[bold yellow]{content[idx: idx + len(query)]}[/bold yellow]", 1)
            else:
                snippet = content[:300] + ("..." if len(content) > 300 else "")

            console.print(Panel(
                snippet,
                title=(
                    f"[dim]{f.topic[:55]}  ·  cycle {f.cycle_num}  ·  "
                    f"{ts_str}  ·  {f.word_count}w  ·  task: {f.task_id}[/]"
                ),
                border_style="dim",
            ))

        console.print(
            f"[dim]  octane research report <task-id>              — full report\n"
            f"  octane research recall \"{query}\" --limit 20  — show more[/]"
        )
        return

    # ── Fallback: web_pages stored during any 'octane ask' run ──────────────
    console.print(f"[dim]No research findings for '{query}'. Searching web pages…[/]")

    from octane.tools.pg_client import PgClient
    pg = PgClient()
    await pg.connect()

    if not pg.available:
        console.print("[yellow]No findings and Postgres unavailable.[/]")
        return

    rows = await pg.fetch(
        """
        SELECT url, title, word_count, fetched_at,
               LEFT(content, 600) AS preview
        FROM   web_pages
        WHERE  content ILIKE $1 OR url ILIKE $1 OR title ILIKE $1
        ORDER  BY fetched_at DESC
        LIMIT  $2
        """,
        f"%{query}%", limit,
    )
    total_pages = await pg.fetchval(
        "SELECT COUNT(*) FROM web_pages WHERE content ILIKE $1 OR url ILIKE $1 OR title ILIKE $1",
        f"%{query}%",
    )
    await pg.close()

    if not rows:
        console.print(f"[yellow]No results in web pages either for: [bold]{query}[/bold][/]")
        console.print(
            "[dim]Tip: run 'octane ask \"<question>\"' to fetch relevant pages, "
            "or 'octane research start \"<topic>\"' for a background task.[/]"
        )
        return

    console.print(Panel(
        f"[bold]{len(rows)}[/] web page(s) matching [bold cyan]{query}[/]"
        + (f"  [dim]({total_pages} total matches)[/]" if total_pages and int(total_pages) > len(rows) else ""),
        title="[bold]Web Page Recall[/]",
        border_style="cyan",
    ))

    for r in rows:
        ts      = str(r["fetched_at"])[:19] + " UTC"
        title   = r["title"] or "(no title)"
        content = str(r["preview"])
        idx     = content.lower().find(query.lower())
        if idx >= 0:
            start   = max(0, idx - 80)
            end     = min(len(content), idx + len(query) + 180)
            snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
            snippet = snippet.replace(content[idx: idx + len(query)],
                                      f"[bold yellow]{content[idx: idx + len(query)]}[/bold yellow]", 1)
        else:
            snippet = content[:300] + ("..." if len(content) > 300 else "")

        console.print(Panel(
            snippet,
            title=f"[dim]{title[:60]}  ·  {ts}  ·  {r['word_count']}w[/]  [cyan]{r['url'][:80]}[/]",
            border_style="dim",
        ))

    console.print(
        f"[dim]  octane store pages \"{query}\"          — full web page browser\n"
        f"  octane research start \"{query}\"      — start a background research task[/]"
    )
