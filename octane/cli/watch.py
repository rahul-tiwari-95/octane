"""octane watch sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_shadow_config, _ensure_shadow_group, _patch_shadow_busygroup

watch_app = typer.Typer(
    name="watch",
    help="📡 Background stock / asset monitors (powered by Shadows).",
    no_args_is_help=True,
)


@watch_app.command("start")
def watch_start(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL or BTC"),
    interval_hours: float = typer.Option(1.0, "--every", "-e", help="Poll interval in hours"),
):
    """📈 Start a perpetual background monitor for a ticker.

    Schedules ``monitor_ticker`` as a Shadows perpetual task, then
    launches the Octane background worker subprocess if it is not
    already running.

    Examples::

        octane watch start AAPL
        octane watch start BTC --every 0.5
    """
    asyncio.run(_watch_start(ticker.upper(), interval_hours))


async def _watch_start(ticker: str, interval_hours: float):
    import subprocess
    from datetime import timedelta
    from shadows import Shadow
    from octane.tasks.monitor import monitor_ticker
    from octane.tasks.worker_process import read_pid

    shadow_name, redis_url = _get_shadow_config()
    await _ensure_shadow_group(shadow_name, redis_url)

    console.print(f"[dim]Connecting to Redis at {redis_url}...[/]")
    try:
        every = timedelta(hours=interval_hours)
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            shadow.register(monitor_ticker)
            await shadow.add(monitor_ticker, key=ticker)(ticker=ticker)
        console.print(
            f"[green]✅ Scheduled monitor for [bold]{ticker}[/bold] "
            f"(every {every}, key={ticker})[/]"
        )
    except Exception as exc:
        console.print(f"[red]Failed to schedule task: {exc}[/]")
        raise typer.Exit(1)

    existing_pid = read_pid()
    if existing_pid:
        console.print(f"[dim]Worker already running (PID {existing_pid}) — task picked up automatically.[/]")
        return

    console.print("[dim]Starting Octane background worker...[/]")
    import sys
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "octane.tasks.worker_process",
            "--shadows-name", shadow_name,
            "--redis-url", redis_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    import time
    time.sleep(0.8)
    pid = read_pid()
    if pid:
        console.print(f"[green]🚀 Worker started (PID {pid})[/]")
    else:
        console.print(f"[green]🚀 Worker launched (PID ~{proc.pid})[/]")

    console.print(
        f"\n[dim]Run [bold]octane watch status[/bold] to see running monitors.\n"
        f"Run [bold]octane watch latest {ticker}[/bold] to see the latest quote.[/]"
    )


@watch_app.command("stop")
def watch_stop():
    """🛑 Stop the Octane background worker."""
    from octane.tasks.worker_process import read_pid, _remove_pid
    import os
    import signal as _signal

    pid = read_pid()
    if pid is None:
        console.print("[yellow]No worker is running.[/]")
        return

    try:
        os.kill(pid, _signal.SIGTERM)
        _remove_pid()
        console.print(f"[green]✅ Worker (PID {pid}) stopped.[/]")
    except ProcessLookupError:
        _remove_pid()
        console.print(f"[yellow]Worker PID {pid} was already gone — cleaned up.[/]")
    except PermissionError:
        console.print(f"[red]Permission denied stopping PID {pid}.[/]")


@watch_app.command("status")
def watch_status():
    """📊 Show running monitors and worker status."""
    asyncio.run(_watch_status())


async def _watch_status():
    from shadows import Shadow
    from octane.tasks.worker_process import read_pid
    from octane.tasks import octane_tasks

    _patch_shadow_busygroup()
    shadow_name, redis_url = _get_shadow_config()

    pid = read_pid()
    worker_line = (
        f"[green]🟢 Running (PID {pid})[/]" if pid else "[red]🔴 Not running[/]"
    )
    console.print(Panel(worker_line, title="[bold]Octane Worker[/]", border_style="cyan"))

    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            for task in octane_tasks:
                shadow.register(task)
            snapshot = await shadow.snapshot()

        table = Table(title="Scheduled / Running Tasks", show_lines=False)
        table.add_column("Key", style="cyan")
        table.add_column("Function", style="green")
        table.add_column("When (UTC)", style="yellow")
        table.add_column("State", style="white")

        for exe in snapshot.future:
            table.add_row(
                exe.key,
                exe.function.__name__,
                exe.when.strftime("%Y-%m-%d %H:%M:%S"),
                "[dim]scheduled[/]",
            )
        for exe in snapshot.running:
            table.add_row(
                exe.key,
                exe.function.__name__,
                exe.when.strftime("%Y-%m-%d %H:%M:%S"),
                f"[bold green]running[/] on {exe.worker}",
            )

        total = snapshot.total_tasks
        if total == 0:
            console.print("[dim]No tasks scheduled. Run: octane watch start <ticker>[/]")
        else:
            console.print(table)
            console.print(f"[dim]Total: {total} task(s)  ·  Workers active: {len(snapshot.workers)}[/]")

    except Exception as exc:
        console.print(f"[yellow]Could not reach Redis: {exc}[/]")
        console.print("[dim]Is Redis running? Check: redis-cli ping[/]")


@watch_app.command("latest")
def watch_latest(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
):
    """💹 Show the latest stored quote for a ticker."""
    asyncio.run(_watch_latest(ticker.upper()))


async def _watch_latest(ticker: str):
    from octane.tools.redis_client import RedisClient
    import json

    redis = RedisClient()
    key = f"watch:{ticker}:latest"
    raw = await redis.get(key)
    await redis.close()

    if not raw:
        console.print(
            f"[yellow]No data yet for [bold]{ticker}[/bold]. "
            f"Run: octane watch start {ticker}[/]"
        )
        return

    try:
        data = json.loads(raw)
    except Exception:
        data = {"raw": raw}

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(Panel(table, title=f"[bold green]📈 {ticker} — Latest Quote[/]", border_style="green"))


@watch_app.command("cancel")
def watch_cancel(
    ticker: str = typer.Argument(..., help="Ticker symbol to stop monitoring"),
):
    """❌ Cancel the perpetual monitor for a ticker."""
    asyncio.run(_watch_cancel(ticker.upper()))


async def _watch_cancel(ticker: str):
    from shadows import Shadow

    _patch_shadow_busygroup()
    shadow_name, redis_url = _get_shadow_config()

    try:
        async with Shadow(name=shadow_name, url=redis_url) as shadow:
            await shadow.cancel(ticker)
        console.print(f"[green]✅ Monitor for [bold]{ticker}[/bold] cancelled.[/]")
    except Exception as exc:
        console.print(f"[red]Failed to cancel: {exc}[/]")
        raise typer.Exit(1)
