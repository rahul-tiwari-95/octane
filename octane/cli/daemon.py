"""octane daemon sub-app."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console

daemon_app = typer.Typer(
    name="daemon",
    help="🔧 Manage the Octane background daemon.",
    no_args_is_help=True,
)


@daemon_app.command("start")
def daemon_start(
    foreground: bool = typer.Option(
        False, "--foreground", "-f",
        help="Run in foreground (don't fork to background). Useful for debugging.",
    ),
    topology: str = typer.Option(
        "auto", "--topology", "-t",
        help="Topology: auto|compact|balanced|power",
    ),
):
    """🚀 Start the Octane daemon."""
    from octane.daemon.client import is_daemon_running

    if is_daemon_running():
        console.print("[yellow]⚠ Daemon is already running.[/]")
        return

    if foreground:
        console.print("[dim]Starting daemon in foreground... (Ctrl+C to stop)[/]")
        asyncio.run(_daemon_start_foreground(topology))
    else:
        console.print("[dim]Starting daemon in background...[/]")
        asyncio.run(_daemon_start_background(topology))


async def _daemon_start_foreground(topology: str):
    from octane.daemon.lifecycle import DaemonLifecycle

    lifecycle = DaemonLifecycle(topology_name=topology, foreground=True)
    try:
        await lifecycle.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Daemon stopped.[/]")
        console.print("[dim]Daemon stopped - [/]")


async def _daemon_start_background(topology: str):
    from octane.daemon.lifecycle import start_daemon

    success = await start_daemon(topology=topology, foreground=False)
    if success:
        from octane.daemon.client import get_pid_path
        pid = get_pid_path().read_text().strip() if get_pid_path().exists() else "?"
        console.print(f"[green]✅ Daemon started[/] (PID: {pid})")
    else:
        console.print("[red]❌ Failed to start daemon.[/]")
        raise typer.Exit(1)


@daemon_app.command("stop")
def daemon_stop():
    """🛑 Stop the Octane daemon."""
    asyncio.run(_daemon_stop())


async def _daemon_stop():
    from octane.daemon.lifecycle import stop_daemon

    console.print("[dim]Stopping daemon...[/]")
    success = await stop_daemon()
    if success:
        console.print("[green]✅ Daemon stopped.[/]")
    else:
        console.print("[yellow]Daemon was not running.[/]")


@daemon_app.command("status")
def daemon_status_cmd():
    """📊 Show daemon status — PID, uptime, queue, connections."""
    asyncio.run(_daemon_status())


async def _daemon_status():
    from octane.daemon.lifecycle import daemon_status

    result = await daemon_status()

    if not result.get("running"):
        console.print("[dim]Daemon is not running.[/]")
        console.print("[dim]Start with: octane daemon start[/]")
        return

    if result.get("status") == "error":
        console.print(f"[red]Error: {result.get('error')}[/]")
        return

    data = result.get("data", {})
    daemon_info = data.get("daemon", {})
    queue_info = data.get("queue", {})

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Key", style="cyan")
    tbl.add_column("Value", style="white")

    tbl.add_row("Status", f"[green]{daemon_info.get('status', '?')}[/]")
    tbl.add_row("PID", str(daemon_info.get("pid", "?")))
    tbl.add_row("Uptime", f"{daemon_info.get('uptime_seconds', 0):.0f}s")
    tbl.add_row("Topology", daemon_info.get("topology", "?"))
    tbl.add_row("Queue Depth", str(queue_info.get("size", 0)))

    depth = queue_info.get("depth_by_priority", {})
    if any(v > 0 for v in depth.values()):
        depth_str = "  ".join(f"{k}={v}" for k, v in depth.items() if v > 0)
        tbl.add_row("Queue Detail", depth_str)

    conns = daemon_info.get("connections", {})
    for svc in ("redis", "postgres", "bodega"):
        info = conns.get(svc, {})
        status = info.get("status", "unknown")
        emoji = "✅" if status == "connected" else "⚠️" if status == "degraded" else "❌"
        latency = info.get("latency_ms", 0)
        lat_str = f" ({latency:.0f}ms)" if latency > 0 else ""
        tbl.add_row(svc.capitalize(), f"{emoji} {status}{lat_str}")

    models = daemon_info.get("models", {})
    for mid, minfo in models.items():
        idle = minfo.get("idle_seconds", 0)
        reqs = minfo.get("request_count", 0)
        tbl.add_row(f"Model: {mid}", f"idle {idle:.0f}s, {reqs} reqs")

    console.print(Panel(tbl, title="[bold cyan]🔧 Octane Daemon[/]", border_style="cyan"))


@daemon_app.command("drain")
def daemon_drain():
    """💧 Drain the daemon — stop accepting new tasks, finish running ones."""
    asyncio.run(_daemon_drain())


async def _daemon_drain():
    from octane.daemon.client import DaemonClient, is_daemon_running

    if not is_daemon_running():
        console.print("[dim]Daemon is not running.[/]")
        return

    client = DaemonClient()
    if not await client.connect():
        console.print("[red]Cannot connect to daemon.[/]")
        return

    try:
        response = await client.request("drain", {})
        if response.get("status") == "ok":
            console.print("[green]✅ Daemon entering drain mode.[/]")
        else:
            console.print(f"[red]Error: {response.get('error', 'unknown')}[/]")
    finally:
        await client.close()


@daemon_app.command("watch")
def daemon_watch(
    interval: float = typer.Option(
        1.0, "--interval", "-i",
        help="Refresh interval in seconds.",
    ),
    log_lines: int = typer.Option(
        18, "--log-lines", "-l",
        help="Number of recent log lines to show.",
    ),
):
    """👁  Live dashboard — queue table + log stream.

    Shows a continuously refreshed table of all pending/active requests
    (one row per request), connection health, and a scrolling log window.

    Press Ctrl+C to open the pause prompt — type a task ID to pause
    that specific request while all others continue.  Press Enter alone
    to cancel and resume normal watch mode.
    """
    asyncio.run(_daemon_watch(interval, log_lines))


async def _daemon_watch(interval: float, log_lines: int):
    import signal as _signal
    from collections import deque
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text as RText
    from octane.daemon.client import DaemonClient, is_daemon_running

    if not is_daemon_running():
        console.print("[red]Daemon is not running.[/]  Start it with: [cyan]octane daemon start[/]")
        return

    logs: deque[str] = deque(maxlen=log_lines)
    pause_requested = False

    def _request_pause(*_):
        nonlocal pause_requested
        pause_requested = True

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(_signal.SIGINT, _request_pause)

    def _build_layout(
        daemon_info: dict,
        queue_info: dict,
        requests: list[dict],
        connections: dict,
    ) -> Layout:
        from rich.layout import Layout as _Layout
        from rich.panel import Panel as _Panel
        from rich.table import Table as _Table
        from rich.text import Text as _Text

        layout = _Layout()
        layout.split_column(
            _Layout(name="header", size=3),
            _Layout(name="body"),
            _Layout(name="logs", size=log_lines + 2),
        )

        uptime = daemon_info.get("uptime_seconds", 0)
        pid = daemon_info.get("pid", "?")
        topo = daemon_info.get("topology", "?")
        q_size = queue_info.get("size", 0)
        header_txt = (
            f"  🔥 [bold cyan]Octane Daemon Watch[/]  "
            f"PID=[yellow]{pid}[/]  Topology=[cyan]{topo}[/]  "
            f"Uptime=[white]{uptime:.0f}s[/]  Queue=[white]{q_size}[/]  "
            f"[dim](Ctrl+C to pause a request)[/]"
        )
        layout["header"].update(_Panel(header_txt, border_style="cyan", padding=(0, 1)))

        body_layout = _Layout()
        body_layout.split_row(
            _Layout(name="requests", ratio=3),
            _Layout(name="health", ratio=1),
        )
        layout["body"].update(body_layout)

        req_tbl = _Table(show_header=True, box=None, padding=(0, 1), expand=True)
        req_tbl.add_column("Task ID", style="dim", width=12)
        req_tbl.add_column("Command", style="cyan", width=12)
        req_tbl.add_column("Priority", style="yellow", width=8)
        req_tbl.add_column("Wait", style="white", width=7)
        req_tbl.add_column("Aged", style="dim", width=5)
        req_tbl.add_column("State", style="white", width=8)

        if requests:
            for req in requests:
                tid = req.get("task_id", "?")[:12]
                cmd = req.get("command", "?")[:12]
                pri = req.get("priority", "?")
                wait = f"{req.get('wait_sec', 0):.1f}s"
                aged = str(req.get("aged_count", 0))
                paused = req.get("paused", False)
                state_str = "[yellow]⏸ paused[/]" if paused else "[green]▶ queued[/]"
                req_tbl.add_row(tid, cmd, pri, wait, aged, state_str)
        else:
            req_tbl.add_row("[dim]—[/]", "[dim]queue empty[/]", "", "", "", "")

        body_layout["requests"].update(
            _Panel(req_tbl, title="[bold]📋 Requests[/]", border_style="dim")
        )

        health_tbl = _Table(show_header=False, box=None, padding=(0, 1), expand=True)
        health_tbl.add_column("Svc", style="cyan", no_wrap=True)
        health_tbl.add_column("Status", no_wrap=True)
        for svc in ("redis", "postgres", "bodega"):
            info = connections.get(svc, {})
            st = info.get("status", "unknown")
            lat = info.get("latency_ms", 0)
            lat_str = f" {lat:.0f}ms" if lat > 0 else ""
            if st == "connected":
                ico = f"[green]OK[/]{lat_str}"
            elif st == "degraded":
                ico = f"[yellow]DEG[/]{lat_str}"
            else:
                ico = "[red]DOWN[/]"
            health_tbl.add_row(svc[:8], ico)

        body_layout["health"].update(
            _Panel(health_tbl, title="[bold]🔌 Health[/]", border_style="dim")
        )

        log_content = "\n".join(logs) if logs else "(waiting for logs…)"
        try:
            log_text = _Text.from_markup(log_content)
        except Exception:
            log_text = _Text(log_content, style="dim")
        layout["logs"].update(
            _Panel(log_text, title="[bold]📜 Logs[/]", border_style="dim")
        )

        return layout

    client = DaemonClient()
    if not await client.connect():
        console.print("[red]Cannot connect to daemon socket.[/]")
        return

    try:
        with Live(console=console, refresh_per_second=max(1, int(1 / interval)), screen=True) as live:
            while True:
                if pause_requested:
                    pause_requested = False
                    live.stop()
                    console.print("\n[bold yellow]⏸  Pause a request[/]")
                    console.print("[dim]Enter a Task ID to pause (or press Enter to cancel):[/] ", end="")
                    try:
                        tid_input = await loop.run_in_executor(None, input)
                        tid_input = tid_input.strip()
                        if tid_input:
                            pause_resp = await client.request("pause_request", {"task_id": tid_input})
                            if pause_resp.get("status") == "ok":
                                console.print(f"[green]✅ Request {tid_input[:12]}… paused.[/]")
                                console.print("[dim]Resume with: octane daemon resume <task_id>[/]")
                            else:
                                console.print(f"[red]Error: {pause_resp.get('error', 'unknown')}[/]")
                        else:
                            console.print("[dim]Cancelled.[/]")
                    except (EOFError, KeyboardInterrupt):
                        pass
                    live.start()
                    continue

                try:
                    status_resp = await client.request("status", {})
                    list_resp = await client.request("list_requests", {})
                except Exception as exc:
                    logs.appendleft(f"[red]poll error: {exc}[/]")
                    await asyncio.sleep(interval)
                    continue

                data = status_resp.get("data", {})
                daemon_info = data.get("daemon", {})
                queue_info = data.get("queue", {})
                connections = daemon_info.get("connections", {})
                requests = list_resp.get("data", {}).get("requests", [])

                log_entries = data.get("recent_logs", [])
                for entry in log_entries:
                    logs.appendleft(str(entry))

                layout = _build_layout(daemon_info, queue_info, requests, connections)
                live.update(layout)
                await asyncio.sleep(interval)

    except asyncio.CancelledError:
        pass
    finally:
        loop.remove_signal_handler(_signal.SIGINT)
        await client.close()
        console.print("\n[dim]Watch stopped.[/]")


@daemon_app.command("resume")
def daemon_resume(
    task_id: str = typer.Argument(..., help="Task ID to resume (from 'octane daemon watch')."),
):
    """▶️  Resume a paused daemon request."""
    asyncio.run(_daemon_resume(task_id))


async def _daemon_resume(task_id: str):
    from octane.daemon.client import DaemonClient, is_daemon_running

    if not is_daemon_running():
        console.print("[dim]Daemon is not running.[/]")
        return

    client = DaemonClient()
    if not await client.connect():
        console.print("[red]Cannot connect to daemon.[/]")
        return

    try:
        resp = await client.request("resume_request", {"task_id": task_id})
        if resp.get("status") == "ok":
            console.print(f"[green]✅ Request {task_id[:16]}… resumed.[/]")
        else:
            console.print(f"[red]Error: {resp.get('error', 'unknown')}[/]")
    finally:
        await client.close()
