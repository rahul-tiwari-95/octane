"""octane ui sub-app — start/stop/status for the web dashboard."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import typer

from octane.cli._shared import console

ui_app = typer.Typer(
    name="ui",
    help="🖥  Manage the Octane web dashboard (Mission Control).",
    no_args_is_help=True,
)

_PID_FILE = os.path.expanduser("~/.octane/ui.pid")
_DEFAULT_PORT = int(os.environ.get("OCTANE_UI_PORT", "44480"))


def _read_pid() -> int | None:
    try:
        with open(_PID_FILE) as f:
            pid = int(f.read().strip())
        # Check if running
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        # Clean up stale file
        if os.path.exists(_PID_FILE):
            os.unlink(_PID_FILE)
        return None


def _kill_port_holders(port: int) -> list[int]:
    """Find PIDs listening on *port* and terminate them. Returns killed PIDs."""
    killed: list[int] = []
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        pids = {int(p) for p in out.strip().splitlines() if p.strip()}
        # Don't kill ourselves
        pids.discard(os.getpid())
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return killed


@ui_app.command("start")
def ui_start(
    port: int = typer.Option(
        _DEFAULT_PORT, "--port", "-p", help="Port to bind (default: 44480)."
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", "-H", help="Host to bind."
    ),
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Run in foreground (don't daemonize)."
    ),
    dev: bool = typer.Option(
        False, "--dev", help="Enable auto-reload for development."
    ),
):
    """🚀 Start the Octane Mission Control web UI."""
    if _read_pid():
        console.print("[yellow]⚠ UI server is already running.[/]")
        return

    # Auto-kill anything already holding the port
    killed = _kill_port_holders(port)
    if killed:
        import time
        console.print(
            f"[yellow]⚠ Killed {len(killed)} process(es) on port {port}: "
            f"{killed}[/]"
        )
        time.sleep(0.5)  # let OS release the socket

    os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "octane.ui.app:app",
        "--host", host,
        "--port", str(port),
    ]
    if dev:
        cmd.append("--reload")

    if foreground:
        console.print(f"[dim]Starting Mission Control on http://{host}:{port} …[/]")
        console.print("[dim]Press Ctrl+C to stop.[/]")
        with open(_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        try:
            os.execvp(sys.executable, cmd)
        finally:
            if os.path.exists(_PID_FILE):
                os.unlink(_PID_FILE)
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        with open(_PID_FILE, "w") as f:
            f.write(str(proc.pid))
        console.print(
            f"[green]● Mission Control started[/] on "
            f"[cyan]http://localhost:{port}[/]  (PID {proc.pid})"
        )
        console.print(
            f"  [dim]Also accessible via http://octane.local:{port}[/]"
        )


@ui_app.command("stop")
def ui_stop():
    """🛑 Stop the Octane Mission Control web UI."""
    pid = _read_pid()
    if not pid:
        console.print("[yellow]⚠ UI server is not running.[/]")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[red]● Mission Control stopped[/] (PID {pid})")
    except ProcessLookupError:
        console.print("[yellow]⚠ Process already gone.[/]")

    if os.path.exists(_PID_FILE):
        os.unlink(_PID_FILE)


@ui_app.command("status")
def ui_status():
    """📊 Show UI server status."""
    pid = _read_pid()
    if pid:
        console.print(
            f"[green]● Mission Control is running[/]  PID {pid}  "
            f"Port {_DEFAULT_PORT}"
        )
    else:
        console.print("[dim]● Mission Control is not running.[/]")
