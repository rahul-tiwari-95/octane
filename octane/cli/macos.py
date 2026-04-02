"""octane/cli/macos.py — macOS native integration commands.

Commands:
  octane macos setup           — Check permissions, create folders
  octane macos status          — Show permission & shadow status
  octane macos imessage on     — Enable iMessage shadow
  octane macos imessage off    — Disable iMessage shadow
  octane macos imessage send   — Send a one-shot iMessage
  octane macos notes create    — Create an Apple Note
  octane macos calendar        — Show upcoming events
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer

from octane.cli._shared import console, err_console

macos_app = typer.Typer(help="macOS native integration — iMessage, Notes, Calendar.")
imessage_app = typer.Typer(help="iMessage integration.")
macos_app.add_typer(imessage_app, name="imessage")


# ── Setup & Status ────────────────────────────────────────────────────────────

@macos_app.command("setup")
def macos_setup():
    """Check macOS permissions and create required folders."""
    asyncio.run(_macos_setup())


async def _macos_setup():
    from octane.macos.permissions import (
        check_all_permissions,
        permission_guidance,
        IS_MACOS,
    )

    if not IS_MACOS:
        err_console.print("[red]This command requires macOS.[/red]")
        raise typer.Exit(1)

    console.print("[bold cyan]Octane macOS Setup[/bold cyan]")
    console.print()

    perms = await check_all_permissions()

    from rich.table import Table

    table = Table(title="Permission Status", show_lines=False)
    table.add_column("Permission", style="bold")
    table.add_column("Status")

    for name, granted in perms.items():
        if name == "is_macos":
            continue
        icon = "✅" if granted else "❌"
        label = name.replace("_", " ").title()
        table.add_row(label, icon)

    console.print(table)

    missing = [k for k, v in perms.items() if k != "is_macos" and not v]
    if missing:
        console.print()
        console.print("[yellow]Some permissions are missing:[/yellow]")
        console.print()
        console.print(permission_guidance())
    else:
        console.print()
        console.print("[green]All permissions granted! ✅[/green]")

    # Create Octane data directories
    from pathlib import Path

    octane_dir = Path.home() / ".octane"
    for sub in ["macos", "macos/imessage"]:
        d = octane_dir / sub
        d.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Config directory: {octane_dir / 'macos'}[/dim]")


@macos_app.command("status")
def macos_status():
    """Show macOS integration status."""
    asyncio.run(_macos_status())


async def _macos_status():
    from octane.macos.permissions import check_all_permissions, IS_MACOS
    from pathlib import Path
    import json as _json

    if not IS_MACOS:
        err_console.print("[red]This command requires macOS.[/red]")
        raise typer.Exit(1)

    perms = await check_all_permissions()

    console.print("[bold cyan]macOS Integration Status[/bold cyan]")
    console.print()

    # Permissions
    for name, granted in perms.items():
        if name == "is_macos":
            continue
        icon = "✅" if granted else "❌"
        label = name.replace("_", " ").title()
        console.print(f"  {icon} {label}")

    # iMessage shadow status
    console.print()
    config_file = Path.home() / ".octane" / "macos" / "imessage" / "config.json"
    if config_file.exists():
        cfg = _json.loads(config_file.read_text())
        enabled = cfg.get("enabled", False)
        contacts = cfg.get("approved_contacts", [])
        console.print(
            f"  iMessage Shadow: {'[green]enabled[/green]' if enabled else '[dim]disabled[/dim]'}"
        )
        if contacts:
            console.print(f"  Approved contacts: {', '.join(contacts)}")
    else:
        console.print("  iMessage Shadow: [dim]not configured[/dim]")


# ── iMessage ──────────────────────────────────────────────────────────────────

@imessage_app.command("on")
def imessage_on(
    contacts: list[str] = typer.Option(
        ..., "--contact", "-c",
        help="Approved phone/email. Can specify multiple.",
    ),
    self_mode: bool = typer.Option(
        False, "--self",
        help="Also pick up messages you send (iPhone → Mac self-messaging).",
    ),
):
    """Enable iMessage shadow monitor.

    Examples:
        octane macos imessage on -c "+15551234567"
        octane macos imessage on -c "+15551234567" --self   # respond to your own messages from iPhone
    """
    import json as _json
    from pathlib import Path

    config_dir = Path.home() / ".octane" / "macos" / "imessage"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    config = {
        "enabled": True,
        "approved_contacts": contacts,
        "poll_interval": 5.0,
        "allow_self": self_mode,
    }
    config_file.write_text(_json.dumps(config, indent=2))

    console.print("[green]iMessage shadow enabled[/green]")
    console.print(f"  Approved contacts: {', '.join(contacts)}")
    if self_mode:
        console.print("  [cyan]Self-mode: ON[/cyan] — messages you send from iPhone will be picked up")
    console.print()
    console.print("[dim]Test now:    octane macos imessage watch     (runs in foreground)[/dim]")
    console.print("[dim]Production:  octane daemon start              (shadow runs inside daemon)[/dim]")


@imessage_app.command("off")
def imessage_off():
    """Disable iMessage shadow monitor."""
    import json as _json
    from pathlib import Path

    config_file = Path.home() / ".octane" / "macos" / "imessage" / "config.json"
    if config_file.exists():
        config = _json.loads(config_file.read_text())
        config["enabled"] = False
        config_file.write_text(_json.dumps(config, indent=2))

    console.print("[yellow]iMessage shadow disabled[/yellow]")


@imessage_app.command("watch")
def imessage_watch():
    """Run iMessage shadow in the foreground (for testing).

    Reads config from 'octane macos imessage on'. Ctrl-C to stop.
    """
    asyncio.run(_imessage_watch())


async def _imessage_watch():
    import json as _json
    import logging as _logging
    from pathlib import Path
    from octane.macos.imessage_shadow import IMessageShadow

    # Enable console logging so all logger calls are visible
    _logging.basicConfig(level=_logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _logging.getLogger("octane").setLevel(_logging.DEBUG)

    config_file = Path.home() / ".octane" / "macos" / "imessage" / "config.json"
    if not config_file.exists():
        err_console.print("[red]Not configured. Run: octane macos imessage on -c '+1...' --self[/red]")
        raise typer.Exit(1)

    cfg = _json.loads(config_file.read_text())
    if not cfg.get("enabled"):
        err_console.print("[red]Shadow disabled. Run: octane macos imessage on -c '+1...' --self[/red]")
        raise typer.Exit(1)

    contacts = cfg.get("approved_contacts", [])
    allow_self = cfg.get("allow_self", False)
    poll_interval = cfg.get("poll_interval", 5.0)

    if not contacts:
        err_console.print("[red]No approved contacts configured.[/red]")
        raise typer.Exit(1)

    shadow = IMessageShadow(
        approved_contacts=contacts,
        allow_self=allow_self,
        poll_interval=poll_interval,
    )

    console.print("[bold cyan]iMessage Shadow — Foreground Mode[/bold cyan]")
    console.print(f"  Contacts: {', '.join(contacts)}")
    console.print(f"  Self-mode: {'ON' if allow_self else 'OFF'}")
    console.print(f"  Poll interval: {poll_interval}s")
    console.print("[dim]Press Ctrl-C to stop[/dim]")
    console.print()

    try:
        await shadow.start()
        # Block until interrupted
        while shadow.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await shadow.stop()
        console.print("\n[yellow]iMessage shadow stopped.[/yellow]")


@imessage_app.command("send")
def imessage_send(
    message: str = typer.Argument(None, help="Message text. Use --stdin to read from pipe."),
    to: str = typer.Option(..., "--to", help="Recipient phone number or Apple ID."),
    stdin: bool = typer.Option(False, "--stdin", help="Read message from stdin."),
):
    """Send a one-shot iMessage.

    Examples:
        octane macos imessage send "Hello from Octane" --to "+15551234567"
        echo "Report ready" | octane macos imessage send --stdin --to "+15551234567"
        octane search news "AI" --json | octane synthesize run --stdin | octane macos imessage send --stdin --to "+15551234567"
    """
    asyncio.run(_imessage_send(message, to, stdin))


async def _imessage_send(message: str | None, to: str, use_stdin: bool):
    from octane.macos.applescript import AppleScriptBridge

    if use_stdin:
        if sys.stdin.isatty():
            err_console.print("[red]No stdin data. Pipe content or use a message argument.[/red]")
            raise typer.Exit(1)
        message = sys.stdin.read().strip()
    elif not message:
        err_console.print("[red]Provide a message or use --stdin.[/red]")
        raise typer.Exit(1)

    if not message:
        err_console.print("[red]Empty message.[/red]")
        raise typer.Exit(1)

    bridge = AppleScriptBridge()
    ok, status = await bridge.send_imessage(to, message)

    if ok:
        console.print(f"[green]✓ iMessage sent to {to}[/green] ({len(message)} chars)")
    else:
        err_console.print(f"[red]Failed: {status}[/red]")
        raise typer.Exit(1)


# ── Notes ─────────────────────────────────────────────────────────────────────

@macos_app.command("notes")
def notes_create(
    title: str = typer.Argument(..., help="Note title."),
    body: str = typer.Option("", "--body", "-b", help="Note body text."),
    folder: str = typer.Option("Octane", "--folder", "-f", help="Notes folder."),
    stdin: bool = typer.Option(False, "--stdin", help="Read body from stdin."),
):
    """Create an Apple Note.

    Examples:
        octane macos notes "Research Summary" --body "NVDA beat earnings..."
        octane synthesize run --stdin | octane macos notes "Daily Briefing" --stdin
    """
    asyncio.run(_notes_create(title, body, folder, stdin))


async def _notes_create(title: str, body: str, folder: str, use_stdin: bool):
    from octane.macos.applescript import AppleScriptBridge

    if use_stdin:
        if sys.stdin.isatty():
            err_console.print("[red]No stdin data.[/red]")
            raise typer.Exit(1)
        body = sys.stdin.read().strip()

    if not body:
        body = "(empty note)"

    bridge = AppleScriptBridge()
    ok, status = await bridge.create_note(title, body, folder)

    if ok:
        console.print(f"[green]✓ Note created: {title}[/green] (folder: {folder})")
    else:
        err_console.print(f"[red]Failed: {status}[/red]")
        raise typer.Exit(1)


# ── Calendar ──────────────────────────────────────────────────────────────────

@macos_app.command("calendar")
def calendar_read(
    hours: int = typer.Option(24, "--hours", "-h", help="Hours ahead to look."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
):
    """Show upcoming calendar events."""
    asyncio.run(_calendar_read(hours, output_json))


async def _calendar_read(hours: int, output_json: bool):
    from octane.macos.applescript import AppleScriptBridge

    bridge = AppleScriptBridge()
    events = await bridge.read_calendar(hours)

    if output_json:
        print(json.dumps({"events": events, "hours_ahead": hours}, indent=2))
        return

    if not events:
        console.print("[dim]No upcoming events in the next {hours}h.[/dim]")
        return

    from rich.table import Table

    table = Table(title=f"Calendar — next {hours}h", show_lines=False)
    table.add_column("Event", style="bold")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Location")
    table.add_column("Calendar", style="dim")

    for e in events:
        table.add_row(
            e["summary"],
            e["start_date"],
            e["end_date"],
            e["location"] or "—",
            e["calendar"],
        )

    console.print(table)
