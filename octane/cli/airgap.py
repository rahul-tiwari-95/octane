"""octane airgap sub-app — network kill switch."""

from __future__ import annotations

import typer
from rich.panel import Panel

from octane.cli._shared import console

airgap_app = typer.Typer(
    name="airgap",
    help="🛡️  Control network access (air-gap mode).",
    no_args_is_help=True,
)


@airgap_app.command("on")
def airgap_on(
    reason: str = typer.Option("", "--reason", "-r", help="Why you're enabling airgap"),
):
    """🔴 Enable airgap mode — disable all outbound network access."""
    from octane.security.airgap import AirgapManager

    mgr = AirgapManager()
    meta = mgr.enable(reason=reason)
    console.print(Panel(
        f"[red bold]● AIRGAP ON[/]\n\n"
        f"All outbound network access blocked.\n"
        f"  • Web searches: [red]BLOCKED[/]\n"
        f"  • Bodega Intel API: [red]BLOCKED[/]\n"
        f"  • Finance data: [red]BLOCKED[/]\n\n"
        f"Octane works with local models and cached Postgres data only.\n\n"
        f"[dim]Reason: {meta.get('reason') or 'Not specified'}[/]\n"
        f"[dim]Run [bold]octane airgap off[/dim] to restore network access.",
        title="Air-gap Mode Enabled",
        border_style="red",
    ))


@airgap_app.command("off")
def airgap_off():
    """🟢 Disable airgap mode — restore network access."""
    from octane.security.airgap import AirgapManager

    mgr = AirgapManager()

    if not mgr.status().get("active"):
        console.print("[dim]Airgap is not currently active.[/]")
        return

    mgr.disable()
    console.print(Panel(
        "[green bold]● NETWORK RESTORED[/]\n\n"
        "Airgap mode disabled. All network access re-enabled.\n"
        "  • Web searches: [green]ACTIVE[/]\n"
        "  • Bodega Intel API: [green]ACTIVE[/]\n"
        "  • Finance data: [green]ACTIVE[/]",
        title="Air-gap Mode Disabled",
        border_style="green",
    ))


@airgap_app.command("status")
def airgap_status():
    """📊 Show current airgap status."""
    from octane.security.airgap import AirgapManager

    mgr = AirgapManager()
    s = mgr.status()

    if s.get("active"):
        console.print(Panel(
            f"[red bold]● AIRGAP ACTIVE[/]\n\n"
            f"Enabled at : {s.get('enabled_at', 'unknown')}\n"
            f"Reason     : {s.get('reason') or 'Not specified'}\n"
            f"Enabled by : {s.get('enabled_by', 'unknown')}\n\n"
            f"[dim]Run [bold]octane airgap off[/dim] to restore network access.",
            title="Air-gap Status",
            border_style="red",
        ))
    else:
        console.print(Panel(
            "[green bold]● NETWORK ACTIVE[/]\n\nAll outbound connections permitted.",
            title="Air-gap Status",
            border_style="green",
        ))


def register(app: typer.Typer) -> None:
    pass  # airgap_app added as sub-app in cli/__init__.py
