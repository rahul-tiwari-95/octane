"""octane vault sub-app — manage hardware-backed encrypted vaults."""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console

vault_app = typer.Typer(
    name="vault",
    help="🔐 Manage Touch ID-protected encrypted vaults.",
    no_args_is_help=True,
)

_VAULT_DESCRIPTIONS = {
    "finance":  "Portfolio positions, broker tokens, trade history",
    "health":   "Biomarkers, WHOOP/Oura exports, blood work",
    "research": "Confidential findings and research notes",
    "code":     "API keys, tokens, credentials",
}


def _get_vm():
    from octane.security.vault import VaultManager, VaultSetupError
    vm = VaultManager()
    if not vm.auth_available():
        console.print(
            "[yellow]⚠  octane-auth binary not found.[/]\n"
            "  Build it: [dim]cd octane-auth && bash build.sh[/]\n"
            "  Vault operations require Touch ID. Continuing in setup mode only."
        )
    return vm


@vault_app.command("create")
def vault_create(
    name: str = typer.Argument(..., help="Vault name (finance|health|research|code)"),
):
    """🔒 Create a new vault. Stores AES key in Keychain with Touch ID protection."""
    from octane.security.vault import VaultError, VaultSetupError

    vm = _get_vm()
    try:
        vm.create(name)
        desc = _VAULT_DESCRIPTIONS.get(name, "Custom vault")
        console.print(Panel(
            f"[green]✓[/] Vault [bold]{name}[/] created.\n"
            f"[dim]{desc}[/]\n\n"
            f"Key stored in macOS Keychain (Touch ID protected).\n"
            f"Vault file: [dim]~/.octane/vaults/{name}.enc[/]",
            title="Vault Created",
            border_style="green",
        ))
    except VaultSetupError as e:
        console.print(f"[red]Setup required:[/] {e}")
        raise typer.Exit(1)
    except VaultError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@vault_app.command("status")
def vault_status():
    """📊 Show all vaults — name, size, Touch ID status."""
    from octane.security.vault import VaultManager, VaultSetupError

    vm = VaultManager()
    vaults = vm.status()

    if not vaults:
        console.print("[dim]No vaults found. Create one: [/][bold]octane vault create finance[/]")
        return

    table = Table(title="Octane Vaults", show_header=True, header_style="bold cyan")
    table.add_column("Name",      style="bold")
    table.add_column("Description")
    table.add_column("Size",      justify="right")
    table.add_column("Touch ID",  justify="center")
    table.add_column("Location",  style="dim")

    for v in vaults:
        name = v["name"]
        size = v["size_bytes"]
        size_str = f"{size} B" if size < 1024 else f"{size // 1024} KB"
        touchid = "🔐 Yes" if v["keychain_protected"] else "⚠  Binary missing"
        desc = _VAULT_DESCRIPTIONS.get(name, "Custom vault")
        table.add_row(name, desc, size_str, touchid, f"~/.octane/vaults/{name}.enc")

    console.print(table)


@vault_app.command("write")
def vault_write(
    vault: str  = typer.Argument(..., help="Vault name"),
    key: str    = typer.Argument(..., help="Secret key name"),
    value: str  = typer.Argument(..., help="Secret value"),
):
    """✏️  Write a secret to a vault. Triggers Touch ID."""
    from octane.security.vault import VaultError, VaultNotFoundError, VaultLockedError, VaultSetupError

    vm = _get_vm()
    try:
        vm.write(vault, key, value)
        console.print(f"[green]✓[/] Wrote [bold]{key}[/] to [bold]{vault}[/] vault.")
    except VaultLockedError:
        console.print("[red]✗[/] Touch ID authentication failed or cancelled.")
        raise typer.Exit(1)
    except (VaultNotFoundError, VaultSetupError, VaultError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@vault_app.command("read")
def vault_read(
    vault: str = typer.Argument(..., help="Vault name"),
    key: str   = typer.Argument(..., help="Secret key name"),
):
    """🔍 Read a secret from a vault. Triggers Touch ID."""
    from octane.security.vault import VaultError, VaultNotFoundError, VaultLockedError, VaultSetupError

    vm = _get_vm()
    try:
        value = vm.read(vault, key)
        if value is None:
            console.print(f"[yellow]Key '{key}' not found in {vault} vault.[/]")
            raise typer.Exit(1)
        # Print to stdout (raw) so it can be piped
        import sys
        sys.stdout.write(value + "\n")
    except VaultLockedError:
        console.print("[red]✗[/] Touch ID authentication failed or cancelled.")
        raise typer.Exit(1)
    except (VaultNotFoundError, VaultSetupError, VaultError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@vault_app.command("list-keys")
def vault_list_keys(
    vault: str = typer.Argument(..., help="Vault name"),
):
    """📋 List all keys stored in a vault. Triggers Touch ID."""
    from octane.security.vault import VaultError, VaultNotFoundError, VaultLockedError, VaultSetupError

    vm = _get_vm()
    try:
        data = vm.read_all(vault)
        if not data:
            console.print(f"[dim]{vault} vault is empty.[/]")
            return
        table = Table(title=f"{vault} vault", show_header=True, header_style="bold cyan")
        table.add_column("Key", style="bold")
        table.add_column("Value (masked)")
        for k, v in sorted(data.items()):
            masked = v[:4] + "•" * (len(v) - 4) if len(v) > 4 else "••••"
            table.add_row(k, masked)
        console.print(table)
    except VaultLockedError:
        console.print("[red]✗[/] Touch ID authentication failed or cancelled.")
        raise typer.Exit(1)
    except (VaultNotFoundError, VaultSetupError, VaultError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@vault_app.command("destroy")
def vault_destroy(
    name: str = typer.Argument(..., help="Vault name to permanently delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """💥 Permanently destroy a vault and all its secrets. Touch ID required."""
    from octane.security.vault import VaultError, VaultNotFoundError, VaultLockedError, VaultSetupError

    if not yes:
        confirmed = typer.confirm(
            f"[red]This will permanently delete the '{name}' vault and ALL its secrets. "
            "This cannot be undone. Continue?[/]"
        )
        if not confirmed:
            console.print("[dim]Cancelled.[/]")
            return

    vm = _get_vm()
    try:
        vm.destroy(name)
        console.print(f"[red]✓[/] Vault [bold]{name}[/] destroyed. Key removed from Keychain.")
    except VaultLockedError:
        console.print("[red]✗[/] Touch ID authentication failed or cancelled.")
        raise typer.Exit(1)
    except (VaultNotFoundError, VaultSetupError, VaultError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    pass  # vault_app added as sub-app in cli/__init__.py
