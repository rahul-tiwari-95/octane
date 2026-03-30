"""octane health / sysstat commands."""

from __future__ import annotations

import asyncio
import time

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_synapse

# Tight timeout for all Bodega probe calls in health/sysstat.
# When the inference engine is busy (37b running), HTTP requests to Bodega
# queue behind the active inference job and can block for the full generation
# time (30-180s).  3 seconds is enough to confirm "up and idle" vs "busy".
_BODEGA_PROBE_TIMEOUT = 3.0


def register(app: typer.Typer) -> None:
    app.command()(health)
    app.command()(sysstat)


def health():
    """🩺 System health — RAM, CPU, loaded model, server status."""
    asyncio.run(_health())


async def _health():
    from octane.agents.sysstat.monitor import Monitor
    from octane.tools.bodega_inference import BodegaInferenceClient
    from octane.tools.topology import ModelTier, detect_topology, get_topology

    t0 = time.monotonic()

    # ── Phase 1: local metrics (psutil — zero network, instant) ──────────────
    system = Monitor().snapshot()

    ram_used = system.get("ram_used_gb", 0)
    ram_total = system.get("ram_total_gb", 0)
    ram_pct = system.get("ram_percent", 0)
    ram_color = "green" if ram_pct < 70 else "yellow" if ram_pct < 90 else "red"

    sys_table = Table(show_header=False, box=None, padding=(0, 2))
    sys_table.add_column("Metric", style="cyan")
    sys_table.add_column("Value", style="white")
    sys_table.add_row("RAM", f"[{ram_color}]{ram_used:.1f} / {ram_total:.1f} GB ({ram_pct}%)[/]")
    sys_table.add_row("CPU", f"{system.get('cpu_percent', '?')}% ({system.get('cpu_count', '?')} cores)")
    sys_table.add_row("RAM Available", f"{system.get('ram_available_gb', 0):.1f} GB")

    console.print(Panel(sys_table, title="[bold cyan]⚙ System Resources[/]", border_style="cyan"))

    # ── Phase 2: Bodega probe (3 s hard timeout) ──────────────────────────────
    bodega = BodegaInferenceClient(timeout=_BODEGA_PROBE_TIMEOUT)
    model: dict = {}
    server_health: dict = {}
    bodega_status = "ok"

    try:
        model, server_health = await asyncio.wait_for(
            asyncio.gather(bodega.current_model(), bodega.health()),
            timeout=_BODEGA_PROBE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        bodega_status = "busy"
        model = {"error": "inference in progress — try again when generation completes"}
        server_health = {"status": "busy"}
    except Exception as exc:
        bodega_status = "offline"
        model = {"error": str(exc)}
        server_health = {"status": "offline"}
    finally:
        await bodega.close()

    # Reflect actual vLLM status (e.g. "unhealthy" = server up, no models)
    if bodega_status == "ok" and server_health.get("status") not in ("ok", None):
        bodega_status = "no_models"

    bodega_table = Table(show_header=False, box=None, padding=(0, 2))
    bodega_table.add_column("Metric", style="magenta")
    bodega_table.add_column("Value", style="white")

    _STATUS_DISPLAY = {
        "ok":         "✅ running",
        "no_models":  "🟡 online (no models loaded)",
        "busy":       "⏳ busy (inference running)",
        "offline":    "🔴 offline",
    }
    bodega_table.add_row("Server", _STATUS_DISPLAY.get(bodega_status, bodega_status))

    if "error" in model:
        color = "yellow" if bodega_status == "busy" else "red"
        bodega_table.add_row("Models", f"[{color}]{model['error']}[/]")
    elif not model.get("loaded"):
        bodega_table.add_row("Models", "[yellow]no model loaded[/]")
    else:
        all_models = model.get("all_models", [model.get("model_path", "unknown")])
        for m in all_models:
            bodega_table.add_row("Loaded", f"[green]{m}[/]")

    console.print(Panel(bodega_table, title="[bold magenta]🧠 Bodega Inference Engine[/]", border_style="magenta"))

    # ── Phase 3: topology (local config — no network) ─────────────────────────
    topo_name = detect_topology()
    topo = get_topology(topo_name)
    loaded_ids = set(model.get("all_models", []))

    topo_table = Table(show_header=False, box=None, padding=(0, 2))
    topo_table.add_column("Tier", style="yellow", width=8)
    topo_table.add_column("Model", style="white")
    topo_table.add_column("Status", style="dim")

    _TIER_LABELS = {
        ModelTier.FAST:   "FAST  ",
        ModelTier.MID:    "MID   ",
        ModelTier.REASON: "REASON",
    }
    _TIER_ROLES = {
        ModelTier.FAST:   "routing · classification · extraction",
        ModelTier.MID:    "chunk summarisation · mid-depth reasoning",
        ModelTier.REASON: "full synthesis · evaluation",
    }
    for tier in (ModelTier.FAST, ModelTier.MID, ModelTier.REASON):
        cfg = topo.resolve_config(tier)
        is_loaded = any(
            cfg.model_id.lower() in m.lower() or cfg.model_path.lower() in m.lower()
            for m in loaded_ids
        ) if loaded_ids else False

        if bodega_status == "busy" and not loaded_ids:
            status = "[yellow]● inference running[/]"
        elif is_loaded:
            status = "[green]● loaded[/]"
        else:
            status = "[dim]○ will auto-load[/]"

        model_line = f"[bold]{cfg.model_id}[/]  [dim]{_TIER_ROLES[tier]}[/]"
        topo_table.add_row(_TIER_LABELS[tier], model_line, status)

    console.print(Panel(
        topo_table,
        title=f"[bold yellow]⚡ Octane Topology — [cyan]{topo_name}[/cyan][/]",
        border_style="yellow",
    ))

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    console.print(f"\n[dim]Rendered in {elapsed_ms}ms[/]")


def sysstat():
    """📊 Live system snapshot — RAM, CPU, loaded model."""
    asyncio.run(_sysstat())


async def _sysstat():
    from octane.agents.sysstat.monitor import Monitor
    from octane.tools.bodega_inference import BodegaInferenceClient

    t0 = time.monotonic()

    # ── Phase 1: psutil (instant, no network) ────────────────────────────────
    system = Monitor().snapshot()

    ram_used = system.get("ram_used_gb", 0)
    ram_total = system.get("ram_total_gb", 0)
    ram_pct = system.get("ram_percent", 0)
    ram_color = "green" if ram_pct < 70 else "yellow" if ram_pct < 90 else "red"

    sys_tbl = Table(show_header=False, box=None, padding=(0, 2))
    sys_tbl.add_column("Metric", style="cyan")
    sys_tbl.add_column("Value", style="white")
    sys_tbl.add_row("RAM", f"[{ram_color}]{ram_used:.1f} / {ram_total:.1f} GB ({ram_pct}%)[/]")
    sys_tbl.add_row("CPU", f"{system.get('cpu_percent', '?')}% ({system.get('cpu_count', '?')} cores)")
    sys_tbl.add_row("Available", f"{system.get('ram_available_gb', 0):.1f} GB free")

    console.print(Panel(sys_tbl, title="[bold cyan]💻 System[/]", border_style="cyan"))

    # ── Phase 2: Bodega status (3 s timeout) ─────────────────────────────────
    bodega = BodegaInferenceClient(timeout=_BODEGA_PROBE_TIMEOUT)
    mod_tbl = Table(show_header=False, box=None, padding=(0, 2))
    mod_tbl.add_column("Key", style="magenta")
    mod_tbl.add_column("Value", style="white")

    try:
        model = await asyncio.wait_for(
            bodega.current_model(),
            timeout=_BODEGA_PROBE_TIMEOUT,
        )
        if "error" in model:
            mod_tbl.add_row("Status", f"[yellow]⚠ {model['error']}[/]")
        elif not model.get("loaded"):
            mod_tbl.add_row("Status", "[yellow]no model loaded[/]")
        else:
            all_models = model.get("all_models", [model.get("model_path", "?")])
            for m in all_models:
                mod_tbl.add_row("Loaded", f"[green]{m}[/]")
            if model.get("context_length"):
                mod_tbl.add_row("Context", f"{model['context_length']:,} tokens")
    except asyncio.TimeoutError:
        mod_tbl.add_row("Status", "[yellow]⏳ busy — inference in progress[/]")
    except Exception as exc:
        mod_tbl.add_row("Status", f"[red]offline — {exc}[/]")
    finally:
        await bodega.close()

    console.print(Panel(mod_tbl, title="[bold magenta]🧠 Model[/]", border_style="magenta"))

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    console.print(f"[dim]{elapsed_ms}ms[/]")
