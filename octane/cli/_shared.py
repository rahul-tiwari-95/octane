"""Shared helpers used across all octane CLI modules."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Pre-compiled ANSI escape code stripper — used when writing to log files.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


# ── PowerFlags — unified config for power commands ────────────────────────────

@dataclass
class PowerFlags:
    """Parsed power flags for investigate/compare/ask commands.

    Attributes:
        deep:         Depth level (None=normal, int=deep with N dimensions).
        sources:      Source types to query (e.g. ["arxiv", "youtube", "web"]).
        cite:         Include inline citations and a Sources section.
        verify:       Add trust-level labels (CONFIRMED/LIKELY/UNVERIFIED).
    """
    deep: int | None = None
    sources: list[str] = field(default_factory=lambda: ["web"])
    cite: bool = False
    verify: bool = False

    @property
    def max_dimensions(self) -> int | None:
        """Return max_dimensions from deep level, or None for default."""
        if self.deep is not None:
            # --deep:N means N dimensions, clamped to 2-16
            return max(2, min(16, self.deep))
        return None

    @property
    def has_extractors(self) -> bool:
        """True if any non-web source is requested."""
        return any(s != "web" for s in self.sources)


def parse_deep_flag(raw: str | None) -> int | None:
    """Parse --deep flag value.

    Accepts: None, "", "8", "12", ":8", ":12"
    Returns: None (not deep) or int (depth level).
    """
    if raw is None:
        return None
    raw = raw.strip().lstrip(":")
    if not raw:
        return 8  # default deep level
    try:
        return int(raw)
    except ValueError:
        return 8


def parse_sources_flag(raw: str | None) -> list[str]:
    """Parse --sources flag value.

    Accepts: None, "arxiv", "arxiv,youtube", "arxiv,youtube,web"
    Returns: List of source type strings.
    """
    if not raw:
        return ["web"]
    sources = [s.strip().lower() for s in raw.split(",") if s.strip()]
    valid = {"web", "arxiv", "youtube"}
    filtered = [s for s in sources if s in valid]
    return filtered if filtered else ["web"]


# ── test_runs tee ─────────────────────────────────────────────────────────────
# Every octane command saves its terminal output (ANSI-stripped) to
# test_runs/<timestamp>_<cmd>.log.  Upload these files to share bug reports.

class _TeeFile:
    """File-like object that writes to stdout (with colour) and a plain log file."""

    def __init__(self, log_path: Path) -> None:
        self._stdout = sys.stdout  # snapshot before Rich's Live can replace it
        self._log = open(log_path, "w", encoding="utf-8")  # noqa: WPS515

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._log.write(_ANSI_RE.sub("", s))  # strip ANSI for readable log
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._log.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()

    def isatty(self) -> bool:
        return hasattr(self._stdout, "isatty") and self._stdout.isatty()


def _build_console() -> Console:
    """Return a Console that tees output to test_runs/ if possible."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "test_runs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cmd_parts = [a for a in sys.argv[1:] if not a.startswith("-")]
        cmd_slug = "_".join(cmd_parts[:3])[:40].replace(" ", "_") if cmd_parts else "octane"
        log_path = log_dir / f"{ts}_{cmd_slug}.log"
        tee = _TeeFile(log_path)
        return Console(file=tee, force_terminal=True, highlight=False)
    except Exception:
        return Console()


console = _build_console()


def _get_synapse():
    """Get or create the global SynapseEventBus."""
    from octane.models.synapse import SynapseEventBus
    if not hasattr(_get_synapse, "_bus"):
        _get_synapse._bus = SynapseEventBus()
    return _get_synapse._bus


async def _try_daemon_route(command: str, payload: dict):
    """Try to route a command through the daemon if it's running.

    Yields string chunks if daemon handles the request.
    Returns without yielding if daemon is not running (caller should fallback).
    """
    from octane.daemon.client import is_daemon_running, DaemonClient

    if not is_daemon_running():
        return

    client = DaemonClient()
    if not await client.connect(timeout=2.0):
        return

    try:
        async for resp in client.stream(command, payload, timeout=600.0):
            status = resp.get("status")
            if status == "stream":
                chunk = resp.get("chunk")
                if chunk:
                    yield chunk
            elif status == "done":
                return
            elif status == "error":
                raise RuntimeError(resp.get("error", "Unknown daemon error"))
    finally:
        await client.close()


def _get_shadow_config() -> tuple[str, str]:
    """Return (shadow_name, redis_url) for the Shadows background worker."""
    from octane.config import settings
    return "octane", settings.redis_url


async def _ensure_shadow_group(shadow_name: str, redis_url: str) -> None:
    """Pre-create the Shadows consumer group, silently ignoring BUSYGROUP.

    Also monkey-patches Shadow.__aenter__ to fix the broken BUSYGROUP check
    in the Shadows library.  In recent redis-py builds, repr(ResponseError)
    returns 'server:ResponseError' (no message text), so the Shadows check
    ``"BUSYGROUP" not in repr(e)`` always re-raises.  We patch it to use
    str(e) instead.
    """
    import redis.asyncio as aioredis
    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.xgroup_create(
            name=f"{shadow_name}:stream",
            groupname="shadows-workers",
            id="0-0",
            mkstream=True,
        )
    except Exception:
        pass  # BUSYGROUP or any other error — group already exists
    finally:
        await client.aclose()

    # Patch Shadows' broken BUSYGROUP check (repr → str)
    _patch_shadow_busygroup()


_shadow_patched = False


def _patch_shadow_busygroup() -> None:
    """Monkey-patch Shadow.__aenter__ to use str(e) instead of repr(e).

    In recent redis-py (>=5.x), repr(ResponseError) returns
    'server:ResponseError' without the message text, so Shadows'
    ``"BUSYGROUP" not in repr(e)`` check always re-raises.
    This patch replaces repr(e) with str(e) so BUSYGROUP is detected.
    """
    global _shadow_patched
    if _shadow_patched:
        return
    _shadow_patched = True

    try:
        import asyncio
        import redis.exceptions
        from shadows import Shadow

        _original_aenter = Shadow.__aenter__

        async def _fixed_aenter(self):
            try:
                return await _original_aenter(self)
            except redis.exceptions.RedisError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists — safe to ignore
                    return self
                raise

        Shadow.__aenter__ = _fixed_aenter
    except ImportError:
        pass  # shadows not installed


def _print_dag_trace(trace, events, dag_nodes, dag_reason: str) -> None:
    """Print a compact DAG execution trace table for --verbose mode."""
    console.print()
    console.print(Panel(
        f"[bold]DAG nodes:[/] {dag_nodes}  ·  [bold]Reasoning:[/] {dag_reason[:120] or 'keyword fallback'}",
        title="[bold dim]⚙ DAG Execution Trace[/]",
        border_style="dim",
    ))

    dispatch_events = [e for e in events if e.event_type in ("dispatch", "egress")]
    if dispatch_events:
        tbl = Table(show_header=True, box=None, padding=(0, 2))
        tbl.add_column("Δt", style="dim", width=9, justify="right")
        tbl.add_column("Step", style="cyan", width=20)
        tbl.add_column("Agent", style="green", width=14)
        tbl.add_column("Detail", style="white")

        t0 = trace.started_at
        for e in dispatch_events:
            offset = f"+{(e.timestamp - t0).total_seconds() * 1000:.0f}ms" if t0 else "—"
            if e.event_type == "dispatch":
                agent = e.target or "—"
                detail = (e.payload or {}).get("instruction", "")[:80]
            else:
                agent = "evaluator"
                agents_used = (e.payload or {}).get("agents_used", [])
                ok = (e.payload or {}).get("tasks_succeeded", "?")
                total = (e.payload or {}).get("tasks_total", "?")
                detail = f"[dim]agents={agents_used} {ok}/{total} succeeded[/]"
            tbl.add_row(offset, e.event_type, agent, detail)
        console.print(tbl)
