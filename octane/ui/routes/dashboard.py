"""Dashboard route — aggregated system overview for Mission Control."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter

router = APIRouter(tags=["dashboard"])

_START_TIME = time.monotonic()
_TRACE_DIR = Path.home() / ".octane" / "traces"


def _uptime_str() -> str:
    """Human-readable uptime since UI server started."""
    seconds = int(time.monotonic() - _START_TIME)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _recent_traces(limit: int = 10) -> list[dict[str, Any]]:
    """Read the most recent trace files for the query timeline."""
    if not _TRACE_DIR.is_dir():
        return []
    trace_files = sorted(_TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for tf in trace_files[:limit]:
        import json
        events = []
        for line in tf.read_text().splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not events:
            continue
        first = events[0]
        last = events[-1]
        query = ""
        for e in events:
            if e.get("event_type") == "ingress":
                query = e.get("payload", {}).get("query", "")[:120]
                break
        total_ms = sum(e.get("duration_ms", 0) for e in events)
        success = not any(e.get("error") for e in events)
        results.append({
            "trace_id": tf.stem,
            "query": query,
            "timestamp": first.get("timestamp", ""),
            "duration_ms": round(total_ms, 1),
            "event_count": len(events),
            "success": success,
        })
    return results


@router.get("/dashboard")
async def dashboard() -> dict[str, Any]:
    """Aggregated dashboard data for Mission Control."""
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)

    return {
        "uptime": _uptime_str(),
        "system": {
            "cpu_percent": cpu_pct,
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "ram_used_gb": round(mem.used / (1024**3), 1),
            "ram_percent": mem.percent,
        },
        "recent_traces": _recent_traces(10),
    }
