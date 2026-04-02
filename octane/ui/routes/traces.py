"""Traces route — Synapse event traces for query lifecycle visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["traces"])

_TRACE_DIR = Path.home() / ".octane" / "traces"


def _load_trace(trace_id: str) -> list[dict[str, Any]]:
    """Load all events from a trace file."""
    # Sanitize trace_id to prevent path traversal
    safe_id = Path(trace_id).name
    trace_file = _TRACE_DIR / f"{safe_id}.jsonl"
    if not trace_file.is_file():
        # Try partial match
        matches = list(_TRACE_DIR.glob(f"{safe_id}*.jsonl"))
        if not matches:
            return []
        trace_file = matches[0]
    events = []
    for line in trace_file.read_text().splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


@router.get("/traces")
async def list_traces(limit: int = 50) -> dict[str, Any]:
    """List recent traces with summary info."""
    if not _TRACE_DIR.is_dir():
        return {"traces": [], "total": 0}
    trace_files = sorted(
        _TRACE_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    results = []
    for tf in trace_files[:limit]:
        events = _load_trace(tf.stem)
        if not events:
            continue
        query = ""
        agents = set()
        for e in events:
            if e.get("event_type") == "ingress":
                query = e.get("payload", {}).get("query", "")[:120]
            if e.get("event_type") == "agent_complete":
                agents.add(e.get("source", ""))
        total_ms = sum(e.get("duration_ms", 0) for e in events)
        results.append({
            "trace_id": tf.stem,
            "query": query,
            "timestamp": events[0].get("timestamp", ""),
            "duration_ms": round(total_ms, 1),
            "event_count": len(events),
            "agents": sorted(agents),
            "success": not any(e.get("error") for e in events),
        })
    return {"traces": results, "total": len(results)}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Get full event waterfall for a specific trace."""
    events = _load_trace(trace_id)
    if not events:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "trace_id": trace_id,
        "events": events,
        "event_count": len(events),
    }


@router.get("/traces-events/recent")
async def recent_events(limit: int = 200) -> dict[str, Any]:
    """Return flattened recent Synapse events for globe seeding.

    Reads the N most recent trace files and returns all their events,
    sorted by timestamp descending, capped at *limit*.
    """
    if not _TRACE_DIR.is_dir():
        return {"events": [], "total": 0}
    trace_files = sorted(
        _TRACE_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    all_events: list[dict[str, Any]] = []
    for tf in trace_files[:30]:  # read up to 30 recent traces
        events = _load_trace(tf.stem)
        all_events.extend(events)
        if len(all_events) >= limit:
            break
    # Sort by timestamp descending, cap
    all_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    capped = all_events[:limit]
    return {"events": capped, "total": len(capped)}
