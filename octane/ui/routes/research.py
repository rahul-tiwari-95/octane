"""Research route — active research tasks, findings, shadows status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["research"])

_OCTANE_DIR = Path.home() / ".octane"


def _read_research_state() -> list[dict[str, Any]]:
    """Read research task state from the state file."""
    state_file = _OCTANE_DIR / "research_tasks.json"
    if not state_file.is_file():
        return []
    try:
        data = json.loads(state_file.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
    except (json.JSONDecodeError, OSError):
        pass
    return []


@router.get("/research/tasks")
async def list_research_tasks() -> dict[str, Any]:
    """List active and recent research tasks."""
    tasks = _read_research_state()
    return {
        "tasks": tasks,
        "total": len(tasks),
        "active": sum(1 for t in tasks if t.get("status") in ("running", "pending")),
    }


@router.get("/research/findings")
async def list_findings(topic: str = "", limit: int = 50) -> dict[str, Any]:
    """List research findings, optionally filtered by topic."""
    findings_dir = _OCTANE_DIR / "findings"
    if not findings_dir.is_dir():
        return {"findings": [], "total": 0}
    results = []
    for f in sorted(findings_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            if topic and topic.lower() not in json.dumps(data).lower():
                continue
            results.append(data)
            if len(results) >= limit:
                break
        except (json.JSONDecodeError, OSError):
            continue
    return {"findings": results, "total": len(results)}
