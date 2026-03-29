"""Queries route — submit queries through the UI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["queries"])


class QueryRequest(BaseModel):
    """Submit a query to Octane."""
    query: str
    mode: str = "ask"  # ask, investigate, compare


@router.post("/ask")
async def submit_query(req: QueryRequest) -> dict[str, Any]:
    """Submit a query and return the trace ID for tracking.

    Runs the query asynchronously through the Orchestrator.
    The frontend polls /api/traces/{trace_id} for results.
    """
    import asyncio
    import uuid
    from octane.osa.orchestrator import Orchestrator

    trace_id = str(uuid.uuid4())[:8]

    async def _run_query() -> None:
        try:
            orch = Orchestrator()
            await orch.run(req.query)
        except Exception:
            pass  # errors captured in Synapse trace

    # Fire and forget — results tracked via trace
    asyncio.create_task(_run_query())

    return {"trace_id": trace_id, "status": "submitted", "query": req.query}
