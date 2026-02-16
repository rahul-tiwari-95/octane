"""Synapse — the observability nervous system.

Every state transition in Octane produces a SynapseEvent.
The SynapseEventBus collects them in-memory (Phase 1).
Phase 4: replace with Redis Stream for real-time streaming.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class SynapseEvent(BaseModel):
    """A single event in the Synapse nervous system.

    These events are the atoms of observability. Every decomposition,
    dispatch, agent start/stop, error, and final output gets one.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(
        description="Ties this event to a specific user query lifecycle"
    )
    event_type: str = Field(
        description="Type: ingress, decomposition, dispatch, agent_start, "
        "agent_complete, agent_error, synthesis, egress"
    )
    source: str = Field(description="Component that emitted this event (e.g., 'osa.decomposer')")
    target: str = Field(default="", description="Target component (e.g., 'web_agent')")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data: query, task DAG, results, errors, etc.",
    )
    error: str = Field(default="", description="Error message if this is an error event")
    duration_ms: float = Field(default=0.0, description="Duration of the operation, if applicable")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SynapseTrace(BaseModel):
    """Complete trace of a query's lifecycle.

    Assembled from all SynapseEvents sharing the same correlation_id.
    """

    correlation_id: str
    events: list[SynapseEvent] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    success: bool = True
    started_at: datetime | None = None
    completed_at: datetime | None = None


class SynapseEventBus:
    """In-memory event bus for Synapse events.

    Phase 1: Simple list storage with query by correlation_id.
    Phase 4: Replace with Redis Stream for real-time event streaming.
    """

    def __init__(self) -> None:
        self._events: list[SynapseEvent] = []

    def emit(self, event: SynapseEvent) -> None:
        """Emit an event to the bus."""
        self._events.append(event)

    def get_trace(self, correlation_id: str) -> SynapseTrace:
        """Assemble a full trace for a given correlation_id."""
        events = [e for e in self._events if e.correlation_id == correlation_id]
        events.sort(key=lambda e: e.timestamp)

        agents_used = list({e.source for e in events} | {e.target for e in events if e.target})

        trace = SynapseTrace(
            correlation_id=correlation_id,
            events=events,
            agents_used=agents_used,
        )

        if events:
            trace.started_at = events[0].timestamp
            trace.completed_at = events[-1].timestamp
            total = (trace.completed_at - trace.started_at).total_seconds() * 1000
            trace.total_duration_ms = round(total, 2)
            trace.success = not any(e.error for e in events)

        return trace

    def get_all_correlation_ids(self) -> list[str]:
        """Return all unique correlation IDs."""
        return list({e.correlation_id for e in self._events})

    def get_recent_traces(self, limit: int = 10) -> list[SynapseTrace]:
        """Return the most recent traces."""
        cids = self.get_all_correlation_ids()
        traces = [self.get_trace(cid) for cid in cids]
        traces.sort(key=lambda t: t.started_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return traces[:limit]

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    @property
    def event_count(self) -> int:
        return len(self._events)
