"""Synapse — the observability nervous system.

Every state transition in Octane produces a SynapseEvent.
The SynapseEventBus collects them in-memory AND persists to
~/.octane/traces/<correlation_id>.jsonl so octane trace works
across processes.

Phase 4: replace file persistence with Redis Stream.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Default trace directory — created on first write
_TRACE_DIR = Path.home() / ".octane" / "traces"


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
    """In-memory event bus with file-based persistence.

    Events are stored in memory for fast in-process access AND
    appended to ~/.octane/traces/<correlation_id>.jsonl so they
    survive process restarts and octane trace works cross-process.

    Phase 4: Replace file persistence with Redis Stream.
    """

    def __init__(self, trace_dir: Path | None = None, persist: bool = True) -> None:
        self._events: list[SynapseEvent] = []
        self._trace_dir = trace_dir or _TRACE_DIR
        self._persist = persist

    def emit(self, event: SynapseEvent) -> None:
        """Emit an event — stores in memory and appends to trace file."""
        self._events.append(event)
        if self._persist:
            self._write_to_file(event)

    def _write_to_file(self, event: SynapseEvent) -> None:
        """Append event as a JSON line to the correlation_id trace file."""
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            trace_file = self._trace_dir / f"{event.correlation_id}.jsonl"
            with trace_file.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception:
            # Never crash the pipeline because of tracing
            pass

    def get_trace(self, correlation_id: str) -> SynapseTrace:
        """Assemble a full trace — checks memory first, then disk."""
        events = [e for e in self._events if e.correlation_id == correlation_id]

        # If not in memory (different process), load from disk
        if not events:
            events = self._load_from_file(correlation_id)

        events.sort(key=lambda e: e.timestamp)
        agents_used = list(
            {e.source for e in events} | {e.target for e in events if e.target}
        )

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

    def _load_from_file(self, correlation_id: str) -> list[SynapseEvent]:
        """Load events from a persisted JSONL trace file."""
        trace_file = self._trace_dir / f"{correlation_id}.jsonl"
        if not trace_file.exists():
            return []
        events = []
        try:
            with trace_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(SynapseEvent.model_validate_json(line))
        except Exception:
            pass
        return events

    def list_traces(self, limit: int = 20) -> list[str]:
        """List recent correlation IDs from persisted trace files (newest first)."""
        if not self._trace_dir.exists():
            return []
        files = sorted(
            self._trace_dir.glob("*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return [f.stem for f in files[:limit]]

    def get_all_correlation_ids(self) -> list[str]:
        """Return all unique correlation IDs (in-memory)."""
        return list({e.correlation_id for e in self._events})

    def get_recent_traces(self, limit: int = 10) -> list[SynapseTrace]:
        """Return the most recent in-memory traces."""
        cids = self.get_all_correlation_ids()
        traces = [self.get_trace(cid) for cid in cids]
        traces.sort(
            key=lambda t: t.started_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return traces[:limit]

    def clear(self) -> None:
        """Clear in-memory events (does not delete files)."""
        self._events.clear()

    @property
    def event_count(self) -> int:
        return len(self._events)
