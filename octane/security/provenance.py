"""Provenance tracking — chain of custody for every stored finding.

Every row written by Octane to Postgres carries a JSONB provenance column
that records *how* and *when* the data was produced. This lets users run
`octane audit <id>` and see the complete origin chain.

Structure:
    {
        "source":     "bodega_intel_api" | "web_scrape" | "user_import" | ...,
        "command":    "octane investigate" | "octane ask --deep" | ...,
        "trace_id":   "abc123...",       # Synapse correlation ID
        "model":      "qwen-0.9b",       # LLM model used (nullable)
        "agent":      "web",             # agent that produced the data
        "airgap":     false,             # was airgap active during collection?
        "timestamp":  "2026-03-22T...",  # UTC ISO-8601
        "session_id": "...",             # optional Octane session
        "depth":      1                  # research round / depth level
    }

Usage:
    prov = build_provenance(
        source="bodega_intel_api",
        command="octane investigate",
        trace_id=correlation_id,
        agent="web",
    )
    # prov is a dict — store as JSONB in Postgres.

    # Retrieve audit trail:
    record = ProvenanceRecord.from_dict(prov)
    print(record.format())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from octane.security.airgap import is_airgap_active


@dataclass
class ProvenanceRecord:
    """Structured provenance record for a stored data item."""

    source: str
    command: str
    timestamp: str
    airgap: bool
    trace_id: str = ""
    model: str = ""
    agent: str = ""
    session_id: str = ""
    depth: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceRecord":
        known = {"source", "command", "timestamp", "airgap", "trace_id",
                 "model", "agent", "session_id", "depth"}
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            source=d.get("source", ""),
            command=d.get("command", ""),
            timestamp=d.get("timestamp", ""),
            airgap=d.get("airgap", False),
            trace_id=d.get("trace_id", ""),
            model=d.get("model", ""),
            agent=d.get("agent", ""),
            session_id=d.get("session_id", ""),
            depth=d.get("depth", 0),
            extras=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "command": self.command,
            "timestamp": self.timestamp,
            "airgap": self.airgap,
        }
        if self.trace_id:
            d["trace_id"] = self.trace_id
        if self.model:
            d["model"] = self.model
        if self.agent:
            d["agent"] = self.agent
        if self.session_id:
            d["session_id"] = self.session_id
        if self.depth:
            d["depth"] = self.depth
        d.update(self.extras)
        return d

    def format(self) -> str:
        """Human-readable audit trail string."""
        lines = [
            f"  Source   : {self.source}",
            f"  Command  : {self.command}",
            f"  Timestamp: {self.timestamp}",
            f"  Air-gap  : {'ON' if self.airgap else 'OFF'}",
        ]
        if self.trace_id:
            lines.append(f"  Trace ID : {self.trace_id[:16]}...")
        if self.model:
            lines.append(f"  Model    : {self.model}")
        if self.agent:
            lines.append(f"  Agent    : {self.agent}")
        if self.depth:
            lines.append(f"  Depth    : {self.depth}")
        for k, v in self.extras.items():
            lines.append(f"  {k.title()}: {v}")
        return "\n".join(lines)


def build_provenance(
    source: str,
    command: str,
    trace_id: str = "",
    model: str = "",
    agent: str = "",
    session_id: str = "",
    depth: int = 0,
    **extras: Any,
) -> dict[str, Any]:
    """Build a provenance dict ready to store as JSONB in Postgres.

    Args:
        source:     Where the data came from (e.g. "bodega_intel_api", "web_scrape").
        command:    The Octane command that triggered this (e.g. "octane investigate").
        trace_id:   Synapse correlation ID.
        model:      LLM model name if synthesis was involved.
        agent:      Agent name ("web", "code", "memory", etc.).
        session_id: Optional session identifier.
        depth:      Research depth level (0 = top-level).
        **extras:   Any additional key-value pairs to include.

    Returns:
        dict suitable for JSONB storage.
    """
    record = ProvenanceRecord(
        source=source,
        command=command,
        timestamp=datetime.now(timezone.utc).isoformat(),
        airgap=is_airgap_active(),
        trace_id=trace_id,
        model=model,
        agent=agent,
        session_id=session_id,
        depth=depth,
        extras=extras,
    )
    return record.to_dict()
