"""ResearchTask and ResearchFinding — Pydantic models for the research pipeline.

ResearchTask   — metadata for a live research topic (stored in Redis)
ResearchFinding — a single synthesised output from one research cycle (stored in Postgres)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ResearchTask(BaseModel):
    """Metadata for a named research topic scheduled via Shadows.

    Stored as JSON at ``research:task:{id}`` in Redis.
    Also tracked in the ``research:active`` Redis SET.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Short 8-char ID used as the Shadows task key.",
    )
    topic: str
    interval_hours: float = Field(default=6.0, description="Cycle cadence in hours.")
    depth: str = Field(
        default="deep",
        description="Research depth: shallow | deep | exhaustive",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    status: str = Field(
        default="active",
        description="active | stopped",
    )
    cycle_count: int = Field(default=0, description="Total cycles completed.")
    finding_count: int = Field(default=0, description="Total findings stored.")

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def age_hours(self) -> float:
        """Hours since this task was created."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 3600


class ResearchFinding(BaseModel):
    """A single synthesised research output from one cycle run.

    Persisted in the ``research_findings`` Postgres table.
    Also reconstructable from a dict (row returned by asyncpg).
    """

    id: int = Field(default=0, description="Auto-assigned Postgres SERIAL id.")
    task_id: str
    cycle_num: int
    topic: str
    content: str
    agents_used: list[str] = []
    sources: list[str] = []
    word_count: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @classmethod
    def from_row(cls, row: dict) -> "ResearchFinding":
        """Construct from an asyncpg record dict."""
        return cls(
            id=row.get("id", 0),
            task_id=row["task_id"],
            cycle_num=row.get("cycle_num", 0),
            topic=row.get("topic", ""),
            content=row.get("content", ""),
            agents_used=list(row.get("agents_used") or []),
            sources=list(row.get("sources") or []),
            word_count=row.get("word_count", 0),
            created_at=row.get("created_at") or datetime.now(timezone.utc),
        )

    @property
    def preview(self) -> str:
        """First 200 chars of content."""
        return self.content[:200].rstrip() + ("…" if len(self.content) > 200 else "")
