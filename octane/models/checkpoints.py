"""Checkpoint model — pipeline state snapshot.

A Checkpoint captures the complete pipeline state at a decision point,
enabling revert-to-checkpoint when a user declines a decision without
losing any work already completed.

Checkpoints are created at:
1. After decomposition (type='plan')
2. Before each high-risk task execution (type='pre_execution')
3. After parallel group completion / before synthesis (type='post_execution')
4. Before final synthesis (type='pre_synthesis')

Session 16: Phase 1 — in-memory storage only.  Phase 4: Redis persistence.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from octane.models.dag import TaskDAG
from octane.models.decisions import Decision


class Checkpoint(BaseModel):
    """Snapshot of pipeline state at a decision point.

    accumulated_results is ALWAYS preserved on revert — completed agent calls
    are never re-fetched.  Only the downstream tasks that were affected by
    the declined decision are re-executed.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    checkpoint_type: str = "plan"
    """'plan' | 'pre_execution' | 'post_execution' | 'pre_synthesis'."""

    # ── Pipeline state ───────────────────────────────────────────────────
    dag: TaskDAG
    completed_tasks: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)

    # ── Results so far ──────────────────────────────────────────────────
    accumulated_results: dict[str, Any] = Field(default_factory=dict)
    """task_id → AgentResponse.model_dump() — preserved across reverts."""

    # ── Decision state ──────────────────────────────────────────────────
    decisions: list[Decision] = Field(default_factory=list)
    approved_decisions: list[str] = Field(default_factory=list)
    """IDs of decisions approved up to this checkpoint."""

    # ── Context state ───────────────────────────────────────────────────
    memory_context: dict[str, Any] = Field(default_factory=dict)
    user_profile: dict[str, Any] = Field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.dag.nodes)

    @property
    def progress_pct(self) -> float:
        total = len(self.dag.nodes)
        if total == 0:
            return 100.0
        return round(len(self.completed_tasks) / total * 100, 1)
