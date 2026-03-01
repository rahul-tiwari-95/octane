"""Decision Ledger models — Decision, DecisionLedger.

Every non-trivial OSA decision is logged here.  The HIL Manager reviews
decisions that exceed risk or confidence thresholds and routes them to
the human for approval, modification, or decline.

Risk levels (lowest → highest):
    low      — read-only, reversible, high-confidence actions (auto-approved)
    medium   — mutations or moderate-confidence operations
    high     — irreversible or external-write operations
    critical — financial, destructive, or personally-sensitive actions

Session 16: Phase 1 HIL — Decision Ledger + in-memory Checkpoint plumbing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """A single decision made by OSA during pipeline execution.

    Logged in the Decision Ledger. May require human approval before
    the pipeline proceeds past this point.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── What ────────────────────────────────────────────────────────────
    action: str
    """Human-readable description: 'Route to Web Agent for AMD comparison'."""

    reasoning: str = ""
    """Why OSA made this choice."""

    # ── Risk ────────────────────────────────────────────────────────────
    risk_level: str = "low"
    """'low' | 'medium' | 'high' | 'critical'."""

    confidence: float = 1.0
    """0.0 → 1.0. Below 0.60 triggers HIL review regardless of risk."""

    uncertainty_reason: str = ""
    """Human-readable explanation if confidence < 1.0."""

    # ── Evidence ────────────────────────────────────────────────────────
    sources: list[str] = Field(default_factory=list)
    """URLs, memory chunk IDs, or data references used to justify the decision."""

    code_preview: str | None = None
    """If the decision involves code execution, show the code here."""

    # ── Status ──────────────────────────────────────────────────────────
    status: str = "pending"
    """
    'pending'          → awaiting review (auto or human)
    'auto_approved'    → passed policy rules, no human needed
    'human_approved'   → user explicitly approved
    'human_modified'   → user approved with changes (see human_feedback)
    'human_declined'   → user rejected; pipeline should replan
    """

    human_feedback: str = ""
    """User's modification instruction or reason for declining."""

    # ── Linkage ─────────────────────────────────────────────────────────
    task_id: str = ""
    """Which TaskNode in the DAG triggered this decision."""

    reversible: bool = True
    """Whether the action can be undone after execution."""

    agent: str = ""
    """Agent name that will execute this decision."""

    @property
    def needs_human(self) -> bool:
        """True when this decision requires human review before proceeding."""
        return (
            self.risk_level in ("high", "critical")
            or self.confidence < 0.60
            or (self.risk_level == "medium" and self.confidence < 0.75)
        )

    @property
    def is_resolved(self) -> bool:
        """True when a final status has been set (not pending)."""
        return self.status != "pending"

    @property
    def is_approved(self) -> bool:
        """True when the decision was approved (auto or human)."""
        return self.status in ("auto_approved", "human_approved", "human_modified")


class DecisionLedger(BaseModel):
    """All decisions for a single pipeline execution."""

    correlation_id: str
    decisions: list[Decision] = Field(default_factory=list)

    def add(self, decision: Decision) -> None:
        """Append a decision to the ledger."""
        self.decisions.append(decision)

    @property
    def pending(self) -> list[Decision]:
        """Decisions not yet resolved."""
        return [d for d in self.decisions if not d.is_resolved]

    @property
    def needs_human_review(self) -> list[Decision]:
        """Decisions that require explicit human review."""
        return [d for d in self.decisions if d.needs_human and not d.is_resolved]

    @property
    def auto_approved(self) -> list[Decision]:
        """Decisions that were auto-approved by policy."""
        return [d for d in self.decisions if d.status == "auto_approved"]

    @property
    def declined(self) -> list[Decision]:
        """Decisions that the human explicitly declined."""
        return [d for d in self.decisions if d.status == "human_declined"]

    @property
    def any_declined(self) -> bool:
        """True if at least one decision was declined — pipeline should replan."""
        return any(d.status == "human_declined" for d in self.decisions)

    def summary(self) -> dict[str, Any]:
        """Compact summary for Synapse event payloads."""
        return {
            "total": len(self.decisions),
            "auto_approved": len(self.auto_approved),
            "needs_human": len(self.needs_human_review),
            "declined": len(self.declined),
        }
