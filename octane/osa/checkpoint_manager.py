"""Checkpoint Manager — creates, stores, and restores pipeline snapshots.

Session 16: Phase 1 — in-memory dict storage.
Phase 4: Replace _store with Redis HSET/HGET (TTL: 24h per correlation_id).

Usage pattern in Orchestrator:
    cp = await checkpoint_mgr.create(
        correlation_id=cid,
        dag=dag,
        results=accumulated,
        decisions=ledger.decisions,
        checkpoint_type="plan",
    )
    ...
    # User declined a decision — revert without losing completed work
    restored_cp = await checkpoint_mgr.revert(cid, cp.id)
"""

from __future__ import annotations

from typing import Any

import structlog

from octane.models.checkpoints import Checkpoint
from octane.models.dag import TaskDAG
from octane.models.decisions import Decision

logger = structlog.get_logger().bind(component="osa.checkpoint_mgr")


class CheckpointManager:
    """Creates, stores, and restores pipeline state snapshots.

    All checkpoints for a given correlation_id are stored in insertion
    order.  ``revert()`` returns the target checkpoint and drops any
    checkpoints that were created after it (they are now invalidated by
    the replan).
    """

    def __init__(self) -> None:
        # correlation_id → list[Checkpoint] in creation order
        self._store: dict[str, list[Checkpoint]] = {}

    async def create(
        self,
        correlation_id: str,
        dag: TaskDAG,
        results: dict[str, Any],
        decisions: list[Decision],
        checkpoint_type: str = "plan",
        memory_context: dict[str, Any] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create and store a new checkpoint.

        Args:
            correlation_id: Pipeline correlation ID.
            dag: Current TaskDAG (may be partial if replanning mid-flight).
            results: task_id → AgentResponse (accumulated so far).
            decisions: All decisions logged so far.
            checkpoint_type: One of 'plan', 'pre_execution', 'post_execution', 'pre_synthesis'.
            memory_context: Optional memory context dict.
            user_profile: Optional user profile dict.

        Returns:
            The newly created Checkpoint.
        """
        cp = Checkpoint(
            correlation_id=correlation_id,
            checkpoint_type=checkpoint_type,
            dag=dag,
            completed_tasks=list(results.keys()),
            pending_tasks=[n.task_id for n in dag.nodes if n.task_id not in results],
            accumulated_results={
                tid: resp if isinstance(resp, dict) else resp.model_dump()
                for tid, resp in results.items()
            },
            decisions=list(decisions),
            approved_decisions=[d.id for d in decisions if d.is_approved],
            memory_context=memory_context or {},
            user_profile=user_profile or {},
        )
        self._store.setdefault(correlation_id, []).append(cp)

        logger.debug(
            "checkpoint_created",
            correlation_id=correlation_id,
            checkpoint_id=cp.id,
            type=checkpoint_type,
            completed=len(cp.completed_tasks),
            pending=len(cp.pending_tasks),
        )
        return cp

    async def revert(self, correlation_id: str, checkpoint_id: str) -> Checkpoint:
        """Restore pipeline state to a previous checkpoint.

        All checkpoints created AFTER the target are dropped.
        The accumulated_results in the target are preserved — no
        upstream agent calls are repeated.

        Args:
            correlation_id: Pipeline correlation ID.
            checkpoint_id: ID of the checkpoint to revert to.

        Returns:
            The target Checkpoint.

        Raises:
            KeyError: If correlation_id or checkpoint_id is not found.
        """
        checkpoints = self._store.get(correlation_id, [])
        target = next((cp for cp in checkpoints if cp.id == checkpoint_id), None)
        if target is None:
            raise KeyError(
                f"Checkpoint {checkpoint_id!r} not found for correlation_id {correlation_id!r}"
            )
        idx = checkpoints.index(target)
        # Drop any checkpoints created after this one
        self._store[correlation_id] = checkpoints[: idx + 1]

        logger.info(
            "checkpoint_reverted",
            correlation_id=correlation_id,
            checkpoint_id=checkpoint_id,
            type=target.checkpoint_type,
            restored_results=len(target.accumulated_results),
        )
        return target

    def list_checkpoints(self, correlation_id: str) -> list[Checkpoint]:
        """Return all checkpoints for a correlation_id in creation order."""
        return list(self._store.get(correlation_id, []))

    def latest(self, correlation_id: str) -> Checkpoint | None:
        """Return the most recently created checkpoint, or None."""
        cps = self._store.get(correlation_id, [])
        return cps[-1] if cps else None

    def clear(self, correlation_id: str) -> None:
        """Remove all checkpoints for a correlation_id (called after session ends)."""
        self._store.pop(correlation_id, None)
