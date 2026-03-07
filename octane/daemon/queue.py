"""Priority Queue — OS-level task scheduling for the Octane Daemon.

Implements a multi-level priority queue inspired by operating system process
schedulers.  Four priority levels (P0–P3) ensure interactive queries never
starve behind batch jobs, while priority aging prevents lower-priority tasks
from waiting forever.

Priority levels:
    P0 — Interactive.  User is waiting at the terminal.  Max latency budget.
    P1 — Shadow.       Background research jobs (Shadows/monitors).
    P2 — Scheduled.    Timed tasks (e.g., "every morning" from eyeso).
    P3 — Batch.        Bulk operations, imports, re-indexing.

Starvation prevention (priority aging):
    Every AGING_INTERVAL_SEC, items waiting longer than their age threshold
    get bumped one priority level up.  P3 → P2 after 30s, P2 → P1 after 60s,
    P1 → P0 after 120s.  This guarantees every task eventually executes, even
    under heavy interactive load.

Drain support:
    When the daemon is shutting down, the queue enters drain mode — no new
    items accepted, existing items processed until empty.

Implementation: asyncio.PriorityQueue with (priority, sequence, item) tuples.
The sequence number breaks ties within the same priority (FIFO within level).
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="daemon_queue")


# ── Priority Levels ───────────────────────────────────────────────────────────


class Priority(int, enum.Enum):
    """Task priority levels. Lower value = higher priority."""

    P0_INTERACTIVE = 0   # User is waiting
    P1_SHADOW = 1        # Background research
    P2_SCHEDULED = 2     # Timed tasks
    P3_BATCH = 3         # Bulk operations


# ── Aging thresholds (seconds waiting before bump) ────────────────────────────

AGING_THRESHOLDS: dict[Priority, float] = {
    Priority.P3_BATCH: 30.0,       # P3 → P2 after 30s
    Priority.P2_SCHEDULED: 60.0,   # P2 → P1 after 60s
    Priority.P1_SHADOW: 120.0,     # P1 → P0 after 120s
    Priority.P0_INTERACTIVE: 0.0,  # P0 can't be bumped further
}

# How often to run the aging sweep (seconds)
AGING_INTERVAL_SEC: float = 10.0


# ── Queue Item ────────────────────────────────────────────────────────────────


@dataclass(order=False)
class QueueItem:
    """A single task waiting in the priority queue.

    Attributes:
        task_id:         Unique identifier for this task.
        priority:        Current priority level (may change via aging).
        original_priority: Priority at submission (never changes).
        command:         The command to execute (e.g., "ask", "research").
        payload:         Command-specific data (query, flags, etc.).
        submitted_at:    Monotonic timestamp when item entered the queue.
        correlation_id:  Links to Synapse trace events.
        aged_count:      Number of times this item has been aged up.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: Priority = Priority.P0_INTERACTIVE
    original_priority: Priority = Priority.P0_INTERACTIVE
    command: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.monotonic)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aged_count: int = 0

    @property
    def wait_time(self) -> float:
        """Seconds this item has been waiting (monotonic)."""
        return time.monotonic() - self.submitted_at

    def __lt__(self, other: QueueItem) -> bool:
        """Comparison for heapq — lower priority value wins."""
        return self.priority.value < other.priority.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueueItem):
            return NotImplemented
        return self.task_id == other.task_id

    def __hash__(self) -> int:
        return hash(self.task_id)


# ── Priority Queue ────────────────────────────────────────────────────────────


class DaemonQueue:
    """Multi-level priority queue with aging and drain support.

    Thread-safe through asyncio primitives. NOT thread-safe for
    multi-threaded access — designed for single-event-loop use.

    The queue uses a list sorted by (priority, sequence) tuples internally.
    This is simpler and more testable than asyncio.PriorityQueue for our
    use case because we need to re-sort after aging.

    Args:
        max_size:  Maximum number of items in queue. 0 = unlimited.
    """

    def __init__(self, max_size: int = 0) -> None:
        self._items: list[tuple[int, int, QueueItem]] = []  # (priority, seq, item)
        self._seq: int = 0  # Monotonic sequence for FIFO within priority
        self._max_size = max_size
        self._draining = False
        self._not_empty = asyncio.Event()
        self._lock = asyncio.Lock()

        # Stats
        self._total_submitted: int = 0
        self._total_completed: int = 0
        self._total_aged: int = 0

    @property
    def size(self) -> int:
        """Current number of items in queue."""
        return len(self._items)

    @property
    def is_draining(self) -> bool:
        """True if queue is in drain mode (no new submissions)."""
        return self._draining

    @property
    def is_empty(self) -> bool:
        return len(self._items) == 0

    async def submit(self, item: QueueItem) -> bool:
        """Add a task to the queue.

        Returns True if accepted, False if queue is draining or full.
        """
        async with self._lock:
            if self._draining:
                logger.warning("queue_rejected_draining", task_id=item.task_id)
                return False

            if self._max_size > 0 and len(self._items) >= self._max_size:
                logger.warning(
                    "queue_rejected_full",
                    task_id=item.task_id,
                    size=len(self._items),
                    max_size=self._max_size,
                )
                return False

            self._seq += 1
            self._items.append((item.priority.value, self._seq, item))
            self._items.sort(key=lambda t: (t[0], t[1]))
            self._total_submitted += 1
            self._not_empty.set()

            logger.debug(
                "queue_submitted",
                task_id=item.task_id,
                priority=item.priority.name,
                command=item.command,
                depth=len(self._items),
            )
            return True

    async def get(self, timeout: float | None = None) -> QueueItem | None:
        """Get the highest-priority item from the queue.

        Blocks until an item is available or timeout expires.
        Returns None on timeout or if queue is empty and draining.

        Args:
            timeout: Max seconds to wait. None = wait forever.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        while True:
            async with self._lock:
                if self._items:
                    _, _, item = self._items.pop(0)
                    if not self._items:
                        self._not_empty.clear()
                    self._total_completed += 1
                    return item

                if self._draining:
                    return None

            # Wait for an item
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None

            try:
                if remaining is not None:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=remaining)
                else:
                    await self._not_empty.wait()
            except asyncio.TimeoutError:
                return None

    async def peek(self) -> QueueItem | None:
        """Look at the next item without removing it."""
        async with self._lock:
            if self._items:
                return self._items[0][2]
            return None

    async def drain(self) -> None:
        """Enter drain mode — no new submissions accepted.

        Existing items remain and can be get()'d until empty.
        """
        async with self._lock:
            self._draining = True
            # Wake up anyone waiting so they can see drain state
            self._not_empty.set()
        logger.info("queue_draining", remaining=len(self._items))

    async def reset(self) -> None:
        """Exit drain mode and clear the queue. Used for testing and restarts."""
        async with self._lock:
            self._items.clear()
            self._draining = False
            self._not_empty.clear()

    async def remove(self, task_id: str) -> bool:
        """Remove a task from the queue by task_id.

        Returns True if found and removed, False if not found.
        Used when a task completes or is cancelled.
        """
        async with self._lock:
            for i, (_, _, item) in enumerate(self._items):
                if item.task_id == task_id:
                    self._items.pop(i)
                    self._total_completed += 1
                    if not self._items:
                        self._not_empty.clear()
                    logger.debug("queue_item_removed", task_id=task_id)
                    return True
            return False

    async def age_items(self, now: float | None = None) -> int:
        """Run priority aging sweep.

        Items waiting longer than their age threshold get bumped up
        one priority level. Returns the number of items aged.

        This should be called periodically (every AGING_INTERVAL_SEC).
        """
        if now is None:
            now = time.monotonic()

        aged = 0
        async with self._lock:
            new_items: list[tuple[int, int, QueueItem]] = []
            for pri, seq, item in self._items:
                threshold = AGING_THRESHOLDS.get(item.priority, 0.0)
                wait = now - item.submitted_at

                if threshold > 0 and wait >= threshold and item.priority.value > 0:
                    # Bump up one level
                    old_name = item.priority.name
                    new_priority = Priority(item.priority.value - 1)
                    item.priority = new_priority
                    item.aged_count += 1
                    aged += 1
                    self._total_aged += 1
                    logger.debug(
                        "queue_item_aged",
                        task_id=item.task_id,
                        from_priority=old_name,
                        to_priority=new_priority.name,
                        wait_seconds=round(wait, 1),
                    )
                    new_items.append((new_priority.value, seq, item))
                else:
                    new_items.append((pri, seq, item))

            self._items = sorted(new_items, key=lambda t: (t[0], t[1]))

        return aged

    def depth_by_priority(self) -> dict[str, int]:
        """Return count of items per priority level."""
        counts: dict[str, int] = {p.name: 0 for p in Priority}
        for pri, _, item in self._items:
            counts[item.priority.name] = counts.get(item.priority.name, 0) + 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        """Queue state for monitoring."""
        return {
            "size": len(self._items),
            "draining": self._draining,
            "depth_by_priority": self.depth_by_priority(),
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_aged": self._total_aged,
        }
