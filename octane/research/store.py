"""ResearchStore — durable storage for research tasks and findings.

Storage layers:
    Redis  — task metadata (JSON hash), log ring-buffer (LIST, cap 200),
             cycle counter (INCR), active-task index (SET)
    Postgres — ``research_findings`` table; each cycle's synthesis is one row

Graceful degradation:
    - If Redis is unreachable: log_entry / get_log silently no-op.
    - If Postgres is unreachable: add_finding / get_findings return empty.
    - Never raises to the caller (all errors logged + swallowed).

Dependency injection:
    Pass ``_redis`` and/or ``_pg`` to inject pre-built clients in tests.
    In production leave them None — the store connects lazily.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import structlog

from octane.research.models import ResearchFinding, ResearchTask

logger = structlog.get_logger().bind(component="research.store")

# ── Postgres DDL ─────────────────────────────────────────────────────────────

_DDL_FINDINGS = """
CREATE TABLE IF NOT EXISTS research_findings (
    id          SERIAL      PRIMARY KEY,
    task_id     TEXT        NOT NULL,
    cycle_num   INTEGER     NOT NULL DEFAULT 0,
    topic       TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    agents_used TEXT[]      NOT NULL DEFAULT '{}',
    sources     TEXT[]      NOT NULL DEFAULT '{}',
    word_count  INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_research_task_id
    ON research_findings (task_id);
CREATE INDEX IF NOT EXISTS idx_research_task_created
    ON research_findings (task_id, created_at DESC);
"""

# Max log entries kept in the Redis ring buffer per task
_LOG_CAP = 200


class ResearchStore:
    """Read/write interface for research tasks and findings.

    Args:
        redis_url:  Redis connection string (defaults to settings).
        postgres_url: asyncpg DSN (defaults to settings).
        _redis:     Pre-built ``redis.asyncio.Redis`` (inject for tests).
        _pg:        Pre-built asyncpg ``Connection`` (inject for tests).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        postgres_url: str | None = None,
        *,
        _redis=None,
        _pg=None,
    ) -> None:
        if redis_url is None and _redis is None:
            from octane.config import settings
            redis_url = settings.redis_url
        if postgres_url is None and _pg is None:
            from octane.config import settings
            postgres_url = settings.postgres_url

        self._redis_url = redis_url
        self._postgres_url = postgres_url
        self._redis_client = _redis      # injected or None → lazy
        self._pg_conn = _pg              # injected or None → lazy
        self._schema_created = False

    # ── Redis helpers ─────────────────────────────────────────────────────

    async def _redis(self):
        """Return a live redis.asyncio.Redis connection (cached)."""
        if self._redis_client is None:
            try:
                import redis.asyncio as aioredis  # type: ignore
                self._redis_client = aioredis.from_url(
                    self._redis_url, decode_responses=True
                )
            except Exception as exc:
                logger.warning("research_store_redis_connect_failed", error=str(exc))
                return None
        return self._redis_client

    # ── Postgres helpers ──────────────────────────────────────────────────

    async def _pg(self):
        """Return a live asyncpg connection (cached, schema auto-created)."""
        if self._pg_conn is None:
            try:
                import asyncpg  # type: ignore
                self._pg_conn = await asyncpg.connect(self._postgres_url)
            except Exception as exc:
                logger.warning("research_store_pg_connect_failed", error=str(exc))
                return None
        if not self._schema_created and self._pg_conn is not None:
            await self.ensure_schema()
        return self._pg_conn

    async def ensure_schema(self) -> None:
        """Create the research_findings table if it doesn't exist."""
        pg = self._pg_conn
        if pg is None:
            return
        try:
            await pg.execute(_DDL_FINDINGS)
            self._schema_created = True
            logger.debug("research_schema_ready")
        except Exception as exc:
            logger.warning("research_schema_create_failed", error=str(exc))

    # ── Task metadata (Redis) ─────────────────────────────────────────────

    async def register_task(self, task: ResearchTask) -> None:
        """Persist task metadata and add to the active-task index."""
        r = await self._redis()
        if r is None:
            return
        try:
            payload = task.model_dump_json()
            await r.set(f"research:task:{task.id}", payload)
            await r.sadd("research:active", task.id)
            logger.info("research_task_registered", task_id=task.id, topic=task.topic)
        except Exception as exc:
            logger.warning("research_register_failed", error=str(exc))

    async def get_task(self, task_id: str) -> ResearchTask | None:
        """Retrieve task metadata from Redis."""
        r = await self._redis()
        if r is None:
            return None
        try:
            raw = await r.get(f"research:task:{task_id}")
            if raw is None:
                return None
            return ResearchTask.model_validate_json(raw)
        except Exception as exc:
            logger.warning("research_get_task_failed", error=str(exc))
            return None

    async def list_tasks(self) -> list[ResearchTask]:
        """Return all tasks in the active-task index."""
        r = await self._redis()
        if r is None:
            return []
        try:
            ids = await r.smembers("research:active")
            tasks: list[ResearchTask] = []
            for tid in ids:
                task = await self.get_task(tid)
                if task:
                    tasks.append(task)
            return sorted(tasks, key=lambda t: t.created_at)
        except Exception as exc:
            logger.warning("research_list_tasks_failed", error=str(exc))
            return []

    async def update_task_status(self, task_id: str, status: str) -> None:
        """Update the status field of an existing task."""
        task = await self.get_task(task_id)
        if task is None:
            return
        task.status = status
        r = await self._redis()
        if r is None:
            return
        try:
            await r.set(f"research:task:{task_id}", task.model_dump_json())
            if status == "stopped":
                await r.srem("research:active", task_id)
            logger.info("research_task_status_updated", task_id=task_id, status=status)
        except Exception as exc:
            logger.warning("research_update_status_failed", error=str(exc))

    async def increment_cycle(self, task_id: str) -> int:
        """Atomically increment the cycle counter; return new value."""
        r = await self._redis()
        if r is None:
            return 0
        try:
            val = await r.incr(f"research:cycle:{task_id}")
            # Also update finding_count in task metadata
            task = await self.get_task(task_id)
            if task:
                task.cycle_count = val
                await r.set(f"research:task:{task_id}", task.model_dump_json())
            return int(val)
        except Exception as exc:
            logger.warning("research_cycle_incr_failed", error=str(exc))
            return 0

    # ── Log ring buffer (Redis LIST) ──────────────────────────────────────

    async def log_entry(self, task_id: str, message: str) -> None:
        """Append a timestamped log line to the ring buffer (capped at 200)."""
        r = await self._redis()
        if r is None:
            return
        try:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            entry = f"[{ts}] {message}"
            key = f"research:log:{task_id}"
            async with r.pipeline() as pipe:
                pipe.rpush(key, entry)
                pipe.ltrim(key, -_LOG_CAP, -1)
                await pipe.execute()
        except Exception as exc:
            logger.warning("research_log_failed", error=str(exc))

    async def get_log(self, task_id: str, n: int = 50) -> list[str]:
        """Return the last *n* log entries (newest last)."""
        r = await self._redis()
        if r is None:
            return []
        try:
            entries = await r.lrange(f"research:log:{task_id}", -n, -1)
            return list(entries)
        except Exception as exc:
            logger.warning("research_get_log_failed", error=str(exc))
            return []

    async def log_length(self, task_id: str) -> int:
        """Return total number of log entries in the ring buffer."""
        r = await self._redis()
        if r is None:
            return 0
        try:
            return int(await r.llen(f"research:log:{task_id}"))
        except Exception:
            return 0

    # ── Findings (Postgres) ───────────────────────────────────────────────

    async def add_finding(
        self,
        task_id: str,
        cycle_num: int,
        topic: str,
        content: str,
        agents_used: list[str] | None = None,
        sources: list[str] | None = None,
        word_count: int | None = None,
    ) -> ResearchFinding | None:
        """Insert one research finding into Postgres.

        Also increments the task's finding_count in Redis.
        """
        pg = await self._pg()
        if pg is None:
            logger.warning("research_add_finding_no_pg", task_id=task_id)
            return None

        agents = agents_used or []
        srcs = sources or []
        wc = word_count if word_count is not None else len(content.split())

        try:
            row = await pg.fetchrow(
                """
                INSERT INTO research_findings
                    (task_id, cycle_num, topic, content, agents_used, sources, word_count)
                VALUES ($1, $2, $3, $4, $5::text[], $6::text[], $7)
                RETURNING *
                """,
                task_id, cycle_num, topic, content, agents, srcs, wc,
            )
            finding = ResearchFinding.from_row(dict(row))
            logger.info(
                "research_finding_stored",
                task_id=task_id,
                cycle=cycle_num,
                words=wc,
            )

            # Update finding_count in Redis metadata
            r = await self._redis()
            if r:
                task = await self.get_task(task_id)
                if task:
                    task.finding_count += 1
                    await r.set(f"research:task:{task_id}", task.model_dump_json())

            return finding
        except Exception as exc:
            logger.warning("research_add_finding_failed", error=str(exc), task_id=task_id)
            return None

    async def get_findings(self, task_id: str) -> list[ResearchFinding]:
        """Retrieve all findings for a task, oldest first."""
        pg = await self._pg()
        if pg is None:
            return []
        try:
            rows = await pg.fetch(
                """
                SELECT * FROM research_findings
                WHERE task_id = $1
                ORDER BY created_at ASC
                """,
                task_id,
            )
            return [ResearchFinding.from_row(dict(r)) for r in rows]
        except Exception as exc:
            logger.warning("research_get_findings_failed", error=str(exc))
            return []

    async def close(self) -> None:
        """Close Redis and Postgres connections."""
        if self._redis_client:
            try:
                await self._redis_client.aclose()
            except Exception:
                pass
        if self._pg_conn:
            try:
                await self._pg_conn.close()
            except Exception:
                pass
