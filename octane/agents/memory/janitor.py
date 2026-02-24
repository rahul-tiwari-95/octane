"""Janitor — tier management for the memory system.

Responsibilities:
  - sweep()   : list all live Redis keys for a session (fast, always works)
  - promote() : if a Postgres chunk has been accessed 3+ times and is not in
                Redis, push it back to hot cache (called lazily on recall hits)
  - clean()   : delete Postgres rows older than 30 days with zero accesses

Postgres operations are skipped gracefully if pg_client is None or unavailable.
"""

from __future__ import annotations

import json
import structlog

from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="memory.janitor")

PROMOTE_THRESHOLD = 3     # accesses before Redis promotion
STALE_DAYS = 30           # Postgres rows older than this with 0 accesses → delete


class Janitor:
    """Scans and manages memory tiers. Eviction is handled by Redis TTL naturally."""

    def __init__(
        self,
        redis: RedisClient | None = None,
        pg=None,  # PgClient | None
    ) -> None:
        self._redis = redis or RedisClient()
        self._pg = pg

    async def sweep(self, session_id: str = "default") -> dict:
        """List all live Redis keys for a session."""
        pattern = f"memory:{session_id}:*"
        keys = await self._redis.keys_matching(pattern)
        logger.debug("janitor_sweep", session_id=session_id, keys_found=len(keys))
        return {
            "session_id": session_id,
            "live_keys": len(keys),
            "keys": keys,
        }

    async def clean(self, session_id: str | None = None) -> dict:
        """Delete stale Postgres rows (older than STALE_DAYS, never accessed).

        If session_id is given, cleans only that session. Otherwise cleans all.
        Returns a dict with the number of rows deleted.
        """
        if not self._pg or not self._pg.available:
            return {"deleted": 0, "reason": "postgres_unavailable"}

        if session_id:
            deleted = await self._pg.fetchval(
                """
                WITH deleted AS (
                    DELETE FROM memory_chunks
                    WHERE session_id = $1
                      AND access_count = 0
                      AND created_at < NOW() - INTERVAL '30 days'
                    RETURNING id
                ) SELECT COUNT(*) FROM deleted
                """,
                session_id,
            )
        else:
            deleted = await self._pg.fetchval(
                """
                WITH deleted AS (
                    DELETE FROM memory_chunks
                    WHERE access_count = 0
                      AND created_at < NOW() - INTERVAL '30 days'
                    RETURNING id
                ) SELECT COUNT(*) FROM deleted
                """
            )

        count = int(deleted or 0)
        if count > 0:
            logger.info("janitor_cleaned", deleted=count, session_id=session_id or "all")
        return {"deleted": count}

    async def promote_hot(self, session_id: str = "default") -> int:
        """Promote frequently-accessed Postgres rows to Redis hot cache.

        Rows with access_count >= PROMOTE_THRESHOLD that are not already in Redis
        get pushed to the hot cache. Returns number of rows promoted.
        """
        if not self._pg or not self._pg.available:
            return 0

        rows = await self._pg.fetch(
            """
            SELECT id, session_id, slot, content, query
            FROM memory_chunks
            WHERE session_id = $1 AND access_count >= $2
            ORDER BY access_count DESC LIMIT 50
            """,
            session_id, PROMOTE_THRESHOLD,
        )

        promoted = 0
        for row in rows:
            key = f"memory:{row['session_id']}:{row['slot']}"
            existing = await self._redis.get(key)
            if existing is None:
                payload = json.dumps({
                    "query": row["query"],
                    "answer": row["content"],
                    "metadata": {"promoted_by": "janitor"},
                })
                await self._redis.set(key, payload, ttl=86400)
                promoted += 1

        if promoted:
            logger.info("janitor_promoted", promoted=promoted, session_id=session_id)
        return promoted
