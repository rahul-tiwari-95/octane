"""Memory Writer — decides what to persist and dual-writes to Redis + Postgres.

Persistence criteria (at least one must be true):
  - Response is longer than 80 chars (substantive answer)
  - Response contains numbers/data (facts worth recalling)

Write strategy:
  - ALWAYS write to Redis hot cache (TTL 24h) — fast recall within session
  - ALSO write to Postgres warm tier if pg_client is available — survives restarts
  - Cold tier (pgVector embeddings) — Phase 3, stub for now

Key format: memory:{session_id}:{slot}
TTL: 24 hours (Redis)
"""

from __future__ import annotations

import json
import re
import structlog

from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="memory.writer")

MEMORY_TTL = 86400  # 24 hours


class MemoryWriter:
    """Evaluates output quality and persists worthy facts to Redis and Postgres."""

    def __init__(
        self,
        redis: RedisClient | None = None,
        pg=None,  # PgClient | None — optional import avoids circular deps
    ) -> None:
        self._redis = redis or RedisClient()
        self._pg = pg  # injected by MemoryAgent after PgClient connects

    async def write(
        self,
        key: str,
        query: str,
        answer: str,
        metadata: dict | None = None,
        session_id: str = "default",
        agent_used: str = "",
    ) -> bool:
        """
        Persist query+answer if it meets quality bar.
        Writes to Redis always, Postgres when available.
        Returns True if stored, False if skipped.
        """
        if not self._worth_storing(answer):
            logger.debug("memory_skip_low_quality", key=key)
            return False

        # ── Redis hot cache ────────────────────────────────────────────────
        payload = json.dumps({
            "query": query,
            "answer": answer,
            "metadata": metadata or {},
        }, ensure_ascii=False)

        await self._redis.set(key, payload, ttl=MEMORY_TTL)
        logger.info("memory_stored_redis", key=key, answer_len=len(answer))

        # ── Postgres warm tier ─────────────────────────────────────────────
        if self._pg and self._pg.available:
            # Extract slot from key: "memory:{session_id}:{slot}"
            parts = key.split(":", 2)
            slot = parts[2] if len(parts) == 3 else key

            ok = await self._pg.execute(
                """
                INSERT INTO memory_chunks
                    (session_id, slot, content, query, agent_used, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                session_id,
                slot,
                answer,
                query,
                agent_used,
                json.dumps(metadata or {}),
            )
            if ok:
                logger.info("memory_stored_postgres", slot=slot, session_id=session_id)

        return True

    def _worth_storing(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 30:
            return False
        if re.search(r"\d", answer):
            return True
        if len(answer) > 80:
            return True
        return False
