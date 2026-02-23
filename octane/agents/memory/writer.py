"""Memory Writer — decides what to persist and stores it in Redis.

Persistence criteria (at least one must be true):
  - Response is longer than 80 chars (substantive answer)
  - Response contains numbers/data (facts worth recalling)
  - Response was produced by a successful agent (success=True)

Key format: memory:{session_id}:{slot}
TTL: 24 hours by default (configurable)
"""

from __future__ import annotations

import json
import re
import structlog

from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="memory.writer")

MEMORY_TTL = 86400  # 24 hours


class MemoryWriter:
    """Evaluates output quality and persists worthy facts to Redis."""

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis or RedisClient()

    async def write(
        self,
        key: str,
        query: str,
        answer: str,
        metadata: dict | None = None,
    ) -> bool:
        """
        Persist query+answer if it meets quality bar.
        Returns True if stored, False if skipped.
        """
        if not self._worth_storing(answer):
            logger.debug("memory_skip_low_quality", key=key)
            return False

        payload = json.dumps({
            "query": query,
            "answer": answer,
            "metadata": metadata or {},
        }, ensure_ascii=False)

        await self._redis.set(key, payload, ttl=MEMORY_TTL)
        logger.info("memory_stored", key=key, answer_len=len(answer))
        return True

    def _worth_storing(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 30:
            return False
        # Contains numbers/data → likely factual
        if re.search(r"\d", answer):
            return True
        # Long enough to be substantive
        if len(answer) > 80:
            return True
        return False
