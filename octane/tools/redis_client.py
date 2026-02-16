"""Redis client for Memory hot cache + Shadows backend (stub for Phase 1).

Full implementation when Shadows and Memory Agent are integrated.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="redis_client")


class RedisClient:
    """Async Redis client.

    Phase 1: Stub. Methods exist but log warnings.
    Phase 2: Full implementation with redis-py or aioredis.
    """

    def __init__(self, url: str | None = None) -> None:
        self.url = url
        self._client = None

    async def connect(self) -> None:
        """Connect to Redis."""
        logger.warning("redis_client_stub", msg="Redis client is a stub in Phase 1")

    async def close(self) -> None:
        """Close the Redis connection."""
        pass

    async def get(self, key: str) -> str | None:
        """Get a value (stub)."""
        logger.warning("redis_get_stub", key=key)
        return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Set a value (stub)."""
        logger.warning("redis_set_stub", key=key)
