"""Janitor — scans Redis keys for a session and reports live/expired counts.

In-process fallback: scans the RedisClient dict store.
Runs on demand (called by MemoryAgent during READ to give context hints).
"""

from __future__ import annotations

import structlog

from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="memory.janitor")


class Janitor:
    """Scans memory keys and reports stats. Expired entries are evicted lazily by Redis."""

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis or RedisClient()

    async def sweep(self, session_id: str = "default") -> dict:
        """List all live memory keys for a session."""
        pattern = f"memory:{session_id}:*"
        keys = await self._redis.keys_matching(pattern)
        logger.debug("janitor_sweep", session_id=session_id, keys_found=len(keys))
        return {
            "session_id": session_id,
            "live_keys": len(keys),
            "keys": keys,
        }
