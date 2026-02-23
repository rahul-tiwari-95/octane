"""Redis client for Memory hot cache.

Tries to connect to Redis on first use. If Redis is unavailable,
falls back to an in-process dict so the Memory Agent always works.
Graceful degradation — same pattern as Bodega clients.
"""

from __future__ import annotations

import time
import structlog

logger = structlog.get_logger().bind(component="redis_client")

# TTL default: 1 hour
DEFAULT_TTL = 3600


class RedisClient:
    """Async Redis client with in-process dict fallback.

    Priority:
        1. Real Redis via redis-py (if installed + server up)
        2. In-process dict with manual TTL (always works)
    """

    def __init__(self, url: str | None = None) -> None:
        from octane.config import settings
        self.url = url or settings.redis_url
        self._redis = None          # redis.asyncio client (lazy)
        self._fallback: dict[str, tuple[str, float]] = {}  # key → (value, expire_at)
        self._use_fallback = False  # set True once Redis is confirmed unavailable

    async def _get_redis(self):
        """Lazy connect to Redis. Sets _use_fallback if unavailable."""
        if self._use_fallback:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis  # type: ignore
            client = aioredis.from_url(self.url, decode_responses=True)
            await client.ping()
            self._redis = client
            logger.info("redis_connected", url=self.url)
            return self._redis
        except Exception as e:
            logger.warning("redis_unavailable_using_fallback", error=str(e))
            self._use_fallback = True
            return None

    async def get(self, key: str) -> str | None:
        """Retrieve value by key. Returns None if missing or expired."""
        r = await self._get_redis()
        if r:
            try:
                return await r.get(key)
            except Exception as e:
                logger.warning("redis_get_error", key=key, error=str(e))

        # Fallback dict
        entry = self._fallback.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if expire_at and time.time() > expire_at:
            del self._fallback[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl: int = DEFAULT_TTL) -> None:
        """Store key→value with optional TTL (seconds)."""
        r = await self._get_redis()
        if r:
            try:
                if ttl:
                    await r.setex(key, ttl, value)
                else:
                    await r.set(key, value)
                return
            except Exception as e:
                logger.warning("redis_set_error", key=key, error=str(e))

        # Fallback dict
        expire_at = time.time() + ttl if ttl else 0.0
        self._fallback[key] = (value, expire_at)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        r = await self._get_redis()
        if r:
            try:
                await r.delete(key)
                return
            except Exception as e:
                logger.warning("redis_delete_error", key=key, error=str(e))
        self._fallback.pop(key, None)

    async def keys_matching(self, pattern: str) -> list[str]:
        """Return all keys matching a pattern (e.g. 'session:abc:*')."""
        r = await self._get_redis()
        if r:
            try:
                return await r.keys(pattern)
            except Exception as e:
                logger.warning("redis_keys_error", error=str(e))
        # Fallback: simple prefix match (strip trailing *)
        prefix = pattern.rstrip("*")
        now = time.time()
        return [
            k for k, (_, exp) in self._fallback.items()
            if k.startswith(prefix) and (exp == 0 or now < exp)
        ]

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
