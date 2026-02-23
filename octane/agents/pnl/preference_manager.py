"""Preference Manager — CRUD on user preferences backed by Redis.

Key scheme: pnl:{user_id}:pref:{key}
Examples:
    pnl:default:pref:verbosity      → "concise" | "detailed"
    pnl:default:pref:expertise      → "beginner" | "intermediate" | "advanced"
    pnl:default:pref:domains        → "technology,finance,science"
    pnl:default:pref:response_style → "bullets" | "prose" | "code-first"

Preferences persist in Redis (no TTL — they're permanent until changed).
"""

from __future__ import annotations

import structlog

from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="pnl.preferences")

# Default profile — applied when no preference has been set yet
DEFAULTS: dict[str, str] = {
    "verbosity": "concise",
    "expertise": "advanced",
    "domains": "technology,finance",
    "response_style": "prose",
}

PREF_TTL = 0  # No expiry — preferences are permanent


class PreferenceManager:
    """CRUD for user preferences in Redis."""

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis or RedisClient()

    def _key(self, user_id: str, pref: str) -> str:
        return f"pnl:{user_id}:pref:{pref}"

    async def get(self, user_id: str, key: str) -> str | None:
        val = await self._redis.get(self._key(user_id, key))
        if val is None:
            return DEFAULTS.get(key)
        return val

    async def set(self, user_id: str, key: str, value: str) -> None:
        await self._redis.set(self._key(user_id, key), value, ttl=PREF_TTL)
        logger.info("preference_set", user_id=user_id, key=key, value=value)

    async def get_all(self, user_id: str) -> dict[str, str]:
        """Return full preference profile, filling gaps with defaults."""
        profile: dict[str, str] = {}
        for key, default in DEFAULTS.items():
            val = await self._redis.get(self._key(user_id, key))
            profile[key] = val if val is not None else default
        return profile

    async def delete(self, user_id: str, key: str) -> None:
        await self._redis.delete(self._key(user_id, key))
        logger.info("preference_deleted", user_id=user_id, key=key)
