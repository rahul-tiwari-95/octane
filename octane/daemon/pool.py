"""Connection Pool Manager — shared pools for the Octane Daemon.

Instead of each CLI invocation creating its own Redis/Postgres/Bodega
connections (cold start every time), the daemon maintains warm pools
that are shared across all concurrent tasks.

Pool sizing per topology:
    compact:  Redis 2, Postgres 2, Bodega 1 (minimal footprint)
    balanced: Redis 4, Postgres 5, Bodega 2 (parallel processing)
    power:    Redis 8, Postgres 10, Bodega 4 (max throughput)

Health checks run periodically to detect connection failures early.
Auto-reconnect on failure — graceful degradation, never crash.

Playwright browser pool is lazy-initialized (only when web browsing
is needed) because Chromium uses ~200-400 MB of RAM.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

from octane.daemon.state import DaemonState

logger = structlog.get_logger().bind(component="pool_manager")


# ── Pool Sizing ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoolSizing:
    """Connection pool sizes per topology."""

    redis_pool_size: int = 4
    postgres_min_size: int = 1
    postgres_max_size: int = 5
    bodega_max_connections: int = 2
    health_check_interval: float = 30.0  # Seconds between health checks


POOL_SIZES: dict[str, PoolSizing] = {
    "compact": PoolSizing(
        redis_pool_size=2,
        postgres_min_size=1,
        postgres_max_size=2,
        bodega_max_connections=1,
        health_check_interval=60.0,
    ),
    "balanced": PoolSizing(
        redis_pool_size=4,
        postgres_min_size=1,
        postgres_max_size=5,
        bodega_max_connections=2,
        health_check_interval=30.0,
    ),
    "power": PoolSizing(
        redis_pool_size=8,
        postgres_min_size=2,
        postgres_max_size=10,
        bodega_max_connections=4,
        health_check_interval=15.0,
    ),
}


class PoolManager:
    """Manages shared connection pools for Redis, Postgres, and Bodega.

    Pools are initialized lazily — first access creates the pool.
    Health checks run in the background and update DaemonState.

    The PoolManager WRAPS existing client classes (RedisClient, PgClient,
    BodegaInferenceClient) — it doesn't replace them.  Consumers get
    clients from the pool, use them, and return them.

    Args:
        state:          DaemonState for health reporting.
        topology_name:  Determines pool sizing.
    """

    def __init__(
        self,
        state: DaemonState,
        topology_name: str = "balanced",
    ) -> None:
        self.state = state
        self.topology_name = topology_name
        self.sizing = POOL_SIZES.get(topology_name, POOL_SIZES["balanced"])

        # Client instances (lazy init)
        self._redis = None
        self._postgres = None
        self._bodega = None

        # Health check task
        self._health_task: asyncio.Task | None = None
        self._running = False

        # Connection stats
        self._redis_ops: int = 0
        self._pg_ops: int = 0
        self._bodega_ops: int = 0

    # ── Redis ─────────────────────────────────────────────────────────────

    async def get_redis(self):
        """Get the shared Redis client. Creates on first call.

        Returns the RedisClient instance. The client handles its own
        connection pooling internally via redis-py's pool.
        """
        if self._redis is None:
            from octane.tools.redis_client import RedisClient
            self._redis = RedisClient()
            # Trigger initial connection attempt
            await self._redis._get_redis()
            # Update state health
            if self._redis._use_fallback:
                await self.state.update_health(
                    "redis", False, error="using in-process fallback"
                )
            else:
                await self.state.update_health("redis", True)
            logger.info("redis_pool_initialized")
        self._redis_ops += 1
        return self._redis

    # ── Postgres ──────────────────────────────────────────────────────────

    async def get_postgres(self):
        """Get the shared Postgres client. Creates and connects on first call."""
        if self._postgres is None:
            from octane.tools.pg_client import PgClient
            self._postgres = PgClient()
            connected = await self._postgres.connect()
            if connected:
                await self.state.update_health("postgres", True)
            else:
                await self.state.update_health(
                    "postgres", False, error="connection failed"
                )
            logger.info(
                "postgres_pool_initialized",
                available=self._postgres.available,
            )
        self._pg_ops += 1
        return self._postgres

    # ── Bodega ────────────────────────────────────────────────────────────

    async def get_bodega(self):
        """Get the shared Bodega inference client."""
        if self._bodega is None:
            from octane.tools.bodega_inference import BodegaInferenceClient
            # 600 s: allows axe-stealth-37b to generate up to ~6000 tokens
            # at ~10 tok/s without timing out, which covers synthesis workloads.
            self._bodega = BodegaInferenceClient(timeout=600.0)
            # Quick health check
            try:
                health = await self._bodega.health()
                if health.get("status") == "ok":
                    await self.state.update_health("bodega", True)
                else:
                    await self.state.update_health(
                        "bodega", False, error="unhealthy response"
                    )
            except Exception as exc:
                await self.state.update_health(
                    "bodega", False, error=str(exc)
                )
            logger.info("bodega_client_initialized")
        self._bodega_ops += 1
        return self._bodega

    # ── Health Checks ─────────────────────────────────────────────────────

    async def start_health_checks(self) -> None:
        """Start periodic health check loop."""
        if self._running:
            return
        self._running = True
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(
            "health_checks_started",
            interval=self.sizing.health_check_interval,
        )

    async def stop_health_checks(self) -> None:
        """Stop periodic health checks."""
        self._running = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        self._health_task = None

    async def _health_loop(self) -> None:
        """Background: periodically check all connections."""
        try:
            while self._running:
                await asyncio.sleep(self.sizing.health_check_interval)
                if not self._running:
                    break
                await self._check_all_health()
        except asyncio.CancelledError:
            pass

    async def _check_all_health(self) -> None:
        """Run health checks on all initialized pools."""
        # Redis
        if self._redis is not None:
            try:
                t0 = time.monotonic()
                r = await self._redis._get_redis()
                if r is not None:
                    await r.ping()
                    latency = (time.monotonic() - t0) * 1000
                    await self.state.update_health("redis", True, latency)
                else:
                    await self.state.update_health(
                        "redis", False, error="fallback mode"
                    )
            except Exception as exc:
                await self.state.update_health("redis", False, error=str(exc))

        # Postgres
        if self._postgres is not None:
            try:
                t0 = time.monotonic()
                val = await self._postgres.fetchval("SELECT 1")
                latency = (time.monotonic() - t0) * 1000
                if val == 1:
                    await self.state.update_health("postgres", True, latency)
                else:
                    await self.state.update_health(
                        "postgres", False, error="unexpected response"
                    )
            except Exception as exc:
                await self.state.update_health("postgres", False, error=str(exc))

        # Bodega
        if self._bodega is not None:
            try:
                t0 = time.monotonic()
                health = await self._bodega.health()
                latency = (time.monotonic() - t0) * 1000
                if health.get("status") == "ok":
                    await self.state.update_health("bodega", True, latency)
                else:
                    await self.state.update_health(
                        "bodega", False, error="unhealthy"
                    )
            except Exception as exc:
                await self.state.update_health("bodega", False, error=str(exc))

    # ── Shutdown ──────────────────────────────────────────────────────────

    async def close_all(self) -> None:
        """Gracefully close all pools."""
        await self.stop_health_checks()

        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None

        if self._postgres is not None:
            try:
                await self._postgres.close()
            except Exception:
                pass
            self._postgres = None

        if self._bodega is not None:
            try:
                await self._bodega.close()
            except Exception:
                pass
            self._bodega = None

        logger.info("all_pools_closed")

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Pool state for monitoring."""
        return {
            "topology": self.topology_name,
            "sizing": {
                "redis": self.sizing.redis_pool_size,
                "postgres_max": self.sizing.postgres_max_size,
                "bodega_max": self.sizing.bodega_max_connections,
            },
            "initialized": {
                "redis": self._redis is not None,
                "postgres": self._postgres is not None,
                "bodega": self._bodega is not None,
            },
            "ops": {
                "redis": self._redis_ops,
                "postgres": self._pg_ops,
                "bodega": self._bodega_ops,
            },
        }
