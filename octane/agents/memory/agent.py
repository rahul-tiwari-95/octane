"""Memory Agent coordinator.

Two operations:

WRITE (called by OSA Orchestrator after every successful response):
    MemoryRouter classifies + generates key
    MemoryWriter dual-writes: Redis hot cache + Postgres warm tier

READ (called when query mentions recall/remember/last time etc.):
    Waterfall: Redis → Postgres → None
    Redis hit → return immediately
    Postgres hit → promote back to Redis, return
    All miss → return None

The OSA Orchestrator injects answers into memory after each pipeline run.
The Evaluator receives prior context so answers are coherent across turns.
"""

from __future__ import annotations

import json
import structlog

from octane.agents.base import BaseAgent
from octane.agents.memory.janitor import Janitor
from octane.agents.memory.router import MemoryRouter
from octane.agents.memory.writer import MemoryWriter
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="memory.agent")


class MemoryAgent(BaseAgent):
    """Memory Agent — Redis hot tier + Postgres warm tier with graceful fallback."""

    name = "memory"

    def __init__(
        self,
        synapse,
        redis: RedisClient | None = None,
        pg=None,  # PgClient | None
    ) -> None:
        super().__init__(synapse)
        self._redis = redis or RedisClient()
        self._pg = pg  # None until connect() is called
        self.mem_router = MemoryRouter()
        self.writer = MemoryWriter(redis=self._redis, pg=self._pg)
        self.janitor = Janitor(redis=self._redis)

    async def connect_pg(self) -> bool:
        """Connect to Postgres and wire it into the writer.

        Called by Orchestrator on startup. Safe to call if Postgres is down.
        Returns True if connected, False if unavailable (Redis-only fallback).
        """
        from octane.tools.pg_client import PgClient
        pg = PgClient()
        connected = await pg.connect()
        if connected:
            self._pg = pg
            self.writer._pg = pg  # wire into writer
            logger.info("memory_pg_connected")
        else:
            logger.info("memory_pg_unavailable_redis_only")
        return connected

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route to READ or WRITE based on query intent."""
        route = self.mem_router.route(request.query, session_id=request.session_id or "default")

        if route["intent"] == "read":
            return await self._read(request, route)
        else:
            return await self._write(request, route)

    async def _read(self, request: AgentRequest, route: dict) -> AgentResponse:
        """Retrieve stored memory — waterfall: Redis → Postgres → none."""
        result = await self.recall(
            session_id=route["session_id"],
            query=request.query,
        )

        if result:
            return AgentResponse(
                agent=self.name, success=True,
                output=f"From memory: {result}",
                data={"found": True, "key": route["key"]},
                correlation_id=request.correlation_id,
            )

        return AgentResponse(
            agent=self.name, success=True,
            output="Nothing relevant found in memory for this session.",
            data={"found": False},
            correlation_id=request.correlation_id,
        )

    async def _write(self, request: AgentRequest, route: dict) -> AgentResponse:
        """Store a query+answer pair (used internally by Orchestrator)."""
        answer = request.metadata.get("answer", request.query)
        stored = await self.writer.write(
            key=route["key"],
            query=request.query,
            answer=answer,
            metadata={"correlation_id": request.correlation_id},
            session_id=route["session_id"],
        )
        return AgentResponse(
            agent=self.name, success=True,
            output="Stored." if stored else "Skipped (low quality or short answer).",
            data={"stored": stored, "key": route["key"]},
            correlation_id=request.correlation_id,
        )

    # ── Public helpers (called directly by Orchestrator) ─────────────────

    async def remember(self, session_id: str, query: str, answer: str) -> None:
        """Called by Orchestrator after every pipeline run to persist the answer."""
        route = self.mem_router.route(query, session_id=session_id)
        await self.writer.write(
            key=route["key"],
            query=query,
            answer=answer,
            session_id=session_id,
        )

    async def recall(self, session_id: str, query: str) -> str | None:
        """Called by Orchestrator before pipeline to inject prior context.

        Waterfall strategy:
          1. Redis exact key  → return immediately
          2. Redis broad scan → return most recent
          3. Postgres by slot → promote to Redis, return
          4. All miss         → return None
        """
        route = self.mem_router.route(query, session_id=session_id)

        # ── 1. Redis exact key ─────────────────────────────────────────────
        raw = await self._redis.get(route["key"])

        # ── 2. Redis broad scan ────────────────────────────────────────────
        if raw is None:
            sweep = await self.janitor.sweep(session_id)
            keys = sweep.get("keys", [])
            for k in reversed(keys):  # most recent first
                raw = await self._redis.get(k)
                if raw:
                    break

        if raw:
            try:
                stored = json.loads(raw)
                return stored.get("answer", raw)
            except (json.JSONDecodeError, TypeError):
                return raw

        # ── 3. Postgres warm tier ──────────────────────────────────────────
        if self._pg and self._pg.available:
            slot = route["slot"]
            row = await self._pg.fetchrow(
                """
                SELECT id, content FROM memory_chunks
                WHERE session_id = $1 AND slot = $2
                ORDER BY accessed_at DESC LIMIT 1
                """,
                session_id, slot,
            )
            if row:
                content = row["content"]
                chunk_id = row["id"]

                # Promote back to Redis hot cache
                await self._redis.set(route["key"], json.dumps({
                    "query": query,
                    "answer": content,
                    "metadata": {"promoted_from": "postgres"},
                }), ttl=86400)

                # Bump access tracking
                await self._pg.execute(
                    """
                    UPDATE memory_chunks
                    SET access_count = access_count + 1, accessed_at = NOW()
                    WHERE id = $1
                    """,
                    chunk_id,
                )
                logger.info("memory_promoted_postgres_to_redis", slot=slot, session_id=session_id)
                return content

        return None
