"""Memory Agent coordinator.

Two operations:

WRITE (called by OSA Orchestrator after every successful response):
    MemoryRouter classifies + generates key
    MemoryWriter stores query+answer in Redis (with quality filter)

READ (called when query mentions recall/remember/last time etc.):
    MemoryRouter generates key
    Redis GET → return stored answer as context
    Janitor provides stats for debugging

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
    """Memory Agent — hot-tier (Redis) read/write with in-process fallback."""

    name = "memory"

    def __init__(self, synapse, redis: RedisClient | None = None) -> None:
        super().__init__(synapse)
        self._redis = redis or RedisClient()
        self.mem_router = MemoryRouter()
        self.writer = MemoryWriter(redis=self._redis)
        self.janitor = Janitor(redis=self._redis)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route to READ or WRITE based on query intent."""
        route = self.mem_router.route(request.query, session_id=request.session_id or "default")

        if route["intent"] == "read":
            return await self._read(request, route)
        else:
            # Direct write requests ("remember that ...", "store this ...")
            return await self._write(request, route)

    async def _read(self, request: AgentRequest, route: dict) -> AgentResponse:
        """Retrieve stored memory for this session."""
        # First try the exact key
        raw = await self._redis.get(route["key"])

        if raw is None:
            # Broad scan: find any key for this session
            sweep = await self.janitor.sweep(route["session_id"])
            all_keys = sweep.get("keys", [])

            if not all_keys:
                return AgentResponse(
                    agent=self.name, success=True,
                    output=f"No memories found for this session yet.",
                    data={"found": False},
                    correlation_id=request.correlation_id,
                )

            # Return the most recent key's content
            raw = await self._redis.get(all_keys[-1])

        if raw:
            try:
                stored = json.loads(raw)
                prior_q = stored.get("query", "")
                prior_a = stored.get("answer", raw)
                output = f"From memory — you previously asked: '{prior_q}'\nAnswer: {prior_a}"
            except (json.JSONDecodeError, TypeError):
                output = f"From memory: {raw}"

            return AgentResponse(
                agent=self.name, success=True,
                output=output,
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
        )

    async def recall(self, session_id: str, query: str) -> str | None:
        """Called by Orchestrator before pipeline to inject prior context."""
        route = self.mem_router.route(query, session_id=session_id)

        # Try exact key first
        raw = await self._redis.get(route["key"])
        if raw is None:
            # Scan session for any relevant key
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
        return None
