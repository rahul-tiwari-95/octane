"""Memory Agent coordinator.

Orchestrates: Router → (Hot/Warm/Cold tier) + Writer + Janitor.
Hot = Redis, Warm = Postgres, Cold = pgVector.
"""

from __future__ import annotations

from octane.agents.base import BaseAgent
from octane.models.schemas import AgentRequest, AgentResponse


class MemoryAgent(BaseAgent):
    """Memory Agent — coordinates three-tier memory system.

    Sub-agents:
        - Router: decides which tier to query
        - Writer: evaluates what to persist and where
        - Janitor: promotes/demotes between tiers (background)
    """

    name = "memory"

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Phase 1 stub — returns placeholder."""
        return AgentResponse(
            agent=self.name,
            success=True,
            output=f"[Memory Agent stub] Would search memory for: {request.query}",
            data={"status": "stub"},
        )
