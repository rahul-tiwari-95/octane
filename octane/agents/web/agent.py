"""Web Agent coordinator.

Orchestrates sub-agents: QueryStrategist → Fetcher → Synthesizer.
Handles finance, news, search, and entertainment queries.
"""

from __future__ import annotations

from octane.agents.base import BaseAgent
from octane.models.schemas import AgentRequest, AgentResponse


class WebAgent(BaseAgent):
    """Web Agent — coordinates internet data retrieval.

    Sub-agents:
        - QueryStrategist: generates search variations
        - Fetcher: calls Bodega Intel APIs
        - Browser: Playwright for JS-heavy sites (stub)
        - Synthesizer: raw data → structured intelligence
    """

    name = "web"

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Phase 1 stub — returns placeholder."""
        return AgentResponse(
            agent=self.name,
            success=True,
            output=f"[Web Agent stub] Would search for: {request.query}",
            data={"status": "stub"},
        )
