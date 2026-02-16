"""P&L Agent coordinator.

Tracks user preferences, feedback, and engagement.
Consulted by OSA.Evaluator before generating final output.
"""

from __future__ import annotations

from octane.agents.base import BaseAgent
from octane.models.schemas import AgentRequest, AgentResponse


class PnLAgent(BaseAgent):
    """P&L Agent — user personalization and learning.

    Sub-agents:
        - PreferenceManager: CRUD on user preferences
        - FeedbackLearner: processes like/dislike/time-spent signals
        - Profile: aggregates preferences into a profile dict
    """

    name = "pnl"

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Phase 1 stub — returns default user profile."""
        return AgentResponse(
            agent=self.name,
            success=True,
            output="Default user profile",
            data={
                "status": "stub",
                "profile": {
                    "user_id": "default",
                    "expertise_level": "advanced",
                    "preferred_verbosity": "concise",
                    "domains": ["technology", "finance"],
                },
            },
        )
