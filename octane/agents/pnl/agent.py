"""P&L Agent coordinator.

Tracks user preferences and provides a profile to the Evaluator.
The profile shapes tone, verbosity, and detail level of every response.

Usage by Orchestrator:
    profile = await pnl_agent.get_profile(user_id)
    # pass profile into evaluator as style context
"""

from __future__ import annotations

import structlog

from octane.agents.base import BaseAgent
from octane.agents.pnl.preference_manager import PreferenceManager
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="pnl.agent")


class PnLAgent(BaseAgent):
    """P&L Agent — user personalization layer."""

    name = "pnl"

    def __init__(self, synapse, redis: RedisClient | None = None) -> None:
        super().__init__(synapse)
        redis = redis or RedisClient()
        self.prefs = PreferenceManager(redis=redis)

    async def get_profile(self, user_id: str = "default") -> dict[str, str]:
        """Return the full preference profile for a user."""
        return await self.prefs.get_all(user_id)

    async def set_preference(self, user_id: str, key: str, value: str) -> None:
        """Directly set a preference (called by CLI or chat commands)."""
        await self.prefs.set(user_id, key, value)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Handle explicit preference queries or updates from the pipeline.

        Supported commands via query:
            "set verbosity concise"
            "set expertise beginner"
            "show preferences"
        """
        user_id = request.metadata.get("user_id", "default")
        query = request.query.strip().lower()

        # Handle explicit set commands
        if query.startswith("set "):
            parts = query.split(None, 2)
            if len(parts) == 3:
                _, key, value = parts
                await self.prefs.set(user_id, key, value)
                return AgentResponse(
                    agent=self.name, success=True,
                    output=f"Preference updated: {key} = {value}",
                    correlation_id=request.correlation_id,
                )

        # Default: return current profile
        profile = await self.prefs.get_all(user_id)
        profile_text = "\n".join(f"  {k}: {v}" for k, v in profile.items())
        return AgentResponse(
            agent=self.name, success=True,
            output=f"User profile for '{user_id}':\n{profile_text}",
            data={"profile": profile},
            correlation_id=request.correlation_id,
        )
