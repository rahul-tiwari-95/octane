"""OSA Router — TaskNode → agent instance mapping.

Deterministic routing. Maps agent names to agent instances.
"""

from __future__ import annotations

import structlog

from octane.agents.base import BaseAgent
from octane.agents.web.agent import WebAgent
from octane.agents.code.agent import CodeAgent
from octane.agents.memory.agent import MemoryAgent
from octane.agents.sysstat.agent import SysStatAgent
from octane.agents.pnl.agent import PnLAgent
from octane.models.synapse import SynapseEventBus
from octane.tools.bodega_inference import BodegaInferenceClient
from octane.tools.bodega_intel import BodegaIntelClient
from octane.tools.redis_client import RedisClient
from octane.tools.pg_client import PgClient
from octane.tools.structured_store import ArtifactStore, WebPageStore

logger = structlog.get_logger().bind(component="osa.router")


class Router:
    """Maps agent names to agent instances."""

    def __init__(self, synapse: SynapseEventBus, bodega: BodegaInferenceClient | None = None) -> None:
        self.synapse = synapse
        self.bodega = bodega or BodegaInferenceClient()
        self.intel = BodegaIntelClient()
        self.redis = RedisClient()
        self.pg = PgClient()
        self._page_store = WebPageStore(self.pg)
        self._artifact_store = ArtifactStore(self.pg)

        self._agents: dict[str, BaseAgent] = {
            "web": WebAgent(synapse, intel=self.intel, bodega=self.bodega,
                            page_store=self._page_store),
            "code": CodeAgent(synapse, bodega=self.bodega,
                              artifact_store=self._artifact_store),
            "memory": MemoryAgent(synapse, redis=self.redis),
            "sysstat": SysStatAgent(synapse, self.bodega),
            "pnl": PnLAgent(synapse, redis=self.redis),
        }

    def get_agent(self, name: str) -> BaseAgent | None:
        """Resolve an agent name to an instance."""
        agent = self._agents.get(name)
        if not agent:
            logger.warning("unknown_agent", name=name)
        return agent

    def list_agents(self) -> list[str]:
        """Return all registered agent names."""
        return list(self._agents.keys())
