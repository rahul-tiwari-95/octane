"""BaseAgent — abstract base class for ALL Octane agents.

Every agent (Web, Code, Memory, SysStat, P&L) extends this.
Provides timing, error handling, and Synapse event emission.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import structlog

from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent, SynapseEventBus

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Base class for all Octane agents.

    Subclasses must:
        1. Set `name` class attribute (e.g., "web", "code", "memory")
        2. Implement `execute(request)` with the agent's core logic

    The `run()` method wraps `execute()` with:
        - Timing
        - Error handling
        - Synapse event emission (agent_start, agent_complete, agent_error)
    """

    name: str = "base"

    def __init__(self, synapse: SynapseEventBus) -> None:
        self.synapse = synapse
        self.log = logger.bind(agent=self.name)

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Execute the agent with timing, error handling, and Synapse tracing.

        This is the public entry point. Do NOT override this — override execute().
        """
        start = time.perf_counter()

        # Emit start event
        self.synapse.emit(SynapseEvent(
            correlation_id=request.correlation_id,
            event_type="agent_start",
            source=self.name,
            payload={"query": request.query},
        ))

        try:
            response = await self.execute(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            response.duration_ms = duration_ms
            response.correlation_id = request.correlation_id

            # Emit completion event
            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id,
                event_type="agent_complete",
                source=self.name,
                duration_ms=duration_ms,
                payload={"success": response.success, "output_preview": response.output[:200]},
            ))

            self.log.info(
                "agent_complete",
                correlation_id=request.correlation_id,
                duration_ms=duration_ms,
                success=response.success,
            )
            return response

        except Exception as e:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)

            # Emit error event
            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id,
                event_type="agent_error",
                source=self.name,
                error=str(e),
                duration_ms=duration_ms,
            ))

            self.log.error(
                "agent_error",
                correlation_id=request.correlation_id,
                error=str(e),
                duration_ms=duration_ms,
            )

            return AgentResponse(
                agent=self.name,
                success=False,
                error=str(e),
                correlation_id=request.correlation_id,
                duration_ms=duration_ms,
            )

    @abstractmethod
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Core agent logic. Subclasses MUST implement this.

        Args:
            request: The standardized agent request

        Returns:
            AgentResponse with results
        """
        ...
