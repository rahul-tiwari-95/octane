"""OSA Orchestrator — the main loop.

Every user query flows through here. The Orchestrator:
1. Receives the query
2. Runs Guard (safety check)
3. Runs Decomposer (query → TaskDAG)
4. Runs Router (TaskNode → agent mapping)
5. Dispatches tasks to agents
6. Collects results
7. Runs Evaluator (quality gate)
8. Returns final output
"""

from __future__ import annotations

import uuid

import structlog

from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent, SynapseEventBus
from octane.osa.decomposer import Decomposer
from octane.osa.router import Router
from octane.osa.evaluator import Evaluator
from octane.osa.policy import PolicyEngine
from octane.osa.guard import Guard

logger = structlog.get_logger().bind(component="osa")


class Orchestrator:
    """The brain of Octane. Every query flows through here.

    Coordinates decomposition, routing, dispatch, evaluation,
    and emits Synapse events at every state transition.
    """

    def __init__(self, synapse: SynapseEventBus) -> None:
        self.synapse = synapse
        self.decomposer = Decomposer()
        self.router = Router(synapse)
        self.evaluator = Evaluator()
        self.policy = PolicyEngine()
        self.guard = Guard()

    async def run(self, query: str, session_id: str = "cli") -> str:
        """Main orchestration loop.

        Args:
            query: The user's query
            session_id: Session identifier for multi-turn context

        Returns:
            Final output string for the user
        """
        correlation_id = str(uuid.uuid4())

        # STEP 1: INGRESS — log the incoming query
        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="ingress",
            source="user",
            target="osa",
            payload={"query": query, "session_id": session_id},
        ))

        # STEP 2: GUARD — safety check on input
        guard_result = await self.guard.check_input(query)
        if not guard_result["safe"]:
            self.synapse.emit(SynapseEvent(
                correlation_id=correlation_id,
                event_type="guard_block",
                source="osa.guard",
                error=guard_result.get("reason", "Input blocked by guard"),
            ))
            return f"⚠ Query blocked: {guard_result.get('reason', 'Safety check failed')}"

        # STEP 3: DECOMPOSE — query → TaskDAG
        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="decomposition_start",
            source="osa.decomposer",
            payload={"query": query},
        ))

        dag = await self.decomposer.decompose(query)

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="decomposition_complete",
            source="osa.decomposer",
            payload={
                "nodes": len(dag.nodes),
                "reasoning": dag.reasoning,
            },
        ))

        # STEP 4: ROUTE & DISPATCH — execute each task via agents
        request = AgentRequest(
            query=query,
            correlation_id=correlation_id,
            session_id=session_id,
            source="osa",
        )

        results: list[AgentResponse] = []
        for wave in dag.execution_order():
            # In Phase 1: sequential execution
            # Phase 2+: asyncio.gather for parallel waves
            for node in wave:
                self.synapse.emit(SynapseEvent(
                    correlation_id=correlation_id,
                    event_type="dispatch",
                    source="osa.router",
                    target=node.agent,
                    payload={"task_id": node.task_id, "instruction": node.instruction},
                ))

                agent = self.router.get_agent(node.agent)
                if agent:
                    task_request = AgentRequest(
                        query=node.instruction,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        source="osa",
                        metadata=node.metadata,
                    )
                    response = await agent.run(task_request)
                    results.append(response)
                else:
                    logger.warning("no_agent_found", agent=node.agent)
                    results.append(AgentResponse(
                        agent=node.agent,
                        success=False,
                        error=f"No agent found for: {node.agent}",
                        correlation_id=correlation_id,
                    ))

        # STEP 5: EVALUATE — assemble and quality-gate the output
        output = await self.evaluator.evaluate(query, results)

        # STEP 6: EGRESS — log final output
        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="egress",
            source="osa",
            target="user",
            payload={
                "output_preview": output[:500],
                "agents_used": [r.agent for r in results],
                "all_succeeded": all(r.success for r in results),
            },
        ))

        return output
