"""OSA Orchestrator — the main loop.

Every user query flows through here. The Orchestrator:
1. pre_flight  — checks Bodega is reachable, loads grunt model if needed
2. Guard       — safety check on input
3. Decompose   — query → TaskDAG (LLM-powered, keyword fallback)
4. Route       — TaskNode → agent instance
5. Dispatch    — execute tasks (parallel within wave, sequential across waves)
6. Evaluate    — LLM synthesis of all agent results
7. Egress      — Synapse event + return output

Session 10:
- conversation_history buffer for multi-turn chat continuity
- run_stream() accepts optional conversation_history
- DAG execution metadata surfaced in egress event for --verbose flag
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import structlog

from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent, SynapseEventBus
from octane.models.dag import TaskNode, TaskDAG
from octane.osa.decomposer import Decomposer
from octane.osa.router import Router
from octane.osa.evaluator import Evaluator
from octane.osa.policy import PolicyEngine
from octane.osa.guard import Guard
from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="osa")


class Orchestrator:
    """The brain of Octane. Every query flows through here."""

    def __init__(self, synapse: SynapseEventBus) -> None:
        self.synapse = synapse
        self.bodega = BodegaInferenceClient()
        self.decomposer = Decomposer(bodega=self.bodega)
        self.router = Router(synapse, bodega=self.bodega)
        self.evaluator = Evaluator(bodega=self.bodega)
        self.policy = PolicyEngine()
        self.guard = Guard()
        self._preflight_done = False
        # Memory agent reference (resolved lazily from router to avoid circular import)
        self._memory_agent = None
        self._pg_connected = False

    async def pre_flight(self) -> dict:
        """Check Bodega is reachable and a model is loaded.

        Called once before the first query. If Bodega is down, the
        Decomposer and Evaluator will gracefully fall back to
        keyword heuristics and string concatenation respectively.

        Returns a status dict for display in the CLI.
        """
        status = {"bodega_reachable": False, "model_loaded": False, "model": None, "note": ""}

        try:
            health = await self.bodega.health()
            if health.get("status") == "ok":
                status["bodega_reachable"] = True

                model_info = await self.bodega.current_model()
                if "error" not in model_info and model_info:
                    status["model_loaded"] = True
                    status["model"] = model_info.get("model_path") or model_info.get("model")
                else:
                    status["note"] = "Bodega reachable but no model loaded — LLM features disabled"
                    logger.warning("bodega_no_model_loaded")
            else:
                status["note"] = "Bodega server not reachable — using keyword fallback"
                logger.warning("bodega_unreachable")

        except Exception as exc:
            status["note"] = f"Bodega check failed: {exc} — using keyword fallback"
            logger.warning("bodega_preflight_error", error=str(exc))

        # Connect Postgres warm tier on first startup (non-blocking, graceful fallback)
        if not self._pg_connected:
            await self._connect_memory_pg()

        self._preflight_done = True

        self.synapse.emit(SynapseEvent(
            correlation_id="preflight",
            event_type="preflight",
            source="osa",
            payload=status,
        ))

        return status

    async def _connect_memory_pg(self) -> None:
        """Connect MemoryAgent to Postgres warm tier. Safe to call even if Postgres is down."""
        memory_agent = self._get_memory_agent()
        if memory_agent is not None:
            try:
                await memory_agent.connect_pg()
            except Exception as exc:
                # Never crash the pipeline over an infra connection failure.
                # MemoryAgent will operate in Redis-only fallback mode.
                logger.warning("memory_pg_connect_error", error=str(exc))
        self._pg_connected = True

    async def run(
        self,
        query: str,
        session_id: str = "cli",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Main orchestration loop.

        Args:
            query: The user's query.
            session_id: Session identifier for memory recall/write.
            conversation_history: Optional rolling buffer of prior turns
                [{"role": "user"|"assistant", "content": "..."}].
                When provided, injected into the Evaluator prompt so the
                LLM has direct multi-turn context (supplements memory recall).
        """
        # Run pre_flight lazily on first query if not already done
        if not self._preflight_done:
            await self.pre_flight()

        correlation_id = str(uuid.uuid4())

        # STEP 1: INGRESS
        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="ingress",
            source="user",
            target="osa",
            payload={"query": query, "session_id": session_id},
        ))

        # STEP 2: GUARD — safety check (fast, parallel with decompose in Phase 3+)
        guard_result = await self.guard.check_input(query)
        if not guard_result["safe"]:
            self.synapse.emit(SynapseEvent(
                correlation_id=correlation_id,
                event_type="guard_block",
                source="osa.guard",
                error=guard_result.get("reason", "Input blocked by guard"),
            ))
            return f"⚠ Query blocked: {guard_result.get('reason', 'Safety check failed')}"

        # STEP 3: DECOMPOSE
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
                "source": dag.nodes[0].metadata.get("source", "unknown") if dag.nodes else "unknown",
                "template": dag.nodes[0].metadata.get("template", "") if dag.nodes else "",
                # Full DAG serialisation — consumed by `octane workflow export`
                "dag_nodes_json": [n.model_dump() for n in dag.nodes],
                "dag_original_query": dag.original_query,
            },
        ))

        # GUARD RAIL: reject if the decomposer mapped to an agent we don't have
        for node in dag.nodes:
            if not self.router.get_agent(node.agent):
                logger.warning("unknown_agent_rejected", agent=node.agent, query=query[:80])
                return (
                    "I'm not sure how to help with that. "
                    "Try rephrasing, or ask about web search, code, news, finance, or system status."
                )

        # STEP 3.5: MEMORY RECALL — inject prior context for chat continuity
        memory_agent = self._get_memory_agent()
        prior_context: str | None = None
        if memory_agent and session_id != "cli":
            prior_context = await memory_agent.recall(session_id, query)
            if prior_context:
                logger.info("memory_recalled", session_id=session_id, preview=prior_context[:60])

        # STEP 3.6: USER PROFILE — fetch preferences to shape Evaluator tone
        user_profile: dict = {}
        pnl_agent = self.router.get_agent("pnl")
        if pnl_agent:
            try:
                user_id = session_id.split("_")[0] if "_" in session_id else "default"
                user_profile = await pnl_agent.get_profile(user_id)
            except Exception:
                pass

        # STEP 4: ROUTE & DISPATCH — parallel within each wave
        results: list[AgentResponse] = []
        accumulated: dict[str, AgentResponse] = {}

        for wave in dag.execution_order():
            # Build requests for this wave
            wave_requests = []
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
                    # DATA INJECTION: enrich instruction with upstream results
                    instruction = _inject_upstream_data(
                        node, accumulated, dag.original_query
                    )
                    task_request = AgentRequest(
                        query=instruction,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        source="osa",
                        metadata=node.metadata,
                    )
                    wave_requests.append((node, agent, task_request))
                else:
                    logger.warning("no_agent_found", agent=node.agent)
                    accumulated[node.task_id] = AgentResponse(
                        agent=node.agent,
                        success=False,
                        error=f"No agent registered for: {node.agent}",
                        correlation_id=correlation_id,
                    )

            # Execute wave in parallel
            if wave_requests:
                wave_responses = await asyncio.gather(
                    *[agent.run(req) for _, agent, req in wave_requests],
                    return_exceptions=False,
                )
                for (node, _, _), response in zip(wave_requests, wave_responses):
                    accumulated[node.task_id] = response
                    results.append(response)

        # STEP 5: EVALUATE — synthesize all results (inject prior memory context + user profile)
        output = await self.evaluator.evaluate(
            query, results,
            prior_context=prior_context,
            user_profile=user_profile,
            conversation_history=conversation_history,
        )

        # STEP 5.5: MEMORY WRITE — persist the answer for future recall
        if memory_agent and session_id != "cli":
            await memory_agent.remember(session_id, query, output)

        # STEP 6: EGRESS
        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="egress",
            source="osa",
            target="user",
            payload={
                "output_preview": output[:200],
                "agents_used": [r.agent for r in results],
                "tasks_total": len(results),
                "tasks_succeeded": sum(1 for r in results if r.success),
                "dag_nodes": len(dag.nodes),
                "dag_reasoning": dag.reasoning,
            },
        ))

        return output

    def _get_memory_agent(self):
        """Lazily resolve MemoryAgent from the router (avoids circular import)."""
        if self._memory_agent is None:
            self._memory_agent = self.router.get_agent("memory")
        return self._memory_agent

    async def run_stream(
        self,
        query: str,
        session_id: str = "cli",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Like run(), but streams Evaluator tokens as they arrive.

        Runs the full pipeline (guard → decompose → dispatch) synchronously,
        then streams the Evaluator output chunk-by-chunk.

        Args:
            query: The user's query.
            session_id: Session identifier for memory recall/write.
            conversation_history: Optional rolling buffer of prior turns for
                direct multi-turn context injection into the Evaluator.

        Yields:
            Text fragments from the LLM as they are generated.
            The final complete string is also written to memory.
        """
        if not self._preflight_done:
            await self.pre_flight()

        correlation_id = str(uuid.uuid4())

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="ingress",
            source="user",
            target="osa",
            payload={"query": query, "session_id": session_id, "mode": "stream"},
        ))

        # Guard
        guard_result = await self.guard.check_input(query)
        if not guard_result["safe"]:
            yield f"⚠ Query blocked: {guard_result.get('reason', 'Safety check failed')}"
            return

        # Decompose
        dag = await self.decomposer.decompose(query)

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="decomposition_complete",
            source="osa.decomposer",
            payload={
                "nodes": len(dag.nodes),
                "reasoning": dag.reasoning,
                "source": dag.nodes[0].metadata.get("source", "unknown") if dag.nodes else "unknown",
                "template": dag.nodes[0].metadata.get("template", "") if dag.nodes else "",
                "dag_nodes_json": [n.model_dump() for n in dag.nodes],
                "dag_original_query": dag.original_query,
            },
        ))

        # Guard rail: unknown agent
        for node in dag.nodes:
            if not self.router.get_agent(node.agent):
                logger.warning("unknown_agent_rejected", agent=node.agent, query=query[:80])
                yield (
                    "I'm not sure how to help with that. "
                    "Try rephrasing, or ask about web search, code, news, finance, or system status."
                )
                return

        # Memory recall
        memory_agent = self._get_memory_agent()
        prior_context: str | None = None
        if memory_agent and session_id != "cli":
            prior_context = await memory_agent.recall(session_id, query)
            if prior_context:
                logger.info("memory_recalled", session_id=session_id, preview=prior_context[:60])

        # User profile
        user_profile: dict = {}
        pnl_agent = self.router.get_agent("pnl")
        if pnl_agent:
            try:
                user_id = session_id.split("_")[0] if "_" in session_id else "default"
                user_profile = await pnl_agent.get_profile(user_id)
            except Exception:
                pass

        # Dispatch
        results: list[AgentResponse] = []
        accumulated: dict[str, AgentResponse] = {}

        for wave in dag.execution_order():
            wave_requests = []
            for node in wave:
                agent = self.router.get_agent(node.agent)
                if agent:
                    instruction = _inject_upstream_data(
                        node, accumulated, dag.original_query
                    )
                    task_request = AgentRequest(
                        query=instruction,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        source="osa",
                        metadata=node.metadata,
                    )
                    wave_requests.append((node, agent, task_request))
                else:
                    accumulated[node.task_id] = AgentResponse(
                        agent=node.agent, success=False,
                        error=f"No agent registered for: {node.agent}",
                        correlation_id=correlation_id,
                    )

            if wave_requests:
                wave_responses = await asyncio.gather(
                    *[agent.run(req) for _, agent, req in wave_requests],
                )
                for (node, _, _), response in zip(wave_requests, wave_responses):
                    accumulated[node.task_id] = response
                    results.append(response)

        # Stream evaluate — collect full output for memory write
        full_output_parts: list[str] = []
        async for chunk in self.evaluator.evaluate_stream(
            query, results,
            prior_context=prior_context,
            user_profile=user_profile,
            conversation_history=conversation_history,
        ):
            full_output_parts.append(chunk)
            yield chunk

        full_output = "".join(full_output_parts).strip()

        # Memory write
        if memory_agent and session_id != "cli":
            await memory_agent.remember(session_id, query, full_output)

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="egress",
            source="osa",
            target="user",
            payload={
                "output_preview": full_output[:200],
                "agents_used": [r.agent for r in results],
                "mode": "stream",
                "dag_nodes": len(dag.nodes),
                "dag_reasoning": dag.reasoning,
            },
        ))

    async def run_from_dag(
        self,
        dag: TaskDAG,
        query: str,
        session_id: str = "cli",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Execute a pre-built TaskDAG, bypassing Guard and Decomposer.

        Used by ``octane workflow run`` to replay a saved workflow template.
        The DAG's nodes are dispatched directly through the Router → Evaluator
        pipeline.  Guard is skipped because the template was already vetted
        when it was first exported.

        Args:
            dag: Pre-built TaskDAG (from WorkflowTemplate.to_dag()).
            query: The user-facing query string (shown in egress event).
            session_id: Session ID for memory read/write.
            conversation_history: Optional multi-turn context buffer.

        Yields:
            Text fragments from the Evaluator, same as run_stream().
        """
        if not self._preflight_done:
            await self.pre_flight()

        correlation_id = str(uuid.uuid4())

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="ingress",
            source="user",
            target="osa",
            payload={"query": query, "session_id": session_id, "mode": "workflow"},
        ))

        # Validate agents exist
        for node in dag.nodes:
            if not self.router.get_agent(node.agent):
                yield (
                    f"⚠ Workflow references unknown agent '{node.agent}'. "
                    "Check the template and re-export."
                )
                return

        # Memory + profile (same as run_stream)
        memory_agent = self._get_memory_agent()
        prior_context: str | None = None
        if memory_agent and session_id != "cli":
            prior_context = await memory_agent.recall(session_id, query)

        user_profile: dict = {}
        pnl_agent = self.router.get_agent("pnl")
        if pnl_agent:
            try:
                user_id = session_id.split("_")[0] if "_" in session_id else "default"
                user_profile = await pnl_agent.get_profile(user_id)
            except Exception:
                pass

        # Dispatch — parallel within each wave
        results: list[AgentResponse] = []
        accumulated: dict[str, AgentResponse] = {}

        for wave in dag.execution_order():
            wave_requests = []
            for node in wave:
                agent = self.router.get_agent(node.agent)
                if agent:
                    instruction = _inject_upstream_data(node, accumulated, query)
                    self.synapse.emit(SynapseEvent(
                        correlation_id=correlation_id,
                        event_type="dispatch",
                        source="osa.router",
                        target=node.agent,
                        payload={"task_id": node.task_id, "instruction": node.instruction},
                    ))
                    task_request = AgentRequest(
                        query=instruction,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        source="osa",
                        metadata=node.metadata,
                    )
                    wave_requests.append((node, agent, task_request))
                else:
                    accumulated[node.task_id] = AgentResponse(
                        agent=node.agent, success=False,
                        error=f"No agent: {node.agent}",
                        correlation_id=correlation_id,
                    )

            if wave_requests:
                wave_responses = await asyncio.gather(
                    *[agent.run(req) for _, agent, req in wave_requests],
                )
                for (node, _, _), response in zip(wave_requests, wave_responses):
                    accumulated[node.task_id] = response
                    results.append(response)

        # Evaluate → stream
        full_output_parts: list[str] = []
        async for chunk in self.evaluator.evaluate_stream(
            query, results,
            prior_context=prior_context,
            user_profile=user_profile,
            conversation_history=conversation_history,
        ):
            full_output_parts.append(chunk)
            yield chunk

        full_output = "".join(full_output_parts).strip()

        if memory_agent and session_id != "cli":
            await memory_agent.remember(session_id, query, full_output)

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="egress",
            source="osa",
            target="user",
            payload={
                "output_preview": full_output[:200],
                "agents_used": [r.agent for r in results],
                "mode": "workflow",
                "dag_nodes": len(dag.nodes),
                "dag_reasoning": dag.reasoning,
            },
        ))


# ── Module-level helpers ──────────────────────────────────────────────────────

def _inject_upstream_data(
    node: TaskNode,
    accumulated: dict[str, AgentResponse],
    original_query: str,
) -> str:
    """Build the instruction for a node, injecting results from its dependencies.

    For root nodes (no depends_on): instruction is unchanged.
    For dependent nodes: upstream outputs are prepended as DATA CONTEXT so the
    agent (especially CodeAgent) has real data to work with instead of guessing.
    """
    if not node.depends_on:
        return node.instruction

    upstream_parts: list[str] = []
    for dep_id in node.depends_on:
        dep_response = accumulated.get(dep_id)
        if dep_response and dep_response.success and dep_response.output:
            output = dep_response.output.strip()
            if len(output) > 800:
                output = output[:800] + "\n... [truncated]"
            upstream_parts.append(f"[Data from {dep_response.agent} agent]\n{output}")

    if not upstream_parts:
        return node.instruction

    context_block = "\n\n".join(upstream_parts)
    return (
        f"{node.instruction}\n\n"
        f"Use the following real data retrieved from upstream agents:\n\n"
        f"{context_block}"
    )
