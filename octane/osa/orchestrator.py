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

Session 16:
- HILManager wired: PolicyEngine.assess_dag() → HILManager.review_ledger()
- CheckpointManager creates plan checkpoint after decomposition
- run() and run_stream() emit hil_summary in egress events
- HIL is non-interactive by default (auto-approves) — octane chat sets
  interactive=True to enable human review prompts
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent, SynapseEventBus
from octane.models.dag import TaskNode, TaskDAG
from octane.osa.decomposer import Decomposer
from octane.osa.router import Router
from octane.osa.evaluator import Evaluator
from octane.osa.policy import PolicyEngine
from octane.osa.guard import Guard
from octane.osa.hil_manager import HILManager
from octane.osa.checkpoint_manager import CheckpointManager
from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="osa")

# ── Timeout constants (module-level so tests can monkeypatch them) ────────────
# Per-chunk timeout for the eval stream. 120 s accommodates slow reasoning models
# with long <think> blocks. Tests monkeypatch this to 0.1 s for fast execution.
_EVAL_CHUNK_TIMEOUT: float = 120.0
# Safety cap on _eval_gen.aclose() so a hung drain never blocks a cancellation.
_EVAL_ACLOSE_TIMEOUT: float = 5.0


class Orchestrator:
    """The brain of Octane. Every query flows through here."""

    def __init__(self, synapse: SynapseEventBus, hil_interactive: bool = False) -> None:
        self.synapse = synapse
        self.bodega = BodegaInferenceClient()
        self.decomposer = Decomposer(bodega=self.bodega)
        self.router = Router(synapse, bodega=self.bodega)
        self.evaluator = Evaluator(bodega=self.bodega)
        self.policy = PolicyEngine()
        self.guard = Guard()
        self.hil = HILManager(interactive=hil_interactive)
        self.checkpoint_mgr = CheckpointManager()
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

        # STEP 3.5: HIL — assess DAG risk, create plan checkpoint, review decisions
        ledger = self.policy.assess_dag(dag)
        await self.checkpoint_mgr.create(
            correlation_id=correlation_id,
            dag=dag,
            results={},
            decisions=ledger.decisions,
            checkpoint_type="plan",
        )
        await self.hil.review_ledger(ledger, user_profile={})

        self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="hil_complete",
            source="osa.hil",
            payload=ledger.summary(),
        ))

        # STEP 3.6: MEMORY RECALL — inject prior context for chat continuity
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
                    # DATA INJECTION: enrich instruction with upstream results (text + structured)
                    instruction, upstream_results = _inject_upstream_data(
                        node, accumulated, dag.original_query
                    )
                    task_request = AgentRequest(
                        query=instruction,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        source="osa",
                        metadata=node.metadata,
                        context={"upstream_results": upstream_results},
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

            # Execute wave in parallel — 90 s ceiling so a hung agent never blocks forever
            if wave_requests:
                try:
                    wave_responses = await asyncio.wait_for(
                        asyncio.gather(
                            *[agent.run(req) for _, agent, req in wave_requests],
                            return_exceptions=False,
                        ),
                        timeout=90.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "wave_dispatch_timeout",
                        wave_agents=[n.agent for n, _, _ in wave_requests],
                    )
                    wave_responses = [
                        AgentResponse(
                            agent=n.agent,
                            success=False,
                            error="Agent timed out (90 s ceiling)",
                            correlation_id=correlation_id,
                        )
                        for n, _, _ in wave_requests
                    ]
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
                "hil_summary": ledger.summary(),
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

        # HIL — assess DAG risk + checkpoint (same as run())
        ledger = self.policy.assess_dag(dag)
        await self.checkpoint_mgr.create(
            correlation_id=correlation_id,
            dag=dag,
            results={},
            decisions=ledger.decisions,
            checkpoint_type="plan",
        )
        await self.hil.review_ledger(ledger, user_profile={})

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
                    instruction, upstream_results = _inject_upstream_data(
                        node, accumulated, dag.original_query
                    )
                    task_request = AgentRequest(
                        query=instruction,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        source="osa",
                        metadata=node.metadata,
                        context={"upstream_results": upstream_results},
                    )
                    wave_requests.append((node, agent, task_request))
                else:
                    accumulated[node.task_id] = AgentResponse(
                        agent=node.agent, success=False,
                        error=f"No agent registered for: {node.agent}",
                        correlation_id=correlation_id,
                    )

            if wave_requests:
                try:
                    wave_responses = await asyncio.wait_for(
                        asyncio.gather(
                            *[agent.run(req) for _, agent, req in wave_requests],
                        ),
                        timeout=90.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "wave_dispatch_timeout",
                        wave_agents=[n.agent for n, _, _ in wave_requests],
                    )
                    wave_responses = [
                        AgentResponse(
                            agent=n.agent,
                            success=False,
                            error="Agent timed out (90 s ceiling)",
                            correlation_id=correlation_id,
                        )
                        for n, _, _ in wave_requests
                    ]
                for (node, _, _), response in zip(wave_requests, wave_responses):
                    accumulated[node.task_id] = response
                    results.append(response)

        # Stream evaluate — collect full output for memory write.
        # asyncio.timeout() inside an async generator is unreliable in Python
        # 3.13 when the generator is suspended between yields.  Instead we put
        # a hard wall-clock cap on each individual __anext__() call, which is a
        # plain coroutine that asyncio.wait_for() handles correctly.
        full_output_parts: list[str] = []
        _eval_gen = self.evaluator.evaluate_stream(
            query, results,
            prior_context=prior_context,
            user_profile=user_profile,
            conversation_history=conversation_history,
        )
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        _eval_gen.__anext__(),
                        timeout=_EVAL_CHUNK_TIMEOUT,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        "eval_stream_chunk_timeout",
                        timeout=_EVAL_CHUNK_TIMEOUT,
                        partial_words=len("".join(full_output_parts).split()),
                    )
                    # Fall back to raw concatenation so we always produce output
                    if not full_output_parts:
                        concat = "\n\n---\n\n".join(r.output for r in results if r.output)
                        full_output_parts.append(concat)
                        yield concat
                    break
                full_output_parts.append(chunk)
                yield chunk
        finally:
            try:
                await asyncio.wait_for(_eval_gen.aclose(), timeout=_EVAL_ACLOSE_TIMEOUT)
            except Exception:
                pass

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
                "word_count": len(full_output.split()),
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
                    instruction, upstream_results = _inject_upstream_data(node, accumulated, query)
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
                        context={"upstream_results": upstream_results},
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

        # Evaluate → stream (same per-__anext__ timeout guard as run_stream)
        full_output_parts: list[str] = []
        _eval_gen = self.evaluator.evaluate_stream(
            query, results,
            prior_context=prior_context,
            user_profile=user_profile,
            conversation_history=conversation_history,
        )
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        _eval_gen.__anext__(),
                        timeout=_EVAL_CHUNK_TIMEOUT,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        "eval_stream_chunk_timeout",
                        timeout=_EVAL_CHUNK_TIMEOUT,
                        partial_words=len("".join(full_output_parts).split()),
                    )
                    if not full_output_parts:
                        concat = "\n\n---\n\n".join(r.output for r in results if r.output)
                        full_output_parts.append(concat)
                        yield concat
                    break
                full_output_parts.append(chunk)
                yield chunk
        finally:
            try:
                await asyncio.wait_for(_eval_gen.aclose(), timeout=_EVAL_ACLOSE_TIMEOUT)
            except Exception:
                pass

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
) -> tuple[str, dict[str, Any]]:
    """Build the instruction and structured upstream data for a node.

    Returns:
        (instruction_str, upstream_results_dict)

        instruction_str — enriched text instruction with upstream outputs
            prepended as DATA CONTEXT (consumed by the LLM pipeline).
        upstream_results_dict — raw structured results keyed by dep node_id
            (consumed by CatalystRegistry; bypasses LLM entirely).

    For root nodes (no depends_on): instruction is unchanged, upstream_results empty.
    """
    if not node.depends_on:
        return node.instruction, {}

    upstream_parts: list[str] = []
    upstream_results: dict[str, Any] = {}

    for dep_id in node.depends_on:
        dep_response = accumulated.get(dep_id)
        if dep_response and dep_response.success:
            # Structured payload — whatever the upstream agent returned as data
            if dep_response.data:
                upstream_results[dep_id] = dep_response.data
            elif dep_response.output:
                # Fallback: agents that only set output (no structured data)
                upstream_results[dep_id] = {"output": dep_response.output}
            # Text payload — for LLM context injection
            if dep_response.output:
                output = dep_response.output.strip()
                if len(output) > 800:
                    output = output[:800] + "\n... [truncated]"
                upstream_parts.append(f"[Data from {dep_response.agent} agent]\n{output}")

    if not upstream_parts:
        return node.instruction, upstream_results

    context_block = "\n\n".join(upstream_parts)
    instruction = (
        f"{node.instruction}\n\n"
        f"Use the following real data retrieved from upstream agents:\n\n"
        f"{context_block}"
    )
    return instruction, upstream_results
