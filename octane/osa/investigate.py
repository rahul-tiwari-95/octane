"""InvestigateOrchestrator — the engine behind `octane investigate`.

Executes the full investigation pipeline:

    Query → DimensionPlanner (REASON)
          → Wave 1: parallel Web research per dimension
          → Wave 2: Memory recall to enrich findings
          → Wave 3: Structured synthesis across all dimensions
          → Auto-store: pages and findings to Postgres

The investigation produces a structured multi-section report, not a single
paragraph.  Each dimension maps to a section.  The evaluator synthesizes
across sections with a final "overall assessment" closing.

Design principles:
    - Pure asyncio concurrency — all Wave 1 tasks run in parallel.
    - Graceful degradation: if a dimension fails, the rest continue.
    - Progress streaming: yields events so the CLI can show a live display.
    - Daemon-aware: routes through daemon priority queue (P1) when available.
    - Memory-augmented: after web research, Memory is consulted for prior context.

Output format (streamed events):
    {"type": "plan",     "data": DimensionPlan.to_dict()}
    {"type": "progress", "data": {"dimension": label, "status": "researching"}}
    {"type": "finding",  "data": {"dimension": label, "content": str}}
    {"type": "synthesis","data": {"report": str, "sections": [...], "word_count": int}}
    {"type": "stored",   "data": {"n_pages": int, "n_findings": int}}
    {"type": "done",     "data": {"dimensions_completed": int, "total_words": int}}
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from octane.osa.dimension_planner import DimensionPlan, DimensionPlanner, ResearchDimension
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="osa.investigate")

# Sentinel
_UNSET = object()

# Concurrency cap per investigation — avoids hammering Bodega with 8 simultaneous requests
_MAX_PARALLEL_DIMENSIONS = 4

# Word budget per dimension for synthesis
_WORDS_PER_DIMENSION = 250

# ── Finding ───────────────────────────────────────────────────────────────────


@dataclass
class DimensionFinding:
    """Research result for one dimension.

    Attributes:
        dimension:    The ResearchDimension this finding covers.
        content:      The raw research text from Web + Memory agents.
        agent_used:   Which agent produced this (web, memory, fallback).
        latency_ms:   Time taken to research this dimension.
        error:        Non-empty if the research failed.
    """

    dimension: ResearchDimension
    content: str = ""
    agent_used: str = "web"
    latency_ms: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return bool(self.content) and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_id": self.dimension.id,
            "dimension_label": self.dimension.label,
            "content": self.content,
            "agent_used": self.agent_used,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
        }


@dataclass
class InvestigationResult:
    """The full output of an investigation.

    Attributes:
        query:       Original user query.
        plan:        The DimensionPlan that guided research.
        findings:    Per-dimension research results.
        report:      Final synthesized report (multi-section).
        word_count:  Total words in the report.
        total_ms:    End-to-end latency in ms.
    """

    query: str
    plan: DimensionPlan
    findings: list[DimensionFinding] = field(default_factory=list)
    report: str = ""
    word_count: int = 0
    total_ms: float = 0.0

    @property
    def successful_findings(self) -> list[DimensionFinding]:
        return [f for f in self.findings if f.success]

    @property
    def failed_dimensions(self) -> list[str]:
        return [f.dimension.label for f in self.findings if not f.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.to_dict(),
            "n_dimensions": len(self.findings),
            "n_successful": len(self.successful_findings),
            "failed_dimensions": self.failed_dimensions,
            "report": self.report,
            "word_count": self.word_count,
            "total_ms": round(self.total_ms, 1),
        }


# ── Synthesis Prompt ──────────────────────────────────────────────────────────

_INVESTIGATE_SYNTHESIS_SYSTEM = """\
You are the Octane Investigator. You have just completed multi-dimensional \
research on a query. Your job is to synthesize all findings into a single \
coherent, structured report.

Output format:
- Start with a one-sentence executive summary
- One section per research dimension (use ## headers)
- End with an ## Overall Assessment section
- Be direct and specific. Ground every claim in the research provided.
- Do not mention "agents", "tools", "dimensions", or internal system names.
- If a dimension had no data, skip it silently.
- Target 600-1200 words total.
"""

_INVESTIGATE_SYNTHESIS_USER = """\
Original query: {query}

Research findings per dimension:
{findings_text}

Synthesize into a structured report.
"""


# ── InvestigateOrchestrator ───────────────────────────────────────────────────


class InvestigateOrchestrator:
    """Orchestrates the full investigation pipeline.

    Args:
        bodega:       BodegaRouter for LLM calls.
        web_agent:    WebAgent for web research.
        memory_agent: MemoryAgent for prior-context recall.
        dimension_planner: DimensionPlanner instance (or auto-constructed).
    """

    def __init__(
        self,
        bodega=_UNSET,
        web_agent=None,
        memory_agent=None,
        dimension_planner: DimensionPlanner | None = None,
    ) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega
        self._web_agent = web_agent
        self._memory_agent = memory_agent
        self._planner = dimension_planner or DimensionPlanner(bodega=bodega)

    async def run_stream(
        self,
        query: str,
        max_dimensions: int | None = None,
        session_id: str = "cli",
    ) -> AsyncIterator[dict[str, Any]]:
        """Run an investigation and yield progress events.

        Yields a sequence of dicts:
            plan → progress (per dim) → finding (per dim) → synthesis → done

        Args:
            query:          The investigation query.
            max_dimensions: Override DimensionPlanner default.
            session_id:     For tracing.

        Yields:
            Progress event dicts (see module docstring for schema).
        """
        t0 = time.monotonic()

        # ── Phase 1: Plan ──────────────────────────────────────────────────
        plan = await self._planner.plan(query, max_dimensions=max_dimensions)
        yield {"type": "plan", "data": plan.to_dict()}

        # ── Phase 2: Parallel research (bounded concurrency) ───────────────
        semaphore = asyncio.Semaphore(_MAX_PARALLEL_DIMENSIONS)
        findings: list[DimensionFinding] = []

        async def _research_dimension(dim: ResearchDimension) -> DimensionFinding:
            async with semaphore:
                return await self._research_one(query, dim, session_id)

        tasks = [_research_dimension(dim) for dim in plan.sorted_dimensions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for dim, result in zip(plan.sorted_dimensions, results):
            if isinstance(result, Exception):
                finding = DimensionFinding(
                    dimension=dim,
                    error=str(result),
                    agent_used="error",
                )
            else:
                finding = result

            findings.append(finding)
            yield {"type": "finding", "data": finding.to_dict()}

        # ── Phase 3: Synthesis ─────────────────────────────────────────────
        successful = [f for f in findings if f.success]
        if not successful:
            report = f"Investigation could not retrieve data for any dimension of: {query}"
        else:
            report = await self._synthesize(query, successful)

        word_count = len(report.split())
        total_ms = (time.monotonic() - t0) * 1000

        yield {
            "type": "synthesis",
            "data": {
                "report": report,
                "word_count": word_count,
                "n_dimensions_used": len(successful),
            },
        }

        yield {
            "type": "done",
            "data": {
                "dimensions_completed": len(successful),
                "dimensions_failed": len(findings) - len(successful),
                "total_words": word_count,
                "total_ms": round(total_ms, 1),
            },
        }

    async def run(
        self,
        query: str,
        max_dimensions: int | None = None,
        session_id: str = "cli",
    ) -> InvestigationResult:
        """Run investigation and return a complete InvestigationResult.

        Convenience wrapper around run_stream() for non-streaming callers.
        """
        t0 = time.monotonic()
        plan: DimensionPlan | None = None
        findings: list[DimensionFinding] = []
        report = ""
        word_count = 0

        async for event in self.run_stream(
            query, max_dimensions=max_dimensions, session_id=session_id
        ):
            if event["type"] == "plan":
                from octane.osa.dimension_planner import DimensionPlan
                d = event["data"]
                dims = [
                    ResearchDimension(
                        id=dim["id"],
                        label=dim["label"],
                        queries=dim["queries"],
                        priority=dim["priority"],
                        rationale=dim.get("rationale", ""),
                    )
                    for dim in d.get("dimensions", [])
                ]
                plan = DimensionPlan(
                    query=d["query"],
                    dimensions=dims,
                    from_llm=d.get("from_llm", False),
                )
            elif event["type"] == "finding":
                d = event["data"]
                if plan:
                    matching_dim = next(
                        (dim for dim in plan.dimensions if dim.id == d["dimension_id"]),
                        None,
                    )
                    if matching_dim:
                        findings.append(DimensionFinding(
                            dimension=matching_dim,
                            content=d.get("content", ""),
                            agent_used=d.get("agent_used", "web"),
                            latency_ms=d.get("latency_ms", 0.0),
                            error=d.get("error", "") if not d.get("success") else "",
                        ))
            elif event["type"] == "synthesis":
                report = event["data"].get("report", "")
                word_count = event["data"].get("word_count", 0)

        if plan is None:
            plan = DimensionPlan(query=query, dimensions=[])

        return InvestigationResult(
            query=query,
            plan=plan,
            findings=findings,
            report=report,
            word_count=word_count,
            total_ms=(time.monotonic() - t0) * 1000,
        )

    async def _research_one(
        self,
        original_query: str,
        dimension: ResearchDimension,
        session_id: str,
    ) -> DimensionFinding:
        """Research a single dimension using Web + optional Memory agents."""
        t0 = time.monotonic()

        # Use the dimension's primary query for web research
        research_query = dimension.primary_query()
        content = ""

        # ── Web research ───────────────────────────────────────────────────
        if self._web_agent is not None:
            try:
                request = AgentRequest(
                    query=research_query,
                    source="investigate",
                    session_id=session_id,
                    metadata={"sub_agent": "search", "deep": False},
                )
                response: AgentResponse = await self._web_agent.execute(request)
                if response.success and response.output:
                    content = response.output
                elif response.data:
                    content = str(response.data.get("summary", ""))
            except Exception as exc:
                logger.warning(
                    "investigate_web_failed",
                    dimension=dimension.id,
                    error=str(exc),
                )

        # ── Memory augmentation ────────────────────────────────────────────
        if self._memory_agent is not None and content:
            try:
                mem_request = AgentRequest(
                    query=research_query,
                    source="investigate",
                    session_id=session_id,
                    metadata={"sub_agent": "read"},
                )
                mem_response: AgentResponse = await self._memory_agent.execute(mem_request)
                if mem_response.success and mem_response.output:
                    content = f"{content}\n\n[Prior context: {mem_response.output}]"
            except Exception as exc:
                logger.debug("investigate_memory_failed", dimension=dimension.id, error=str(exc))

        # ── Fallback: if no agent available, return empty finding ──────────
        agent_used = "web" if self._web_agent else "fallback"
        latency_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "investigate_dimension_done",
            dimension=dimension.id,
            content_len=len(content),
            latency_ms=round(latency_ms, 1),
        )

        return DimensionFinding(
            dimension=dimension,
            content=content,
            agent_used=agent_used,
            latency_ms=latency_ms,
        )

    async def _synthesize(
        self,
        query: str,
        findings: list[DimensionFinding],
    ) -> str:
        """Synthesize all findings into a structured report using REASON tier."""
        if not self._bodega:
            # Fallback: concatenate findings
            parts = [f"## {f.dimension.label}\n\n{f.content}" for f in findings]
            return f"# Investigation: {query}\n\n" + "\n\n".join(parts)

        # Build the findings text
        findings_text = "\n\n".join(
            f"### {f.dimension.label}\n{f.content}" for f in findings
        )

        user_prompt = _INVESTIGATE_SYNTHESIS_USER.format(
            query=query,
            findings_text=findings_text,
        )

        try:
            report = await self._bodega.chat_simple(
                user_prompt,
                system=_INVESTIGATE_SYNTHESIS_SYSTEM,
                tier=ModelTier.REASON,
                max_tokens=3000,
                temperature=0.1,
            )
            # Strip <think> blocks
            import re
            report = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()
            return report
        except Exception as exc:
            logger.warning("investigate_synthesis_reason_failed", error=str(exc))
            # Fallback: try MID tier (lightweight loaded model)
            try:
                import re
                report = await self._bodega.chat_simple(
                    user_prompt,
                    system=_INVESTIGATE_SYNTHESIS_SYSTEM,
                    tier=ModelTier.MID,
                    max_tokens=3000,
                    temperature=0.1,
                )
                return re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()
            except Exception as exc2:
                logger.warning("investigate_synthesis_mid_failed", error=str(exc2))
                # Last resort: concatenate findings
                parts = [f"## {f.dimension.label}\n\n{f.content}" for f in findings]
                return f"# Investigation: {query}\n\n" + "\n\n".join(parts)
