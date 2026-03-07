"""CompareOrchestrator — the engine behind `octane compare`.

Executes the full comparison pipeline:

    Query → ComparisonPlanner (REASON)
          → Parallel research: each item × each dimension
          → Cross-reference: Code Agent builds comparison matrix
          → Tradeoff synthesis: REASON tier structured side-by-side report
          → Auto-store findings

The comparison produces a structured report with:
    - Per-dimension comparison rows (what each item scores on that dimension)
    - A pros/cons summary per item
    - Overall tradeoff analysis and recommendation context

Design:
    - Matrix parallelism: items × dimensions all fire simultaneously.
    - Bounded concurrency: semaphore(4) prevents Bodega overload.
    - Graceful degradation: missing cells are marked "No data" in the matrix.
    - Structured output: report is deterministic table + narrative.

Progress events streamed:
    {"type": "plan",       "data": ComparisonPlan.to_dict()}
    {"type": "progress",   "data": {"item": label, "dimension": label, "status": "researching"}}
    {"type": "cell",       "data": {"item": label, "dimension": label, "content": str}}
    {"type": "matrix",     "data": {"cells": {...}, "n_cells": int}}
    {"type": "synthesis",  "data": {"report": str, "word_count": int}}
    {"type": "done",       "data": {"n_cells": int, "n_successful": int, "total_ms": float}}
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from octane.osa.comparison_planner import (
    ComparisonItem,
    ComparisonDimension,
    ComparisonPlan,
    ComparisonPlanner,
)
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="osa.compare")

_UNSET = object()
_MAX_PARALLEL_CELLS = 4

# ── Synthesis Prompt ──────────────────────────────────────────────────────────

_COMPARE_SYNTHESIS_SYSTEM = """\
You are the Octane Comparator. You have just completed multi-dimensional \
research comparing multiple items. Synthesize into a structured comparison report.

Output format:
- Start with a one-sentence overview of what's being compared
- ## Dimension-by-Dimension Comparison (one brief paragraph per dimension)
- ## Summary Table (markdown table: rows=dimensions, columns=items)
- ## Verdict (1-2 paragraphs: who wins on what, when to choose each)
- Be direct and specific. Ground every claim in the research provided.
- Do not mention "agents", "tools", or internal system names.
- Target 600-1000 words.
"""

_COMPARE_SYNTHESIS_USER = """\
Comparison query: {query}

Items being compared: {items}

Research matrix (item × dimension):
{matrix_text}

Synthesize into a structured comparison report.
"""


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class ComparisonCell:
    """Research result for one (item, dimension) pair.

    Attributes:
        item:       The item being researched.
        dimension:  The dimension of comparison.
        content:    Research text.
        latency_ms: Time to research.
        error:      Non-empty if failed.
    """

    item: ComparisonItem
    dimension: ComparisonDimension
    content: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def cell_key(self) -> str:
        return f"{self.item.id}::{self.dimension.id}"

    @property
    def success(self) -> bool:
        return bool(self.content) and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item.id,
            "item_label": self.item.label,
            "dimension_id": self.dimension.id,
            "dimension_label": self.dimension.label,
            "content": self.content,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
        }


@dataclass
class ComparisonResult:
    """Full output of a comparison.

    Attributes:
        query:     Original user query.
        plan:      ComparisonPlan.
        cells:     All (item, dimension) research cells.
        report:    Synthesized comparison report.
        word_count: Report word count.
        total_ms:  End-to-end latency.
    """

    query: str
    plan: ComparisonPlan
    cells: list[ComparisonCell] = field(default_factory=list)
    report: str = ""
    word_count: int = 0
    total_ms: float = 0.0

    @property
    def successful_cells(self) -> list[ComparisonCell]:
        return [c for c in self.cells if c.success]

    def cell_for(self, item_id: str, dimension_id: str) -> ComparisonCell | None:
        for cell in self.cells:
            if cell.item.id == item_id and cell.dimension.id == dimension_id:
                return cell
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.to_dict(),
            "n_cells": len(self.cells),
            "n_successful": len(self.successful_cells),
            "report": self.report,
            "word_count": self.word_count,
            "total_ms": round(self.total_ms, 1),
        }


# ── CompareOrchestrator ───────────────────────────────────────────────────────


class CompareOrchestrator:
    """Orchestrates the full comparison pipeline.

    Args:
        bodega:             BodegaRouter for LLM calls.
        web_agent:          WebAgent for per-cell web research.
        memory_agent:       MemoryAgent for prior-context recall.
        comparison_planner: ComparisonPlanner instance.
    """

    def __init__(
        self,
        bodega=_UNSET,
        web_agent=None,
        memory_agent=None,
        comparison_planner: ComparisonPlanner | None = None,
    ) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega
        self._web_agent = web_agent
        self._memory_agent = memory_agent
        self._planner = comparison_planner or ComparisonPlanner(bodega=bodega)

    async def run_stream(
        self,
        query: str,
        session_id: str = "cli",
    ) -> AsyncIterator[dict[str, Any]]:
        """Run comparison and yield progress events.

        Yields: plan → cell (per item×dim) → matrix → synthesis → done
        """
        t0 = time.monotonic()

        # ── Phase 1: Plan ──────────────────────────────────────────────────
        plan = await self._planner.plan(query)
        yield {"type": "plan", "data": plan.to_dict()}

        # ── Phase 2: Parallel matrix research ─────────────────────────────
        semaphore = asyncio.Semaphore(_MAX_PARALLEL_CELLS)
        cells: list[ComparisonCell] = []

        async def _research_cell(
            item: ComparisonItem,
            dimension: ComparisonDimension,
        ) -> ComparisonCell:
            async with semaphore:
                return await self._research_one_cell(query, item, dimension, session_id)

        tasks = [
            _research_cell(item, dim)
            for item, dim in plan.task_matrix
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (item, dim), result in zip(plan.task_matrix, results):
            if isinstance(result, Exception):
                cell = ComparisonCell(
                    item=item,
                    dimension=dim,
                    error=str(result),
                )
            else:
                cell = result
            cells.append(cell)
            yield {"type": "cell", "data": cell.to_dict()}

        # ── Phase 3: Matrix summary event ──────────────────────────────────
        matrix_summary = {
            f"{c.item.label}::{c.dimension.label}": c.content[:200]
            for c in cells if c.success
        }
        yield {
            "type": "matrix",
            "data": {
                "cells": matrix_summary,
                "n_cells": len(cells),
                "n_successful": sum(1 for c in cells if c.success),
            },
        }

        # ── Phase 4: Synthesis ─────────────────────────────────────────────
        successful_cells = [c for c in cells if c.success]
        if not successful_cells:
            report = f"Comparison could not retrieve data for: {query}"
        else:
            report = await self._synthesize(query, plan, successful_cells)

        word_count = len(report.split())
        total_ms = (time.monotonic() - t0) * 1000

        yield {
            "type": "synthesis",
            "data": {"report": report, "word_count": word_count},
        }
        yield {
            "type": "done",
            "data": {
                "n_cells": len(cells),
                "n_successful": len(successful_cells),
                "total_ms": round(total_ms, 1),
            },
        }

    async def run(
        self,
        query: str,
        session_id: str = "cli",
    ) -> ComparisonResult:
        """Run comparison and return a complete ComparisonResult."""
        t0 = time.monotonic()
        plan_obj: ComparisonPlan | None = None
        cells: list[ComparisonCell] = []
        report = ""
        word_count = 0

        async for event in self.run_stream(query, session_id=session_id):
            if event["type"] == "plan":
                d = event["data"]
                items = [
                    ComparisonItem(
                        id=i["id"], label=i["label"],
                        canonical_query=i.get("canonical_query", ""),
                    )
                    for i in d.get("items", [])
                ]
                dimensions = [
                    ComparisonDimension(
                        id=dim["id"], label=dim["label"],
                        query_template=dim.get("query_template", ""),
                        priority=dim.get("priority", 1),
                    )
                    for dim in d.get("dimensions", [])
                ]
                plan_obj = ComparisonPlan(
                    query=d["query"],
                    items=items,
                    dimensions=dimensions,
                    from_llm=d.get("from_llm", False),
                )
            elif event["type"] == "cell":
                d = event["data"]
                if plan_obj:
                    item = next((i for i in plan_obj.items if i.id == d["item_id"]), None)
                    dim = next((dim for dim in plan_obj.dimensions if dim.id == d["dimension_id"]), None)
                    if item and dim:
                        cells.append(ComparisonCell(
                            item=item,
                            dimension=dim,
                            content=d.get("content", ""),
                            latency_ms=d.get("latency_ms", 0.0),
                            error="" if d.get("success") else d.get("error", "unknown"),
                        ))
            elif event["type"] == "synthesis":
                report = event["data"].get("report", "")
                word_count = event["data"].get("word_count", 0)

        if plan_obj is None:
            plan_obj = ComparisonPlan(query=query)

        return ComparisonResult(
            query=query,
            plan=plan_obj,
            cells=cells,
            report=report,
            word_count=word_count,
            total_ms=(time.monotonic() - t0) * 1000,
        )

    async def _research_one_cell(
        self,
        original_query: str,
        item: ComparisonItem,
        dimension: ComparisonDimension,
        session_id: str,
    ) -> ComparisonCell:
        """Research a single (item, dimension) cell."""
        t0 = time.monotonic()

        # Build the search query for this cell
        search_query = item.query_for_dimension(dimension)
        content = ""

        if self._web_agent is not None:
            try:
                request = AgentRequest(
                    query=search_query,
                    source="compare",
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
                    "compare_cell_web_failed",
                    item=item.id,
                    dimension=dimension.id,
                    error=str(exc),
                )

        latency_ms = (time.monotonic() - t0) * 1000
        return ComparisonCell(
            item=item,
            dimension=dimension,
            content=content,
            latency_ms=latency_ms,
        )

    async def _synthesize(
        self,
        query: str,
        plan: ComparisonPlan,
        cells: list[ComparisonCell],
    ) -> str:
        """Synthesize comparison matrix into a structured report."""
        if not self._bodega:
            # Fallback: simple concatenation
            parts: list[str] = []
            for dim in plan.sorted_dimensions:
                parts.append(f"## {dim.label}")
                for item in plan.items:
                    cell = next(
                        (c for c in cells
                         if c.item.id == item.id and c.dimension.id == dim.id),
                        None,
                    )
                    content = cell.content[:300] if cell and cell.success else "No data"
                    parts.append(f"**{item.label}**: {content}")
                parts.append("")
            return "\n".join(parts)

        # Build matrix text
        matrix_parts: list[str] = []
        for dim in plan.sorted_dimensions:
            matrix_parts.append(f"\n### {dim.label}")
            for item in plan.items:
                cell = next(
                    (c for c in cells
                     if c.item.id == item.id and c.dimension.id == dim.id),
                    None,
                )
                content = cell.content[:400] if cell and cell.success else "No data available"
                matrix_parts.append(f"**{item.label}**: {content}")

        matrix_text = "\n".join(matrix_parts)
        items_str = ", ".join(i.label for i in plan.items)

        user_prompt = _COMPARE_SYNTHESIS_USER.format(
            query=query,
            items=items_str,
            matrix_text=matrix_text,
        )

        try:
            report = await self._bodega.chat_simple(
                user_prompt,
                system=_COMPARE_SYNTHESIS_SYSTEM,
                tier=ModelTier.REASON,
                max_tokens=2500,
                temperature=0.1,
            )
            import re
            report = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()
            return report
        except Exception as exc:
            logger.warning("compare_synthesis_reason_failed", error=str(exc))
            # Fallback: try MID tier (lightweight loaded model)
            try:
                import re
                report = await self._bodega.chat_simple(
                    user_prompt,
                    system=_COMPARE_SYNTHESIS_SYSTEM,
                    tier=ModelTier.MID,
                    max_tokens=2500,
                    temperature=0.1,
                )
                return re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()
            except Exception as exc2:
                logger.warning("compare_synthesis_mid_failed", error=str(exc2))
                return matrix_text
