"""DimensionPlanner — decompose a query into independent research dimensions.

The DimensionPlanner is the brain of `octane investigate`.  Given a query
like "Is NVDA overvalued at current levels?", it identifies 4-8 independent
research dimensions, each of which can be investigated in parallel:

    1. Current valuation metrics (P/E, P/S, EV/EBITDA vs sector)
    2. Revenue and earnings growth trajectory
    3. Analyst price targets and consensus
    4. Technical price action and momentum
    5. Competitive positioning (AMD, Intel, QCOM)
    6. Market sentiment and insider activity
    7. Macro risk factors (rates, export restrictions, AI cycle)

Each dimension is a self-contained research angle.  The InvestigateOrchestrator
runs them in parallel via asyncio.gather(), then cross-references the findings.

Design:
    - Uses REASON tier (8B model) — this is the hardest cognitive task in S25.
    - Produces deterministic structured JSON (temperature=0).
    - Falls back gracefully: if Bodega unavailable, returns keyword-derived
      dimensions so `octane investigate` still works offline.
    - Max dimensions: 8 (beyond 8, parallel overhead exceeds value).
    - Min dimensions: 2 (single-dimension = just use `octane ask`).

Output schema:
    {
        "query": str,
        "dimensions": [
            {
                "id": str,             # Short slug: "valuation_metrics"
                "label": str,          # Human label: "Valuation Metrics"
                "queries": list[str],  # 1-3 search queries for this dimension
                "priority": int,       # 1=highest, 8=lowest (for display ordering)
                "rationale": str       # Why this dimension matters for the query
            }
        ]
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="osa.dimension_planner")

# Sentinel: distinguish "use default BodegaRouter" from "explicitly no LLM"
_UNSET = object()

_MAX_DIMENSIONS = 8
_MIN_DIMENSIONS = 2

# ── System Prompt ─────────────────────────────────────────────────────────────

_DIMENSION_PLANNER_SYSTEM = """\
You are a research strategist. Given a query, identify the independent research \
dimensions needed to fully investigate it. Each dimension is a separate angle \
that can be researched in parallel.

Output ONLY valid JSON in this exact format (no markdown, no commentary):
{
  "dimensions": [
    {
      "id": "short_slug",
      "label": "Human Readable Label",
      "queries": ["search query 1", "search query 2"],
      "priority": 1,
      "rationale": "Why this dimension matters"
    }
  ]
}

Rules:
- Between 3 and 8 dimensions. No more, no less.
- Each dimension must be INDEPENDENT — researchable without the others.
- "queries" should be 1-3 specific search queries optimized for web search.
- "priority" is 1 (most important) to 8 (least). Assign unique values.
- "id" uses underscores, lowercase, 2-4 words max.
- Dimensions should cover breadth: quantitative + qualitative + contextual.
- If the query is about a specific asset, always include: current data, \
  recent news, and a comparative/competitive dimension.
"""


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class ResearchDimension:
    """A single independent research dimension.

    Attributes:
        id:        Short slug identifier (e.g. "valuation_metrics").
        label:     Human-readable label (e.g. "Valuation Metrics").
        queries:   1-3 specific search queries for this dimension.
        priority:  Importance ranking (1=highest, lower = research first).
        rationale: Why this dimension matters for the original query.
    """

    id: str
    label: str
    queries: list[str]
    priority: int = 1
    rationale: str = ""

    def primary_query(self) -> str:
        """Return the highest-priority search query for this dimension."""
        return self.queries[0] if self.queries else self.label


@dataclass
class DimensionPlan:
    """The full investigation plan for a query.

    Attributes:
        query:       The original user query.
        dimensions:  List of independent research dimensions, sorted by priority.
        from_llm:    True if produced by LLM, False if keyword fallback.
    """

    query: str
    dimensions: list[ResearchDimension] = field(default_factory=list)
    from_llm: bool = True

    @property
    def sorted_dimensions(self) -> list[ResearchDimension]:
        """Return dimensions sorted by priority (1 first)."""
        return sorted(self.dimensions, key=lambda d: d.priority)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "from_llm": self.from_llm,
            "dimensions": [
                {
                    "id": d.id,
                    "label": d.label,
                    "queries": d.queries,
                    "priority": d.priority,
                    "rationale": d.rationale,
                }
                for d in self.sorted_dimensions
            ],
        }


# ── DimensionPlanner ──────────────────────────────────────────────────────────


class DimensionPlanner:
    """Decomposes a query into independent research dimensions.

    Uses REASON tier for maximum planning quality.  Falls back to
    keyword-derived heuristics when Bodega is unavailable.

    Args:
        bodega: BodegaRouter instance.  Pass None to disable LLM (keyword
                fallback only).  Defaults to auto-configured BodegaRouter.
        max_dimensions: Maximum dimensions to generate (default 8).
    """

    def __init__(
        self,
        bodega=_UNSET,
        max_dimensions: int = _MAX_DIMENSIONS,
    ) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega
        self.max_dimensions = max_dimensions

    async def plan(
        self,
        query: str,
        max_dimensions: int | None = None,
    ) -> DimensionPlan:
        """Decompose a query into research dimensions.

        Args:
            query:          The investigation query.
            max_dimensions: Override instance default.

        Returns:
            DimensionPlan with sorted dimensions.
        """
        limit = max_dimensions or self.max_dimensions

        if not self._bodega:
            logger.debug("dimension_planner_fallback_no_bodega", query=query[:60])
            return self._keyword_fallback(query, limit)

        try:
            raw = await self._bodega.chat_simple(
                f"Query to investigate: {query}",
                system=_DIMENSION_PLANNER_SYSTEM,
                tier=ModelTier.MID,
                max_tokens=800,
                temperature=0.0,
            )
            return self._parse_response(query, raw, limit)

        except Exception as exc:
            logger.warning("dimension_planner_llm_failed", error=str(exc))
            return self._keyword_fallback(query, limit)

    def _parse_response(
        self,
        query: str,
        raw: str,
        limit: int,
    ) -> DimensionPlan:
        """Parse LLM JSON response into a DimensionPlan.

        Strips markdown fences and <think> blocks before parsing.
        Falls back to keyword heuristics on any parse error.
        """
        # Strip <think>...</think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        # Find the JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("dimension_planner_no_json", raw=raw[:200])
            return self._keyword_fallback(query, limit)

        try:
            data = json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("dimension_planner_json_error", error=str(exc), raw=raw[:200])
            return self._keyword_fallback(query, limit)

        raw_dims = data.get("dimensions", [])
        if not isinstance(raw_dims, list) or len(raw_dims) < _MIN_DIMENSIONS:
            logger.warning(
                "dimension_planner_insufficient_dims",
                count=len(raw_dims),
                min=_MIN_DIMENSIONS,
            )
            return self._keyword_fallback(query, limit)

        dimensions: list[ResearchDimension] = []
        for i, dim in enumerate(raw_dims[:limit]):
            if not isinstance(dim, dict):
                continue
            dim_id = str(dim.get("id", f"dimension_{i + 1}")).strip()
            label = str(dim.get("label", dim_id.replace("_", " ").title())).strip()
            queries = dim.get("queries", [label])
            if not isinstance(queries, list):
                queries = [str(queries)]
            # Ensure queries are strings and non-empty
            queries = [str(q).strip() for q in queries if str(q).strip()][:3]
            if not queries:
                queries = [label]
            priority = int(dim.get("priority", i + 1))
            rationale = str(dim.get("rationale", "")).strip()
            dimensions.append(
                ResearchDimension(
                    id=dim_id,
                    label=label,
                    queries=queries,
                    priority=priority,
                    rationale=rationale,
                )
            )

        if len(dimensions) < _MIN_DIMENSIONS:
            return self._keyword_fallback(query, limit)

        logger.info(
            "dimension_plan_built",
            query=query[:60],
            n_dimensions=len(dimensions),
            from_llm=True,
        )
        return DimensionPlan(query=query, dimensions=dimensions, from_llm=True)

    def _keyword_fallback(self, query: str, limit: int) -> DimensionPlan:
        """Produce a sensible set of research dimensions from keywords.

        This runs without any LLM call.  It provides enough structure
        for `octane investigate` to be useful even when Bodega is down.
        """
        query_lower = query.lower()

        # Detect domain from keywords
        is_finance = any(
            kw in query_lower
            for kw in ["stock", "price", "valuation", "invest", "market", "earnings",
                       "revenue", "p/e", "pe ratio", "overvalued", "undervalued",
                       "portfolio", "shares", "ticker", "etf"]
        )
        is_company = any(
            kw in query_lower
            for kw in ["company", "corp", "inc", "ltd", "apple", "google", "nvidia",
                       "nvda", "aapl", "goog", "msft", "microsoft", "amazon", "amzn",
                       "tesla", "tsla", "meta"]
        )
        is_tech = any(
            kw in query_lower
            for kw in ["ai", "machine learning", "software", "tech", "model", "llm",
                       "gpu", "chip", "semiconductor", "cloud"]
        )

        if is_finance and is_company:
            dims = [
                ResearchDimension(
                    id="current_data",
                    label="Current Market Data",
                    queries=[f"{query} current price data", f"{query} market cap"],
                    priority=1,
                    rationale="Latest quantitative data for the subject",
                ),
                ResearchDimension(
                    id="recent_news",
                    label="Recent News & Developments",
                    queries=[f"{query} latest news", f"{query} recent developments"],
                    priority=2,
                    rationale="Recent events affecting the subject",
                ),
                ResearchDimension(
                    id="analyst_consensus",
                    label="Analyst Views",
                    queries=[f"{query} analyst rating", f"{query} price target"],
                    priority=3,
                    rationale="Expert consensus and price targets",
                ),
                ResearchDimension(
                    id="competitive_landscape",
                    label="Competitive Landscape",
                    queries=[f"{query} competitors comparison", f"{query} vs peers"],
                    priority=4,
                    rationale="How the subject compares to alternatives",
                ),
                ResearchDimension(
                    id="risk_factors",
                    label="Risk Factors",
                    queries=[f"{query} risks", f"{query} headwinds threats"],
                    priority=5,
                    rationale="Key risks and downside scenarios",
                ),
            ]
        elif is_tech:
            dims = [
                ResearchDimension(
                    id="current_state",
                    label="Current State of Technology",
                    queries=[f"{query} current state", f"{query} latest developments"],
                    priority=1,
                    rationale="Current capabilities and state of the art",
                ),
                ResearchDimension(
                    id="key_players",
                    label="Key Players",
                    queries=[f"{query} leading companies", f"{query} top solutions"],
                    priority=2,
                    rationale="Who is leading in this space",
                ),
                ResearchDimension(
                    id="use_cases",
                    label="Use Cases & Applications",
                    queries=[f"{query} use cases", f"{query} applications examples"],
                    priority=3,
                    rationale="Practical applications and real-world use",
                ),
                ResearchDimension(
                    id="future_outlook",
                    label="Future Outlook",
                    queries=[f"{query} future predictions", f"{query} roadmap 2026"],
                    priority=4,
                    rationale="Where this is heading",
                ),
            ]
        else:
            # Generic multi-angle decomposition
            dims = [
                ResearchDimension(
                    id="overview",
                    label="Overview & Context",
                    queries=[query, f"{query} overview"],
                    priority=1,
                    rationale="Foundational understanding",
                ),
                ResearchDimension(
                    id="current_developments",
                    label="Current Developments",
                    queries=[f"{query} latest news", f"{query} 2026"],
                    priority=2,
                    rationale="Recent news and developments",
                ),
                ResearchDimension(
                    id="analysis",
                    label="Analysis & Perspectives",
                    queries=[f"{query} analysis", f"{query} expert opinion"],
                    priority=3,
                    rationale="Analytical perspectives",
                ),
                ResearchDimension(
                    id="implications",
                    label="Implications & Impact",
                    queries=[f"{query} impact", f"{query} implications"],
                    priority=4,
                    rationale="What this means and why it matters",
                ),
            ]

        dims = dims[:limit]
        logger.info(
            "dimension_plan_keyword_fallback",
            query=query[:60],
            n_dimensions=len(dims),
        )
        return DimensionPlan(query=query, dimensions=dims, from_llm=False)
