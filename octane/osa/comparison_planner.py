"""ComparisonPlanner — decompose a comparison query into items × dimensions.

The ComparisonPlanner is the brain of `octane compare`.  Given a query
like "NVDA vs AMD", it identifies:

    Items:       ["NVDA", "AMD"]
    Dimensions:  ["Valuation", "Revenue Growth", "AI Market Share",
                  "Technical Roadmap", "Analyst Sentiment"]

This produces a research matrix: each item is investigated on each dimension
in parallel.  2 items × 5 dimensions = 10 parallel research tasks.

Design:
    - Uses REASON tier for maximum planning quality.
    - Handles N-way comparisons (not just 2): "Python vs Go vs Rust"
    - Falls back gracefully when Bodega unavailable.
    - Auto-detects comparison syntax: "X vs Y", "X versus Y", "compare X and Y",
      "X or Y", etc.
    - If query doesn't mention specific items, extracts them from context.

Output schema:
    {
        "items": [
            {"id": "nvda", "label": "NVDA", "canonical_query": "NVIDIA stock"}
        ],
        "dimensions": [
            {
                "id": "valuation",
                "label": "Valuation",
                "query_template": "{item} valuation P/E ratio",
                "priority": 1
            }
        ]
    }
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="osa.comparison_planner")

# Sentinel: distinguish "use default BodegaRouter" from "explicitly no LLM"
_UNSET = object()

_MAX_ITEMS = 6          # More than 6 items = unwieldy comparison
_MAX_DIMENSIONS = 7     # More than 7 dimensions = diminishing returns
_MIN_ITEMS = 2
_MIN_DIMENSIONS = 2

# ── System Prompt ─────────────────────────────────────────────────────────────

_COMPARISON_PLANNER_SYSTEM = """\
You are a research strategist for structured comparisons. Given a comparison \
query, identify the items being compared and the dimensions to compare them on.

Output ONLY valid JSON in this exact format (no markdown, no commentary):
{
  "items": [
    {
      "id": "short_slug",
      "label": "Human Readable Label",
      "canonical_query": "best search query to find information about this item"
    }
  ],
  "dimensions": [
    {
      "id": "dimension_slug",
      "label": "Dimension Label",
      "query_template": "{item} dimension query",
      "priority": 1
    }
  ]
}

Rules:
- 2 to 6 items. If query mentions more than 6, keep the most relevant 6.
- 3 to 7 comparison dimensions. Cover: quantitative metrics, qualitative \
  factors, recent news/sentiment, and future outlook.
- "query_template" uses {item} as a placeholder — it will be substituted \
  with each item's label when searching.
- "id" uses underscores, lowercase. "priority" is 1 (most important) to 7.
- Dimensions should be PARALLEL — each must make sense for every item.
- If comparing financial assets, include: price/valuation, growth, \
  analyst sentiment, competitive moat.
- If comparing technologies, include: capabilities, adoption, ecosystem, \
  performance benchmarks.
"""

# ── Patterns for detecting comparison queries ─────────────────────────────────

_VS_PATTERNS = [
    r"\b(?P<a>.+?)\s+vs\.?\s+(?P<b>.+)",
    r"\b(?P<a>.+?)\s+versus\s+(?P<b>.+)",
    r"\bcompare\s+(?P<a>.+?)\s+(?:and|with|to)\s+(?P<b>.+)",
    r"\b(?P<a>.+?)\s+or\s+(?P<b>.+?)\b(?:\?|$)",
]

_COMPILED_VS = [re.compile(p, re.IGNORECASE) for p in _VS_PATTERNS]


def extract_items_from_query(query: str) -> list[str] | None:
    """Try to extract comparison items from a query string.

    Handles: "X vs Y", "X versus Y", "compare X and Y", "X or Y".
    Returns None if no comparison pattern detected.
    """
    for pattern in _COMPILED_VS:
        m = pattern.search(query.strip())
        if m:
            a = m.group("a").strip().strip("'\"")
            b = m.group("b").strip().strip("'\"")
            if a and b:
                # Handle multi-item "A vs B vs C"
                items = [a]
                # Check if b itself is a "vs" chain
                parts = re.split(r"\s+vs\.?\s+|\s+versus\s+", b, flags=re.IGNORECASE)
                items.extend(p.strip() for p in parts if p.strip())
                return items[:_MAX_ITEMS]
    return []


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class ComparisonItem:
    """One item in the comparison.

    Attributes:
        id:              Short slug (e.g. "nvda").
        label:           Human label (e.g. "NVDA").
        canonical_query: Best search query for this item.
    """

    id: str
    label: str
    canonical_query: str = ""

    def query_for_dimension(self, dimension: "ComparisonDimension") -> str:
        """Build the search query for this item on a specific dimension."""
        template = dimension.query_template or f"{{item}} {dimension.label}"
        return template.replace("{item}", self.label)


@dataclass
class ComparisonDimension:
    """One dimension of comparison.

    Attributes:
        id:             Short slug (e.g. "valuation").
        label:          Human label (e.g. "Valuation").
        query_template: Template with {item} placeholder.
        priority:       1=most important, 7=least.
    """

    id: str
    label: str
    query_template: str = ""
    priority: int = 1


@dataclass
class ComparisonPlan:
    """The full comparison plan: items × dimensions.

    Attributes:
        query:      The original user query.
        items:      What's being compared.
        dimensions: Along which axes.
        from_llm:   True if LLM-generated, False if fallback.
    """

    query: str
    items: list[ComparisonItem] = field(default_factory=list)
    dimensions: list[ComparisonDimension] = field(default_factory=list)
    from_llm: bool = True

    @property
    def sorted_dimensions(self) -> list[ComparisonDimension]:
        return sorted(self.dimensions, key=lambda d: d.priority)

    @property
    def task_matrix(self) -> list[tuple[ComparisonItem, ComparisonDimension]]:
        """Return all (item, dimension) pairs for parallel research."""
        return [
            (item, dim)
            for dim in self.sorted_dimensions
            for item in self.items
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "from_llm": self.from_llm,
            "items": [
                {"id": i.id, "label": i.label, "canonical_query": i.canonical_query}
                for i in self.items
            ],
            "dimensions": [
                {
                    "id": d.id,
                    "label": d.label,
                    "query_template": d.query_template,
                    "priority": d.priority,
                }
                for d in self.sorted_dimensions
            ],
            "n_tasks": len(self.task_matrix),
        }


# ── ComparisonPlanner ─────────────────────────────────────────────────────────


class ComparisonPlanner:
    """Identifies items and comparison dimensions from a query.

    Uses REASON tier for maximum quality.  Falls back to keyword-derived
    dimensions when Bodega is unavailable.

    Args:
        bodega: BodegaRouter instance or None to disable LLM.
    """

    def __init__(self, bodega=_UNSET) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega

    async def plan(self, query: str) -> ComparisonPlan:
        """Decompose a comparison query into a ComparisonPlan.

        Args:
            query: The comparison query (e.g. "NVDA vs AMD").

        Returns:
            ComparisonPlan with items and dimensions.
        """
        if not self._bodega:
            logger.debug("comparison_planner_fallback_no_bodega", query=query[:60])
            return self._keyword_fallback(query)

        try:
            _t0 = time.monotonic()
            print("[octane] Planning: sending to bodega-raptor-8b (MID tier)...", file=sys.stderr, flush=True)
            raw = await asyncio.wait_for(
                self._bodega.chat_simple(
                    f"Comparison query: {query}",
                    system=_COMPARISON_PLANNER_SYSTEM,
                    tier=ModelTier.MID,
                    max_tokens=800,
                    temperature=0.0,
                ),
                timeout=45.0,
            )
            print(f"[octane] Planning done in {time.monotonic() - _t0:.1f}s", file=sys.stderr, flush=True)
            return self._parse_response(query, raw)

        except asyncio.TimeoutError:
            print("[octane] Planning timed out after 45s — using keyword fallback", file=sys.stderr, flush=True)
            logger.warning("comparison_planner_timeout", query=query[:60])
            return self._keyword_fallback(query)
        except Exception as exc:
            print(f"[octane] Planning failed: {exc}", file=sys.stderr, flush=True)
            logger.warning("comparison_planner_llm_failed", error=str(exc))
            return self._keyword_fallback(query)

    def _parse_response(self, query: str, raw: str) -> ComparisonPlan:
        """Parse LLM JSON response into a ComparisonPlan."""
        # Strip <think> blocks and markdown fences
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            return self._keyword_fallback(query)

        try:
            data = json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            return self._keyword_fallback(query)

        raw_items = data.get("items", [])
        raw_dims = data.get("dimensions", [])

        if len(raw_items) < _MIN_ITEMS or len(raw_dims) < _MIN_DIMENSIONS:
            return self._keyword_fallback(query)

        items: list[ComparisonItem] = []
        for item in raw_items[:_MAX_ITEMS]:
            if not isinstance(item, dict):
                continue
            items.append(ComparisonItem(
                id=str(item.get("id", "item")).strip(),
                label=str(item.get("label", "")).strip(),
                canonical_query=str(item.get("canonical_query", "")).strip(),
            ))

        dimensions: list[ComparisonDimension] = []
        for i, dim in enumerate(raw_dims[:_MAX_DIMENSIONS]):
            if not isinstance(dim, dict):
                continue
            dimensions.append(ComparisonDimension(
                id=str(dim.get("id", f"dim_{i}")).strip(),
                label=str(dim.get("label", "")).strip(),
                query_template=str(dim.get("query_template", "")).strip(),
                priority=int(dim.get("priority", i + 1)),
            ))

        if len(items) < _MIN_ITEMS or len(dimensions) < _MIN_DIMENSIONS:
            return self._keyword_fallback(query)

        logger.info(
            "comparison_plan_built",
            query=query[:60],
            n_items=len(items),
            n_dimensions=len(dimensions),
            from_llm=True,
        )
        return ComparisonPlan(
            query=query,
            items=items,
            dimensions=dimensions,
            from_llm=True,
        )

    def _keyword_fallback(self, query: str) -> ComparisonPlan:
        """Produce a comparison plan from keywords without LLM."""
        # Try to extract items from the query syntax
        extracted = extract_items_from_query(query)

        if extracted and len(extracted) >= _MIN_ITEMS:
            items = [
                ComparisonItem(
                    id=label.lower().replace(" ", "_").replace("/", "_"),
                    label=label,
                    canonical_query=f"{label} overview analysis",
                )
                for label in extracted
            ]
        else:
            # Generic fallback: treat the whole query as item A vs "alternatives"
            items = [
                ComparisonItem(id="option_a", label="Option A", canonical_query=query),
                ComparisonItem(id="option_b", label="Option B", canonical_query=f"alternatives to {query}"),
            ]

        query_lower = query.lower()
        is_finance = any(
            kw in query_lower
            for kw in ["stock", "etf", "nvda", "aapl", "msft", "fund", "invest",
                       "price", "valuation", "market cap"]
        )

        if is_finance:
            dimensions = [
                ComparisonDimension("valuation", "Valuation Metrics", "{item} valuation P/E ratio", 1),
                ComparisonDimension("growth", "Revenue & Earnings Growth", "{item} revenue growth earnings", 2),
                ComparisonDimension("analyst", "Analyst Sentiment", "{item} analyst rating price target", 3),
                ComparisonDimension("momentum", "Technical Momentum", "{item} stock price momentum", 4),
                ComparisonDimension("news", "Recent News", "{item} latest news developments", 5),
            ]
        else:
            dimensions = [
                ComparisonDimension("overview", "Overview", "{item} overview features", 1),
                ComparisonDimension("strengths", "Strengths", "{item} strengths advantages", 2),
                ComparisonDimension("weaknesses", "Weaknesses", "{item} weaknesses limitations", 3),
                ComparisonDimension("use_cases", "Use Cases", "{item} best use cases when to use", 4),
                ComparisonDimension("community", "Community & Ecosystem", "{item} community ecosystem support", 5),
            ]

        logger.info(
            "comparison_plan_keyword_fallback",
            query=query[:60],
            n_items=len(items),
            n_dimensions=len(dimensions),
        )
        return ComparisonPlan(
            query=query,
            items=items,
            dimensions=dimensions,
            from_llm=False,
        )
