"""Query Strategist — generates search variations from a user query.

Session 10: LLM-powered multi-variation strategy generation.
For each query, produces 2-3 search variations targeting the most
relevant Bodega Intelligence API (search, news, finance).
Falls back to single-variation passthrough if LLM is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog

from octane.tools.topology import ModelTier
from octane.utils.clock import month_year, today_str

logger = structlog.get_logger().bind(component="web.query_strategist")

# Sentinel: distinguish "use default BodegaRouter" from "explicitly no LLM"
_UNSET = object()

_STRATEGIST_SYSTEM_BASE = """\
You are a search query optimizer. Given a user query, return a JSON array of \
2-3 search variations that maximise coverage of the topic.

Each variation must have:
  - "query": the search string (concise, specific)
  - "api": one of "search", "news", "finance"
  - "rationale": one short phrase explaining why

Rules:
- Use "finance" only for stock prices, earnings, market data.
- Use "news" for recent events, announcements, headlines.
- Use "search" for factual/technical/general queries.
- Variations should be meaningfully different, not just synonyms.
- For time-sensitive queries (prices, news, current events), include the \
  current month and year in at least one search variation to ensure fresh results.
- Return ONLY valid JSON. No prose, no markdown fences."""

_FALLBACK_APIS = {
    "stock", "price", "ticker", "earnings", "market", "share",
    "ipo", "dividend", "valuation", "revenue", "capex",
}
_NEWS_APIS = {
    "news", "latest", "recent", "today", "headline", "announced",
    "breaking", "update", "this week", "yesterday",
}


class QueryStrategist:
    """Generates 1-3 targeted search strategies for a query.

    With LLM: returns 2-3 semantically varied queries across appropriate APIs.
    Without LLM: returns a single strategy using keyword-based API selection.
    """

    def __init__(self, bodega=_UNSET) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega

    async def strategize(
        self, query: str, context: dict | None = None
    ) -> list[dict[str, str]]:
        """Generate search strategies for the given query.

        Returns:
            List of dicts with keys: query, api, rationale
        """
        if self._bodega is not None:
            try:
                return await self._llm_strategize(query, context)
            except Exception as exc:
                logger.warning("llm_strategize_failed", error=str(exc), fallback="keyword")

        # Fallback: single keyword-based strategy
        return [{"query": query, "api": self._keyword_api(query), "rationale": "keyword fallback"}]

    async def _llm_strategize(
        self, query: str, context: dict | None = None
    ) -> list[dict[str, str]]:
        """Use LLM to generate multi-variation strategies."""
        context_hint = ""
        if context:
            sub_agent = context.get("sub_agent", "")
            if sub_agent in ("finance", "web_finance"):
                context_hint = " [Context: this is a finance/market data query]"
            elif sub_agent in ("news", "web_news"):
                context_hint = " [Context: this is a news query]"

        prompt = f'Today is {today_str()} ({month_year()}).\nQuery: "{query}"{context_hint}\n\nGenerate 2-3 search variations.'

        raw = await asyncio.wait_for(
            self._bodega.chat_simple(
                prompt=prompt,
                system=_STRATEGIST_SYSTEM_BASE,
                tier=ModelTier.FAST,
                temperature=0.3,
                max_tokens=2048,  # reasoning models burn 300-1000 tokens on <think>; need headroom
            ),
            timeout=45.0,  # 90M reasoning model can take 20-30s on first inference
        )

        # Extract content after </think> block — preserve reasoning as debug log
        if "</think>" in raw:
            think_part, _, clean = raw.partition("</think>")
            logger.debug("model_reasoning_trace", trace=think_part.replace("<think>", "").strip()[:300])
        else:
            clean = raw
        # Extract JSON array — handle models that wrap in markdown fences
        json_match = re.search(r"\[.*\]", clean, flags=re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON array in LLM response: {clean[:100]}")

        strategies = json.loads(json_match.group(0))

        # Validate and normalise
        valid: list[dict[str, str]] = []
        valid_apis = {"search", "news", "finance"}
        for s in strategies:
            if isinstance(s, dict) and "query" in s and "api" in s:
                api = s["api"] if s["api"] in valid_apis else "search"
                valid.append({
                    "query": str(s["query"])[:200],
                    "api": api,
                    "rationale": str(s.get("rationale", ""))[:100],
                })
        if not valid:
            raise ValueError("LLM returned no valid strategies")

        logger.debug("strategies_generated", count=len(valid), query=query[:60])
        return valid[:3]  # cap at 3

    def _keyword_api(self, query: str) -> str:
        """Choose the best API based on keywords in the query."""
        q = query.lower()
        if any(w in q for w in _FALLBACK_APIS):
            return "finance"
        if any(w in q for w in _NEWS_APIS):
            return "news"
        return "search"
