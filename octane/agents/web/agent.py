"""Web Agent — fetches real data from Bodega Intelligence APIs.

Routes to the correct API based on the sub_agent hint in the task metadata:
    web_finance  → market_data() + optional timeseries()
    web_news     → news_search() → LLM synthesis
    web_search   → web_search() (Beru/Brave) → LLM synthesis

Session 10: QueryStrategist generates 2-3 search variations.
            Synthesizer uses LLM to turn raw results into structured intelligence.
"""

from __future__ import annotations

import re

import structlog

from octane.agents.base import BaseAgent
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.bodega_intel import BodegaIntelClient
from octane.agents.web.synthesizer import Synthesizer
from octane.agents.web.query_strategist import QueryStrategist

logger = structlog.get_logger().bind(component="web_agent")


class WebAgent(BaseAgent):
    """Web Agent — coordinates internet data retrieval via Bodega Intelligence."""

    name = "web"

    def __init__(self, synapse, intel: BodegaIntelClient | None = None, bodega=None) -> None:
        super().__init__(synapse)
        self._intel = intel or BodegaIntelClient()
        self._synthesizer = Synthesizer(bodega=bodega)
        self._strategist = QueryStrategist(bodega=bodega)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route to the correct Bodega Intelligence API based on sub_agent hint."""
        sub_agent = request.metadata.get("sub_agent", "search")
        query = request.query

        # Accept both short forms ("finance") and template-prefixed forms ("web_finance")
        if sub_agent in ("finance", "web_finance"):
            return await self._fetch_finance(query, request)
        elif sub_agent in ("news", "web_news"):
            return await self._fetch_news(query, request)
        else:
            # "search", "web_search", or any unknown → general Brave search
            return await self._fetch_search(query, request)

    # ── Finance ───────────────────────────────────────────────────────────

    async def _fetch_finance(self, query: str, request: AgentRequest) -> AgentResponse:
        """Fetch market data. Extracts ticker symbol from query if present."""
        ticker = self._extract_ticker(query)

        if ticker:
            raw = await self._intel.market_data(ticker)
            if "error" in raw:
                return AgentResponse(
                    agent=self.name, success=False,
                    error=f"Finance API error: {raw['error']}",
                    correlation_id=request.correlation_id,
                )
            summary = self._format_market_data(raw, ticker)
        else:
            # No ticker found — fall back to web search for financial query
            raw = await self._intel.web_search(query, count=3)
            if "error" in raw:
                return AgentResponse(
                    agent=self.name, success=False,
                    error=f"Search fallback error: {raw['error']}",
                    correlation_id=request.correlation_id,
                )
            summary = self._format_web_results(raw, query)

        return AgentResponse(
            agent=self.name, success=True,
            output=summary, data=raw,
            correlation_id=request.correlation_id,
        )

    def _extract_ticker(self, query: str) -> str | None:
        """Extract a ticker symbol from the query (e.g. NVDA, AAPL, TSLA)."""
        # Explicit all-caps ticker: NVDA, AAPL, TSLA, MSFT etc.
        match = re.search(r'\b([A-Z]{2,5})\b', query)
        if match:
            return match.group(1)
        # Known name→ticker map for common queries
        name_map = {
            "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
            "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
            "tesla": "TSLA", "meta": "META", "netflix": "NFLX",
            "amd": "AMD", "intel": "INTC", "qualcomm": "QCOM",
        }
        query_lower = query.lower()
        for name, ticker in name_map.items():
            if name in query_lower:
                return ticker
        return None

    def _format_market_data(self, raw: dict, ticker: str) -> str:
        """Turn market_data API response into a clean summary string."""
        md = raw.get("market_data", raw)
        price = md.get("price", "?")
        change = md.get("change", 0)
        change_pct = md.get("change_percent", 0)
        volume = md.get("volume", 0)
        market_cap = md.get("market_cap", 0)
        direction = "▲" if change >= 0 else "▼"
        cap_str = f"${market_cap / 1e12:.2f}T" if market_cap > 1e12 else f"${market_cap / 1e9:.1f}B"
        return (
            f"{ticker}: ${price:.2f} {direction}{abs(change_pct):.2f}% today | "
            f"Volume: {volume:,} | Market Cap: {cap_str}"
        )

    # ── News ──────────────────────────────────────────────────────────────

    async def _fetch_news(self, query: str, request: AgentRequest) -> AgentResponse:
        """Fetch news articles for the query, then synthesize with LLM."""
        # Use QueryStrategist to potentially refine the news query
        strategies = await self._strategist.strategize(
            query, context={"sub_agent": "news"}
        )
        # Use the first strategy's query for news lookup
        search_query = strategies[0].get("query", query)

        raw = await self._intel.news_search(search_query, period="3d", max_results=7)
        if "error" in raw:
            return AgentResponse(
                agent=self.name, success=False,
                error=f"News API error: {raw['error']}",
                correlation_id=request.correlation_id,
            )

        articles = raw.get("articles", [])
        summary = await self._synthesizer.synthesize_news(query, articles)

        return AgentResponse(
            agent=self.name, success=True,
            output=summary, data=raw,
            correlation_id=request.correlation_id,
        )

    # ── Web Search ────────────────────────────────────────────────────────

    async def _fetch_search(self, query: str, request: AgentRequest) -> AgentResponse:
        """General web search via Beru/Brave, with multi-variation strategy + LLM synthesis."""
        # Generate 2-3 search variations
        strategies = await self._strategist.strategize(query)

        # Execute the first strategy (primary) — future: fan-out + dedup across all
        primary = strategies[0]
        raw = await self._intel.web_search(primary["query"], count=6)
        if "error" in raw:
            return AgentResponse(
                agent=self.name, success=False,
                error=f"Search API error: {raw['error']}",
                correlation_id=request.correlation_id,
            )

        web = raw.get("web", {})
        results = web.get("results", [])
        if not results:
            results = raw.get("discussions", {}).get("results", [])

        summary = await self._synthesizer.synthesize_search(query, results)
        return AgentResponse(
            agent=self.name, success=True,
            output=summary, data=raw,
            correlation_id=request.correlation_id,
        )

