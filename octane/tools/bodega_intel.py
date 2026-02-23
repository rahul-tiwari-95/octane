"""Async client for Bodega Intelligence APIs (external data).

All services run on a single consolidated server at :44469.
Requires Bearer token authentication.

Route map (confirmed from live server):
    Web search:    GET /api/v1/beru/search/web?query=...&count=...
    News search:   GET /api/v1/news/api/v1/news/search?q=...&period=...&max_results=...
    Finance:       GET /api/v1/finance/api/v1/finance/market/{symbol}
    Timeseries:    GET /api/v1/finance/api/v1/finance/timeseries/{symbol}?period=...
    Headlines:     GET /api/v1/news/api/v1/news/headlines
    News topic:    GET /api/v1/news/api/v1/news/topics/{topic}

Used ONLY by the Web Agent's Fetcher sub-agent.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from octane.config import settings

logger = structlog.get_logger().bind(component="bodega_intel")


class BodegaIntelClient:
    """Async client for Bodega Intelligence consolidated server (:44469).

    Single httpx client, single base URL, Bearer auth on every request.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or settings.bodega_intel_url).rstrip("/")
        self.api_key = api_key or settings.bodega_intel_api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._auth_headers(),
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── Web Search (Beru) ──────────────────────────────────────────────────

    async def web_search(self, query: str, count: int = 5) -> dict[str, Any]:
        """General web search via Beru/Brave Search.

        Returns raw Brave search results with web, discussions, etc.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/web",
                params={"query": query, "count": count},
            )
            response.raise_for_status()
            result = response.json()
            logger.debug("web_search_ok", query=query, count=count)
            return result
        except httpx.HTTPError as e:
            logger.error("web_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    # ── News ──────────────────────────────────────────────────────────────

    async def news_search(
        self,
        query: str,
        period: str = "3d",
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Search news articles by keyword.

        period: '1d', '3d', '7d', '1m'
        """
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/news/api/v1/news/search",
                params={
                    "q": query,
                    "period": period,
                    "max_results": max_results,
                    "language": "en",
                    "country": "US",
                    "decode_urls": "true",
                },
            )
            response.raise_for_status()
            result = response.json()
            logger.debug("news_search_ok", query=query, count=result.get("count", 0))
            return result
        except httpx.HTTPError as e:
            logger.error("news_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def news_headlines(self) -> dict[str, Any]:
        """Get top breaking news headlines."""
        client = await self._get_client()
        try:
            response = await client.get("/api/v1/news/api/v1/news/headlines")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("news_headlines_failed", error=str(e))
            return {"error": str(e)}

    async def news_by_topic(self, topic: str, period: str = "1d") -> dict[str, Any]:
        """Get news by topic (TECHNOLOGY, BUSINESS, SCIENCE, etc.)."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/news/api/v1/news/topics/{topic}",
                params={"period": period},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("news_by_topic_failed", topic=topic, error=str(e))
            return {"error": str(e)}

    # ── Finance ───────────────────────────────────────────────────────────

    async def market_data(self, ticker: str) -> dict[str, Any]:
        """Get real-time market data for a ticker (price, change, volume, etc.)."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/finance/api/v1/finance/market/{ticker.upper()}"
            )
            response.raise_for_status()
            result = response.json()
            logger.debug("market_data_ok", ticker=ticker)
            return result
        except httpx.HTTPError as e:
            logger.error("market_data_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    async def timeseries(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> dict[str, Any]:
        """Get historical OHLCV time series for a ticker."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/finance/api/v1/finance/timeseries/{ticker.upper()}",
                params={"period": period, "interval": interval},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("timeseries_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    async def finance_complete(self, ticker: str) -> dict[str, Any]:
        """Get complete financial data: market + statements + timeseries."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/finance/api/v1/finance/complete/{ticker.upper()}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("finance_complete_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    # ── Health ────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Check consolidated server health."""
        client = await self._get_client()
        try:
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e)}
