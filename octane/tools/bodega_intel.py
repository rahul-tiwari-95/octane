"""Async client for Bodega Intelligence APIs (external data).

Search (:1111), Finance (:8030), News (:8032), Entertainment (:8031).
Used ONLY by the Web Agent's Fetcher sub-agent.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from octane.config import settings

logger = structlog.get_logger().bind(component="bodega_intel")


class BodegaIntelClient:
    """Async client for Bodega Intelligence external data APIs.

    Provides search, finance, news, and entertainment data.
    All methods return raw API responses as dicts.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._clients: dict[str, httpx.AsyncClient] = {}

    def _get_client(self, base_url: str) -> httpx.AsyncClient:
        if base_url not in self._clients or self._clients[base_url].is_closed:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                timeout=self.timeout,
            )
        return self._clients[base_url]

    async def close(self) -> None:
        """Close all HTTP clients."""
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()

    # ---- Search (Beru Search :1111) ----

    async def search(self, query: str) -> dict[str, Any]:
        """AI-powered search via Beru Search."""
        client = self._get_client(settings.bodega_search_url)
        try:
            response = await client.get(
                "/intelligence/search",
                params={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def image_search(self, query: str) -> dict[str, Any]:
        """Image search via Beru Search."""
        client = self._get_client(settings.bodega_search_url)
        try:
            response = await client.get(
                "/search/images",
                params={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("image_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    # ---- Finance (:8030) ----

    async def market_data(self, ticker: str) -> dict[str, Any]:
        """Get real-time market data for a ticker."""
        client = self._get_client(settings.bodega_finance_url)
        try:
            response = await client.get(f"/api/v1/finance/market/{ticker}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("market_data_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    async def timeseries(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> dict[str, Any]:
        """Get historical time series data for a ticker."""
        client = self._get_client(settings.bodega_finance_url)
        try:
            response = await client.get(
                f"/api/v1/finance/timeseries/{ticker}",
                params={"period": period, "interval": interval},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("timeseries_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    # ---- News (:8032) ----

    async def headlines(self) -> dict[str, Any]:
        """Get breaking news headlines."""
        client = self._get_client(settings.bodega_news_url)
        try:
            response = await client.get("/api/v1/news/headlines")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("headlines_failed", error=str(e))
            return {"error": str(e)}

    async def news_by_topic(
        self, topic: str = "TECHNOLOGY", period: str = "1d"
    ) -> dict[str, Any]:
        """Get news by topic. Topics: TECHNOLOGY, BUSINESS, SCIENCE, etc."""
        client = self._get_client(settings.bodega_news_url)
        try:
            response = await client.get(
                f"/api/v1/news/topics/{topic}",
                params={"period": period},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("news_by_topic_failed", topic=topic, error=str(e))
            return {"error": str(e)}

    # ---- Entertainment (:8031) ----

    async def movie_search(self, query: str) -> dict[str, Any]:
        """Search for movies."""
        client = self._get_client(settings.bodega_entertainment_url)
        try:
            response = await client.get(
                "/api/v1/entertainment/movies/search",
                params={"q": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("movie_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def tv_search(self, query: str) -> dict[str, Any]:
        """Search for TV shows."""
        client = self._get_client(settings.bodega_entertainment_url)
        try:
            response = await client.get(
                "/api/v1/entertainment/tv/search",
                params={"q": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("tv_search_failed", query=query, error=str(e))
            return {"error": str(e)}
