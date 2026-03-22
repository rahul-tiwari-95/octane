"""Async client for Bodega Intelligence APIs (external data).

All services run on a single consolidated server at :44467.
Requires Bearer token authentication (user key).

Route map (mount prefix + flow path):
    ── Beru / Brave Search ──────────────────────────────────────────────────
    Web search:        GET /api/v1/beru/search/web?query=...&count=...
    Image search:      GET /api/v1/beru/search/images?query=...
    Video search:      GET /api/v1/beru/search/videos?query=...
    Discussions:       GET /api/v1/beru/search/discussions?query=...
    Products:          GET /api/v1/beru/search/products?query=...
    Person:            GET /api/v1/beru/search/person?query=...
    Movie (Brave):     GET /api/v1/beru/search/movie?query=...
    FAQ:               GET /api/v1/beru/search/faq?query=...
    Locations:         GET /api/v1/beru/search/locations?query=...
    Combined:          GET /api/v1/beru/search/combined?query=...

    ── News ────────────────────────────────────────────────────────────────
    Search:            GET /api/v1/news/api/v1/news/search?q=...&period=...&max_results=...
    Headlines:         GET /api/v1/news/api/v1/news/headlines
    By topic:          GET /api/v1/news/api/v1/news/topics/{topic}
    By location:       GET /api/v1/news/api/v1/news/locations/{location}
    By site:           GET /api/v1/news/api/v1/news/sites/{domain}
    Trending:          GET /api/v1/news/api/v1/news/trending

    ── Finance ─────────────────────────────────────────────────────────────
    Market data:       GET /api/v1/finance/api/v1/finance/market/{symbol}
    Complete:          GET /api/v1/finance/api/v1/finance/complete?symbol=...
    Timeseries:        GET /api/v1/finance/api/v1/finance/timeseries/{symbol}?period=...
    Statements:        GET /api/v1/finance/api/v1/finance/statements/{symbol}
    Options chain:     GET /api/v1/finance/api/v1/finance/options/{symbol}

    ── Entertainment (TMDB) ────────────────────────────────────────────────
    Movie search:      GET /api/v1/entertainment/api/v1/entertainment/movies/search?query=...
    Movie details:     GET /api/v1/entertainment/api/v1/entertainment/movies/{movie_id}
    TV search:         GET /api/v1/entertainment/api/v1/entertainment/tv/search?query=...
    Popular movies:    GET /api/v1/entertainment/api/v1/entertainment/movies/popular
    Top-rated movies:  GET /api/v1/entertainment/api/v1/entertainment/movies/top-rated

    ── Music (YouTube Music) ───────────────────────────────────────────────
    Search:            GET /api/v1/music/search?query=...
    Artist:            GET /api/v1/music/artist/{artist_identifier}
    Song lyrics:       GET /api/v1/music/song/{video_id}/lyrics

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

    # ── Beru / Web Search ─────────────────────────────────────────────────
    # NOTE: All /api/v1/beru/search/* endpoints are typed `-> str` in the
    # server and return json.dumps(data) — so response.json() yields a Python
    # str, not a dict.  _unwrap_beru() handles that one extra decode step.

    @staticmethod
    def _unwrap_beru(raw: Any) -> dict[str, Any]:
        """Unwrap the extra JSON-encode that all beru/search endpoints apply.

        FastAPI serialises the `-> str` return value as a quoted JSON string.
        httpx's response.json() gives us that Python str; json.loads() gives
        us the actual dict.

        The beru server also wraps the Brave Search payload inside a
        ``beru_response`` key alongside a ``query_details`` key.  We hoist the
        ``beru_response`` contents to the top level so callers can access
        ``result["web"]["results"]`` directly, consistent with the old layout.
        """
        import json as _json
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    raw = parsed
                else:
                    return {"result": parsed}
            except Exception:
                return {"error": f"Unexpected beru string response: {raw[:120]}"}
        if not isinstance(raw, dict):
            return {"error": f"Unexpected beru response type: {type(raw).__name__}"}

        # Hoist beru_response to top level so web/discussions/news are directly accessible
        beru = raw.get("beru_response")
        if isinstance(beru, dict):
            merged = dict(raw)      # preserve query_details etc.
            merged.update(beru)     # web, discussions, news, videos, mixed, infobox
            return merged
        return raw

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
            result = self._unwrap_beru(response.json())
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

    async def finance_statements(self, ticker: str) -> dict[str, Any]:
        """Get financial statements (income, balance sheet, cash flow)."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/finance/api/v1/finance/statements/{ticker.upper()}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("finance_statements_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    async def finance_options(self, ticker: str) -> dict[str, Any]:
        """Get options chain (calls + puts) for a ticker."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/finance/api/v1/finance/options/{ticker.upper()}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("finance_options_failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    # ── Beru Extended Search ───────────────────────────────────────────────

    async def search_images(self, query: str) -> dict[str, Any]:
        """Image search via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/images",
                params={"query": query},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_images_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_videos(self, query: str) -> dict[str, Any]:
        """Video search via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/videos",
                params={"query": query},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_videos_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_discussions(self, query: str) -> dict[str, Any]:
        """Forum/discussion search (Reddit, forums, etc.) via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/discussions",
                params={"query": query},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_discussions_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_products(self, query: str) -> dict[str, Any]:
        """Product/shopping search via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/products",
                params={"query": query},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_products_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_person(self, query: str) -> dict[str, Any]:
        """Person/biography search via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/person",
                params={"query": query},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_person_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_locations(self, query: str) -> dict[str, Any]:
        """Location/place search via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/locations",
                params={"query": query},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_locations_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_combined(self, query: str, count: int = 5) -> dict[str, Any]:
        """Combined multi-type search (web + news + discussions) via Beru/Brave."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/beru/search/combined",
                params={"query": query, "count": count},
            )
            response.raise_for_status()
            return self._unwrap_beru(response.json())
        except httpx.HTTPError as e:
            logger.error("search_combined_failed", query=query, error=str(e))
            return {"error": str(e)}

    # ── News Extended ─────────────────────────────────────────────────────

    async def news_trending(self) -> dict[str, Any]:
        """Get trending news topics."""
        client = await self._get_client()
        try:
            response = await client.get("/api/v1/news/api/v1/news/trending")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("news_trending_failed", error=str(e))
            return {"error": str(e)}

    async def news_by_location(self, location: str, period: str = "1d") -> dict[str, Any]:
        """Get news filtered by geographic location."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/news/api/v1/news/locations/{location}",
                params={"period": period},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("news_by_location_failed", location=location, error=str(e))
            return {"error": str(e)}

    async def news_by_site(self, domain: str, period: str = "7d") -> dict[str, Any]:
        """Get news from a specific publication/domain (e.g. 'reuters.com')."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/news/api/v1/news/sites/{domain}",
                params={"period": period},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("news_by_site_failed", domain=domain, error=str(e))
            return {"error": str(e)}

    # ── Entertainment (TMDB) ──────────────────────────────────────────────

    async def movies_search(self, query: str) -> dict[str, Any]:
        """Search movies/TV via TMDB."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/entertainment/api/v1/entertainment/movies/search",
                params={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("movies_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def movie_details(self, movie_id: int | str) -> dict[str, Any]:
        """Get detailed info for a TMDB movie by ID."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/entertainment/api/v1/entertainment/movies/{movie_id}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("movie_details_failed", movie_id=movie_id, error=str(e))
            return {"error": str(e)}

    async def tv_search(self, query: str) -> dict[str, Any]:
        """Search TV shows via TMDB."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/entertainment/api/v1/entertainment/tv/search",
                params={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("tv_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def movies_popular(self) -> dict[str, Any]:
        """Get currently popular movies from TMDB."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/entertainment/api/v1/entertainment/movies/popular"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("movies_popular_failed", error=str(e))
            return {"error": str(e)}

    # ── Music (YouTube Music) ─────────────────────────────────────────────

    async def music_search(self, query: str) -> dict[str, Any]:
        """Search music (artists, albums, songs) via YouTube Music."""
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/v1/music/search",
                params={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("music_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def music_artist(self, artist_identifier: str) -> dict[str, Any]:
        """Get artist profile and discography via YouTube Music."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/music/artist/{artist_identifier}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("music_artist_failed", artist=artist_identifier, error=str(e))
            return {"error": str(e)}

    async def song_lyrics(self, video_id: str) -> dict[str, Any]:
        """Get song lyrics by YouTube video ID."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/v1/music/song/{video_id}/lyrics"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("song_lyrics_failed", video_id=video_id, error=str(e))
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
