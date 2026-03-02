"""Web Agent — fetches real data from Bodega Intelligence APIs.

Routes to the correct API based on the sub_agent hint in the task metadata:
    web_finance  → market_data() + optional timeseries()
    web_news     → news_search() + web_search() (parallel) → full-text extraction → LLM synthesis
    web_search   → web_search() (Beru/Brave) → full-text extraction → LLM synthesis

The primary search backbone is /api/v1/beru/search/web (Google/Brave/Bing).
For news-flavoured queries the dedicated news API runs in parallel so both
fresh articles AND broader web results feed the synthesiser.

Session 10: QueryStrategist generates 2-3 search variations.
            Synthesizer uses LLM to turn raw results into structured intelligence.
Session 15: ContentExtractor (trafilatura) + BrowserAgent (Playwright) wire in for
            deep full-text synthesis; snippets remain as automatic fallback.
"""

from __future__ import annotations

import asyncio
import re

import structlog

from octane.agents.base import BaseAgent
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.bodega_intel import BodegaIntelClient
from octane.agents.web.synthesizer import Synthesizer
from octane.agents.web.query_strategist import QueryStrategist
from octane.agents.web.depth_analyzer import DepthAnalyzer
from octane.agents.web.content_extractor import ContentExtractor, ExtractedContent
from octane.agents.web.browser import BrowserAgent
from octane.utils.clock import month_year, today_str
from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="web_agent")


class WebAgent(BaseAgent):
    """Web Agent — coordinates internet data retrieval via Bodega Intelligence."""

    name = "web"

    def __init__(
        self,
        synapse,
        intel: BodegaIntelClient | None = None,
        bodega=None,
        extractor: ContentExtractor | None = None,
        browser: BrowserAgent | None = None,
        page_store=None,  # optional WebPageStore — injected by Orchestrator (18A)
    ) -> None:
        super().__init__(synapse)
        self._bodega = bodega
        self._intel = intel or BodegaIntelClient()
        self._synthesizer = Synthesizer(bodega=bodega)
        self._strategist = QueryStrategist(bodega=bodega)
        self._depth_analyzer = DepthAnalyzer(bodega=bodega)
        self._extractor = extractor or ContentExtractor()
        self._browser = browser or BrowserAgent(interactive=False)
        self._page_store = page_store  # WebPageStore | None

    async def _store_pages(self, extracted: list[ExtractedContent]) -> None:
        """Persist successfully extracted pages to WebPageStore (fire-and-forget)."""
        if self._page_store is None:
            return
        for item in extracted:
            if item.text and item.method not in ("unavailable", "failed"):
                try:
                    await self._page_store.store(
                        url=item.url,
                        content=item.text,
                        title=item.title if hasattr(item, "title") else "",
                    )
                except Exception as exc:
                    logger.debug("web_page_store_failed", url=item.url, error=str(exc))

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route to the correct Bodega Intelligence API based on sub_agent hint."""
        sub_agent = request.metadata.get("sub_agent", "search")
        query = request.query
        deep: bool = bool(request.metadata.get("deep", False))

        # Accept both short forms ("finance") and template-prefixed forms ("web_finance")
        if sub_agent in ("finance", "web_finance"):
            return await self._fetch_finance(query, request)
        elif sub_agent in ("news", "web_news"):
            return await self._fetch_news(query, request, deep=deep)
        else:
            # "search", "web_search", or any unknown → general Brave search
            return await self._fetch_search(query, request, deep=deep)

    # ── Finance ───────────────────────────────────────────────────────────

    _TIMESERIES_KEYWORDS = frozenset([
        "chart", "graph", "plot", "history", "historical", "timeseries",
        "time series", "last month", "past month", "last week", "over time",
        "price history", "trend", "ohlc", "candlestick", "30 day", "30-day",
    ])

    def _wants_timeseries(self, query: str) -> bool:
        """Return True when the query needs historical price data, not just a snapshot."""
        q = query.lower()
        return any(kw in q for kw in self._TIMESERIES_KEYWORDS)

    async def _fetch_finance(self, query: str, request: AgentRequest) -> AgentResponse:
        """Fetch market data. Extracts ticker symbol from query if present.

        When the query requests a chart / historical data, also fetches the
        30-day timeseries and includes it as a CSV-formatted block so that a
        downstream CodeAgent can plot it directly without re-fetching.
        """
        ticker = await self._extract_ticker(query)

        if ticker:
            raw = await self._intel.market_data(ticker)
            if "error" in raw:
                return AgentResponse(
                    agent=self.name, success=False,
                    error=f"Finance API error: {raw['error']}",
                    correlation_id=request.correlation_id,
                )
            summary = self._format_market_data(raw, ticker)

            # If the query wants a chart/history, fetch the 30-day timeseries too
            if self._wants_timeseries(query):
                ts_raw = await self._intel.timeseries(ticker, period="1mo", interval="1d")
                ts_rows = ts_raw.get("time_series", [])
                if ts_rows:
                    summary = summary + "\n\n" + self._format_timeseries_csv(ticker, ts_rows)
                    raw = {"market_data": raw, "timeseries": ts_raw}
        else:
            # No ticker found — fall back to a date-enriched web search so
            # Brave returns current results, not cached pages from months ago.
            dated_query = f"{query} {month_year()}"
            raw = await self._intel.web_search(dated_query, count=3)
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

    async def _extract_ticker(self, query: str) -> str | None:
        """Extract a ticker symbol from the query, with LLM+web as a smart fallback.

        Resolution order:
        1. Regex — explicit all-caps ticker already in the query (e.g. "NVDA earnings").
        2. Known name→ticker map — fast lookup for common companies.
        3. Web search + LLM — for unknown companies like DoorDash, Palantir, Carvana, etc.
           Queries Brave for "{query} stock ticker symbol", asks the LLM to extract it.
           This lets users write "DoorDash price" and still get DASH resolved correctly.
        """
        # Fast path 1: explicit all-caps ticker: NVDA, AAPL, TSLA, MSFT etc.
        match = re.search(r'\b([A-Z]{2,5})\b', query)
        if match:
            return match.group(1)
        # Fast path 2: known name→ticker map for common queries
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
        # Slow path 3: web search + LLM resolution for unknown company names
        return await self._resolve_ticker_via_web(query)

    async def _resolve_ticker_via_web(self, query: str) -> str | None:
        """Use a quick web search + LLM to resolve a company name to a stock ticker.

        Example: "DoorDash price" → search "DoorDash stock ticker symbol"
                 → LLM reads snippets → returns "DASH"

        The LLM is kept deliberately constrained (max_tokens=10, temperature=0.0) so
        this is a near-instant lookup rather than a full synthesis pass.
        """
        if self._bodega is None:
            return None
        try:
            raw = await self._intel.web_search(
                f"{query} stock ticker symbol", count=3
            )
            results = raw.get("web", {}).get("results", [])[:3]
            if not results:
                return None

            snippets = "\n".join(
                f"- {r.get('title', '')}: {r.get('description', '')[:150]}"
                for r in results
            )
            prompt = (
                f'Query: "{query}"\n\n'
                f"Web search results:\n{snippets}\n\n"
                f"What is the stock ticker symbol for the company in this query? "
                f"Reply with ONLY the ticker (e.g. AAPL, TSLA, DASH). "
                f'If no ticker can be determined, reply with "NONE".'
            )
            ticker_raw = await self._bodega.chat_simple(
                prompt=prompt,
                system="You are a financial assistant. Extract stock ticker symbols from text.",
                tier=ModelTier.FAST,
                temperature=0.0,
                max_tokens=10,
            )
            ticker = ticker_raw.strip().upper().split()[0] if ticker_raw else ""
            if re.match(r'^[A-Z]{1,5}$', ticker) and ticker != "NONE":
                logger.info("ticker_resolved_via_web", query=query, ticker=ticker)
                return ticker
            return None
        except Exception as exc:
            logger.warning("ticker_web_resolve_failed", query=query, error=str(exc))
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

    def _format_timeseries_csv(self, ticker: str, rows: list[dict]) -> str:
        """Serialise timeseries rows as a labelled CSV block for downstream agents.

        Downstream CodeAgent can parse this directly — no extra API calls needed.
        Each row: {'timestamp': '2026-01-26 00:00:00', 'open':…, 'high':…,
                   'low':…, 'close':…, 'volume':…}
        """
        lines = [f"# {ticker} 30-day price history (date,close,volume)"]
        lines.append("date,close,volume")
        for row in rows:
            ts = row.get("timestamp", "")
            # Normalise '2026-01-26 00:00:00' → '2026-01-26'
            date_str = ts[:10] if ts else ""
            close = round(row.get("close", 0), 2)
            volume = row.get("volume", 0)
            lines.append(f"{date_str},{close},{volume}")
        return "\n".join(lines)

    def _format_web_results(self, raw: dict, query: str) -> str:
        """Turn a generic web_search response into a readable summary string.

        Called when a finance query has no recognised ticker symbol and falls
        back to a broad web search (e.g. 'netscout systems price').
        """
        web = raw.get("web", {})
        results = web.get("results", [])
        if not results:
            # Some APIs return top-level 'results' or 'discussions'
            results = raw.get("results", []) or raw.get("discussions", {}).get("results", [])

        if not results:
            return f"No web results found for '{query}'."

        lines = [f"Web results for '{query}' (as of {today_str()}):"]
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "").strip()
            url = r.get("url", r.get("link", ""))
            snippet = r.get("description", r.get("snippet", "")).strip()
            if title:
                lines.append(f"{i}. {title}")
            if snippet:
                lines.append(f"   {snippet[:120]}")
            if url:
                lines.append(f"   {url}")
        return "\n".join(lines)

    # ── News ──────────────────────────────────────────────────────────────

    async def _enrich_with_browser(
        self,
        extracted: list[ExtractedContent],
        max_attempts: int = 2,
    ) -> list[ExtractedContent]:
        """Retry failed trafilatura extractions with a headless BrowserAgent.

        At most ``max_attempts`` browser scrapes are performed to keep
        latency bounded.  The BrowserAgent is always instantiated with
        ``interactive=False`` — no headed window or terminal prompt is ever
        shown in automated contexts.
        """
        attempts = 0
        result = list(extracted)
        for i, item in enumerate(result):
            if attempts >= max_attempts:
                break
            if not item.text or item.method in ("unavailable", "failed"):
                raw_text: str | None = await self._browser.scrape(item.url)
                if raw_text:
                    text = raw_text[:3_000]
                    result[i] = ExtractedContent(
                        url=item.url,
                        text=text,
                        word_count=len(text.split()),
                        method="browser",
                    )
                    attempts += 1
                    logger.debug(
                        "browser_enrich_success",
                        url=item.url,
                        chars=len(text),
                    )
        return result

    async def _fetch_news(self, query: str, request: AgentRequest, deep: bool = False) -> AgentResponse:
        """Fetch news for the query using two parallel sources, then synthesize.

        Sources:
        1. News API  (/api/v1/news/…/search)  — recent articles with structured metadata.
        2. Beru web  (/api/v1/beru/search/web) — Google/Brave/Bing; catches analysis,
           opinion pieces, and blog posts that the news feed misses.

        Both calls run concurrently.  URLs from both are deduplicated and fed into
        the same content-extraction + synthesis pipeline so the LLM always has the
        richest available source material.

        When ``deep=True`` (or ``--deep`` flag), a second round of targeted follow-up
        queries is generated from Round-1 findings and executed in parallel, giving
        much broader topic coverage before final synthesis.
        """
        strategies = await self._strategist.strategize(
            query, context={"sub_agent": "news"}
        )
        search_query = strategies[0].get("query", query)

        # ── Round 1: news API + beru web search in parallel ──────────────────
        news_raw, web_raw = await asyncio.gather(
            self._intel.news_search(search_query, period="3d", max_results=7),
            self._intel.web_search(f"{search_query} {today_str()}", count=6),
            return_exceptions=True,
        )

        if isinstance(news_raw, Exception):
            logger.warning("news_api_failed", error=str(news_raw))
            news_raw = {}
        if isinstance(web_raw, Exception):
            logger.warning("web_search_for_news_failed", error=str(web_raw))
            web_raw = {}

        articles = news_raw.get("articles", [])
        web_results = web_raw.get("web", {}).get("results", [])

        seen: set[str] = set()
        combined_urls: list[str] = []

        for a in articles:
            u = a.get("url") or a.get("link") or a.get("href", "")
            if u and u not in seen:
                seen.add(u)
                combined_urls.append(u)
        for r in web_results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                combined_urls.append(u)

        # ── Round-1 extraction ────────────────────────────────────────────────
        r1_usable: list[ExtractedContent] = []
        if combined_urls:
            extracted = await self._extractor.extract_batch(combined_urls, top_n=6)
            extracted = await self._enrich_with_browser(extracted)
            r1_usable = [
                a for a in extracted
                if a.text and a.method not in ("unavailable", "failed")
            ]
            if r1_usable:
                await self._store_pages(r1_usable)

        logger.info(
            "news_round1_complete",
            query=query[:60],
            n_urls=len(combined_urls),
            n_extracted=len(r1_usable),
        )

        # ── Round 2: iterative deepening (always on for news; more on --deep) ─
        all_usable = list(r1_usable)
        r2_rounds = 2 if deep else 1
        for round_num in range(r2_rounds):
            if not all_usable:
                break  # nothing to analyse — skip deepening
            findings = [a.text[:200] for a in all_usable[:6] if a.text]
            max_fups = 5 if deep else 3
            followups = await self._depth_analyzer.generate_followups(
                original_query=query,
                findings=findings,
                max_followups=max_fups,
            )
            if not followups:
                break

            logger.info(
                "news_deepening_round",
                round=round_num + 2,
                n_followups=len(followups),
                queries=[f["query"][:60] for f in followups],
            )

            # Fan out follow-up searches (mix of news and web based on api hint)
            deep_tasks = []
            for f in followups:
                api = f.get("api", "search")
                q = f["query"]
                if api == "news":
                    deep_tasks.append(self._intel.news_search(q, period="3d", max_results=5))
                else:
                    deep_tasks.append(self._intel.web_search(q, count=5))

            deep_raws = await asyncio.gather(*deep_tasks, return_exceptions=True)
            new_urls: list[str] = []
            for raw_resp in deep_raws:
                if isinstance(raw_resp, Exception):
                    continue
                for a in raw_resp.get("articles", []):
                    u = a.get("url") or a.get("link") or a.get("href", "")
                    if u and u not in seen:
                        seen.add(u)
                        new_urls.append(u)
                for r in raw_resp.get("web", {}).get("results", []):
                    u = r.get("url", "")
                    if u and u not in seen:
                        seen.add(u)
                        new_urls.append(u)

            if new_urls:
                r2_extracted = await self._extractor.extract_batch(new_urls, top_n=6)
                r2_extracted = await self._enrich_with_browser(r2_extracted)
                r2_usable = [
                    a for a in r2_extracted
                    if a.text and a.method not in ("unavailable", "failed")
                ]
                if r2_usable:
                    await self._store_pages(r2_usable)
                    all_usable.extend(r2_usable)
                    logger.info(
                        "news_deepening_extracted",
                        round=round_num + 2,
                        new_pages=len(r2_usable),
                        total=len(all_usable),
                    )

        # ── Synthesis ─────────────────────────────────────────────────────────
        if all_usable:
            summary = await self._synthesizer.synthesize_with_content(query, all_usable)
            return AgentResponse(
                agent=self.name, success=True,
                output=summary, data={"news": news_raw, "web": web_raw},
                correlation_id=request.correlation_id,
            )

        # Fallback: snippet-based synthesis from whatever we got
        logger.info(
            "news_extraction_fallback_to_summaries",
            query=query,
            n_articles=len(articles),
            n_web=len(web_results),
        )
        if articles:
            summary = await self._synthesizer.synthesize_news(query, articles)
        elif web_results:
            summary = await self._synthesizer.synthesize_search(query, web_results)
        else:
            summary = f"No results found for '{query}'."

        return AgentResponse(
            agent=self.name, success=True,
            output=summary, data={"news": news_raw, "web": web_raw},
            correlation_id=request.correlation_id,
        )

    # ── Web Search ────────────────────────────────────────────────────────

    async def _fetch_search(self, query: str, request: AgentRequest, deep: bool = False) -> AgentResponse:
        """General web search via Beru/Brave, with multi-variation strategy + LLM synthesis.

        Extraction cascade:
        1. trafilatura (ContentExtractor) on result URLs — fast, quiet.
        2. Headless BrowserAgent on any failed URLs (max 2 retries).
        3. synthesize_with_content() if any usable text was extracted.
        4. Fallback to synthesize_search() with snippets if all extraction fails.

        When ``deep=True``, a DepthAnalyzer pass generates 3-5 follow-up queries
        from Round-1 findings and runs them in parallel before final synthesis —
        dramatically broadening topic coverage.
        """
        # Generate 2-3 search variations
        strategies = await self._strategist.strategize(query)

        # Fan out all strategies in parallel, then deduplicate results by URL
        search_tasks = [
            self._intel.web_search(s["query"], count=6)
            for s in strategies
        ]
        raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results from all strategies, deduplicating by URL
        seen_urls: set[str] = set()
        results: list[dict] = []
        raw = {}
        for i, raw_resp in enumerate(raw_results):
            if isinstance(raw_resp, Exception):
                logger.warning("strategy_search_failed", query=strategies[i]["query"], error=str(raw_resp))
                continue
            if "error" in raw_resp:
                continue
            if not raw:
                raw = raw_resp  # keep first successful raw for data passthrough
            for r in raw_resp.get("web", {}).get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(r)
            for r in raw_resp.get("discussions", {}).get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(r)

        if not results:
            return AgentResponse(
                agent=self.name, success=False,
                error="Search API returned no results",
                correlation_id=request.correlation_id,
            )

        # ── Round-1 extraction ────────────────────────────────────────────────
        urls = [r.get("url", "") for r in results if r.get("url")]
        r1_usable: list[ExtractedContent] = []
        if urls:
            extracted = await self._extractor.extract_batch(urls, top_n=5)
            extracted = await self._enrich_with_browser(extracted)
            r1_usable = [
                a for a in extracted
                if a.text and a.method not in ("unavailable", "failed")
            ]
            if r1_usable:
                await self._store_pages(r1_usable)

        logger.info(
            "search_round1_complete",
            query=query[:60],
            n_strategies=len(strategies),
            n_results=len(results),
            n_extracted=len(r1_usable),
        )

        # ── Round 2: iterative deepening (--deep flag) ───────────────────────
        all_usable = list(r1_usable)
        if deep and all_usable:
            findings = [a.text[:200] for a in all_usable[:6] if a.text]
            followups = await self._depth_analyzer.generate_followups(
                original_query=query,
                findings=findings,
                max_followups=5,
            )
            if followups:
                logger.info(
                    "search_deepening_round2",
                    n_followups=len(followups),
                    queries=[f["query"][:60] for f in followups],
                )
                deep_tasks = [
                    self._intel.web_search(f["query"], count=5) for f in followups
                ]
                deep_raws = await asyncio.gather(*deep_tasks, return_exceptions=True)
                new_urls: list[str] = []
                for raw_resp in deep_raws:
                    if isinstance(raw_resp, Exception):
                        continue
                    for r in raw_resp.get("web", {}).get("results", []):
                        u = r.get("url", "")
                        if u and u not in seen_urls:
                            seen_urls.add(u)
                            new_urls.append(u)
                    for r in raw_resp.get("discussions", {}).get("results", []):
                        u = r.get("url", "")
                        if u and u not in seen_urls:
                            seen_urls.add(u)
                            new_urls.append(u)

                if new_urls:
                    r2_extracted = await self._extractor.extract_batch(new_urls, top_n=6)
                    r2_extracted = await self._enrich_with_browser(r2_extracted)
                    r2_usable = [
                        a for a in r2_extracted
                        if a.text and a.method not in ("unavailable", "failed")
                    ]
                    if r2_usable:
                        await self._store_pages(r2_usable)
                        all_usable.extend(r2_usable)
                        logger.info(
                            "search_deepening_extracted",
                            new_pages=len(r2_usable),
                            total=len(all_usable),
                        )

        # ── Synthesis ─────────────────────────────────────────────────────────
        if all_usable:
            summary = await self._synthesizer.synthesize_with_content(query, all_usable)
            return AgentResponse(
                agent=self.name, success=True,
                output=summary, data=raw,
                correlation_id=request.correlation_id,
            )

        # Fallback: snippet-based synthesis (original behaviour)
        logger.info("search_extraction_fallback_to_snippets", query=query, n_results=len(results))
        summary = await self._synthesizer.synthesize_search(query, results)
        return AgentResponse(
            agent=self.name, success=True,
            output=summary, data=raw,
            correlation_id=request.correlation_id,
        )

