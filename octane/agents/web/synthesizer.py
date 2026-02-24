"""Synthesizer — raw results → structured intelligence with sources.

Session 10: LLM-powered synthesis for news and web search.
Finance data passes through as-is (already structured numbers).
"""

from __future__ import annotations

import re

import structlog

from octane.utils.clock import today_human

logger = structlog.get_logger().bind(component="web.synthesizer")

_NEWS_SYSTEM_BASE = """\
You are a news analyst. Given a list of raw news articles, produce a brief \
structured summary with exactly this format:

Key stories:
1. <headline> — <1-sentence insight> (<source>, <date>)
2. ...
3. ...

Rules:
- Maximum 3 stories. Pick the most significant.
- Each insight must be one sentence of factual analysis, not just a restatement.
- Do not invent facts. Use only what is in the provided articles.
- Do not mention "agent", "tool", or "API".
- If an article date is more than 30 days before today, note it as potentially outdated.
- End with: Sources: <comma-separated publisher names>"""

_SEARCH_SYSTEM_BASE = """\
You are a research assistant. Given raw web search results, synthesize the \
key information relevant to the user's query into 2-4 concise bullet points.

Rules:
- Ground every point in the provided results. Do not invent facts.
- Be direct. Skip filler phrases like "According to the results..."
- Do not mention "agent", "tool", or "API".
- If a result contains a date that is not recent (relative to today), flag it as stale.
- If the results don't answer the query, say so briefly."""


def _news_system() -> str:
    """Return the news system prompt with today's date injected."""
    return f"Today's date: {today_human()}.\n\n{_NEWS_SYSTEM_BASE}"


def _search_system() -> str:
    """Return the search system prompt with today's date injected."""
    return f"Today's date: {today_human()}.\n\n{_SEARCH_SYSTEM_BASE}"


class Synthesizer:
    """Turns raw Bodega Intelligence API results into structured intelligence.

    - News: LLM produces "Key stories: 1. ..." summary.
    - Search: LLM produces 2-4 bullet-point synthesis.
    - Finance: passes through unchanged (already structured numbers).
    - Falls back to plain text formatting if LLM is unavailable.
    """

    def __init__(self, bodega=None) -> None:
        self._bodega = bodega

    async def synthesize_news(self, query: str, articles: list[dict]) -> str:
        """Synthesize a list of news articles into a structured summary."""
        if not articles:
            return f"No recent news found for: {query}"

        # Format articles for the LLM prompt
        article_lines: list[str] = []
        for i, art in enumerate(articles[:7], 1):
            title = art.get("title", "Untitled")
            pub = art.get("publisher", {}).get("title", "")
            date = art.get("published date", "")[:10]
            summary = art.get("summary", art.get("description", ""))[:200]
            article_lines.append(
                f"[{i}] {title}\n"
                f"    Source: {pub} · {date}\n"
                f"    {summary}"
            )

        raw_text = "\n\n".join(article_lines)
        prompt = f'Query: "{query}"\n\nArticles:\n{raw_text}'

        if self._bodega is not None:
            try:
                result = await self._bodega.chat_simple(
                    prompt=prompt,
                    system=_news_system(),
                    temperature=0.2,
                    max_tokens=512,
                )
                return _strip_think(result.strip())
            except Exception as exc:
                logger.warning("news_synthesis_failed", error=str(exc), fallback="plain")

        # Fallback: plain numbered list
        lines = [f"Recent news for '{query}':"]
        for i, art in enumerate(articles[:5], 1):
            title = art.get("title", "Untitled")
            pub = art.get("publisher", {}).get("title", "")
            date = art.get("published date", "")[:16]
            lines.append(f"{i}. {title}")
            if pub or date:
                lines.append(f"   {pub} · {date}")
        return "\n".join(lines)

    async def synthesize_search(self, query: str, results: list[dict]) -> str:
        """Synthesize web search results into structured bullet points."""
        if not results:
            return f"No web results found for: {query}"

        result_lines: list[str] = []
        for i, r in enumerate(results[:6], 1):
            title = r.get("title", "Untitled")
            desc = r.get("description", r.get("extra_snippets", [""])[0] if isinstance(r.get("extra_snippets"), list) else "")
            url = r.get("url", "")
            result_lines.append(
                f"[{i}] {title}\n"
                f"    {str(desc)[:250]}\n"
                f"    URL: {url}"
            )

        raw_text = "\n\n".join(result_lines)
        prompt = f'Query: "{query}"\n\nSearch results:\n{raw_text}'

        if self._bodega is not None:
            try:
                result = await self._bodega.chat_simple(
                    prompt=prompt,
                    system=_search_system(),
                    temperature=0.2,
                    max_tokens=512,
                )
                return _strip_think(result.strip())
            except Exception as exc:
                logger.warning("search_synthesis_failed", error=str(exc), fallback="plain")

        # Fallback: plain list
        lines = [f"Web results for '{query}':"]
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "Untitled")
            desc = r.get("description", "")
            lines.append(f"{i}. {title}")
            if desc:
                lines.append(f"   {str(desc)[:150]}")
        return "\n".join(lines)


def _strip_think(text: str) -> str:
    """Remove any <think>...</think> blocks left in by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
