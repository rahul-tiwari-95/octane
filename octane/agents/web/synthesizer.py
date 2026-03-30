"""Synthesizer — raw results → structured intelligence with sources.

Session 10: LLM-powered synthesis for news and web search.
Finance data passes through as-is (already structured numbers).
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import structlog

from octane.tools.topology import ModelTier
from octane.utils.clock import today_human

if TYPE_CHECKING:
    from octane.agents.web.content_extractor import ExtractedContent

logger = structlog.get_logger().bind(component="web.synthesizer")

# Sentinel: distinguish "use default BodegaRouter" from "explicitly no LLM"
_UNSET = object()

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

_CHUNK_SYSTEM_BASE = """\
You are a precise summarizer. Given a long article excerpt, extract the \
key facts, figures, and insights relevant to the user's query.

Rules:
- Output a dense summary in 150–250 words.
- Preserve specific numbers, dates, names, and statistics exactly as written.
- Focus only on what is most relevant to the query. Omit tangential content.
- Do not add analysis beyond what is stated in the text. Do not invent facts.
- Do not mention "agent", "tool", or "API"."""

_FULL_TEXT_SYSTEM_BASE = """\
You are a research analyst with access to full article text. Synthesize the \
provided articles into a thorough, grounded response to the user's query.

Format:
- Lead with a direct 1–2 sentence answer to the query.
- Then 3–5 bullet points covering the key facts, trends, or insights.
- End with: Sources: <comma-separated domain names>

Rules:
- Ground every claim in the provided article texts.
- Preserve specific numbers, dates, and statistics exactly as written.
- Do not mention "agent", "tool", or "API".
- If sources conflict on a fact, note the conflict explicitly.
- If the articles don't answer the query, say so clearly and state what they do cover."""

_DEEP_SYNTHESIS_SYSTEM_BASE = """\
You are a senior research analyst producing a comprehensive intelligence briefing.
You have full article text from multiple authoritative sources. Your job is \
thorough analysis — Octane users rely on you for depth, not brevity.

Produce a STRUCTURED report covering ALL of the following sections:

## Summary
2-3 sentences answering the query directly with the single most important finding.

## Key Developments
5-8 specific factual bullet points. Each must include: date (if known), actor, \
and a concrete detail (number, name, direct quote, or specific action taken). \
No vague generalities.

## Background & Context
2-3 paragraphs explaining the history, causes, and key actors that make \
these developments significant. Connect the dots between events explicitly.

## International Reactions & Implications
What key parties (governments, organisations, analysts) have said or done \
in response. Include direct quotes from sources where available.

## What's Next
2-4 bullet points on the most likely near-term developments based on the \
evidence. Label each as [likely], [possible], or [uncertain].

## Sources
Comma-separated list of domain names used.

Rules:
- Every claim must be grounded directly in the provided article texts.
- Preserve specific dates, numbers, names, and statistics exactly as written.
- If sources conflict on a fact, note the conflict explicitly: "Sources differ: ...".
- Do NOT add analysis unsupported by the sources.
- Do not mention "agent", "tool", or "API".
- Target 700-1200 words total. Be thorough, not brief. Missing a section is \
  only acceptable if the sources genuinely provide no information for it."""


def _news_system() -> str:
    """Return the news system prompt with today's date injected."""
    return f"Today's date: {today_human()}.\n\n{_NEWS_SYSTEM_BASE}"


def _search_system() -> str:
    """Return the search system prompt with today's date injected."""
    return f"Today's date: {today_human()}.\n\n{_SEARCH_SYSTEM_BASE}"


def _chunk_system() -> str:
    """Return the chunk-summarizer system prompt with today's date injected."""
    return f"Today's date: {today_human()}.\n\n{_CHUNK_SYSTEM_BASE}"


def _full_text_system() -> str:
    """Return the full-text synthesis system prompt with today's date injected."""
    return f"Today's date: {today_human()}.\n\n{_FULL_TEXT_SYSTEM_BASE}"


def _deep_synthesis_system() -> str:
    """Return the deep-mode synthesis system prompt (REASON tier, structured report)."""
    return f"Today's date: {today_human()}.\n\n{_DEEP_SYNTHESIS_SYSTEM_BASE}"


class Synthesizer:
    """Turns raw Bodega Intelligence API results into structured intelligence.

    - News: LLM produces "Key stories: 1. ..." summary.
    - Search: LLM produces 2-4 bullet-point synthesis.
    - Finance: passes through unchanged (already structured numbers).
    - Full-text: deep synthesis from extracted article text (ContentExtractor output).
    - Falls back to plain text formatting if LLM is unavailable.
    """

    # Full-text synthesis thresholds — standard mode
    _MAX_CHARS_DIRECT: int = 3_000   # Use text directly if at or below this
    _MAX_CHARS_CHUNK: int = 6_000    # Truncate to this before chunk-summarizing

    # Full-text synthesis thresholds — deep mode (allow much larger direct window)
    _MAX_CHARS_DIRECT_DEEP: int = 8_000   # Direct synthesis up to 8 K chars in deep mode
    _MAX_CHARS_CHUNK_DEEP: int = 14_000   # Truncate before chunk-summarizing in deep mode

    def __init__(self, bodega=_UNSET) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
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
            from octane.utils.response_templates import apply_template
            try:
                result = await asyncio.wait_for(
                    self._bodega.chat_simple(
                        prompt=prompt,
                        system=apply_template(_news_system(), "news"),
                        tier=ModelTier.MID,
                        temperature=0.2,
                        max_tokens=512,
                    ),
                    timeout=30.0,  # news synthesis: 30 s cap
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
                result = await asyncio.wait_for(
                    self._bodega.chat_simple(
                        prompt=prompt,
                        system=_search_system(),
                        tier=ModelTier.MID,
                        temperature=0.2,
                        max_tokens=512,
                    ),
                    timeout=30.0,  # search synthesis: 30 s cap
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

    # ── Full-text synthesis ────────────────────────────────────────────────

    async def _summarize_chunk(self, text: str, query: str, deep: bool = False) -> str:
        """Pre-summarize a long article chunk to ~250-500 words.

        Takes the first _MAX_CHARS_CHUNK (or _MAX_CHARS_CHUNK_DEEP) chars of
        ``text`` and asks the LLM to compress it into a dense, query-focused
        summary.  Falls back to a plain truncation if the LLM is unavailable.
        """
        char_limit = self._MAX_CHARS_CHUNK_DEEP if deep else self._MAX_CHARS_CHUNK
        chunk_tokens = 600 if deep else 350
        truncated = text[:char_limit]
        prompt = f'Query context: "{query}"\n\nArticle excerpt:\n{truncated}'
        if self._bodega is not None:
            try:
                result = await asyncio.wait_for(
                    self._bodega.chat_simple(
                        prompt=prompt,
                        system=_chunk_system(),
                        tier=ModelTier.MID,
                        temperature=0.1,
                        max_tokens=chunk_tokens,
                    ),
                    timeout=25.0,  # chunk summarization: 25 s cap
                )
                return _strip_think(result.strip())
            except Exception as exc:
                logger.warning("chunk_summarize_failed", error=str(exc), fallback="truncate")
        # Fallback: first 500 chars as a rough stand-in
        return text[:500]

    async def synthesize_with_content(
        self,
        query: str,
        extracted_articles: list[ExtractedContent],
        deep: bool = False,
    ) -> str:
        """Synthesize full article text into deep, grounded intelligence.

        Pipeline:
        1. Skip articles with no usable text (method="unavailable"/"failed").
        2. For each article whose text exceeds the direct-use threshold, run a
           chunk-summarize pass first (_summarize_chunk) to compress it.
        3. Assemble the (possibly compressed) texts and call the LLM with
           a deeper full-text synthesis prompt.
        4. Falls back to a plain formatted listing if LLM is unavailable.

        Args:
            query:               The user's original query.
            extracted_articles:  List of ExtractedContent from ContentExtractor.
            deep:                When True, uses REASON tier, 3 000-token output,
                                 up to 10 articles, and a structured multi-section
                                 report prompt. Default False = legacy brief mode.

        At most 10 articles (deep) or 5 articles (standard) are used in the
        final synthesis to keep prompt size predictable.
        """
        # 1. Filter: keep only articles that have actual extracted text
        usable = [
            a for a in extracted_articles
            if a.text and a.method not in ("unavailable", "failed")
        ]
        if not usable:
            logger.info("synthesize_with_content_no_usable_text", query=query)
            return f"No article text could be extracted for: {query}"

        # Choose thresholds and LLM params based on mode
        article_limit = 10 if deep else 5
        max_chars_direct = self._MAX_CHARS_DIRECT_DEEP if deep else self._MAX_CHARS_DIRECT
        synthesis_tokens = 3000 if deep else 768
        synthesis_tier = ModelTier.REASON if deep else ModelTier.MID
        from octane.utils.response_templates import apply_template
        _raw_system = _deep_synthesis_system() if deep else _full_text_system()
        synthesis_system = apply_template(_raw_system, "search")
        synthesis_timeout = 120.0 if deep else 60.0  # REASON model is slower

        logger.info(
            "synthesize_with_content_start",
            query=query[:60],
            n_articles=len(usable),
            using=min(len(usable), article_limit),
            deep=deep,
        )

        # 2. Pre-process: chunk-summarize articles that exceed the direct limit
        article_blocks: list[str] = []
        for i, article in enumerate(usable[:article_limit], 1):
            text = article.text
            if len(text) > max_chars_direct:
                logger.debug(
                    "chunk_summarizing_article",
                    index=i,
                    url=article.url,
                    chars=len(text),
                    deep=deep,
                )
                text = await self._summarize_chunk(text, query, deep=deep)
            # Use domain as the source label (e.g. "reuters.com")
            parts = article.url.split("/")
            source_label = parts[2] if len(parts) > 2 else article.url
            article_blocks.append(f"[{i}] Source: {source_label}\n{text}")

        # 3. Assemble final prompt and synthesize
        assembled = "\n\n---\n\n".join(article_blocks)
        prompt = f'Query: "{query}"\n\nArticle texts:\n\n{assembled}'

        if self._bodega is not None:
            try:
                result = await asyncio.wait_for(
                    self._bodega.chat_simple(
                        prompt=prompt,
                        system=synthesis_system,
                        tier=synthesis_tier,
                        temperature=0.2,
                        max_tokens=synthesis_tokens,
                    ),
                    timeout=synthesis_timeout,
                )
                return _strip_think(result.strip())
            except asyncio.TimeoutError:
                logger.warning(
                    "full_text_synthesis_failed",
                    error=f"synthesis timeout ({synthesis_timeout}s)",
                    fallback="plain",
                    deep=deep,
                )
            except Exception as exc:
                logger.warning(
                    "full_text_synthesis_failed",
                    error=str(exc) or type(exc).__name__,
                    fallback="plain",
                    deep=deep,
                )

        # 4. Plain fallback: numbered list with short snippets
        lines = [f"Full-text results for '{query}':"]
        for i, article in enumerate(usable[:article_limit], 1):
            parts = article.url.split("/")
            domain = parts[2] if len(parts) > 2 else article.url
            snippet = article.text[:200].replace("\n", " ")
            lines.append(f"\n[{i}] {domain}")
            lines.append(f"    {snippet}…")
        return "\n".join(lines)


def _strip_think(text: str) -> str:
    """Remove any <think>...</think> blocks left in by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
