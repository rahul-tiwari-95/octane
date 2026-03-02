"""Depth Analyzer — iterative query deepening for web/news searches.

After a Round-1 search produces initial findings, DepthAnalyzer reads
the extracted snippets and generates 3-5 targeted follow-up queries
that drill deeper into the threads the first round surfaced.

Example:
    Round 1: "israel iran latest news"
    Findings: US-Israel struck Iran, Khamenei fate unclear
    Follow-ups:
        - "Khamenei death confirmed reports [month_year]"
        - "Israel military operations in Iran aftermath [month_year]"
        - "US involvement Israel Iran war international response [month_year]"
        - "Iran leadership succession vacuum [month_year]"
        - "Iran retaliation threat after Israel strike [month_year]"

Uses ModelTier.FAST (90M) for the follow-up generation — lightweight,
fast, keeps overall latency low while multiplying search breadth.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog

from octane.tools.topology import ModelTier
from octane.utils.clock import month_year, today_str

logger = structlog.get_logger().bind(component="web.depth_analyzer")

_SENTINEL = object()

_DEPTH_SYSTEM = """\
You are a search research analyst. You have just received the first round of \
search results for a user query. Your job is to identify the most important \
threads, gaps, and follow-up angles — then generate 3-5 targeted follow-up \
search queries that will deepen the research.

Return a JSON array. Each item must have:
  - "query": the follow-up search string (specific, includes date context)
  - "api": one of "search", "news", "finance"
  - "rationale": one short phrase (max 8 words) explaining the angle

Rules:
- Each follow-up must be MEANINGFULLY different from the original query.
- Include the current month/year in queries about ongoing events.
- Focus on: key actors, aftermath, causes, international reactions, timeline.
- Use "news" api for breaking developments. "search" for background/context.
- DO NOT repeat the original query. Expand and drill down.
- Return ONLY valid JSON. No prose, no markdown fences."""


class DepthAnalyzer:
    """Generates targeted follow-up queries from initial search findings.

    Args:
        bodega: BodegaRouter instance (or compatible duck-type).
                Pass ``None`` to disable LLM follow-up generation.
    """

    def __init__(self, bodega=_SENTINEL) -> None:
        if bodega is _SENTINEL:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega

    async def generate_followups(
        self,
        original_query: str,
        findings: list[str],
        max_followups: int = 4,
    ) -> list[dict[str, str]]:
        """Analyse findings from Round 1 and generate deeper follow-up queries.

        Args:
            original_query: The user's original search query.
            findings:       List of text snippets / extracted content from Round 1.
                            Can be article titles, descriptions, or full paragraphs.
            max_followups:  Maximum number of follow-up queries to return (default 4).

        Returns:
            List of strategy dicts (same format as QueryStrategist):
            [{"query": str, "api": str, "rationale": str}, ...]
        """
        if self._bodega is None or not findings:
            logger.debug("depth_analyzer_skipped", reason="no_bodega_or_no_findings")
            return []

        try:
            return await self._llm_followups(original_query, findings, max_followups)
        except Exception as exc:
            logger.warning("depth_analysis_failed", error=str(exc))
            return []

    async def _llm_followups(
        self,
        original_query: str,
        findings: list[str],
        max_followups: int,
    ) -> list[dict[str, str]]:
        """Use FAST-tier LLM to generate targeted follow-up queries."""
        # Truncate findings to avoid context overflow — 150 chars each, max 8 items
        condensed = "\n".join(
            f"- {f[:150]}" for f in findings[:8] if f and f.strip()
        )
        if not condensed:
            return []

        prompt = (
            f'Today is {today_str()} ({month_year()}).\n'
            f'Original query: "{original_query}"\n\n'
            f"Round-1 findings (excerpts):\n{condensed}\n\n"
            f"Generate {max_followups} targeted follow-up queries to deepen this research."
        )

        raw = await asyncio.wait_for(
            self._bodega.chat_simple(
                prompt=prompt,
                system=_DEPTH_SYSTEM,
                tier=ModelTier.FAST,
                temperature=0.4,
                max_tokens=2048,  # headroom for <think> + JSON output
            ),
            timeout=45.0,
        )

        # Strip <think> reasoning block if present
        if "</think>" in raw:
            think_part, _, clean = raw.partition("</think>")
            logger.debug(
                "depth_analyzer_reasoning",
                trace=think_part.replace("<think>", "").strip()[:300],
            )
        else:
            clean = raw

        # Extract JSON array
        json_match = re.search(r"\[.*\]", clean, flags=re.DOTALL)
        if not json_match:
            logger.warning("depth_analyzer_no_json", response_preview=clean[:120])
            return []

        strategies = json.loads(json_match.group(0))

        valid: list[dict[str, str]] = []
        valid_apis = {"search", "news", "finance"}
        for s in strategies:
            if isinstance(s, dict) and "query" in s:
                api = s.get("api", "search")
                if api not in valid_apis:
                    api = "search"
                valid.append({
                    "query": str(s["query"])[:200],
                    "api": api,
                    "rationale": str(s.get("rationale", ""))[:100],
                })

        result = valid[:max_followups]
        logger.info(
            "depth_followups_generated",
            original=original_query[:60],
            count=len(result),
            queries=[q["query"][:60] for q in result],
        )
        return result
