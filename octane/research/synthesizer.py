"""ResearchSynthesizer — generates narrative reports from stored research findings.

Consumes findings from ResearchStore (Postgres) and produces a coherent summary
using either the Bodega LLM (preferred) or a plain-text fallback.

Usage (CLI):
    octane research report <task_id>
    octane research report <task_id> --cycles 3
    octane research report <task_id> --since 2026-01-01
    octane research report <task_id> --export ~/reports/nvda.md

Usage (programmatic):
    synth = ResearchSynthesizer(store, bodega=bodega)
    report = await synth.generate(task_id)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from octane.research.models import ResearchFinding
    from octane.research.store import ResearchStore

from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="research.synthesizer")

# Max characters of findings text sent to LLM in a single prompt.
# Above this threshold findings are compressed first.
_MAX_DIRECT_CHARS = 6_000

_SYNTHESIS_SYSTEM = """\
You are a research analyst. You have been given a series of research findings
collected over multiple investigation cycles on a specific topic.

Your task:
1. Synthesize all findings into a cohesive, well-structured report.
2. Highlight key insights, recurring themes, and any contradictions.
3. Include a brief executive summary at the top.
4. End with a "Key Takeaways" section (3–5 bullets).

Output format: clean markdown with headers. Be analytical, not just descriptive.
Do not include meta-commentary about the synthesis process itself.
"""


class ResearchSynthesizer:
    """Produces narrative reports from accumulated research findings.

    Args:
        store:  ResearchStore instance (for fetching findings).
        bodega: BodegaInferenceClient (optional — falls back to plain text).
    """

    def __init__(self, store: "ResearchStore", bodega=None) -> None:
        self._store = store
        self._bodega = bodega

    async def generate(
        self,
        task_id: str,
        *,
        cycles: int | None = None,
        since: datetime | None = None,
        max_tokens: int = 1500,
    ) -> str:
        """Generate a synthesized report for the given task.

        Args:
            task_id:    The research task ID.
            cycles:     If set, use only the last N cycle findings.
            since:      If set, use only findings created on/after this timestamp.
            max_tokens: Maximum tokens for the LLM response.

        Returns:
            Markdown-formatted report string.
        """
        findings = await self._store.get_findings(task_id)

        if not findings:
            return f"## Research Report\n\nNo findings available for task `{task_id}`."

        # Apply filters
        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            findings = [f for f in findings if (f.created_at or datetime.min.replace(tzinfo=timezone.utc)) >= since]

        if cycles is not None:
            findings = findings[-cycles:] if cycles > 0 else []

        if not findings:
            return (
                f"## Research Report\n\n"
                f"No findings match the requested filter for task `{task_id}`."
            )

        logger.info(
            "synthesizer_generating",
            task_id=task_id,
            finding_count=len(findings),
            has_llm=self._bodega is not None,
        )

        if self._bodega is None:
            return self._format_plain(findings, task_id)

        return await self._synthesize_with_llm(findings, task_id, max_tokens=max_tokens)

    # ── LLM synthesis ─────────────────────────────────────────────────────────

    async def _synthesize_with_llm(
        self,
        findings: list["ResearchFinding"],
        task_id: str,
        max_tokens: int = 1500,
    ) -> str:
        topic = findings[0].topic if findings else "Unknown"
        combined = self._combine_findings(findings)

        # Compress if too long
        if len(combined) > _MAX_DIRECT_CHARS:
            combined = await self._compress(combined, topic)

        prompt = (
            f"Topic: {topic}\n"
            f"Cycles researched: {len(findings)}\n"
            f"Total words collected: {sum(f.word_count for f in findings)}\n\n"
            f"---\n\n"
            f"{combined}"
        )

        try:
            result = await self._bodega.chat_simple(
                prompt=prompt,
                system=_SYNTHESIS_SYSTEM,
                tier=ModelTier.REASON,   # final narrative synthesis → deep model
                temperature=0.3,
                max_tokens=max_tokens,
            )
            # Strip <think> blocks from reasoning models
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result
        except Exception as exc:
            logger.warning("synthesizer_llm_failed", error=str(exc), task_id=task_id)
            return self._format_plain(findings, task_id)

    async def _compress(self, text: str, topic: str) -> str:
        """Compress long findings text before synthesis."""
        if self._bodega is None:
            return text[:_MAX_DIRECT_CHARS]

        compress_prompt = (
            f"The following is a collection of research findings about '{topic}'. "
            f"Compress them into the most important facts, removing duplication. "
            f"Preserve key data points, quotes, and source names.\n\n{text}"
        )
        try:
            compressed = await self._bodega.chat_simple(
                prompt=compress_prompt,
                system="You are a research editor. Compress information faithfully.",
                tier=ModelTier.MID,   # compression: mid-tier sufficient
                temperature=0.1,
                max_tokens=800,
            )
            return re.sub(r"<think>.*?</think>", "", compressed, flags=re.DOTALL).strip()
        except Exception:
            return text[:_MAX_DIRECT_CHARS]

    # ── Plain-text fallback ───────────────────────────────────────────────────

    def _format_plain(
        self,
        findings: list["ResearchFinding"],
        task_id: str,
    ) -> str:
        """Format findings as plain markdown without LLM synthesis."""
        topic = findings[0].topic if findings else "Unknown"
        total_words = sum(f.word_count for f in findings)

        lines = [
            f"## Research Report: {topic}",
            f"",
            f"**Task ID:** `{task_id}`  ",
            f"**Findings:** {len(findings)}  ",
            f"**Total words:** {total_words:,}",
            f"",
            "---",
            "",
        ]

        for i, finding in enumerate(findings, 1):
            created = ""
            if finding.created_at:
                created = finding.created_at.strftime("%Y-%m-%d %H:%M UTC")

            lines.append(f"### Cycle {finding.cycle_num} — {created}")
            lines.append("")
            lines.append(finding.content)
            lines.append("")
            if finding.sources:
                lines.append(f"*Sources: {', '.join(finding.sources[:3])}*")
            lines.append("")
            if i < len(findings):
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _combine_findings(findings: list["ResearchFinding"]) -> str:
        """Combine findings into a single text block for the LLM."""
        parts = []
        for f in findings:
            header = f"=== Cycle {f.cycle_num} ==="
            if f.created_at:
                header += f" ({f.created_at.strftime('%Y-%m-%d')})"
            parts.append(f"{header}\n{f.content}")
        return "\n\n".join(parts)
