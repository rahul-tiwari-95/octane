"""Unit tests for ResearchSynthesizer.

Tests are fully offline — no Redis, Postgres, or Bodega required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from octane.research.models import ResearchFinding
from octane.research.synthesizer import ResearchSynthesizer, _MAX_DIRECT_CHARS

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_finding(
    task_id: str = "task-001",
    cycle_num: int = 1,
    topic: str = "AI chips",
    content: str = "NVDA posted record revenue.",
    sources: list[str] | None = None,
    word_count: int = 4,
    created_at: datetime | None = None,
) -> ResearchFinding:
    return ResearchFinding(
        task_id=task_id,
        cycle_num=cycle_num,
        topic=topic,
        content=content,
        sources=sources or ["https://example.com"],
        word_count=word_count,
        created_at=created_at or datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


def _make_store(findings: list[ResearchFinding]) -> MagicMock:
    store = MagicMock()
    store.get_findings = AsyncMock(return_value=findings)
    return store


# ── TestPlainFallback ─────────────────────────────────────────────────────────


class TestPlainFallback:
    """ResearchSynthesizer.generate() without a Bodega client."""

    async def test_generate_returns_plain_string_no_bodega(self):
        findings = [_make_finding()]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_generate_includes_topic_in_header(self):
        findings = [_make_finding(topic="GPU Market")]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001")
        assert "GPU Market" in result

    async def test_generate_includes_all_cycle_content(self):
        findings = [
            _make_finding(cycle_num=1, content="First finding about NVDA."),
            _make_finding(cycle_num=2, content="Second finding about AMD."),
        ]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001")
        assert "First finding about NVDA." in result
        assert "Second finding about AMD." in result

    async def test_generate_empty_findings_returns_message(self):
        synth = ResearchSynthesizer(_make_store([]), bodega=None)
        result = await synth.generate("task-999")
        assert "No findings" in result
        assert "task-999" in result

    async def test_plain_format_includes_task_id(self):
        findings = [_make_finding(task_id="abc-123")]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("abc-123")
        assert "abc-123" in result

    async def test_plain_format_includes_sources(self):
        findings = [_make_finding(sources=["https://reuters.com"])]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001")
        assert "reuters.com" in result


# ── TestCycleFilter ───────────────────────────────────────────────────────────


class TestCycleFilter:
    """Filtering findings by cycles= parameter."""

    async def test_generate_filters_last_n_cycles(self):
        findings = [_make_finding(cycle_num=i, content=f"cycle {i}") for i in range(1, 6)]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001", cycles=2)
        assert "cycle 4" in result
        assert "cycle 5" in result
        # Earlier cycles should not appear
        assert "cycle 1" not in result
        assert "cycle 2" not in result

    async def test_generate_cycles_zero_shows_nothing(self):
        """cycles=0 means 'last 0' → slice returns empty → 'no findings match' message."""
        findings = [_make_finding()]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001", cycles=0)
        assert "No findings match" in result

    async def test_generate_cycles_larger_than_available_returns_all(self):
        findings = [_make_finding(cycle_num=i) for i in range(1, 4)]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate("task-001", cycles=100)
        # All 3 findings should be present
        assert result.count("Cycle") >= 3


# ── TestSinceFilter ───────────────────────────────────────────────────────────


class TestSinceFilter:
    """Filtering findings by since= parameter."""

    async def test_generate_filters_by_since(self):
        findings = [
            _make_finding(
                cycle_num=1,
                content="old finding",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            _make_finding(
                cycle_num=2,
                content="new finding",
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            ),
        ]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate(
            "task-001",
            since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        assert "new finding" in result
        assert "old finding" not in result

    async def test_generate_since_naive_datetime_treated_as_utc(self):
        """A naive since datetime should be treated as UTC without crashing."""
        findings = [
            _make_finding(
                cycle_num=1,
                content="my finding",
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        ]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        # Naive datetime — should not raise
        result = await synth.generate(
            "task-001",
            since=datetime(2026, 1, 1),  # no tzinfo
        )
        assert "my finding" in result

    async def test_generate_since_no_match_returns_filter_message(self):
        findings = [
            _make_finding(
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        ]
        synth = ResearchSynthesizer(_make_store(findings), bodega=None)
        result = await synth.generate(
            "task-001",
            since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert "No findings match" in result


# ── TestLLMSynthesis ──────────────────────────────────────────────────────────


class TestLLMSynthesis:
    """ResearchSynthesizer.generate() with a Bodega mock."""

    def _make_bodega(self, response: str = "## Synthesis\n\nResult.") -> MagicMock:
        bodega = MagicMock()
        bodega.chat_simple = AsyncMock(return_value=response)
        return bodega

    async def test_generate_calls_chat_simple_with_bodega(self):
        findings = [_make_finding()]
        bodega = self._make_bodega()
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        result = await synth.generate("task-001")
        bodega.chat_simple.assert_called_once()
        assert "Synthesis" in result

    async def test_generate_llm_result_returned_verbatim(self):
        findings = [_make_finding()]
        bodega = self._make_bodega(response="Custom LLM output here.")
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        result = await synth.generate("task-001")
        assert "Custom LLM output here." in result

    async def test_generate_llm_failure_falls_back_to_plain(self):
        findings = [_make_finding(topic="Fallback Test")]
        bodega = MagicMock()
        bodega.chat_simple = AsyncMock(side_effect=RuntimeError("LLM offline"))
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        result = await synth.generate("task-001")
        # Should fall back to plain format — topic still present
        assert "Fallback Test" in result
        assert isinstance(result, str)

    async def test_generate_strips_think_blocks(self):
        findings = [_make_finding()]
        bodega = self._make_bodega(
            response="<think>internal reasoning</think>\n## Report\n\nConclusion."
        )
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        result = await synth.generate("task-001")
        assert "<think>" not in result
        assert "internal reasoning" not in result
        assert "Conclusion." in result

    async def test_generate_passes_system_prompt(self):
        findings = [_make_finding()]
        bodega = self._make_bodega()
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        await synth.generate("task-001")
        call_kwargs = bodega.chat_simple.call_args
        assert "system" in call_kwargs.kwargs
        assert len(call_kwargs.kwargs["system"]) > 20  # non-trivial system prompt


# ── TestCombineFindings ───────────────────────────────────────────────────────


class TestCombineFindings:
    """Internal _combine_findings helper."""

    def test_combine_includes_all_cycles(self):
        findings = [
            _make_finding(cycle_num=1, content="Alpha"),
            _make_finding(cycle_num=2, content="Beta"),
        ]
        combined = ResearchSynthesizer._combine_findings(findings)
        assert "Alpha" in combined
        assert "Beta" in combined
        assert "Cycle 1" in combined
        assert "Cycle 2" in combined

    def test_combine_includes_date(self):
        findings = [
            _make_finding(
                cycle_num=1,
                created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
            )
        ]
        combined = ResearchSynthesizer._combine_findings(findings)
        assert "2026-03-15" in combined

    def test_combine_empty_returns_empty(self):
        combined = ResearchSynthesizer._combine_findings([])
        assert combined == ""


# ── TestCompression ───────────────────────────────────────────────────────────


class TestCompression:
    """Compression triggered when combined text exceeds _MAX_DIRECT_CHARS."""

    async def test_long_findings_triggers_compression_call(self):
        """With many large findings, _compress should be called."""
        # Each finding ~700 chars so total > _MAX_DIRECT_CHARS (6000)
        findings = [
            _make_finding(cycle_num=i, content="x" * 700)
            for i in range(1, 12)
        ]
        bodega = MagicMock()
        bodega.chat_simple = AsyncMock(return_value="Compressed output.")
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        await synth.generate("task-001")
        # chat_simple called at least twice: compress + synthesis
        assert bodega.chat_simple.call_count >= 2

    async def test_short_findings_no_compression(self):
        """With short findings, chat_simple called exactly once (synthesis only)."""
        findings = [_make_finding(content="Short.")]
        bodega = MagicMock()
        bodega.chat_simple = AsyncMock(return_value="Result.")
        synth = ResearchSynthesizer(_make_store(findings), bodega=bodega)
        await synth.generate("task-001")
        assert bodega.chat_simple.call_count == 1
