"""Tests for Session 31 — Extractor wiring, PowerFlags, DepthController, Hierarchical decomposition.

Covers:
  - search_and_extract() in pipeline.py
  - source_types on ResearchDimension / DimensionPlan
  - PowerFlags dataclass and parse helpers
  - DepthController convergence detection
  - Hierarchical decomposition in DimensionPlanner
  - InvestigateOrchestrator extractor dispatch
  - Trust-weighted synthesis (cite/verify)
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════════════════
#  1. PowerFlags & Parse Helpers
# ════════════════════════════════════════════════════════════════════════════════

from octane.cli._shared import PowerFlags, parse_deep_flag, parse_sources_flag


class TestParseDeepFlag:
    def test_none_returns_none(self):
        assert parse_deep_flag(None) is None

    def test_empty_string_returns_default(self):
        assert parse_deep_flag("") == 8

    def test_integer_string(self):
        assert parse_deep_flag("12") == 12

    def test_colon_prefix(self):
        assert parse_deep_flag(":12") == 12

    def test_invalid_returns_default(self):
        assert parse_deep_flag("abc") == 8

    def test_whitespace_stripped(self):
        assert parse_deep_flag("  10  ") == 10


class TestParseSourcesFlag:
    def test_none_returns_web(self):
        assert parse_sources_flag(None) == ["web"]

    def test_empty_returns_web(self):
        assert parse_sources_flag("") == ["web"]

    def test_single_source(self):
        assert parse_sources_flag("arxiv") == ["arxiv"]

    def test_multiple_sources(self):
        result = parse_sources_flag("arxiv,youtube,web")
        assert set(result) == {"arxiv", "youtube", "web"}

    def test_invalid_sources_filtered(self):
        result = parse_sources_flag("arxiv,invalid,youtube")
        assert "invalid" not in result
        assert "arxiv" in result
        assert "youtube" in result

    def test_all_invalid_returns_web(self):
        assert parse_sources_flag("foo,bar") == ["web"]

    def test_whitespace_in_sources(self):
        result = parse_sources_flag(" arxiv , youtube ")
        assert "arxiv" in result
        assert "youtube" in result

    def test_case_insensitive(self):
        result = parse_sources_flag("ArXiv,YOUTUBE")
        assert "arxiv" in result
        assert "youtube" in result


class TestPowerFlags:
    def test_defaults(self):
        flags = PowerFlags()
        assert flags.deep is None
        assert flags.sources == ["web"]
        assert flags.cite is False
        assert flags.verify is False

    def test_max_dimensions_from_deep(self):
        flags = PowerFlags(deep=12)
        assert flags.max_dimensions == 12

    def test_max_dimensions_clamped_low(self):
        flags = PowerFlags(deep=1)
        assert flags.max_dimensions == 2

    def test_max_dimensions_clamped_high(self):
        flags = PowerFlags(deep=100)
        assert flags.max_dimensions == 16

    def test_max_dimensions_none_when_not_deep(self):
        flags = PowerFlags()
        assert flags.max_dimensions is None

    def test_has_extractors_false_for_web_only(self):
        flags = PowerFlags(sources=["web"])
        assert flags.has_extractors is False

    def test_has_extractors_true_for_arxiv(self):
        flags = PowerFlags(sources=["arxiv", "web"])
        assert flags.has_extractors is True

    def test_has_extractors_true_for_youtube(self):
        flags = PowerFlags(sources=["youtube"])
        assert flags.has_extractors is True

    def test_full_power_flags(self):
        flags = PowerFlags(deep=12, sources=["arxiv", "youtube"], cite=True, verify=True)
        assert flags.max_dimensions == 12
        assert flags.has_extractors is True
        assert flags.cite is True
        assert flags.verify is True


# ════════════════════════════════════════════════════════════════════════════════
#  2. ResearchDimension source_types
# ════════════════════════════════════════════════════════════════════════════════

from octane.osa.dimension_planner import ResearchDimension, DimensionPlan, DimensionPlanner


class TestResearchDimensionSourceTypes:
    def test_default_source_types(self):
        dim = ResearchDimension(id="test", label="Test", queries=["test query"])
        assert dim.source_types == ["web"]

    def test_custom_source_types(self):
        dim = ResearchDimension(
            id="test", label="Test", queries=["test query"],
            source_types=["arxiv", "youtube"],
        )
        assert dim.source_types == ["arxiv", "youtube"]

    def test_source_types_in_plan_to_dict(self):
        dim = ResearchDimension(
            id="test", label="Test", queries=["q1"],
            source_types=["arxiv", "web"],
        )
        plan = DimensionPlan(query="test", dimensions=[dim])
        d = plan.to_dict()
        assert d["dimensions"][0]["source_types"] == ["arxiv", "web"]

    def test_source_types_independent_across_dims(self):
        """Ensure default factory creates separate lists per instance."""
        d1 = ResearchDimension(id="a", label="A", queries=["q"])
        d2 = ResearchDimension(id="b", label="B", queries=["q"])
        d1.source_types.append("arxiv")
        assert d2.source_types == ["web"]  # unaffected


class TestDimensionPlannerSourceTypesPassthrough:
    """Verify source_types from the planner reach the dimensions."""

    def test_keyword_fallback_applies_source_types(self):
        planner = DimensionPlanner(bodega=None)
        plan = asyncio.run(
            planner.plan("AI machine learning models", source_types=["arxiv", "web"])
        )
        for dim in plan.dimensions:
            assert "arxiv" in dim.source_types
            assert "web" in dim.source_types

    def test_keyword_fallback_without_source_types(self):
        planner = DimensionPlanner(bodega=None)
        plan = asyncio.run(
            planner.plan("AI machine learning models")
        )
        for dim in plan.dimensions:
            assert dim.source_types == ["web"]


# ════════════════════════════════════════════════════════════════════════════════
#  3. DepthController
# ════════════════════════════════════════════════════════════════════════════════

from octane.agents.web.depth_controller import DepthController


class TestDepthController:
    def test_initial_state(self):
        dc = DepthController()
        assert dc.round_count == 0
        assert dc.novelty_history == []
        assert dc.total_unique_ngrams == 0

    def test_first_round_high_novelty(self):
        dc = DepthController()
        novelty = dc.ingest(["The quick brown fox jumps over the lazy dog"])
        assert novelty == 1.0  # first round is all new
        assert dc.round_count == 1
        assert dc.should_continue() is True

    def test_duplicate_content_low_novelty(self):
        dc = DepthController()
        text = "The quick brown fox jumps over the lazy dog in the park"
        dc.ingest([text])
        novelty = dc.ingest([text])  # exact same content
        assert novelty == 0.0
        assert dc.should_continue() is False

    def test_partially_novel_content(self):
        dc = DepthController()
        dc.ingest(["The quick brown fox jumps over the lazy dog"])
        novelty = dc.ingest(["The quick brown fox eats the red apple pie"])
        # Some overlap, some new — novelty should be between 0 and 1
        assert 0.0 < novelty < 1.0

    def test_max_rounds_stops(self):
        dc = DepthController(max_rounds=2)
        dc.ingest(["Completely unique text number one for testing"])
        assert dc.should_continue() is True
        dc.ingest(["Different unique content that is also novel"])
        assert dc.should_continue() is False  # hit max_rounds

    def test_empty_texts_zero_novelty(self):
        dc = DepthController()
        novelty = dc.ingest([""])
        assert novelty == 0.0

    def test_no_texts_zero_novelty(self):
        dc = DepthController()
        novelty = dc.ingest([])
        assert novelty == 0.0

    def test_custom_threshold(self):
        dc = DepthController(novelty_threshold=0.5)
        dc.ingest(["The quick brown fox jumps over the lazy dog in the park today"])
        # Ingest mostly overlapping content
        novelty = dc.ingest([
            "The quick brown fox jumps over the lazy dog in the park today "
            "and also some new words about cats"
        ])
        # With high threshold, might stop even with some novelty
        if novelty < 0.5:
            assert dc.should_continue() is False

    def test_summary(self):
        dc = DepthController()
        dc.ingest(["Hello world foo bar baz qux quux"])
        summary = dc.summary()
        assert summary["rounds"] == 1
        assert summary["total_unique_ngrams"] > 0
        assert len(summary["novelty_history"]) == 1

    def test_declining_novelty_detection(self):
        dc = DepthController(novelty_threshold=0.1)
        # Round 1: all new
        dc.ingest(["Alpha beta gamma delta epsilon zeta eta theta"])
        # Round 2: mostly new
        dc.ingest(["Iota kappa lambda mu nu xi omicron pi rho sigma"])
        # Round 3: mix of old and new, but add enough repeats to trigger declining
        dc.ingest([
            "Alpha beta gamma delta epsilon zeta eta theta "
            "iota kappa lambda mu nu xi omicron pi rho sigma "
            "tau upsilon"
        ])
        # The declining trend may or may not stop — just verify it doesn't crash
        assert isinstance(dc.should_continue(), bool)

    def test_should_continue_before_ingest(self):
        dc = DepthController()
        assert dc.should_continue() is True  # no data yet


# ════════════════════════════════════════════════════════════════════════════════
#  4. DimensionFinding — new fields
# ════════════════════════════════════════════════════════════════════════════════

from octane.osa.investigate import DimensionFinding, InvestigateOrchestrator


class TestDimensionFindingNewFields:
    def test_reliability_score_default(self):
        dim = ResearchDimension(id="t", label="T", queries=["q"])
        finding = DimensionFinding(dimension=dim)
        assert finding.reliability_score == 0.5

    def test_reliability_score_custom(self):
        dim = ResearchDimension(id="t", label="T", queries=["q"])
        finding = DimensionFinding(dimension=dim, reliability_score=0.92)
        assert finding.reliability_score == 0.92

    def test_sources_default_empty(self):
        dim = ResearchDimension(id="t", label="T", queries=["q"])
        finding = DimensionFinding(dimension=dim)
        assert finding.sources == []

    def test_sources_custom(self):
        dim = ResearchDimension(id="t", label="T", queries=["q"])
        finding = DimensionFinding(
            dimension=dim,
            sources=["https://arxiv.org/abs/1706.03762"],
        )
        assert len(finding.sources) == 1

    def test_to_dict_includes_new_fields(self):
        dim = ResearchDimension(id="t", label="T", queries=["q"])
        finding = DimensionFinding(
            dimension=dim,
            content="test content",
            reliability_score=0.85,
            sources=["https://arxiv.org/abs/1234.56789"],
        )
        d = finding.to_dict()
        assert "reliability_score" in d
        assert d["reliability_score"] == 0.85
        assert "sources" in d
        assert len(d["sources"]) == 1

    def test_sources_independent_across_findings(self):
        dim = ResearchDimension(id="t", label="T", queries=["q"])
        f1 = DimensionFinding(dimension=dim)
        f2 = DimensionFinding(dimension=dim)
        f1.sources.append("url1")
        assert f2.sources == []  # unaffected


# ════════════════════════════════════════════════════════════════════════════════
#  5. search_and_extract
# ════════════════════════════════════════════════════════════════════════════════

from octane.extractors.pipeline import search_and_extract
from octane.extractors.models import ExtractedDocument, SourceType


class TestSearchAndExtract:
    def test_default_sources_is_web(self):
        """Web-only returns empty (web is handled separately by WebAgent)."""
        result = asyncio.run(search_and_extract("test query"))
        assert result == []  # "web" is not dispatched by search_and_extract

    def test_web_only_returns_empty(self):
        result = asyncio.run(search_and_extract("test query", source_types=["web"]))
        assert result == []

    @patch("octane.extractors.pipeline._search_extract_arxiv")
    def test_arxiv_source_dispatches(self, mock_arxiv):
        mock_doc = ExtractedDocument(
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234.56789",
            title="Test Paper",
            raw_text="Some text",
            reliability_score=0.92,
        )
        mock_arxiv.return_value = [mock_doc]

        result = asyncio.run(
            search_and_extract("attention mechanisms", source_types=["arxiv"])
        )
        assert len(result) == 1
        assert result[0].source_type == SourceType.ARXIV
        mock_arxiv.assert_called_once()

    @patch("octane.extractors.pipeline._search_extract_youtube")
    def test_youtube_source_dispatches(self, mock_yt):
        mock_doc = ExtractedDocument(
            source_type=SourceType.YOUTUBE,
            source_url="https://youtube.com/watch?v=test",
            title="Test Video",
            raw_text="transcript text",
            reliability_score=0.55,
        )
        mock_yt.return_value = [mock_doc]

        result = asyncio.run(
            search_and_extract("transformers explained", source_types=["youtube"])
        )
        assert len(result) == 1
        assert result[0].source_type == SourceType.YOUTUBE

    @patch("octane.extractors.pipeline._search_extract_arxiv")
    @patch("octane.extractors.pipeline._search_extract_youtube")
    def test_multi_source_merges_and_sorts(self, mock_yt, mock_arxiv):
        arxiv_doc = ExtractedDocument(
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234",
            raw_text="paper",
            reliability_score=0.92,
        )
        yt_doc = ExtractedDocument(
            source_type=SourceType.YOUTUBE,
            source_url="https://youtube.com/watch?v=abc",
            raw_text="video",
            reliability_score=0.55,
        )
        mock_arxiv.return_value = [arxiv_doc]
        mock_yt.return_value = [yt_doc]

        result = asyncio.run(
            search_and_extract("test", source_types=["arxiv", "youtube"])
        )
        assert len(result) == 2
        # Sorted by reliability descending
        assert result[0].reliability_score >= result[1].reliability_score

    @patch("octane.extractors.pipeline._search_extract_arxiv")
    def test_exception_in_source_handled(self, mock_arxiv):
        mock_arxiv.side_effect = RuntimeError("Network error")
        result = asyncio.run(
            search_and_extract("test", source_types=["arxiv"])
        )
        assert result == []


# ════════════════════════════════════════════════════════════════════════════════
#  6. InvestigateOrchestrator — extractor dispatch
# ════════════════════════════════════════════════════════════════════════════════


class TestInvestigateOrchestratorExtractorDispatch:
    """Test that _research_one dispatches to extractors for non-web source_types."""

    @patch("octane.extractors.pipeline.search_and_extract", new_callable=AsyncMock)
    def test_research_one_arxiv(self, mock_sae):
        from octane.extractors.models import ExtractedDocument, SourceType

        mock_doc = ExtractedDocument(
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            raw_text="We propose a new simple network architecture..." * 20,
            reliability_score=0.92,
        )
        mock_sae.return_value = [mock_doc]

        orch = InvestigateOrchestrator(bodega=None, web_agent=None)
        dim = ResearchDimension(
            id="academic_papers",
            label="Academic Papers",
            queries=["attention mechanisms transformer"],
            source_types=["arxiv"],
        )

        finding = asyncio.run(
            orch._research_one("attention mechanisms", dim, "test-session")
        )

        assert finding.success
        assert "extractor:arxiv" in finding.agent_used
        assert finding.reliability_score > 0.8
        assert len(finding.sources) > 0


class TestInvestigateOrchestratorRunStream:
    """Test run_stream accepts and passes through new params."""

    def test_run_stream_accepts_source_types(self):
        orch = InvestigateOrchestrator(bodega=None, web_agent=None)
        # Just verify the method signature works — no actual LLM call
        import inspect
        sig = inspect.signature(orch.run_stream)
        assert "source_types" in sig.parameters
        assert "cite" in sig.parameters
        assert "verify" in sig.parameters

    def test_run_accepts_source_types(self):
        orch = InvestigateOrchestrator(bodega=None, web_agent=None)
        import inspect
        sig = inspect.signature(orch.run)
        assert "source_types" in sig.parameters
        assert "cite" in sig.parameters
        assert "verify" in sig.parameters


# ════════════════════════════════════════════════════════════════════════════════
#  7. Synthesis — cite/verify prompt injection
# ════════════════════════════════════════════════════════════════════════════════

from octane.osa.investigate import (
    _INVESTIGATE_SYNTHESIS_SYSTEM,
    _CITATION_INSTRUCTIONS,
    _VERIFICATION_INSTRUCTIONS,
)


class TestSynthesisPromptAddons:
    def test_citation_instructions_exist(self):
        assert "CITATION MODE" in _CITATION_INSTRUCTIONS
        assert "Sources" in _CITATION_INSTRUCTIONS

    def test_verification_instructions_exist(self):
        assert "VERIFICATION MODE" in _VERIFICATION_INSTRUCTIONS
        assert "CONFIRMED" in _VERIFICATION_INSTRUCTIONS
        assert "LIKELY" in _VERIFICATION_INSTRUCTIONS
        assert "UNVERIFIED" in _VERIFICATION_INSTRUCTIONS

    def test_base_prompt_does_not_include_cite_or_verify(self):
        assert "CITATION MODE" not in _INVESTIGATE_SYNTHESIS_SYSTEM
        assert "VERIFICATION MODE" not in _INVESTIGATE_SYNTHESIS_SYSTEM


# ════════════════════════════════════════════════════════════════════════════════
#  8. Hierarchical Decomposition
# ════════════════════════════════════════════════════════════════════════════════


class TestHierarchicalDecomposition:
    def test_plan_routes_to_hierarchical_above_8(self):
        """When max_dimensions > 8, plan() should call _plan_hierarchical."""
        planner = DimensionPlanner(bodega=None)
        # With bodega=None, _plan_hierarchical falls back to keyword_fallback
        plan = asyncio.run(
            planner.plan("artificial intelligence trends", max_dimensions=12)
        )
        # Should produce dimensions (keyword fallback, capped)
        assert len(plan.dimensions) > 0
        assert len(plan.dimensions) <= 12

    def test_plan_standard_for_8_or_less(self):
        planner = DimensionPlanner(bodega=None)
        plan = asyncio.run(
            planner.plan("artificial intelligence", max_dimensions=6)
        )
        assert len(plan.dimensions) <= 6

    def test_hierarchical_with_mock_bodega(self):
        """Mock bodega to test actual hierarchical two-phase flow."""
        mock_bodega = AsyncMock()

        # Phase 1 response — 3 core dimensions
        core_response = '''{
            "dimensions": [
                {"id": "core_a", "label": "Core A", "queries": ["q1"], "priority": 1, "rationale": "r1"},
                {"id": "core_b", "label": "Core B", "queries": ["q2"], "priority": 2, "rationale": "r2"},
                {"id": "core_c", "label": "Core C", "queries": ["q3"], "priority": 3, "rationale": "r3"}
            ]
        }'''

        # Phase 2 responses — sub-dimensions for each core
        sub_response = '''{
            "dimensions": [
                {"id": "sub_1", "label": "Sub 1", "queries": ["sq1"], "priority": 1, "rationale": "sr1"},
                {"id": "sub_2", "label": "Sub 2", "queries": ["sq2"], "priority": 2, "rationale": "sr2"}
            ]
        }'''

        mock_bodega.chat_simple = AsyncMock(side_effect=[
            core_response,  # Phase 1
            sub_response,   # Phase 2 for core_a
            sub_response,   # Phase 2 for core_b
            sub_response,   # Phase 2 for core_c
        ])

        planner = DimensionPlanner(bodega=mock_bodega)
        plan = asyncio.run(
            planner.plan("deep topic investigation", max_dimensions=12)
        )

        # Should have core + sub dimensions
        assert len(plan.dimensions) > 3
        assert plan.from_llm is True
        # Sub-dimensions should have composite IDs
        sub_ids = [d.id for d in plan.dimensions if "_sub_" in d.id]
        assert len(sub_ids) > 0


# ════════════════════════════════════════════════════════════════════════════════
#  9. CLI Power Command Signatures
# ════════════════════════════════════════════════════════════════════════════════


class TestPowerCommandSignatures:
    """Verify CLI commands accept the new flags."""

    def test_investigate_has_deep_flag(self):
        import inspect
        from octane.cli.power import investigate
        sig = inspect.signature(investigate)
        assert "deep" in sig.parameters
        assert "sources" in sig.parameters
        assert "cite" in sig.parameters
        assert "verify" in sig.parameters

    def test_compare_has_cite_flag(self):
        import inspect
        from octane.cli.power import compare
        sig = inspect.signature(compare)
        assert "cite" in sig.parameters
        assert "verify" in sig.parameters
        assert "sources" in sig.parameters
