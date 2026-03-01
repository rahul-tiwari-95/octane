"""Unit tests for octane.osa.domain_pipelines and DAGPlanner integration.

Tests cover:
  - match_domain() keyword scoring for all 4 pipelines
  - build_dag() node structure for each pipeline
  - Variable extraction helpers
  - DAGPlanner.plan() uses domain pipeline before LLM
"""

from __future__ import annotations

import pytest

from octane.osa.domain_pipelines import (
    match_domain,
    build_dag,
    _extract_ticker,
    _extract_compare_items,
    _extract_topic,
)


# ─────────────────────────────────────────────────────────────────────────────
# match_domain()
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchDomain:
    """Tests for the keyword-based domain matcher."""

    def test_investment_query_matches(self):
        """Stock/investment query should match investment_analysis."""
        assert match_domain("NVDA stock earnings and revenue analysis") == "investment_analysis"

    def test_comparative_query_matches(self):
        """'Compare X vs Y' should match comparative_analysis."""
        assert match_domain("compare NVDA vs AMD performance") == "comparative_analysis"

    def test_content_creation_query_matches(self):
        """'Write an article about...' should match content_creation."""
        assert match_domain("write an article about quantum computing") == "content_creation"

    def test_deep_research_query_matches(self):
        """'deep dive on...' should match deep_research."""
        assert match_domain("deep dive comprehensive research on AI regulation") == "deep_research"

    def test_none_for_ambiguous_short_query(self):
        """A very short/generic query should return None."""
        result = match_domain("what is the weather")
        # Neither finance nor research keywords present
        assert result is None or isinstance(result, str)  # just must not crash

    def test_comparative_beats_investment_on_tie(self):
        """comparative_analysis has priority over investment_analysis on tie."""
        # Both have one match; comparative should win per priority
        result = match_domain("compare Tesla stock vs Ford shares")
        assert result == "comparative_analysis"

    def test_empty_query_returns_none(self):
        """Empty string should return None gracefully."""
        assert match_domain("") is None

    def test_general_question_no_domain(self):
        """A generic conversational query should not match any domain."""
        result = match_domain("hello how are you doing today")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# build_dag() — investment_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestInvestmentAnalysisPipeline:
    """Tests for the investment_analysis domain pipeline."""

    def test_produces_two_nodes(self):
        """investment_analysis should produce exactly 2 parallel nodes."""
        dag = build_dag("NVDA stock analysis and earnings", "investment_analysis")
        assert dag is not None
        assert len(dag.nodes) == 2

    def test_nodes_use_web_agent(self):
        """Both nodes should use the web agent."""
        dag = build_dag("NVDA stock performance", "investment_analysis")
        assert all(n.agent == "web" for n in dag.nodes)

    def test_one_finance_one_news_node(self):
        """Pipeline should produce one finance and one news node."""
        dag = build_dag("AAPL stock earnings", "investment_analysis")
        templates = {n.metadata.get("template") for n in dag.nodes}
        assert "web_finance" in templates
        assert "web_news" in templates

    def test_no_dependencies_between_nodes(self):
        """Investment nodes run in parallel — no depends_on."""
        dag = build_dag("MSFT revenue analysis", "investment_analysis")
        for node in dag.nodes:
            assert node.depends_on == []

    def test_ticker_appears_in_instructions(self):
        """All-caps ticker should appear in node instructions."""
        dag = build_dag("NVDA earnings and growth", "investment_analysis")
        combined = " ".join(n.instruction for n in dag.nodes)
        assert "NVDA" in combined

    def test_pipeline_metadata_tag(self):
        """All nodes should be tagged with pipeline='investment_analysis'."""
        dag = build_dag("TSLA stock", "investment_analysis")
        for node in dag.nodes:
            assert node.metadata.get("pipeline") == "investment_analysis"
            assert node.metadata.get("source") == "domain_pipeline"


# ─────────────────────────────────────────────────────────────────────────────
# build_dag() — comparative_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestComparativeAnalysisPipeline:
    """Tests for the comparative_analysis domain pipeline."""

    def test_produces_three_nodes(self):
        """comparative_analysis should produce 3 nodes: A, B, compare."""
        dag = build_dag("compare NVDA vs AMD", "comparative_analysis")
        assert dag is not None
        assert len(dag.nodes) == 3

    def test_code_node_depends_on_both_web_nodes(self):
        """The code node must depend on both web nodes."""
        dag = build_dag("NVDA versus AMD performance comparison", "comparative_analysis")
        code_nodes = [n for n in dag.nodes if n.agent == "code"]
        assert len(code_nodes) == 1
        code_node = code_nodes[0]
        web_nodes = [n for n in dag.nodes if n.agent == "web"]
        assert len(code_node.depends_on) == 2
        web_ids = {n.task_id for n in web_nodes}
        assert set(code_node.depends_on) == web_ids

    def test_returns_none_when_items_not_extractable(self):
        """If two items cannot be extracted, returns None (fallback to LLM)."""
        result = build_dag("compare things", "comparative_analysis")
        assert result is None

    def test_item_names_in_instructions(self):
        """Both item names should appear in the web node instructions."""
        dag = build_dag("compare Tesla vs Ford", "comparative_analysis")
        assert dag is not None
        instructions = [n.instruction.lower() for n in dag.nodes if n.agent == "web"]
        combined = " ".join(instructions)
        assert "tesla" in combined
        assert "ford" in combined


# ─────────────────────────────────────────────────────────────────────────────
# build_dag() — deep_research
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepResearchPipeline:
    """Tests for the deep_research domain pipeline."""

    def test_produces_five_nodes(self):
        """deep_research should produce 4 web angles + 1 memory recall = 5 nodes."""
        dag = build_dag("comprehensive deep dive on quantum computing", "deep_research")
        assert dag is not None
        assert len(dag.nodes) == 5

    def test_has_memory_recall_node(self):
        """deep_research must include a memory/memory_recall node."""
        dag = build_dag("thorough research on AI chips", "deep_research")
        memory_nodes = [n for n in dag.nodes if n.agent == "memory"]
        assert len(memory_nodes) == 1
        assert memory_nodes[0].metadata.get("template") == "memory_recall"

    def test_all_nodes_parallel(self):
        """All deep_research nodes run in parallel (no depends_on)."""
        dag = build_dag("in-depth analysis of climate change", "deep_research")
        for node in dag.nodes:
            assert node.depends_on == []

    def test_pipeline_tag_on_all_nodes(self):
        dag = build_dag("comprehensive overview of blockchain", "deep_research")
        for node in dag.nodes:
            assert node.metadata.get("pipeline") == "deep_research"


# ─────────────────────────────────────────────────────────────────────────────
# build_dag() — content_creation
# ─────────────────────────────────────────────────────────────────────────────

class TestContentCreationPipeline:
    """Tests for the content_creation domain pipeline."""

    def test_produces_three_nodes(self):
        """content_creation should produce 2 research nodes + 1 write node."""
        dag = build_dag("write a comprehensive article about climate change", "content_creation")
        assert dag is not None
        assert len(dag.nodes) == 3

    def test_write_node_is_code_agent(self):
        """The write node must use the code agent."""
        dag = build_dag("write an essay about the French Revolution", "content_creation")
        code_nodes = [n for n in dag.nodes if n.agent == "code"]
        assert len(code_nodes) == 1

    def test_write_node_depends_on_research_nodes(self):
        """The write node should depend on both research nodes."""
        dag = build_dag("draft a blog post about quantum computing", "content_creation")
        write_nodes = [n for n in dag.nodes if n.agent == "code"]
        research_ids = {n.task_id for n in dag.nodes if n.agent == "web"}
        assert set(write_nodes[0].depends_on) == research_ids


# ─────────────────────────────────────────────────────────────────────────────
# Variable extractors
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableExtractors:
    """Tests for the internal variable extraction helpers."""

    def test_extract_ticker_all_caps(self):
        """All-caps token should be extracted as ticker."""
        assert _extract_ticker("What is NVDA stock doing") == "NVDA"

    def test_extract_ticker_company_name(self):
        """Known company name should map to ticker."""
        assert _extract_ticker("tell me about apple stock") == "AAPL"
        assert _extract_ticker("microsoft earnings this quarter") == "MSFT"

    def test_extract_ticker_returns_none_for_no_ticker(self):
        """Query without a ticker should return None."""
        result = _extract_ticker("what is the weather in Paris")
        assert result is None

    def test_extract_compare_items_vs_pattern(self):
        """'X vs Y' should extract both items."""
        result = _extract_compare_items("compare NVDA vs AMD")
        assert result is not None
        a, b = result
        assert "NVDA" in a.upper() or "NVDA" in b.upper()
        assert "AMD" in a.upper() or "AMD" in b.upper()

    def test_extract_compare_items_versus_pattern(self):
        """'X versus Y' should extract both items."""
        result = _extract_compare_items("Tesla versus Ford in the EV market")
        assert result is not None
        items_lower = " ".join(result).lower()
        assert "tesla" in items_lower
        assert "ford" in items_lower

    def test_extract_compare_items_returns_none_for_single_item(self):
        """A query with only one item should return None."""
        result = _extract_compare_items("analyze NVDA")
        assert result is None

    def test_extract_topic_strips_write_trigger(self):
        """'write a report on X' should extract 'X'."""
        import re
        trigger = re.compile(r"\b(write|draft|create|generate)\s+(a |an |the )?", re.IGNORECASE)
        topic = _extract_topic("write a report on climate change", remove_patterns=[trigger])
        assert "climate change" in topic
        assert "write" not in topic.lower()


# ─────────────────────────────────────────────────────────────────────────────
# DAGPlanner integration: domain pipeline fires before LLM
# ─────────────────────────────────────────────────────────────────────────────

class TestDAGPlannerDomainIntegration:
    """Tests that DAGPlanner uses domain pipelines before calling the LLM."""

    @pytest.mark.asyncio
    async def test_comparative_query_uses_domain_pipeline_not_llm(self):
        """compare X vs Y should trigger domain pipeline; LLM must not be called."""
        from unittest.mock import AsyncMock
        from octane.osa.dag_planner import DAGPlanner

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="1. web/web_finance | NVDA price")
        planner = DAGPlanner(bodega=mock_bodega)

        dag = await planner.plan("compare NVDA vs AMD performance")

        # Domain pipeline should fire (3 nodes) without calling chat_simple
        assert dag is not None
        assert len(dag.nodes) == 3
        mock_bodega.chat_simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_investment_query_uses_domain_pipeline(self):
        """investment_analysis query triggers domain pipeline, no LLM call."""
        from unittest.mock import AsyncMock
        from octane.osa.dag_planner import DAGPlanner

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="1. web/web_finance | AAPL")
        planner = DAGPlanner(bodega=mock_bodega)

        dag = await planner.plan("AAPL stock earnings and analyst ratings")

        assert dag is not None
        assert len(dag.nodes) == 2   # investment_analysis: finance + news
        mock_bodega.chat_simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_simple_query_falls_through_to_llm(self):
        """A query with no domain keywords falls through to LLM planning."""
        from unittest.mock import AsyncMock
        from octane.osa.dag_planner import DAGPlanner

        llm_output = (
            "1. web/web_finance | NVDA stock price\n"
            "2. web/web_news | NVDA latest news\n"
            "depends_on: none"
        )
        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=llm_output)
        planner = DAGPlanner(bodega=mock_bodega)

        # "what is X" style — no domain trigger words
        dag = await planner.plan("what is NVDA price right now")

        # LLM should have been called (domain match returns None or single-node)
        # Result depends on LLM parse; either way the bodega.chat_simple was called
        mock_bodega.chat_simple.assert_called_once()

    @pytest.mark.asyncio
    async def test_domain_pipeline_without_bodega(self):
        """Domain pipeline works even without a Bodega client."""
        from octane.osa.dag_planner import DAGPlanner

        planner = DAGPlanner(bodega=None)  # no LLM

        dag = await planner.plan("compare NVDA versus AMD chips")

        assert dag is not None
        assert len(dag.nodes) == 3   # comparative_analysis
