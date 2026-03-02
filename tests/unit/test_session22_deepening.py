"""Tests for Session 22 features.

Feature-1: QueryStrategist timeout/max_tokens fix (llm_strategize_failed)
Feature-2: DepthAnalyzer — iterative follow-up query generation
Feature-3: WebAgent iterative deepening (_fetch_search deep=True, _fetch_news deep=True)
Feature-4: Orchestrator.run_stream extra_metadata propagation
Feature-5: --deep flag wiring (octane ask --deep → extra_metadata → WebAgent)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Feature-1: QueryStrategist timeout / max_tokens
# ---------------------------------------------------------------------------


class TestQueryStrategistTimeoutFix:
    """QueryStrategist must use max_tokens=2048 and timeout=45.0."""

    def test_max_tokens_at_least_2048(self):
        """_llm_strategize must allow enough tokens for <think> + JSON output."""
        import inspect
        from octane.agents.web.query_strategist import QueryStrategist

        src = inspect.getsource(QueryStrategist._llm_strategize)
        # max_tokens must be 2048 or higher
        assert "max_tokens=2048" in src or "max_tokens=4096" in src, (
            "QueryStrategist._llm_strategize must set max_tokens >= 2048 "
            "to accommodate reasoning model <think> blocks"
        )

    def test_timeout_at_least_30s(self):
        """Timeout must be >= 30 s — 90M reasoning model needs time."""
        import inspect
        from octane.agents.web.query_strategist import QueryStrategist

        src = inspect.getsource(QueryStrategist._llm_strategize)
        # Extract the timeout= argument value
        import re
        match = re.search(r"timeout=(\d+\.?\d*)", src)
        assert match, "timeout= argument not found in _llm_strategize"
        timeout_val = float(match.group(1))
        assert timeout_val >= 30.0, (
            f"QueryStrategist timeout is {timeout_val}s — must be >= 30s "
            "for reasoning models with <think> blocks"
        )

    @pytest.mark.asyncio
    async def test_think_block_no_close_tag_raises_value_error(self):
        """If <think> never closes (max_tokens hit), ValueError is caught as fallback."""
        from octane.agents.web.query_strategist import QueryStrategist

        mock_bodega = AsyncMock()
        # Simulate truncated response — <think> started but never closed
        mock_bodega.chat_simple = AsyncMock(
            return_value="<think>\nOkay let me analyse this query carefully...\n"
        )
        qs = QueryStrategist(bodega=mock_bodega)
        # Should fall back to keyword strategy, not raise
        strategies = await qs.strategize("israel iran latest news")
        assert len(strategies) == 1
        assert strategies[0]["rationale"] == "keyword fallback"

    @pytest.mark.asyncio
    async def test_think_block_with_close_tag_parses_json(self):
        """<think>...</think> followed by valid JSON must be parsed correctly."""
        from octane.agents.web.query_strategist import QueryStrategist

        valid_json = json.dumps([
            {"query": "israel iran war update", "api": "news", "rationale": "recent events"},
            {"query": "iran retaliation after strike", "api": "news", "rationale": "aftermath"},
        ])
        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(
            return_value=f"<think>\nLet me think...\n</think>\n{valid_json}"
        )
        qs = QueryStrategist(bodega=mock_bodega)
        strategies = await qs.strategize("israel iran latest news")
        assert len(strategies) == 2
        assert strategies[0]["api"] == "news"
        assert "israel" in strategies[0]["query"].lower()


# ---------------------------------------------------------------------------
# Feature-2: DepthAnalyzer
# ---------------------------------------------------------------------------


class TestDepthAnalyzer:
    """DepthAnalyzer generates targeted follow-up queries from initial findings."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_bodega(self):
        """DepthAnalyzer(bodega=None) always returns empty list."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        da = DepthAnalyzer(bodega=None)
        result = await da.generate_followups("test query", ["some finding"])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_findings(self):
        """Empty findings → skip LLM call → return []."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        mock_bodega = AsyncMock()
        da = DepthAnalyzer(bodega=mock_bodega)
        result = await da.generate_followups("test query", [])
        # Should not call the LLM at all
        mock_bodega.chat_simple.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_valid_json_followups(self):
        """LLM returning valid JSON array → parsed into strategy dicts."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        followup_json = json.dumps([
            {"query": "Khamenei death confirmed reports May 2025", "api": "news", "rationale": "leader fate"},
            {"query": "Israel military operations Iran aftermath", "api": "news", "rationale": "military ops"},
            {"query": "USA involvement Israel Iran war", "api": "search", "rationale": "US role"},
        ])
        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=followup_json)
        da = DepthAnalyzer(bodega=mock_bodega)
        result = await da.generate_followups(
            "israel iran latest news",
            ["US-Israel struck Iran, Khamenei fate unclear"],
        )
        assert len(result) == 3
        assert result[0]["api"] == "news"
        assert "khamenei" in result[0]["query"].lower()

    @pytest.mark.asyncio
    async def test_respects_max_followups_cap(self):
        """max_followups=2 → at most 2 strategies returned even if LLM gives more."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        followup_json = json.dumps([
            {"query": "q1", "api": "news", "rationale": "r1"},
            {"query": "q2", "api": "search", "rationale": "r2"},
            {"query": "q3", "api": "search", "rationale": "r3"},
            {"query": "q4", "api": "news", "rationale": "r4"},
        ])
        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=followup_json)
        da = DepthAnalyzer(bodega=mock_bodega)
        result = await da.generate_followups("query", ["finding"], max_followups=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """LLM exception → caught → return empty list (never raise)."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("network error"))
        da = DepthAnalyzer(bodega=mock_bodega)
        result = await da.generate_followups("query", ["finding"])
        assert result == []

    @pytest.mark.asyncio
    async def test_strips_think_block(self):
        """<think>...</think> before JSON is stripped before parsing."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        inner_json = json.dumps([
            {"query": "Iran leadership vacuum", "api": "news", "rationale": "succession"},
        ])
        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(
            return_value=f"<think>\nAnalysing the situation...\n</think>\n{inner_json}"
        )
        da = DepthAnalyzer(bodega=mock_bodega)
        result = await da.generate_followups("israel iran news", ["findings here"])
        assert len(result) == 1
        assert "iran" in result[0]["query"].lower()

    @pytest.mark.asyncio
    async def test_invalid_api_normalised_to_search(self):
        """Unknown 'api' value in LLM response → normalised to 'search'."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        followup_json = json.dumps([
            {"query": "some query", "api": "unknown_api", "rationale": "test"},
        ])
        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=followup_json)
        da = DepthAnalyzer(bodega=mock_bodega)
        result = await da.generate_followups("test", ["finding"])
        assert result[0]["api"] == "search"

    def test_uses_fast_tier(self):
        """DepthAnalyzer must use ModelTier.FAST to keep latency low."""
        import inspect
        from octane.agents.web.depth_analyzer import DepthAnalyzer
        from octane.tools.topology import ModelTier

        src = inspect.getsource(DepthAnalyzer._llm_followups)
        assert "ModelTier.FAST" in src, (
            "DepthAnalyzer must route to ModelTier.FAST (small model) for follow-up generation"
        )


# ---------------------------------------------------------------------------
# Feature-3: WebAgent iterative deepening
# ---------------------------------------------------------------------------


class TestWebAgentDeepMode:
    """WebAgent._fetch_search and _fetch_news run DepthAnalyzer in deep mode."""

    def _make_agent(self, *, depth_followups=None):
        """Build a WebAgent with fully mocked dependencies."""
        from octane.agents.web.agent import WebAgent
        from octane.models.synapse import SynapseEventBus

        synapse = SynapseEventBus()
        mock_bodega = AsyncMock()
        mock_intel = AsyncMock()
        mock_extractor = AsyncMock()
        mock_browser = AsyncMock()

        # Extractor returns one good result
        from octane.agents.web.content_extractor import ExtractedContent
        mock_extractor.extract_batch = AsyncMock(return_value=[
            ExtractedContent(url="https://example.com/1", text="Article text about Israel Iran war.", word_count=50, method="trafilatura"),
        ])
        mock_browser.scrape = AsyncMock(return_value=None)

        # Strategist: one strategy
        agent = WebAgent(
            synapse=synapse,
            intel=mock_intel,
            bodega=mock_bodega,
            extractor=mock_extractor,
            browser=mock_browser,
        )
        # Patch strategist to return one strategy
        agent._strategist = AsyncMock()
        agent._strategist.strategize = AsyncMock(return_value=[
            {"query": "israel iran news", "api": "news", "rationale": "test"},
        ])
        # Patch synthesizer
        agent._synthesizer = AsyncMock()
        agent._synthesizer.synthesize_with_content = AsyncMock(return_value="Synthesized output")
        agent._synthesizer.synthesize_search = AsyncMock(return_value="Snippet output")
        agent._synthesizer.synthesize_news = AsyncMock(return_value="News output")

        # Patch depth analyzer
        agent._depth_analyzer = AsyncMock()
        if depth_followups is not None:
            agent._depth_analyzer.generate_followups = AsyncMock(return_value=depth_followups)
        else:
            agent._depth_analyzer.generate_followups = AsyncMock(return_value=[])

        return agent, mock_intel

    @pytest.mark.asyncio
    async def test_fetch_search_deep_calls_depth_analyzer(self):
        """_fetch_search with deep=True calls DepthAnalyzer.generate_followups."""
        from octane.models.schemas import AgentRequest

        followups = [
            {"query": "Iran leadership vacuum", "api": "search", "rationale": "succession"},
        ]
        agent, mock_intel = self._make_agent(depth_followups=followups)

        # Round 1 returns some results
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [{"url": "https://example.com/1", "title": "Israel Iran War"}]}
        })

        request = AgentRequest(
            query="israel iran news", source="test",
            metadata={"sub_agent": "search", "deep": True},
        )
        response = await agent._fetch_search("israel iran news", request, deep=True)
        assert response.success is True
        agent._depth_analyzer.generate_followups.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_search_no_deep_skips_depth_analyzer(self):
        """_fetch_search with deep=False does NOT call DepthAnalyzer."""
        from octane.models.schemas import AgentRequest

        agent, mock_intel = self._make_agent()
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [{"url": "https://example.com/1", "title": "Test"}]}
        })
        request = AgentRequest(query="test", source="test", metadata={"sub_agent": "search"})
        await agent._fetch_search("test", request, deep=False)
        agent._depth_analyzer.generate_followups.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_news_always_calls_depth_analyzer(self):
        """_fetch_news always runs one DepthAnalyzer pass (not gated on --deep)."""
        from octane.models.schemas import AgentRequest
        from octane.agents.web.content_extractor import ExtractedContent

        followups = [{"query": "Iran leadership", "api": "news", "rationale": "succession"}]
        agent, mock_intel = self._make_agent(depth_followups=followups)

        mock_intel.news_search = AsyncMock(return_value={
            "articles": [{"url": "https://news.com/1", "title": "War Update"}]
        })
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [{"url": "https://example.com/2", "title": "Analysis"}]}
        })

        request = AgentRequest(query="israel iran news", source="test", metadata={"sub_agent": "news"})
        response = await agent._fetch_news("israel iran news", request, deep=False)
        assert response.success is True
        agent._depth_analyzer.generate_followups.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_news_deep_runs_two_rounds(self):
        """deep=True on _fetch_news runs DepthAnalyzer twice (2 deepening rounds)."""
        from octane.models.schemas import AgentRequest

        followups = [{"query": "Iran leadership", "api": "news", "rationale": "succession"}]
        agent, mock_intel = self._make_agent(depth_followups=followups)

        mock_intel.news_search = AsyncMock(return_value={
            "articles": [{"url": "https://news.com/1", "title": "War Update"}]
        })
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [{"url": "https://example.com/2", "title": "Analysis"}]}
        })

        request = AgentRequest(query="israel iran news", source="test", metadata={"sub_agent": "news", "deep": True})
        await agent._fetch_news("israel iran news", request, deep=True)
        # deep=True → 2 rounds
        assert agent._depth_analyzer.generate_followups.call_count == 2


# ---------------------------------------------------------------------------
# Feature-4: Orchestrator.run_stream extra_metadata propagation
# ---------------------------------------------------------------------------


class TestOrchestratorExtraMetadata:
    """run_stream extra_metadata merges into each AgentRequest.metadata."""

    @pytest.mark.asyncio
    async def test_extra_metadata_merged_into_agent_requests(self):
        """extra_metadata={"deep": True} appears in every dispatched AgentRequest."""
        from octane.osa.orchestrator import Orchestrator
        from octane.models.synapse import SynapseEventBus
        from octane.models.dag import TaskDAG, TaskNode
        from octane.models.schemas import AgentResponse

        synapse = SynapseEventBus()
        osa = Orchestrator(synapse)
        osa._preflight_done = True

        # Stub out guard → allow
        osa.guard = AsyncMock()
        osa.guard.check_input = AsyncMock(return_value={"safe": True})

        # Build a minimal DAG with one web node
        web_node = TaskNode(
            task_id="t1",
            agent="web",
            instruction="latest news",
            metadata={"sub_agent": "news"},
        )
        mock_dag = MagicMock()
        mock_dag.nodes = [web_node]
        mock_dag.original_query = "latest news"
        mock_dag.reasoning = "web query"
        mock_dag.execution_order = MagicMock(return_value=[[web_node]])

        osa.decomposer = AsyncMock()
        osa.decomposer.decompose = AsyncMock(return_value=mock_dag)

        # Capture the AgentRequest passed to agent.run
        captured_requests: list = []

        async def fake_run(req):
            captured_requests.append(req)
            return AgentResponse(agent="web", success=True, output="ok", correlation_id=req.correlation_id)

        mock_web_agent = AsyncMock()
        mock_web_agent.run = fake_run
        osa.router = MagicMock()
        osa.router.get_agent = MagicMock(return_value=mock_web_agent)

        osa.policy = MagicMock()
        osa.policy.assess_dag = MagicMock(return_value=MagicMock(decisions=[]))
        osa.checkpoint_mgr = AsyncMock()
        osa.checkpoint_mgr.create = AsyncMock()
        osa.hil = AsyncMock()
        osa.hil.review_ledger = AsyncMock()

        # Evaluator streams one chunk then stops
        async def fake_evaluate_stream(*args, **kwargs):
            yield "response text"

        osa.evaluator = AsyncMock()
        osa.evaluator.evaluate_stream = fake_evaluate_stream

        # Run with extra_metadata
        chunks = []
        async for chunk in osa.run_stream("latest news", extra_metadata={"deep": True}):
            chunks.append(chunk)

        assert len(captured_requests) == 1
        assert captured_requests[0].metadata.get("deep") is True
        assert captured_requests[0].metadata.get("sub_agent") == "news"  # original preserved


# ---------------------------------------------------------------------------
# Feature-5: --deep flag wired end-to-end
# ---------------------------------------------------------------------------


class TestDeepFlagWiring:
    """octane ask --deep passes deep=True through the whole stack."""

    def test_ask_command_has_deep_option(self):
        """The ask Typer command must have a --deep option."""
        import inspect
        import octane.main as main_mod

        src = inspect.getsource(main_mod.ask)
        assert "--deep" in src, "octane ask must define a --deep option"

    def test_ask_async_accepts_deep_param(self):
        """_ask() must accept a 'deep' keyword argument."""
        import inspect
        import octane.main as main_mod

        sig = inspect.signature(main_mod._ask)
        assert "deep" in sig.parameters, "_ask() must have a 'deep' parameter"

    def test_run_stream_accepts_extra_metadata(self):
        """Orchestrator.run_stream must accept extra_metadata parameter."""
        import inspect
        from octane.osa.orchestrator import Orchestrator

        sig = inspect.signature(Orchestrator.run_stream)
        assert "extra_metadata" in sig.parameters, (
            "Orchestrator.run_stream must accept extra_metadata= kwarg"
        )

    def test_fetch_search_accepts_deep_param(self):
        """WebAgent._fetch_search must accept a 'deep' parameter."""
        import inspect
        from octane.agents.web.agent import WebAgent

        sig = inspect.signature(WebAgent._fetch_search)
        assert "deep" in sig.parameters

    def test_fetch_news_accepts_deep_param(self):
        """WebAgent._fetch_news must accept a 'deep' parameter."""
        import inspect
        from octane.agents.web.agent import WebAgent

        sig = inspect.signature(WebAgent._fetch_news)
        assert "deep" in sig.parameters
