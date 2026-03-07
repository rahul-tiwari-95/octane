"""Tests for Session 23 features.

Feature-1: Synthesizer deep mode (REASON tier, 3000 tokens, 10 articles, multi-section prompt)
Feature-2: MSRDecider (ambiguity detection, MCQ generation, JSON parsing)
Feature-3: DepthAnalyzer user_context injection
Feature-4: WebAgent Synapse event emission (_emit helper, web_search_round, etc.)
Feature-5: WebAgent MSR wiring (clarification_hook called from request.context)
Feature-6: Orchestrator clarification_hook threading through run_stream
Feature-7: octane trace --verbose rendering helpers
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Feature-1: Synthesizer deep mode
# ─────────────────────────────────────────────────────────────────────────────


class TestSynthesizerDeepMode:
    """synthesize_with_content(deep=True) must use REASON tier, 3000 tokens, 10 articles."""

    def test_deep_synthesis_system_prompt_exists(self):
        """_DEEP_SYNTHESIS_SYSTEM_BASE must exist and contain structured sections."""
        from octane.agents.web import synthesizer as syn_mod

        assert hasattr(syn_mod, "_DEEP_SYNTHESIS_SYSTEM_BASE"), (
            "_DEEP_SYNTHESIS_SYSTEM_BASE constant must exist in synthesizer.py"
        )
        base = syn_mod._DEEP_SYNTHESIS_SYSTEM_BASE
        for section in ("## Summary", "## Key Developments", "## Background", "## What's Next"):
            assert section in base, f"Deep synthesis prompt missing section: {section}"

    def test_deep_synthesis_system_helper_exists(self):
        """_deep_synthesis_system() must exist and inject today's date."""
        from octane.agents.web.synthesizer import _deep_synthesis_system

        result = _deep_synthesis_system()
        assert "Today's date:" in result
        assert "## Summary" in result

    def test_synthesize_with_content_accepts_deep_param(self):
        """synthesize_with_content must accept deep=False (default) and deep=True."""
        import inspect
        from octane.agents.web.synthesizer import Synthesizer

        sig = inspect.signature(Synthesizer.synthesize_with_content)
        params = sig.parameters
        assert "deep" in params, "synthesize_with_content must have a 'deep' parameter"
        assert params["deep"].default is False, "'deep' parameter default must be False"

    def test_deep_mode_uses_10_article_limit(self):
        """deep=True must allow up to 10 articles (vs 5 in standard mode)."""
        src = inspect.getsource(
            __import__("octane.agents.web.synthesizer", fromlist=["Synthesizer"]).Synthesizer.synthesize_with_content
        )
        # deep=True path should use 10
        assert "10" in src, "deep mode must reference article_limit=10"
        assert "5" in src, "standard mode must reference article_limit=5"

    def test_deep_mode_uses_reason_tier(self):
        """deep=True must use ModelTier.REASON for synthesis."""
        src = inspect.getsource(
            __import__("octane.agents.web.synthesizer", fromlist=["Synthesizer"]).Synthesizer.synthesize_with_content
        )
        assert "ModelTier.REASON" in src, "deep synthesis must use ModelTier.REASON"

    def test_deep_mode_uses_3000_tokens(self):
        """deep=True must set max_tokens >= 3000 for synthesis."""
        src = inspect.getsource(
            __import__("octane.agents.web.synthesizer", fromlist=["Synthesizer"]).Synthesizer.synthesize_with_content
        )
        # Look for 3000 token value
        match = re.search(r"synthesis_tokens\s*=\s*(\d+)\s*if deep", src)
        assert match, "synthesis_tokens assignment not found"
        assert int(match.group(1)) >= 3000, f"deep synthesis_tokens must be >= 3000, got {match.group(1)}"

    def test_deep_mode_uses_higher_char_threshold(self):
        """deep=True must use a higher MAX_CHARS_DIRECT to allow larger direct windows."""
        from octane.agents.web.synthesizer import Synthesizer

        assert hasattr(Synthesizer, "_MAX_CHARS_DIRECT_DEEP"), (
            "Synthesizer must have _MAX_CHARS_DIRECT_DEEP class attribute"
        )
        assert Synthesizer._MAX_CHARS_DIRECT_DEEP > Synthesizer._MAX_CHARS_DIRECT, (
            "_MAX_CHARS_DIRECT_DEEP must be larger than _MAX_CHARS_DIRECT"
        )

    @pytest.mark.asyncio
    async def test_deep_synthesis_calls_reason_tier_llm(self):
        """synthesize_with_content(deep=True) must call bodega with REASON tier."""
        from octane.agents.web.synthesizer import Synthesizer
        from octane.tools.topology import ModelTier

        @dataclass
        class FakeArticle:
            url: str = "https://example.com/article"
            text: str = "Some article text " * 50  # short enough for direct synthesis
            method: str = "trafilatura"
            word_count: int = 100

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="## Summary\nTest synthesis output")
        synthesizer = Synthesizer(bodega=mock_bodega)

        articles = [FakeArticle() for _ in range(3)]
        result = await synthesizer.synthesize_with_content("test query", articles, deep=True)

        # Verify chat_simple was called with REASON tier
        assert mock_bodega.chat_simple.called, "chat_simple must be called"
        call_kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert call_kwargs.get("tier") == ModelTier.REASON, (
            f"Expected REASON tier for deep synthesis, got {call_kwargs.get('tier')}"
        )
        assert call_kwargs.get("max_tokens", 0) >= 3000, (
            f"Expected max_tokens >= 3000, got {call_kwargs.get('max_tokens')}"
        )

    @pytest.mark.asyncio
    async def test_standard_synthesis_uses_mid_tier(self):
        """synthesize_with_content(deep=False) must use MID tier (unchanged behaviour)."""
        from octane.agents.web.synthesizer import Synthesizer
        from octane.tools.topology import ModelTier

        @dataclass
        class FakeArticle:
            url: str = "https://example.com/article"
            text: str = "Short article text"
            method: str = "trafilatura"
            word_count: int = 10

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="Test output")
        synthesizer = Synthesizer(bodega=mock_bodega)

        articles = [FakeArticle()]
        await synthesizer.synthesize_with_content("test query", articles, deep=False)

        call_kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert call_kwargs.get("tier") == ModelTier.MID, (
            f"Standard synthesis should use MID tier, got {call_kwargs.get('tier')}"
        )
        assert call_kwargs.get("max_tokens", 0) < 2000, (
            f"Standard synthesis max_tokens should be < 2000, got {call_kwargs.get('max_tokens')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature-2: MSRDecider
# ─────────────────────────────────────────────────────────────────────────────


class TestMSRDecider:
    """MSRDecider must parse LLM JSON and return MSRResult correctly."""

    def test_msr_dataclasses_exist(self):
        """MSRQuestion and MSRResult dataclasses must exist."""
        from octane.agents.web.msr_decider import MSRQuestion, MSRResult

        q = MSRQuestion(text="What's your focus?", options=["A", "B", "C"])
        assert q.text == "What's your focus?"
        assert len(q.options) == 3

        r = MSRResult(should_ask=True, questions=[q])
        assert r.should_ask is True
        assert len(r.questions) == 1

    def test_msr_decider_init_with_no_bodega(self):
        """MSRDecider(bodega=None) must not crash and always return should_ask=False."""
        from octane.agents.web.msr_decider import MSRDecider

        decider = MSRDecider(bodega=None)
        assert decider is not None

    @pytest.mark.asyncio
    async def test_decide_returns_no_ask_when_no_bodega(self):
        """MSRDecider with no bodega must return MSRResult(should_ask=False)."""
        from octane.agents.web.msr_decider import MSRDecider

        decider = MSRDecider(bodega=None)
        result = await decider.decide("test query", ["finding 1", "finding 2"])
        assert result.should_ask is False
        assert result.questions == []

    @pytest.mark.asyncio
    async def test_decide_returns_no_ask_when_no_findings(self):
        """MSRDecider with empty findings must return should_ask=False without calling LLM."""
        from octane.agents.web.msr_decider import MSRDecider

        mock_bodega = AsyncMock()
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("test query", [])
        assert result.should_ask is False
        mock_bodega.chat_simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_decide_parses_should_ask_false(self):
        """MSRDecider must parse {should_ask: false} correctly."""
        from octane.agents.web.msr_decider import MSRDecider

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(
            return_value='{"should_ask": false, "questions": []}'
        )
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("NVDA stock price", ["NVDA is $120"])
        assert result.should_ask is False
        assert result.questions == []

    @pytest.mark.asyncio
    async def test_decide_parses_should_ask_true_with_questions(self):
        """MSRDecider must parse {should_ask: true, questions: [...]} correctly."""
        from octane.agents.web.msr_decider import MSRDecider

        llm_response = """{
            "should_ask": true,
            "questions": [
                {
                    "text": "What aspect are you most interested in?",
                    "options": ["Military operations", "Diplomatic talks", "Economic sanctions", "Regional reactions"]
                },
                {
                    "text": "What time period?",
                    "options": ["Last 24 hours", "Last week", "Historical context"]
                }
            ]
        }"""

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=llm_response)
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("israel iran latest news", ["Round-1 finding 1"])

        assert result.should_ask is True
        assert len(result.questions) == 2
        assert result.questions[0].text == "What aspect are you most interested in?"
        assert len(result.questions[0].options) == 4
        assert "Military operations" in result.questions[0].options

    @pytest.mark.asyncio
    async def test_decide_strips_think_block(self):
        """MSRDecider must strip <think>...</think> reasoning before parsing JSON."""
        from octane.agents.web.msr_decider import MSRDecider

        llm_response = (
            "<think>This query is about multiple aspects...</think>"
            '{"should_ask": false, "questions": []}'
        )

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=llm_response)
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("clear query", ["finding"])

        # Should not crash despite <think> block
        assert result.should_ask is False

    @pytest.mark.asyncio
    async def test_decide_caps_questions_at_max(self):
        """MSRDecider must not return more than max_questions questions."""
        from octane.agents.web.msr_decider import MSRDecider

        # Return 5 questions but cap is 3
        many_questions = [
            {"text": f"Question {i}?", "options": ["A", "B", "C"]}
            for i in range(5)
        ]
        llm_response = f'{{"should_ask": true, "questions": {__import__("json").dumps(many_questions)}}}'

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=llm_response)
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("test", ["finding"], max_questions=3)

        assert result.should_ask is True
        assert len(result.questions) <= 3, f"Got {len(result.questions)} questions, expected <= 3"

    @pytest.mark.asyncio
    async def test_decide_returns_no_ask_on_llm_error(self):
        """MSRDecider must gracefully handle LLM errors by returning should_ask=False."""
        from octane.agents.web.msr_decider import MSRDecider

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("LLM error"))
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("test query", ["finding 1"])

        assert result.should_ask is False  # graceful degradation
        assert result.questions == []

    @pytest.mark.asyncio
    async def test_decide_returns_no_ask_on_bad_json(self):
        """MSRDecider must return should_ask=False if LLM returns non-JSON."""
        from octane.agents.web.msr_decider import MSRDecider

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="I cannot determine this.")
        decider = MSRDecider(bodega=mock_bodega)
        result = await decider.decide("test", ["finding"])

        assert result.should_ask is False

    def test_msr_uses_fast_tier(self):
        """MSRDecider must use ModelTier.FAST for efficiency."""
        src = inspect.getsource(
            __import__("octane.agents.web.msr_decider", fromlist=["MSRDecider"]).MSRDecider._llm_decide
        )
        assert "ModelTier.FAST" in src, "MSRDecider must use ModelTier.FAST"


# ─────────────────────────────────────────────────────────────────────────────
# Feature-3: DepthAnalyzer user_context injection
# ─────────────────────────────────────────────────────────────────────────────


class TestDepthAnalyzerUserContext:
    """generate_followups must accept and inject user_context into the LLM prompt."""

    def test_generate_followups_accepts_user_context(self):
        """generate_followups must have user_context parameter."""
        sig = inspect.signature(
            __import__(
                "octane.agents.web.depth_analyzer", fromlist=["DepthAnalyzer"]
            ).DepthAnalyzer.generate_followups
        )
        assert "user_context" in sig.parameters, (
            "generate_followups must have user_context parameter"
        )
        assert sig.parameters["user_context"].default is None, (
            "user_context must default to None"
        )

    @pytest.mark.asyncio
    async def test_user_context_injected_into_prompt(self):
        """When user_context is provided, it must appear in the LLM prompt."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        captured_prompt = {}

        async def capture_chat_simple(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            return "[]"

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=capture_chat_simple)
        analyzer = DepthAnalyzer(bodega=mock_bodega)

        await analyzer.generate_followups(
            original_query="israel iran news",
            findings=["Finding 1", "Finding 2"],
            user_context="Military operations & timeline",
        )

        prompt = captured_prompt.get("prompt", "")
        assert "Military operations & timeline" in prompt, (
            "user_context must appear in the DepthAnalyzer prompt"
        )

    @pytest.mark.asyncio
    async def test_no_user_context_does_not_crash(self):
        """generate_followups without user_context must work exactly as before."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(
            return_value='[{"query": "follow up", "api": "search", "rationale": "test"}]'
        )
        analyzer = DepthAnalyzer(bodega=mock_bodega)

        results = await analyzer.generate_followups(
            original_query="test query",
            findings=["finding 1", "finding 2"],
        )
        assert isinstance(results, list)
        assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Feature-4: WebAgent Synapse event emission
# ─────────────────────────────────────────────────────────────────────────────


class TestWebAgentSynapseEvents:
    """WebAgent must have an _emit() helper and use it during search rounds."""

    def test_emit_helper_exists(self):
        """WebAgent must have an _emit() method."""
        from octane.agents.web.agent import WebAgent

        assert hasattr(WebAgent, "_emit"), "WebAgent must have _emit() helper method"

    def test_emit_helper_signature(self):
        """_emit() must accept correlation_id, event_type, and payload."""
        sig = inspect.signature(
            __import__(
                "octane.agents.web.agent", fromlist=["WebAgent"]
            ).WebAgent._emit
        )
        params = list(sig.parameters.keys())
        assert "correlation_id" in params, "_emit must have correlation_id param"
        assert "event_type" in params, "_emit must have event_type param"
        assert "payload" in params, "_emit must have payload param"

    def test_agent_imports_synapse_event(self):
        """agent.py must import SynapseEvent for event emission."""
        src = inspect.getsource(
            __import__("octane.agents.web.agent", fromlist=["WebAgent"])
        )
        assert "SynapseEvent" in src, "agent.py must import SynapseEvent"

    def test_web_search_round_event_emitted(self):
        """agent.py must emit 'web_search_round' events."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert '"web_search_round"' in src or "'web_search_round'" in src, (
            "WebAgent must emit web_search_round events"
        )

    def test_web_depth_analysis_event_emitted(self):
        """agent.py must emit 'web_depth_analysis' events."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert '"web_depth_analysis"' in src or "'web_depth_analysis'" in src, (
            "WebAgent must emit web_depth_analysis events"
        )

    def test_msr_decision_event_emitted(self):
        """agent.py must emit 'msr_decision' events."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert '"msr_decision"' in src or "'msr_decision'" in src, (
            "WebAgent must emit msr_decision events"
        )

    def test_web_synthesis_event_emitted(self):
        """agent.py must emit 'web_synthesis' events."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert '"web_synthesis"' in src or "'web_synthesis'" in src, (
            "WebAgent must emit web_synthesis events"
        )

    def test_emit_never_raises(self):
        """_emit() must silently swallow all exceptions (never crash the pipeline)."""
        from octane.agents.web.agent import WebAgent

        # Create a broken synapse that raises on emit
        broken_synapse = MagicMock()
        broken_synapse.emit = MagicMock(side_effect=RuntimeError("synapse broken"))

        agent = WebAgent(synapse=broken_synapse, intel=MagicMock(), bodega=None)
        # Should not raise even with broken synapse
        agent._emit("test-cid", "web_search_round", {"test": "data"})

    def test_msr_decider_wired_in_agent(self):
        """WebAgent must instantiate an MSRDecider."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert "MSRDecider" in src, "WebAgent must use MSRDecider"
        assert "msr_decider" in src.lower(), "WebAgent must have _msr_decider attribute"


# ─────────────────────────────────────────────────────────────────────────────
# Feature-5: WebAgent MSR clarification_hook wiring
# ─────────────────────────────────────────────────────────────────────────────


class TestWebAgentMSRWiring:
    """clarification_hook from request.context must be called when MSR triggers."""

    @pytest.mark.asyncio
    async def test_clarification_hook_called_when_msr_triggers(self):
        """When MSRDecider says should_ask=True, clarification_hook must be awaited."""
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent
        from octane.agents.web.msr_decider import MSRQuestion, MSRResult
        from octane.models.schemas import AgentRequest

        # Setup: MSRDecider says should_ask=True
        mock_msr = AsyncMock()
        mock_msr.decide = AsyncMock(return_value=MSRResult(
            should_ask=True,
            questions=[MSRQuestion(
                text="What aspect?",
                options=["Military", "Diplomatic", "Economic"],
            )]
        ))

        # Track if hook was called
        hook_calls = []

        async def fake_hook(questions):
            hook_calls.append(questions)
            return "Military operations"

        # Build a minimal WebAgent with mocked dependencies
        synapse = MagicMock()
        synapse.emit = MagicMock()

        intel = AsyncMock()
        intel.web_search = AsyncMock(return_value={"web": {"results": [
            {"url": "https://example.com/1", "title": "Test", "description": "Test article"},
        ]}})
        intel.news_search = AsyncMock(return_value={"articles": []})

        bodega = AsyncMock()
        bodega.chat_simple = AsyncMock(return_value="test response")

        # Return a real ExtractedContent so r1_usable is non-empty (needed to
        # satisfy the `if deep and r1_usable:` guard that gates the MSR block)
        fake_article = ExtractedContent(
            url="https://example.com/1",
            text="Israel and Iran tensions escalate significantly today. " * 20,
            word_count=120,
            method="trafilatura",
        )
        extractor = AsyncMock()
        extractor.extract_batch = AsyncMock(return_value=[fake_article])

        browser = AsyncMock()
        browser.scrape = AsyncMock(return_value=None)

        agent = WebAgent(synapse=synapse, intel=intel, bodega=bodega,
                         extractor=extractor, browser=browser)
        agent._msr_decider = mock_msr  # inject mock MSRDecider
        agent._strategist = AsyncMock()
        agent._strategist.strategize = AsyncMock(return_value=[{"query": "test query"}])
        agent._depth_analyzer = AsyncMock()
        agent._depth_analyzer.generate_followups = AsyncMock(return_value=[])
        agent._store_pages = AsyncMock()

        request = AgentRequest(
            query="israel iran news",
            correlation_id="test-cid",
            metadata={"sub_agent": "search"},
            context={"clarification_hook": fake_hook},
        )

        # Run with deep=True to trigger MSR path
        await agent._fetch_search("israel iran news", request, deep=True)

        # Verify hook was called with questions from MSRDecider
        assert len(hook_calls) == 1, f"clarification_hook must be called once, got {len(hook_calls)}"
        assert hook_calls[0][0].text == "What aspect?", (
            "Hook must be called with MSRQuestion objects"
        )

    @pytest.mark.asyncio
    async def test_hook_not_called_when_msr_skips(self):
        """clarification_hook must NOT be called when MSRDecider says should_ask=False."""
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent
        from octane.agents.web.msr_decider import MSRResult
        from octane.models.schemas import AgentRequest

        mock_msr = AsyncMock()
        mock_msr.decide = AsyncMock(return_value=MSRResult(should_ask=False, questions=[]))

        hook_calls = []

        async def fake_hook(questions):
            hook_calls.append(questions)
            return None

        synapse = MagicMock()
        synapse.emit = MagicMock()
        intel = AsyncMock()
        intel.web_search = AsyncMock(return_value={"web": {"results": [
            {"url": "https://example.com/1", "title": "Test", "description": "Test article"},
        ]}})
        bodega = AsyncMock()
        fake_article = ExtractedContent(
            url="https://example.com/1",
            text="NVDA stock price is $120 today. " * 10,
            word_count=60,
            method="trafilatura",
        )
        extractor = AsyncMock()
        extractor.extract_batch = AsyncMock(return_value=[fake_article])
        browser = AsyncMock()
        browser.scrape = AsyncMock(return_value=None)

        agent = WebAgent(synapse=synapse, intel=intel, bodega=bodega,
                         extractor=extractor, browser=browser)
        agent._msr_decider = mock_msr
        agent._strategist = AsyncMock()
        agent._strategist.strategize = AsyncMock(return_value=[{"query": "test"}])
        agent._depth_analyzer = AsyncMock()
        agent._depth_analyzer.generate_followups = AsyncMock(return_value=[])
        agent._store_pages = AsyncMock()

        request = AgentRequest(
            query="NVDA stock price",
            correlation_id="test-cid",
            metadata={"sub_agent": "search"},
            context={"clarification_hook": fake_hook},
        )

        await agent._fetch_search("NVDA stock price", request, deep=True)
        assert len(hook_calls) == 0, (
            "Hook must not be called when MSR decides should_ask=False"
        )

    @pytest.mark.asyncio
    async def test_no_hook_in_context_skips_msr(self):
        """When no clarification_hook in context, MSRDecider must NOT be called.

        Provides real ExtractedContent so r1_usable is non-empty — this verifies
        that the MISSING HOOK (not empty content) prevents MSR from running.
        """
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent
        from octane.models.schemas import AgentRequest

        msr_called = []

        class SpyMSRDecider:
            async def decide(self, *args, **kwargs):
                msr_called.append(True)
                return MagicMock(should_ask=False, questions=[])

        synapse = MagicMock()
        synapse.emit = MagicMock()
        intel = AsyncMock()
        intel.web_search = AsyncMock(return_value={"web": {"results": [
            {"url": "https://example.com/1", "title": "Test", "description": "Test article"},
        ]}})
        bodega = AsyncMock()
        bodega.chat_simple = AsyncMock(return_value="synthesis result")

        # Provide real content so r1_usable is non-empty
        fake_article = ExtractedContent(
            url="https://example.com/1",
            text="Test article content for MSR skip test. " * 20,
            word_count=100,
            method="trafilatura",
        )
        extractor = AsyncMock()
        extractor.extract_batch = AsyncMock(return_value=[fake_article])
        browser = AsyncMock()
        browser.scrape = AsyncMock(return_value=None)

        agent = WebAgent(synapse=synapse, intel=intel, bodega=bodega,
                         extractor=extractor, browser=browser)
        agent._msr_decider = SpyMSRDecider()
        agent._strategist = AsyncMock()
        agent._strategist.strategize = AsyncMock(return_value=[{"query": "test"}])
        agent._depth_analyzer = AsyncMock()
        agent._depth_analyzer.generate_followups = AsyncMock(return_value=[])
        agent._store_pages = AsyncMock()

        # No clarification_hook in context
        request = AgentRequest(
            query="test query",
            correlation_id="test-cid",
            metadata={"sub_agent": "search"},
            context={},  # no hook!
        )

        await agent._fetch_search("test query", request, deep=True)
        # MSR decide should not be called when there's no hook
        assert len(msr_called) == 0, (
            "MSRDecider.decide must not be called when no clarification_hook in context"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature-6: Orchestrator clarification_hook threading
# ─────────────────────────────────────────────────────────────────────────────


class TestOrchestratorClarificationHook:
    """run_stream must accept clarification_hook and inject it into AgentRequest.context."""

    def test_run_stream_accepts_clarification_hook(self):
        """run_stream must have a clarification_hook parameter."""
        from octane.osa.orchestrator import Orchestrator

        sig = inspect.signature(Orchestrator.run_stream)
        assert "clarification_hook" in sig.parameters, (
            "run_stream must accept a clarification_hook parameter"
        )
        assert sig.parameters["clarification_hook"].default is None, (
            "clarification_hook must default to None"
        )

    def test_run_stream_injects_hook_into_context(self):
        """run_stream must inject clarification_hook into AgentRequest.context."""
        src = inspect.getsource(
            __import__("octane.osa.orchestrator", fromlist=["Orchestrator"]).Orchestrator.run_stream
        )
        # The hook must be added to the task_context dict that feeds AgentRequest
        assert "clarification_hook" in src, (
            "run_stream source must reference clarification_hook injection into context"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature-7: octane trace --verbose rendering
# ─────────────────────────────────────────────────────────────────────────────


class TestTraceVerboseRendering:
    """octane trace --verbose must render new web event types."""

    def test_verbose_flag_in_trace_command(self):
        """trace command must accept --verbose / -v flag."""
        import octane.main as main_mod

        src = inspect.getsource(main_mod.trace)
        assert "--verbose" in src or "verbose" in src, (
            "trace command must have --verbose flag"
        )

    def test_print_verbose_web_trace_function_exists(self):
        """_print_verbose_web_trace helper function must exist in main.py."""
        import octane.main as main_mod

        assert hasattr(main_mod, "_print_verbose_web_trace"), (
            "_print_verbose_web_trace function must exist in main.py"
        )

    def test_verbose_trace_handles_empty_events(self):
        """_print_verbose_web_trace must handle empty event lists without crashing."""
        import octane.main as main_mod
        from datetime import datetime, timezone

        # Should not raise with no web events
        main_mod._print_verbose_web_trace([], t0=datetime.now(timezone.utc))

    def test_new_event_types_in_event_colours(self):
        """New web event types must have entries in _EVENT_COLOURS."""
        import octane.main as main_mod

        for evt_type in ("web_search_round", "web_depth_analysis", "msr_decision", "web_synthesis"):
            assert evt_type in main_mod._EVENT_COLOURS, (
                f"_EVENT_COLOURS must include '{evt_type}'"
            )

    def test_new_event_types_in_event_icons(self):
        """New web event types must have entries in _EVENT_ICONS."""
        import octane.main as main_mod

        for evt_type in ("web_search_round", "web_depth_analysis", "msr_decision", "web_synthesis"):
            assert evt_type in main_mod._EVENT_ICONS, (
                f"_EVENT_ICONS must include '{evt_type}'"
            )

    def test_trace_async_accepts_verbose_param(self):
        """_trace async function must accept verbose parameter."""
        import octane.main as main_mod

        sig = inspect.signature(main_mod._trace)
        assert "verbose" in sig.parameters, (
            "_trace async function must accept verbose parameter"
        )
        assert sig.parameters["verbose"].default is False, (
            "verbose must default to False"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Synthesis deep param flows through agent.py
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentPassesDeepToSynthesizer:
    """Both _fetch_news and _fetch_search must pass deep=deep to synthesize_with_content."""

    def test_fetch_news_passes_deep(self):
        """_fetch_news must call synthesize_with_content(query, all_usable, deep=deep)."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod.WebAgent._fetch_news)
        assert "deep=deep" in src, (
            "_fetch_news must pass deep=deep to synthesize_with_content()"
        )

    def test_fetch_search_passes_deep(self):
        """_fetch_search must call synthesize_with_content(query, all_usable, deep=deep)."""
        import octane.agents.web.agent as agent_mod

        src = inspect.getsource(agent_mod.WebAgent._fetch_search)
        assert "deep=deep" in src, (
            "_fetch_search must pass deep=deep to synthesize_with_content()"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regression fixes (observed from live run logs — March 2026)
# ─────────────────────────────────────────────────────────────────────────────


class TestWaveTimeoutDeepMode:
    """Bug fix: 90 s wave ceiling killed deep synthesis. Must be 180 s for deep mode."""

    def test_orchestrator_uses_deep_ceiling_variable(self):
        """run_stream source must reference a deep-aware ceiling (180.0)."""
        import octane.osa.orchestrator as orch_mod

        src = inspect.getsource(orch_mod.Orchestrator.run_stream)
        assert "180" in src, (
            "run_stream must use 180 s ceiling for deep mode"
        )
        assert "wave_ceiling" in src or "is_deep" in src, (
            "run_stream must have a conditional wave ceiling variable"
        )

    def test_deep_flag_read_from_extra_metadata(self):
        """run_stream must derive the deep flag from extra_metadata, not hardcode it."""
        import octane.osa.orchestrator as orch_mod

        src = inspect.getsource(orch_mod.Orchestrator.run_stream)
        # Should check extra_metadata for 'deep' key
        assert "extra_metadata" in src and '"deep"' in src or "'deep'" in src, (
            "wave ceiling logic must read 'deep' from extra_metadata"
        )


class TestDepthAnalyzerCodeFenceStripping:
    """Bug fix: LLM returned ```json\n[...]\n``` and depth_analyzer_no_json was logged."""

    def test_code_fence_stripped_before_parse(self):
        """depth_analyzer.py must strip ```json ... ``` fences before JSON extraction."""
        src = inspect.getsource(
            __import__(
                "octane.agents.web.depth_analyzer", fromlist=["DepthAnalyzer"]
            ).DepthAnalyzer._llm_followups
        )
        # Must have a code-fence stripping step
        assert "```" in src or "code_fence" in src.lower() or re.search(r"sub.*```", src), (
            "_llm_followups must strip markdown code fences before JSON extraction"
        )

    @pytest.mark.asyncio
    async def test_fenced_json_parsed_correctly(self):
        """DepthAnalyzer must parse LLM response wrapped in ```json ... ``` fences."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        fenced_response = (
            "```json\n"
            '[{"query": "kali yuga 2026 signs", "api": "search", "rationale": "current evidence"}]\n'
            "```"
        )

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=fenced_response)
        analyzer = DepthAnalyzer(bodega=mock_bodega)

        results = await analyzer.generate_followups(
            original_query="kali yuga and world state 2026",
            findings=["Kali Yuga is the last of four yugas in Hindu cosmology"],
        )

        assert len(results) == 1, (
            f"DepthAnalyzer must parse fenced JSON; got {len(results)} results"
        )
        assert results[0]["query"] == "kali yuga 2026 signs"

    @pytest.mark.asyncio
    async def test_plain_fences_also_stripped(self):
        """DepthAnalyzer must also handle plain ``` fences (no language tag)."""
        from octane.agents.web.depth_analyzer import DepthAnalyzer

        plain_fenced = (
            "```\n"
            '[{"query": "HBM4 supply chain 2026", "api": "news", "rationale": "supply info"}]\n'
            "```"
        )

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value=plain_fenced)
        analyzer = DepthAnalyzer(bodega=mock_bodega)

        results = await analyzer.generate_followups(
            original_query="HBM4 and NVIDIA",
            findings=["HBM4 is the next generation High Bandwidth Memory"],
        )

        assert len(results) == 1, "Plain code-fence JSON must parse correctly"
        assert results[0]["api"] == "news"


class TestMSRPromptAdaptive:
    """Bug fix: MSR showed geopolitical options for a kali yuga / spiritual query."""

    def test_msr_system_prompt_has_no_hardcoded_military_options(self):
        """_MSR_SYSTEM must not contain hardcoded 'Military operations' example options."""
        from octane.agents.web import msr_decider as msr_mod

        assert "Military operations" not in msr_mod._MSR_SYSTEM, (
            "_MSR_SYSTEM must not hardcode 'Military operations' as an example option — "
            "options must be query-adaptive"
        )
        assert "Diplomatic negotiations" not in msr_mod._MSR_SYSTEM, (
            "_MSR_SYSTEM must not hardcode 'Diplomatic negotiations' as an example option"
        )
        assert "Regional geopolitics" not in msr_mod._MSR_SYSTEM, (
            "_MSR_SYSTEM must not hardcode 'Regional geopolitics' as an example option"
        )

    def test_msr_system_prompt_instructs_query_specific_options(self):
        """_MSR_SYSTEM must instruct the LLM to generate query-specific options."""
        from octane.agents.web import msr_decider as msr_mod

        prompt_lower = msr_mod._MSR_SYSTEM.lower()
        # Must contain language telling the LLM to derive options from the query
        assert any(kw in prompt_lower for kw in ("specific to", "relevant to", "derive")), (
            "_MSR_SYSTEM must tell the LLM to generate options specific to the query topic"
        )

    def test_msr_system_prompt_warns_against_generic_categories(self):
        """_MSR_SYSTEM must explicitly warn against reusing generic geopolitical categories."""
        from octane.agents.web import msr_decider as msr_mod

        # We added "Do not reuse generic geopolitical categories" instruction
        assert "Do not reuse" in msr_mod._MSR_SYSTEM, (
            "_MSR_SYSTEM must warn the LLM not to reuse generic category names"
        )
        # Must tell LLM to derive from actual topic
        assert "Derive" in msr_mod._MSR_SYSTEM or "derive" in msr_mod._MSR_SYSTEM, (
            "_MSR_SYSTEM must instruct deriving options from the actual topic"
        )
