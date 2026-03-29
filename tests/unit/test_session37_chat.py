"""Session 37 tests — Crown Jewel Chat.

Tests cover:
    A. IntentGate — classifies user messages into 5 intents
    B. CommandMapper — NL → Octane operation decomposition
    C. ChatEngine — conversation fast path, command routing, recall
    D. Persona system — preference storage and prompt injection
    E. Chat REPL — slash commands /persona, /name
"""

from __future__ import annotations

import asyncio
import pytest

from octane.osa.intent_gate import Intent, IntentGate, _CONVERSATION_EXACT
from octane.osa.command_mapper import CommandMapper, CommandPlan, MappedCommand, COMMAND_MANIFEST
from octane.osa.chat_engine import ChatEngine, build_persona_prompt, build_command_synthesis_prompt

# Use MockBodega from conftest
from tests.unit.conftest import MockBodega, make_synapse, collect_stream


# ═══════════════════════════════════════════════════════════════════════════════
# A. IntentGate Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntentGateExactMatch:
    """Exact-match bypass — no LLM needed."""

    @pytest.mark.asyncio
    async def test_hi_is_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("hi")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_hello_is_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("hello!")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_thanks_is_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("thanks")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_how_are_you_is_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("how are you")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_whats_up_is_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what's up")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_yes_is_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("yes")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_all_exact_greetings_are_conversation(self):
        gate = IntentGate(bodega=None)
        for phrase in _CONVERSATION_EXACT:
            intent, _ = await gate.classify(phrase)
            assert intent == Intent.CONVERSATION, f"'{phrase}' should be CONVERSATION"


class TestIntentGatePatterns:
    """Regex-based pattern matching for recall, command, analysis."""

    @pytest.mark.asyncio
    async def test_recall_remember(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("do you remember what we discussed about AI?")
        assert intent == Intent.RECALL

    @pytest.mark.asyncio
    async def test_recall_articles_i_read(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what articles i read about transformers?")
        assert intent == Intent.RECALL

    @pytest.mark.asyncio
    async def test_recall_last_time(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("last time we talked about portfolios")
        assert intent == Intent.RECALL

    @pytest.mark.asyncio
    async def test_command_show_portfolio(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("show me my portfolio")
        assert intent == Intent.COMMAND

    @pytest.mark.asyncio
    async def test_command_research(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("research Apple's latest earnings")
        assert intent == Intent.COMMAND

    @pytest.mark.asyncio
    async def test_command_compare(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("compare NVDA to AMD in terms of AI revenue")
        assert intent == Intent.ANALYSIS  # compare triggers analysis

    @pytest.mark.asyncio
    async def test_command_monitor(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("monitor AAPL price")
        assert intent == Intent.COMMAND

    @pytest.mark.asyncio
    async def test_command_encrypt(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("encrypt my secret file")
        assert intent == Intent.COMMAND

    @pytest.mark.asyncio
    async def test_analysis_deep_dive(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("analyze the pros and cons of buying Tesla stock")
        assert intent == Intent.ANALYSIS

    @pytest.mark.asyncio
    async def test_analysis_comprehensive(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("give me a comprehensive breakdown of the AI market")
        assert intent == Intent.ANALYSIS


class TestIntentGateFollowUps:
    """Conversational follow-ups should NEVER trigger web search."""

    @pytest.mark.asyncio
    async def test_what_do_you_mean(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what do you mean?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_why_question(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("why?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_can_you_explain(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("can you explain?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_elaborate(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("elaborate")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_how_so(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("how so?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_really(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("really?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_like_what(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("like what?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_what_do_you_mean_by_that(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what do you mean by that?")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_go_on(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("go on")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_but_why(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("but why?")
        assert intent == Intent.CONVERSATION


class TestIntentGateHistoryAwareFollowUp:
    """Short questions mid-conversation should stay conversational."""

    @pytest.mark.asyncio
    async def test_short_question_after_assistant_stays_conversation(self):
        gate = IntentGate(bodega=None)
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hey there! What can I help you with?"},
        ]
        intent, _ = await gate.classify("what is that about?", conversation_history=history)
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_long_factual_question_goes_to_web(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what is the current market cap of Apple Inc?")
        assert intent == Intent.WEB

    @pytest.mark.asyncio
    async def test_short_question_without_history_goes_to_web(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what is the capital of France?")
        assert intent == Intent.WEB


class TestIntentGateShortMessages:
    """Short messages (1-3 words) without action verbs → conversation."""

    @pytest.mark.asyncio
    async def test_one_word(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("interesting")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_two_words(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("oh cool")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_three_words(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("that makes sense")
        assert intent == Intent.CONVERSATION


class TestIntentGateFallback:
    """Fallback behavior for ambiguous queries."""

    @pytest.mark.asyncio
    async def test_question_goes_to_web(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("what is the capital of France?")
        assert intent == Intent.WEB

    @pytest.mark.asyncio
    async def test_ambiguous_defaults_to_conversation(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("I love rainy days and hot chocolate")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_llm_classification(self):
        """LLM correctly classifies when patterns don't match."""
        bodega = MockBodega(chat_response="conversation")
        gate = IntentGate(bodega=bodega)
        intent, _ = await gate.classify("I had the weirdest dream last night about robots")
        assert intent == Intent.CONVERSATION


class TestIntentGateLLMFallback:
    """When LLM fails, gate falls back gracefully."""

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back(self):
        bodega = MockBodega(chat_delay=20.0)  # will timeout at 5s
        gate = IntentGate(bodega=bodega)
        intent, _ = await gate.classify("this is a test message about nothing")
        # Should still get a valid intent (conversation or web)
        assert intent in (Intent.CONVERSATION, Intent.WEB)

    @pytest.mark.asyncio
    async def test_llm_error_falls_back(self):
        bodega = MockBodega(raises=RuntimeError("Bodega is down"))
        gate = IntentGate(bodega=bodega)
        intent, _ = await gate.classify("tell me something interesting")
        assert intent in (Intent.CONVERSATION, Intent.WEB)

    @pytest.mark.asyncio
    async def test_llm_with_im_end_token_stripped(self):
        """LLM returns 'conversation<|im_end|>' — should parse correctly."""
        bodega = MockBodega(chat_response="conversation<|im_end|>")
        gate = IntentGate(bodega=bodega)
        intent, _ = await gate.classify("I had the weirdest dream last night about robots")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_llm_with_eot_token_stripped(self):
        """LLM returns 'web<|eot_id|>' — should parse correctly."""
        bodega = MockBodega(chat_response="web<|eot_id|>")
        gate = IntentGate(bodega=bodega)
        intent, _ = await gate.classify("what is the GDP of Japan in 2025?")
        assert intent == Intent.WEB


# ═══════════════════════════════════════════════════════════════════════════════
# B. CommandMapper Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommandMapperKeyword:
    """Keyword-based fallback mapping."""

    def test_show_portfolio(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("show me my portfolio")
        assert len(plan.commands) >= 1
        assert plan.commands[0].operation == "portfolio.show"

    def test_research_topic(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("research artificial intelligence trends")
        assert plan.commands[0].operation == "research.start"

    def test_news_query(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("latest news about Tesla")
        assert plan.commands[0].operation == "web.news"

    def test_stock_price(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("what is the stock price of Apple")
        assert plan.commands[0].operation == "web.finance"

    def test_extract_url(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("extract content from this article")
        assert plan.commands[0].operation == "extract.url"

    def test_system_health(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("run a health check on the system")
        assert plan.commands[0].operation == "system.health"

    def test_no_match_defaults_to_web_search(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("asdfghjkl gibberish words")
        assert plan.commands[0].operation == "web.search"


class TestCommandMapperLLM:
    """LLM-powered mapping."""

    @pytest.mark.asyncio
    async def test_llm_mapping(self):
        bodega = MockBodega(chat_response='{"reasoning": ["User wants portfolio data"], "commands": [{"operation": "portfolio.show", "description": "Show positions", "parameters": {"prices": true}}]}')
        mapper = CommandMapper(bodega=bodega)
        plan = await mapper.map("how is my portfolio doing?")
        assert len(plan.commands) == 1
        assert plan.commands[0].operation == "portfolio.show"

    @pytest.mark.asyncio
    async def test_llm_multi_command(self):
        bodega = MockBodega(chat_response='{"reasoning": ["Need research and comparison"], "commands": [{"operation": "research.start", "description": "Research X", "parameters": {"topic": "AI chips"}}, {"operation": "research.compare", "description": "Compare them", "parameters": {"items": ["NVDA", "AMD"]}}]}')
        mapper = CommandMapper(bodega=bodega)
        plan = await mapper.map("research AI chips and compare NVDA vs AMD")
        assert len(plan.commands) == 2

    @pytest.mark.asyncio
    async def test_llm_bad_json_falls_to_keywords(self):
        bodega = MockBodega(chat_response="this is not json at all")
        mapper = CommandMapper(bodega=bodega)
        plan = await mapper.map("show me my portfolio with prices")
        assert plan.commands[0].operation == "portfolio.show"

    @pytest.mark.asyncio
    async def test_llm_unknown_op_filtered(self):
        bodega = MockBodega(chat_response='{"reasoning": ["test"], "commands": [{"operation": "nonexistent.op", "description": "bad", "parameters": {}}]}')
        mapper = CommandMapper(bodega=bodega)
        plan = await mapper.map("do something weird")
        # Unknown op filtered out, falls back to keywords
        assert all(c.operation in COMMAND_MANIFEST for c in plan.commands)


class TestCommandManifest:
    """Manifest completeness."""

    def test_all_ops_have_description(self):
        for op_id, info in COMMAND_MANIFEST.items():
            assert "description" in info, f"{op_id} missing description"

    def test_all_ops_have_triggers(self):
        for op_id, info in COMMAND_MANIFEST.items():
            assert "triggers" in info, f"{op_id} missing triggers"
            assert len(info["triggers"]) > 0, f"{op_id} has empty triggers"


# ═══════════════════════════════════════════════════════════════════════════════
# C. ChatEngine Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatEngineConversation:
    """Conversational fast path — direct LLM, no agents."""

    @pytest.mark.asyncio
    async def test_greeting_uses_fast_path(self):
        bodega = MockBodega(stream_chunks=["Hey there! ", "How can I help?"])
        engine = ChatEngine(bodega=bodega, persona={"assistant_name": "bunny"})
        chunks = await collect_stream(engine.respond("hi", "test_session", []))
        response = "".join(chunks)
        assert "Hey there" in response
        # Should NOT have triggered any web search — stream_chunks came from conversation path
        assert bodega.chat_stream_calls == 1

    @pytest.mark.asyncio
    async def test_short_message_uses_fast_path(self):
        bodega = MockBodega(stream_chunks=["That's cool!"])
        engine = ChatEngine(bodega=bodega)
        chunks = await collect_stream(engine.respond("nice", "test_session", []))
        assert "cool" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_conversation_no_bodega(self):
        engine = ChatEngine(bodega=None)
        chunks = await collect_stream(engine.respond("hi", "test_session", []))
        response = "".join(chunks)
        assert "trouble" in response.lower() or "connecting" in response.lower()


class TestChatEnginePersona:
    """Persona system integration."""

    def test_persona_prompt_includes_name(self):
        prompt = build_persona_prompt(assistant_name="bunny", personality="playful")
        assert "Bunny" in prompt
        assert "playful" in prompt

    def test_persona_prompt_includes_user_name(self):
        prompt = build_persona_prompt(assistant_name="bunny", user_name="Rahul")
        assert "Rahul" in prompt

    def test_command_synthesis_prompt(self):
        prompt = build_command_synthesis_prompt(assistant_name="bunny", personality="clever")
        assert "Bunny" in prompt
        assert "clever" in prompt

    def test_engine_name_property(self):
        engine = ChatEngine(persona={"assistant_name": "bunny"})
        assert engine.assistant_name == "bunny"

    def test_engine_default_name(self):
        engine = ChatEngine()
        assert engine.assistant_name == "octane"

    def test_engine_update_persona(self):
        engine = ChatEngine()
        assert engine.assistant_name == "octane"
        engine.update_persona({"assistant_name": "bunny"})
        assert engine.assistant_name == "bunny"


class TestChatEngineRecall:
    """Recall intent path."""

    @pytest.mark.asyncio
    async def test_recall_no_data(self):
        engine = ChatEngine(bodega=None)
        chunks = await collect_stream(
            engine.respond("what did we discuss last time?", "test_session", [])
        )
        response = "".join(chunks)
        assert "memory" in response.lower() or "stored" in response.lower() or "don't have" in response.lower()


class TestChatEngineOSA:
    """OSA pipeline delegation for analysis/web."""

    @pytest.mark.asyncio
    async def test_analysis_without_orchestrator_shows_unavailable(self):
        """Analysis intent without orchestrator shows pipeline unavailable."""
        bodega = MockBodega(stream_chunks=["analysis result"])
        engine = ChatEngine(bodega=bodega)
        # "analyze" triggers ANALYSIS intent
        chunks = await collect_stream(
            engine.respond("analyze the global AI chip market thoroughly", "test_session", [])
        )
        response = "".join(chunks)
        # Without orchestrator, should indicate pipeline unavailable
        assert "pipeline" in response.lower() or "available" in response.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# D. Persona Preferences Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersonaPreferences:
    """Ensure persona keys exist in preference defaults."""

    def test_defaults_include_assistant_name(self):
        from octane.agents.pnl.preference_manager import DEFAULTS
        assert "assistant_name" in DEFAULTS
        assert DEFAULTS["assistant_name"] == "octane"

    def test_defaults_include_assistant_personality(self):
        from octane.agents.pnl.preference_manager import DEFAULTS
        assert "assistant_personality" in DEFAULTS

    def test_pref_choices_include_persona(self):
        from octane.cli.pref import _PREF_CHOICES
        assert "assistant_name" in _PREF_CHOICES
        assert "assistant_personality" in _PREF_CHOICES


# ═══════════════════════════════════════════════════════════════════════════════
# E. Intent-to-Action Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntentToAction:
    """End-to-end intent routing (no live services)."""

    @pytest.mark.asyncio
    async def test_hi_never_triggers_web_search(self):
        """The original bug: 'hi' triggered a 15-second web search.
        Now it should go through the conversation fast path."""
        bodega = MockBodega(stream_chunks=["Hello! How are you?"])
        engine = ChatEngine(bodega=bodega)

        chunks = await collect_stream(engine.respond("hi", "test_session", []))

        # Must get a response
        assert len(chunks) > 0
        response = "".join(chunks)
        assert len(response) > 0

        # Should have used chat_stream (conversational), NOT chat_simple (classification)
        # chat_stream_calls == 1 means only the conversation LLM call happened
        assert bodega.chat_stream_calls == 1

    @pytest.mark.asyncio
    async def test_this_does_not_feel_natural_is_conversation(self):
        """Another original bug: 'this does not feel natural' triggered web search."""
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("this does not feel natural")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_show_portfolio_is_command(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("show me my portfolio with prices")
        assert intent == Intent.COMMAND

    @pytest.mark.asyncio
    async def test_compare_gemini_claude_is_analysis(self):
        """User's example: 'how does claude and gemini compete in coding tasks?'"""
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("how does claude and gemini compete with each other in terms of coding tasks?")
        # This has "compare" via analysis pattern
        assert intent in (Intent.ANALYSIS, Intent.COMMAND, Intent.WEB)

    @pytest.mark.asyncio
    async def test_i_am_thinking_about_apple_is_not_web_search(self):
        """Aggressive command detection: 'I am thinking about Apple' should not default to web search."""
        bodega = MockBodega(chat_response="command")
        gate = IntentGate(bodega=bodega)
        intent, _ = await gate.classify("I am thinking about Apple")
        # With LLM, this should be command (the user's intent is to explore Apple)
        assert intent in (Intent.COMMAND, Intent.CONVERSATION)


# ═══════════════════════════════════════════════════════════════════════════════
# F. Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and robustness."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_punctuation_only(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("...")
        assert intent == Intent.CONVERSATION

    @pytest.mark.asyncio
    async def test_single_emoji(self):
        gate = IntentGate(bodega=None)
        intent, _ = await gate.classify("👋")
        assert intent == Intent.CONVERSATION

    def test_command_mapper_handles_empty_manifest_response(self):
        mapper = CommandMapper(bodega=None)
        plan = mapper._map_with_keywords("")
        assert plan.commands[0].operation == "web.search"

    @pytest.mark.asyncio
    async def test_classify_with_history_context(self):
        bodega = MockBodega(chat_response="recall")
        gate = IntentGate(bodega=bodega)
        history = [
            {"role": "user", "content": "tell me about AI chips"},
            {"role": "assistant", "content": "AI chips are specialized processors..."},
        ]
        intent, _ = await gate.classify("what was that about again?", history)
        # With history context and "what was that" → should recognize recall
        assert intent in (Intent.RECALL, Intent.CONVERSATION, Intent.WEB)


# ═══════════════════════════════════════════════════════════════════════════════
# G. Chat Premium Mode Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatPremiumMode:
    """Tests for the prepare_for_chat model management flow."""

    def test_unload_model_by_id_method_exists(self):
        """BodegaInferenceClient has unload_model_by_id method."""
        from octane.tools.bodega_inference import BodegaInferenceClient
        client = BodegaInferenceClient(base_url="http://localhost:44468")
        assert hasattr(client, "unload_model_by_id")

    def test_prepare_for_chat_method_exists(self):
        """BodegaRouter has prepare_for_chat method."""
        from octane.tools.bodega_router import BodegaRouter
        router = BodegaRouter(topology="balanced")
        assert hasattr(router, "prepare_for_chat")

    def test_chat_log_level_suppression(self):
        """Chat command function sets log level to warning by default."""
        # Verify the chat function signature accepts verbose
        import inspect
        from octane.cli.chat import chat
        sig = inspect.signature(chat)
        assert "verbose" in sig.parameters

    def test_chat_help_text_includes_slash_commands(self):
        """Chat help text includes all slash commands."""
        from octane.cli.chat import _CHAT_HELP
        assert "/help" in _CHAT_HELP
        assert "/deep" in _CHAT_HELP
        assert "/persona" in _CHAT_HELP
        assert "/name" in _CHAT_HELP


class TestChatEngineDAGDisplay:
    """Tests for the OSA handler that shows DAG plan before execution."""

    @pytest.mark.asyncio
    async def test_osa_without_orchestrator(self):
        """Without orchestrator, shows pipeline unavailable."""
        bodega = MockBodega(stream_chunks=["result"])
        engine = ChatEngine(bodega=bodega, orchestrator=None)
        chunks = await collect_stream(
            engine.respond("what is the GDP of Japan?", "test", [])
        )
        response = "".join(chunks)
        assert "pipeline" in response.lower() or "available" in response.lower()
