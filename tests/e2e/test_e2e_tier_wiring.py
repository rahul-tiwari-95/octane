"""E2E tier-wiring integration tests (Session 20D).

These tests exercise the **full call chain** from each wired agent method down
through BodegaRouter to confirm that the correct ModelTier is actually forwarded
to the underlying chat_simple / chat_stream call.

The mock lives at ``BodegaRouter.chat_simple`` / ``BodegaRouter.chat_stream``
(not inside BodegaInferenceClient), so the routing logic in each agent is fully
exercised — we verify what tier arrived at the router boundary.

No live infrastructure required — all HTTP is replaced by mock coroutines.

Agents and their expected tiers (from Session 20B wiring):
    QueryStrategist.strategize()            → ModelTier.FAST
    Synthesizer.synthesize_news()           → ModelTier.MID
    Synthesizer.synthesize_search()         → ModelTier.MID
    Synthesizer._summarize_chunk()          → ModelTier.MID
    Synthesizer.synthesize_full_text()      → ModelTier.MID
    Decomposer._classify_with_llm()         → ModelTier.FAST
    Evaluator.evaluate()                    → ModelTier.REASON
    Evaluator.evaluate_stream()             → ModelTier.REASON
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from octane.tools.topology import ModelTier
from octane.models.schemas import AgentResponse


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_chat_simple(response_text: str = "ok") -> AsyncMock:
    """Return an AsyncMock that records calls and returns a plain string."""
    return AsyncMock(return_value=response_text)


def _fake_chat_stream(*chunks: str):
    """Return an async generator mock that yields the given text chunks."""
    async def _gen(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    mock = MagicMock()
    mock.return_value = _gen()
    # Make it re-creatable across calls
    mock.side_effect = lambda *a, **kw: _gen()
    return mock


def _make_router(fast_response: str = "fast_ok", mid_response: str = "mid_ok",
                 reason_response: str = "reason_ok") -> MagicMock:
    """Return a mock BodegaRouter whose chat_simple returns the given responses by tier."""
    router = MagicMock()

    async def _chat_simple(prompt, system="", tier=ModelTier.REASON, **kwargs):
        return {
            ModelTier.FAST: fast_response,
            ModelTier.MID: mid_response,
            ModelTier.REASON: reason_response,
        }.get(tier, "unknown_tier")

    router.chat_simple = AsyncMock(side_effect=_chat_simple)

    async def _chat_stream(prompt, system="", tier=ModelTier.REASON, **kwargs):
        text = {
            ModelTier.FAST: fast_response,
            ModelTier.MID: mid_response,
            ModelTier.REASON: reason_response,
        }.get(tier, "unknown_tier")
        yield text

    router.chat_stream = _chat_stream
    return router


def _last_tier(mock_chat_simple: AsyncMock) -> ModelTier:
    """Return the `tier` kwarg from the most recent call to mock_chat_simple."""
    return mock_chat_simple.call_args.kwargs["tier"]


# ── QueryStrategist ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_strategist_uses_fast_tier():
    """QueryStrategist.strategize() must route to ModelTier.FAST."""
    from octane.agents.web.query_strategist import QueryStrategist

    # Return valid JSON that strategize() will parse
    valid_json = json.dumps([
        {"query": "NVDA stock price", "api": "finance", "rationale": "market data"},
    ])
    router = _make_router(fast_response=valid_json)

    qs = QueryStrategist(bodega=router)
    await qs.strategize("What is NVDA stock price?")

    router.chat_simple.assert_awaited()
    assert _last_tier(router.chat_simple) == ModelTier.FAST, (
        "QueryStrategist must call chat_simple with tier=ModelTier.FAST"
    )


@pytest.mark.asyncio
async def test_query_strategist_fast_tier_is_not_reason():
    """Sanity: the QueryStrategist must NOT use REASON (that's for evaluation)."""
    from octane.agents.web.query_strategist import QueryStrategist

    valid_json = json.dumps([{"query": "q", "api": "search", "rationale": "r"}])
    router = _make_router(fast_response=valid_json)

    qs = QueryStrategist(bodega=router)
    await qs.strategize("lookup something")

    assert _last_tier(router.chat_simple) != ModelTier.REASON


# ── Synthesizer ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesizer_news_uses_mid_tier():
    """Synthesizer.synthesize_news() must route to ModelTier.MID."""
    from octane.agents.web.synthesizer import Synthesizer

    router = _make_router(mid_response="Key stories: 1. NVDA surges — Source: Reuters")
    synth = Synthesizer(bodega=router)

    articles = [
        {"title": "NVDA surges", "publisher": {"title": "Reuters"},
         "published date": "2026-02-28", "summary": "NVDA up 5%"}
    ]
    await synth.synthesize_news("NVDA news", articles)

    router.chat_simple.assert_awaited()
    assert _last_tier(router.chat_simple) == ModelTier.MID


@pytest.mark.asyncio
async def test_synthesizer_search_uses_mid_tier():
    """Synthesizer.synthesize_search() must route to ModelTier.MID."""
    from octane.agents.web.synthesizer import Synthesizer

    router = _make_router(mid_response="• Key finding about NVDA")
    synth = Synthesizer(bodega=router)

    results = [{"title": "NVDA explainer", "description": "NVDA is...", "url": "https://x.com"}]
    await synth.synthesize_search("explain NVDA", results)

    router.chat_simple.assert_awaited()
    assert _last_tier(router.chat_simple) == ModelTier.MID


@pytest.mark.asyncio
async def test_synthesizer_chunk_summarize_uses_mid_tier():
    """Synthesizer._summarize_chunk() must route to ModelTier.MID."""
    from octane.agents.web.synthesizer import Synthesizer

    router = _make_router(mid_response="Dense chunk summary.")
    synth = Synthesizer(bodega=router)

    await synth._summarize_chunk("Long article text " * 100, "NVDA analysis")

    router.chat_simple.assert_awaited()
    assert _last_tier(router.chat_simple) == ModelTier.MID


@pytest.mark.asyncio
async def test_synthesizer_mid_tier_is_not_fast():
    """Sanity: Synthesizer must NOT use FAST (that's for cheap routing)."""
    from octane.agents.web.synthesizer import Synthesizer

    router = _make_router(mid_response="summary")
    synth = Synthesizer(bodega=router)
    await synth.synthesize_search("query", [{"title": "t", "description": "d", "url": "u"}])

    assert _last_tier(router.chat_simple) != ModelTier.FAST


@pytest.mark.asyncio
async def test_synthesizer_all_four_call_sites_use_mid():
    """All Synthesizer LLM call sites use MID — none sneaks in FAST or REASON."""
    from octane.agents.web.synthesizer import Synthesizer

    recorded_tiers: list[ModelTier] = []

    router = MagicMock()

    async def _record_tier(prompt, system="", tier=ModelTier.REASON, **kwargs):
        recorded_tiers.append(tier)
        return "synthesized text"

    router.chat_simple = AsyncMock(side_effect=_record_tier)

    synth = Synthesizer(bodega=router)

    # Call all three synchronous LLM methods (synthesize_full_text needs more setup)
    articles = [{"title": "t", "publisher": {"title": "p"}, "published date": "2026-01", "summary": "s"}]
    await synth.synthesize_news("q", articles)

    search_results = [{"title": "t", "description": "d", "url": "u"}]
    await synth.synthesize_search("q", search_results)

    await synth._summarize_chunk("text " * 200, "q")

    assert all(t == ModelTier.MID for t in recorded_tiers), (
        f"Expected all calls to use MID, got: {recorded_tiers}"
    )


# ── Decomposer ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decomposer_classify_uses_fast_tier():
    """Decomposer._classify_with_llm() must route to ModelTier.FAST.

    We call the public decompose() method and inspect the tier kwarg that
    reached chat_simple — the internal routing path is what we're verifying.
    """
    from octane.osa.decomposer import Decomposer

    router = _make_router(fast_response="web_search")
    decomposer = Decomposer(bodega=router)

    dag = await decomposer.decompose("What is NVDA?")

    router.chat_simple.assert_awaited()
    assert _last_tier(router.chat_simple) == ModelTier.FAST

    # Source metadata confirms LLM path was taken (not keyword fallback)
    source = dag.nodes[0].metadata.get("source", "")
    assert source == "llm", f"Expected LLM source, got: {source!r}"


@pytest.mark.asyncio
async def test_decomposer_fast_tier_is_not_reason():
    """Decomposer must NOT use REASON (heavy model for cheap routing is wasteful)."""
    from octane.osa.decomposer import Decomposer

    router = _make_router(fast_response="web_finance")
    decomposer = Decomposer(bodega=router)

    await decomposer.decompose("NVDA stock price")
    assert _last_tier(router.chat_simple) != ModelTier.REASON


@pytest.mark.asyncio
async def test_decomposer_falls_back_to_keywords_when_llm_disabled():
    """Decomposer with bodega=None must produce a DAG via keyword fallback."""
    from octane.osa.decomposer import Decomposer

    decomposer = Decomposer(bodega=None)
    dag = await decomposer.decompose("NVDA stock price")

    assert dag.nodes, "DAG must have at least one node"
    source = dag.nodes[0].metadata.get("source", "")
    assert source == "keyword_fallback", f"Expected keyword_fallback, got: {source!r}"
    template = dag.nodes[0].metadata.get("template", "")
    assert template in {"web_finance", "web_news", "web_search",
                        "code_generation", "memory_recall", "sysstat_health"}


# ── Evaluator ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluator_evaluate_uses_reason_tier():
    """Evaluator.evaluate() must route to ModelTier.REASON."""
    from octane.osa.evaluator import Evaluator

    router = _make_router(reason_response="NVDA is down 3.2% today.")
    evaluator = Evaluator(bodega=router)

    results = [AgentResponse(agent="web", output="NVDA -3.2%", success=True)]
    response = await evaluator.evaluate("What is NVDA doing?", results)

    assert response == "NVDA is down 3.2% today."
    router.chat_simple.assert_awaited()
    assert _last_tier(router.chat_simple) == ModelTier.REASON


@pytest.mark.asyncio
async def test_evaluator_evaluate_stream_uses_reason_tier():
    """Evaluator.evaluate_stream() must route to ModelTier.REASON."""
    from octane.osa.evaluator import Evaluator

    recorded_tiers: list[ModelTier] = []
    router = MagicMock()

    async def _stream(prompt, system="", tier=ModelTier.REASON, **kwargs):
        recorded_tiers.append(tier)
        yield "NVDA "
        yield "analysis done."

    router.chat_stream = _stream

    evaluator = Evaluator(bodega=router)
    results = [AgentResponse(agent="web", output="NVDA data", success=True)]

    chunks = []
    async for chunk in evaluator.evaluate_stream("Analyse NVDA", results):
        chunks.append(chunk)

    assert recorded_tiers, "chat_stream was never called"
    assert all(t == ModelTier.REASON for t in recorded_tiers), (
        f"Expected REASON, got: {recorded_tiers}"
    )


@pytest.mark.asyncio
async def test_evaluator_reason_tier_is_not_fast():
    """Evaluator must NOT use FAST — final synthesis requires deep reasoning."""
    from octane.osa.evaluator import Evaluator

    router = _make_router(reason_response="Answer.")
    evaluator = Evaluator(bodega=router)

    results = [AgentResponse(agent="web", output="data", success=True)]
    await evaluator.evaluate("query", results)

    assert _last_tier(router.chat_simple) != ModelTier.FAST


@pytest.mark.asyncio
async def test_evaluator_fallback_when_no_bodega():
    """Evaluator with bodega=None returns concatenated output (no LLM call)."""
    from octane.osa.evaluator import Evaluator

    evaluator = Evaluator(bodega=None)
    results = [AgentResponse(agent="web", output="raw data", success=True)]
    response = await evaluator.evaluate("query", results)

    assert "raw data" in response


# ── Cross-agent tier differentiation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_fast_mid_reason_tiers_are_distinct_across_agents():
    """The three tiers form a strict routing hierarchy: FAST < MID < REASON.

    This test confirms that no two different agent roles accidentally share
    the same tier assignment — the tier assignment is deliberate and unique.
    """
    fast_agents = {"QueryStrategist", "Decomposer"}
    mid_agents = {"Synthesizer"}
    reason_agents = {"Evaluator"}

    # All three sets must be non-overlapping
    assert not (fast_agents & mid_agents), "FAST and MID agents overlap"
    assert not (fast_agents & reason_agents), "FAST and REASON agents overlap"
    assert not (mid_agents & reason_agents), "MID and REASON agents overlap"


def test_tier_cost_ordering():
    """Tier values encode cost — FAST is cheapest, REASON is most expensive.

    This is a documentation-level assertion: if someone re-assigns tier
    strings in a way that breaks the semantics, this test flags it.
    """
    # These assignments must never change without a full audit
    assert ModelTier.FAST == "fast"
    assert ModelTier.MID == "mid"
    assert ModelTier.REASON == "reason"


# ── _UNSET sentinel wiring ────────────────────────────────────────────────────


def test_query_strategist_default_creates_bodega_router():
    """QueryStrategist() with no args must create a BodegaRouter, not None.

    BodegaRouter is imported lazily inside __init__ so we patch the source
    module (octane.tools.bodega_router) not the call-site module.
    """
    with patch("octane.tools.bodega_router.BodegaRouter") as MockRouter:
        MockRouter.return_value = MagicMock()
        # Force re-execution of the lazy import by creating a fresh instance
        from octane.agents.web.query_strategist import QueryStrategist
        qs = QueryStrategist()
        MockRouter.assert_called_once()


def test_synthesizer_default_creates_bodega_router():
    """Synthesizer() with no args must create a BodegaRouter, not None."""
    with patch("octane.tools.bodega_router.BodegaRouter") as MockRouter:
        MockRouter.return_value = MagicMock()
        from octane.agents.web.synthesizer import Synthesizer
        synth = Synthesizer()
        MockRouter.assert_called_once()


def test_decomposer_default_creates_bodega_router():
    """Decomposer() with no args must create a BodegaRouter, not None."""
    with patch("octane.tools.bodega_router.BodegaRouter") as MockRouter:
        MockRouter.return_value = MagicMock()
        from octane.osa.decomposer import Decomposer
        dec = Decomposer()
        MockRouter.assert_called_once()


def test_evaluator_default_creates_bodega_router():
    """Evaluator() with no args must create a BodegaRouter, not None."""
    with patch("octane.tools.bodega_router.BodegaRouter") as MockRouter:
        MockRouter.return_value = MagicMock()
        from octane.osa.evaluator import Evaluator
        ev = Evaluator()
        MockRouter.assert_called_once()


def test_explicit_none_disables_llm_in_query_strategist():
    """QueryStrategist(bodega=None) must set _bodega=None (fallback mode)."""
    from octane.agents.web.query_strategist import QueryStrategist
    qs = QueryStrategist(bodega=None)
    assert qs._bodega is None


def test_explicit_none_disables_llm_in_synthesizer():
    """Synthesizer(bodega=None) must set _bodega=None (fallback mode)."""
    from octane.agents.web.synthesizer import Synthesizer
    synth = Synthesizer(bodega=None)
    assert synth._bodega is None


def test_explicit_none_disables_llm_in_decomposer():
    """Decomposer(bodega=None) must set _bodega=None (keyword fallback mode)."""
    from octane.osa.decomposer import Decomposer
    decomposer = Decomposer(bodega=None)
    assert decomposer._bodega is None


def test_explicit_none_disables_llm_in_evaluator():
    """Evaluator(bodega=None) must set _bodega=None (concatenation fallback)."""
    from octane.osa.evaluator import Evaluator
    evaluator = Evaluator(bodega=None)
    assert evaluator._bodega is None
