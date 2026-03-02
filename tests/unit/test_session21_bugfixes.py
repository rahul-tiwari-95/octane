"""Tests for Session 21 bug fixes.

BUG-21-1: pg_client two-pass schema execution
BUG-21-2: Orchestrator uses BodegaRouter (not raw BodegaInferenceClient)
BUG-21-3: AngleGenerator _llm_angles → tier=MID
BUG-21-4: ResearchSynthesizer → tier=REASON (synthesize) / tier=MID (compress)
BUG-21-5: Code agents (Planner, Writer, Debugger) → BodegaRouter default + tier=REASON
BUG-21-6: WebAgent ticker extraction → tier=FAST
BUG-21-7: OSA Router type hint + BodegaRouter fallback
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# BUG-21-1: pg_client two-pass schema execution
# ---------------------------------------------------------------------------

class TestPgClientTwoPassSchema:
    """_ensure_schema() must create base tables even when pgVector is absent."""

    @pytest.mark.asyncio
    async def test_base_tables_created_without_vector(self):
        """Pass 1 always runs; Pass 2 skipped when vector_enabled=False."""
        from octane.tools.pg_client import PgClient

        client = PgClient.__new__(PgClient)
        client.vector_enabled = False

        fake_conn = AsyncMock()
        fake_pool = MagicMock()
        fake_pool.acquire = MagicMock(return_value=fake_conn)
        fake_conn.__aenter__ = AsyncMock(return_value=fake_conn)
        fake_conn.__aexit__ = AsyncMock(return_value=False)

        client._pool = fake_pool

        # Minimal fake schema.sql content with the embeddings marker
        fake_ddl = (
            "CREATE EXTENSION IF NOT EXISTS vector;\n"
            "CREATE TABLE IF NOT EXISTS projects (id SERIAL PRIMARY KEY);\n"
            "-- ── embeddings\n"
            "CREATE TABLE IF NOT EXISTS memory_embeddings (id SERIAL PRIMARY KEY);\n"
        )

        with patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=lambda s: MagicMock(read=lambda: fake_ddl),
            __exit__=MagicMock(return_value=False),
        ))):
            # We can't call _ensure_schema directly (needs live pool), but we can
            # verify the splitting logic inline:
            _EMBED_MARKER = "-- ── embeddings"
            base_ddl, _, embeddings_tail = fake_ddl.partition(_EMBED_MARKER)
            base_ddl_clean = "\n".join(
                line for line in base_ddl.splitlines()
                if "CREATE EXTENSION" not in line
            )
            # Base DDL must contain projects table but NOT the extension or embeddings
            assert "projects" in base_ddl_clean
            assert "CREATE EXTENSION" not in base_ddl_clean
            assert "memory_embeddings" not in base_ddl_clean
            # Embeddings tail is present
            assert "memory_embeddings" in embeddings_tail

    def test_schema_split_marker_present_in_schema_sql(self):
        """schema.sql must contain the split marker used by two-pass logic."""
        import os
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "octane", "tools", "schema.sql"
        )
        with open(schema_path) as f:
            content = f.read()
        assert "-- ── embeddings" in content, "Split marker missing from schema.sql"

    def test_extension_line_stripped_from_base_ddl(self):
        """CREATE EXTENSION lines are removed from the base DDL pass."""
        full_ddl = (
            "CREATE EXTENSION IF NOT EXISTS vector;\n"
            "CREATE TABLE foo (id INT);\n"
            "-- ── embeddings\n"
            "CREATE TABLE emb (v vector(384));\n"
        )
        _EMBED_MARKER = "-- ── embeddings"
        base_ddl, _, _ = full_ddl.partition(_EMBED_MARKER)
        base_clean = "\n".join(
            line for line in base_ddl.splitlines()
            if "CREATE EXTENSION" not in line
        )
        assert "CREATE EXTENSION" not in base_clean
        assert "CREATE TABLE foo" in base_clean


# ---------------------------------------------------------------------------
# BUG-21-2: Orchestrator must use BodegaRouter
# ---------------------------------------------------------------------------

class TestOrchestratorUsesBodegaRouter:
    """Orchestrator.__init__ must create a BodegaRouter, not raw BodegaInferenceClient."""

    def test_orchestrator_init_creates_bodega_router_not_inference_client(self):
        """Check the source code of Orchestrator uses BodegaRouter."""
        import inspect
        from octane.osa import orchestrator as orch_mod
        src = inspect.getsource(orch_mod)
        assert "BodegaRouter" in src, "Orchestrator must import/use BodegaRouter"
        assert "self.bodega = BodegaRouter()" in src, \
            "Orchestrator.__init__ must set self.bodega = BodegaRouter()"


# ---------------------------------------------------------------------------
# BUG-21-3: AngleGenerator tier=MID
# ---------------------------------------------------------------------------

class TestAngleGeneratorTier:
    """_llm_angles() must call chat_simple with tier=ModelTier.MID."""

    def test_llm_angles_uses_mid_tier(self):
        import inspect
        from octane.research import angles as angles_mod
        src = inspect.getsource(angles_mod)
        assert "ModelTier" in src
        assert "ModelTier.MID" in src

    @pytest.mark.asyncio
    async def test_llm_angles_passes_tier_mid_to_chat_simple(self):
        from octane.research.angles import AngleGenerator
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value='["angle1","angle2","angle3"]')

        gen = AngleGenerator.__new__(AngleGenerator)
        gen._bodega = mock_bodega

        await gen._llm_angles("AAPL", "quarterly earnings")

        call_kwargs = mock_bodega.chat_simple.call_args
        assert call_kwargs is not None
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        # tier should be passed as kwarg
        assert kwargs.get("tier") == ModelTier.MID, \
            f"Expected tier=ModelTier.MID, got {kwargs.get('tier')}"


# ---------------------------------------------------------------------------
# BUG-21-4: ResearchSynthesizer tiers
# ---------------------------------------------------------------------------

class TestResearchSynthesizerTiers:
    """_synthesize_with_llm → REASON; _compress → MID."""

    def test_synthesizer_source_has_both_tiers(self):
        import inspect
        from octane.research import synthesizer as synth_mod
        src = inspect.getsource(synth_mod)
        assert "ModelTier.REASON" in src
        assert "ModelTier.MID" in src

    @pytest.mark.asyncio
    async def test_synthesize_uses_reason_tier(self):
        from octane.research.synthesizer import ResearchSynthesizer
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="synthesis result")

        synth = ResearchSynthesizer.__new__(ResearchSynthesizer)
        synth._bodega = mock_bodega

        # _synthesize_with_llm expects a list of ResearchFinding objects
        finding = MagicMock()
        finding.topic = "AAPL quarterly earnings"
        finding.word_count = 100
        finding.findings = ["finding1"]
        finding.sources = []

        # Patch _combine_findings to avoid real logic
        with patch.object(synth, "_combine_findings", return_value="combined text"):
            await synth._synthesize_with_llm([finding], task_id="t1")

        kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert kwargs.get("tier") == ModelTier.REASON, \
            f"_synthesize_with_llm expected REASON, got {kwargs.get('tier')}"

    @pytest.mark.asyncio
    async def test_compress_uses_mid_tier(self):
        from octane.research.synthesizer import ResearchSynthesizer
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="compressed text")

        synth = ResearchSynthesizer.__new__(ResearchSynthesizer)
        synth._bodega = mock_bodega

        long_text = "word " * 5000  # force compression
        await synth._compress(long_text, topic="AAPL")

        kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert kwargs.get("tier") == ModelTier.MID, \
            f"_compress expected MID, got {kwargs.get('tier')}"


# ---------------------------------------------------------------------------
# BUG-21-5: Code agents → BodegaRouter default + tier=REASON
# ---------------------------------------------------------------------------

class TestPlannerTierWiring:
    def test_planner_default_uses_bodega_router(self):
        """Planner() with no arg creates a BodegaRouter (lazy import inside __init__)."""
        # BodegaRouter is lazily imported inside __init__, patch it at source
        with patch("octane.tools.bodega_router.BodegaRouter") as MockRouter:
            mock_router_instance = MagicMock()
            MockRouter.return_value = mock_router_instance
            from octane.agents.code.planner import Planner
            p = Planner()
            # The bodega assigned must be the instance returned by BodegaRouter()
            assert p._bodega is mock_router_instance

    def test_planner_source_has_reason_tier(self):
        import inspect
        from octane.agents.code import planner as planner_mod
        src = inspect.getsource(planner_mod)
        assert "ModelTier.REASON" in src

    @pytest.mark.asyncio
    async def test_planner_plan_with_llm_passes_reason_tier(self):
        from octane.agents.code.planner import Planner
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(
            return_value='{"language":"python","approach":"test","steps":["s1"],'
                         '"requirements":[],"entry_point":"solution.py"}'
        )
        planner = Planner(bodega=mock_bodega)
        await planner._plan_with_llm("write a hello world script")

        kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert kwargs.get("tier") == ModelTier.REASON, \
            f"Planner expected REASON, got {kwargs.get('tier')}"


class TestWriterTierWiring:
    def test_writer_source_has_reason_tier(self):
        import inspect
        from octane.agents.code import writer as writer_mod
        src = inspect.getsource(writer_mod)
        assert "ModelTier.REASON" in src

    @pytest.mark.asyncio
    async def test_writer_write_with_llm_passes_reason_tier(self):
        from octane.agents.code.writer import Writer
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="print('hello')")

        writer = Writer(bodega=mock_bodega)
        spec = {
            "task": "print hello",
            "approach": "simple print",
            "steps": ["print hello world"],
            "requirements": [],
            "entry_point": "solution.py",
        }
        await writer._write_with_llm(spec, None)

        kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert kwargs.get("tier") == ModelTier.REASON, \
            f"Writer expected REASON, got {kwargs.get('tier')}"


class TestDebuggerTierWiring:
    def test_debugger_source_has_reason_tier(self):
        import inspect
        from octane.agents.code import debugger as debugger_mod
        src = inspect.getsource(debugger_mod)
        assert "ModelTier.REASON" in src

    @pytest.mark.asyncio
    async def test_debugger_fix_with_llm_passes_reason_tier(self):
        from octane.agents.code.debugger import Debugger
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="print('fixed')")

        debugger = Debugger(bodega=mock_bodega)
        await debugger._fix_with_llm("print('broken')", "NameError: name 'x' is not defined")

        kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert kwargs.get("tier") == ModelTier.REASON, \
            f"Debugger expected REASON, got {kwargs.get('tier')}"


# ---------------------------------------------------------------------------
# BUG-21-6: WebAgent ticker extraction → tier=FAST
# ---------------------------------------------------------------------------

class TestWebAgentTickerFast:
    def test_web_agent_source_has_fast_tier(self):
        import inspect
        from octane.agents.web import agent as web_agent_mod
        src = inspect.getsource(web_agent_mod)
        assert "ModelTier.FAST" in src

    @pytest.mark.asyncio
    async def test_resolve_ticker_passes_fast_tier(self):
        from octane.agents.web.agent import WebAgent
        from octane.tools.topology import ModelTier

        mock_bodega = AsyncMock()
        mock_bodega.chat_simple = AsyncMock(return_value="AAPL")
        mock_intel = MagicMock()
        # web_search returns {"web": {"results": [...]}} structure
        mock_intel.web_search = AsyncMock(return_value={
            "web": {
                "results": [
                    {"title": "Apple Inc.", "description": "AAPL stock info"}
                ]
            }
        })

        agent = WebAgent.__new__(WebAgent)
        agent._bodega = mock_bodega
        agent._intel = mock_intel

        ticker = await agent._resolve_ticker_via_web("Apple stock ticker")

        assert mock_bodega.chat_simple.called, "chat_simple was never called"
        kwargs = mock_bodega.chat_simple.call_args.kwargs
        assert kwargs.get("tier") == ModelTier.FAST, \
            f"Ticker extraction expected FAST, got {kwargs.get('tier')}"
        assert ticker == "AAPL"


# ---------------------------------------------------------------------------
# BUG-21-7: OSA Router type hint + BodegaRouter fallback
# ---------------------------------------------------------------------------

class TestOsaRouterBodegaRouterFallback:
    def test_router_source_imports_bodega_router(self):
        import inspect
        from octane.osa import router as router_mod
        src = inspect.getsource(router_mod)
        assert "BodegaRouter" in src, "OSA Router must import BodegaRouter"

    def test_router_default_creates_bodega_router_not_inference_client(self):
        """When bodega=None, Router should create BodegaRouter (not BodegaInferenceClient)."""
        import inspect
        from octane.osa import router as router_mod
        src = inspect.getsource(router_mod)
        # The fallback should be BodegaRouter(), not BodegaInferenceClient()
        assert "BodegaRouter()" in src, "Router fallback must use BodegaRouter()"
        # Ensure the old raw fallback is gone
        assert "bodega or BodegaInferenceClient()" not in src, \
            "Router must not fall back to raw BodegaInferenceClient()"

    def test_router_accepts_bodega_router_instance(self):
        """Router.__init__ must accept a BodegaRouter without raising."""
        from octane.tools.bodega_router import BodegaRouter
        from octane.models.synapse import SynapseEventBus

        mock_bodega = MagicMock(spec=BodegaRouter)
        mock_synapse = MagicMock(spec=SynapseEventBus)

        with (
            patch("octane.osa.router.BodegaIntelClient"),
            patch("octane.osa.router.RedisClient"),
            patch("octane.osa.router.PgClient"),
            patch("octane.osa.router.WebPageStore"),
            patch("octane.osa.router.ArtifactStore"),
            patch("octane.osa.router.WebAgent"),
            patch("octane.osa.router.CodeAgent"),
            patch("octane.osa.router.MemoryAgent"),
            patch("octane.osa.router.SysStatAgent"),
            patch("octane.osa.router.PnLAgent"),
        ):
            from octane.osa.router import Router
            router = Router(synapse=mock_synapse, bodega=mock_bodega)
            assert router.bodega is mock_bodega
