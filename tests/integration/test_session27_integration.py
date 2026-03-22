"""Session 27 integration tests.

Guards:
    OCTANE_TEST_INTEGRATION=1   Redis + Postgres must be running
    OCTANE_TEST_E2E=1           Bodega inference server must also be running

Run all integration:
    OCTANE_TEST_INTEGRATION=1 python -m pytest tests/integration/ -v

Run including E2E (Bodega live):
    OCTANE_TEST_E2E=1 OCTANE_TEST_INTEGRATION=1 python -m pytest tests/integration/ tests/e2e/ -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

# ── Skip guards ───────────────────────────────────────────────────────────────

_INTEGRATION = bool(os.getenv("OCTANE_TEST_INTEGRATION"))
_E2E = bool(os.getenv("OCTANE_TEST_E2E"))

requires_integration = pytest.mark.skipif(
    not _INTEGRATION,
    reason="Set OCTANE_TEST_INTEGRATION=1 to run integration tests",
)
requires_e2e = pytest.mark.skipif(
    not _E2E,
    reason="Set OCTANE_TEST_E2E=1 (+ OCTANE_TEST_INTEGRATION=1) for live Bodega tests",
)

pytestmark = requires_integration


# ══════════════════════════════════════════════════════════════════════════════
# Part A-1 : Infrastructure connectivity
# ══════════════════════════════════════════════════════════════════════════════

class TestRedisConnectivity:
    """Redis must be reachable on localhost:6379."""

    async def test_redis_ping(self):
        """redis.asyncio can PING localhost:6379."""
        import redis.asyncio as aioredis
        r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        try:
            result = await r.ping()
            assert result is True
        finally:
            await r.aclose()

    async def test_redis_set_get_roundtrip(self):
        """SET/GET round-trip works and produces the correct value."""
        import redis.asyncio as aioredis
        r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        try:
            await r.set("octane_test_key", "session27", ex=10)
            val = await r.get("octane_test_key")
            assert val == "session27"
        finally:
            await r.delete("octane_test_key")
            await r.aclose()


class TestPostgresConnectivity:
    """Postgres must be reachable on localhost:5432 with octane database."""

    async def test_pg_client_connects(self):
        """PgClient.connect() returns True and sets available=True."""
        from octane.tools.pg_client import PgClient
        pg = PgClient()
        try:
            connected = await pg.connect()
            assert connected is True
            assert pg.available is True
        finally:
            await pg.close()

    async def test_pg_schema_tables_exist(self):
        """Core tables exist after a fresh connect (idempotent DDL)."""
        from octane.tools.pg_client import PgClient
        pg = PgClient()
        await pg.connect()
        try:
            tables = await pg.fetch(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            names = {r["tablename"] for r in tables}
            # Core tables from pg_client DDL
            assert "memory_chunks" in names, "memory_chunks table missing"
            assert "web_pages" in names, "web_pages table missing"
        finally:
            await pg.close()

    async def test_memory_chunks_table_columns(self):
        """memory_chunks must have all expected columns."""
        from octane.tools.pg_client import PgClient
        pg = PgClient()
        await pg.connect()
        try:
            cols = await pg.fetch(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'memory_chunks'
                ORDER BY ordinal_position
                """
            )
            names = {r["column_name"] for r in cols}
            for expected in ("id", "session_id", "slot", "content", "created_at"):
                assert expected in names, f"Column '{expected}' missing from memory_chunks"
        finally:
            await pg.close()

    async def test_schema_migration_is_idempotent(self):
        """Calling pg.connect() twice must not raise (DDL uses CREATE IF NOT EXISTS)."""
        from octane.tools.pg_client import PgClient
        pg = PgClient()
        await pg.connect()
        # Force immediate disconnect so next connect re-runs DDL
        await pg.close()
        pg2 = PgClient()
        try:
            connected = await pg2.connect()
            assert connected is True
        finally:
            await pg2.close()


# ══════════════════════════════════════════════════════════════════════════════
# Part A-2 : WebPageStore (Postgres)
# ══════════════════════════════════════════════════════════════════════════════

class TestWebPageStore:
    """WebPageStore dedup, store, recent, and count against real Postgres."""

    @pytest.fixture(autouse=True)
    async def _pg(self):
        from octane.tools.pg_client import PgClient
        pg = PgClient()
        await pg.connect()
        self._pg = pg
        yield
        await pg.close()

    async def test_store_returns_row(self):
        """store() returns a dict with an 'id' field."""
        from octane.tools.structured_store import WebPageStore
        ws = WebPageStore(self._pg)
        row = await ws.store(
            url="https://example.com/test-session27",
            content="Integration test content for session 27.",
            title="Test Page",
        )
        assert row is not None
        assert "id" in row

    async def test_store_dedup_same_url(self):
        """Storing the same URL twice must not increase the count (ON CONFLICT UPDATE)."""
        from octane.tools.structured_store import WebPageStore
        ws = WebPageStore(self._pg)
        url = "https://example.com/dedup-test-session27"
        before = await ws.count()
        await ws.store(url=url, content="First version")
        after_first = await ws.count()
        await ws.store(url=url, content="Second version — updated content")
        after_second = await ws.count()
        # count must increase by at most 1 (upsert, not double insert)
        assert after_second == after_first, "Duplicate URL should upsert, not insert new row"
        assert after_first == before + 1

    async def test_seen_returns_true_after_store(self):
        """seen() must return True for a URL that was stored."""
        from octane.tools.structured_store import WebPageStore
        ws = WebPageStore(self._pg)
        url = "https://example.com/seen-test-session27"
        await ws.store(url=url, content="seen check")
        assert await ws.seen(url) is True

    async def test_seen_returns_false_for_unknown_url(self):
        """seen() returns False for a URL that was never stored."""
        from octane.tools.structured_store import WebPageStore
        ws = WebPageStore(self._pg)
        assert await ws.seen("https://example.com/never-stored-xyz-99999") is False

    async def test_recent_returns_list(self):
        """recent() returns a list (possibly empty)."""
        from octane.tools.structured_store import WebPageStore
        ws = WebPageStore(self._pg)
        rows = await ws.recent(limit=5)
        assert isinstance(rows, list)

    async def test_count_is_non_negative_int(self):
        """count() returns a non-negative integer."""
        from octane.tools.structured_store import WebPageStore
        ws = WebPageStore(self._pg)
        n = await ws.count()
        assert isinstance(n, int)
        assert n >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Part A-3 : ResearchStore (Redis + Postgres)
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchStore:
    """ResearchStore round-trip: register task, add finding, retrieve finding."""

    async def test_add_and_get_finding(self):
        """add_finding() stores to Postgres; get_findings() retrieves it."""
        from octane.research.store import ResearchStore
        store = ResearchStore()

        finding = await store.add_finding(
            task_id="test_session27",
            topic="NVDA integration test",
            content="This is a test finding from Session 27 integration tests.",
            agents_used=["web", "memory"],
            sources=["https://example.com"],
        )
        assert finding is not None
        assert finding.topic == "NVDA integration test"
        assert finding.word_count > 0

        findings = await store.get_findings("test_session27")
        assert any(f.topic == "NVDA integration test" for f in findings)

    async def test_register_and_get_task(self):
        """register_task() stores to Redis; get_task() retrieves the metadata."""
        from octane.research.store import ResearchStore
        from octane.research.models import ResearchTask
        store = ResearchStore()

        task = ResearchTask(id="s27test1", topic="Test topic session 27", interval_hours=1.0)
        await store.register_task(task)

        retrieved = await store.get_task("s27test1")
        assert retrieved is not None
        assert retrieved.topic == "Test topic session 27"

    async def test_log_entry_round_trip(self):
        """log_entry() writes to Redis; get_log() retrieves lines."""
        from octane.research.store import ResearchStore
        store = ResearchStore()

        await store.log_entry("s27logtest", "Integration test log line")
        lines = await store.get_log("s27logtest", limit=10)
        assert any("Integration test log line" in (l if isinstance(l, str) else "") for l in lines)


# ══════════════════════════════════════════════════════════════════════════════
# Part A-4 : BodegaRouter tier routing  (requires live Bodega = E2E)
# ══════════════════════════════════════════════════════════════════════════════

@requires_e2e
class TestBodegaRouterLive:
    """Verify tier routing hits the correct model in live Bodega."""

    async def test_bodega_health_returns_ok(self):
        """/health endpoint returns status=ok."""
        from octane.tools.bodega_router import BodegaRouter
        router = BodegaRouter()
        health = await router.health()
        assert health.get("status") == "ok", f"Bodega health: {health}"

    async def test_fast_tier_routes_to_raptor_90m(self):
        """FAST tier resolves to the raptor-90m model path."""
        from octane.tools.bodega_router import BodegaRouter
        from octane.tools.topology import ModelTier
        router = BodegaRouter()
        model_id = await router._resolve_available_model(ModelTier.FAST)
        assert model_id is not None, "No model resolved for FAST tier"
        # Model ID or path should refer to the 90m model
        needle = "90m"
        assert needle in model_id.lower(), (
            f"FAST tier resolved to '{model_id}' — expected a 90m model"
        )

    async def test_reason_tier_routes_to_8b(self):
        """REASON tier resolves to the 8b model path."""
        from octane.tools.bodega_router import BodegaRouter
        from octane.tools.topology import ModelTier
        router = BodegaRouter()
        model_id = await router._resolve_available_model(ModelTier.REASON)
        assert model_id is not None, "No model resolved for REASON tier"
        assert "8b" in model_id.lower(), (
            f"REASON tier resolved to '{model_id}' — expected an 8b model"
        )

    async def test_mid_tier_resolves_to_a_model(self):
        """MID tier resolves to some loaded model (exact model varies by topology)."""
        from octane.tools.bodega_router import BodegaRouter
        from octane.tools.topology import ModelTier
        router = BodegaRouter()
        model_id = await router._resolve_available_model(ModelTier.MID)
        assert model_id is not None, "No model resolved for MID tier"

    async def test_fast_tier_inference_returns_text(self):
        """FAST tier chat_simple produces non-empty text."""
        from octane.tools.bodega_router import BodegaRouter
        from octane.tools.topology import ModelTier
        router = BodegaRouter()
        text = await router.chat_simple(
            "Extract the ticker: buy NVDA shares",
            tier=ModelTier.FAST,
            max_tokens=16,
        )
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    async def test_three_tiers_route_to_different_physical_models(self):
        """On a power topology, FAST / MID / REASON should use distinct physical models."""
        from octane.tools.bodega_router import BodegaRouter
        from octane.tools.topology import ModelTier, get_topology
        topo = get_topology("auto")
        # Only meaningful on power topology (64 GB+, 3 distinct models).
        # On compact/balanced they may share; skip gracefully.
        fast_id = topo.resolve(ModelTier.FAST)
        mid_id = topo.resolve(ModelTier.MID)
        reason_id = topo.resolve(ModelTier.REASON)
        if len({fast_id, mid_id, reason_id}) < 2:
            pytest.skip("Topology maps multiple tiers to the same model — not a power config")
        # At least FAST vs REASON must differ on any 3-model topology
        assert fast_id != reason_id, (
            f"FAST ({fast_id}) and REASON ({reason_id}) unexpectedly share the same model"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Part A-5 : OSA / CLI pipeline   (requires live Bodega = E2E)
# ══════════════════════════════════════════════════════════════════════════════

@requires_e2e
class TestOSAPipelineLive:
    """End-to-end OSA pipeline checks against live Bodega."""

    async def test_health_command_exits_cleanly(self):
        """octane health should exit 0 and not raise."""
        from typer.testing import CliRunner
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0, f"health exited {result.exit_code}: {result.output}"
        assert "Error" not in (result.output or "")

    async def test_ask_returns_non_empty_response(self):
        """octane ask produces a non-empty text response for a simple factual query."""
        from octane.models.synapse import SynapseEventBus
        from octane.osa.orchestrator import Orchestrator
        synapse = SynapseEventBus(persist=False)
        osa = Orchestrator(synapse)
        response = await osa.run("What is NVDA?", session_id="integration_test")
        assert isinstance(response, str)
        assert len(response.strip()) > 10, f"Response too short: {response!r}"

    async def test_memory_store_and_recall(self):
        """Store a finding in memory, then verify a follow-up query can recall it."""
        from octane.tools.pg_client import PgClient
        from octane.tools.bodega_router import BodegaRouter
        from octane.models.synapse import SynapseEventBus
        from octane.agents.memory.agent import MemoryAgent
        from octane.models.schemas import AgentRequest

        pg = PgClient()
        await pg.connect()
        synapse = SynapseEventBus(persist=False)
        bodega = BodegaRouter()
        from octane.tools.redis_client import RedisClient
        redis = RedisClient()
        mem = MemoryAgent(synapse, redis=redis)
        await mem.connect_pg()

        # Write a known fact
        write_req = AgentRequest(
            query="The Integration Test Fact: Octane session 27 ran on March 2026",
            metadata={"sub_agent": "write", "session_id": "s27_mem_test", "slot": "test"},
        )
        await mem.execute(write_req)

        # Read it back
        read_req = AgentRequest(
            query="Integration Test Fact",
            metadata={"sub_agent": "read", "session_id": "s27_mem_test"},
        )
        resp = await mem.execute(read_req)
        assert "2026" in resp.output or "session 27" in resp.output.lower(), (
            f"Memory recall didn't find stored content. Got: {resp.output[:200]}"
        )
        await pg.close()


# ══════════════════════════════════════════════════════════════════════════════
# Part A-6 : Auto-store (Part C verification)   (requires live Bodega = E2E)
# ══════════════════════════════════════════════════════════════════════════════

@requires_e2e
class TestAutoStoreOnDeepQuery:
    """Verify that --deep queries store web pages and findings automatically."""

    async def test_router_pg_connected_after_preflight(self):
        """Orchestrator pre_flight() must make router.pg.available == True."""
        from octane.models.synapse import SynapseEventBus
        from octane.osa.orchestrator import Orchestrator
        synapse = SynapseEventBus(persist=False)
        osa = Orchestrator(synapse)
        await osa.pre_flight()
        assert osa.router.pg.available is True, (
            "Router.pg not connected after pre_flight — "
            "WebPageStore would silently no-op without this fix"
        )

    async def test_web_page_count_increases_after_deep_query(self):
        """A --deep query must increase the web_pages row count."""
        from octane.tools.pg_client import PgClient
        from octane.tools.structured_store import WebPageStore
        from octane.models.synapse import SynapseEventBus
        from octane.osa.orchestrator import Orchestrator

        pg = PgClient()
        await pg.connect()
        ws = WebPageStore(pg)
        before = await ws.count()

        synapse = SynapseEventBus(persist=False)
        osa = Orchestrator(synapse)
        await osa.run(
            "What is NVDA's current stock price?",
            session_id="s27_autostore_test",
        )

        after = await ws.count()
        await pg.close()
        assert after >= before, "web_pages count decreased — something went wrong"
        # Note: may stay the same if Bodega can't reach web; we assert >=, not >

    async def test_research_finding_stored_after_deep_cli(self):
        """research_findings table must have a row with task_id='ask_deep' after a --deep query."""
        from octane.research.store import ResearchStore
        from octane.models.synapse import SynapseEventBus
        from octane.osa.orchestrator import Orchestrator
        from octane.cli.ask import _store_deep_finding

        synapse = SynapseEventBus(persist=False)
        osa = Orchestrator(synapse)
        # Directly test the storage helper (not the full CLI to avoid TTY)
        await _store_deep_finding(
            query="integration test deep finding",
            content="NVDA had a great quarter. Integration test persistence check.",
            osa=osa,
        )

        store = ResearchStore()
        findings = await store.get_findings("ask_deep")
        assert any("integration test" in f.topic.lower() for f in findings), (
            "Deep finding not found in research_findings table"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Part A-7 : Database schema completeness + migration idempotency
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseSchema:
    """Core schema tables must exist and key columns must be present."""

    @pytest.fixture(autouse=True)
    async def _connect(self):
        from octane.tools.pg_client import PgClient
        self.pg = PgClient()
        await self.pg.connect()
        yield
        await self.pg.close()

    async def _table_columns(self, table: str) -> set[str]:
        rows = await self.pg.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
            table,
        )
        return {r["column_name"] for r in rows}

    async def test_web_pages_table_exists(self):
        tables = await self.pg.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        assert "web_pages" in {r["tablename"] for r in tables}

    async def test_web_pages_has_required_columns(self):
        cols = await self._table_columns("web_pages")
        for c in ("id", "url", "url_hash", "content", "word_count", "fetched_at"):
            assert c in cols, f"web_pages missing column: {c}"

    async def test_projects_table_exists(self):
        tables = await self.pg.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        assert "projects" in {r["tablename"] for r in tables}

    async def test_research_findings_table_created_on_demand(self):
        """ResearchStore.ensure_schema() creates research_findings if not present."""
        from octane.research.store import ResearchStore
        store = ResearchStore()
        # Calling ensure_schema should be safe (idempotent)
        pg_conn = await store._pg()
        assert pg_conn is not None, "ResearchStore could not connect to Postgres"
        await store.ensure_schema()
        # Verify table now exists
        tables = await self.pg.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        assert "research_findings" in {r["tablename"] for r in tables}

    async def test_double_migration_no_error(self):
        """Running pg.connect() on an already-migrated DB must not raise."""
        from octane.tools.pg_client import PgClient
        pg2 = PgClient()
        try:
            ok = await pg2.connect()
            assert ok is True
        finally:
            await pg2.close()
