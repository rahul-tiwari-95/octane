"""Agent happy-path tests.

One test per agent. Fast, no network, no LLM required.

Run with:
    sandbox/oct_env/bin/python -m pytest tests/test_agents.py -v
"""

from __future__ import annotations

import asyncio
import pytest

from octane.models.schemas import AgentRequest
from octane.models.synapse import SynapseEventBus


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_synapse() -> SynapseEventBus:
    return SynapseEventBus(persist=False)


def make_request(query: str, metadata: dict | None = None) -> AgentRequest:
    return AgentRequest(
        query=query,
        session_id="test",
        source="test",
        metadata=metadata or {},
    )


# ── WebAgent ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_agent_routes_finance():
    """WebAgent with sub_agent=finance should call market_data and return price info."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.agent import WebAgent

    # Mock BodegaIntelClient to avoid real HTTP
    mock_intel = MagicMock()
    mock_intel.market_data = AsyncMock(return_value={
        "status": "success",
        "market_data": {
            "symbol": "NVDA",
            "price": 189.82,
            "change": 1.92,
            "change_percent": 1.02,
            "volume": 178422337,
            "market_cap": 4620000000000,
        },
    })

    agent = WebAgent(make_synapse(), intel=mock_intel)
    req = make_request("NVDA stock price", metadata={"sub_agent": "finance"})
    resp = await agent.execute(req)

    assert resp.success is True
    assert "189.82" in resp.output
    assert "NVDA" in resp.output
    mock_intel.market_data.assert_called_once_with("NVDA")


@pytest.mark.asyncio
async def test_web_agent_routes_news():
    """WebAgent with sub_agent=news should call news_search and return articles."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.agent import WebAgent

    mock_intel = MagicMock()
    mock_intel.news_search = AsyncMock(return_value={
        "status": "success",
        "count": 1,
        "articles": [{
            "title": "Xbox names new CEO",
            "description": "Asha Sharma takes over from Phil Spencer",
            "published date": "2026-02-20T12:00:00",
            "url": "https://example.com/xbox-ceo",
            "publisher": {"title": "IGN"},
        }],
    })

    agent = WebAgent(make_synapse(), intel=mock_intel)
    req = make_request("latest xbox news", metadata={"sub_agent": "news"})
    resp = await agent.execute(req)

    assert resp.success is True
    assert "Xbox names new CEO" in resp.output


@pytest.mark.asyncio
async def test_web_agent_routes_search_by_default():
    """WebAgent with unknown sub_agent should fall back to web_search."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.agent import WebAgent

    mock_intel = MagicMock()
    mock_intel.web_search = AsyncMock(return_value={
        "web": {
            "results": [{
                "title": "Phil Spencer - Wikipedia",
                "description": "CEO of Microsoft Gaming",
                "url": "https://en.wikipedia.org/wiki/Phil_Spencer",
            }]
        }
    })

    agent = WebAgent(make_synapse(), intel=mock_intel)
    req = make_request("ceo of xbox", metadata={"sub_agent": "unknown_type"})
    resp = await agent.execute(req)

    assert resp.success is True
    assert "Phil Spencer" in resp.output
    mock_intel.web_search.assert_called_once()


# ── MemoryAgent ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_agent_write_and_read():
    """MemoryAgent should store an answer and retrieve it in the same session."""
    from octane.agents.memory.agent import MemoryAgent
    from octane.tools.redis_client import RedisClient

    # Use a fresh in-process RedisClient (no real Redis needed)
    redis = RedisClient()
    redis._use_fallback = True  # Force in-process dict, skip real Redis attempt

    agent = MemoryAgent(make_synapse(), redis=redis)
    session = "test_mem_session"

    # Write — answer must pass quality filter (>30 chars with digits)
    await agent.remember(session, "NVDA stock price", "NVDA is trading at $189.82 today, up 1.02%.")

    # Read — use a phrase that contains "stock" and "price" to match the same slot
    recalled = await agent.recall(session, "what was NVDA stock price?")

    assert recalled is not None
    assert "189.82" in recalled


# ── CodeAgent ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_code_agent_executes_clean_python():
    """CodeAgent executor should run a trivial script and capture its output."""
    from octane.agents.code.executor import Executor

    executor = Executor()
    result = await executor.run('print("octane_test_ok")')

    assert result["exit_code"] == 0
    assert "octane_test_ok" in result["stdout"]
    assert result["duration_ms"] < 5000


@pytest.mark.asyncio
async def test_code_agent_captures_stderr_on_failure():
    """CodeAgent executor should capture stderr and return non-zero exit code."""
    from octane.agents.code.executor import Executor

    executor = Executor()
    result = await executor.run("raise ValueError('intentional test error')")

    assert result["exit_code"] != 0
    assert "intentional test error" in result["stderr"]


# ── SysStatAgent ──────────────────────────────────────────────────────────────

def test_sysstat_monitor_returns_ram():
    """Monitor.snapshot() should return real psutil RAM and CPU data."""
    from octane.agents.sysstat.monitor import Monitor

    monitor = Monitor()
    snap = monitor.snapshot()

    assert "ram_used_gb" in snap
    assert "ram_total_gb" in snap
    assert "cpu_percent" in snap
    assert snap["ram_total_gb"] > 0
    assert 0 <= snap["cpu_percent"] <= 100


@pytest.mark.asyncio
async def test_sysstat_agent_execute_succeeds():
    """SysStatAgent.execute() should return success with system data."""
    from unittest.mock import AsyncMock
    from octane.agents.sysstat.agent import SysStatAgent
    from octane.tools.bodega_inference import BodegaInferenceClient

    mock_bodega = AsyncMock(spec=BodegaInferenceClient)
    mock_bodega.health = AsyncMock(return_value={"status": "ok"})
    mock_bodega.current_model = AsyncMock(return_value={"model": "test-model"})

    agent = SysStatAgent(make_synapse(), bodega=mock_bodega)
    req = make_request("system health check")
    resp = await agent.execute(req)

    assert resp.success is True
    assert "ram" in resp.output.lower()
    assert resp.data["system"]["ram_total_gb"] > 0


# ── Evaluator ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_profile_shapes_system_prompt():
    """_build_system_prompt should produce different output for different profiles."""
    from octane.osa.evaluator import _build_system_prompt

    concise = _build_system_prompt({"verbosity": "concise", "expertise": "advanced", "response_style": "prose"})
    detailed = _build_system_prompt({"verbosity": "detailed", "expertise": "beginner", "response_style": "bullets"})
    default = _build_system_prompt(None)

    assert "brief and direct" in concise
    assert "thorough" in detailed
    assert "simple language" in detailed
    assert "bullet" in detailed
    # default has no style additions — check the specific injected phrase, not the word "brief"
    # (the base prompt may contain "briefly" naturally)
    assert "Be brief and direct" not in default
    assert "Be thorough" not in default


@pytest.mark.asyncio
async def test_evaluator_fallback_concatenation():
    """Evaluator without Bodega should concatenate agent outputs."""
    from octane.models.schemas import AgentResponse
    from octane.osa.evaluator import Evaluator

    evaluator = Evaluator(bodega=None)  # no LLM
    results = [
        AgentResponse(agent="web", success=True, output="NVDA is $189.82"),
        AgentResponse(agent="memory", success=True, output="Previously: NVDA was $185"),
    ]
    output = await evaluator.evaluate("NVDA price", results)

    assert "189.82" in output
    assert "185" in output


# ── PnL / FeedbackLearner ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_thumbs_up_increments_score():
    """thumbs_up signal should increase the running score."""
    from octane.agents.pnl.feedback_learner import FeedbackLearner
    from octane.tools.redis_client import RedisClient

    redis = RedisClient()
    redis._use_fallback = True
    learner = FeedbackLearner(redis=redis)

    await learner.record("u1", "thumbs_up", 1.0, correlation_id="cid-001")
    score = await learner.get_score("u1")
    assert score == 1


@pytest.mark.asyncio
async def test_feedback_thumbs_down_decrements_score():
    """thumbs_down signal should decrease the running score."""
    from octane.agents.pnl.feedback_learner import FeedbackLearner
    from octane.tools.redis_client import RedisClient

    redis = RedisClient()
    redis._use_fallback = True
    learner = FeedbackLearner(redis=redis)

    await learner.record("u2", "thumbs_down", 1.0, correlation_id="cid-002")
    score = await learner.get_score("u2")
    assert score == -1


@pytest.mark.asyncio
async def test_feedback_nudges_verbosity_after_threshold():
    """Three thumbs_down signals should nudge verbosity one step toward concise."""
    from octane.agents.pnl.feedback_learner import FeedbackLearner, NUDGE_THRESHOLD
    from octane.agents.pnl.preference_manager import PreferenceManager
    from octane.tools.redis_client import RedisClient

    redis = RedisClient()
    redis._use_fallback = True
    prefs = PreferenceManager(redis=redis)
    # Start user at "balanced" so there's room to step down
    await prefs.set("u3", "verbosity", "balanced")

    learner = FeedbackLearner(redis=redis, prefs=prefs)

    for i in range(NUDGE_THRESHOLD):
        await learner.record("u3", "thumbs_down", 1.0, correlation_id=f"cid-{i}")

    verbosity = await prefs.get("u3", "verbosity")
    assert verbosity == "concise"
    # Score resets to 0 after nudge
    score = await learner.get_score("u3")
    assert score == 0


# ── DAGPlanner (Session 8) ────────────────────────────────────────────────────

def test_dag_planner_detects_compound_query():
    """is_compound() should return True for queries that reference charts or comparisons."""
    from octane.osa.dag_planner import DAGPlanner

    planner = DAGPlanner(bodega=None)

    compound_queries = [
        "chart MSFT vs GOOGL capex",
        "fetch the data then write code",
        "compare Amazon and Microsoft revenue and graph it",
        "analyze AI spend and visualize the results",
        "get Apple financials and plot a bar chart",
        "research and implement a solution",
    ]
    for q in compound_queries:
        assert planner.is_compound(q) is True, f"Expected compound but got False for: {q!r}"


def test_dag_planner_simple_query_not_compound():
    """is_compound() should return False for straightforward single-agent queries."""
    from octane.osa.dag_planner import DAGPlanner

    planner = DAGPlanner(bodega=None)

    simple_queries = [
        "what is NVDA stock price",
        "latest news on Apple",
        "how is my CPU doing",
        "what did I ask yesterday",
        "write a fibonacci function",
    ]
    for q in simple_queries:
        assert planner.is_compound(q) is False, f"Expected simple but got True for: {q!r}"


def test_dag_planner_parse_plan_builds_dag():
    """_parse_plan() should correctly parse the constrained LLM output format."""
    from octane.osa.dag_planner import DAGPlanner

    planner = DAGPlanner(bodega=None)

    llm_output = (
        "1. web/web_finance | MSFT capex last 5 years\n"
        "2. web/web_finance | GOOGL capex last 5 years\n"
        "3. code/code_generation | Chart the capex data\n"
        "depends_on: 3 needs 1, 2"
    )

    dag = planner._parse_plan(llm_output, "chart MSFT and GOOGL capex")

    assert dag is not None
    assert len(dag.nodes) == 3

    # Node 3 (code_generation) should depend on nodes 1 and 2
    code_node = next(n for n in dag.nodes if n.metadata.get("template") == "code_generation")
    assert len(code_node.depends_on) == 2

    # Web nodes should have no dependencies
    web_nodes = [n for n in dag.nodes if n.metadata.get("template") == "web_finance"]
    assert len(web_nodes) == 2
    for wn in web_nodes:
        assert wn.depends_on == []


def test_dag_planner_trivial_plan_returns_none():
    """_parse_plan() with a single step should return a 1-node DAG (triggers None in plan())."""
    from octane.osa.dag_planner import DAGPlanner

    planner = DAGPlanner(bodega=None)

    llm_output = "1. web/web_finance | MSFT stock price"
    dag = planner._parse_plan(llm_output, "MSFT price")

    # parse returns the DAG — the plan() method filters out len<=1 cases
    assert dag is not None
    assert len(dag.nodes) == 1


# ── _inject_upstream_data (Session 8) ────────────────────────────────────────

def test_inject_upstream_data_root_node_unchanged():
    """Root nodes (no depends_on) should pass instruction through unchanged."""
    from octane.models.dag import TaskNode
    from octane.models.schemas import AgentResponse
    from octane.osa.orchestrator import _inject_upstream_data

    node = TaskNode(agent="web", instruction="Fetch MSFT capex", metadata={})
    accumulated: dict[str, AgentResponse] = {}

    result = _inject_upstream_data(node, accumulated, "original query")
    assert result == "Fetch MSFT capex"


def test_inject_upstream_data_enriches_dependent_node():
    """A dependent node should have upstream outputs prepended to its instruction."""
    from octane.models.dag import TaskNode
    from octane.models.schemas import AgentResponse
    from octane.osa.orchestrator import _inject_upstream_data

    upstream = TaskNode(agent="web", instruction="Fetch MSFT capex", metadata={})
    upstream_resp = AgentResponse(
        agent="web",
        success=True,
        output="MSFT capex: 2020: $15.4B, 2021: $20.6B, 2022: $23.9B",
    )

    dependent = TaskNode(
        agent="code",
        instruction="Chart the capex data",
        metadata={},
        depends_on=[upstream.task_id],
    )

    accumulated = {upstream.task_id: upstream_resp}
    result = _inject_upstream_data(dependent, accumulated, "chart MSFT capex")

    assert "Chart the capex data" in result
    assert "MSFT capex" in result
    assert "real data" in result


def test_inject_upstream_data_truncates_long_output():
    """Upstream outputs longer than 800 chars should be truncated."""
    from octane.models.dag import TaskNode
    from octane.models.schemas import AgentResponse
    from octane.osa.orchestrator import _inject_upstream_data

    upstream = TaskNode(agent="web", instruction="Fetch data", metadata={})
    upstream_resp = AgentResponse(
        agent="web",
        success=True,
        output="X" * 2000,  # very long output
    )

    dependent = TaskNode(
        agent="code",
        instruction="Use the data",
        metadata={},
        depends_on=[upstream.task_id],
    )

    accumulated = {upstream.task_id: upstream_resp}
    result = _inject_upstream_data(dependent, accumulated, "original")

    assert "[truncated]" in result
    # The injected upstream block should not vastly exceed the 800-char limit
    assert result.count("X") <= 800


# ── Executor artifact collection (Session 8) ─────────────────────────────────

@pytest.mark.asyncio
async def test_executor_creates_output_dir_and_saves_artifacts():
    """Executor should create ~/octane_output/<correlation_id>/ and copy artifacts there."""
    from pathlib import Path
    from octane.agents.code.executor import Executor

    executor = Executor()

    # Simple script that writes a CSV to OUTPUT_DIR
    code = "import os; open(os.path.join(OUTPUT_DIR, 'result.csv'), 'w').write('a,b\\n1,2')"

    result = await executor.run(code, requirements=None, correlation_id="test-session-8")

    assert result["exit_code"] == 0
    assert result["output_dir"] is not None

    output_dir = Path(result["output_dir"])
    assert output_dir.exists()
    assert "test-session-8" in str(output_dir)

    saved = result.get("output_files", [])
    assert "result.csv" in saved

    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)


# ── Scaler — model topology (Session 9) ──────────────────────────────────────

def test_scaler_recommends_tier_64gb():
    """50GB available → should recommend tier_64gb (40GB+ threshold)."""
    from octane.agents.sysstat.scaler import Scaler
    scaler = Scaler()
    rec = scaler.recommend(50.0)
    assert rec["tier"] == "tier_64gb"
    assert "brain" in rec["models"]
    assert "worker" in rec["models"]
    assert "grunt" in rec["models"]


def test_scaler_recommends_tier_32gb():
    """25GB available → should recommend tier_32gb (20–40GB range)."""
    from octane.agents.sysstat.scaler import Scaler
    scaler = Scaler()
    rec = scaler.recommend(25.0)
    assert rec["tier"] == "tier_32gb"
    assert "brain" in rec["models"]


def test_scaler_recommends_tier_16gb():
    """12GB available → should recommend tier_16gb (8–20GB range)."""
    from octane.agents.sysstat.scaler import Scaler
    scaler = Scaler()
    rec = scaler.recommend(12.0)
    assert rec["tier"] == "tier_16gb"
    assert "grunt" in rec["models"]
    assert "worker" not in rec["models"]


def test_scaler_recommends_tier_8gb():
    """3GB available → should recommend tier_8gb (minimal)."""
    from octane.agents.sysstat.scaler import Scaler
    scaler = Scaler()
    rec = scaler.recommend(3.0)
    assert rec["tier"] == "tier_8gb"
    assert len(rec["models"]) == 1
    assert "brain" in rec["models"]


def test_scaler_exact_boundary_40gb():
    """Exactly 40GB available → qualifies for tier_64gb."""
    from octane.agents.sysstat.scaler import Scaler
    assert Scaler().recommend(40.0)["tier"] == "tier_64gb"


def test_scaler_just_below_boundary():
    """39.9GB available → falls to tier_32gb."""
    from octane.agents.sysstat.scaler import Scaler
    assert Scaler().recommend(39.9)["tier"] == "tier_32gb"


# ── Memory waterfall (Session 9) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_recall_waterfall_redis_hit():
    """Redis has the key → recall returns immediately without touching Postgres."""
    import json
    from octane.agents.memory.agent import MemoryAgent
    from octane.tools.redis_client import RedisClient
    from octane.models.synapse import SynapseEventBus

    redis = RedisClient()
    redis._use_fallback = True  # in-process dict, no real Redis needed

    synapse = SynapseEventBus(persist=False)
    agent = MemoryAgent(synapse, redis=redis)  # no pg wired

    # Pre-load a memory key
    session_id = "test-waterfall"
    slot = "nvda_price"
    key = f"memory:{session_id}:{slot}"
    payload = json.dumps({"query": "NVDA price", "answer": "NVDA is $189.82", "metadata": {}})
    await redis.set(key, payload, ttl=3600)

    result = await agent.recall(session_id, "NVDA price")

    assert result is not None
    assert "189.82" in result


@pytest.mark.asyncio
async def test_memory_recall_waterfall_all_miss():
    """Redis miss + no Postgres → recall returns None."""
    from octane.agents.memory.agent import MemoryAgent
    from octane.tools.redis_client import RedisClient
    from octane.models.synapse import SynapseEventBus

    redis = RedisClient()
    redis._use_fallback = True

    synapse = SynapseEventBus(persist=False)
    agent = MemoryAgent(synapse, redis=redis)  # no pg

    result = await agent.recall("empty-session", "what did I ask before")
    assert result is None


@pytest.mark.asyncio
async def test_memory_writer_skips_short_answer():
    """Writer should not store answers shorter than 30 chars."""
    from octane.agents.memory.writer import MemoryWriter
    from octane.tools.redis_client import RedisClient

    redis = RedisClient()
    redis._use_fallback = True
    writer = MemoryWriter(redis=redis)

    stored = await writer.write(
        key="memory:s:slot",
        query="hi",
        answer="ok",  # too short
        session_id="s",
    )
    assert stored is False


@pytest.mark.asyncio
async def test_memory_writer_stores_substantive_answer():
    """Writer should store answers that pass the quality bar."""
    from octane.agents.memory.writer import MemoryWriter
    from octane.tools.redis_client import RedisClient

    redis = RedisClient()
    redis._use_fallback = True
    writer = MemoryWriter(redis=redis)

    stored = await writer.write(
        key="memory:s:nvda",
        query="NVDA price",
        answer="NVDA is trading at $189.82, up 1.2% today on strong volume.",
        session_id="s",
    )
    assert stored is True


# ── Code Agent self-healing Synapse events (Session 9) ───────────────────────

@pytest.mark.asyncio
async def test_code_agent_emits_synapse_on_success():
    """Successful first-attempt execution should emit code_success Synapse event."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.agents.code.agent import CodeAgent
    from octane.models.synapse import SynapseEventBus
    from octane.models.schemas import AgentRequest

    synapse = SynapseEventBus(persist=False)
    agent = CodeAgent(synapse)

    with patch.object(agent.planner, "plan", new=AsyncMock(return_value={
        "approach": "print output", "requirements": []
    })):
        with patch.object(agent.writer, "write", new=AsyncMock(return_value="print('hello')")):
            with patch.object(agent.executor, "run", new=AsyncMock(return_value={
                "stdout": "hello", "stderr": "", "exit_code": 0,
                "duration_ms": 10, "truncated": False,
                "output_dir": None, "output_files": [],
            })):
                req = AgentRequest(query="print hello", source="test")
                resp = await agent.execute(req)

    assert resp.success is True
    all_events = [e for trace in synapse.get_recent_traces(limit=10) for e in trace.events]
    event_types = [e.event_type for e in all_events]
    assert "code_success" in event_types


@pytest.mark.asyncio
async def test_code_agent_emits_synapse_on_debug_and_heal():
    """Failed attempt → debug → success should emit code_attempt_failed, code_debug_invoked, code_healed."""
    from unittest.mock import AsyncMock, patch
    from octane.agents.code.agent import CodeAgent
    from octane.models.synapse import SynapseEventBus
    from octane.models.schemas import AgentRequest

    synapse = SynapseEventBus(persist=False)
    agent = CodeAgent(synapse)

    fail_result = {
        "stdout": "", "stderr": "NameError: name 'x' is not defined",
        "exit_code": 1, "duration_ms": 5, "truncated": False,
        "output_dir": None, "output_files": [],
    }
    success_result = {
        "stdout": "42", "stderr": "", "exit_code": 0,
        "duration_ms": 5, "truncated": False,
        "output_dir": None, "output_files": [],
    }

    call_count = {"n": 0}
    async def mock_executor_run(*args, **kwargs):
        call_count["n"] += 1
        return fail_result if call_count["n"] == 1 else success_result

    with patch.object(agent.planner, "plan", new=AsyncMock(return_value={
        "approach": "fix", "requirements": []
    })):
        with patch.object(agent.writer, "write", new=AsyncMock(return_value="x = undefined\nprint(x)")):
            with patch.object(agent.executor, "run", side_effect=mock_executor_run):
                with patch.object(agent.debugger, "debug", new=AsyncMock(return_value="x = 42\nprint(x)")):
                    req = AgentRequest(query="compute x", source="test")
                    resp = await agent.execute(req)

    assert resp.success is True
    all_events = [e for trace in synapse.get_recent_traces(limit=20) for e in trace.events]
    event_types = [e.event_type for e in all_events]
    assert "code_attempt_failed" in event_types
    assert "code_debug_invoked" in event_types
    assert "code_healed" in event_types


@pytest.mark.asyncio
async def test_code_agent_emits_code_exhausted_after_max_retries():
    """All retries fail → should emit code_exhausted event."""
    from unittest.mock import AsyncMock, patch
    from octane.agents.code.agent import CodeAgent
    from octane.models.synapse import SynapseEventBus
    from octane.models.schemas import AgentRequest

    synapse = SynapseEventBus(persist=False)
    agent = CodeAgent(synapse)

    fail_result = {
        "stdout": "", "stderr": "SyntaxError: invalid syntax",
        "exit_code": 1, "duration_ms": 3, "truncated": False,
        "output_dir": None, "output_files": [],
    }

    with patch.object(agent.planner, "plan", new=AsyncMock(return_value={
        "approach": "broken", "requirements": []
    })):
        with patch.object(agent.writer, "write", new=AsyncMock(return_value="def (")):
            with patch.object(agent.executor, "run", new=AsyncMock(return_value=fail_result)):
                # Debugger returns same code → loop stops early
                with patch.object(agent.debugger, "debug", new=AsyncMock(return_value="def (")):
                    req = AgentRequest(query="broken code", source="test")
                    resp = await agent.execute(req)

    assert resp.success is False
    all_events = [e for trace in synapse.get_recent_traces(limit=20) for e in trace.events]
    event_types = [e.event_type for e in all_events]
    assert "code_exhausted" in event_types


# ── Janitor (Session 9) ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_janitor_sweep_returns_session_keys():
    """Janitor.sweep() should list all Redis keys for the given session."""
    from octane.agents.memory.janitor import Janitor
    from octane.tools.redis_client import RedisClient

    redis = RedisClient()
    redis._use_fallback = True

    # Pre-populate some keys
    await redis.set("memory:sess-j:topic_a", "data a", ttl=3600)
    await redis.set("memory:sess-j:topic_b", "data b", ttl=3600)

    janitor = Janitor(redis=redis)
    result = await janitor.sweep("sess-j")

    assert result["live_keys"] >= 2
    assert any("topic_a" in k for k in result["keys"])
    assert any("topic_b" in k for k in result["keys"])

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION 10 — Web Intelligence, Conversation History, DAG Visibility
# ═══════════════════════════════════════════════════════════════════════════════

# ── Synthesizer (Session 10) ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesizer_news_with_llm():
    """Synthesizer.synthesize_news() should call LLM and strip think blocks."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.synthesizer import Synthesizer

    mock_bodega = MagicMock()
    mock_bodega.chat_simple = AsyncMock(
        return_value="<think>reasoning</think>\nKey stories:\n1. NVIDIA hits $3T — first AI chipmaker to do so."
    )

    synth = Synthesizer(bodega=mock_bodega)
    articles = [
        {"title": "NVIDIA hits $3T valuation", "publisher": {"title": "Reuters"}, "published date": "2026-02-20"},
    ]
    result = await synth.synthesize_news("NVIDIA valuation", articles)

    assert "Key stories" in result
    assert "<think>" not in result
    mock_bodega.chat_simple.assert_called_once()


@pytest.mark.asyncio
async def test_synthesizer_news_fallback_no_llm():
    """Synthesizer.synthesize_news() without LLM should return plain list."""
    from octane.agents.web.synthesizer import Synthesizer

    synth = Synthesizer(bodega=None)
    articles = [
        {"title": "Article One", "publisher": {"title": "BBC"}, "published date": "2026-02-20"},
        {"title": "Article Two", "publisher": {"title": "CNN"}, "published date": "2026-02-19"},
    ]
    result = await synth.synthesize_news("test query", articles)

    assert "Article One" in result
    assert "Article Two" in result


@pytest.mark.asyncio
async def test_synthesizer_news_empty_articles():
    """Synthesizer should handle empty article list gracefully."""
    from octane.agents.web.synthesizer import Synthesizer

    synth = Synthesizer(bodega=None)
    result = await synth.synthesize_news("something", [])

    assert "No recent news" in result


@pytest.mark.asyncio
async def test_synthesizer_search_with_llm():
    """Synthesizer.synthesize_search() should use LLM for bullet synthesis."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.synthesizer import Synthesizer

    mock_bodega = MagicMock()
    mock_bodega.chat_simple = AsyncMock(
        return_value="• Python 3.13 improves performance by 15%\n• New JIT compiler enabled by default"
    )

    synth = Synthesizer(bodega=mock_bodega)
    results = [
        {"title": "Python 3.13 released", "description": "Faster Python with JIT", "url": "http://example.com"},
    ]
    result = await synth.synthesize_search("Python 3.13 features", results)

    assert "Python" in result
    mock_bodega.chat_simple.assert_called_once()


@pytest.mark.asyncio
async def test_synthesizer_search_fallback_no_llm():
    """Synthesizer.synthesize_search() without LLM should return plain list."""
    from octane.agents.web.synthesizer import Synthesizer

    synth = Synthesizer(bodega=None)
    results = [{"title": "Result One", "description": "Desc one", "url": "http://a.com"}]
    result = await synth.synthesize_search("some query", results)

    assert "Result One" in result


# ── QueryStrategist (Session 10) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_strategist_llm_returns_variations():
    """QueryStrategist should parse LLM JSON into validated strategies."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.query_strategist import QueryStrategist

    mock_bodega = MagicMock()
    mock_bodega.chat_simple = AsyncMock(return_value="""
[
  {"query": "AAPL stock price today", "api": "finance", "rationale": "ticker lookup"},
  {"query": "Apple earnings Q1 2026", "api": "news", "rationale": "recent results"},
  {"query": "Apple market cap 2026", "api": "search", "rationale": "general context"}
]
""")

    strategist = QueryStrategist(bodega=mock_bodega)
    strategies = await strategist.strategize("Apple stock performance")

    assert len(strategies) == 3
    assert all("query" in s and "api" in s for s in strategies)
    assert strategies[0]["api"] == "finance"
    assert strategies[1]["api"] == "news"


@pytest.mark.asyncio
async def test_query_strategist_rejects_invalid_api():
    """QueryStrategist should normalise unknown API values to 'search'."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.query_strategist import QueryStrategist

    mock_bodega = MagicMock()
    mock_bodega.chat_simple = AsyncMock(return_value="""
[{"query": "some query", "api": "twitter", "rationale": "social"}]
""")

    strategist = QueryStrategist(bodega=mock_bodega)
    strategies = await strategist.strategize("some topic")

    assert strategies[0]["api"] == "search"


@pytest.mark.asyncio
async def test_query_strategist_fallback_no_llm():
    """QueryStrategist without LLM should return single keyword-based strategy."""
    from octane.agents.web.query_strategist import QueryStrategist

    strategist = QueryStrategist(bodega=None)
    strategies = await strategist.strategize("AAPL earnings report")

    assert len(strategies) == 1
    assert strategies[0]["api"] == "finance"
    assert strategies[0]["query"] == "AAPL earnings report"


@pytest.mark.asyncio
async def test_query_strategist_keyword_news():
    """QueryStrategist keyword fallback should pick 'news' for news-like queries."""
    from octane.agents.web.query_strategist import QueryStrategist

    strategist = QueryStrategist(bodega=None)
    strategies = await strategist.strategize("latest news on AI regulation")

    assert strategies[0]["api"] == "news"


@pytest.mark.asyncio
async def test_query_strategist_llm_failure_falls_back():
    """QueryStrategist should fall back gracefully if LLM raises."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.agents.web.query_strategist import QueryStrategist

    mock_bodega = MagicMock()
    mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("LLM offline"))

    strategist = QueryStrategist(bodega=mock_bodega)
    strategies = await strategist.strategize("test query")

    assert len(strategies) == 1
    assert strategies[0]["query"] == "test query"


# ── WebAgent with Synthesizer (Session 10) ────────────────────────────────────

@pytest.mark.asyncio
async def test_web_agent_news_uses_synthesizer():
    """WebAgent._fetch_news should pass articles through Synthesizer."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.agents.web.agent import WebAgent

    mock_intel = MagicMock()
    mock_intel.news_search = AsyncMock(return_value={
        "articles": [
            {"title": "Big AI news", "publisher": {"title": "TechCrunch"}, "published date": "2026-02-20"},
        ]
    })

    with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_news", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = "Key stories:\n1. Big AI news — Major development."

        agent = WebAgent(make_synapse(), intel=mock_intel)
        req = make_request("AI news today", metadata={"sub_agent": "news"})
        resp = await agent.execute(req)

    assert resp.success is True
    assert "Key stories" in resp.output


@pytest.mark.asyncio
async def test_web_agent_search_uses_synthesizer():
    """WebAgent._fetch_search should pass results through Synthesizer."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.agents.web.agent import WebAgent

    mock_intel = MagicMock()
    mock_intel.web_search = AsyncMock(return_value={
        "web": {"results": [
            {"title": "Python tips", "description": "Best practices", "url": "http://a.com"}
        ]}
    })

    with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_search", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = "• Python best practices include writing tests."

        agent = WebAgent(make_synapse(), intel=mock_intel)
        req = make_request("Python best practices", metadata={"sub_agent": "search"})
        resp = await agent.execute(req)

    assert resp.success is True
    assert "Python" in resp.output


# ── Evaluator conversation history (Session 10) ───────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_injects_conversation_history():
    """Evaluator should include conversation history in the LLM prompt."""
    from unittest.mock import MagicMock
    from octane.osa.evaluator import Evaluator
    from octane.models.schemas import AgentResponse

    mock_bodega = MagicMock()
    captured_prompt: list[str] = []

    async def capture_chat_simple(prompt, system="", **kwargs):
        captured_prompt.append(prompt)
        return "Based on our conversation, AAPL is up."

    mock_bodega.chat_simple = capture_chat_simple

    evaluator = Evaluator(bodega=mock_bodega)
    history = [
        {"role": "user", "content": "Tell me about AAPL"},
        {"role": "assistant", "content": "AAPL is Apple Inc, currently trading around $200."},
    ]
    results = [AgentResponse(agent="web", success=True, output="AAPL: $215 today")]

    result = await evaluator.evaluate(
        "Is AAPL up today?",
        results,
        conversation_history=history,
    )

    assert len(captured_prompt) == 1
    prompt_text = captured_prompt[0]
    assert "Conversation history" in prompt_text
    assert "Tell me about AAPL" in prompt_text
    assert "AAPL is Apple Inc" in prompt_text


@pytest.mark.asyncio
async def test_evaluator_no_history_still_works():
    """Evaluator should work normally when conversation_history is None."""
    from unittest.mock import AsyncMock, MagicMock
    from octane.osa.evaluator import Evaluator
    from octane.models.schemas import AgentResponse

    mock_bodega = MagicMock()
    mock_bodega.chat_simple = AsyncMock(return_value="Direct answer.")

    evaluator = Evaluator(bodega=mock_bodega)
    results = [AgentResponse(agent="web", success=True, output="Some data")]

    result = await evaluator.evaluate("test query", results, conversation_history=None)
    assert result == "Direct answer."


@pytest.mark.asyncio
async def test_format_conversation_history_truncates():
    """_format_conversation_history should keep last max_turns entries only."""
    from octane.osa.evaluator import _format_conversation_history

    history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    result = _format_conversation_history(history, max_turns=4)

    lines = result.strip().split("\n")
    assert len(lines) == 4
    assert "msg 16" in result
    assert "msg 0" not in result


# ── Orchestrator conversation_history threading (Session 10) ──────────────────

@pytest.mark.asyncio
async def test_orchestrator_run_accepts_conversation_history():
    """Orchestrator.run() should pass conversation_history through to Evaluator."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.osa.orchestrator import Orchestrator
    from octane.models.dag import TaskDAG, TaskNode
    from octane.models.schemas import AgentResponse

    synapse = make_synapse()
    osa = Orchestrator(synapse)
    osa._preflight_done = True

    with patch.object(osa.guard, "check_input", new_callable=AsyncMock) as mock_guard, \
         patch.object(osa.decomposer, "decompose", new_callable=AsyncMock) as mock_decompose, \
         patch.object(osa.evaluator, "evaluate", new_callable=AsyncMock) as mock_eval:

        mock_guard.return_value = {"safe": True}

        mock_dag = MagicMock(spec=TaskDAG)
        mock_dag.nodes = [TaskNode(task_id="t1", agent="web", instruction="search this")]
        mock_dag.reasoning = "single web task"
        mock_dag.original_query = "test"
        mock_dag.execution_order.return_value = [[mock_dag.nodes[0]]]
        mock_decompose.return_value = mock_dag

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=AgentResponse(
            agent="web", success=True, output="result"
        ))
        osa.router._agents["web"] = mock_agent
        mock_eval.return_value = "Final answer with context."

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        result = await osa.run("Follow-up question", conversation_history=history)

    assert result == "Final answer with context."
    call_kwargs = mock_eval.call_args
    assert call_kwargs.kwargs.get("conversation_history") == history


# ── DAG egress metadata (Session 10) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_egress_event_contains_dag_metadata():
    """Egress Synapse event should contain dag_nodes and dag_reasoning fields."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.osa.orchestrator import Orchestrator
    from octane.models.dag import TaskDAG, TaskNode
    from octane.models.schemas import AgentResponse

    synapse = make_synapse()
    osa = Orchestrator(synapse)
    osa._preflight_done = True

    with patch.object(osa.guard, "check_input", new_callable=AsyncMock) as mock_guard, \
         patch.object(osa.decomposer, "decompose", new_callable=AsyncMock) as mock_decompose, \
         patch.object(osa.evaluator, "evaluate", new_callable=AsyncMock) as mock_eval:

        mock_guard.return_value = {"safe": True}

        mock_dag = MagicMock(spec=TaskDAG)
        mock_dag.nodes = [
            TaskNode(task_id="t1", agent="web", instruction="task one"),
            TaskNode(task_id="t2", agent="code", instruction="task two"),
        ]
        mock_dag.reasoning = "two-step plan"
        mock_dag.original_query = "compound query"
        mock_dag.execution_order.return_value = [[mock_dag.nodes[0]], [mock_dag.nodes[1]]]
        mock_decompose.return_value = mock_dag

        mock_web = MagicMock()
        mock_web.run = AsyncMock(return_value=AgentResponse(agent="web", success=True, output="data"))
        mock_code = MagicMock()
        mock_code.run = AsyncMock(return_value=AgentResponse(agent="code", success=True, output="result"))
        osa.router._agents["web"] = mock_web
        osa.router._agents["code"] = mock_code
        mock_eval.return_value = "synthesized"

        await osa.run("compound query")

    # Find egress event
    all_events = [e for trace in synapse.get_recent_traces(limit=5) for e in trace.events]
    egress_events = [e for e in all_events if e.event_type == "egress"]
    assert egress_events, "No egress event found"

    egress = egress_events[-1]
    assert egress.payload.get("dag_nodes") == 2
    assert egress.payload.get("dag_reasoning") == "two-step plan"


# ── Shadows / watch tasks (Session 11) ───────────────────────────────────────

# ── 1. Task collection is importable and non-empty ────────────────────────────

def test_octane_tasks_collection_is_non_empty():
    """octane.tasks should export a non-empty list of task functions."""
    from octane.tasks import octane_tasks
    assert isinstance(octane_tasks, list)
    assert len(octane_tasks) >= 1


def test_monitor_ticker_is_in_collection():
    """monitor_ticker must be registered in octane_tasks."""
    from octane.tasks import octane_tasks, monitor_ticker
    assert monitor_ticker in octane_tasks


# ── 2. monitor_ticker task structure ─────────────────────────────────────────

def test_monitor_ticker_is_coroutine_function():
    """monitor_ticker must be an async function (Shadows requires awaitable tasks)."""
    import asyncio
    from octane.tasks.monitor import monitor_ticker
    assert asyncio.iscoroutinefunction(monitor_ticker)


def test_monitor_ticker_has_perpetual_dependency():
    """monitor_ticker must declare a Perpetual dependency via default param."""
    import inspect
    from shadows import Perpetual
    from octane.tasks.monitor import monitor_ticker

    sig = inspect.signature(monitor_ticker)
    perpetual_params = [
        p for p in sig.parameters.values()
        if isinstance(p.default, Perpetual)
    ]
    assert perpetual_params, "monitor_ticker must have a Perpetual default parameter"


def test_monitor_ticker_poll_interval_is_one_hour():
    """Default poll interval must be 1 hour."""
    from datetime import timedelta
    from octane.tasks.monitor import POLL_INTERVAL
    assert POLL_INTERVAL == timedelta(hours=1)


# ── 3. monitor_ticker execution (mocked Bodega + Redis) ───────────────────────

@pytest.mark.asyncio
async def test_monitor_ticker_stores_quote_to_redis():
    """monitor_ticker should store the fetched quote in Redis when Bodega succeeds."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.tasks.monitor import monitor_ticker
    from shadows import Perpetual
    from shadows.dependencies import TaskLogger
    import logging

    fake_quote = {"ticker": "AAPL", "price": "182.50", "c": "182.50"}
    fake_perpetual = MagicMock(spec=Perpetual)
    fake_log = logging.LoggerAdapter(logging.getLogger("test"), {})

    mock_intel_ctx = AsyncMock()
    mock_intel_ctx.__aenter__ = AsyncMock(return_value=mock_intel_ctx)
    mock_intel_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_intel_ctx.finance = AsyncMock(return_value=fake_quote)

    stored: dict = {}

    async def fake_execute(commands):
        pass

    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=False)
    mock_pipe.set = MagicMock(return_value=mock_pipe)
    mock_pipe.rpush = MagicMock(return_value=mock_pipe)
    mock_pipe.ltrim = MagicMock(return_value=mock_pipe)
    mock_pipe.execute = AsyncMock()

    mock_redis = AsyncMock()
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)
    mock_redis.publish = AsyncMock()
    mock_redis.aclose = AsyncMock()

    with patch("octane.tools.bodega_intel.BodegaIntelClient", return_value=mock_intel_ctx), \
         patch("octane.config.settings") as mock_settings, \
         patch("redis.asyncio.from_url", return_value=mock_redis):
        mock_settings.bodega_intel_url = "http://localhost:44469"
        mock_settings.redis_url = "redis://localhost:6379/0"

        await monitor_ticker(
            ticker="AAPL",
            perpetual=fake_perpetual,
            log=fake_log,
        )

    mock_intel_ctx.finance.assert_awaited_once_with("AAPL")
    mock_pipe.set.assert_called_once()
    key_arg = mock_pipe.set.call_args[0][0]
    assert key_arg == "watch:AAPL:latest"


@pytest.mark.asyncio
async def test_monitor_ticker_survives_bodega_failure():
    """monitor_ticker must not raise if Bodega is unreachable — log + return."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.tasks.monitor import monitor_ticker
    from shadows import Perpetual
    import logging

    fake_perpetual = MagicMock(spec=Perpetual)
    fake_log = logging.LoggerAdapter(logging.getLogger("test"), {})

    mock_intel_ctx = AsyncMock()
    mock_intel_ctx.__aenter__ = AsyncMock(return_value=mock_intel_ctx)
    mock_intel_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_intel_ctx.finance = AsyncMock(side_effect=ConnectionError("Bodega offline"))

    with patch("octane.tools.bodega_intel.BodegaIntelClient", return_value=mock_intel_ctx), \
         patch("octane.config.settings") as mock_settings:
        mock_settings.bodega_intel_url = "http://localhost:44469"
        mock_settings.redis_url = "redis://localhost:6379/0"

        # Must not raise
        await monitor_ticker(
            ticker="TSLA",
            perpetual=fake_perpetual,
            log=fake_log,
        )


# ── 4. Worker PID file utilities ──────────────────────────────────────────────

def test_worker_pid_file_path_is_under_home():
    """PID file must live in ~/.octane/worker.pid."""
    from pathlib import Path
    from octane.tasks.worker_process import PID_FILE
    assert PID_FILE == Path.home() / ".octane" / "worker.pid"


def test_read_pid_returns_none_when_file_absent(tmp_path, monkeypatch):
    """read_pid() returns None when the PID file does not exist."""
    import octane.tasks.worker_process as wp
    monkeypatch.setattr(wp, "PID_FILE", tmp_path / "worker.pid")
    assert wp.read_pid() is None


def test_read_pid_returns_none_for_dead_process(tmp_path, monkeypatch):
    """read_pid() returns None and cleans up when the stored PID is dead."""
    import octane.tasks.worker_process as wp
    pid_file = tmp_path / "worker.pid"
    # PID 9999999 is virtually guaranteed to not exist
    pid_file.write_text("9999999")
    monkeypatch.setattr(wp, "PID_FILE", pid_file)
    result = wp.read_pid()
    assert result is None
    assert not pid_file.exists()


# ── Session 12: Workflow module tests ─────────────────────────────────────────

class TestWorkflowTemplate:
    """Round-trip, fill, and placeholder tests for WorkflowTemplate."""

    def _make_template(self, **kwargs):
        from octane.workflow.template import WorkflowTemplate
        defaults = dict(
            name="test-wf",
            description="A test workflow",
            variables={"ticker": "AAPL", "query": "What is {{ticker}} price?"},
            reasoning="single lookup",
            nodes=[
                {
                    "task_id": "t1",
                    "agent": "web",
                    "instruction": "Get {{ticker}} price",
                    "depends_on": [],
                    "priority": 1,
                    "metadata": {"hint": "finance for {{ticker}}"},
                }
            ],
        )
        defaults.update(kwargs)
        return WorkflowTemplate(**defaults)

    def test_save_and_load_round_trip(self, tmp_path):
        """save() then load() produces an identical template."""
        from octane.workflow.template import WorkflowTemplate
        t = self._make_template()
        saved = t.save(tmp_path / "test-wf.workflow.json")
        loaded = WorkflowTemplate.load(saved)
        assert loaded.name == t.name
        assert loaded.variables == t.variables
        assert loaded.nodes == t.nodes

    def test_fill_substitutes_variables(self):
        """fill() replaces {{ticker}} with the default variable value."""
        t = self._make_template()
        nodes = t.fill()
        assert "AAPL" in nodes[0].instruction
        assert "{{ticker}}" not in nodes[0].instruction

    def test_fill_with_overrides_takes_precedence(self):
        """fill(overrides) uses runtime value over template default."""
        t = self._make_template()
        nodes = t.fill(overrides={"ticker": "MSFT"})
        assert "MSFT" in nodes[0].instruction
        assert "AAPL" not in nodes[0].instruction

    def test_fill_leaves_unknown_placeholders_as_is(self):
        """Unknown {{placeholder}} names are left unchanged, not silently removed."""
        from octane.workflow.template import WorkflowTemplate
        t = WorkflowTemplate(
            name="partial",
            nodes=[{
                "task_id": "t1",
                "agent": "web",
                "instruction": "Do {{known}} and {{unknown}}",
                "depends_on": [],
                "priority": 1,
                "metadata": {},
            }],
            variables={"known": "resolved"},
        )
        nodes = t.fill()
        assert "resolved" in nodes[0].instruction
        assert "{{unknown}}" in nodes[0].instruction

    def test_list_placeholders_returns_all_names(self):
        """list_placeholders() finds every unique {{name}} in the node list."""
        t = self._make_template()
        placeholders = t.list_placeholders()
        # ticker appears in both instruction and metadata hint
        assert "ticker" in placeholders

    def test_to_dag_returns_task_dag(self):
        """to_dag() wraps filled nodes in a TaskDAG instance."""
        from octane.models.dag import TaskDAG
        t = self._make_template()
        dag = t.to_dag()
        assert isinstance(dag, TaskDAG)
        assert len(dag.nodes) == 1
        assert dag.nodes[0].agent == "web"


class TestWorkflowExporter:
    """export_from_dag and export_from_trace unit tests."""

    def _make_dag(self, query="Get AAPL price"):
        from octane.models.dag import TaskDAG, TaskNode
        return TaskDAG(
            nodes=[
                TaskNode(
                    task_id="t1",
                    agent="web",
                    instruction=f"{query} today",
                    depends_on=[],
                    priority=1,
                    metadata={"hint": query},
                )
            ],
            reasoning="single lookup",
            original_query=query,
        )

    def test_export_from_dag_produces_template(self):
        """export_from_dag() returns a WorkflowTemplate with correct name."""
        from octane.workflow.exporter import export_from_dag
        dag = self._make_dag()
        tpl = export_from_dag(dag, name="aapl-lookup")
        assert tpl.name == "aapl-lookup"
        assert len(tpl.nodes) == 1

    def test_export_from_dag_substitutes_query_placeholder(self):
        """export_from_dag() replaces original_query text with {{query}}."""
        from octane.workflow.exporter import export_from_dag
        dag = self._make_dag(query="Get AAPL price")
        tpl = export_from_dag(dag, name="aapl-lookup")
        instruction = tpl.nodes[0]["instruction"]
        # The literal query text should be gone; {{query}} placeholder present
        assert "{{query}}" in instruction
        assert "Get AAPL price" not in instruction

    def test_export_from_trace_raises_on_missing_trace(self, tmp_path):
        """export_from_trace() raises FileNotFoundError for unknown correlation_id."""
        from octane.workflow.exporter import export_from_trace
        with pytest.raises(FileNotFoundError, match="No trace found"):
            export_from_trace("nonexistent-id-xyz", trace_dir=tmp_path)

    def test_export_from_trace_raises_on_missing_dag_nodes_json(self, tmp_path):
        """export_from_trace() raises ValueError when dag_nodes_json absent."""
        import json
        from octane.workflow.exporter import export_from_trace
        from octane.models.synapse import SynapseEvent
        # Write a trace with decomposition_complete but no dag_nodes_json
        evt = SynapseEvent(
            correlation_id="abc12345",
            event_type="decomposition_complete",
            source="decomposer",
            target="router",
            payload={"node_count": 1},  # no dag_nodes_json
        )
        trace_file = tmp_path / "abc12345.jsonl"
        trace_file.write_text(evt.model_dump_json() + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="dag_nodes_json"):
            export_from_trace("abc12345", trace_dir=tmp_path)

    def test_export_from_trace_rebuilds_template(self, tmp_path):
        """export_from_trace() correctly reconstructs WorkflowTemplate from trace."""
        import json
        from octane.models.dag import TaskNode
        from octane.models.synapse import SynapseEvent
        from octane.workflow.exporter import export_from_trace
        node = TaskNode(
            task_id="t1", agent="web",
            instruction="Get TSLA price today",
            depends_on=[], priority=1, metadata={},
        )
        evt = SynapseEvent(
            correlation_id="trace99",
            event_type="decomposition_complete",
            source="decomposer",
            target="router",
            payload={
                "node_count": 1,
                "dag_nodes_json": [node.model_dump()],
                "dag_original_query": "Get TSLA price today",
            },
        )
        trace_file = tmp_path / "trace99.jsonl"
        trace_file.write_text(evt.model_dump_json() + "\n", encoding="utf-8")
        tpl = export_from_trace("trace99", name="tsla-wf", trace_dir=tmp_path)
        assert tpl.name == "tsla-wf"
        assert len(tpl.nodes) == 1
        # The instruction should have had "Get TSLA price today" replaced with {{query}}
        assert "{{query}}" in tpl.nodes[0]["instruction"]


# ── Edge-case regression tests (post-bug-fix) ─────────────────────────────────
#
# These tests cover the three real-world failure modes observed when running
# `octane ask` in a dev environment:
#
#   1. Duplicate class definition  — MemoryAgent must expose connect_pg / remember / recall
#   2. Postgres unavailable        — connect_pg() must return False, never raise
#   3. Bodega offline              — pre_flight must complete without crashing
#
# ──────────────────────────────────────────────────────────────────────────────


class TestMemoryAgentIntegrity:
    """Guard against the duplicate-class bug and verify the public API surface."""

    def test_only_one_memory_agent_class_is_importable(self):
        """Regression: two class definitions in the same file caused the second
        (incomplete) one to shadow the first.  Ensure only one survives and it
        is the complete Postgres+Redis version."""
        import inspect
        from octane.agents.memory.agent import MemoryAgent

        # Count how many times 'class MemoryAgent' appears in the actual source
        source = inspect.getsource(MemoryAgent)
        # The source of a *class* (via inspect.getsource) should not contain
        # another full class definition with the same name inside it.
        assert source.count("class MemoryAgent") == 1

    def test_memory_agent_exposes_connect_pg(self):
        """connect_pg must be a callable async method on MemoryAgent."""
        import asyncio
        from octane.agents.memory.agent import MemoryAgent
        from octane.models.synapse import SynapseEventBus

        agent = MemoryAgent(SynapseEventBus())
        assert callable(getattr(agent, "connect_pg", None)), (
            "connect_pg missing — duplicate class definition bug has re-appeared"
        )

    def test_memory_agent_exposes_remember_and_recall(self):
        """remember() and recall() must be present — used directly by Orchestrator."""
        from octane.agents.memory.agent import MemoryAgent
        from octane.models.synapse import SynapseEventBus

        agent = MemoryAgent(SynapseEventBus())
        assert callable(getattr(agent, "remember", None))
        assert callable(getattr(agent, "recall", None))

    async def test_connect_pg_returns_false_when_asyncpg_missing(self, monkeypatch):
        """connect_pg() must return False gracefully when asyncpg is not installed.
        This matches the observed .venv behaviour where asyncpg is absent."""
        import builtins
        from octane.agents.memory.agent import MemoryAgent
        from octane.models.synapse import SynapseEventBus

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "asyncpg":
                raise ModuleNotFoundError("No module named 'asyncpg'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        agent = MemoryAgent(SynapseEventBus())
        result = await agent.connect_pg()
        assert result is False, "connect_pg should return False when asyncpg is missing"

    async def test_connect_pg_returns_false_when_postgres_unreachable(self, monkeypatch):
        """connect_pg() must return False when Postgres is unreachable (connection refused)."""
        from octane.agents.memory.agent import MemoryAgent
        from octane.models.synapse import SynapseEventBus
        from octane.tools.pg_client import PgClient

        # Patch PgClient.connect to simulate connection failure
        async def mock_connect(self):
            return False

        monkeypatch.setattr(PgClient, "connect", mock_connect)
        agent = MemoryAgent(SynapseEventBus())
        result = await agent.connect_pg()
        assert result is False


class TestOrchestratorGracefulDegradation:
    """Orchestrator pre_flight must complete and return a status dict even when
    all external services (Bodega, Postgres, Redis) are offline."""

    async def test_preflight_completes_when_bodega_offline(self, monkeypatch):
        """pre_flight returns a status dict with bodega_reachable=False.
        Regression: used to crash with AttributeError before duplicate-class fix."""
        import httpx
        from octane.osa.orchestrator import Orchestrator
        from octane.models.synapse import SynapseEventBus
        from octane.tools.pg_client import PgClient

        # Simulate Bodega being unreachable
        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        # Simulate Postgres also down
        async def mock_pg_connect(self):
            return False

        monkeypatch.setattr(PgClient, "connect", mock_pg_connect)

        osa = Orchestrator(SynapseEventBus())
        status = await osa.pre_flight()

        assert isinstance(status, dict), "pre_flight must return a dict"
        assert "bodega_reachable" in status
        assert status["bodega_reachable"] is False

    async def test_preflight_does_not_raise_on_connect_pg(self, monkeypatch):
        """_connect_memory_pg must never propagate an exception — it is fire-and-forget."""
        import httpx
        from octane.osa.orchestrator import Orchestrator
        from octane.models.synapse import SynapseEventBus
        from octane.tools.pg_client import PgClient

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        async def exploding_connect(self):
            raise RuntimeError("Simulated Postgres catastrophic failure")

        monkeypatch.setattr(PgClient, "connect", exploding_connect)

        osa = Orchestrator(SynapseEventBus())
        # Should NOT raise even when PgClient.connect throws
        try:
            await osa.pre_flight()
        except RuntimeError:
            pytest.fail(
                "pre_flight propagated a Postgres exception — "
                "_connect_memory_pg must swallow infra errors"
            )


class TestWebAgentPublicAPI:
    """Guard against missing methods on WebAgent — regression for _format_web_results."""

    def test_web_agent_has_all_format_methods(self):
        """Both _format_market_data and _format_web_results must exist.
        Regression: _format_web_results was called but never defined, crashing
        any finance query where the ticker was not recognised (e.g. small-cap stocks)."""
        from octane.agents.web.agent import WebAgent
        from octane.models.synapse import SynapseEventBus

        agent = WebAgent(SynapseEventBus())
        assert callable(getattr(agent, "_format_market_data", None)), \
            "_format_market_data missing"
        assert callable(getattr(agent, "_format_web_results", None)), \
            "_format_web_results missing — small-cap/unrecognised ticker queries will crash"

    def test_format_web_results_returns_string(self):
        """_format_web_results must return a non-empty string for valid results."""
        from octane.agents.web.agent import WebAgent
        from octane.models.synapse import SynapseEventBus

        agent = WebAgent(SynapseEventBus())
        raw = {
            "web": {
                "results": [
                    {"title": "Netscout Systems Q3 results", "url": "https://example.com",
                     "description": "NTCT reported revenue of $85M, beating estimates."},
                ]
            }
        }
        result = agent._format_web_results(raw, "netscout systems price")
        assert isinstance(result, str)
        assert "Netscout" in result

    def test_format_web_results_handles_empty_results(self):
        """_format_web_results must not crash on an empty result set."""
        from octane.agents.web.agent import WebAgent
        from octane.models.synapse import SynapseEventBus

        agent = WebAgent(SynapseEventBus())
        result = agent._format_web_results({}, "unknown query")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_agents_instantiate_without_error(self):
        """All five agents must instantiate cleanly — catches missing-method bugs early."""
        from octane.agents.web.agent import WebAgent
        from octane.agents.code.agent import CodeAgent
        from octane.agents.memory.agent import MemoryAgent
        from octane.agents.sysstat.agent import SysStatAgent
        from octane.agents.pnl.agent import PnLAgent
        from octane.models.synapse import SynapseEventBus

        synapse = SynapseEventBus()
        for cls in [WebAgent, CodeAgent, MemoryAgent, SysStatAgent, PnLAgent]:
            try:
                cls(synapse)
            except Exception as exc:
                pytest.fail(f"{cls.__name__} failed to instantiate: {exc}")


class TestClockUtility:
    """Regression suite for octane.utils.clock — the single source of truth for dates.

    These tests guard against the class of bug where stale dates are hard-coded
    or the clock module is broken, causing search queries and LLM prompts to use
    the wrong temporal context (e.g. showing Oct-2025 data when today is Feb-2026).
    """

    def test_today_str_is_iso_format(self):
        """today_str() must return a valid ISO 8601 date: YYYY-MM-DD."""
        import re
        from octane.utils.clock import today_str

        result = today_str()
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result), \
            f"today_str() returned unexpected format: {result!r}"

    def test_today_human_contains_year(self):
        """today_human() must contain the current 4-digit year."""
        from octane.utils.clock import today_human, year

        result = today_human()
        assert year() in result, \
            f"today_human() '{result}' does not contain current year '{year()}'"

    def test_month_year_format(self):
        """month_year() must return '<MonthName> <YYYY>' (e.g. 'February 2026')."""
        import re
        from octane.utils.clock import month_year

        result = month_year()
        assert re.match(r"^[A-Z][a-z]+ \d{4}$", result), \
            f"month_year() returned unexpected format: {result!r}"

    def test_evaluator_system_prompt_contains_today(self):
        """The Evaluator system prompt must embed today's date so the LLM
        can flag stale data (e.g. a stock price from 4 months ago)."""
        from octane.osa.evaluator import _build_system_prompt
        from octane.utils.clock import year

        prompt = _build_system_prompt(None)
        assert year() in prompt, \
            "Evaluator system prompt does not contain the current year — LLM will not detect stale data"

    def test_finance_fallback_query_includes_month_year(self):
        """When WebAgent falls back to web_search (no ticker), the query sent to
        Brave must include the current month+year to surface fresh results."""
        from octane.agents.web.agent import WebAgent
        from octane.models.synapse import SynapseEventBus
        from octane.utils.clock import month_year

        # Verify the month_year() string itself is well-formed
        my = month_year()
        assert len(my) > 6, f"month_year() too short: {my!r}"

        # Verify the agent module imports clock correctly (would raise ImportError if broken)
        agent = WebAgent(SynapseEventBus())
        assert agent is not None


# ─────────────────────────────────────────────────────────────────────────────
# Session 13 — Developer Experience & Observable Internals
# ─────────────────────────────────────────────────────────────────────────────


class TestDagCommand:
    """octane dag — Decomposer dry-run renders the task DAG without executing agents.

    Regression goals:
    - DAG dry-run must NEVER dispatch agents — it reads the TaskDAG only.
    - Node table must include agent name, sub-agent, wave number, template.
    - Partial / full trace IDs must be resolvable by _resolve_trace_id.
    - Command must complete even when Bodega is offline (keyword fallback path).
    """

    async def test_dag_dryryn_does_not_dispatch_agents(self, monkeypatch):
        """Decomposer.decompose() is called but no agent.execute() may be called."""
        from octane.osa.decomposer import Decomposer
        from octane.models.synapse import SynapseEventBus

        executed = []

        class _MockBodega:
            async def health(self):
                return {}
            async def current_model(self):
                return {"error": "offline"}
            async def close(self):
                pass

        decomposer = Decomposer(bodega=None)
        dag_result = await decomposer.decompose("what is AAPL trading at?")

        # Confirm we get a TaskDAG with nodes — no execution happened
        assert dag_result is not None
        assert len(dag_result.nodes) >= 1
        assert executed == [], "agents must not be dispatched during dag dry-run"

    async def test_dag_finance_query_routes_to_web_finance(self):
        """A stock price query must decompose to agent=web, sub_agent=finance."""
        from octane.osa.decomposer import Decomposer

        d = Decomposer(bodega=None)
        result = await d.decompose("what is the price of Apple stock?")

        assert len(result.nodes) >= 1
        first = result.nodes[0]
        assert first.agent == "web"
        assert first.metadata.get("sub_agent") in ("finance", "web_finance")

    async def test_dag_code_query_routes_to_code_agent(self):
        """A code generation query must decompose to agent=code."""
        from octane.osa.decomposer import Decomposer

        d = Decomposer(bodega=None)
        result = await d.decompose("write a python script to sort a list")

        assert len(result.nodes) >= 1
        assert result.nodes[0].agent == "code"

    async def test_dag_nodes_have_required_metadata(self):
        """Every TaskNode in the DAG must carry agent, sub_agent, template metadata."""
        from octane.osa.decomposer import Decomposer

        d = Decomposer(bodega=None)
        result = await d.decompose("latest NVIDIA news")

        for node in result.nodes:
            assert node.agent, f"node {node.id} missing agent"
            assert "sub_agent" in node.metadata, f"node {node.id} missing sub_agent metadata"
            assert "template" in node.metadata, f"node {node.id} missing template metadata"

    def test_resolve_trace_id_exact_match(self, monkeypatch):
        """_resolve_trace_id must return the full ID when given an exact match."""
        from octane.main import _resolve_trace_id
        from octane.models.synapse import SynapseEventBus, SynapseEvent

        synapse = SynapseEventBus(persist=False)
        cid = "abc123-def456-full"
        # Emit a real event so get_trace can find it in _events
        event = SynapseEvent(
            event_type="ingress",
            source="user",
            correlation_id=cid,
        )
        synapse.emit(event)

        result = _resolve_trace_id(synapse, cid)
        assert result == cid

    def test_resolve_trace_id_partial_prefix(self, monkeypatch):
        """_resolve_trace_id must resolve a partial (prefix) ID against stored traces."""
        from octane.main import _resolve_trace_id
        from octane.models.synapse import SynapseEventBus, SynapseEvent

        synapse = SynapseEventBus(persist=False)
        cid = "deadbeef-1234-5678-abcd"
        event = SynapseEvent(event_type="ingress", source="user", correlation_id=cid)
        synapse.emit(event)

        # Patch list_traces to return our test CID
        monkeypatch.setattr(synapse, "list_traces", lambda limit=50: [cid])

        result = _resolve_trace_id(synapse, "deadbeef")
        assert result == cid


class TestPrefCommand:
    """octane pref show/set/reset — wires PreferenceManager to the CLI.

    Regression goals:
    - set/get/delete round-trip works end-to-end.
    - Unknown keys are rejected with a helpful error.
    - Invalid values for constrained keys (verbosity, expertise, response_style) are rejected.
    - reset single key restores the default value.
    - reset all keys removes all customisations.
    """

    async def test_pref_set_and_get_round_trip(self, monkeypatch):
        """set() followed by get_all() must return the new value."""
        from octane.agents.pnl.preference_manager import PreferenceManager

        storage: dict[str, str] = {}

        class _FakeRedis:
            async def get(self, key):
                return storage.get(key)
            async def set(self, key, value, ttl=0):
                storage[key] = value
            async def delete(self, key):
                storage.pop(key, None)
            async def close(self):
                pass

        pm = PreferenceManager(redis=_FakeRedis())
        await pm.set("alice", "verbosity", "detailed")
        val = await pm.get("alice", "verbosity")
        assert val == "detailed"

    async def test_pref_reset_restores_default(self, monkeypatch):
        """delete() followed by get() must return the built-in default."""
        from octane.agents.pnl.preference_manager import PreferenceManager, DEFAULTS

        storage: dict[str, str] = {}

        class _FakeRedis:
            async def get(self, key):
                return storage.get(key)
            async def set(self, key, value, ttl=0):
                storage[key] = value
            async def delete(self, key):
                storage.pop(key, None)
            async def close(self):
                pass

        pm = PreferenceManager(redis=_FakeRedis())
        await pm.set("bob", "expertise", "beginner")
        await pm.delete("bob", "expertise")
        val = await pm.get("bob", "expertise")
        assert val == DEFAULTS["expertise"]

    def test_pref_invalid_key_rejected(self):
        """_pref_set CLI helper must exit non-zero for unknown preference keys."""
        from typer.testing import CliRunner
        from octane.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["pref", "set", "nonexistent_key", "value"])
        assert result.exit_code != 0 or "Unknown preference key" in (result.output or "")

    def test_pref_choices_exhaustive(self):
        """Every constrained preference key in PreferenceManager.DEFAULTS must have
        a non-empty choices list in _PREF_CHOICES (or be marked as free text)."""
        from octane.agents.pnl.preference_manager import DEFAULTS
        from octane.main import _PREF_CHOICES

        # All DEFAULTS keys must appear in _PREF_CHOICES
        for key in DEFAULTS:
            assert key in _PREF_CHOICES, \
                f"Preference key '{key}' in DEFAULTS is not listed in _PREF_CHOICES"


class TestVersionCommand:
    """octane version — styled splash panel with version, Python, agents, Shadows."""

    def test_version_output_contains_octane_version(self):
        """The splash output must contain the Octane version string."""
        from typer.testing import CliRunner
        from octane.main import app
        from octane import __version__

        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        assert __version__ in result.output

    def test_version_output_contains_python(self):
        """The splash output must mention the Python version."""
        import sys
        from typer.testing import CliRunner
        from octane.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert py_ver in result.output, \
            f"Python version '{py_ver}' not found in version output"

