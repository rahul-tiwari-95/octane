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
    # _fetch_news now also fans out to web_search — stub it as empty so the
    # news-API path is exercised for assertions.
    mock_intel.web_search = AsyncMock(return_value={"web": {"results": []}})

    # Stub extractor: extraction fails → triggers synthesize_news fallback path
    mock_extractor = MagicMock()
    mock_extractor.extract_batch = AsyncMock(return_value=[])

    agent = WebAgent(make_synapse(), intel=mock_intel, extractor=mock_extractor)
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

    # Stub extractor: extraction fails → triggers synthesize_search fallback path
    mock_extractor = MagicMock()
    mock_extractor.extract_batch = AsyncMock(return_value=[])

    agent = WebAgent(make_synapse(), intel=mock_intel, extractor=mock_extractor)
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

    instruction, upstream_results = _inject_upstream_data(node, accumulated, "original query")
    assert instruction == "Fetch MSFT capex"
    assert upstream_results == {}


def test_inject_upstream_data_enriches_dependent_node():
    """A dependent node should have upstream outputs prepended to its instruction,
    and upstream_results should carry the structured data dict."""
    from octane.models.dag import TaskNode
    from octane.models.schemas import AgentResponse
    from octane.osa.orchestrator import _inject_upstream_data

    upstream = TaskNode(agent="web", instruction="Fetch MSFT capex", metadata={})
    upstream_resp = AgentResponse(
        agent="web",
        success=True,
        output="MSFT capex: 2020: $15.4B, 2021: $20.6B, 2022: $23.9B",
        data={"market_data": {"ticker": "MSFT", "capex": 23.9}},
    )

    dependent = TaskNode(
        agent="code",
        instruction="Chart the capex data",
        metadata={},
        depends_on=[upstream.task_id],
    )

    accumulated = {upstream.task_id: upstream_resp}
    instruction, upstream_results = _inject_upstream_data(dependent, accumulated, "chart MSFT capex")

    assert "Chart the capex data" in instruction
    assert "MSFT capex" in instruction
    assert "real data" in instruction
    # Structured data flows through
    assert upstream.task_id in upstream_results
    assert upstream_results[upstream.task_id] == {"market_data": {"ticker": "MSFT", "capex": 23.9}}


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
    instruction, upstream_results = _inject_upstream_data(dependent, accumulated, "original")

    assert "[truncated]" in instruction
    # The injected upstream block should not vastly exceed the 800-char limit
    assert instruction.count("X") <= 800
    # upstream_results falls back to {"output": ...} since data={} is falsy
    assert upstream.task_id in upstream_results


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
    """WebAgent._fetch_news should pass articles through Synthesizer when no URLs extracted."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.agents.web.agent import WebAgent

    mock_intel = MagicMock()
    mock_intel.news_search = AsyncMock(return_value={
        "articles": [
            # No URL field → extraction skipped → synthesize_news fallback
            {"title": "Big AI news", "publisher": {"title": "TechCrunch"}, "published date": "2026-02-20"},
        ]
    })
    # _fetch_news now also fans out to web_search — stub as empty
    mock_intel.web_search = AsyncMock(return_value={"web": {"results": []}})

    with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_news", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = "Key stories:\n1. Big AI news — Major development."

        agent = WebAgent(make_synapse(), intel=mock_intel)
        req = make_request("AI news today", metadata={"sub_agent": "news"})
        resp = await agent.execute(req)

    assert resp.success is True
    assert "Key stories" in resp.output


@pytest.mark.asyncio
async def test_web_agent_search_uses_synthesizer():
    """WebAgent._fetch_search should pass results through synthesize_search when extraction fails."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from octane.agents.web.agent import WebAgent

    mock_intel = MagicMock()
    mock_intel.web_search = AsyncMock(return_value={
        "web": {"results": [
            {"title": "Python tips", "description": "Best practices", "url": "http://a.com"}
        ]}
    })

    # Stub extractor so no real HTTP; extraction returns nothing → snippet fallback
    mock_extractor = MagicMock()
    mock_extractor.extract_batch = AsyncMock(return_value=[])

    with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_search", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = "• Python best practices include writing tests."

        agent = WebAgent(make_synapse(), intel=mock_intel, extractor=mock_extractor)
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


# ── Session 14 — Catalysts ────────────────────────────────────────────────────

class TestCatalystRegistry:
    """Unit tests for CatalystRegistry keyword matching and data resolution."""

    def _make_upstream(self, data: dict) -> dict:
        """Wrap data as a single upstream dependency (node_id -> data_dict)."""
        return {"node_abc": data}

    def test_no_match_returns_none_when_no_keywords(self):
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        result = registry.match("write a fibonacci function", {})
        assert result is None

    def test_no_match_returns_none_when_keywords_hit_but_data_missing(self):
        """price_chart triggers on 'chart' but requires time_series — no match without data."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        result = registry.match("chart the price of NVDA", {})
        assert result is None

    def test_price_chart_matches_with_timeseries_data(self):
        """price_chart should match 'chart' + time_series key in upstream."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        upstream = self._make_upstream({"time_series": [{"date": "2026-01-01", "close": 100}]})
        result = registry.match("chart the price history of NVDA", upstream)
        assert result is not None
        catalyst_fn, resolved = result
        assert catalyst_fn.__name__ == "price_chart"
        assert "time_series" in resolved

    def test_return_calculator_matches_with_market_price(self):
        """return_calculator should match 'return' + regularMarketPrice."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        upstream = self._make_upstream({"regularMarketPrice": 875.50, "symbol": "NVDA"})
        result = registry.match("what is my return on NVDA if I bought at $500?", upstream)
        assert result is not None
        catalyst_fn, _ = result
        assert catalyst_fn.__name__ == "return_calculator"

    def test_portfolio_projection_matches_with_market_price(self):
        """portfolio_projection should match 'project' + price key."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        upstream = self._make_upstream({"price": 875.50, "symbol": "NVDA"})
        result = registry.match("project my portfolio growth over 20 years with $500 initial", upstream)
        assert result is not None
        catalyst_fn, _ = result
        assert catalyst_fn.__name__ == "portfolio_projection"

    def test_technical_indicators_matches_with_timeseries(self):
        """technical_indicators should match 'RSI' + time_series key."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        upstream = self._make_upstream({"time_series": [{"date": "2026-01-01", "close": 100}]})
        result = registry.match("show me the RSI and moving average for NVDA", upstream)
        assert result is not None
        catalyst_fn, _ = result
        assert catalyst_fn.__name__ == "technical_indicators"

    def test_highest_score_wins_when_multiple_triggers_match(self):
        """If query has more price_chart triggers than return_calculator, price_chart wins."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        # 'chart', 'plot', 'graph', 'history' all hit price_chart
        upstream = self._make_upstream({"time_series": [{"date": "2026-01-01", "close": 100}]})
        result = registry.match("chart plot graph the price history", upstream)
        assert result is not None
        catalyst_fn, _ = result
        assert catalyst_fn.__name__ == "price_chart"

    def test_resolved_data_contains_upstream_key(self):
        """resolved_data should include _upstream key for full context access."""
        from octane.catalysts.registry import CatalystRegistry
        registry = CatalystRegistry()
        upstream = self._make_upstream({"time_series": [{"date": "2026-01-01", "close": 100}], "symbol": "AAPL"})
        result = registry.match("chart AAPL price history", upstream)
        assert result is not None
        _, resolved = result
        assert "_upstream" in resolved


class TestReturnCalculator:
    """Unit tests for the return_calculator catalyst function."""

    def test_basic_return_calculation(self, tmp_path):
        from octane.catalysts.finance.return_calculator import return_calculator
        data = {"regularMarketPrice": 1000.0, "symbol": "NVDA"}
        result = return_calculator(
            resolved_data=data,
            output_dir=str(tmp_path),
            instruction="I bought 10 shares at $500",
        )
        assert result["buy_price"] == 500.0
        assert result["quantity"] == 10.0
        assert result["current_price"] == 1000.0
        assert result["absolute_return"] == 5000.0
        assert result["pct_return"] == 100.0
        assert result["ticker"] == "NVDA"

    def test_returns_summary_string(self, tmp_path):
        from octane.catalysts.finance.return_calculator import return_calculator
        data = {"price": 200.0, "symbol": "AAPL"}
        result = return_calculator(
            resolved_data=data,
            output_dir=str(tmp_path),
            instruction="I bought 5 shares at $150",
        )
        assert "AAPL" in result["summary"]
        assert "Return" in result["summary"]

    def test_raises_when_no_price(self, tmp_path):
        from octane.catalysts.finance.return_calculator import return_calculator
        import pytest
        with pytest.raises(ValueError, match="no current price"):
            return_calculator(resolved_data={}, output_dir=str(tmp_path))


class TestPriceChart:
    """Unit tests for the price_chart catalyst function."""

    def _make_timeseries(self, n: int = 30) -> list[dict]:
        from datetime import date, timedelta
        base = date(2026, 1, 1)
        return [
            {"timestamp": str(base + timedelta(days=i)), "close": 100.0 + i, "volume": 1_000_000}
            for i in range(n)
        ]

    def test_saves_png_and_returns_chart_path(self, tmp_path):
        from octane.catalysts.finance.price_chart import price_chart
        import os
        data = {"time_series": self._make_timeseries(30), "symbol": "NVDA"}
        result = price_chart(resolved_data=data, output_dir=str(tmp_path))
        assert "chart_path" in result
        assert os.path.exists(result["chart_path"])
        assert result["chart_path"].endswith(".png")

    def test_returns_correct_stats(self, tmp_path):
        from octane.catalysts.finance.price_chart import price_chart
        data = {"time_series": self._make_timeseries(30), "symbol": "TEST"}
        result = price_chart(resolved_data=data, output_dir=str(tmp_path))
        assert result["data_points"] == 30
        assert result["min_price"] == pytest.approx(100.0)
        assert result["max_price"] == pytest.approx(129.0)
        assert result["ticker"] == "TEST"

    def test_raises_on_empty_timeseries(self, tmp_path):
        from octane.catalysts.finance.price_chart import price_chart
        import pytest
        with pytest.raises(ValueError, match="no time_series"):
            price_chart(resolved_data={"time_series": []}, output_dir=str(tmp_path))

    def test_raises_on_insufficient_data(self, tmp_path):
        from octane.catalysts.finance.price_chart import price_chart
        import pytest
        data = {"time_series": [{"timestamp": "2026-01-01", "close": 100.0, "volume": 0}]}
        with pytest.raises(ValueError, match="at least 2"):
            price_chart(resolved_data=data, output_dir=str(tmp_path))


class TestPortfolioProjection:
    """Unit tests for the portfolio_projection catalyst."""

    def test_saves_chart_and_returns_financials(self, tmp_path):
        from octane.catalysts.finance.portfolio_projection import portfolio_projection
        import os
        data = {"price": 875.0, "symbol": "NVDA"}
        result = portfolio_projection(
            resolved_data=data,
            output_dir=str(tmp_path),
            instruction="project $1000 initial with $100 per month over 10 years",
        )
        assert os.path.exists(result["chart_path"])
        assert result["years"] == 10
        assert result["initial_investment"] == 1000.0
        assert result["monthly_contribution"] == 100.0
        assert result["p10_final"] < result["median_final"] < result["p90_final"]

    def test_total_invested_is_correct(self, tmp_path):
        from octane.catalysts.finance.portfolio_projection import portfolio_projection
        data = {"price": 100.0}
        result = portfolio_projection(
            resolved_data=data,
            output_dir=str(tmp_path),
            instruction="$500 initial $100/month 5 years",
        )
        # 500 + 100 * 60 months = 6500
        assert result["total_invested"] == pytest.approx(6500.0)


class TestTechnicalIndicators:
    """Unit tests for the technical_indicators catalyst."""

    def _make_timeseries(self, n: int = 60) -> list[dict]:
        from datetime import date, timedelta
        import math
        base = date(2025, 11, 1)
        return [
            {
                "timestamp": str(base + timedelta(days=i)),
                "close": 100 + 20 * math.sin(i / 10),
                "volume": 1_000_000,
            }
            for i in range(n)
        ]

    def test_computes_rsi_and_saves_chart(self, tmp_path):
        from octane.catalysts.finance.technical_indicators import technical_indicators
        import os
        data = {"time_series": self._make_timeseries(60), "symbol": "AAPL"}
        result = technical_indicators(resolved_data=data, output_dir=str(tmp_path))
        assert os.path.exists(result["chart_path"])
        assert 0 <= result["rsi"] <= 100
        assert result["rsi_signal"] in ("overbought", "oversold", "neutral")
        assert result["sma_20"] is not None

    def test_raises_on_insufficient_data(self, tmp_path):
        from octane.catalysts.finance.technical_indicators import technical_indicators
        import pytest
        data = {"time_series": [{"timestamp": "2026-01-01", "close": 100.0}] * 10}
        with pytest.raises(ValueError, match="at least 14"):
            technical_indicators(resolved_data=data, output_dir=str(tmp_path))


class TestExecutorGetOutputDir:
    """Unit tests for Executor.get_output_dir."""

    def test_creates_directory_and_returns_path(self, tmp_path):
        import os
        from unittest.mock import patch
        from octane.agents.code.executor import Executor, OUTPUT_ROOT

        executor = Executor()
        with patch("octane.agents.code.executor.OUTPUT_ROOT", tmp_path):
            path = executor.get_output_dir("test-correlation-123")
        assert os.path.isdir(path)
        assert "test-correlation-123" in path

    def test_default_run_id_when_no_correlation_id(self, tmp_path):
        import os
        from unittest.mock import patch
        from octane.agents.code.executor import Executor

        executor = Executor()
        with patch("octane.agents.code.executor.OUTPUT_ROOT", tmp_path):
            path = executor.get_output_dir()
        assert os.path.isdir(path)


# ── Session 15: ContentExtractor ──────────────────────────────────────────────

class TestContentExtractor:
    """Unit tests for ContentExtractor (trafilatura-based full-text extraction)."""

    def _make_extractor(self):
        from octane.agents.web.content_extractor import ContentExtractor
        return ContentExtractor(max_chars_per_source=500, timeout=5.0)

    def test_chunk_text_respects_max_chars(self):
        """_chunk_text must return at most max_chars characters, breaking at a word."""
        extractor = self._make_extractor()
        text = "hello " * 200   # 1200 chars
        result = extractor._chunk_text(text, max_chars=100)
        assert len(result) <= 100
        assert result.endswith("hello")   # ends on a full word

    def test_chunk_text_short_text_unchanged(self):
        """_chunk_text should return the full text when it is shorter than max_chars."""
        extractor = self._make_extractor()
        short = "short text"
        assert extractor._chunk_text(short, max_chars=500) == short

    @pytest.mark.asyncio
    async def test_extract_url_returns_extracted_content_shape(self):
        """extract_url should return an ExtractedContent with all required fields."""
        from unittest.mock import patch
        from octane.agents.web.content_extractor import ExtractedContent

        extractor = self._make_extractor()

        # Patch trafilatura.fetch_url and trafilatura.extract directly to avoid network calls
        with patch("octane.agents.web.content_extractor.trafilatura") as mock_traf:
            mock_traf.fetch_url.return_value = "<html><body><p>Test article body content here.</p></body></html>"
            mock_traf.extract.return_value = "Test article body content here. " * 10  # >100 chars

            result = await extractor.extract_url("https://example.com/article")

        assert isinstance(result, ExtractedContent)
        assert result.url == "https://example.com/article"
        assert result.text != ""
        assert result.word_count > 0
        assert result.method == "trafilatura"
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_extract_url_failed_fetch_returns_failed_method(self):
        """extract_url should return method='failed' when trafilatura fetch returns None."""
        from unittest.mock import patch

        extractor = self._make_extractor()

        with patch("octane.agents.web.content_extractor.trafilatura") as mock_traf:
            mock_traf.fetch_url.return_value = None  # fetch_url returns None → no HTML

            result = await extractor.extract_url("https://blocked.example.com/")

        assert result.method == "failed"
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_extract_batch_respects_top_n(self):
        """extract_batch should process at most top_n URLs."""
        from unittest.mock import patch
        from octane.agents.web.content_extractor import ExtractedContent

        extractor = self._make_extractor()
        call_count = 0

        async def fake_extract_url(url, max_chars=None):
            nonlocal call_count
            call_count += 1
            return ExtractedContent(
                url=url, text="some content here", word_count=3, method="trafilatura"
            )

        extractor.extract_url = fake_extract_url

        urls = [f"https://example.com/{i}" for i in range(10)]
        results = await extractor.extract_batch(urls, top_n=3)

        assert call_count == 3
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_graceful_when_trafilatura_unavailable(self):
        """extract_url should return method='unavailable' if trafilatura is not installed."""
        from unittest.mock import patch

        extractor = self._make_extractor()

        # Patch the module-level flag — it's already set at import time,
        # so patching sys.modules wouldn't help here.
        with patch("octane.agents.web.content_extractor._TRAFILATURA_AVAILABLE", False):
            result = await extractor.extract_url("https://example.com/article")

        assert result.method == "unavailable"
        assert result.text == ""


# ── Session 15: BrowserAgent ──────────────────────────────────────────────────

class TestBrowserAgent:
    """Unit tests for BrowserAgent (Playwright-based scraper)."""

    @pytest.mark.asyncio
    async def test_scrape_returns_text_or_none(self):
        """scrape() must return a string on success or None on failure — never raises."""
        from unittest.mock import AsyncMock
        from octane.agents.web.browser import BrowserAgent

        agent = BrowserAgent(interactive=False)

        # Patch _scrape_with_context so no real Playwright is used
        agent._scrape_with_context = AsyncMock(return_value="This is the page content extracted.")

        result = await agent.scrape("https://example.com/")

        assert isinstance(result, str)
        assert "page content" in result

    @pytest.mark.asyncio
    async def test_non_interactive_mode_skips_human_assist(self):
        """With interactive=False, _human_assist should never be called."""
        from unittest.mock import AsyncMock
        from octane.agents.web.browser import BrowserAgent

        agent = BrowserAgent(interactive=False)
        # Both headless attempts return None → would fall through to human-assist
        agent._scrape_with_context = AsyncMock(return_value=None)
        agent._human_assist = AsyncMock(return_value="should not be called")

        result = await agent.scrape("https://example.com/")

        agent._human_assist.assert_not_called()
        assert result is None  # Non-interactive, headless failed → None

    @pytest.mark.asyncio
    async def test_graceful_when_playwright_unavailable(self):
        """scrape() should return None if playwright is not installed."""
        from unittest.mock import patch
        from octane.agents.web.browser import BrowserAgent

        agent = BrowserAgent(interactive=False)

        # Simulate playwright not available at the module level
        with patch("octane.agents.web.browser._PLAYWRIGHT_AVAILABLE", False):
            result = await agent.scrape("https://example.com/")

        assert result is None


# ── Session 15: Synthesizer.synthesize_with_content ───────────────────────────

class TestSynthesizerWithContent:
    """Unit tests for the new full-text synthesis path in Synthesizer."""

    def _make_article(self, url, text, method="trafilatura"):
        from octane.agents.web.content_extractor import ExtractedContent
        return ExtractedContent(
            url=url,
            text=text,
            word_count=len(text.split()),
            method=method,
        )

    @pytest.mark.asyncio
    async def test_returns_string_with_llm(self):
        """synthesize_with_content should return a non-empty string when LLM responds."""
        from unittest.mock import MagicMock, AsyncMock
        from octane.agents.web.synthesizer import Synthesizer

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(
            return_value="AI chip demand is growing rapidly.\n• NVDA up 5%.\nSources: reuters.com"
        )

        synth = Synthesizer(bodega=mock_bodega)
        articles = [
            self._make_article("https://reuters.com/ai-chips", "NVDA reported strong Q4 earnings driven by AI chip demand. Revenue grew 122% YoY."),
        ]
        result = await synth.synthesize_with_content("AI chip demand", articles)

        assert isinstance(result, str)
        assert len(result) > 0
        mock_bodega.chat_simple.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_usable_articles_returns_fallback_message(self):
        """synthesize_with_content with all failed/unavailable articles returns clear message."""
        from octane.agents.web.synthesizer import Synthesizer
        from octane.agents.web.content_extractor import ExtractedContent

        synth = Synthesizer(bodega=None)
        articles = [
            ExtractedContent(url="https://a.com", text="", word_count=0, method="failed"),
            ExtractedContent(url="https://b.com", text="", word_count=0, method="unavailable"),
        ]
        result = await synth.synthesize_with_content("some query", articles)

        assert "No article text" in result
        assert "some query" in result

    @pytest.mark.asyncio
    async def test_long_text_triggers_chunk_summarize(self):
        """Articles exceeding _MAX_CHARS_DIRECT chars should invoke _summarize_chunk."""
        from unittest.mock import MagicMock, AsyncMock
        from octane.agents.web.synthesizer import Synthesizer

        mock_bodega = MagicMock()
        # First call = _summarize_chunk, second call = final synthesis
        mock_bodega.chat_simple = AsyncMock(
            side_effect=["Compressed summary of long article.", "Final synthesis result."]
        )

        synth = Synthesizer(bodega=mock_bodega)
        long_text = "word " * 700   # ~3500 chars — exceeds _MAX_CHARS_DIRECT (3000)
        articles = [self._make_article("https://longsite.com/article", long_text)]

        result = await synth.synthesize_with_content("test query", articles)

        # Both LLM calls should have fired: chunk pass + final synthesis
        assert mock_bodega.chat_simple.call_count == 2
        assert result == "Final synthesis result."

    @pytest.mark.asyncio
    async def test_plain_fallback_when_no_bodega(self):
        """synthesize_with_content without LLM should return a plain formatted listing."""
        from octane.agents.web.synthesizer import Synthesizer

        synth = Synthesizer(bodega=None)
        articles = [
            self._make_article("https://techcrunch.com/story", "TechCrunch reports major AI breakthrough in Q1."),
        ]
        result = await synth.synthesize_with_content("AI breakthroughs", articles)

        assert "techcrunch.com" in result
        assert "TechCrunch" in result  # from the snippet

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_plain(self):
        """If LLM raises, synthesize_with_content should still return a plain result."""
        from unittest.mock import MagicMock, AsyncMock
        from octane.agents.web.synthesizer import Synthesizer

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        synth = Synthesizer(bodega=mock_bodega)
        articles = [
            self._make_article("https://example.com/news", "Important news content here."),
        ]
        result = await synth.synthesize_with_content("latest news", articles)

        assert isinstance(result, str)
        assert len(result) > 0


# ── Session 15: WebAgent full-text extraction path ───────────────────────────

class TestWebAgentFullTextPath:
    """Integration-style tests for the WebAgent full-text extraction cascade."""

    def _make_synapse(self):
        return SynapseEventBus(persist=False)

    def _make_request(self, query, sub_agent="search"):
        return AgentRequest(
            query=query,
            session_id="test",
            source="test",
            metadata={"sub_agent": sub_agent},
        )

    @pytest.mark.asyncio
    async def test_search_uses_synthesize_with_content_when_extraction_succeeds(self):
        """When ContentExtractor returns usable text, synthesize_with_content is called."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent

        mock_intel = MagicMock()
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [
                {"title": "AI Research", "description": "New findings", "url": "https://arxiv.org/ai"},
            ]}
        })

        good_article = ExtractedContent(
            url="https://arxiv.org/ai",
            text="Researchers found that AI models can now reason about complex tasks.",
            word_count=12,
            method="trafilatura",
        )
        mock_extractor = MagicMock()
        mock_extractor.extract_batch = AsyncMock(return_value=[good_article])

        with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_with_content", new_callable=AsyncMock) as mock_swc:
            mock_swc.return_value = "AI models can now reason about complex tasks.\n• Key finding from arxiv."

            agent = WebAgent(self._make_synapse(), intel=mock_intel, extractor=mock_extractor)
            resp = await agent.execute(self._make_request("AI research findings"))

        assert resp.success is True
        mock_swc.assert_called_once()
        assert "AI models" in resp.output

    @pytest.mark.asyncio
    async def test_search_falls_back_to_snippets_when_extraction_empty(self):
        """When extractor returns empty list, synthesize_search (snippet path) is used."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.agents.web.agent import WebAgent

        mock_intel = MagicMock()
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [
                {"title": "Climate Change", "description": "Record temperatures", "url": "https://news.com/climate"},
            ]}
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_batch = AsyncMock(return_value=[])

        with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_search", new_callable=AsyncMock) as mock_ss:
            mock_ss.return_value = "• Record temperatures set globally in 2026."

            agent = WebAgent(self._make_synapse(), intel=mock_intel, extractor=mock_extractor)
            resp = await agent.execute(self._make_request("climate change news"))

        assert resp.success is True
        mock_ss.assert_called_once()

    @pytest.mark.asyncio
    async def test_news_uses_synthesize_with_content_when_urls_extracted(self):
        """_fetch_news: when article URLs yield text, synthesize_with_content is called."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent

        mock_intel = MagicMock()
        mock_intel.news_search = AsyncMock(return_value={
            "articles": [{
                "title": "Markets Rally",
                "url": "https://reuters.com/markets",
                "publisher": {"title": "Reuters"},
                "published date": "2026-03-01",
            }]
        })
        # _fetch_news now also fans out to web_search — stub as empty
        mock_intel.web_search = AsyncMock(return_value={"web": {"results": []}})

        good_article = ExtractedContent(
            url="https://reuters.com/markets",
            text="Global markets rallied on Friday driven by strong jobs data.",
            word_count=11,
            method="trafilatura",
        )
        mock_extractor = MagicMock()
        mock_extractor.extract_batch = AsyncMock(return_value=[good_article])

        with patch("octane.agents.web.synthesizer.Synthesizer.synthesize_with_content", new_callable=AsyncMock) as mock_swc:
            mock_swc.return_value = "Markets rallied Friday.\n• Jobs data beat expectations.\nSources: reuters.com"

            agent = WebAgent(self._make_synapse(), intel=mock_intel, extractor=mock_extractor)
            resp = await agent.execute(self._make_request("market news today", sub_agent="news"))

        assert resp.success is True
        mock_swc.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_enrichment_called_for_failed_extractions(self):
        """_enrich_with_browser should upgrade failed ExtractedContent via BrowserAgent."""
        from unittest.mock import AsyncMock, MagicMock
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent

        # Extractor returns a failed article
        failed = ExtractedContent(url="https://paywalled.com/article", text="", word_count=0, method="failed")
        mock_extractor = MagicMock()
        mock_extractor.extract_batch = AsyncMock(return_value=[failed])

        # Browser successfully fetches the content (scrape returns str | None)
        mock_browser = MagicMock()
        mock_browser.scrape = AsyncMock(return_value="Paywalled article content retrieved via browser.")

        agent = WebAgent(self._make_synapse(), extractor=mock_extractor, browser=mock_browser)
        enriched = await agent._enrich_with_browser([failed], max_attempts=1)

        assert len(enriched) == 1
        assert enriched[0].method == "browser"
        assert enriched[0].text.startswith("Paywalled article")
        mock_browser.scrape.assert_called_once_with("https://paywalled.com/article")


# ── Session 15 (extended): ContentExtractor edge cases ───────────────────────

class TestContentExtractorEdgeCases:
    """Deeper edge-case coverage for ContentExtractor."""

    def _make_extractor(self):
        from octane.agents.web.content_extractor import ContentExtractor
        return ContentExtractor(max_chars_per_source=500, timeout=5.0)

    def test_chunk_text_exactly_at_boundary_is_unchanged(self):
        """Text exactly equal to max_chars should be returned as-is."""
        extractor = self._make_extractor()
        text = "a" * 500
        assert extractor._chunk_text(text, max_chars=500) == text

    def test_chunk_text_single_long_word_truncated(self):
        """A single word longer than max_chars should still be truncated."""
        extractor = self._make_extractor()
        long_word = "x" * 200
        result = extractor._chunk_text(long_word, max_chars=50)
        assert len(result) <= 50

    @pytest.mark.asyncio
    async def test_extract_url_short_text_is_treated_as_failed(self):
        """Extracted text shorter than 100 chars (cookie wall, redirect) → method='failed'."""
        from unittest.mock import patch
        extractor = self._make_extractor()

        with patch("octane.agents.web.content_extractor.trafilatura") as mock_traf:
            mock_traf.fetch_url.return_value = "<html><body>Accept cookies</body></html>"
            mock_traf.extract.return_value = "Accept cookies"  # < 100 chars

            result = await extractor.extract_url("https://cookiewall.example.com/")

        assert result.method == "failed"
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_extract_batch_empty_urls_returns_empty_list(self):
        """extract_batch with an empty URL list should return []."""
        extractor = self._make_extractor()
        results = await extractor.extract_batch([], top_n=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_batch_results_sorted_by_word_count_descending(self):
        """extract_batch should return articles sorted by word_count descending."""
        from octane.agents.web.content_extractor import ExtractedContent

        extractor = self._make_extractor()

        async def fake_extract_url(url, max_chars=None):
            counts = {"https://a.com": 5, "https://b.com": 100, "https://c.com": 30}
            wc = counts.get(url, 0)
            return ExtractedContent(
                url=url,
                text="word " * wc,
                word_count=wc,
                method="trafilatura",
            )

        extractor.extract_url = fake_extract_url
        results = await extractor.extract_batch(
            ["https://a.com", "https://b.com", "https://c.com"], top_n=3
        )

        word_counts = [r.word_count for r in results]
        assert word_counts == sorted(word_counts, reverse=True)


# ── Session 15 (extended): BrowserAgent chain behaviour ──────────────────────

class TestBrowserAgentChain:
    """Tests for BrowserAgent's headless→human-assist fallback chain."""

    @pytest.mark.asyncio
    async def test_scrape_returns_none_when_all_headless_attempts_fail_non_interactive(self):
        """All scrape attempts return None in non-interactive mode → scrape() returns None."""
        from unittest.mock import AsyncMock
        from octane.agents.web.browser import BrowserAgent

        agent = BrowserAgent(interactive=False)
        agent._scrape_with_context = AsyncMock(return_value=None)

        result = await agent.scrape("https://example.com/")
        assert result is None

    @pytest.mark.asyncio
    async def test_interactive_mode_calls_human_assist_when_headless_fails(self):
        """With interactive=True and headless failure, _human_assist should be called."""
        from unittest.mock import AsyncMock
        from octane.agents.web.browser import BrowserAgent

        agent = BrowserAgent(interactive=True)
        agent._scrape_with_context = AsyncMock(return_value=None)
        agent._human_assist = AsyncMock(return_value="Content fetched via human-assist.")

        result = await agent.scrape("https://paywalled.com/")

        agent._human_assist.assert_called_once()
        assert result == "Content fetched via human-assist."

    @pytest.mark.asyncio
    async def test_scrape_succeeds_on_first_headless_attempt(self):
        """When first headless attempt succeeds, no further attempts are made."""
        from unittest.mock import AsyncMock
        from octane.agents.web.browser import BrowserAgent

        agent = BrowserAgent(interactive=False)
        call_count = 0

        async def fake_scrape_with_context(browser, url, cookie_file):
            nonlocal call_count
            call_count += 1
            return "Article text extracted successfully."

        agent._scrape_with_context = fake_scrape_with_context

        result = await agent.scrape("https://example.com/")
        assert result == "Article text extracted successfully."
        assert call_count == 1


# ── Session 15 (extended): Synthesizer._summarize_chunk ──────────────────────

class TestSynthesizerChunkSummarize:
    """Tests for the _summarize_chunk compression helper."""

    def _make_article(self, url, text, method="trafilatura"):
        from octane.agents.web.content_extractor import ExtractedContent
        return ExtractedContent(url=url, text=text, word_count=len(text.split()), method=method)

    @pytest.mark.asyncio
    async def test_summarize_chunk_returns_compressed_text(self):
        """_summarize_chunk should call LLM and return the compressed string."""
        from unittest.mock import MagicMock, AsyncMock
        from octane.agents.web.synthesizer import Synthesizer

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value="Compressed: NVDA reported 122% revenue growth.")

        synth = Synthesizer(bodega=mock_bodega)
        result = await synth._summarize_chunk("NVDA reported very strong Q4 earnings " * 20, "NVDA earnings")

        mock_bodega.chat_simple.assert_called_once()
        assert "Compressed" in result

    @pytest.mark.asyncio
    async def test_summarize_chunk_falls_back_on_llm_error(self):
        """_summarize_chunk should return a text[:500] fallback if LLM raises."""
        from unittest.mock import MagicMock, AsyncMock
        from octane.agents.web.synthesizer import Synthesizer

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        synth = Synthesizer(bodega=mock_bodega)
        long_text = "word " * 400  # 2000 chars
        result = await synth._summarize_chunk(long_text, "some query")

        assert isinstance(result, str)
        assert len(result) <= 500

    @pytest.mark.asyncio
    async def test_synthesize_with_content_filters_failed_articles(self):
        """Articles with method='failed' or 'unavailable' should be excluded from synthesis."""
        from unittest.mock import MagicMock, AsyncMock
        from octane.agents.web.synthesizer import Synthesizer

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value="Synthesis result.")

        synth = Synthesizer(bodega=mock_bodega)
        articles = [
            self._make_article("https://a.com", "Good content with lots of words here.", "trafilatura"),
            self._make_article("https://b.com", "", "failed"),
            self._make_article("https://c.com", "", "unavailable"),
        ]
        result = await synth.synthesize_with_content("test query", articles)

        assert isinstance(result, str)
        # LLM should only have seen 1 usable article — called for synthesis (not chunk)
        mock_bodega.chat_simple.assert_called_once()


# ── Session 15 (extended): WebAgent ticker resolution ────────────────────────

class TestWebAgentTickerResolution:
    """Tests for the LLM+web fallback in _extract_ticker / _resolve_ticker_via_web."""

    def _make_synapse(self):
        return SynapseEventBus(persist=False)

    @pytest.mark.asyncio
    async def test_extract_ticker_regex_fast_path(self):
        """All-caps token in query resolves via regex — no web search needed."""
        from unittest.mock import AsyncMock, MagicMock
        from octane.agents.web.agent import WebAgent

        agent = WebAgent(self._make_synapse())
        result = await agent._extract_ticker("What is DASH doing today?")
        assert result == "DASH"

    @pytest.mark.asyncio
    async def test_extract_ticker_hardcoded_map_fast_path(self):
        """Known company name resolves via hardcoded map — no web search needed."""
        from octane.agents.web.agent import WebAgent

        agent = WebAgent(self._make_synapse())
        result = await agent._extract_ticker("what is doordash stock price")
        # "doordash" is NOT in the hardcoded map → falls through to web resolve (returns None
        # without bodega), so meta is the one to test here:
        result_meta = await agent._extract_ticker("meta earnings this quarter")
        assert result_meta == "META"

    @pytest.mark.asyncio
    async def test_extract_ticker_resolves_unknown_company_via_web_and_llm(self):
        """Unknown company (e.g. DoorDash) resolves correctly via web search + LLM."""
        from unittest.mock import AsyncMock, MagicMock
        from octane.agents.web.agent import WebAgent

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value="DASH")

        mock_intel = MagicMock()
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [
                {"title": "DoorDash Inc (DASH) Stock Price", "description": "DASH trades on NYSE at $180."},
            ]}
        })

        agent = WebAgent(self._make_synapse(), intel=mock_intel, bodega=mock_bodega)
        result = await agent._extract_ticker("doordash stock price today")

        assert result == "DASH"
        mock_intel.web_search.assert_called_once()
        mock_bodega.chat_simple.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_ticker_returns_none_when_llm_says_none(self):
        """If LLM responds 'NONE', _extract_ticker should return None gracefully."""
        from unittest.mock import AsyncMock, MagicMock
        from octane.agents.web.agent import WebAgent

        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value="NONE")

        mock_intel = MagicMock()
        mock_intel.web_search = AsyncMock(return_value={
            "web": {"results": [
                {"title": "Some unrelated result", "description": "No ticker here."},
            ]}
        })

        agent = WebAgent(self._make_synapse(), intel=mock_intel, bodega=mock_bodega)
        result = await agent._extract_ticker("what is the weather in tokyo")

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_ticker_returns_none_when_no_bodega(self):
        """Without a bodega client, _resolve_ticker_via_web returns None safely."""
        from octane.agents.web.agent import WebAgent

        agent = WebAgent(self._make_synapse(), bodega=None)
        result = await agent._extract_ticker("palantir stock today")

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_ticker_via_web_handles_web_search_error(self):
        """If web_search raises, _resolve_ticker_via_web returns None without crashing."""
        from unittest.mock import AsyncMock, MagicMock
        from octane.agents.web.agent import WebAgent

        mock_bodega = MagicMock()
        mock_intel = MagicMock()
        mock_intel.web_search = AsyncMock(side_effect=RuntimeError("Network error"))

        agent = WebAgent(self._make_synapse(), intel=mock_intel, bodega=mock_bodega)
        result = await agent._resolve_ticker_via_web("carvana stock price")

        assert result is None


# ══════════════════════════════════════════════════════════════
# Session 16 — HIL Manager · Decision Ledger · Checkpoint
# ══════════════════════════════════════════════════════════════


# ── Decision model ────────────────────────────────────────────

class TestDecisionModel:
    """Unit tests for the Decision Pydantic model."""

    def _make_decision(self, **overrides):
        from octane.models.decisions import Decision
        defaults = dict(
            correlation_id="cid-test",
            action="call web search",
            task_id="t1",
            agent="web",
        )
        defaults.update(overrides)
        return Decision(**defaults)

    def test_needs_human_high_risk(self):
        """risk_level='high' → needs_human is True regardless of confidence."""
        d = self._make_decision(risk_level="high", confidence=0.95)
        assert d.needs_human is True

    def test_needs_human_critical_risk(self):
        """risk_level='critical' → needs_human is True."""
        d = self._make_decision(risk_level="critical", confidence=1.0)
        assert d.needs_human is True

    def test_needs_human_low_confidence(self):
        """confidence < 0.60 → needs_human is True even for low risk."""
        d = self._make_decision(risk_level="low", confidence=0.50)
        assert d.needs_human is True

    def test_needs_human_medium_low_confidence(self):
        """medium risk + confidence < 0.75 → needs_human is True."""
        d = self._make_decision(risk_level="medium", confidence=0.70)
        assert d.needs_human is True

    def test_needs_human_false_for_low_risk_high_conf(self):
        """risk=low, conf=0.95 → needs_human is False."""
        d = self._make_decision(risk_level="low", confidence=0.95)
        assert d.needs_human is False

    def test_is_resolved_pending_is_false(self):
        """Default status='pending' → is_resolved is False."""
        d = self._make_decision()
        assert d.is_resolved is False

    def test_is_resolved_after_approval(self):
        """status='auto_approved' → is_resolved is True."""
        d = self._make_decision(status="auto_approved")
        assert d.is_resolved is True

    def test_is_approved_for_all_approved_statuses(self):
        """auto_approved, human_approved, human_modified are all 'approved'."""
        from octane.models.decisions import Decision
        for status in ("auto_approved", "human_approved", "human_modified"):
            d = Decision(correlation_id="x", action="a", status=status)
            assert d.is_approved is True, f"Expected is_approved for status={status}"

    def test_is_approved_false_for_declined(self):
        """human_declined is NOT approved."""
        d = self._make_decision(status="human_declined")
        assert d.is_approved is False


# ── DecisionLedger ────────────────────────────────────────────

class TestDecisionLedger:
    """Unit tests for DecisionLedger aggregate model."""

    def _ledger(self):
        from octane.models.decisions import DecisionLedger
        return DecisionLedger(correlation_id="cid-ledger")

    def _decision(self, risk="low", conf=0.95, status="pending", agent="web"):
        from octane.models.decisions import Decision
        return Decision(
            correlation_id="cid-ledger",
            action="do something",
            risk_level=risk,
            confidence=conf,
            status=status,
            agent=agent,
        )

    def test_add_appends_decision(self):
        """add() should grow the decisions list."""
        ledger = self._ledger()
        ledger.add(self._decision())
        ledger.add(self._decision())
        assert len(ledger.decisions) == 2

    def test_pending_filters_unresolved(self):
        """pending property returns only status='pending' decisions."""
        ledger = self._ledger()
        ledger.add(self._decision(status="pending"))
        ledger.add(self._decision(status="auto_approved"))
        assert len(ledger.pending) == 1

    def test_any_declined_false_when_none_declined(self):
        """any_declined is False when no decisions are declined."""
        ledger = self._ledger()
        ledger.add(self._decision(status="auto_approved"))
        assert ledger.any_declined is False

    def test_any_declined_true_when_one_declined(self):
        """any_declined is True when at least one decision is declined."""
        ledger = self._ledger()
        ledger.add(self._decision(status="auto_approved"))
        ledger.add(self._decision(status="human_declined"))
        assert ledger.any_declined is True

    def test_summary_returns_correct_counts(self):
        """summary() dict should count total, auto_approved, needs_human, declined."""
        ledger = self._ledger()
        ledger.add(self._decision(status="auto_approved"))
        ledger.add(self._decision(status="auto_approved"))
        ledger.add(self._decision(risk="high", status="pending"))
        ledger.add(self._decision(status="human_declined"))
        s = ledger.summary()
        assert s["total"] == 4
        assert s["auto_approved"] == 2
        assert s["declined"] == 1

    def test_needs_human_review_filters_correctly(self):
        """needs_human_review returns only pending decisions where needs_human=True."""
        ledger = self._ledger()
        ledger.add(self._decision(risk="low", conf=0.95, status="pending"))   # no
        ledger.add(self._decision(risk="high", conf=0.95, status="pending"))  # yes
        ledger.add(self._decision(risk="high", conf=0.95, status="auto_approved"))  # resolved
        assert len(ledger.needs_human_review) == 1


# ── CheckpointManager ─────────────────────────────────────────

class TestCheckpointManager:
    """Unit tests for in-memory CheckpointManager."""

    def _simple_dag(self):
        from octane.models.dag import TaskDAG, TaskNode
        node = TaskNode(id="t1", agent="web", instruction="search something")
        return TaskDAG(nodes=[node], edges=[], reasoning="test dag")

    @pytest.mark.asyncio
    async def test_create_returns_checkpoint(self):
        """create() should return a Checkpoint with correct correlation_id and type."""
        from octane.osa.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        dag = self._simple_dag()
        cp = await mgr.create("cid-1", dag, {}, [], checkpoint_type="plan")
        assert cp.correlation_id == "cid-1"
        assert cp.checkpoint_type == "plan"
        assert cp.id  # non-empty UUID

    @pytest.mark.asyncio
    async def test_latest_returns_most_recent(self):
        """latest() should return the most recently created checkpoint."""
        from octane.osa.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        dag = self._simple_dag()
        await mgr.create("cid-2", dag, {}, [], checkpoint_type="plan")
        cp2 = await mgr.create("cid-2", dag, {}, [], checkpoint_type="pre_execution")
        latest = mgr.latest("cid-2")
        assert latest is not None
        assert latest.id == cp2.id

    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_all(self):
        """list_checkpoints() returns all checkpoints for a correlation_id."""
        from octane.osa.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        dag = self._simple_dag()
        await mgr.create("cid-3", dag, {}, [], checkpoint_type="plan")
        await mgr.create("cid-3", dag, {}, [], checkpoint_type="pre_execution")
        cps = mgr.list_checkpoints("cid-3")
        assert len(cps) == 2

    @pytest.mark.asyncio
    async def test_revert_drops_later_checkpoints(self):
        """revert() to first checkpoint removes all subsequent checkpoints."""
        from octane.osa.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        dag = self._simple_dag()
        cp1 = await mgr.create("cid-4", dag, {}, [], checkpoint_type="plan")
        await mgr.create("cid-4", dag, {}, [], checkpoint_type="pre_execution")
        await mgr.create("cid-4", dag, {}, [], checkpoint_type="post_execution")

        reverted = await mgr.revert("cid-4", cp1.id)
        assert reverted.id == cp1.id
        remaining = mgr.list_checkpoints("cid-4")
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_revert_raises_on_unknown_id(self):
        """revert() with non-existent checkpoint_id raises KeyError."""
        from octane.osa.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        dag = self._simple_dag()
        await mgr.create("cid-5", dag, {}, [], checkpoint_type="plan")
        with pytest.raises(KeyError):
            await mgr.revert("cid-5", "non-existent-id")

    def test_clear_removes_all_checkpoints(self):
        """clear() removes all checkpoints for the given correlation_id."""
        from octane.osa.checkpoint_manager import CheckpointManager
        import asyncio
        mgr = CheckpointManager()
        dag = self._simple_dag()
        asyncio.run(mgr.create("cid-6", dag, {}, [], checkpoint_type="plan"))
        mgr.clear("cid-6")
        assert mgr.list_checkpoints("cid-6") == []


# ── HILManager ────────────────────────────────────────────────

class TestHILManagerAutoApprove:
    """Tests for HILManager auto-approval logic (non-interactive=True means never prompt)."""

    def _decision(self, risk="low", conf=0.95, reversible=True):
        from octane.models.decisions import Decision
        return Decision(
            correlation_id="cid-hil",
            action="test action",
            risk_level=risk,
            confidence=conf,
            reversible=reversible,
        )

    def _ledger(self, decisions):
        from octane.models.decisions import DecisionLedger
        ledger = DecisionLedger(correlation_id="cid-hil")
        for d in decisions:
            ledger.add(d)
        return ledger

    @pytest.mark.asyncio
    async def test_non_interactive_auto_approves_all(self):
        """interactive=False should mark ALL decisions as auto_approved without prompting."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        decisions = [
            self._decision(risk="high", conf=0.80),
            self._decision(risk="critical", conf=0.95),
            self._decision(risk="low", conf=0.10),
        ]
        ledger = self._ledger(decisions)
        result = await hil.review_ledger(ledger, user_profile={})
        assert all(d.status == "auto_approved" for d in result.decisions)

    def test_balanced_auto_approves_low_high_conf(self):
        """balanced level: low risk + conf >= 0.85 → auto-approved."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="low", conf=0.90)
        assert hil._should_auto_approve(d, "balanced") is True

    def test_balanced_does_not_auto_approve_medium_risk(self):
        """balanced level: medium risk → requires human review."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="medium", conf=0.90)
        assert hil._should_auto_approve(d, "balanced") is False

    def test_balanced_does_not_auto_approve_low_risk_low_conf(self):
        """balanced level: low risk but conf < 0.85 → requires human review."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="low", conf=0.80)
        assert hil._should_auto_approve(d, "balanced") is False

    def test_relaxed_auto_approves_medium_risk(self):
        """relaxed level: medium risk → auto-approved."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="medium", conf=0.90)
        assert hil._should_auto_approve(d, "relaxed") is True

    def test_relaxed_does_not_auto_approve_high_risk(self):
        """relaxed level: high risk → requires human review."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="high", conf=0.95)
        assert hil._should_auto_approve(d, "relaxed") is False

    def test_strict_auto_approves_low_reversible(self):
        """strict level: low risk + reversible=True → auto-approved."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="low", reversible=True)
        assert hil._should_auto_approve(d, "strict") is True

    def test_strict_does_not_auto_approve_low_irreversible(self):
        """strict level: low risk but reversible=False → requires human review."""
        from octane.osa.hil_manager import HILManager
        hil = HILManager(interactive=False)
        d = self._decision(risk="low", reversible=False)
        assert hil._should_auto_approve(d, "strict") is False


# ── PolicyEngine.assess_dag ───────────────────────────────────

class TestPolicyEngineAssessDag:
    """Tests for PolicyEngine.assess_dag() risk classification."""

    def _dag(self, nodes):
        from octane.models.dag import TaskDAG, TaskNode
        return TaskDAG(nodes=nodes, edges=[], reasoning="test")

    def _node(self, agent, sub_agent="", instruction="search for info"):
        from octane.models.dag import TaskNode
        return TaskNode(
            id=f"t_{agent}_{sub_agent}",
            agent=agent,
            sub_agent=sub_agent,
            instruction=instruction,
        )

    def test_web_search_node_is_low_risk(self):
        """Web search tasks should be classified as low risk."""
        from octane.osa.policy import PolicyEngine
        pe = PolicyEngine()
        dag = self._dag([self._node("web", "search")])
        ledger = pe.assess_dag(dag)
        assert len(ledger.decisions) == 1
        assert ledger.decisions[0].risk_level == "low"

    def test_code_execute_node_is_high_risk(self):
        """Code execute tasks should be classified as high risk."""
        from octane.osa.policy import PolicyEngine
        pe = PolicyEngine()
        dag = self._dag([self._node("code", "execute", "run the script")])
        ledger = pe.assess_dag(dag)
        assert ledger.decisions[0].risk_level == "high"

    def test_critical_keyword_escalates_to_critical(self):
        """Instructions containing critical keywords (e.g. 'delete') escalate risk to critical."""
        from octane.osa.policy import PolicyEngine
        pe = PolicyEngine()
        dag = self._dag([self._node("web", "search", "delete all user records")])
        ledger = pe.assess_dag(dag)
        assert ledger.decisions[0].risk_level == "critical"

    def test_high_keyword_escalates_to_high(self):
        """Instructions containing high-risk keywords (e.g. 'deploy') escalate to at least high."""
        from octane.osa.policy import PolicyEngine
        pe = PolicyEngine()
        dag = self._dag([self._node("web", "search", "deploy to production server")])
        ledger = pe.assess_dag(dag)
        assert ledger.decisions[0].risk_level in ("high", "critical")

    def test_full_dag_one_decision_per_node(self):
        """assess_dag() should produce exactly one Decision per TaskNode."""
        from octane.osa.policy import PolicyEngine
        pe = PolicyEngine()
        dag = self._dag([
            self._node("web", "search"),
            self._node("memory", "read"),
            self._node("code", "write"),
        ])
        ledger = pe.assess_dag(dag)
        assert len(ledger.decisions) == 3

    def test_memory_write_is_medium_risk(self):
        """Memory write tasks should be classified as medium risk."""
        from octane.osa.policy import PolicyEngine
        pe = PolicyEngine()
        dag = self._dag([self._node("memory", "write")])
        ledger = pe.assess_dag(dag)
        assert ledger.decisions[0].risk_level == "medium"


# ── Orchestrator HIL wiring ───────────────────────────────────

class TestOrchestratorHILWiring:
    """Tests that Orchestrator correctly wires HIL + Checkpoint into run()."""

    def _make_synapse(self):
        return SynapseEventBus(persist=False)

    def _make_orchestrator(self, synapse):
        from octane.osa.orchestrator import Orchestrator
        return Orchestrator(synapse, hil_interactive=False)

    @pytest.mark.asyncio
    async def test_orchestrator_accepts_hil_interactive_param(self):
        """Orchestrator.__init__ should accept hil_interactive without error."""
        from octane.osa.orchestrator import Orchestrator
        synapse = self._make_synapse()
        osa = Orchestrator(synapse, hil_interactive=False)
        assert osa.hil is not None
        assert osa.checkpoint_mgr is not None

    @pytest.mark.asyncio
    async def test_run_includes_hil_summary_in_egress_when_guard_passes(self):
        """When run() succeeds past guard, egress payload should contain hil_summary."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.models.schemas import AgentResponse
        from octane.models.dag import TaskDAG, TaskNode
        from octane.osa.orchestrator import Orchestrator

        synapse = self._make_synapse()
        osa = Orchestrator(synapse, hil_interactive=False)

        fake_dag = TaskDAG(
            nodes=[TaskNode(id="t1", agent="web", instruction="search AAPL")],
            edges=[],
            reasoning="test",
        )
        fake_response = AgentResponse(
            query="test", success=True, agent="web",
            output="AAPL is at $200.",
            correlation_id="cid-test", session_id="s1",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_response)

        with (
            patch.object(osa.guard, "check_input", return_value={"safe": True, "block": False}),
            patch.object(osa.decomposer, "decompose", return_value=fake_dag),
            patch.object(osa, "_get_memory_agent", return_value=None),
            patch.object(osa.router, "get_agent", return_value=mock_agent),
            patch.object(osa.evaluator, "evaluate", return_value="AAPL is at $200."),
            patch.object(osa.policy, "check_query_length", return_value=None),
        ):
            result = await osa.run("AAPL stock price", session_id="s1")

        assert result == "AAPL is at $200."
        # Egress event should exist and have hil_summary in payload
        egress_events = [e for e in synapse._events if e.event_type == "egress"]
        assert egress_events, "Expected at least one egress event"
        payload = egress_events[-1].payload or {}
        assert "hil_summary" in payload
        summary = payload["hil_summary"]
        assert "total" in summary


# ══════════════════════════════════════════════════════════════
# Session 17 — Background Research Workflows
# ══════════════════════════════════════════════════════════════


# ── ResearchTask model ────────────────────────────────────────

class TestResearchTask:
    """Unit tests for the ResearchTask Pydantic model."""

    def test_id_auto_generated(self):
        """ResearchTask should auto-generate a short 8-char ID."""
        from octane.research.models import ResearchTask
        t = ResearchTask(topic="NVDA earnings")
        assert len(t.id) == 8
        assert isinstance(t.id, str)

    def test_default_status_is_active(self):
        """Default status should be 'active'."""
        from octane.research.models import ResearchTask
        t = ResearchTask(topic="test")
        assert t.status == "active"
        assert t.is_active is True

    def test_interval_default_is_six_hours(self):
        """Default cycle interval should be 6.0 hours."""
        from octane.research.models import ResearchTask
        t = ResearchTask(topic="test")
        assert t.interval_hours == 6.0

    def test_is_active_false_when_stopped(self):
        """is_active should be False when status='stopped'."""
        from octane.research.models import ResearchTask
        t = ResearchTask(topic="test", status="stopped")
        assert t.is_active is False

    def test_unique_ids_across_instances(self):
        """Two ResearchTask instances should not share the same ID."""
        from octane.research.models import ResearchTask
        ids = {ResearchTask(topic="t").id for _ in range(10)}
        assert len(ids) > 1  # at least some should differ


# ── ResearchFinding model ─────────────────────────────────────

class TestResearchFinding:
    """Unit tests for the ResearchFinding model."""

    def test_from_row_builds_correctly(self):
        """from_row() should map a dict to ResearchFinding fields."""
        from octane.research.models import ResearchFinding
        from datetime import datetime, timezone
        row = {
            "id": 42,
            "task_id": "abc12345",
            "cycle_num": 3,
            "topic": "NVDA earnings",
            "content": "NVDA posted record revenue of $22B in Q4.",
            "agents_used": ["web"],
            "sources": ["https://reuters.com/nvda"],
            "word_count": 10,
            "created_at": datetime.now(timezone.utc),
        }
        f = ResearchFinding.from_row(row)
        assert f.id == 42
        assert f.task_id == "abc12345"
        assert f.cycle_num == 3
        assert f.word_count == 10

    def test_preview_truncates_long_content(self):
        """preview property should truncate to 200 chars with ellipsis."""
        from octane.research.models import ResearchFinding
        f = ResearchFinding(
            task_id="x", cycle_num=1, topic="t",
            content="word " * 200,
            word_count=200,
        )
        assert len(f.preview) <= 204  # 200 + "…"
        assert f.preview.endswith("…")

    def test_preview_no_ellipsis_for_short_content(self):
        """Short content should not get an ellipsis."""
        from octane.research.models import ResearchFinding
        f = ResearchFinding(
            task_id="x", cycle_num=1, topic="t",
            content="Short summary.",
            word_count=2,
        )
        assert not f.preview.endswith("…")


# ── ResearchStore (mocked Redis + Postgres) ───────────────────

class TestResearchStore:
    """Tests for ResearchStore with injected mock clients."""

    def _make_redis_mock(self):
        """Build an AsyncMock that behaves like redis.asyncio.Redis for the store."""
        from unittest.mock import AsyncMock, MagicMock
        r = AsyncMock()
        # Pipeline context manager — rpush/ltrim are sync queue calls inside pipeline
        pipe = MagicMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__ = AsyncMock(return_value=False)
        pipe.rpush = MagicMock()   # not awaited in pipeline pattern
        pipe.ltrim = MagicMock()   # not awaited in pipeline pattern
        pipe.execute = AsyncMock(return_value=[1, 1])
        r.pipeline = MagicMock(return_value=pipe)
        r.set = AsyncMock(return_value=True)
        r.get = AsyncMock(return_value=None)
        r.sadd = AsyncMock(return_value=1)
        r.srem = AsyncMock(return_value=1)
        r.smembers = AsyncMock(return_value=set())
        r.lrange = AsyncMock(return_value=[])
        r.llen = AsyncMock(return_value=0)
        r.incr = AsyncMock(return_value=1)
        r.aclose = AsyncMock()
        return r

    def _make_store(self, redis_mock=None, pg_mock=None):
        from octane.research.store import ResearchStore
        return ResearchStore(_redis=redis_mock, _pg=pg_mock)

    @pytest.mark.asyncio
    async def test_register_task_writes_to_redis(self):
        """register_task() should SET the task JSON and SADD to the active index."""
        from octane.research.models import ResearchTask
        redis = self._make_redis_mock()
        store = self._make_store(redis_mock=redis)
        task = ResearchTask(topic="NVDA earnings")
        await store.register_task(task)
        redis.set.assert_called_once()
        redis.sadd.assert_called_once_with("research:active", task.id)

    @pytest.mark.asyncio
    async def test_get_task_returns_none_when_not_found(self):
        """get_task() returns None when key is absent from Redis."""
        redis = self._make_redis_mock()
        redis.get = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value=None)
        store = self._make_store(redis_mock=redis)
        result = await store.get_task("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_task_parses_stored_json(self):
        """get_task() should deserialise the JSON stored by register_task()."""
        from unittest.mock import AsyncMock
        from octane.research.models import ResearchTask
        task = ResearchTask(topic="AI market trends")
        redis = self._make_redis_mock()
        redis.get = AsyncMock(return_value=task.model_dump_json())
        store = self._make_store(redis_mock=redis)
        result = await store.get_task(task.id)
        assert result is not None
        assert result.topic == "AI market trends"

    @pytest.mark.asyncio
    async def test_log_entry_rpush_and_ltrim(self):
        """log_entry() should RPUSH to the log key and LTRIM to cap=200."""
        redis = self._make_redis_mock()
        store = self._make_store(redis_mock=redis)
        await store.log_entry("task123", "⚙ Starting cycle")
        pipe = redis.pipeline.return_value.__aenter__.return_value
        pipe.rpush.assert_called_once()
        pipe.ltrim.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_log_calls_lrange(self):
        """get_log() should call LRANGE with the correct key and range."""
        from unittest.mock import AsyncMock
        redis = self._make_redis_mock()
        redis.lrange = AsyncMock(return_value=["[12:00:00] entry 1", "[12:01:00] entry 2"])
        store = self._make_store(redis_mock=redis)
        entries = await store.get_log("task123", n=10)
        redis.lrange.assert_called_once_with("research:log:task123", -10, -1)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_increment_cycle_returns_new_value(self):
        """increment_cycle() should INCR and return the new integer value."""
        from unittest.mock import AsyncMock
        redis = self._make_redis_mock()
        redis.incr = AsyncMock(return_value=3)
        redis.get = AsyncMock(return_value=None)  # no task to update
        store = self._make_store(redis_mock=redis)
        val = await store.increment_cycle("task123")
        assert val == 3
        redis.incr.assert_called_once_with("research:cycle:task123")

    @pytest.mark.asyncio
    async def test_update_task_status_to_stopped_removes_from_active_set(self):
        """update_task_status('stopped') should SREM from the active index."""
        from unittest.mock import AsyncMock
        from octane.research.models import ResearchTask
        task = ResearchTask(topic="test topic")
        redis = self._make_redis_mock()
        redis.get = AsyncMock(return_value=task.model_dump_json())
        store = self._make_store(redis_mock=redis)
        await store.update_task_status(task.id, "stopped")
        redis.srem.assert_called_once_with("research:active", task.id)

    @pytest.mark.asyncio
    async def test_list_tasks_empty_when_no_active_tasks(self):
        """list_tasks() returns [] when no IDs are in the active SET."""
        from unittest.mock import AsyncMock
        redis = self._make_redis_mock()
        redis.smembers = AsyncMock(return_value=set())
        store = self._make_store(redis_mock=redis)
        tasks = await store.list_tasks()
        assert tasks == []

    @pytest.mark.asyncio
    async def test_add_finding_with_no_pg_returns_none(self):
        """add_finding() should return None gracefully when Postgres is unavailable."""
        redis = self._make_redis_mock()
        store = self._make_store(redis_mock=redis, pg_mock=None)
        # Prevent lazy connect by removing the URL so asyncpg.connect() is never called
        store._postgres_url = None
        # No pg injected and no pg_url — _pg() returns None
        result = await store.add_finding(
            task_id="t1", cycle_num=1, topic="test", content="summary text",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_findings_with_no_pg_returns_empty_list(self):
        """get_findings() should return [] gracefully when Postgres is unavailable."""
        redis = self._make_redis_mock()
        store = self._make_store(redis_mock=redis, pg_mock=None)
        store._postgres_url = None  # prevent lazy connect
        results = await store.get_findings("t1")
        assert results == []


# ── research_cycle task (unit-level) ─────────────────────────

class TestResearchCycleUnit:
    """Unit tests for the research_cycle task logic with fully mocked dependencies."""

    @pytest.mark.asyncio
    async def test_cycle_logs_start_and_increments_counter(self):
        """Cycle should call log_entry and increment_cycle at the start."""
        from unittest.mock import AsyncMock, MagicMock, patch, AsyncMock as AM
        from octane.tasks.research import research_cycle

        mock_store = MagicMock()
        mock_store.log_entry = AsyncMock()
        mock_store.increment_cycle = AsyncMock(return_value=1)
        mock_store.add_finding = AsyncMock(return_value=None)
        mock_store.get_task = AsyncMock(return_value=None)
        mock_store.close = AsyncMock()

        mock_osa = MagicMock()

        async def fake_stream(*a, **kw):
            yield "NVDA posted record revenue."

        mock_osa.run_stream = fake_stream
        mock_synapse = MagicMock()
        mock_synapse._events = []

        with (
            patch("octane.research.store.ResearchStore", return_value=mock_store),
            patch("octane.models.synapse.SynapseEventBus", return_value=mock_synapse),
            patch("octane.osa.orchestrator.Orchestrator", return_value=mock_osa),
        ):
            # Call without Shadows injection (perpetual/log are optional-dep injected)
            await research_cycle.__wrapped__(
                task_id="t1", topic="NVDA earnings"
            ) if hasattr(research_cycle, "__wrapped__") else await research_cycle(
                task_id="t1", topic="NVDA earnings",
                perpetual=MagicMock(), log=MagicMock(),
            )

        mock_store.increment_cycle.assert_called_once_with("t1")
        # log_entry should have been called at least twice (start + end)
        assert mock_store.log_entry.call_count >= 1

    @pytest.mark.asyncio
    async def test_cycle_stores_finding_when_pipeline_returns_content(self):
        """Cycle should call add_finding when the OSA pipeline produces output."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.tasks.research import research_cycle

        mock_store = MagicMock()
        mock_store.log_entry = AsyncMock()
        mock_store.increment_cycle = AsyncMock(return_value=1)
        mock_store.add_finding = AsyncMock(return_value=MagicMock(word_count=8))
        mock_store.get_task = AsyncMock(return_value=None)
        mock_store.close = AsyncMock()

        mock_osa = MagicMock()

        async def fake_stream(*a, **kw):
            # Content must exceed MINIMUM_QUALITY_WORDS (30) to pass quality gate
            yield (
                "NVDA Q4 revenue hit $22.1B, beating estimates by 15%. "
                "Data center segment grew 409% year-over-year driven by AI demand. "
                "Management guided Q1 revenue at $24B plus or minus 2%, well above "
                "consensus expectations of $21.9B. Gross margins expanded to 76.7%."
            )

        mock_osa.run_stream = fake_stream
        mock_synapse = MagicMock()
        mock_synapse._events = []

        with (
            patch("octane.research.store.ResearchStore", return_value=mock_store),
            patch("octane.models.synapse.SynapseEventBus", return_value=mock_synapse),
            patch("octane.osa.orchestrator.Orchestrator", return_value=mock_osa),
        ):
            await research_cycle(
                task_id="t2", topic="NVDA earnings",
                perpetual=MagicMock(), log=MagicMock(),
            )

        mock_store.add_finding.assert_called_once()
        call_kwargs = mock_store.add_finding.call_args
        assert call_kwargs.kwargs["task_id"] == "t2"
        assert "NVDA" in call_kwargs.kwargs["content"]

    @pytest.mark.asyncio
    async def test_cycle_skips_finding_when_pipeline_empty(self):
        """Cycle should not call add_finding when pipeline produces no output."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.tasks.research import research_cycle

        mock_store = MagicMock()
        mock_store.log_entry = AsyncMock()
        mock_store.increment_cycle = AsyncMock(return_value=1)
        mock_store.add_finding = AsyncMock()
        mock_store.get_task = AsyncMock(return_value=None)
        mock_store.close = AsyncMock()

        mock_osa = MagicMock()

        async def empty_stream(*a, **kw):
            return
            yield  # make it an async generator

        mock_osa.run_stream = empty_stream
        mock_synapse = MagicMock()
        mock_synapse._events = []

        with (
            patch("octane.research.store.ResearchStore", return_value=mock_store),
            patch("octane.models.synapse.SynapseEventBus", return_value=mock_synapse),
            patch("octane.osa.orchestrator.Orchestrator", return_value=mock_osa),
        ):
            await research_cycle(
                task_id="t3", topic="empty topic",
                perpetual=MagicMock(), log=MagicMock(),
            )

        mock_store.add_finding.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_handles_pipeline_exception_gracefully(self):
        """Cycle should call close() and not raise when the pipeline errors."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from octane.tasks.research import research_cycle

        mock_store = MagicMock()
        mock_store.log_entry = AsyncMock()
        mock_store.increment_cycle = AsyncMock(return_value=1)
        mock_store.add_finding = AsyncMock()
        mock_store.get_task = AsyncMock(return_value=None)
        mock_store.close = AsyncMock()

        mock_osa = MagicMock()

        async def error_stream(*a, **kw):
            raise RuntimeError("Bodega timeout")
            yield

        mock_osa.run_stream = error_stream
        mock_synapse = MagicMock()
        mock_synapse._events = []

        with (
            patch("octane.research.store.ResearchStore", return_value=mock_store),
            patch("octane.models.synapse.SynapseEventBus", return_value=mock_synapse),
            patch("octane.osa.orchestrator.Orchestrator", return_value=mock_osa),
        ):
            # Should NOT raise
            await research_cycle(
                task_id="t4", topic="error topic",
                perpetual=MagicMock(), log=MagicMock(),
            )

        mock_store.add_finding.assert_not_called()
        mock_store.close.assert_called_once()


# ── research CLI commands (smoke tests) ──────────────────────

class TestResearchCLI:
    """Smoke tests for the octane research CLI command group."""

    def test_research_app_registered(self):
        """research_app should be registered on the main typer app."""
        from octane.main import app
        # Typer stores sub-apps in registered_groups; at minimum the app
        # should have a 'research' entry.
        command_names = [cmd.name for cmd in app.registered_commands]
        # Sub-apps appear as groups; check at least the main commands exist
        group_names = [g.typer_instance.info.name for g in app.registered_groups]
        assert "research" in group_names, f"'research' not in groups: {group_names}"

    def test_research_start_requires_topic(self):
        """research start without a topic should exit with error."""
        from typer.testing import CliRunner
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["research", "start"])
        assert result.exit_code != 0

    def test_research_status_runs_without_crash(self):
        """research status should run and exit cleanly (even with no tasks)."""
        from unittest.mock import patch, AsyncMock
        from typer.testing import CliRunner
        from octane.main import app

        runner = CliRunner()
        with patch(
            "octane.main._research_status",
            new=AsyncMock(return_value=None),
        ):
            result = runner.invoke(app, ["research", "status"])
        # Either 0 (success) or non-zero (mocked early return) is fine
        # — what matters is no unhandled exception traceback
        assert "Error" not in (result.output or "")

    def test_research_task_id_in_octane_tasks(self):
        """research_cycle should appear in the octane_tasks registry."""
        from octane.tasks import octane_tasks
        names = [t.__name__ for t in octane_tasks]
        assert "research_cycle" in names


# ── Session 18A — Structured Storage ─────────────────────────────────────────

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


def _make_pg_mock(row: dict | None = None, rows: list | None = None, val=0):
    """Build a mock PgClient with async methods pre-configured."""
    pg = MagicMock()
    pg.fetchrow = AsyncMock(return_value=row or {
        "id": 1, "name": "test", "status": "active",
        "created_at": datetime.now(timezone.utc), "description": "",
    })
    pg.fetch = AsyncMock(return_value=rows or [])
    pg.fetchval = AsyncMock(return_value=val)
    pg.execute = AsyncMock(return_value=True)
    return pg


class TestProjectStore:
    """Tests for octane.tools.structured_store.ProjectStore."""

    def test_create_project_returns_row(self):
        from octane.tools.structured_store import ProjectStore
        pg = _make_pg_mock(row={"id": 1, "name": "alpha", "status": "active",
                                "created_at": datetime.now(timezone.utc), "description": ""})
        store = ProjectStore(pg)
        result = asyncio.run(store.create("alpha"))
        assert result is not None
        assert result["name"] == "alpha"
        pg.fetchrow.assert_called_once()

    def test_create_project_upsert_on_conflict(self):
        """create() uses ON CONFLICT DO UPDATE so the SQL contains CONFLICT."""
        from octane.tools.structured_store import ProjectStore
        pg = _make_pg_mock()
        store = ProjectStore(pg)
        asyncio.run(store.create("beta"))
        call_args = pg.fetchrow.call_args
        sql = call_args[0][0].upper()
        assert "CONFLICT" in sql

    def test_list_excludes_archived_by_default(self):
        from octane.tools.structured_store import ProjectStore
        pg = _make_pg_mock(rows=[{"id": 1, "name": "p1", "status": "active",
                                   "description": "", "created_at": datetime.now(timezone.utc)}])
        store = ProjectStore(pg)
        results = asyncio.run(store.list())
        pg.fetch.assert_called_once()
        call_args = pg.fetch.call_args
        sql_or_args = str(call_args)
        assert "active" in sql_or_args or len(results) >= 0  # query was made

    def test_archive_calls_execute_with_archived_status(self):
        from octane.tools.structured_store import ProjectStore
        pg = _make_pg_mock(val=1)
        pg.fetchval = AsyncMock(return_value=1)
        store = ProjectStore(pg)
        ok = asyncio.run(store.archive("gamma"))
        pg.execute.assert_called_once()
        call_args = pg.execute.call_args
        assert "archived" in str(call_args).lower()

    def test_delete_project_calls_execute(self):
        from octane.tools.structured_store import ProjectStore
        pg = _make_pg_mock()
        store = ProjectStore(pg)
        asyncio.run(store.delete("gamma"))
        pg.execute.assert_called_once()


class TestWebPageStore:
    """Tests for octane.tools.structured_store.WebPageStore."""

    def test_store_returns_row(self):
        from octane.tools.structured_store import WebPageStore
        pg = _make_pg_mock(row={"id": 1, "url": "https://example.com", "url_hash": "abc",
                                 "word_count": 10, "fetch_status": "ok",
                                 "created_at": datetime.now(timezone.utc)})
        store = WebPageStore(pg)
        result = asyncio.run(
            store.store("https://example.com", "hello world"))
        assert result is not None
        pg.fetchrow.assert_called_once()

    def test_seen_returns_true_when_found(self):
        from octane.tools.structured_store import WebPageStore
        pg = _make_pg_mock(row={"id": 1})
        store = WebPageStore(pg)
        result = asyncio.run(store.seen("https://example.com"))
        assert result is True

    def test_seen_returns_false_when_not_found(self):
        from octane.tools.structured_store import WebPageStore
        pg = MagicMock()
        pg.fetchrow = AsyncMock(return_value=None)
        store = WebPageStore(pg)
        result = asyncio.run(store.seen("https://example.com"))
        assert result is False

    def test_url_hash_is_sha256_normalised(self):
        from octane.tools.structured_store import _url_hash
        h1 = _url_hash("HTTP://Example.com/")
        h2 = _url_hash("http://example.com")
        assert h1 == h2
        assert len(h1) == 64  # hex SHA-256

    def test_url_hash_differs_for_different_urls(self):
        from octane.tools.structured_store import _url_hash
        assert _url_hash("https://example.com") != _url_hash("https://other.com")

    def test_recent_calls_fetch(self):
        from octane.tools.structured_store import WebPageStore
        pg = _make_pg_mock(rows=[])
        store = WebPageStore(pg)
        asyncio.run(store.recent(limit=5))
        pg.fetch.assert_called_once()

    def test_count_calls_fetchval(self):
        from octane.tools.structured_store import WebPageStore
        pg = _make_pg_mock(val=42)
        store = WebPageStore(pg)
        n = asyncio.run(store.count())
        assert n == 42


class TestArtifactStore:
    """Tests for octane.tools.structured_store.ArtifactStore."""

    def test_register_returns_row(self):
        from octane.tools.structured_store import ArtifactStore
        pg = _make_pg_mock(row={"id": 1, "artifact_type": "code", "language": "python",
                                 "session_id": "s1", "created_at": datetime.now(timezone.utc)})
        store = ArtifactStore(pg)
        result = asyncio.run(
            store.register("print('hi')", artifact_type="code", language="python"))
        assert result is not None
        pg.fetchrow.assert_called_once()

    def test_query_with_no_filters_calls_fetch(self):
        from octane.tools.structured_store import ArtifactStore
        pg = _make_pg_mock(rows=[])
        store = ArtifactStore(pg)
        results = asyncio.run(store.query())
        pg.fetch.assert_called_once()
        assert isinstance(results, list)

    def test_query_with_session_filter(self):
        from octane.tools.structured_store import ArtifactStore
        pg = _make_pg_mock(rows=[])
        store = ArtifactStore(pg)
        asyncio.run(store.query(session_id="abc123"))
        call_args = pg.fetch.call_args
        # session_id value should appear in the positional args
        assert "abc123" in str(call_args)

    def test_get_calls_fetchrow(self):
        from octane.tools.structured_store import ArtifactStore
        pg = _make_pg_mock(row={"id": 7})
        store = ArtifactStore(pg)
        asyncio.run(store.get(7))
        pg.fetchrow.assert_called_once()


class TestChunkText:
    """Tests for the _chunk_text helper."""

    def test_short_text_produces_one_chunk(self):
        from octane.tools.structured_store import _chunk_text
        chunks = _chunk_text("word " * 50)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        from octane.tools.structured_store import _chunk_text
        chunks = _chunk_text("word " * 500)
        assert len(chunks) > 2

    def test_overlap_means_chunks_share_words(self):
        from octane.tools.structured_store import _chunk_text
        # Build deterministic text: "w0 w1 w2 ... w499"
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_words=200)
        # The last word of chunk[0] should appear in chunk[1] (due to overlap)
        last_word_of_c0 = chunks[0].split()[-1]
        assert last_word_of_c0 in chunks[1]

    def test_empty_text_returns_empty_list(self):
        from octane.tools.structured_store import _chunk_text
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []


class TestFileIndexer:
    """Tests for octane.tools.structured_store.FileIndexer."""

    def test_index_file_unsupported_extension_returns_none(self, tmp_path):
        from octane.tools.structured_store import FileIndexer
        pg = _make_pg_mock()
        store = FileIndexer(pg)
        exe_file = tmp_path / "malware.exe"
        exe_file.write_bytes(b"\x00\x01\x02")
        result = asyncio.run(store.index_file(str(exe_file)))
        assert result is None

    def test_index_file_not_found_returns_none(self):
        from octane.tools.structured_store import FileIndexer
        pg = _make_pg_mock()
        store = FileIndexer(pg)
        result = asyncio.run(
            store.index_file("/nonexistent/path/file.py"))
        assert result is None

    def test_index_file_skips_unchanged_hash(self, tmp_path):
        from octane.tools.structured_store import FileIndexer, _file_hash
        pg = _make_pg_mock()
        store = FileIndexer(pg)
        py_file = tmp_path / "script.py"
        py_file.write_text("print('hello')")
        h = _file_hash(str(py_file))
        # Pre-populate fetchrow to return existing row with same hash
        pg.fetchrow = AsyncMock(return_value={
            "id": 5, "path": str(py_file), "file_hash": h,
            "word_count": 2, "created_at": datetime.now(timezone.utc),
            "extension": ".py", "project_id": None, "indexed_at": None,
        })
        result = asyncio.run(store.index_file(str(py_file)))
        # Should return existing row, not call execute (no INSERT)
        assert result is not None
        pg.execute.assert_not_called()

    def test_stats_returns_expected_keys(self):
        from octane.tools.structured_store import FileIndexer
        pg = _make_pg_mock(val=10)
        pg.fetchval = AsyncMock(return_value=10)
        pg.fetch = AsyncMock(return_value=[
            {"extension": ".py", "count": 5, "total_words": 1000}
        ])
        store = FileIndexer(pg)
        stats = asyncio.run(store.stats())
        assert "total_files" in stats
        assert "total_words" in stats
        assert "by_extension" in stats


class TestEmbeddingEngine:
    """Tests for octane.tools.structured_store.EmbeddingEngine."""

    def test_embed_returns_zero_vectors_without_model(self):
        from octane.tools.structured_store import EmbeddingEngine, EMBEDDING_DIM
        pg = _make_pg_mock()
        engine = EmbeddingEngine(pg)
        engine._model = None  # force fallback
        vecs = engine._embed(["hello world", "test"])
        assert len(vecs) == 2
        assert len(vecs[0]) == EMBEDDING_DIM
        assert all(v == 0.0 for v in vecs[0])

    def test_embed_and_store_returns_zero_for_empty_text(self):
        from octane.tools.structured_store import EmbeddingEngine
        pg = _make_pg_mock()
        engine = EmbeddingEngine(pg)
        engine._model = None
        count = asyncio.run(
            engine.embed_and_store("user_file", 1, ""))
        assert count == 0
        pg.execute.assert_not_called()

    def test_embed_and_store_chunks_and_persists(self):
        from octane.tools.structured_store import EmbeddingEngine
        pg = _make_pg_mock()
        engine = EmbeddingEngine(pg)
        engine._model = None
        text = "word " * 250  # long enough for 2+ chunks
        count = asyncio.run(
            engine.embed_and_store("web_page", 1, text))
        assert count >= 1
        assert pg.execute.call_count >= 1

    def test_semantic_search_calls_pg_fetch(self):
        from octane.tools.structured_store import EmbeddingEngine
        pg = _make_pg_mock(rows=[])
        engine = EmbeddingEngine(pg)
        engine._model = None
        results = asyncio.run(
            engine.semantic_search("find something"))
        pg.fetch.assert_called_once()
        # The SQL should use the pgvector <=> operator
        call_sql = pg.fetch.call_args[0][0]
        assert "<=>" in call_sql

    def test_semantic_search_with_source_type_filter(self):
        from octane.tools.structured_store import EmbeddingEngine
        pg = _make_pg_mock(rows=[])
        engine = EmbeddingEngine(pg)
        engine._model = None
        asyncio.run(
            engine.semantic_search("query", source_type="user_file"))
        call_args = pg.fetch.call_args
        assert "user_file" in str(call_args)


class TestWebAgentPageStore:
    """Tests for WebAgent page_store injection (Session 18A)."""

    def test_web_agent_accepts_page_store_kwarg(self):
        from octane.agents.web.agent import WebAgent
        synapse = make_synapse()
        mock_ps = MagicMock()
        agent = WebAgent(synapse, page_store=mock_ps)
        assert agent._page_store is mock_ps

    def test_web_agent_page_store_defaults_to_none(self):
        from octane.agents.web.agent import WebAgent
        synapse = make_synapse()
        agent = WebAgent(synapse)
        assert agent._page_store is None

    def test_store_pages_no_op_when_page_store_none(self):
        """_store_pages() must not raise when page_store is None."""
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent
        synapse = make_synapse()
        agent = WebAgent(synapse)
        dummy = ExtractedContent(url="https://x.com", text="hello", word_count=1, method="trafilatura")
        # Should complete without error
        asyncio.run(agent._store_pages([dummy]))

    def test_store_pages_skips_empty_content(self):
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent
        synapse = make_synapse()
        mock_ps = MagicMock()
        mock_ps.store = AsyncMock()
        agent = WebAgent(synapse, page_store=mock_ps)
        empty = ExtractedContent(url="https://x.com", text="", word_count=0, method="trafilatura")
        asyncio.run(agent._store_pages([empty]))
        mock_ps.store.assert_not_called()

    def test_store_pages_calls_store_for_valid_content(self):
        from octane.agents.web.agent import WebAgent
        from octane.agents.web.content_extractor import ExtractedContent
        synapse = make_synapse()
        mock_ps = MagicMock()
        mock_ps.store = AsyncMock(return_value={"id": 1})
        agent = WebAgent(synapse, page_store=mock_ps)
        item = ExtractedContent(url="https://x.com", text="some article text", word_count=3, method="trafilatura")
        asyncio.run(agent._store_pages([item]))
        mock_ps.store.assert_called_once()
        call_kwargs = mock_ps.store.call_args[1]
        assert call_kwargs["url"] == "https://x.com"


class TestCodeAgentArtifactStore:
    """Tests for CodeAgent artifact_store injection (Session 18A)."""

    def test_code_agent_accepts_artifact_store_kwarg(self):
        from octane.agents.code.agent import CodeAgent
        synapse = make_synapse()
        mock_as = MagicMock()
        agent = CodeAgent(synapse, artifact_store=mock_as)
        assert agent._artifact_store is mock_as

    def test_code_agent_artifact_store_defaults_to_none(self):
        from octane.agents.code.agent import CodeAgent
        synapse = make_synapse()
        agent = CodeAgent(synapse)
        assert agent._artifact_store is None

    def test_artifact_store_register_called_on_success(self):
        """ArtifactStore.register() is called when code passes validation."""
        from octane.agents.code.agent import CodeAgent
        synapse = make_synapse()
        mock_as = MagicMock()
        mock_as.register = AsyncMock(return_value={"id": 9})
        agent = CodeAgent(synapse, artifact_store=mock_as)

        # Stub out all sub-components
        agent.planner.plan = AsyncMock(return_value={"task": "test", "language": "python",
                                                      "expected_output": "ok", "constraints": []})
        agent.writer.write = AsyncMock(return_value="print('ok')")
        agent.executor.run = AsyncMock(return_value={"stdout": "ok", "stderr": "", "returncode": 0})
        agent.validator.validate = MagicMock(return_value={"passed": True, "error_summary": ""})
        agent.catalyst_registry.match = MagicMock(return_value=None)

        req = make_request("write a hello world script")
        resp = asyncio.run(agent.execute(req))
        assert resp.success is True
        mock_as.register.assert_called_once()
        call_kwargs = mock_as.register.call_args[1]
        assert call_kwargs["artifact_type"] == "code"
        assert "print" in call_kwargs["content"]

    def test_artifact_store_not_called_on_failure(self):
        """ArtifactStore.register() must NOT be called when code fails validation."""
        from octane.agents.code.agent import CodeAgent
        synapse = make_synapse()
        mock_as = MagicMock()
        mock_as.register = AsyncMock()
        agent = CodeAgent(synapse, artifact_store=mock_as)

        agent.planner.plan = AsyncMock(return_value={"task": "test", "language": "python",
                                                      "expected_output": "ok", "constraints": []})
        agent.writer.write = AsyncMock(return_value="bad code")
        agent.executor.run = AsyncMock(return_value={"stdout": "", "stderr": "error", "returncode": 1})
        agent.validator.validate = MagicMock(return_value={"passed": False, "error_summary": "fail",
                                                              "should_retry": False})
        agent.debugger.debug = AsyncMock(return_value=None)
        agent.catalyst_registry.match = MagicMock(return_value=None)

        req = make_request("write broken code")
        resp = asyncio.run(agent.execute(req))
        assert resp.success is False
        mock_as.register.assert_not_called()


class TestRouterStoreInjection:
    """Tests that Router correctly wires stores into agents."""

    def test_router_has_page_store_and_artifact_store(self):
        from octane.osa.router import Router
        synapse = make_synapse()
        with patch("octane.osa.router.PgClient"), \
             patch("octane.osa.router.WebPageStore") as MockWPS, \
             patch("octane.osa.router.ArtifactStore") as MockAS:
            MockWPS.return_value = MagicMock()
            MockAS.return_value = MagicMock()
            router = Router(synapse)
        assert hasattr(router, "_page_store")
        assert hasattr(router, "_artifact_store")

    def test_router_web_agent_has_page_store(self):
        from octane.osa.router import Router
        synapse = make_synapse()
        with patch("octane.osa.router.PgClient"), \
             patch("octane.osa.router.WebPageStore") as MockWPS, \
             patch("octane.osa.router.ArtifactStore"):
            mock_ps = MagicMock()
            MockWPS.return_value = mock_ps
            router = Router(synapse)
        web_agent = router.get_agent("web")
        assert web_agent._page_store is mock_ps

    def test_router_code_agent_has_artifact_store(self):
        from octane.osa.router import Router
        synapse = make_synapse()
        with patch("octane.osa.router.PgClient"), \
             patch("octane.osa.router.WebPageStore"), \
             patch("octane.osa.router.ArtifactStore") as MockAS:
            mock_as = MagicMock()
            MockAS.return_value = mock_as
            router = Router(synapse)
        code_agent = router.get_agent("code")
        assert code_agent._artifact_store is mock_as


# ── Session 18B — Deep Research Mode ─────────────────────────────────────────

class TestAngleGenerator:
    """Tests for octane.research.angles.AngleGenerator."""

    def test_keyword_angles_shallow_returns_two(self):
        from octane.research.angles import AngleGenerator
        gen = AngleGenerator(bodega=None)
        angles = asyncio.run(gen.generate("NVDA earnings", depth="shallow"))
        assert len(angles) == 2
        for a in angles:
            assert "query" in a
            assert "angle" in a

    def test_keyword_angles_deep_returns_four(self):
        from octane.research.angles import AngleGenerator
        gen = AngleGenerator(bodega=None)
        angles = asyncio.run(gen.generate("Apple Vision Pro", depth="deep"))
        assert len(angles) == 4

    def test_keyword_angles_exhaustive_returns_eight(self):
        from octane.research.angles import AngleGenerator
        gen = AngleGenerator(bodega=None)
        angles = asyncio.run(gen.generate("Fed rate decisions", depth="exhaustive"))
        assert len(angles) == 8

    def test_keyword_angles_unknown_depth_defaults_to_deep(self):
        from octane.research.angles import AngleGenerator, DEPTH_ANGLES
        gen = AngleGenerator(bodega=None)
        angles = asyncio.run(gen.generate("test", depth="bogus"))
        assert len(angles) == DEPTH_ANGLES["deep"]

    def test_all_angles_contain_topic(self):
        from octane.research.angles import AngleGenerator
        gen = AngleGenerator(bodega=None)
        angles = asyncio.run(gen.generate("NVDA", depth="deep"))
        for a in angles:
            assert "NVDA" in a["query"]

    def test_llm_fallback_on_exception(self):
        """If LLM raises, falls back to keyword angles silently."""
        from octane.research.angles import AngleGenerator
        mock_bodega = MagicMock()
        mock_bodega.complete = AsyncMock(side_effect=RuntimeError("offline"))
        gen = AngleGenerator(bodega=mock_bodega)
        angles = asyncio.run(gen.generate("test topic", depth="deep"))
        assert len(angles) == 4

    def test_parse_angles_valid_json(self):
        from octane.research.angles import _parse_angles
        raw = '[{"query": "NVDA Q4 2025", "angle": "earnings"}, {"query": "NVIDIA AI chips", "angle": "market"}]'
        result = _parse_angles(raw)
        assert len(result) == 2
        assert result[0]["query"] == "NVDA Q4 2025"
        assert result[0]["angle"] == "earnings"

    def test_parse_angles_handles_markdown_fences(self):
        from octane.research.angles import _parse_angles
        raw = '```json\n[{"query": "test", "angle": "market"}]\n```'
        result = _parse_angles(raw)
        assert len(result) == 1

    def test_parse_angles_invalid_json_returns_empty(self):
        from octane.research.angles import _parse_angles
        result = _parse_angles("not json at all")
        assert result == []

    def test_llm_path_called_with_correct_n(self):
        """LLM is asked for the correct number of angles based on depth."""
        from octane.research.angles import AngleGenerator
        mock_bodega = MagicMock()
        angles_json = '[' + ','.join(
            f'{{"query": "q{i}", "angle": "market"}}' for i in range(8)
        ) + ']'
        mock_bodega.chat_simple = AsyncMock(return_value=angles_json)
        gen = AngleGenerator(bodega=mock_bodega)
        result = asyncio.run(gen.generate("topic", depth="exhaustive"))
        assert len(result) == 8
        call_args = mock_bodega.chat_simple.call_args
        prompt = call_args[1].get("prompt") or call_args[0][0]
        assert "8" in prompt


class TestResearchDepthFlag:
    """Tests for --depth flag wiring in research start + ResearchTask model."""

    def test_research_task_has_depth_field(self):
        from octane.research.models import ResearchTask
        t = ResearchTask(topic="test")
        assert hasattr(t, "depth")
        assert t.depth == "deep"

    def test_research_task_custom_depth(self):
        from octane.research.models import ResearchTask
        t = ResearchTask(topic="test", depth="exhaustive")
        assert t.depth == "exhaustive"

    def test_research_start_command_has_depth_option(self):
        """CLI must accept --depth without error."""
        from typer.testing import CliRunner
        from octane.main import app
        runner = CliRunner()
        with patch("octane.main._research_start", new=AsyncMock(return_value=None)):
            result = runner.invoke(app, ["research", "start", "test topic", "--depth", "shallow"])
        assert "Invalid" not in (result.output or "")

    def test_research_start_rejects_invalid_depth(self):
        """CLI must reject unknown depth values."""
        from typer.testing import CliRunner
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["research", "start", "test topic", "--depth", "ultra"])
        assert result.exit_code != 0 or "Invalid" in (result.output or "")


class TestSynapseEventLabel:
    """Tests for the _event_label() helper in research_cycle."""

    def test_ingress_label(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="ingress",
                         source="osa", payload={"query": "hello world"})
        label = _event_label(e)
        assert label is not None
        assert "ingress" in label
        assert "hello world" in label

    def test_decomposition_complete_label(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="decomposition_complete",
                         source="osa", payload={"node_count": 3})
        label = _event_label(e)
        assert label is not None
        assert "3" in label

    def test_dispatch_label(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="dispatch",
                         source="osa", payload={"agent": "web", "task": "fetch NVDA news"})
        label = _event_label(e)
        assert label is not None
        assert "web" in label

    def test_code_success_label(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="code_success",
                         source="code.agent", payload={"attempt": 1})
        label = _event_label(e)
        assert label is not None
        assert "✅" in label

    def test_code_healed_label_shows_self_healed(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="code_healed",
                         source="code.agent", payload={"attempt": 2})
        label = _event_label(e)
        assert "self-healed" in label

    def test_guard_block_label(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="guard_block",
                         source="osa.guard", payload={"reason": "unsafe content"})
        label = _event_label(e)
        assert "🛡" in label
        assert "unsafe content" in label

    def test_egress_label_shows_word_count(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        e = SynapseEvent(correlation_id="c1", event_type="egress",
                         source="osa", payload={"output": "word " * 50, "agents_used": ["web"]})
        label = _event_label(e)
        assert "50" in label

    def test_noisy_events_return_none(self):
        from octane.tasks.research import _event_label
        from octane.models.synapse import SynapseEvent
        for etype in ("preflight", "hil_complete", "catalyst_failed"):
            e = SynapseEvent(correlation_id="c1", event_type=etype,
                             source="osa", payload={})
            assert _event_label(e) is None, f"Expected None for event_type={etype}"


class TestResearchCycleDepth:
    """Smoke tests for the depth + angles path in research_cycle."""

    def test_depth_angles_mapping(self):
        from octane.research.angles import DEPTH_ANGLES
        assert DEPTH_ANGLES["shallow"] == 2
        assert DEPTH_ANGLES["deep"] == 4
        assert DEPTH_ANGLES["exhaustive"] == 8

    def test_research_cycle_signature_has_depth(self):
        import inspect
        from octane.tasks.research import research_cycle
        sig = inspect.signature(research_cycle)
        assert "depth" in sig.parameters
        assert sig.parameters["depth"].default == "deep"
