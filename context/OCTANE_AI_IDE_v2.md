# OCTANE_AI_IDE.md
# Project Octane — AI IDE Agent Guidelines (v2)
# Place this file in your project root. Reference it in your Copilot context window.
# Last updated: 2026-02-16

---

## 🧠 WHAT IS OCTANE?

Octane is a **local-first agentic operating system** for Apple Silicon. It is NOT a chatbot. It is a hierarchical agent system modeled after biological nervous systems:

- **OSA (Orchestrator & Synapse Agent)** = Brain + Nervous System. Every query flows through it. Every state transition is logged.
- **Specialized Agents** (Web, Code, Memory, SysStat, P&L) = Organs. Each has internal sub-agents.
- **Shadows** = Neural bus. Redis-based task orchestration carrying signals between agents.
- **Synapse** = Observability layer. Every agent transition produces structured events with full traceability.

**Core principles:**
1. Everything runs locally on Apple Silicon via MLX
2. Agents are hierarchical — top-level agents coordinate sub-agents
3. OSA is involved in EVERY state transition (no direct agent-to-agent communication)
4. Deterministic where possible, LLM-powered where necessary
5. Every action is traceable via Synapse events

---

## 🏗️ ARCHITECTURE

```
User Query (CLI)
      ↓
┌─────────────────────────────────────────────────────────────┐
│                         O S A                                │
│            (Orchestrator & Synapse Agent)                     │
│                                                              │
│  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────┐ │
│  │ Decomposer │ │  Router  │ │ Evaluator │ │   Guard     │ │
│  │(big model) │ │(determin)│ │(big model)│ │  (hybrid)   │ │
│  └─────┬──────┘ └────┬─────┘ └─────┬─────┘ └──────┬──────┘ │
│        │             │             │               │         │
│  ┌─────┴─────────────┴─────────────┴───────────────┴──────┐ │
│  │              Synapse EventBus                           │ │
│  │   (structured ingress/egress logs, correlation IDs,     │ │
│  │    reasoning traces, token counts, timing)              │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        │                                     │
│  ┌─────────────────────┴───────────────────────────────────┐ │
│  │           Shadows Task Orchestration                     │ │
│  │   (Redis Streams, at-least-once delivery, scheduling,   │ │
│  │    retries, perpetual tasks, idempotent keys)           │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│  ┌──────────────┐      │                                     │
│  │Policy Engine │      │  (deterministic rules,              │
│  │              │      │   max retries, HITL triggers)       │
│  └──────────────┘      │                                     │
└────────────────────────┼─────────────────────────────────────┘
                         │
        ┌────────┬───────┼────────┬─────────────┐
        ↓        ↓       ↓        ↓             ↓
   ┌────────┐┌───────┐┌───────┐┌────────┐┌──────────┐
   │  Web   ││ Code  ││Memory ││SysStat ││  P & L   │
   │ Agent  ││ Agent ││Agent  ││ Agent  ││  Agent   │
   │        ││       ││       ││        ││          │
   │ Query  ││Planner││Router ││Monitor ││Pref Mgr  │
   │ Strat. ││Writer ││(Hot/  ││Model   ││Feedback  │
   │ Fetch. ││Exec.  ││Warm/  ││Mgr     ││Learner   │
   │ Browsr ││Debug. ││Cold)  ││Scaler  ││Profile   │
   │ Synth. ││Valid. ││Writer ││        ││          │
   │        ││       ││Janitr ││        ││          │
   └────────┘└───────┘└───────┘└────────┘└──────────┘
```

### Flow Rule
```
User → OSA.Guard (parallel safety check)
     → OSA.Decomposer (query → task DAG)
     → OSA.Router (task → agent mapping)
     → Shadows dispatch (parallel agent execution)
     → Agents execute (each with internal sub-agents)
     → Results return to OSA via Synapse events
     → OSA.Evaluator (quality gate)
     → P&L consulted for personalization
     → Output to user
     → P&L records feedback
     → Memory.Writer decides what to persist
```

**CRITICAL: Agents NEVER talk directly to each other. Everything flows through OSA via Synapse.**

---

## 🔧 TECH STACK — DO NOT DEVIATE

| Layer | Technology | Version/Notes |
|-------|-----------|---------------|
| Language | Python | 3.12+ (required by Shadows). Type hints everywhere. `X | None` syntax. |
| Framework | FastAPI | For HTTP API endpoints (future). CLI-first for now. |
| Task Queue | Shadows (shadow-task) | Redis Streams-based. At-least-once delivery. The neural bus. |
| CLI | Typer + Rich | CLI commands. Rich for formatted output and traces. |
| HTTP Client | httpx | Async HTTP client. **NOT requests** (blocking). |
| Schemas | Pydantic v2 | All data models. Use `model_validate()`. |
| Database | PostgreSQL + pgVector | Memory warm/cold tiers. Use asyncpg. |
| Cache | Redis | Memory hot tier + Shadows backend. |
| LLM Inference | Bodega Inference Engine | localhost:44468, OpenAI-compatible API. |
| External Data | Bodega Intelligence | Search :1111, Finance :8030, News :8032, Entertainment :8031 |
| Web Scraping | Trafilatura + Playwright | Trafilatura for clean extraction, Playwright for JS-heavy sites. |
| Testing | pytest + pytest-asyncio | Every agent and sub-agent gets tests. |
| Logging | structlog | Structured JSON logging. |
| Config | pydantic-settings | .env file based. |
| System Metrics | psutil | RAM, CPU monitoring for SysStat Agent. |

---

## 📐 CODING STANDARDS — FOLLOW STRICTLY

### Python Style
```python
# ✅ DO: Type hints, async, Pydantic models, docstrings
async def search(self, query: str, max_results: int = 5) -> SearchResult:
    """Execute web search via Beru Intelligence API."""
    ...

# ❌ DON'T: No types, sync functions for I/O, raw dicts
def search(query, max_results=5):
    return {"results": [...]}
```

### Import Organization
```python
# Standard library
import asyncio
import time
from datetime import datetime, timezone

# Third party
import httpx
import structlog
from pydantic import BaseModel, Field

# Local - tools layer
from octane.tools.bodega_inference import BodegaInferenceClient

# Local - models layer
from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent
```

---

## 🧬 CORE CONTRACTS

### BaseAgent — EVERY agent extends this

```python
from abc import ABC, abstractmethod
from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent, SynapseEventBus
import time
import structlog

logger = structlog.get_logger()

class BaseAgent(ABC):
    """Base class for all Octane agents.
    
    Every agent (top-level and sub-agent) extends this.
    The run() wrapper handles timing, error handling, and Synapse event emission.
    Subclasses implement execute() only.
    """
    
    name: str
    description: str

    def __init__(self, synapse: SynapseEventBus):
        self.synapse = synapse

    @abstractmethod
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Core logic. Subclasses implement this. Do NOT call directly — use run()."""
        ...

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Public entry point. Wraps execute() with timing, logging, Synapse events.
        Do NOT override this method."""
        start = time.perf_counter()
        correlation_id = request.metadata.get("correlation_id", "unknown")
        
        # Emit ingress event
        await self.synapse.emit(SynapseEvent(
            correlation_id=correlation_id,
            event_type="agent_ingress",
            source=request.metadata.get("source", "osa"),
            target=self.name,
            payload_summary=request.query[:200],
        ))
        
        try:
            result = await self.execute(request)
            result.latency_ms = (time.perf_counter() - start) * 1000
            result.agent_name = self.name
            result.success = True
            
            # Emit egress event
            await self.synapse.emit(SynapseEvent(
                correlation_id=correlation_id,
                event_type="agent_egress",
                source=self.name,
                target=request.metadata.get("source", "osa"),
                latency_ms=result.latency_ms,
                success=True,
            ))
            
            return result
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error("agent_execution_failed", agent=self.name, error=str(e))
            
            await self.synapse.emit(SynapseEvent(
                correlation_id=correlation_id,
                event_type="agent_error",
                source=self.name,
                target="osa",
                error=str(e),
                latency_ms=latency,
                success=False,
            ))
            
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    async def health(self) -> bool:
        """Check if this agent's dependencies are reachable."""
        return True
```

### Schemas

```python
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
import uuid

class AgentRequest(BaseModel):
    """Standard input for every agent."""
    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # metadata always includes: correlation_id, source, session_id

class AgentResponse(BaseModel):
    """Standard output from every agent."""
    agent_name: str = ""
    result: Any = None
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    # metadata can include: model_used, tokens_in, tokens_out, sub_agent_traces
```

### SynapseEvent

```python
class SynapseEvent(BaseModel):
    """A single event in the Synapse nervous system.
    Every agent state transition produces one."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str              # Traces full query lifecycle
    event_type: str                  # ingress | egress | dispatch | error | decomposition | routing
    source: str                      # Agent/component that produced this event
    target: str                      # Agent/component this event is directed at
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Content
    payload_summary: str = ""        # Truncated description of payload
    reasoning: str = ""              # Why this decision was made (for Decomposer/Router)
    
    # Metrics
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model_used: str = ""
    
    # Status
    success: bool = True
    error: str = ""

class SynapseTrace(BaseModel):
    """Complete trace of a query's lifecycle."""
    correlation_id: str
    events: list[SynapseEvent] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    agents_used: list[str] = Field(default_factory=list)
    
class SynapseEventBus:
    """In-memory event bus. Phase 4: replace with Redis Stream."""
    
    def __init__(self):
        self._events: list[SynapseEvent] = []
    
    async def emit(self, event: SynapseEvent) -> None:
        self._events.append(event)
        structlog.get_logger().info(
            "synapse_event",
            event_type=event.event_type,
            source=event.source,
            target=event.target,
            correlation_id=event.correlation_id,
        )
    
    def get_trace(self, correlation_id: str) -> SynapseTrace:
        events = [e for e in self._events if e.correlation_id == correlation_id]
        return SynapseTrace(
            correlation_id=correlation_id,
            events=events,
            total_latency_ms=sum(e.latency_ms for e in events),
            agents_used=list(set(e.source for e in events if e.source != "osa")),
        )
```

### TaskDAG — For OSA Decomposer output

```python
class TaskNode(BaseModel):
    """A single task in the decomposed DAG."""
    id: str
    agent: str                       # "web", "code", "memory", "sysstat", "pnl"
    sub_agent: str = ""              # "finance", "news", "search", "planner", etc.
    input_query: str
    instruction: str = ""
    depends_on: list[str] = Field(default_factory=list)  # task IDs this depends on

class TaskDAG(BaseModel):
    """Directed acyclic graph of tasks produced by OSA.Decomposer."""
    correlation_id: str
    nodes: list[TaskNode]
    parallel_groups: list[list[str]] = Field(default_factory=list)  # groups of task IDs to run concurrently
    reasoning: str = ""              # Decomposer's reasoning for this decomposition
```

---

## 📋 AGENT SPECIFICATIONS

### OSA (Orchestrator & Synapse Agent)
**Location:** `octane/osa/`
**Role:** Brain + nervous system. Receives ALL queries, decomposes, routes, evaluates.

| Sub-agent | Type | Model | Responsibility |
|-----------|------|-------|----------------|
| Decomposer | LLM-powered | Big (30B/8B) | Query → TaskDAG. Reasons about what agents/data are needed. |
| Router | Deterministic | None / Small | TaskNode → agent instance mapping. Pattern matching + routing table. |
| Evaluator | LLM-powered | Big (30B/8B) | Reviews assembled results. Quality gate. Can trigger re-execution. |
| Policy Engine | Deterministic | None | Rules: max retries, HITL triggers, allowed actions. Pure Python logic. |
| Guard | Hybrid | Small + regex | Input/output safety. Regex for injection, small model for semantic checks. |

**OSA.Orchestrator main loop:**
```python
async def run(self, query: str, session_id: str) -> str:
    correlation_id = str(uuid.uuid4())
    
    # 1. Guard check (parallel)
    guard_task = asyncio.create_task(self.guard.check(query))
    
    # 2. Get user profile from P&L
    profile = await self.pnl_agent.get_profile()
    
    # 3. Decompose
    dag = await self.decomposer.decompose(query, profile, correlation_id)
    
    # 4. Check guard result
    if not (await guard_task).is_safe:
        return "Query blocked by safety filter."
    
    # 5. Route and dispatch
    results = await self.dispatch(dag, correlation_id)
    
    # 6. Evaluate
    output = await self.evaluator.evaluate(query, results, profile, correlation_id)
    
    # 7. Record in P&L + Memory
    await self.pnl_agent.record_interaction(query, output, correlation_id)
    await self.memory_agent.maybe_persist(query, output, results, correlation_id)
    
    return output
```

### Web Agent
**Location:** `octane/agents/web/`
**Role:** ALL internet-facing data retrieval. Finance, news, search, entertainment.
**Does NOT call Bodega APIs directly from sub-agents.** Uses `BodegaIntelClient` from tools layer.

| Sub-agent | Model | Responsibility |
|-----------|-------|----------------|
| Query Strategist | Small (0.9B/6B) | Takes query + context, generates 1-3 search parameter variations |
| Fetcher | None (HTTP only) | Calls Bodega APIs, uses Trafilatura for HTML extraction |
| Browser | None (Playwright) | JS-rendered pages. Only when Fetcher fails or target needs JS |
| Synthesizer | Medium (6B/14B) | Raw results → structured intelligence with sources |

**WebAgent coordinator pattern:**
```python
class WebAgent(BaseAgent):
    name = "web"
    description = "All internet-facing data retrieval"

    async def execute(self, request: AgentRequest) -> AgentResponse:
        # 1. Generate search strategies
        strategies = await self.query_strategist.run(request)
        
        # 2. Fetch data (may hit multiple Bodega APIs)
        raw_results = await self.fetcher.run(AgentRequest(
            query=request.query,
            context={"strategies": strategies.result},
            metadata=request.metadata,
        ))
        
        # 3. If fetcher failed on any URL, try browser
        failed_urls = raw_results.metadata.get("failed_urls", [])
        if failed_urls:
            browser_results = await self.browser.run(AgentRequest(
                query=request.query,
                context={"urls": failed_urls},
                metadata=request.metadata,
            ))
            raw_results.result.extend(browser_results.result or [])
        
        # 4. Synthesize into structured intelligence
        synthesis = await self.synthesizer.run(AgentRequest(
            query=request.query,
            context={"raw_data": raw_results.result},
            metadata=request.metadata,
        ))
        
        return synthesis
```

**How Bodega APIs map to Web Agent:**
```
Web Agent receives: "NVIDIA stock price"
  → Query Strategist: "This is a finance query"
  → Fetcher: calls localhost:8030/api/v1/finance/market/NVDA
  → Synthesizer: structures the response

Web Agent receives: "Latest AI news"
  → Query Strategist: "This is a news query"
  → Fetcher: calls localhost:8032/api/v1/news/search?q=AI&period=1d
  → Synthesizer: summarizes top stories

Web Agent receives: "What is transformer architecture"
  → Query Strategist: "This is a general search query"
  → Fetcher: calls localhost:1111/intelligence/search?query=transformer+architecture
  → Synthesizer: creates structured summary from search results
```

### Code Agent
**Location:** `octane/agents/code/`
**Role:** Code generation, execution, validation, and self-healing.

| Sub-agent | Model | Responsibility |
|-----------|-------|----------------|
| Planner | Big (30B/8B) | Task description → code specification (language, libs, approach) |
| Writer | Code model (axe-turbo) | Specification → actual code |
| Executor | None (subprocess) | Venv creation, pip install, run code, capture output |
| Debugger | Medium (14B/8B) | stderr + code → diagnosis + fix |
| Validator | Small + deterministic | Check: exit code, output assertions, schema validation |

**Self-healing loop:**
```python
class CodeAgent(BaseAgent):
    name = "code"
    description = "Code generation, execution, and validation"

    async def execute(self, request: AgentRequest) -> AgentResponse:
        # 1. Plan
        spec = await self.planner.run(request)
        
        # 2. Write code
        code_result = await self.writer.run(AgentRequest(
            query=request.query,
            context={"spec": spec.result},
            metadata=request.metadata,
        ))
        
        max_retries = 3  # From PolicyEngine
        for attempt in range(max_retries):
            # 3. Execute
            exec_result = await self.executor.run(AgentRequest(
                query="execute",
                context={"code": code_result.result, "requirements": spec.result.get("requirements", [])},
                metadata=request.metadata,
            ))
            
            # 4. Validate
            validation = await self.validator.run(AgentRequest(
                query="validate",
                context={"execution": exec_result.result, "spec": spec.result},
                metadata=request.metadata,
            ))
            
            if validation.result.get("passed", False):
                return AgentResponse(
                    result={
                        "code": code_result.result,
                        "output": exec_result.result,
                        "validation": validation.result,
                        "attempts": attempt + 1,
                    },
                    metadata={"attempts": attempt + 1},
                )
            
            # 5. Debug and retry
            debug_result = await self.debugger.run(AgentRequest(
                query="debug",
                context={
                    "code": code_result.result,
                    "error": exec_result.result.get("stderr", ""),
                    "validation": validation.result,
                },
                metadata=request.metadata,
            ))
            code_result = debug_result  # Updated code for next attempt
        
        # All retries exhausted
        return AgentResponse(
            result={"code": code_result.result, "error": "Max retries exhausted"},
            success=False,
            error="Code validation failed after 3 attempts",
        )
```

### Memory Agent
**Location:** `octane/agents/memory/`
**Role:** Three-tier memory system: hot (Redis), warm (Postgres), cold (pgVector).

| Sub-agent | Model | Responsibility |
|-----------|-------|----------------|
| Router | Small / heuristic | Decides which tier to query based on request semantics |
| Writer | Small | Evaluates what to persist and in which tier |
| Janitor | None (cron/perpetual) | Background: promote/demote between tiers based on access patterns |

**Tier routing logic:**
```python
class MemoryRouter:
    """Decides which memory tier to query."""
    
    async def route(self, request: AgentRequest) -> str:
        query = request.query.lower()
        context = request.context
        
        # Hot (Redis): session context, recent queries, follow-ups
        if context.get("session_id") and any(kw in query for kw in [
            "just said", "earlier", "previous", "that", "it", "follow up"
        ]):
            return "hot"
        
        # Warm (Postgres): structured queries, user data, preferences
        if any(kw in query for kw in [
            "my portfolio", "my history", "preference", "setting",
            "last week", "all queries", "statistics"
        ]):
            return "warm"
        
        # Cold (pgVector): semantic similarity, deep knowledge search
        return "cold"
```

### SysStat Agent
**Location:** `octane/agents/sysstat/`
**Role:** Resource monitoring, model loading strategy, adaptive scaling. Reports to OSA continuously.

| Sub-agent | Model | Responsibility |
|-----------|-------|----------------|
| Monitor | None (psutil) | RAM, CPU, tokens/sec, queue depth metrics |
| Model Manager | None (HTTP) | Loads/unloads models via Bodega admin API based on RAM |
| Scaler | Deterministic | Adaptive model topology based on resource pressure |

**Model loading strategy:**
```python
MODEL_TOPOLOGIES = {
    "64gb": [
        {"path": "SRSWTI/axe-turbo-31b", "role": "brain", "max_concurrent": 3},
        {"path": "SRSWTI/bodega-raptor-8b-mxfp4", "role": "worker", "max_concurrent": 8},
        {"path": "SRSWTI/bodega-raptor-0.9b", "role": "grunt", "max_concurrent": 50},
    ],
    "32gb": [
        {"path": "SRSWTI/bodega-raptor-8b-mxfp4", "role": "brain", "max_concurrent": 3},
        {"path": "SRSWTI/bodega-raptor-0.9b", "role": "grunt", "max_concurrent": 30},
    ],
    "16gb": [
        {"path": "SRSWTI/bodega-raptor-8b-mxfp4", "role": "brain", "max_concurrent": 2},
        {"path": "SRSWTI/bodega-raptor-0.9b", "role": "grunt", "max_concurrent": 20},
    ],
}
```

**CRITICAL: SysStat is the ONLY agent that calls Bodega admin endpoints (load/unload). No other agent manages models directly.**

### P&L Agent (Persona & Learning)
**Location:** `octane/agents/pnl/`
**Role:** User personalization. Tracks preferences, feedback, engagement. Consulted before synthesis.

| Sub-agent | Model | Responsibility |
|-----------|-------|----------------|
| Preference Manager | None (CRUD) | Read/write user preferences in Postgres |
| Feedback Learner | Small | Processes like/dislike/time-spent signals into preference updates |
| Profile | None (aggregation) | Assembles current user profile from all signals |

**User profile schema:**
```python
class UserProfile(BaseModel):
    user_id: str = "default"
    domain_interests: list[str] = []          # ["finance", "AI", "coding"]
    expertise_level: str = "advanced"          # beginner | intermediate | advanced
    preferred_verbosity: str = "concise"       # concise | balanced | detailed
    preferred_domains: dict[str, float] = {}   # {"finance": 0.8, "news": 0.6}
    liked_content_tags: list[str] = []
    disliked_content_tags: list[str] = []
    total_interactions: int = 0
    last_active: datetime | None = None
```

**P&L is consulted by OSA.Evaluator before generating final output.** The profile influences synthesis tone, detail level, and domain emphasis.

---

## 🔌 EXTERNAL SERVICES — COMPLETE API REFERENCE

### Bodega Inference Engine (LOCAL LLM)
```
Base URL: http://localhost:44468
OpenAI-compatible API. Managed ONLY by SysStat.ModelManager.

POST /v1/chat/completions       — Inference (streaming supported)
POST /v1/admin/load-model       — Load model (SysStat only)
POST /v1/admin/unload-model     — Unload model (SysStat only)
GET  /v1/admin/current-model    — Check loaded model
GET  /v1/models                 — List cached models
GET  /v1/tools                  — List available built-in tools
POST /v1/tools/execute          — Execute a built-in tool
GET  /health                    — Server health
GET  /v1/queue/stats            — Queue performance metrics

Key models (by role):
  Brain (big reasoning):
    - SRSWTI/axe-turbo-31b (64GB)
    - SRSWTI/bodega-centenario-21b-mxfp4 (32-64GB)
    - SRSWTI/bodega-raptor-8b-mxfp4 (16-32GB)

  Worker (intermediate tasks):
    - SRSWTI/bodega-raptor-8b-mxfp4
    - SRSWTI/bodega-vertex-4b
    
  Grunt (fast, high-concurrency):
    - SRSWTI/bodega-raptor-0.9b (400+ tok/s)
    - SRSWTI/bodega-raptor-90m (edge, sub-100M)
    
  Code-specialized:
    - SRSWTI/axe-turbo-1b (150 tok/s, sub-50ms first token)
    - SRSWTI/bodega-solomon-9b (agentic coding)

Load with LoRA adapters:
  POST /v1/admin/load-model
  {"model_path": "...", "lora_paths": ["./adapters/persona.bin"], "lora_scales": [1.0]}

Load with tool calling:
  {"model_path": "...", "tool_call_parser": "qwen3", "reasoning_parser": "qwen3"}

Chat completion with tools:
  POST /v1/chat/completions
  {"model": "current", "messages": [...], "tools": [...], "tool_choice": "auto"}
```

### Bodega Intelligence (EXTERNAL DATA — used by Web Agent only)
```
Beru Search: http://localhost:1111
  GET /intelligence/search?query=...              — AI-powered search
  GET /search/web?query=...                       — Web search
  GET /search/images?query=...                    — Image search

Finance API: http://localhost:8030
  GET /api/v1/finance/market/{ticker}             — Real-time market data
  GET /api/v1/finance/complete?q=...              — Natural language finance query
  GET /api/v1/finance/timeseries/{ticker}?period=1mo&interval=1d — Historical data

News API: http://localhost:8032
  GET /api/v1/news/headlines                      — Breaking news
  GET /api/v1/news/search?q=...&period=3d         — Search news
  GET /api/v1/news/topics/{TOPIC}?period=1d       — Topic: TECHNOLOGY, BUSINESS, etc.

Entertainment API: http://localhost:8031
  GET /api/v1/entertainment/movies/search?q=...   — Movie search
  GET /api/v1/entertainment/tv/search?q=...       — TV show search
```

### PostgreSQL + pgVector
```
Connection: postgresql://localhost:5432/octane
Tables:
  - memory_chunks: (id, content, embedding vector(1536), metadata jsonb, tier, created_at, accessed_at)
  - user_preferences: (user_id, key, value jsonb, updated_at)
  - feedback_signals: (id, correlation_id, signal_type, value, created_at)
  - synapse_events: (id, correlation_id, event_type, source, target, payload jsonb, created_at)

Use asyncpg for ALL database access. Never use synchronous drivers.
```

### Redis
```
URL: redis://localhost:6379/0
Used for:
  1. Shadows task queue (Redis Streams) — managed by shadow-task library
  2. Memory hot cache — session context, recent results (TTL-based)
  3. Synapse event stream (Phase 4) — real-time event bus
```

---

## 🌊 SHADOWS INTEGRATION

Shadows is the neural bus. It handles all async task dispatch between OSA and agents.

### Setup Pattern
```python
from shadows import Shadow, Worker

# In octane/main.py
async def startup():
    shadows = Shadow(name="octane", url="redis://localhost:6379/0")
    
    # Register all agent execute methods as shadow tasks
    shadows.register(web_agent.execute)
    shadows.register(code_agent.execute)
    shadows.register(memory_agent.execute)
    shadows.register(sysstat_agent.execute)
    shadows.register(pnl_agent.execute)
    
    return shadows
```

### Task Dispatch (in OSA Router)
```python
from shadows import CurrentShadow

async def dispatch_task(self, node: TaskNode, request: AgentRequest):
    """Dispatch a task to the appropriate agent via Shadows."""
    agent = self.agent_registry[node.agent]
    
    # Add via Shadows for at-least-once delivery
    await self.shadows.add(agent.execute)(request)
```

### SysStat Perpetual Monitoring
```python
from shadows import Perpetual
from datetime import timedelta

async def monitor_system(
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=30))
) -> None:
    """Runs every 30 seconds, reports to OSA."""
    metrics = await collect_system_metrics()
    await synapse.emit(SynapseEvent(
        correlation_id="sysstat_monitor",
        event_type="sysstat_report",
        source="sysstat",
        target="osa",
        payload_summary=str(metrics),
    ))
```

### Retry for Flaky External APIs
```python
from shadows import ExponentialRetry
from datetime import timedelta

async def fetch_finance_data(
    ticker: str,
    retry: ExponentialRetry = ExponentialRetry(
        attempts=3,
        minimum_delay=timedelta(seconds=1),
        maximum_delay=timedelta(seconds=30),
    )
) -> dict:
    """Fetch finance data with automatic retry on failure."""
    response = await bodega_intel.get_finance(ticker)
    return response
```

---

## 🚫 DO NOT DO THESE THINGS

1. **Do NOT use `requests` library** — Use `httpx` (async). We are fully async.
2. **Do NOT use raw dicts for data** — Use Pydantic models. Always.
3. **Do NOT let agents communicate directly** — Everything flows through OSA + Synapse.
4. **Do NOT call Bodega admin endpoints from anywhere except SysStat.ModelManager** — Model management is centralized.
5. **Do NOT build a web UI yet** — CLI first. React UI is Phase 5+.
6. **Do NOT create monolithic agents** — Each agent coordinates sub-agents. Sub-agents do ONE thing.
7. **Do NOT hardcode model names** — Use config.py. SysStat.ModelManager resolves "brain"/"worker"/"grunt" to actual model paths.
8. **Do NOT use synchronous I/O in any agent** — Everything is `async def`.
9. **Do NOT catch and silence exceptions** — Log them with structlog, emit Synapse error event, then return AgentResponse with success=False.
10. **Do NOT import agents from other agents** — Dependency flows DOWN only: CLI → OSA → Agents → Tools → External Services.
11. **Do NOT bypass Shadows for task dispatch** — In Phase 1, simple `shadows.add()` is fine. But everything goes through Shadows.
12. **Do NOT put LLM calls in deterministic components** — Router and PolicyEngine are deterministic/small-model only. Save big models for Decomposer and Evaluator.

---

## 📁 DEPENDENCY FLOW — NEVER VIOLATE

```
CLI (Typer)
  └── OSA (Orchestrator)
        ├── OSA sub-agents (Decomposer, Router, Evaluator, Guard, Policy)
        ├── Shadows (task dispatch)
        └── Agents (Web, Code, Memory, SysStat, P&L)
              ├── Sub-agents (Query Strategist, Fetcher, Planner, etc.)
              └── Tools (BodegaInferenceClient, BodegaIntelClient, PgClient, RedisClient, Sandbox)
                    └── External Services (localhost:44468, :1111, :8030, :8032, Postgres, Redis)
```

**Import rule:** Dependencies flow DOWN only. Never import upward. Never import across (agent to agent).

---

## 🌐 ENVIRONMENT VARIABLES (.env)

```env
# === Bodega Inference Engine ===
BODEGA_INFERENCE_URL=http://localhost:44468
BODEGA_INFERENCE_TIMEOUT=120

# === Bodega Intelligence APIs ===
BERU_SEARCH_URL=http://localhost:1111
FINANCE_API_URL=http://localhost:8030
NEWS_API_URL=http://localhost:8032
ENTERTAINMENT_API_URL=http://localhost:8031

# === PostgreSQL ===
DATABASE_URL=postgresql://localhost:5432/octane
PGVECTOR_DIMENSION=1536

# === Redis ===
REDIS_URL=redis://localhost:6379/0

# === Model Roles (resolved by SysStat.ModelManager) ===
BRAIN_MODEL=SRSWTI/bodega-raptor-8b-mxfp4
WORKER_MODEL=SRSWTI/bodega-vertex-4b
GRUNT_MODEL=SRSWTI/bodega-raptor-0.9b
CODE_MODEL=SRSWTI/axe-turbo-1b

# === SandboxAgent ===
SANDBOX_TIMEOUT_SECONDS=30
SANDBOX_BASE_DIR=/tmp/octane-sandboxes
SANDBOX_MAX_RETRIES=3

# === SysStat ===
SYSSTAT_MONITOR_INTERVAL_SECONDS=30
SYSSTAT_RAM_THRESHOLD_PERCENT=85

# === P&L ===
DEFAULT_EXPERTISE_LEVEL=advanced
DEFAULT_VERBOSITY=concise

# === Logging ===
LOG_LEVEL=INFO
LOG_FORMAT=json

# === Shadows ===
SHADOWS_NAME=octane
SHADOWS_CONCURRENCY=10
```

---

## 🧪 TESTING PATTERNS

### Agent Test Pattern
```python
import pytest
from unittest.mock import AsyncMock
from octane.agents.web.agent import WebAgent
from octane.models.schemas import AgentRequest
from octane.models.synapse import SynapseEventBus

@pytest.fixture
def synapse():
    return SynapseEventBus()

@pytest.fixture
def web_agent(synapse):
    agent = WebAgent(synapse=synapse)
    # Mock external dependencies
    agent.fetcher.bodega_client = AsyncMock()
    return agent

@pytest.mark.asyncio
async def test_web_agent_finance_query(web_agent):
    """Web Agent correctly routes finance queries to Bodega Finance API."""
    web_agent.fetcher.bodega_client.get_finance.return_value = {
        "ticker": "NVDA", "price": 142.50, "change": -3.2
    }
    
    request = AgentRequest(
        query="NVIDIA stock price",
        metadata={"correlation_id": "test_123", "source": "osa"},
    )
    response = await web_agent.run(request)
    
    assert response.success is True
    assert "NVDA" in str(response.result)
    assert response.latency_ms > 0

@pytest.mark.asyncio
async def test_web_agent_emits_synapse_events(web_agent, synapse):
    """Every agent execution emits ingress and egress Synapse events."""
    request = AgentRequest(
        query="test",
        metadata={"correlation_id": "test_456", "source": "osa"},
    )
    await web_agent.run(request)
    
    trace = synapse.get_trace("test_456")
    assert len(trace.events) >= 2  # at least ingress + egress
    assert trace.events[0].event_type == "agent_ingress"
```

### Shadows Test Pattern
```python
from shadows import Shadow, Worker
from uuid import uuid4

@pytest.fixture
async def test_shadows():
    async with Shadow(name=f"test-{uuid4()}") as shadows:
        yield shadows
        await shadows.clear()

@pytest.mark.asyncio
async def test_task_dispatch_via_shadows(test_shadows):
    """Tasks dispatched via Shadows are executed by workers."""
    results = []
    
    async def mock_agent_execute(query: str):
        results.append(query)
    
    test_shadows.register(mock_agent_execute)
    await test_shadows.add(mock_agent_execute)("test query")
    
    async with Worker(test_shadows) as worker:
        await worker.run_until_finished()
    
    assert results == ["test query"]
```

---

## 💡 PROMPTING TIPS FOR COPILOT

When asking Copilot to generate code for this project:

1. **Always start with:** "Following OCTANE_AI_IDE.md patterns..."
2. **Specify the layer:** "In the tools layer, create..." or "In the OSA orchestrator..."
3. **Reference the contract:** "It must extend BaseAgent with execute() method and emit Synapse events"
4. **Name the sub-agent:** "Create the WebAgent.QueryStrategist sub-agent"
5. **Mention async + httpx:** "Use httpx async client, all methods are async def"
6. **Ask for tests:** "Include pytest-asyncio tests with mocked dependencies"

### Example Prompts

**Creating an agent:**
```
Following OCTANE_AI_IDE.md patterns, create the WebAgent Fetcher sub-agent 
in octane/agents/web/fetcher.py.

It should:
- Extend BaseAgent
- Use BodegaIntelClient from octane/tools/bodega_intel.py
- Accept strategies from QueryStrategist (list of {api: str, params: dict})
- Route to correct Bodega API (finance :8030, news :8032, search :1111)
- Return raw results as list of dicts
- Handle API failures gracefully (continue with successful results)
- Emit Synapse events for each API call
- Include health() check that pings all Bodega endpoints

Also create tests/test_web/test_fetcher.py with:
- Happy path: finance query returns data
- Partial failure: one API fails, others succeed
- Total failure: all APIs down, returns error gracefully
```

**Creating a tool:**
```
Following OCTANE_AI_IDE.md patterns, create the BodegaIntelClient 
in octane/tools/bodega_intel.py.

It should:
- Use httpx.AsyncClient with configurable base URLs from config
- Methods: get_search(query), get_finance(ticker), get_news(query, period), 
  get_timeseries(ticker, period, interval), get_headlines()
- Each method returns parsed JSON response
- Timeout from config (default 30s)
- Structured error handling (httpx.HTTPStatusError → meaningful error messages)
- health() method that pings all 4 services

Endpoints from OCTANE_AI_IDE.md:
  Beru: localhost:1111, Finance: localhost:8030, News: localhost:8032
```

---

## 📌 KEY DECISIONS LOG

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent hierarchy | OSA → Agents → Sub-agents | Biological model. Clear separation. OSA controls all state. |
| Agent communication | Always through OSA + Synapse | No direct agent-to-agent. Full traceability. |
| Task dispatch | Shadows (Redis Streams) | At-least-once delivery, retries, scheduling, perpetual tasks. |
| Deterministic where possible | Router + Policy = no LLM | Avoid OSA bottleneck. Fast, predictable routing. |
| Big model usage | Decomposer + Evaluator only | Expensive reasoning only where it matters. |
| Web Agent consolidation | Finance/News/Search = sub-types, not separate agents | Cleaner. One agent, multiple data sources. |
| Guard inside OSA | Sub-agent, not separate top-level | Centralized safety. Runs parallel with main processing. |
| SysStat owns model management | Only agent calling admin endpoints | Prevents model thrashing from multiple agents. |
| P&L from Day 1 | Not deferred to Phase 3 | Personalization data accumulates from first interaction. |
| CLI-first | No web UI in Week 1 | Agents must be solid before adding React layer. |
| Python 3.12+ | Required by Shadows | Also gives us better typing and performance. |
| Horizontal Phase scaling | All layers basic → all layers deeper | Avoids deep agent with no integration. E2E flow matters most. |
