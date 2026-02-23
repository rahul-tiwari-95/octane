# Project Octane — 4-Phase Build Plan (v2)

## Executive Summary

Octane is a local-first, composable agentic operating system for Apple Silicon. It is architected as a **biological system**: OSA (Orchestrator & Synapse Agent) is the brain and nervous system; Web, Code, Memory, SysStat, and P&L agents are specialized organs; Shadows is the neural bus that carries signals between them.

**Key architectural principle:** Build all layers to basic foundation in Week 1, then add complexity horizontally across all layers simultaneously — not vertically on one.

**Stack:** Python 3.12+ / FastAPI / asyncio + Shadows (Redis Streams) / Bodega Inference Engine (MLX, :44468) / Bodega Intelligence (:1111, :8030-8032) / Postgres + pgVector / Redis

---

## The Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OSA (Orchestrator & Synapse Agent)                    │
│                                                                         │
│  Sub-agents:                                                            │
│  ┌────────────┐ ┌────────┐ ┌───────────┐ ┌───────────────┐ ┌────────┐ │
│  │ Decomposer │ │ Router │ │ Evaluator │ │ Policy Engine │ │ Guard  │ │
│  │ (big model)│ │ (determ│ │(big model)│ │(deterministic)│ │(hybrid)│ │
│  └────────────┘ │ /small)│ └───────────┘ └───────────────┘ └────────┘ │
│                  └────────┘                                             │
│  Infrastructure:                                                        │
│  ┌──────────────────────┐  ┌──────────────────────────────┐            │
│  │ Synapse EventBus     │  │ Shadows Task Orchestration    │            │
│  │ (ingress/egress logs │  │ (Redis Streams, scheduling,   │            │
│  │  trace, correlation) │  │  retries, perpetual tasks)    │            │
│  └──────────────────────┘  └──────────────────────────────┘            │
└───────┬──────────┬──────────────┬──────────────┬──────────────┬────────┘
        │          │              │              │              │
   ┌────┴───┐ ┌───┴────┐  ┌─────┴──────┐ ┌────┴─────┐ ┌──────┴──────┐
   │  Web   │ │  Code  │  │  Memory    │ │ SysStat  │ │    P&L      │
   │  Agent │ │  Agent │  │  Agent     │ │  Agent   │ │   Agent     │
   │        │ │        │  │            │ │          │ │(Persona &   │
   │ ┌────┐ │ │ ┌────┐ │  │ ┌────────┐ │ │ ┌──────┐ │ │ Learning)   │
   │ │Qry │ │ │ │Plan│ │  │ │Router  │ │ │ │Monitr│ │ │ ┌─────────┐ │
   │ │Strt│ │ │ │Wrtr│ │  │ │(Hot/   │ │ │ │Model │ │ │ │Pref Mgr │ │
   │ │Ftch│ │ │ │Exec│ │  │ │Warm/   │ │ │ │Mgr   │ │ │ │Feedback │ │
   │ │Brws│ │ │ │Dbug│ │  │ │Cold)   │ │ │ │Scaler│ │ │ │Learner  │ │
   │ │Synt│ │ │ │Vald│ │  │ │Writer  │ │ │ └──────┘ │ │ │Profile  │ │
   │ └────┘ │ │ └────┘ │  │ │Janitor │ │ │          │ │ └─────────┘ │
   └────────┘ └────────┘  │ └────────┘ │ └──────────┘ └─────────────┘
                           └────────────┘
```

---

## End-to-End Flow Example

### Query: "What's happening with NVIDIA today and should I be worried about my position?"

```
STEP 1: INGRESS
──────────────────────────────────────────────────────────────────
User types query in CLI
    ↓
OSA receives query
    ↓
Synapse creates SynapseEvent:
  {
    correlation_id: "evt_abc123",
    type: "ingress",
    source: "user",
    target: "osa.decomposer",
    payload: "What's happening with NVIDIA today...",
    timestamp: "2026-02-16T10:00:00Z",
    metadata: { session_id: "sess_1", turn: 1 }
  }
    ↓
Guard Agent (parallel) — scans input for safety/injection → PASS
    ↓
P&L Agent consulted — "User is a principal engineer, interested in
  finance, prefers concise technical analysis, has NVDA in watchlist"

STEP 2: DECOMPOSITION
──────────────────────────────────────────────────────────────────
OSA.Decomposer (big model) reasons:
  "This query has two parts:
   1. Current NVIDIA news and market data (factual)
   2. Position risk assessment (analytical, needs user context)
   I need: web.finance + web.news + memory.read(portfolio) + synthesis"

  Produces task DAG:
  {
    nodes: [
      { id: "t1", agent: "web", sub: "finance", input: "NVDA market data" },
      { id: "t2", agent: "web", sub: "news", input: "NVIDIA latest news" },
      { id: "t3", agent: "memory", sub: "read", input: "user portfolio NVDA position" },
      { id: "t4", agent: "osa.synthesize", input: "[t1,t2,t3]",
        instruction: "Analyze NVIDIA position risk based on market data,
                      news, and user's portfolio context" }
    ],
    edges: [
      { from: "t1", to: "t4" },
      { from: "t2", to: "t4" },
      { from: "t3", to: "t4" }
    ],
    parallel_groups: [["t1", "t2", "t3"]]  // these run concurrently
  }

Synapse logs: { type: "decomposition", dag: <above>, model_used: "30B",
                reasoning: "Two-part query: factual + analytical..." }

STEP 3: ROUTING & DISPATCH
──────────────────────────────────────────────────────────────────
OSA.Router (deterministic/small model) maps each node:
  t1 → Web Agent → Query Strategist → Fetcher (Bodega Finance :8030)
  t2 → Web Agent → Query Strategist → Fetcher (Bodega News :8032)
  t3 → Memory Agent → Router → Warm tier (Postgres portfolio table)

OSA dispatches via Shadows:
  await shadows.add(web_agent.execute)(task=t1)
  await shadows.add(web_agent.execute)(task=t2)
  await shadows.add(memory_agent.execute)(task=t3)

Synapse logs each dispatch:
  { type: "dispatch", target: "web_agent", task_id: "t1", ... }
  { type: "dispatch", target: "web_agent", task_id: "t2", ... }
  { type: "dispatch", target: "memory_agent", task_id: "t3", ... }

STEP 4: AGENT EXECUTION (PARALLEL)
──────────────────────────────────────────────────────────────────
Web Agent (t1 — Finance):
  Query Strategist → generates: ["NVDA stock price today", "NVIDIA market cap"]
  Fetcher → hits localhost:8030/api/v1/finance/market/NVDA
  Fetcher → hits localhost:8030/api/v1/finance/timeseries/NVDA?period=5d
  Synthesizer → "NVDA trading at $142.50, down 3.2% today, volume 2x avg..."
  → Returns structured FinanceResult to OSA via Synapse

Web Agent (t2 — News):
  Query Strategist → generates: ["NVIDIA news today", "NVIDIA earnings 2026"]
  Fetcher → hits localhost:8032/api/v1/news/search?q=NVIDIA&period=1d
  Synthesizer → "3 key stories: export restrictions update, earnings preview..."
  → Returns structured NewsResult to OSA via Synapse

Memory Agent (t3 — Portfolio):
  Router → determines this is structured data → Warm tier (Postgres)
  Queries: SELECT * FROM portfolio WHERE ticker='NVDA' AND user_id=...
  → Returns: { shares: 150, avg_cost: 128.30, current_value: 21375 }

SysStat Agent (continuous background):
  Reports: "30B model at 67% utilization, 14B idle, RAM 48/64GB"

STEP 5: SYNTHESIS & EVALUATION
──────────────────────────────────────────────────────────────────
OSA receives all three results via Synapse events
    ↓
OSA.Evaluator (big model) assembles context:
  - Finance data: NVDA $142.50, -3.2%, high volume
  - News: export restrictions, earnings preview, analyst downgrades
  - Portfolio: 150 shares at $128.30 avg cost, still profitable at +11%
  - P&L context: user prefers concise technical analysis
    ↓
Generates final synthesis using big model:
  "NVIDIA is down 3.2% today on elevated volume, primarily driven by
   renewed export restriction concerns. Your position of 150 shares
   is still up ~11% from your cost basis. Key risk: earnings report
   next week could amplify volatility. The export news is regulatory
   noise unless specific new restrictions are announced."
    ↓
Guard Agent (parallel) — scans output for safety → PASS

STEP 6: EGRESS & LEARNING
──────────────────────────────────────────────────────────────────
Synapse logs final event:
  {
    correlation_id: "evt_abc123",
    type: "egress",
    total_latency_ms: 4200,
    agents_used: ["web", "memory", "osa"],
    models_used: ["30B", "6B"],
    tokens_in: 2400,
    tokens_out: 180,
    task_count: 4
  }

Output delivered to user via CLI

P&L Agent records:
  - Query pattern: finance + portfolio analysis
  - If user gives thumbs up/down → reward signal stored
  - Time spent reading → engagement signal
  - Updates preference model for future queries

Memory Agent.Writer evaluates:
  "This synthesis contains current market context worth caching"
  → Writes to Redis hot cache (TTL: 4 hours) for follow-up questions
```

---

## Phase 1: Foundation Across All Layers (Days 1-3)

### Philosophy
Build the skeleton of EVERY component. Nothing is deep — everything is wide. Every agent exists, every sub-agent has a stub, the Synapse bus works, Shadows is integrated. You can trace a query from ingress to egress, even if the processing is basic.

### Day 1: Scaffold + Core Infrastructure

**Project Structure:**
```
octane/
├── octane/
│   ├── __init__.py
│   ├── main.py                      # CLI entry (Typer)
│   ├── config.py                    # Pydantic Settings
│   ├── osa/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # Main OSA loop
│   │   ├── decomposer.py           # Query → task DAG
│   │   ├── router.py               # Task → agent mapping (deterministic)
│   │   ├── evaluator.py            # Output quality gate
│   │   ├── policy.py               # Deterministic rules engine
│   │   ├── guard.py                # Input/output safety checks
│   │   └── synapse.py              # EventBus: SynapseEvent model + logging
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent ABC
│   │   ├── web/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # WebAgent coordinator
│   │   │   ├── query_strategist.py  # Multi-query generation
│   │   │   ├── fetcher.py           # HTTP fetch + Trafilatura
│   │   │   ├── browser.py           # Playwright scraper (stub)
│   │   │   └── synthesizer.py       # Raw data → structured intel
│   │   ├── code/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # CodeAgent coordinator
│   │   │   ├── planner.py           # Task → code spec
│   │   │   ├── writer.py            # Spec → code generation
│   │   │   ├── executor.py          # Venv + subprocess
│   │   │   ├── debugger.py          # Error → fix loop
│   │   │   └── validator.py         # Output verification
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # MemoryAgent coordinator
│   │   │   ├── router.py            # Decide: hot/warm/cold
│   │   │   ├── writer.py            # Decide what/where to persist
│   │   │   └── janitor.py           # Tier promotion/demotion (stub)
│   │   ├── sysstat/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # SysStatAgent coordinator
│   │   │   ├── monitor.py           # RAM/CPU/token metrics
│   │   │   ├── model_manager.py     # Load/unload models via Bodega
│   │   │   └── scaler.py            # Adaptive model topology (stub)
│   │   └── pnl/
│   │       ├── __init__.py
│   │       ├── agent.py             # P&L Agent coordinator
│   │       ├── preference_manager.py # User prefs CRUD
│   │       ├── feedback_learner.py   # Like/dislike/time signals
│   │       └── profile.py           # Aggregated user profile
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── bodega_inference.py      # Async client for :44468
│   │   ├── bodega_intel.py          # Async client for :1111, :8030, :8032
│   │   ├── pg_client.py             # asyncpg + pgVector helpers
│   │   ├── redis_client.py          # Redis connection for Shadows + cache
│   │   └── sandbox.py               # Venv creation + subprocess execution
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py               # AgentRequest, AgentResponse
│   │   ├── synapse.py               # SynapseEvent, SynapseTrace
│   │   └── dag.py                   # TaskDAG, TaskNode, TaskEdge
│   └── utils/
│       ├── __init__.py
│       └── logging.py               # structlog config
├── tests/
│   ├── test_osa/
│   ├── test_web/
│   ├── test_code/
│   ├── test_memory/
│   ├── test_sysstat/
│   └── test_pnl/
├── .env
├── pyproject.toml
├── OCTANE_AI_IDE.md
└── README.md
```

**Day 1 Deliverables:**
- [ ] `pyproject.toml` with all dependencies (shadow-task, httpx, asyncpg, typer, rich, structlog, pydantic-settings, trafilatura)
- [ ] `config.py` with all env vars
- [ ] `BaseAgent` ABC with `run()` wrapper (timing, error handling, Synapse event emission)
- [ ] `SynapseEvent` Pydantic model + `SynapseEventBus` (in-memory list for now, Redis Stream later)
- [ ] `BodegaInferenceClient` (async httpx → :44468)
- [ ] `BodegaIntelClient` (async httpx → :1111, :8030, :8032)
- [ ] Basic Shadows integration: `Shadow()` instance in main, `Worker` startup
- [ ] CLI skeleton: `octane health`, `octane ask "test"`

### Day 2: All Agents — Skeleton + Basic Functionality

Every agent gets its coordinator + sub-agent stubs. Focus on one happy path per agent.

**Web Agent:**
- [ ] `WebAgent.execute()` — coordinates sub-agents
- [ ] `QueryStrategist` — takes query, returns 1-3 search variations (using small model)
- [ ] `Fetcher` — calls Bodega Intel APIs, returns raw results
- [ ] `Synthesizer` — takes raw results, returns structured summary (using small/medium model)
- [ ] Browser stub (returns "not implemented, falling back to Fetcher")
- [ ] Test: "NVIDIA stock" → finance data returned

**Code Agent:**
- [ ] `CodeAgent.execute()` — coordinates sub-agents
- [ ] `Planner` — takes task description, returns code spec (stub: just passes through)
- [ ] `Writer` — generates Python code via Bodega Inference
- [ ] `Executor` — creates venv, runs code, captures output
- [ ] `Debugger` stub (logs error, no auto-fix yet)
- [ ] `Validator` stub (checks exit code only)
- [ ] Test: "print hello world" → code generated, executed, output captured

**Memory Agent:**
- [ ] `MemoryAgent.execute()` — coordinates sub-agents
- [ ] `MemoryRouter` — simple heuristic: recent/session → Redis, structured → Postgres, semantic → pgVector
- [ ] `Writer` — stores chunks with metadata
- [ ] Janitor stub
- [ ] Test: write 5 chunks, read back by similarity

**SysStat Agent:**
- [ ] `SysStatAgent.execute()` — returns system metrics
- [ ] `Monitor` — psutil for RAM/CPU, token counter
- [ ] `ModelManager` — calls Bodega `/v1/admin/current-model`, `/v1/models`
- [ ] Scaler stub
- [ ] Test: returns current RAM usage + loaded model info

**P&L Agent:**
- [ ] `PnLAgent.execute()` — returns user profile
- [ ] `PreferenceManager` — CRUD on Postgres preferences table
- [ ] `FeedbackLearner` — records like/dislike/time-spent signals
- [ ] `Profile` — aggregates preferences into a profile dict
- [ ] Test: record a preference, retrieve updated profile

### Day 3: OSA Core + End-to-End Flow

**OSA:**
- [ ] `Orchestrator.run(query)` — the main loop
- [ ] `Decomposer` — takes query, uses big model to produce TaskDAG
- [ ] `Router` — deterministic mapping: TaskNode.agent → agent instance
- [ ] `Evaluator` stub — passes through (no quality gating yet)
- [ ] `PolicyEngine` — basic rules: max retries = 3, no destructive actions without confirmation
- [ ] `Guard` — input length check, basic regex for injection patterns
- [ ] Synapse events emitted at every state transition
- [ ] Shadows integration: tasks dispatched via `shadows.add()`

**End-to-End Test:**
- [ ] `octane ask "What is NVIDIA stock price?"` → OSA decomposes → Web Agent fetches → Synthesis → Output
- [ ] `octane ask "Write a Python script that prints fibonacci numbers"` → OSA → Code Agent → Output
- [ ] `octane health` → SysStat Agent reports system state
- [ ] Synapse trace viewable: `octane trace <correlation_id>`

**CLI:**
- [ ] `octane ask "<query>"` — auto-routed through OSA
- [ ] `octane health` — SysStat report
- [ ] `octane trace <id>` — view Synapse event log for a query
- [ ] `octane memory search "<query>"` — search memory tiers

### Definition of Done — Phase 1
- [ ] Every agent (Web, Code, Memory, SysStat, P&L) has a working coordinator + at least one functional sub-agent
- [ ] OSA can decompose a query, route to agents, collect results, and return output
- [ ] Synapse events trace the full lifecycle of a query
- [ ] Shadows processes at least one task per agent type
- [ ] CLI demonstrates 3 different query types end-to-end
- [ ] SysStat reports loaded models and resource usage

---

## Phase 2: Deepen Each Layer (Days 4-5)

### Philosophy
Now that the skeleton works end-to-end, add depth to each agent. Make sub-agents smarter, add the self-healing loops, make OSA actually reason.

**Web Agent Deepening:**
- [ ] Query Strategist generates 3+ variations with different search parameters
- [ ] Fetcher handles multiple Bodega APIs (search + finance + news) based on query type
- [ ] Browser Agent functional via Playwright for JS-heavy sites
- [ ] Synthesizer produces structured intelligence with citations/sources
- [ ] Error handling: if one search fails, continue with remaining results

**Code Agent Deepening:**
- [ ] Planner uses big model to create proper specifications
- [ ] Writer generates multi-file Python projects (not just single scripts)
- [ ] Debugger functional: analyzes stderr, produces fixes, feeds back to Executor
- [ ] Validator: runs output assertions, checks against expected behavior
- [ ] Self-healing loop: Writer → Executor → Debugger → Writer (max 3 retries)
- [ ] Code Agent can validate other agents' factual claims

**Memory Agent Deepening:**
- [ ] Router uses small model for intelligent tier selection (not just heuristics)
- [ ] Hot cache (Redis): session context with TTL, frequently accessed data
- [ ] Warm tier (Postgres): structured queries with JSONB metadata filtering
- [ ] Cold tier (pgVector): proper embedding generation + similarity search
- [ ] Writer evaluates what's worth persisting (not everything gets saved)

**SysStat Agent Deepening:**
- [ ] Model Manager implements loading strategies based on RAM:
  - 64GB: 30B (brain) + 14B (worker) + 6B (grunt)
  - 32GB: 14B (brain) + 6B (grunt)
  - 16GB: 8B (brain) + 0.9B (grunt)
- [ ] Monitor tracks per-model metrics: tok/s, queue depth, latency p50/p95
- [ ] Model Manager can hot-swap models via Bodega admin endpoints

**P&L Agent Deepening:**
- [ ] Feedback signals: explicit (thumbs up/down) + implicit (time spent, follow-up queries)
- [ ] Profile enrichment: domain interests, expertise level, preferred verbosity
- [ ] OSA consults P&L before every synthesis step

**OSA Deepening:**
- [ ] Decomposer produces real multi-step DAGs with parallel groups
- [ ] Router handles parallel dispatch via `asyncio.gather()` + Shadows
- [ ] Evaluator reviews assembled output, can request re-execution of failed steps
- [ ] Guard runs input AND output validation in parallel with main processing
- [ ] Synapse events include token counts, model usage, reasoning traces

### Definition of Done — Phase 2
- [ ] Web Agent handles finance, news, AND general search queries through same interface
- [ ] Code Agent self-heals: detects errors, fixes code, re-runs successfully
- [ ] Memory Agent routes to correct tier based on query semantics
- [ ] SysStat loads appropriate model topology for the hardware
- [ ] OSA decomposes complex multi-part queries into parallel DAGs
- [ ] P&L provides user context that visibly improves synthesis quality

---

## Phase 3: Intelligence & Polish (Days 6-7)

### Philosophy
Make the system genuinely intelligent. OSA reasons about routing. Memory provides context-aware augmentation. The system learns from usage.

**Intelligent Routing:**
- [ ] OSA.Decomposer uses Synapse history to improve decomposition
- [ ] Router learns from past successes: "finance queries work best with finance+news parallel"
- [ ] Evaluator implements quality scoring (relevance, completeness, accuracy)
- [ ] Failed evaluations trigger targeted re-execution (not full restart)

**Context-Aware Memory:**
- [ ] Every synthesis step checks Memory for relevant prior context
- [ ] Follow-up queries leverage hot cache: "What about AMD?" → knows we were discussing stocks
- [ ] Memory Writer auto-tags chunks with agent type, query domain, confidence score

**Interactive Mode:**
- [ ] `octane chat` — REPL with persistent session context
- [ ] Multi-turn memory: context accumulates across turns in Redis hot cache
- [ ] `octane trace` — rich formatted Synapse trace with timing visualization
- [ ] `octane sysstat` — live dashboard of model loading, RAM, throughput

**P&L Learning:**
- [ ] Preference model updates based on accumulated feedback
- [ ] Persona influences: synthesis tone, detail level, domain emphasis
- [ ] User can explicitly set preferences: `octane pref set verbosity concise`

**Workflow Templates:**
- [ ] Export Synapse trace as reusable workflow template (JSON)
- [ ] Import workflow template: `octane workflow run market-brief.json`
- [ ] Built-in templates: market-brief, research-deep-dive, code-validate

### Definition of Done — Phase 3
- [ ] Interactive chat maintains context across 10+ turns
- [ ] Memory augmentation demonstrably improves responses
- [ ] P&L personalization visible in output tone/content
- [ ] 3 workflow templates work end-to-end
- [ ] Synapse traces are human-readable and useful for debugging

---

## Phase 4: Hardening & Extensibility (Week 2+)

### Observability
- [ ] Structured logging with correlation IDs across all agents
- [ ] Synapse events persisted to Postgres for long-term analysis
- [ ] Metrics endpoint: `/metrics` for Prometheus-compatible scraping
- [ ] `octane dashboard` — terminal UI with live metrics (textual or rich)

### Error Recovery
- [ ] Agent-level retry with exponential backoff (via Shadows `ExponentialRetry`)
- [ ] Circuit breaker for external APIs (Bodega endpoints)
- [ ] Graceful degradation: if Web Agent fails, OSA attempts answer from Memory
- [ ] SandboxAgent timeout enforcement with process kill

### Shadows Full Integration
- [ ] All inter-agent communication via Shadows tasks (replace raw asyncio)
- [ ] Perpetual tasks for SysStat monitoring (`Perpetual(every=30s)`)
- [ ] Idempotent task scheduling for duplicate query prevention
- [ ] Task chaining with `CurrentShadow()` for follow-up scheduling

### VW Bandit Foundation
- [ ] Log every OSA routing decision + outcome as training data
- [ ] Format: `{context: query_features, action: route_chosen, reward: quality_score}`
- [ ] Offline VW training script from accumulated data
- [ ] Wire reward signal: user feedback + execution success + latency

### LoRA Personalization
- [ ] Export P&L + Memory data as JSONL for LoRA fine-tuning
- [ ] Script: `octane train persona` → runs mlx_lm.lora on accumulated data
- [ ] Load personalized adapters via Bodega `lora_paths` parameter

### Plugin Architecture
- [ ] Agent plugin interface: any Python class implementing BaseAgent can register
- [ ] Workflow marketplace format: JSON workflow templates with metadata
- [ ] `octane plugin install <name>` — installs community agents
- [ ] `octane workflow publish <template>` — exports shareable workflow


---

## ADDENDUM: HIL (Human-in-the-Loop) & Checkpointing System

*Added 2026-02-16. Paste this after the Phase 4 section in OCTANE_4_PHASE_PLAN_v2.md*

---

### New OSA Sub-components

Two new sub-components are added to OSA:

```
OSA Sub-agents (updated):
  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────┐ ┌──────────┐
  │ Decomposer │ │  Router  │ │ Evaluator │ │   Guard     │ │   HIL    │
  │(big model) │ │(determin)│ │(big model)│ │  (hybrid)   │ │ Manager  │
  └────────────┘ └──────────┘ └───────────┘ └─────────────┘ └──────────┘
  ┌──────────────┐ ┌────────────────────┐
  │Policy Engine │ │Checkpoint Manager  │
  │(deterministic│ │(snapshots + revert)│
  └──────────────┘ └────────────────────┘
```

**HIL Manager** — Renders decision reviews, collects user responses (approve/modify/decline), feeds results back to OSA loop.

**Checkpoint Manager** — Creates, stores, and restores pipeline state snapshots. Enables revert-to-checkpoint when user declines a decision.

---

### Three HIL Trigger Categories

| Category | When | Example | User Action Required |
|----------|------|---------|---------------------|
| **Blocked** | System literally cannot proceed | Captcha, login wall, missing credentials, ambiguous instruction | Must resolve to continue |
| **High-stakes** | Action is irreversible or consequential | Code execution, file deletion, external API writes, financial actions | Should review before proceeding |
| **Confidence checkpoint** | Accumulated uncertainty exceeds threshold | Multiple medium-confidence routing decisions, unfamiliar query pattern | Can review or skip |

---

### Decision Ledger

Every non-trivial OSA decision is logged as a `Decision`:

```python
class Decision(BaseModel):
    id: str
    correlation_id: str
    timestamp: datetime
    
    # What
    action: str                          # "Route to Web Agent for AMD comparison"
    reasoning: str                       # Why the agent made this choice
    
    # Risk assessment
    risk_level: str                      # "low" | "medium" | "high" | "critical"
    confidence: float                    # 0.0 to 1.0
    uncertainty_reason: str              # Why confidence < 1.0
    
    # Evidence
    sources: list[str] = []             # URLs, memory IDs, data references
    code_preview: str | None = None     # If Code Agent, show the code
    
    # Status
    status: str = "pending"             # pending | auto_approved | human_approved |
                                        # human_modified | human_declined
    human_feedback: str = ""            # Modification or decline reason
    
    # Linkage
    task_id: str                        # Which TaskNode in the DAG
    reversible: bool = True
```

---

### Auto-Approval Rules (Policy Engine)

**Auto-approve when:**
- Risk is LOW + confidence > 0.85
- Action is read-only (search, memory read, data retrieval)
- Same action type approved by user 5+ times previously (P&L learning)
- User preference: `octane pref set auto_approve_level medium`

**Escalate to HIL when:**
- Risk is HIGH or CRITICAL (regardless of confidence)
- Confidence < 0.60 (regardless of risk)
- 3+ accumulated MEDIUM-risk decisions without checkpoint
- Destructive action (delete, send, execute on external systems)
- Literally blocked (auth, captcha, missing data)
- First encounter with this decision type (no prior pattern)

**User controls:**
```bash
octane pref set hil_level relaxed     # Only HIGH+ interrupts
octane pref set hil_level balanced    # MEDIUM+ when confidence < 0.75 (default)
octane pref set hil_level strict      # Everything except LOW read-only

# Per-action overrides
octane pref set auto_approve web.search true
octane pref set auto_approve code.execute false
```

---

### Checkpoint System

**What a checkpoint captures:**
```python
class Checkpoint(BaseModel):
    id: str
    correlation_id: str
    timestamp: datetime
    
    # Pipeline state
    dag: TaskDAG
    completed_tasks: list[str]
    pending_tasks: list[str]
    
    # Results so far
    accumulated_results: dict[str, Any]    # task_id → AgentResponse
    
    # Decision state
    decisions: list[Decision]
    approved_decisions: list[str]
    
    # Context
    memory_context: dict[str, Any]
    pnl_profile: UserProfile
    synapse_events: list[SynapseEvent]
```

**When checkpoints are created:**
1. After decomposition (the "plan" checkpoint)
2. Before each high-risk task execution
3. After parallel group completion (data gathered, before synthesis)
4. Before final synthesis

**Revert flow:**
```
User declines Decision #4 at Checkpoint #2
    ↓
User provides reason: "qualitative only, skip quantitative"
    ↓
Checkpoint Manager restores state to Checkpoint #2
    ↓
OSA.Decomposer re-plans with:
  - Original query
  - All data from completed tasks (preserved, not re-fetched)
  - Declined decision + user's reason as new context
    ↓
Modified DAG executes only the changed steps
```

**Checkpoint-enabled features (Phase 3+):**
```bash
octane replay <correlation_id>                    # Re-run from first checkpoint
octane branch <correlation_id> --from-checkpoint 2  # Fork with new instructions
octane audit <correlation_id>                     # Full decision + checkpoint trail
```

---

### Phase Integration

| Phase | HIL Scope | Checkpoint Scope |
|-------|-----------|------------------|
| Phase 1 | BLOCKED only (captcha, auth, missing info). Simple `input()` prompts. | In-memory dict snapshots. Basic save/restore. |
| Phase 2 | Add risk/confidence HIL. Decision Ledger. Rich CLI panels. | Checkpoints persisted to Redis. Revert flow working. |
| Phase 3 | Auto-approval learning via P&L. User preference controls. | Replay + branch from checkpoints. |
| Phase 4 | Full audit trail in Postgres. VW bandits learn from approval patterns. | Checkpoint diff visualization. Workflow learning. |

---

### Updated File Structure (new files only)

```
octane/osa/
├── hil_manager.py           # HIL rendering, user input collection, response routing
├── checkpoint_manager.py    # Checkpoint creation, storage, restoration, revert
└── (existing: orchestrator.py, decomposer.py, router.py, evaluator.py, policy.py, guard.py, synapse.py)

octane/models/
├── decisions.py             # Decision, DecisionLedger models
├── checkpoints.py           # Checkpoint model
└── (existing: schemas.py, synapse.py, dag.py)
```

---

## Timeline Summary

| Day | Focus | Key Deliverable |
|-----|-------|-----------------|
| 1 | Scaffold + infrastructure + clients + Shadows | All files exist, clients work, Shadows runs |
| 2 | All 5 agents skeleton + sub-agent stubs | Every agent has basic execute(), one test each |
| 3 | OSA core + end-to-end flow + CLI | `octane ask` works for 3 query types, traces visible |
| 4 | Deepen Web + Code agents | Multi-source search, self-healing code execution |
| 5 | Deepen Memory + SysStat + P&L + OSA reasoning | Intelligent tier routing, model topology, personalization |
| 6 | Interactive mode + context memory + workflow templates | `octane chat` works with memory, 3 templates |
| 7 | Polish + demo prep + documentation | Clean demos, traces, README |

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| OSA becomes bottleneck | Decomposer/Evaluator use big model; Router/Policy/Guard are deterministic or small model |
| Model swapping latency | SysStat pre-loads model topology at startup; avoid mid-query swaps |
| Shadows complexity on Day 1 | Start with basic `shadows.add()`, add retries/perpetual in Phase 2 |
| Too many sub-agents to build | Every sub-agent starts as a stub. Depth comes in Phase 2. |
| pgVector setup complexity | Day 1: just asyncpg + JSONB. Day 2: add pgVector extension |
| Playwright browser agent | Stub in Phase 1. Functional in Phase 2 only for known JS-heavy targets |
