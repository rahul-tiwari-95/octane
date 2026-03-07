# OCTANE_PHASE_PLAN_v3.md
# Project Octane — Master Development Plan
# SRSWTI Research Labs
# Last updated: 2026-03-02

---

## What Octane Is

Octane figures out what to do, and you can see what it's doing so you can tweak the flow based on your preferences. The CLI makes it highly customizable yet powerful — and there are so many catalysts made by Octane and the open-source community that it gives you things to plug in and plug out depending on your needs.

Octane decides the widest breadth and deepest depth on its own. And the best part is you can tweak it as much as you want.

---

## Where We Are (Sessions 1–23)

### Phase 1: Foundation (Sessions 1–3)
Built the agent OS skeleton: 5 agents (Web, Code, Memory, SysStat, P&L), Bodega client, Synapse event bus, CLI with `octane ask/chat/health`. All agents stubbed with real API wiring for Web (Bodega Intel endpoints) and SysStat (psutil). First 35 tests.

### Phase 2: Intelligence (Sessions 4–9)
Activated the LLM brain. Decomposer routes queries to templates via Bodega. Guard checks input safety. Redis + Postgres memory with slot-based recall. P&L personalization with FeedbackLearner. Multi-step DAGs with parallel wave execution and upstream data injection. Code Agent self-healing (Planner → Writer → Executor → Validator → Debugger loop). SysStat topology recommendations. Artifact persistence.

### Phase 3: Web Intelligence + Developer Experience (Sessions 10–13)
Synthesizer for LLM-powered news/search synthesis. QueryStrategist for multi-variation query generation. Multi-turn chat with conversation history. Shadows perpetual tasks (stock monitoring). Workflow templates (export/run/list). Rich CLI: `octane dag` preview, `octane pref` management, `octane trace` visualization, colour-coded footers, partial trace ID resolution.

### Phase 4: Production Hardening (Sessions 14–19)
Catalysts (5 deterministic finance modules). Deep web intelligence (trafilatura + Playwright full-text extraction, cookie persistence). HIL with Decision Ledger, Checkpoint Manager, three approval levels. Normalized Postgres schema (14 tables). WebPageStore, ArtifactStore, FileIndexer, EmbeddingEngine with pgVector. Deep research mode (AngleGenerator, --depth flag, URL dedup, chunk embedding). Domain pipeline templates (investment, research, content, comparative). Three-tier test structure (unit/integration/e2e). ResearchSynthesizer for rolling reports. 9 critical async bugs fixed in Session 18C including two Python 3.13 architectural issues.

### Phase 5: Multi-Model Topology + Tier Routing (Sessions 20–21)
BodegaRouter with ModelTier enum (FAST/MID/REASON/EMBED). Tier-based routing at every LLM call site — QueryStrategist → FAST, Synthesizer → MID, Evaluator → REASON. Adaptive topology detection: compact (16GB M4 Air) / balanced (32-48GB) / power (64GB M1 Max). ModelConfig dataclass with `to_load_params()`. Two-pass schema execution (base tables always, pgVector when available). Full tier audit across all agents including Code sub-agents, AngleGenerator, ResearchSynthesizer, WebAgent ticker extraction. 475 → 496 tests.

### Phase 6: Deep Mode + Multi-Shot Refinement (Sessions 22–23)
`octane ask --deep` with multi-round deepening (3 extraction rounds, 13+ pages, convergence-based). Deep synthesis path: REASON tier, 3000 tokens, 10-article window, structured multi-section report output. MSR (Multi-Shot Refinement): MSRDecider evaluates Round-1 findings for ambiguity, auto-generates MCQ clarification questions, user answers steer follow-up query generation via DepthAnalyzer user_context injection. Verbose trace (`octane trace -v`) with per-URL tables, extraction details, MSR decisions. Live `--monitor` flag showing RAM/CPU during execution. Startup banner with full topology model map. 5 new Synapse event types for full web research reconstruction. 521 → 566 tests.

### Current State (Session 23 complete)
- **566 tests**, 0 failures
- **14 Postgres tables** with FKs and cascading deletes
- **5 finance catalysts** (deterministic, no LLM needed)
- **4 domain pipeline templates** (keyword-matched, no LLM round-trip)
- **BodegaRouter** with 3 model tiers, adaptive topology, speculative decoding
- **Full research pipeline**: AngleGenerator → parallel angles → per-angle OSA → findings → rolling synthesis
- **Deep mode**: `--deep` with MSR clarification, 3-round deepening, structured reports
- **Deep web extraction**: trafilatura + Playwright + cookie persistence + multi-round deepening
- **HIL**: Decision Ledger + Checkpoint Manager + 3 approval levels
- **Three-tier memory**: Redis hot → Postgres warm → pgVector embeddings
- **Shadows perpetual tasks**: stock monitoring + research cycles
- **Verbose trace**: per-URL, per-chunk, per-MSR-decision event reconstruction
- **CLI**: 15+ command groups with Rich formatting, startup topology banner

---

## Session 24: Octane Daemon

**Goal:** Centralized runtime that coordinates all CLI instances, manages the priority queue, and holds shared state.

### Why

Every CLI command currently creates its own Orchestrator, BodegaRouter, and connections. Three terminals = three independent instances competing for the same single-threaded 8B model with no coordination. No priority, no queuing, no shared context.

### What the Daemon Provides

**Priority Queue.** All CLI instances route through the daemon. Interactive `octane ask` (P0) gets served before background research cycles (P2). FAST tier requests go to the 90m model immediately while REASON tier requests queue by priority.

```
P0 (immediate):  octane ask, octane chat (user waiting)
P1 (soon):       MSR clarification, investigate dimensions
P2 (background): research cycles, monitor signals
P3 (batch):      embedding generation, chunk compression
```

**Shared Model State.** Single source of truth for which models are loaded, queue depth per model, RAM available, Bodega health. No per-command pre-flight checks.

**Shared Memory and Context.** Terminal 1's research enriches Terminal 2's queries automatically. Query result cache (Redis, 1h TTL). Active research findings available to all clients. Cross-session Synapse event bus.

**Connection Pooling.** One asyncpg pool, one Redis connection, one httpx client, one Playwright browser — shared across all requests. Cuts 200-500ms startup overhead per command.

**Background Task Coordination.** User query in progress → defer research cycle 60s. Two research tasks at same time → stagger. Embedding backlog → batch instead of one-at-a-time.

### Architecture

```
Terminal 1          Terminal 2          Terminal 3
octane ask          octane research     octane chat
    │                   │                   │
    └───────────────────┼───────────────────┘
                        │
                   Unix Socket
              ~/.octane/daemon.sock
                        │
                        ▼
        ┌───────────────────────────────────┐
        │         Octane Daemon             │
        │                                   │
        │  Priority Queue                   │
        │  Shared State (topology, cache)   │
        │  Connection Pools (PG, Redis, HTTP)│
        │  BodegaRouter (single instance)   │
        │  Synapse Event Bus (shared)       │
        │  Shadows Coordinator              │
        └───────────────────────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
          Bodega    Postgres    Redis
```

### CLI

```bash
octane daemon start       # start (auto-starts on first CLI command)
octane daemon status      # PID, uptime, queue depth, connections
octane daemon stop        # graceful shutdown
```

### Graceful Degradation

Daemon not running → CLI falls back to direct mode (current behavior). Nothing breaks without it. Everything gets better with it.

### Files

```
octane/daemon/
├── server.py       # async Unix socket server, request dispatcher
├── client.py       # thin async client for CLI commands
├── queue.py        # priority queue with tier-aware scheduling
├── state.py        # shared model state, query cache, sessions
└── lifecycle.py    # start/stop/auto-start/PID management
```

---

## Session 25: Power Commands — investigate, compare, chain

**Goal:** CLI commands that use the full agent stack at maximum depth and breadth.

### octane investigate

The "throw everything at it" command. Decomposes a query into 4-8 independent dimensions, researches each in parallel, cross-references findings, produces structured multi-section report.

```bash
octane investigate "Is NVDA overvalued at current levels?"
```

```
Query → DimensionPlanner (REASON) → 4-8 dimensions
    → Wave 1: parallel research per dimension (Web + Memory)
    → Wave 2: Code Agent cross-reference + catalysts
    → Wave 3: Evaluator structured report
    → Auto-store (pages, findings, embeddings)
```

**New component:** `DimensionPlanner` — identifies independent research dimensions from a query. Returns structured JSON: dimension labels, queries per dimension, priority ranking.

### octane compare

Structured multi-dimensional comparison. Parallel research for both sides, quantitative analysis, side-by-side report.

```bash
octane compare "NVDA vs AMD"
```

**New component:** `ComparisonPlanner` — identifies items and comparison dimensions. Each item × each dimension researched in parallel. Code Agent builds comparison matrix. Evaluator produces tradeoff analysis.

### octane chain

Explicit multi-step pipeline. eyeso piping before eyeso exists.

```bash
octane chain \
  "prices: fetch finance NVDA AAPL MSFT" \
  "tech: analyze technical {prices}" \
  "report: synthesize investment-brief {all}"
```

References: `{prev}`, `{step_name}`, `{all}`, `{{variable}}` for templates. `--save` converts chain to reusable workflow.

---

## Session 26: Power Commands — plan, monitor, replay

### octane plan

Goal → action plan with milestones.

```bash
octane plan "Build $100K portfolio in 5 years at $500/month"
```

GoalAnalyzer → feasibility research → Code Agent projections → structured plan with phases, checkboxes, risk scenarios. Offers to set up monitoring.

### octane monitor

Compound multi-signal surveillance via the daemon.

```bash
octane monitor "NVDA" --signals price,news,earnings,sentiment
octane monitor alerts
```

Creates compound Shadows perpetual tasks: price (1h), news (4h), earnings (daily), sentiment (6h). Cross-signal alerts when multiple signals align.

### octane replay

Re-run past analysis with fresh data.

```bash
octane replay <trace-id> --diff
```

Reads original trace DAG, re-runs with current data, highlights what changed vs original.

---

## Session 27: eyeso Language (v0.1)

The purest form of Octane. Composes CLI capabilities into workflows. Two execution modes: interpreted (REPL + scripts) and compiled (deployed via daemon as Shadows tasks).

```eyeso
tickers = ["NVDA", "AAPL", "MSFT"]

parallel for t in $tickers:
    prices.$t = fetch finance $t

for t in $tickers:
    if $prices.$t.change_pct < -3:
        alert "$t down ${prices.$t.change_pct}%"

every morning:
    portfolio = recall "my portfolio"
    drift = analyze portfolio-drift $portfolio $prices
    $drift → notify
```

v0.1: Variables, commands, pipes (→), parallel blocks, for-each, if/else, try/catch, `??` fallback, `every` scheduling. Full spec: EYESO_Programming_Language.md

---

## Session 28+: Future Horizon

**Multimodal Intelligence.** solomon-9b for image understanding. FileIndexer image description, chart verification, image-aware research.

**Community Catalyst Marketplace.** Share and install catalysts, templates, eyeso scripts. GitHub-based index.

**Career Autopilot + Health Intelligence.** Domain flows using existing infrastructure. Job monitoring, scoring, resume tailoring, workout programs, progress tracking.

**Morning Briefing.** 4 AM compound task → portfolio + research + career + news → synthesized update at terminal open.

---

## Session Map

| Session | Title | Tests |
|---------|-------|-------|
| 1-3 | Foundation: agents, CLI, Bodega client | 35 |
| 4-9 | Intelligence: LLM brain, memory, DAGs, Code Agent | 117 |
| 10-13 | Web intelligence, dev experience, Shadows | 101 |
| 14-16 | Catalysts, deep web, HIL | 194 |
| 17 | Long-running research workflows | 220 |
| 18A | Structured storage (14 Postgres tables) | 261 |
| 18B | Deep research mode (angles, depth, dedup) | 285 |
| 18C | Timeout cascade fix (9 bugs, Python 3.13) | 285 |
| 19 | Testing overhaul, domain pipelines, rolling synthesis | 357 |
| 20A-D | BodegaRouter, topology, ModelConfig, E2E tier wiring | 475 |
| 21 | Bug-fix sprint: schema resilience, tier audit (7 bugs) | 496 |
| 22 | Test migration + deep synthesis foundation | 521 |
| 23 | Deep synthesis quality + MSR + verbose trace | 566 |
| **24** | **Octane Daemon** | **600+** |
| **25** | **investigate, compare, chain** | **640+** |
| **26** | **plan, monitor, replay** | **680+** |
| **27** | **eyeso v0.1** | **710+** |
| 28+ | Multimodal, marketplace, career, health, briefing | 750+ |

---

## Architecture After Session 27

```
User
  │
  ├── eyeso scripts (.eyeso)
  ├── octane CLI (ask/chat/investigate/compare/chain/plan/monitor/replay/research)
  └── Community catalysts + templates
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Octane Daemon                              │
│                                                               │
│  Priority Queue (P0 interactive → P3 batch)                  │
│  Shared State (topology, query cache, sessions)              │
│  Connection Pools (Postgres, Redis, Bodega, Playwright)      │
│  Synapse Event Bus (cross-session trace visibility)          │
│  Shadows Coordinator (research, monitors, eyeso deploy)      │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Orchestrator                          │ │
│  │                                                          │ │
│  │  Guard → Decomposer/DimensionPlanner/ComparisonPlanner  │ │
│  │  → Domain Pipelines → DAGPlanner → Parallel Waves       │ │
│  │  → MSR Refinement → Deep Rounds → HIL                   │ │
│  │  → CrossReferencer → StructuredSynthesizer → Memory     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                         │                                     │
│           ┌─────────────┼─────────────┐                      │
│           ▼             ▼             ▼                       │
│     ┌──────────┐  ┌──────────┐  ┌──────────────┐            │
│     │Web Agent │  │Code Agent│  │Memory Agent  │            │
│     │          │  │          │  │              │            │
│     │Strategist│  │Planner   │  │Redis (hot)   │            │
│     │Fetcher   │  │Writer    │  │Postgres (warm)│            │
│     │Extractor │  │Executor  │  │pgVector (cold)│            │
│     │Browser   │  │Validator │  │              │            │
│     │Synthesizer│ │Debugger  │  │              │            │
│     │Depth     │  │Catalysts │  │              │            │
│     │MSRDecider│  │CrossRef  │  │              │            │
│     └──────────┘  └──────────┘  └──────────────┘            │
│           │             │             │                       │
│           ▼             ▼             ▼                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   BodegaRouter                           │ │
│  │                                                          │ │
│  │  FAST (90m)  │  MID (0.9b)  │  REASON (8b)             │ │
│  │  EMBED (MiniLM)  │  VISION (solomon-9b, future)        │ │
│  │                                                          │ │
│  │  Adaptive topology: compact / balanced / power           │ │
│  │  Speculative decoding on REASON tier                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
             Bodega    Postgres    Redis
```

---

## The Vision

Octane is an agent operating system that runs entirely on your machine. It figures out what to do when you give it a problem. It researches while you sleep. It monitors your portfolio, your job search, your research projects. It learns your preferences and adapts its behavior. Every piece of data it collects stays on your hardware.

The daemon is the beating heart — always running, always coordinating, always aware of what every subsystem is doing. The power commands let you throw real problems at it and get real intelligence back. eyeso lets you compose those capabilities into workflows that run autonomously.

That's what makes it an operating system, not an app.