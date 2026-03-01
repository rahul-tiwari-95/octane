# OCTANE_PHASE_PLAN_v3.md
# Project Octane — Master Development Plan
# SRSWTI Research Labs
# Last updated: 2026-03-01

---

## What Octane Is

Octane figures out what to do, and you can see what it's doing so you can tweak the flow based on your preferences. The CLI makes it highly customizable yet powerful — and there are so many catalysts made by Octane and the open-source community that it gives you things to plug in and plug out depending on your needs.

Octane decides the widest breadth and deepest depth on its own. And the best part is you can tweak it as much as you want.

---

## Where We Are (Sessions 1–19)

### Phase 1: Foundation (Sessions 1–3)
Built the agent OS skeleton: 5 agents (Web, Code, Memory, SysStat, P&L), Bodega client, Synapse event bus, CLI with `octane ask/chat/health`. All agents stubbed with real API wiring for Web (Bodega Intel endpoints) and SysStat (psutil). First 35 tests.

### Phase 2: Intelligence (Sessions 4–9)
Activated the LLM brain. Decomposer routes queries to templates via Bodega. Guard checks input safety. Redis + Postgres memory with slot-based recall. P&L personalization with FeedbackLearner. Multi-step DAGs with parallel wave execution and upstream data injection. Code Agent self-healing (Planner → Writer → Executor → Validator → Debugger loop). SysStat topology recommendations. Artifact persistence.

### Phase 3: Web Intelligence + Developer Experience (Sessions 10–13)
Synthesizer for LLM-powered news/search synthesis. QueryStrategist for multi-variation query generation. Multi-turn chat with conversation history. Shadows perpetual tasks (stock monitoring). Workflow templates (export/run/list). Rich CLI: `octane dag` preview, `octane pref` management, `octane trace` visualization, colour-coded footers, partial trace ID resolution. 101 tests.

### Phase 4: Production Hardening (Sessions 14–19)
Catalysts (5 deterministic finance modules: price chart, return calculator, Monte Carlo projection, technical indicators, allocation pie). Deep web intelligence (trafilatura + Playwright full-text extraction, cookie persistence, browser agent). HIL with Decision Ledger, Checkpoint Manager, three approval levels. Normalized Postgres schema (10 tables with proper FKs). WebPageStore, ArtifactStore, FileIndexer, EmbeddingEngine with pgVector. Deep research mode (AngleGenerator, --depth flag, URL dedup, chunk embedding). Domain pipeline templates (investment, research, content, comparative). Three-tier test structure (unit/integration/e2e). ResearchSynthesizer for rolling reports.

**9 critical async bugs found and fixed in Session 18C** including two Python 3.13 architectural issues: asyncio.timeout() unreliable inside suspended generators, and asyncio.wait_for() blocking on httpx connection pool drain during cancellation cleanup.

### Current State
- **357 tests**, 0 failures
- **10 normalized Postgres tables** with FKs and cascading deletes
- **5 finance catalysts** (deterministic, no LLM needed)
- **4 domain pipeline templates** (keyword-matched, no LLM round-trip)
- **Full research pipeline**: AngleGenerator → parallel angles → per-angle OSA → findings in Postgres → rolling synthesis report
- **Deep web extraction**: trafilatura + Playwright + cookie persistence
- **HIL**: Decision Ledger + Checkpoint Manager + 3 approval levels
- **Three-tier memory**: Redis hot → Postgres warm → pgVector embeddings
- **Shadows perpetual tasks**: stock monitoring + research cycles
- **All Bodega call sites** have asyncio timeout guards with documented fallbacks
- **CLI**: 12+ command groups with Rich formatting

---

## Session 20: BodegaRouter + Bug Fixes + E2E Testing

### 20A: Fix What's Broken
- DB migration runner (schema.sql never applied to live Postgres)
- Zombie task cleanup (shadow.cancel() on stop)
- Quality gate (don't store junk when Bodega is down)
- Cross-angle URL dedup within same cycle
- Smart plain-text fallback (deduplicate, group by angle)
- Pre-cycle Bodega health check
- `octane research list` command
- Embedding pipeline live (unblocked by migration)

### 20B: BodegaRouter + Multi-Model Topology
- `ModelTier` enum: FAST (90m) / MID (0.9b) / REASON (8b) / EMBED
- `BodegaRouter` wraps `BodegaInferenceClient` with tier-based routing
- Adaptive topology: compact (16GB) / balanced (32-48GB) / power (64GB+)
- Speculative decoding: raptor-8b + Qwen3-0.6B draft model
- Structured JSON outputs via `response_format` (replaces regex/think-block parsing)
- All call sites updated: tier declared at each LLM call

### 20C: End-to-End Testing
- Integration tests through real Redis + Postgres
- E2E tests against live multi-model Bodega
- Manual verification checklist for M4 Air (compact) and M1 Max (power)
- Target: 400+ tests

---

## Sessions 21–25: The Next Horizon

### Session 21: Multimodal Intelligence

**Goal:** Octane can see.

Load `bodega-solomon-9b` (multimodal) alongside the text models. Wire vision capabilities into existing agents.

**What it enables:**
- FileIndexer image description: user uploads a photo or screenshot → solomon-9b describes it → text stored + embedded → semantically searchable
- Chart verification: Code Agent generates a chart → solomon-9b reads it → confirms data matches expectations
- Image-aware research: research cycle encounters infographics, diagrams, data visualizations → solomon-9b extracts information → findings include visual data
- `octane files add photo.png` → LLM description stored, not just OCR text

**Architecture:** BodegaRouter gets a `VISION` tier that routes to solomon-9b. Call sites that handle images use `tier=ModelTier.VISION`. Fallback: if vision model not loaded, skip image processing gracefully.

**RAM budget:** solomon-9b needs ~6GB. On M1 Max 64GB: fits alongside raptor-8b + 90m + 0.9b. On M4 Air 16GB: swap out 0.9b or load vision on-demand (unload after use).

---

### Session 22: Career Autopilot + Health Intelligence

**Goal:** Wire the second and fourth application flows from the everyday use doc.

**Career Autopilot:**
- `octane career setup` → stores role, salary, skills, preferences in `projects` table
- `career_monitor` Shadows perpetual task: scans job boards every 6h via Web Agent + Playwright
- Code Agent scoring catalyst: match_score based on role + stack + salary + location + industry
- `tracked_jobs` table populated with dedup (URL fingerprint)
- `octane career list` → shows matches ranked by score
- `octane career prep <job-id>` → company research + interview prep pipeline
- Resume tailoring: Memory retrieves base resume, Web Agent fetches JD, Evaluator rewrites sections

**Health Intelligence:**
- `octane health-profile setup` → stores stats, goals, constraints in `projects` table
- TDEE/macro calculator catalyst (deterministic, no LLM)
- Workout program generation via deep_research pipeline + domain-specific synthesis
- Weekly check-in: Memory stores prior data, Code Agent plots trends
- Progress chart catalyst (adapts price_chart for weight/lift tracking)

**Both flows use existing infrastructure** — no new agents needed. Just domain-specific catalysts, perpetual tasks, and project table entries.

---

### Session 23: eyeso Language (v0.1)

**Goal:** A simple scripting language that composes Octane CLI commands into powerful workflows.

**Design principles:**
- Simple to type, powerful to execute
- User focuses on the problem statement, not the plumbing
- Compiles to Octane CLI commands under the hood
- Human-readable, version-controllable, shareable

**Example:**
```eyeso
# morning-briefing.eyeso

research "NVDA earnings outlook" depth=deep every=6h
research "AI drug discovery" depth=exhaustive every=12h

watch NVDA AAPL MSFT

every morning:
  report all-research
  portfolio drift-check
  career new-matches
  briefing compile → email
```

**What it compiles to:**
```bash
octane research start "NVDA earnings outlook" --depth deep --every 6
octane research start "AI drug discovery" --depth exhaustive --every 12
octane watch start NVDA
octane watch start AAPL
octane watch start MSFT
# morning cron:
octane research report --all
octane portfolio check
octane career list --new
octane briefing generate --send
```

**Implementation:**
- `octane/eyeso/parser.py` — tokenizer + AST for eyeso syntax
- `octane/eyeso/compiler.py` — AST → Octane CLI commands
- `octane/eyeso/runtime.py` — executes compiled commands, handles scheduling
- `octane run morning-briefing.eyeso` — CLI entrypoint

**v0.1 scope:** Variable declarations, research/watch/portfolio commands, `every` scheduling blocks, `→` pipe operator for chaining outputs. No conditionals or loops yet.

---

### Session 24: Community Catalyst Marketplace

**Goal:** Users share and install catalysts, workflow templates, and eyeso scripts.

**What's shareable:**
- Catalysts: deterministic Python functions (finance, research, career, health, content)
- Workflow templates: JSON DAG definitions with {{variable}} placeholders
- eyeso scripts: .eyeso files for complex workflows
- Domain pipeline templates: keyword triggers + node definitions

**CLI:**
```bash
octane marketplace search "earnings analysis"
octane marketplace install srswti/earnings-deep-dive
octane marketplace publish my-catalyst.py --public

# Installed catalysts auto-register in CatalystRegistry
# Installed templates appear in octane workflow list
# Installed eyeso scripts runnable via octane run
```

**Implementation:**
- `~/.octane/community/` — local catalog of installed packages
- `octane/marketplace/registry.py` — search, install, publish
- GitHub-based package index (JSON manifest per package)
- Version pinning, dependency declaration, hash verification
- CatalystRegistry auto-discovers installed catalysts on startup

---

### Session 25: Morning Briefing + Compound Perpetual Workflows

**Goal:** Octane works while you sleep and greets you with intelligence.

**Morning briefing flow:**
```
04:00 AM — Shadows triggers compound_briefing task
  │
  ├── Portfolio check: fetch prices, compute drift, flag alerts
  ├── Research updates: check all active research projects for new findings
  ├── Career matches: scan for new job postings matching profile
  ├── News digest: top stories in user's domain interests
  │
  └── Compile into briefing:
      │
      ├── Store in Redis: briefing:latest (HASH)
      └── Generate: briefing.md in ~/octane_output/

07:00 AM — User opens terminal:

  $ octane

  ☀️ Good morning, Rahul. Here's what happened overnight:

  📊 Portfolio: $12,847 (+0.3%) — no drift alerts
  🔬 Research "AI drug discovery": 4 new findings, 1 confirmed multi-source
  💼 Career: 2 new matches (Vercel Principal Eng 92/100, Anthropic EM 85/100)
  📰 News: NVIDIA announces new AI chip, OpenAI raises $10B round

  Run 'octane briefing detail' for full report.
```

**This is the moment Octane stops feeling like a tool and starts feeling like an assistant that was working while you slept.**

---

## Session Map

| Session | Title | Tests After |
|---------|-------|-------------|
| 1-3 | Foundation: agents, CLI, Bodega client | 35 |
| 4-9 | Intelligence: LLM brain, memory, DAGs, Code Agent | 117 |
| 10-13 | Web intelligence, dev experience, Shadows | 101 |
| 14-16 | Catalysts, deep web, HIL | 194 |
| 17 | Long-running research workflows | 220 |
| 18A | Structured storage (10 Postgres tables) | 261 |
| 18B | Deep research mode (angles, depth, dedup) | 285 |
| 18C | Timeout cascade fix (9 bugs, Python 3.13 issues) | 285 |
| 19 | Testing overhaul, domain pipelines, rolling synthesis | 357 |
| **20A** | **Bug fixes, migration runner, quality gate** | **367** |
| **20B** | **BodegaRouter, multi-model topology, structured outputs** | **387** |
| **20C** | **E2E testing on M4 Air + M1 Max** | **400+** |
| 21 | Multimodal (solomon-9b vision) | 420+ |
| 22 | Career Autopilot + Health Intelligence | 450+ |
| 23 | eyeso language v0.1 | 470+ |
| 24 | Community catalyst marketplace | 490+ |
| 25 | Morning briefing + compound perpetual workflows | 510+ |

---

## Architecture After Session 25

```
User
  │
  ├── eyeso scripts (.eyeso)
  ├── octane CLI (ask/chat/research/career/health/portfolio/briefing)
  └── Community catalysts + templates
        │
        ▼
┌─────────────────────────────────────────────┐
│              Octane OSA                       │
│                                               │
│  Guard → Decomposer → Domain Pipelines →     │
│  DAGPlanner → Parallel Wave Dispatch →       │
│  HIL + Checkpoints → Evaluator → Memory      │
└──────────────────┬────────────────────────────┘
                   │
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
   ┌─────────┐ ┌──────────┐ ┌──────────────┐
   │Web Agent│ │Code Agent│ │Memory Agent  │
   │         │ │          │ │              │
   │Strategist│ │Planner  │ │Redis (hot)   │
   │Fetcher   │ │Writer   │ │Postgres (warm)│
   │Extractor │ │Executor │ │pgVector (cold)│
   │Browser   │ │Validator│ │              │
   │Synthesizer│ │Debugger│ │              │
   │          │ │Catalysts│ │              │
   └─────────┘ └──────────┘ └──────────────┘
        │          │              │
        ▼          ▼              ▼
┌─────────────────────────────────────────────┐
│            BodegaRouter                       │
│                                               │
│  FAST (90m)  │  MID (0.9b)  │  REASON (8b)  │
│  VISION (9b) │  EMBED (MiniLM)              │
│                                               │
│  Adaptive topology: compact / balanced / power│
│  Speculative decoding on REASON tier          │
│  Structured JSON outputs via response_format  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           Bodega Inference Engine             │
│  Multi-model registry on localhost:44468     │
│  Apple Silicon MLX · process-isolated        │
└─────────────────────────────────────────────┘
```

---

## The Vision

Octane is an agent operating system that runs entirely on your machine. It figures out what to do when you give it a problem. It researches while you sleep. It monitors your portfolio, your job search, your research projects. It learns your preferences and adapts its behavior. Every piece of data it collects stays on your hardware — your financial data, your health metrics, your career information never leave your machine.

And when you want to customize it, you can. Swap models, install catalysts, write eyeso scripts, tweak pipeline templates, adjust preferences. The CLI gives you full visibility and control. The community gives you leverage.

That's what makes it an operating system, not an app.