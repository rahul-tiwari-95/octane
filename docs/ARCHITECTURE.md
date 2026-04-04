# Octane Architecture

How Octane works internally — from CLI command to AI response.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           YOUR MAC                               │
│                                                                 │
│  ┌──────────┐  ┌────────────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Octane   │  │ BodegaOS       │  │ Postgres │  │ Redis    │ │
│  │ Daemon   │←→│ Sensors        │  │ 16 tables│  │ Shadows  │ │
│  │          │  │ localhost:44468│  │ knowledge│  │ cache    │ │
│  │ 5 agents │  │ MLX/Metal      │  │ findings │  │ queues   │ │
│  │ OSA pipe │  │ multi-model    │  │ extracts │  │          │ │
│  │ Shadows  │  │ spec. decoding │  │ memory   │  │          │ │
│  └────┬─────┘  └────────────────┘  └──────────┘  └──────────┘ │
│       │                                                         │
│  ┌────▼──────────┐  ┌──────────────────────────────────────┐   │
│  │ CLI (~100 cmd)│  │ Mission Control UI                   │   │
│  │ typer+rich    │  │ React 18 + FastAPI + WebSocket       │   │
│  └───────────────┘  │ localhost:44480                      │   │
│                     └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

Every `octane ask` or `octane investigate` command follows this path:

```
CLI → ChatEngine / IntentGate
         │
         ▼
     IntentGate (classify intent)
         │
    ┌────┴──────────────┐
    │ GREETING/SIMPLE   │  → instant response from BodegaOS FAST tier
    │ RESEARCH/DEEP     │  → OSA Pipeline
    │ FINANCIAL         │  → P&L Agent + Web Agent
    └───────────────────┘
         │
         ▼ (research path)
     Orchestrator
         │
         ▼
     Strategist → DimensionPlanner
     (decomposes query into N parallel research angles)
         │
         ▼
     Agent Pool (parallel execution)
     ├── WebAgent    (search, extract, trust-score)
     ├── MemoryAgent (recall prior research)
     ├── CodeAgent   (when computation needed)
     ├── SysStatAgent (when system info needed)
     └── PnLAgent    (when financial context needed)
         │
         ▼
     Evaluator
     (dedup by content-hash, trust-weight, rank)
         │
         ▼
     Synthesizer → BodegaOS REASON tier
     (produce cited report)
         │
         ▼
     Memory persistence (Postgres + local markdown)
         │
         ▼
     Response to CLI / UI
```

---

## Components

### CLI (`octane/cli/`)
~100 commands implemented with `typer` and `rich`. Each command is a standalone module. Commands communicate with the daemon via Unix socket or HTTP, or run standalone for stateless operations.

### OSA Pipeline (`octane/osa/`)
**Orchestrator → Strategist → Agents → Evaluator** — the brain of Octane.

- **Orchestrator** — top-level coordinator, manages agent lifecycle
- **Strategist** — query decomposition and dimension planning
- **Evaluator** — trust-weighted deduplication and result ranking

### Agents (`octane/agents/`)
Five specialized agents, each with its own tool set:

| Agent | Responsibility |
|-------|---------------|
| **WebAgent** | Web search, extraction, YouTube, arXiv, PDF |
| **MemoryAgent** | Postgres recall, semantic search, knowledge base |
| **CodeAgent** | Code generation, execution, sandboxed Python |
| **SysStatAgent** | System vitals, model management, BodegaOS health |
| **PnLAgent** | Portfolio data, financial analysis, user P&L |

### BodegaOS Sensors — Model Tiers

Octane routes requests to BodegaOS across three model tiers:

| Tier | Use | Latency |
|------|-----|---------|
| `FAST` | Greetings, simple classification, quick answers | <1s |
| `MID` | Extraction, summarization, structured outputs | 2-5s |
| `REASON` | Deep research synthesis, multi-step reasoning, citations | 10-30s |

The **BodegaRouter** selects the tier based on task complexity and available models.

### Daemon (`octane/daemon/`)
An always-on background process managing:
- Priority task queue (CRITICAL → HIGH → NORMAL → LOW)
- Agent pool and lifecycle
- Shadows orchestration (perpetual background tasks)
- WebSocket event broadcasting to Mission Control UI

### Shadows (`octane/daemon/shadows/`)
Redis-backed perpetual background tasks. Unlike one-shot daemon tasks, Shadows run continuously:
- `ResearchCycle` — periodically re-researches watched topics
- `PriceMonitor` — watches portfolio tickers
- `FileWatcher` — monitors `~/Octane/inbox/` for new documents

### Database (PostgreSQL — 16 tables)
Key tables:

| Table | Contents |
|-------|----------|
| `research_tasks` | Task metadata, status, dimensions |
| `research_findings` | Individual findings with trust scores |
| `extractions` | Web page and document extractions (content-hash deduped) |
| `memory_nodes` | Long-term semantic memory |
| `portfolio_holdings` | Portfolio positions |
| `audit_log` | Command audit trail |

### Security (`octane/security/`)
- **Vault** — Touch ID authenticated secret storage via Swift helper (`octane-auth`)
- **Air-gap** — network kill-switch that blocks all external traffic
- **Audit log** — append-only Postgres log of all CLI commands

### Mission Control UI (`octane/ui-frontend/`)
React 18 + Vite frontend served by FastAPI at `localhost:44480`:
- System vitals panel (CPU, RAM, GPU, model load)
- Live research event stream via WebSocket
- Globe visualization (research activity map)
- Integrated terminal with persistent sessions
- Portfolio dashboard with 9 chart components

---

## Data Flow — Extraction Pipeline

```
URL or search result
       │
       ▼
  Playwright (headless Chromium) → raw HTML
       │
       ▼
  Trafilatura → clean text
       │
       ▼
  content-hash dedup check (Postgres)
       │ (new content only)
       ▼
  BodegaOS MID tier → structured extraction
  (title, summary, key claims, entities)
       │
       ▼
  Trust scoring (source reputation × claim consistency)
       │
       ▼
  Postgres storage + local markdown mirror (~/.octane/extractions/)
```

---

## File Structure

```
octane/
├── agents/          # 5 specialized agents
├── catalysts/       # Finance data connectors
├── cli/             # ~100 CLI commands (one file per command)
├── config.py        # Pydantic settings (reads from .env)
├── daemon/          # Always-on process + Shadows
├── models/          # Pydantic data models
├── osa/             # Orchestrator, Strategist, Evaluator
├── portfolio/       # Portfolio analysis
├── research/        # DimensionPlanner, Synthesizer, angles
├── security/        # Vault, air-gap, audit
├── tasks/           # Background task definitions
├── tools/           # BodegaOS clients, web tools
├── ui/              # FastAPI backend for Mission Control
├── ui-frontend/     # React 18 Mission Control frontend
├── utils/           # Shared utilities
└── workflow/        # DAG workflow runner
```
