<h3 align="center">Local AI Intelligence OS for Apple Silicon</h3>
<p align="center">
  Built by <a href="https://srswti.com">SRSWTI Research Labs</a><br/>
  <em>Engineering the World's Fastest Retrieval and Inference Algorithmic Systems</em>
</p>

---

## What is SRSWTI?

SRSWTI Research Labs builds infrastructure for private, high-performance AI on consumer hardware. Our mission: **engineering the world's fastest retrieval and inference algorithmic systems.**

We believe the future of AI is local. Not because cloud AI is bad — but because the most valuable data (your finances, your research, your decisions) should never leave your machine. The hardware exists. The models exist. What's missing is the intelligence layer that ties them together.

That's what we build.

**BodegaOS Sensors** is our inference runtime — a native macOS app that runs multiple MLX models simultaneously on Apple Silicon with speculative decoding, continuous batching, structured outputs, and a local HTTP server at `localhost:44468`.

**Octane** is our intelligence OS — a multi-agent research and analysis platform that sits on top of BodegaOS and turns your Mac into a private research terminal.

**Axe** is our coding CLI — agentic code generation and refactoring powered by local models.

Octane is the first product. BodegaOS Sensors is the engine underneath. Together, they're the foundation for everything SRSWTI builds next.

---

## What Octane Does

Octane is a local-first AI operating system for deep research, financial analysis, and knowledge accumulation. It runs entirely on your Apple Silicon Mac. Nothing touches a cloud.

### Deep Research
```bash
octane investigate "AI chip export restrictions" --deep 8 --cite
```
Decomposes your question into 8 independent research dimensions. Searches web, arXiv papers, and YouTube transcripts in parallel. Cross-references claims with trust scoring. Produces a cited report. Stores everything in your local knowledge base for future recall.

### Knowledge Accumulation
```bash
octane recall search "transformer attention"
octane stats
```
Every extraction, every finding, every web page is persisted with content-hash dedup. Your AI gets smarter over time because it remembers what you've already researched. 16 Postgres tables. Local markdown mirrors. Nothing is thrown away.

### Composable Pipes
```bash
octane search arxiv "RAG" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "latest RAG techniques"
```
Unix philosophy applied to AI. Each command reads stdin, writes stdout. Search → Extract → Synthesize. Chain them however you want.

### Mission Control
```bash
octane ui start
```
A real-time dashboard at `localhost:44480` with system vitals, loaded models, live research events, a globe visualization, and a full terminal with persistent sessions.

### Conversational AI with Intent Routing
```bash
octane chat
```
IntentGate classifies every message before the pipeline runs. "hi" gets an instant response. "analyze NVDA" triggers a full research pipeline. The system figures out what you need — you just talk.

---

## Install

### Requirements

- **Apple Silicon Mac** (M1 or later)
- **macOS Tahoe 16.x** (required by BodegaOS Sensors)

### Step 1 — Install BodegaOS Sensors (the inference engine)

BodegaOS Sensors is a native macOS app distributed as a `.dmg`. It provides the MLX inference engine that Octane depends on.

```bash
bash scripts/install_sensors.sh
```

This script will:
- Detect your RAM and download the correct edition (Standard ≤32 GB, Pro >32 GB)
- Save the `.dmg` to `~/Downloads`
- Guide you through installation step by step

After the `.dmg` is installed:
1. Open **BodegaOS Sensors** from Applications
2. Find the **Bodega Inference Engine** toggle and turn it **ON**
3. Wait until the toggle turns **green** — inference is now live at `localhost:44468`

### Step 2 — Install Octane

```bash
git clone https://github.com/srswti/octane
cd octane
bash setup.sh
```

`setup.sh` installs PostgreSQL, Redis, Python dependencies, runs database migrations, and verifies everything is running. Requires Homebrew.

### First Commands

```bash
source .venv/bin/activate
octane health                                      # verify all services are green
octane ask "hello"                                 # quick test
octane ask "explain transformers" --deep 4 --cite  # deep research
octane ui start                                    # Mission Control at localhost:44480
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         YOUR MAC                             │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Octane   │  │ BodegaOS │  │ Postgres │  │ Redis      │ │
│  │ Daemon   │←→│ Sensors  │  │ (16 tbl) │  │ (cache +   │ │
│  │          │  │ Inference│  │          │  │  Shadows)  │ │
│  │ 5 agents │  │ MLX/Metal│  │ knowledge│  │            │ │
│  │ OSA pipe │  │ multi-   │  │ findings │  │ perpetual  │ │
│  │ Shadows  │  │ model    │  │ extracts │  │ background │ │
│  └────┬─────┘  └──────────┘  └──────────┘  │ tasks      │ │
│       │                                     └────────────┘ │
│  ┌────▼─────┐  ┌──────────────┐                            │
│  │ CLI      │  │ Mission      │                            │
│  │ ~100 cmd │  │ Control UI   │                            │
│  └──────────┘  │ React+FastAPI│                            │
│                └──────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

**Octane Daemon** — always-on process managing agents, priority queue, model routing (FAST/MID/REASON tiers), and background Shadows tasks.

**BodegaOS Sensors** — native macOS app providing MLX inference at `localhost:44468`. Multi-model registry, speculative decoding, continuous batching, structured JSON outputs, prompt caching, RAG pipeline. Runs independently.

**OSA Pipeline** — Orchestrator → Strategist → Agents → Evaluator. Decomposes queries, routes to specialized agents (Web, Code, Memory, SysStat, P&L), synthesizes with trust-weighted evaluation.

**Shadows** — Redis-backed perpetual background tasks. Research cycles, price monitors, file watchers. Run while you sleep.

For full architecture details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## What's Stable, What's Beta

| Feature | Status | Notes |
|---------|--------|-------|
| `octane ask` / `octane chat` | ✅ Stable | IntentGate + ChatEngine, 1400+ tests |
| `octane investigate` / `compare` / `chain` | ✅ Stable | Power commands with --deep N, --cite, --verify |
| `octane search` / `extract` / `synthesize` | ✅ Stable | Composable pipes, JSON contracts |
| `octane recall` / `octane stats` | ✅ Stable | 5-table unified search |
| Mission Control UI | ✅ Stable | Dashboard, globe, terminal, WebSocket events |
| Extraction (YouTube, arXiv, PDF, EPUB) | ✅ Stable | Trust scoring, content-hash dedup |
| Security vault + air-gap | ✅ Stable | Touch ID via Swift helper, network kill-switch |
| Daemon + Shadows | ✅ Stable | Priority queue, background tasks |
| Portfolio (import, show, risk) | ⚠️ Beta | CSV import works, analysis needs live BodegaOS |
| Portfolio (dividends, tax lots, crypto) | ⚠️ Beta | Built, needs real-data testing |
| iMessage integration | 🧪 Experimental | Core pipeline works, needs BodegaOS + permissions |
| eyeso scripts | 📋 Planned | Example scripts exist, interpreter in development |

---

## License

Octane is source-available under the [Business Source License 1.1](LICENSE).

- **Free** for personal and non-commercial use
- **Source code is public** and auditable — "your data never leaves your machine" is verifiable, not a marketing promise
- **Commercial use** requires a license from SRSWTI
- **Converts to Apache 2.0** after 4 years

BodegaOS Sensors is proprietary software distributed by SRSWTI Research Labs.

---

## Roadmap: April → July 4th, 2026

| Milestone | Target |
|-----------|--------|
| Public launch, first community users | April 2026 |
| iMessage stable, Portfolio real-data, eyeso v0.1 | May 2026 |
| libp2p distributed inference, Mobile companion | June 2026 |
| eyeso v1.0, OctaneMesh alpha, Bodega 2.0 | July 4th 2026 |

**The July 4th vision:** Your Mac is sovereign. It runs its own models, stores its own knowledge, connects to a mesh of trusted peers for extra compute, and sends you intelligence via iMessage. No subscriptions. No cloud dependencies. No permission needed.

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5-minute install to first command |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture and data flow |
| [docs/BODEGA.md](docs/BODEGA.md) | BodegaOS Sensors — what it is, how to install |
| [docs/CLI_CHEATSHEET.md](docs/CLI_CHEATSHEET.md) | All ~100 commands with examples |
| [docs/USE_CASE_RESEARCH.md](docs/USE_CASE_RESEARCH.md) | Deep research workflows |
| [docs/USE_CASE_FINANCE.md](docs/USE_CASE_FINANCE.md) | Portfolio management and financial analysis |
| [docs/USE_CASE_COMPANION.md](docs/USE_CASE_COMPANION.md) | Always-on Mac AI companion |
| [docs/USE_CASE_OSINT.md](docs/USE_CASE_OSINT.md) | Open source intelligence research |
| [docs/USE_CASE_LOCAL_AI.md](docs/USE_CASE_LOCAL_AI.md) | For local AI power users |
| [docs/EYESO.md](docs/EYESO.md) | eyeso scripting language vision and examples |
| [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md) | What's MIT, what's BSL, and why |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Full April → July 4th roadmap |

---

## Stats

```
44 sessions · 1,416 tests · ~100 CLI commands
5 agents · 16 Postgres tables · 4 extraction pipelines
3 model tiers · Touch ID security · Air-gap mode
React Mission Control with globe visualization
Powered by BodegaOS Sensors · Apple Silicon only
```

---

<p align="center">
  <strong>SRSWTI Research Labs</strong><br/>
  <em>Engineering the World's Fastest Retrieval and Inference Algorithmic Systems</em><br/><br/>
  <a href="https://srswti.com">Website</a> ·
  <a href="https://github.com/srswti">GitHub</a> ·
  <a href="https://huggingface.co/srswti">Models</a>
</p>
