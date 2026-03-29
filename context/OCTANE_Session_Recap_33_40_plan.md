# Octane Sessions 33–40: The Niche Domination Plan (v2)
# Apple Silicon Only. No Ollama. No Compromises.
# SRSWTI Research Labs | 2026-03-26

---

## By Session 40, Octane Is:

1. **A private Bloomberg alternative** — portfolio, dividends, tax lots, earnings, risk, monitoring, crypto
2. **A self-hosted Perplexity alternative** — citation-based deep research with web + arXiv + YouTube + PDF
3. **An OSINT research platform** — entity extraction, relationship mapping, multi-source intelligence
4. **A macOS-native AI companion** — iMessage interface, Apple Notes integration, folder watching, always-on
5. **An OpenClaw skill provider** — Octane intelligence accessible from WhatsApp/Telegram/Slack
6. **Has a React UI** — Finance Dashboard + Research Command Center + Entity Graph
7. **Runs entirely on Apple Silicon** with Touch ID, air-gap mode, and Bodega inference

---

## Design Decisions (Locked)

### No Ollama. Bodega Only.

Octane runs exclusively on Bodega. No Ollama fallback. No Linux support. No generic OpenAI-compatible backend switching.

**Why:** Octane is for people who can afford $3K+ spread over 3 years on Apple Silicon hardware. That's the same person who buys a Bloomberg terminal. They're not price-sensitive — they're quality-sensitive and privacy-sensitive. Supporting Ollama means supporting Linux, which means supporting NVIDIA, which means a different runtime, different memory model, different performance characteristics, different bugs. That's building two products. We build one product, and it's the best product for Apple Silicon.

Bodega gives us capabilities no Ollama setup can match: multi-model registry with per-model subprocess isolation, speculative decoding, continuous batching at 900 tok/s, structured JSON outputs via outlines, native MLX prompt caching, Bodega RAG, vision via solomon-9b, image generation via solomon/keshav/kalamkari — all through one daemon on one port. That's the infrastructure advantage. We don't dilute it.

### Apple Silicon is the Platform

M1/M2/M3/M4/M5 Macs. Unified memory. Metal GPU. Touch ID via Secure Enclave. AppleScript for system automation. iMessage for messaging. Apple Notes for note-taking. Folder Actions for file watching. These are not constraints — they're capabilities that no other platform has in combination.

The M5's Neural Accelerators deliver 4x TTFT speedup over M4 for compute-bound inference. `mx.distributed` enables pipeline-parallel inference across multiple Macs via Ring (Ethernet) or JACCL (Thunderbolt 5 RDMA with order-of-magnitude lower latency). This is where Bodega's future goes — and Octane rides it.

---

## The macOS-Native AI Companion (The Big Idea)

### The Core Problem

Every AI tool today lives inside a browser tab or a terminal window. You go to the AI. The AI doesn't come to you.

Octane should be the opposite. Octane lives on your Mac. It reads your folders. It watches your files change. It sends you iMessages when something matters. It writes to your Apple Notes. It runs while you sleep. When you wake up, there's an iMessage from Octane: "NVDA dropped 4% overnight. RSI hit 28. Here's what changed since your last analysis."

That's not a tool. That's a colleague.

### What macOS Gives Us (via AppleScript + System APIs)

**iMessage (send and receive):**
AppleScript can send iMessages programmatically. The `imsg` CLI tool (open source, built on AppleScript + read-only Messages.db access) enables: sending messages, reading conversation history, and streaming incoming messages in real-time with ~500ms detection latency. No private APIs. Requires Full Disk Access + Automation permissions for Messages.app.

```applescript
tell application "Messages"
    send "NVDA Alert: -4.16%, RSI 28 (oversold)" to buddy "+1234567890" of service id "iMessage"
end tell
```

**This means:** Octane's Shadows monitor detects a cross-signal alert → calls AppleScript → you get an iMessage on your iPhone. You reply "investigate NVDA" → Octane reads the reply → runs the investigation → sends the report back via iMessage. Your phone becomes the Octane interface. No app needed.

**Apple Notes (read and write):**
AppleScript can create and append to Notes:

```applescript
tell application "Notes"
    make new note at folder "Octane" with properties {name:"NVDA Analysis", body:"<html>report here</html>"}
end tell
```

**This means:** `octane investigate "topic" --export notes` creates an Apple Note with the full report. It syncs to your iPhone, iPad, and iCloud automatically. Your research library lives in Notes alongside your own notes.

**Folder Watching (FSEvents):**
Python's `watchdog` library monitors a folder for changes. When a file is added to `~/Octane/inbox/`, the daemon picks it up, extracts it, and stores it.

```bash
# Drop a PDF into the inbox folder
cp ~/Downloads/annual_report.pdf ~/Octane/inbox/
# Octane auto-extracts, chunks, indexes it
# Available immediately for octane ask --file or octane recall
```

**Calendar (read events):**
AppleScript reads calendar events. Shadows can prep you for meetings: "Board meeting in 1h. Key topics from your research: [summary]."

### The Shadows Expansion

Shadows is currently used for background price monitoring and research cycles. With macOS integration, Shadows becomes the **nervous system of your Mac**:

| Shadow Type | Trigger | Action | Output |
|---|---|---|---|
| **Price Monitor** | Every N minutes | Check prices for watchlist | iMessage alert if threshold crossed |
| **News Monitor** | Every N hours | Search web for watched topics | iMessage summary of changes |
| **Research Cycle** | Every N hours (--watch) | Re-run investigation, compute delta | iMessage + Apple Note with changes |
| **Folder Watcher** | File added to ~/Octane/inbox/ | Extract + chunk + index | Available in recall and Q&A |
| **Earnings Prep** | Day before earnings date | Run earnings investigation | iMessage + Note: "NVDA reports tomorrow" |
| **Calendar Prep** | 1 hour before meeting | Read agenda, prep relevant research | iMessage: meeting prep summary |
| **Portfolio Snapshot** | Daily at market close | Store portfolio state | Net worth timeline updated |
| **Morning Briefing** | 7am daily | Aggregate overnight changes | iMessage: personalized morning briefing |

The Morning Briefing is the killer Shadow. Every day at 7am, an iMessage:

```
Good morning, Rahul.

Portfolio: $847,231 (+$2,341 overnight, +0.28%)
  🔴 NVDA: -2.1% after-hours on export restriction news
  🟢 VOO: +0.4%, S&P holding steady

Research Updates:
  "AI chip export policy" — new developments (checked 6h ago)

Earnings This Week:
  NVDA reports Thursday after close. Prep ready.

Calendar:
  Board meeting at 2pm. 3 agenda topics.
```

Reply "investigate NVDA export restrictions" from your iPhone. The daemon runs it. Result comes back as an iMessage.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     YOUR MAC                              │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Octane Daemon│←→│ Shadows       │←→│ AppleScript   │   │
│  │ (always-on)  │  │ (perpetual    │  │ Bridge        │   │
│  │              │  │  background   │  │               │   │
│  │ - Bodega     │  │  tasks)       │  │ - iMessage    │   │
│  │ - Postgres   │  │              │  │ - Notes       │   │
│  │ - Redis      │  │ - monitors   │  │ - Calendar    │   │
│  │ - Extractors │  │ - watchers   │  │ - Folders     │   │
│  │ - Evaluator  │  │ - crons      │  │               │   │
│  └──────┬───────┘  └──────────────┘  └───────────────┘   │
│         │                                                 │
│  ┌──────▼───────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ CLI (terminal)│  │ React UI     │  │ ~/Octane/     │   │
│  │              │  │ (localhost)   │  │ inbox/ (drop) │   │
│  └──────────────┘  └──────────────┘  └───────────────┘   │
│                                                          │
└──────────────────────────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Your iPhone          │
                    │ (iMessage replies)   │
                    │ (Apple Notes sync)   │
                    └─────────────────────┘
```

---

## Session-by-Session Plan

### Session 32: Templates + Export + Recall

**Deliverables:** Output templates (TOML), PDF/DOCX export, `octane recall`, `octane status`
**Tests:** `tests/unit/test_session32_templates_export.py` — 30 tests | **Total: 1,046**

### Session 33: React UI Foundation

**Deliverables:** FastAPI backend + React frontend at localhost:44480. Finance + Research views.
**Tests:** `tests/unit/test_session33_ui_backend.py` — 20 tests | **Total: 1,066**

### Session 34: macOS Native — iMessage + Folder Watching + Notes

**Deliverables:** AppleScript bridge, iMessage send/receive, folder watcher Shadow, Notes export, `octane macos setup/status`
**Files:** `octane/macos/applescript.py`, `folder_watcher.py`, `imessage_handler.py`, `permissions.py`
**Tests:** `tests/unit/test_session34_macos.py` — 25 tests | **Total: 1,091**

### Session 35: Advanced Finance — Dividends, Tax Lots, Net Worth, Crypto

**Deliverables:** Dividend tracker, tax lot FIFO/LIFO, tax-loss harvesting, net worth timeline, XIRR/Sharpe, crypto parsers (Coinbase/Kraken/Binance/Gemini), CoinGecko prices
**Tests:** `tests/unit/test_session35_advanced_finance.py` — 35 tests | **Total: 1,126**

### Session 36: Entity Extraction + Relationship Mapping (OSINT)

**Deliverables:** NER catalyst via Bodega structured JSON, entity store, relationship mapping, `--entity` mode, `octane entities show/graph`
**Tests:** `tests/unit/test_session36_entities.py` — 25 tests | **Total: 1,151**

### Session 37: React UI Polish — Charts + Entity Graph + Shadows Dashboard

**Deliverables:** D3 entity graph, portfolio charts (net worth, allocation, dividends, benchmark), investigation timeline, Shadows management panel
**Tests:** `tests/unit/test_session37_ui_routes.py` — 15 tests | **Total: 1,166**

### Session 38: OpenClaw Skill + Morning Briefing + --watch + Compound Monitor

**Deliverables:** OpenClaw bridge script + SKILL.md, Morning Briefing Shadow, `--watch:Nh` on all commands, compound `octane monitor` with cross-signal alerts
**Tests:** `tests/unit/test_session38_openclaw_monitor.py` — 25 tests | **Total: 1,191**

### Session 39: plan + replay + Chain Upgrades

**Deliverables:** `octane plan` (GoalAnalyzer), `octane replay --diff`, parallel/conditional chain steps, Calendar Prep Shadow
**Tests:** `tests/unit/test_session39_plan_replay.py` — 25 tests | **Total: 1,216**

### Session 40: Launch Polish + Documentation + Demo

**Deliverables:** `setup.sh` hardened, `QUICKSTART.md`, `DEMO.md`, Reddit posts drafted, demo portfolio CSV, final integration test
**Tests:** `tests/unit/test_session40_polish.py` — 15 tests | **Total: 1,231**

---

## Bodega's Future — Sessions 40+ (The Open Source Play)

### What MLX Gives Us

**M5 Neural Accelerators (macOS 26.2+):** 4x TTFT speedup. Bodega users on M5 get this automatically.

**`mx.distributed` Ring (Ethernet):** Pipeline-parallel inference across multiple Macs on any network. Split a 120B model across 4 Mac Minis.

**`mx.distributed` JACCL (Thunderbolt 5 RDMA, macOS 26.2+):** Order-of-magnitude lower latency than Ring. Connect Macs via Thunderbolt cable → they become one inference cluster. Apple's WWDC 2025 featured this.

**`mx.distributed` + NCCL (CUDA):** MLX now ships with NCCL support. Bodega could serve as a unified inference layer across Apple Silicon AND NVIDIA GPUs on the same network.

### Creative Flows Enabled

**Visual Research:** Load raptor-8b (text) + solomon-9b (vision) simultaneously. Investigate a topic using both text articles and image analysis. The report includes visual evidence assessment.

**Real-Time Batch Processing:** Drop 50 PDFs into ~/Octane/inbox/. Bodega's continuous batching at 900 tok/s processes all 50 simultaneously. Minutes, not hours.

**Multi-Mac War Room:** 2 Mac Minis via Thunderbolt. Mac A runs 32B reasoning. Mac B runs 8B fast + 9B vision. Bodega routes automatically based on query complexity.

**LoRA Specialists:** Fine-tune a LoRA adapter on your investment thesis using `mlx-lm`. Load into Bodega with `lora_paths`. The model becomes yours — your domain, your methodology, your style.

---

## Open Source Boundary (Locked Until Session 40)

### Open Source (MIT) — Ships Day One

| Component | PyPI Package | Purpose |
|---|---|---|
| Bodega Inference Engine | `bodega-mlx-engine` | Top of funnel. Every Mac running Bodega → future Octane user |
| eyeso Language Spec | (spec only) | Community writes scripts that run on Octane |
| Catalyst API Interface | `octane-catalysts` | Community writes catalysts |
| Extraction Data Models | `octane-extractors` | ExtractedDocument, TextChunk, SourceType |
| OpenClaw Skill | (ClawHub listing) | Any OpenClaw user can route to Octane |

### Proprietary (BSL)

| Component | Distribution |
|---|---|
| Octane CLI + Daemon | BSL license (personal use free, commercial requires license) |
| Power Commands (investigate, compare, chain, plan, monitor, replay) | Part of Octane |
| BodegaRouter + OSA Pipeline | Part of Octane |
| React UI | Part of Octane |
| Shadows Engine | Part of Octane |
| macOS Integration (iMessage, Notes, folders) | Part of Octane |
| Security Vault | Part of Octane |

### Code Structure for Clean Separation

```
srswti/
├── bodega-mlx-engine/     # MIT — separate repo, separate PyPI
├── octane-catalysts/       # MIT — interface + community catalysts
├── octane-extractors/      # MIT — data models only
├── octane/                 # BSL — the product
│   ├── osa/               # Proprietary orchestration
│   ├── cli/               # Proprietary CLI
│   ├── daemon/            # Proprietary daemon + Shadows
│   ├── macos/             # Proprietary macOS integration
│   ├── security/          # Proprietary vault
│   ├── ui/                # Proprietary React UI
│   ├── portfolio/         # Proprietary finance
│   └── templates/         # Proprietary templates
└── octane-mesh/            # MIT (future) — P2P wire protocol
```

**The insight:** Bodega is the engine. Octane is the car. Open-source the engine. Sell the car.

---

## Revenue Model

| Tier | Price | What |
|---|---|---|
| **Free** | $0 forever | Full CLI, all commands, 5 templates, Bodega, iMessage, local-only |
| **Pro** | $29/month | OctaneMesh, premium templates, premium catalysts, export, priority |
| **Enterprise** | Custom | On-prem, LoRA fine-tuning, dedicated compute, compliance, air-gap cert |
| **Mesh Marketplace** | 15-20% | Contribute compute → earn. Consume compute → spend. |

---

## The Launch

```
Octane v1.0 — Session 40

35,000+ lines of Python · 1,231+ tests · 40 sessions
5 agents · 13+ catalysts · 4 extraction pipelines · 3 model tiers
Touch ID · Air-gap · Bodega-only · Apple Silicon
React UI · iMessage · Apple Notes · Folder watching
OpenClaw skill for WhatsApp/Telegram/Slack
Shadows: monitors, research cycles, morning briefings, calendar prep

"Bloomberg's depth. Your Mac's power. Your data never leaves."
```