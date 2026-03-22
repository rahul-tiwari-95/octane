s# Octane Sessions 28–33: Launch Package
# Private Financial Intelligence + Deep Research for Knowledge Workers
# SRSWTI Research Labs | 2026-03-20

---

## The Positioning

Bloomberg costs $30,000/year and requires fingerprint login.
Perplexity Finance is free but sends your portfolio to the cloud.
Octane costs nothing after hardware, runs entirely on your Apple Silicon, and uses Touch ID to gate your financial data through the Secure Enclave.

Bloomberg's depth. Perplexity's accessibility. Octane's privacy. That's the launch story.

---

## Session 28: Security Vault + Touch ID Integration

### Goal
Hardware-level security that makes "your data never leaves your machine" not just a claim but a cryptographic guarantee.

### What to Build

**Swift Touch ID helper** (`octane-auth/`)

A small compiled Swift CLI (~200 lines) that handles Keychain + Touch ID interaction. Python calls it via subprocess, receives results on stdout. No Python-to-Swift bridge complexity — just a binary.

```
octane-auth store <vault> <key> <value>   # Touch ID → store encrypted in Keychain
octane-auth retrieve <vault> <key>        # Touch ID → decrypt and return
octane-auth check <vault>                 # Touch ID → returns 0 (success) or 1
octane-auth list-vaults                   # list configured vaults
```

Keychain items created with `SecAccessControl` set to `.biometryCurrentSet` — data is physically inaccessible without the enrolled fingerprint. Keys stored in the Secure Enclave never leave the hardware.

**Vault system** (`octane/security/vault.py`)

Each sensitive data domain gets its own vault:

```
finance     — portfolio positions, broker credentials, trade history
health      — biomarkers, WHOOP/Oura exports, blood work
research    — findings marked as confidential
code        — API keys, tokens, credentials
```

Vaults are encrypted-at-rest. The encryption key for each vault is stored in macOS Keychain with Touch ID protection. When you run `octane portfolio`, Touch ID unlocks the finance vault. When the command finishes, the key is discarded from memory.

**CLI:**
```bash
octane vault create finance           # Touch ID → create vault
octane vault lock finance             # discard key from memory
octane vault status                   # show which vaults exist, locked/unlocked
octane vault destroy finance --yes    # permanently delete vault + all data
```

**Air-gap mode** (`octane/security/airgap.py`)

```bash
octane airgap on                      # disable all network access
octane airgap off                     # re-enable
octane airgap status                  # show current mode
```

When airgap is on: no web searches, no Bodega Intel API, no Redis sync, no mesh. CLI shows a red 🔒 indicator. Octane works only with local Bodega models and data already in Postgres.

**Data provenance tracking**

Every row in Postgres gets a `provenance` JSONB column:
```json
{
  "source": "bodega_intel_api",
  "command": "octane investigate",
  "trace_id": "abc12345",
  "timestamp": "2026-03-20T14:23:00Z",
  "airgap": false
}
```

`octane audit <finding-id>` shows the complete chain of custody.

### Session 28 Deliverables
- `octane-auth` Swift binary (compiled, ~200 lines)
- `octane/security/vault.py` — vault creation, lock/unlock, Touch ID integration
- `octane/security/airgap.py` — network kill switch
- Provenance column added to key tables
- `octane vault`, `octane airgap`, `octane audit` CLI commands
- 15+ tests

---

## Session 29: Portfolio Command Group + Finance Catalysts

### Goal
`octane portfolio` becomes the private Bloomberg terminal for your Mac.

### What to Build

**Portfolio import and management:**
```bash
octane portfolio import ~/Downloads/schwab_positions.csv
octane portfolio import ~/Downloads/fidelity_positions.csv --broker fidelity
# Auto-detects CSV format from major brokers: Schwab, Fidelity, Vanguard,
# Interactive Brokers, Robinhood, Webull, E*TRADE
# Parses: ticker, shares, cost_basis, date_acquired
# Stores in finance vault (Touch ID gated)

octane portfolio show
# ┌────────┬────────┬──────────┬──────────┬──────────┬──────────┐
# │ Ticker │ Shares │ Cost     │ Current  │ P&L      │ Weight   │
# ├────────┼────────┼──────────┼──────────┼──────────┼──────────┤
# │ NVDA   │ 50     │ $8,250   │ $8,860   │ +$610    │ 35.2%    │
# │ VOO    │ 20     │ $9,800   │ $10,420  │ +$620    │ 41.4%    │
# │ AAPL   │ 30     │ $5,400   │ $5,940   │ +$540    │ 23.6%    │
# └────────┴────────┴──────────┴──────────┴──────────┴──────────┘
# Total: $25,220 | Cost: $23,450 | P&L: +$1,770 (+7.5%)

octane portfolio analyze
# Runs full investigation on your actual positions:
# - Per-position valuation analysis
# - Concentration risk (any position > 20% of portfolio)
# - Sector exposure breakdown
# - Correlation analysis between holdings
# - Dividend yield and income projection

octane portfolio risk
# Monte Carlo simulation (existing catalyst) on YOUR positions
# Shows: expected range at 1yr, 5yr, max drawdown scenarios
# All deterministic — no LLM, no cloud, no hallucination on the numbers

octane portfolio rebalance --target "40% US equity, 30% international, 20% bonds, 10% REIT"
# Shows proposed trades to reach target allocation
# Does NOT execute trades — shows the plan only
```

**New finance catalysts (deterministic, no LLM):**

- `earnings_calendar` — pulls upcoming earnings dates for your holdings from Bodega Intel API, stores locally
- `options_greeks` — Black-Scholes pricing, delta/gamma/theta/vega for a given option chain (pure math)
- `dividend_analyzer` — yield, payout ratio, growth rate, ex-date tracking
- `sector_exposure` — maps tickers to GICS sectors, calculates portfolio weights per sector
- `correlation_matrix` — pairwise correlation between holdings using historical returns

**Leveraging Bodega capabilities:**

- **Structured JSON outputs** for all LLM-assisted analysis: when `octane portfolio analyze` calls the 8B model for qualitative assessment, use `response_format: { type: "json_schema" }` to get guaranteed-valid JSON back. No regex parsing.
- **Continuous batching** for multi-ticker research: when analyzing a 10-position portfolio, batch all ticker research queries through Bodega's CB engine for 2-3x throughput vs sequential.
- **Prompt caching**: portfolio analysis uses a consistent system prompt. With `prompt_cache_size: 10`, repeated analysis runs get near-instant TTFT.

### Session 29 Deliverables
- `octane/cli/portfolio.py` — import, show, analyze, risk, rebalance
- `octane/catalysts/earnings_calendar.py`
- `octane/catalysts/options_greeks.py`
- `octane/catalysts/dividend_analyzer.py`
- `octane/catalysts/sector_exposure.py`
- `octane/catalysts/correlation_matrix.py`
- Broker CSV parsers for top 7 US brokers
- Portfolio data stored in finance vault (Touch ID gated)
- 25+ tests

---

## Session 30: Deep Research Engine Upgrade

### Goal
Make `octane investigate` and `octane ask --deep` produce research that's genuinely better than anything Perplexity or ChatGPT Deep Research can do — because Octane can go wider, deeper, and store everything permanently.

### What to Build

**Convergence-based deepening** (replacing fixed 3 rounds):

```python
class DepthController:
    """Manages per-dimension deepening with convergence detection."""
    
    async def should_continue(self, dimension, round_num, new_pages):
        # Round yielded < 2 genuinely new pages → this dimension has converged
        if len(new_pages) < 2:
            return False
        # Round yielded new pages but they're mostly duplicates of existing → converging
        if self.novelty_score(new_pages, dimension.existing_pages) < 0.3:
            return False
        # Hard cap at 5 rounds per dimension
        if round_num >= 5:
            return False
        return True
```

This means a rich dimension (lots of new sources each round) gets 4-5 rounds, while a thin dimension (few sources, quickly converges) stops at 2. Data-driven depth, not fixed.

**Cross-reference catalyst** (`octane/catalysts/cross_referencer.py`):

After all dimensions complete, Code Agent runs a cross-reference pass:
- Extract factual claims from each dimension's findings
- Check each claim against findings from other dimensions
- Claims appearing in 3+ independent sources → **confirmed**
- Claims from single source → **unverified**
- Contradictions between sources → **flagged**
- Output: confidence-scored claims with source attribution

This is the capability no chatbot has. A single LLM call can't cross-reference — it sees one context window. Octane's multi-dimensional pipeline has findings from 6-8 independent research passes stored in Postgres. The cross-referencer queries across all of them.

**Citation tracking:**

Every finding stores its source URLs. The final report includes inline citations:

```
NVDA's gross margin expanded to 76% in Q4 2025 [1][2], driven primarily
by data center GPU demand [1][3][4]. However, one analyst notes potential
margin pressure from increased competition in the inference chip market [5].

Sources:
[1] reuters.com/technology/nvidia-q4-earnings-2025
[2] sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=NVDA
[3] cnbc.com/2025/02/nvidia-data-center-revenue
[4] bloomberg.com/news/articles/nvidia-ai-chip-demand  
[5] ft.com/content/nvidia-competition-analysis
```

**Export formats:**

```bash
octane investigate "NVDA valuation" --export report.md
octane investigate "NVDA valuation" --export report.pdf
octane investigate "NVDA valuation" --export report.docx
```

PDF and DOCX generation via `weasyprint` (PDF) and `python-docx` (Word). These are the formats lawyers, consultants, and analysts actually deliver to clients.

**Research library:**

```bash
octane research library
# Shows all stored investigations and deep queries:
# ID        Query                              Date        Dims  Sources  Findings
# inv-001   NVDA valuation analysis            2h ago      6     42       28
# inv-002   AI semiconductor industry          1d ago      8     67       45
# deep-001  Israel Iran latest developments    3d ago      —     13       9

octane research recall "NVDA"
# Searches across all stored investigations for NVDA-related findings
# Uses pgVector semantic search if embeddings available
# Falls back to keyword search otherwise
```

**Leveraging Bodega capabilities:**

- **Speculative decoding** for the REASON tier synthesis: final report generation is single-user, latency-sensitive — exactly the use case spec decoding optimizes. With a Qwen3-0.6B draft model, the 8B target generates the report 2-3x faster.
- **Reasoning parser** (`reasoning_parser: "qwen3"`): for the DimensionPlanner and cross-referencer, enable explicit reasoning mode. The model's chain-of-thought is captured separately from the output, giving you both the reasoning trace (for `octane trace -v`) and the clean structured output.

### Session 30 Deliverables
- `octane/agents/web/depth_controller.py` — convergence-based deepening
- `octane/catalysts/cross_referencer.py` — multi-source claim verification
- Citation tracking in findings and reports
- PDF and DOCX export
- `octane research library` and `octane research recall`
- 20+ tests

---

## Session 31: Watchlist + Earnings Hub + Monitor

### Goal
The always-on intelligence layer. Octane watches your positions and alerts you when something matters.

### What to Build

**Watchlist:**
```bash
octane watch add NVDA AAPL MSFT VOO
octane watch add BTC-USD --crypto

octane watch show
# ┌────────┬──────────┬─────────┬──────────┬──────────┬──────────┐
# │ Ticker │ Price    │ Change  │ Volume   │ RSI      │ Signal   │
# ├────────┼──────────┼─────────┼──────────┼──────────┼──────────┤
# │ NVDA   │ $177.19  │ ▼4.16%  │ 307M     │ 32 🔴    │ Oversold │
# │ AAPL   │ $198.42  │ +0.8%   │ 45M      │ 58       │ Neutral  │
# │ MSFT   │ $412.31  │ +1.2%   │ 22M      │ 62       │ Neutral  │
# │ VOO    │ $521.08  │ -0.1%   │ 8M       │ 55       │ Neutral  │
# └────────┴──────────┴─────────┴──────────┴──────────┴──────────┘
```

RSI and technical signals computed by existing catalysts — deterministic, no LLM.

**Earnings hub:**
```bash
octane earnings
# Upcoming Earnings for Your Holdings
# ┌────────┬──────────────┬──────────┬────────────────────────────┐
# │ Ticker │ Date         │ Time     │ Consensus                  │
# ├────────┼──────────────┼──────────┼────────────────────────────┤
# │ NVDA   │ Mar 26, 2026 │ After    │ EPS $0.89 | Rev $38.1B    │
# │ AAPL   │ Apr 5, 2026  │ After    │ EPS $1.62 | Rev $94.2B    │
# └────────┴──────────────┴──────────┴────────────────────────────┘

octane earnings prep NVDA
# Runs octane investigate focused on NVDA earnings:
# - Last 4 quarters results (beats/misses)
# - Analyst estimate revisions (last 30/60/90 days)  
# - Key metrics to watch (data center rev, gross margin, guidance)
# - Historical stock reaction to earnings (+/- 1 day)
# Stores as research for post-earnings comparison
```

**Compound monitor** (from CLI v2 spec):
```bash
octane monitor "NVDA" --signals price,news,earnings,sentiment

octane monitor alerts
# 🔴 HIGH — NVDA Multi-Signal Alert (2h ago)
#    Price: -4.16% + RSI oversold (32)
#    News: "Nvidia faces new export restrictions"
#    Action: Review position

octane monitor list
octane monitor pause NVDA
octane monitor stop NVDA
```

**Leveraging Bodega capabilities:**

- **Continuous batching** for watchlist price fetches: when refreshing 10+ tickers, batch all through CB engine
- **Bodega RAG** for earnings prep: upload the company's last 10-K/10-Q PDF via `/v1/rag/upload`, then query it with `/v1/rag/query` alongside the web research. The earnings prep combines web intelligence with SEC filing analysis — all local.

### Session 31 Deliverables
- Enhanced `octane/cli/watch.py` — add, show, remove with RSI/signal display
- `octane/cli/earnings.py` — upcoming, prep command
- `octane/cli/monitor.py` — compound signal monitoring via Shadows
- Bodega RAG integration for SEC filing analysis
- 20+ tests

---

## Session 32: plan + replay + chain Upgrades

### Goal
Complete the power command set. Make `octane plan` and `octane replay` production-quality.

### What to Build

**`octane plan`:**
```bash
octane plan "Build a $100K portfolio over 5 years starting with $500/month"
# GoalAnalyzer (REASON tier):
#   1. Feasibility check (is $100K achievable at $500/mo?)
#   2. Required return rate calculation (deterministic catalyst)
#   3. Vehicle research (ETFs, index funds, tax-advantaged accounts)
#   4. Monte Carlo projection (existing catalyst, YOUR parameters)
#   5. Milestone schedule (month-by-month targets)
#   6. Risk scenarios (bear/base/bull)
#
# Output: structured plan with phases, milestones, specific actions
# Stored in research library for future reference
# Offers: octane monitor setup for the plan

octane plan "Transition from frontend to ML engineer in 6 months"
# Same structure but non-financial: skill gap analysis, learning resources,
# job market research, timeline with weekly milestones
```

**`octane replay`:**
```bash
octane replay <trace-id> --diff
# 1. Read original trace DAG
# 2. Re-run same agents with fresh data
# 3. Compare: what changed since last run?
#
# 🔄 Replay: "NVDA valuation analysis" (original: 7 days ago)
# 
# What Changed:
#   Price: $185.40 → $177.19 (-4.4%)
#   RSI: 58 → 32 (neutral → oversold)
#   News: NEW — export restriction concerns
#   Analyst consensus: unchanged ($220 target)
#
# What Didn't Change:
#   Revenue growth: 94% YoY
#   Gross margin: 76%
#
# Updated Verdict: [synthesized comparison]
```

**`octane chain` upgrades:**
```bash
# Parallel steps
octane chain \
  "parallel: prices=fetch NVDA, news=fetch news AI chips" \
  "report: synthesize {prices} {news}"

# Conditional steps
octane chain \
  "data: fetch finance NVDA" \
  "if {data.change_pct} < -3: alert 'NVDA down big'" \
  "synthesize {data}"
```

### Session 32 Deliverables
- `octane/osa/goal_analyzer.py` — GoalAnalyzer for `octane plan`
- `octane/osa/replay.py` — trace reconstruction + fresh data + diff
- Chain parser upgrades: parallel steps, conditionals
- 20+ tests

---

## Session 33: React UI Foundation

### Goal
A local React app served by the daemon at `http://localhost:44480`. Two views: Finance Dashboard and Research Command Center. Data comes from the same daemon the CLI uses — same Postgres, same Redis, same Synapse events.

### What to Build

**Backend (FastAPI WebSocket):**

Already partially designed in the daemon. Add:
- `GET /api/portfolio` — portfolio positions + P&L (Touch ID check)
- `GET /api/watchlist` — current prices + signals
- `GET /api/earnings` — upcoming earnings calendar
- `GET /api/research/library` — all stored investigations
- `GET /api/research/:id` — full investigation report
- `GET /api/monitor/alerts` — active alerts
- `WS /ws/events` — real-time Synapse event stream
- `WS /ws/prices` — live price updates for watchlist

**Frontend (React + Tailwind):**

Two views, toggled via sidebar:

**Finance Dashboard:**
- Portfolio table with live P&L (WebSocket price updates)
- Watchlist with RSI/signals
- Earnings calendar
- Allocation pie chart (rendered client-side from catalyst data)
- "Ask Octane" input that sends queries through the daemon
- Recent research sidebar

**Research Command Center:**
- Active investigations with progress bars
- Research library with search
- Investigation detail view (structured report with citations)
- Live Octane activity feed (Synapse events via WebSocket)
- Daemon status (models loaded, queue depth, RAM usage)

**Build approach:**
- `octane/ui/` — FastAPI app with API routes
- `octane/ui/frontend/` — React app (Vite + React + Tailwind)
- `octane ui start` — builds frontend, starts FastAPI on :44480
- Development: `octane ui dev` — Vite dev server with hot reload

**No external data dependencies.** Every chart, every number, every finding comes from local Postgres and local Bodega. The browser talks to localhost only. No CDN, no analytics, no telemetry. View source and verify.

### Session 33 Deliverables
- FastAPI backend with 10+ API endpoints + 2 WebSocket channels
- React frontend with Finance Dashboard + Research Command Center
- `octane ui start` and `octane ui dev` commands
- Touch ID gate on finance API endpoints
- 15+ backend tests

---

## Session Map

| Session | Title | Key Deliverable |
|---------|-------|-----------------|
| 28 | Security Vault + Touch ID | `octane vault`, `octane airgap`, `octane audit`, Swift Touch ID helper |
| 29 | Portfolio + Finance Catalysts | `octane portfolio`, 5 new catalysts, broker CSV import |
| 30 | Deep Research Engine Upgrade | Convergence deepening, cross-referencer, citations, export PDF/DOCX |
| 31 | Watchlist + Earnings + Monitor | `octane watch show`, `octane earnings prep`, compound `octane monitor` |
| 32 | plan + replay + chain upgrades | `octane plan`, `octane replay --diff`, parallel chain steps |
| 33 | React UI Foundation | Finance Dashboard + Research Command Center at localhost:44480 |

---

## What You Launch With

After Session 33, Octane ships with:

**For the financial professional:**
- Touch ID-gated portfolio with broker CSV import
- 10 deterministic finance catalysts (no hallucination on numbers)
- Earnings hub with SEC filing analysis via Bodega RAG
- Compound monitoring with cross-signal alerts
- Air-gap mode for absolute data isolation
- Monte Carlo projections, options Greeks, correlation analysis
- Everything runs on your Mac, nothing touches a cloud

**For the knowledge worker:**
- `octane investigate` with convergence-based deepening (up to 40+ sources per query)
- Cross-source claim verification with confidence scoring
- Inline citations tracing every claim to its source URL
- Export to PDF and DOCX for client deliverables
- Research library that accumulates knowledge across investigations
- `octane plan` for goal-oriented action planning
- `octane replay` for tracking how analyses change over time

**For both:**
- React dashboard at localhost (Finance view + Research Command Center)
- Real-time WebSocket updates from the daemon
- Complete data provenance and audit trail
- 800+ tests across unit, integration, and E2E

**The pitch:** "Bloomberg's depth. Zero annual cost. Your data never leaves your Mac."