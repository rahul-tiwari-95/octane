# Octane Sessions 40–43: Ship It
# Pipes. UI Polish. iMessage. eyeso. Launch.
# SRSWTI Research Labs | 2026-03-29

---

## Where We Are

38 sessions. ~1,300 tests. ~100 CLI commands. Mission Control UI with globe + terminal. IntentGate + ChatEngine + persona. Knowledge accumulation across 16 Postgres tables. Portfolio with dividends, tax lots, crypto. Touch ID vault.

Sessions 40-43 connect the engine to the world.

---

## Session 40: Modular Pipes + UI Enhancements (Round 1)

### Part A: Composable CLI Pipes

**`octane search` — top-level command group:**

```bash
octane search web "NVDA earnings" --json --limit 10
octane search news "AI chip export" --json --limit 5
octane search youtube "transformer explained" --json --limit 5
octane search arxiv "attention mechanism" --json --limit 10
```

Each wraps existing functions with a `--json` flag that outputs structured JSON to stdout. Without `--json`, shows the existing Rich table. ~150 lines for all four.

**`octane extract --stdin` — reads URLs from stdin:**

```bash
octane search arxiv "RAG" --urls-only | octane extract --stdin --json
```

~80 lines wrapping existing extract logic.

**`octane synthesize` — new command:**

```bash
octane search news "NVDA" --json | octane extract --stdin --json | octane synthesize --stdin --template briefing
```

~120 lines wrapping existing Evaluator.

**`octane ask --deep N`:**

Wire integer support through ask.py. `--deep 6` → `max_dimensions=6`. One-line change.

### Part B: Mission Control UI Enhancements (Round 1)

**BUG FIX: Globe loads empty on fresh start.**

The globe currently only populates from live WebSocket events. On a fresh page load or restart, it's a blank dark sphere. Fix: on dashboard mount, fetch historical event data from the API and seed the globe.

```typescript
// Dashboard.tsx — on mount
useEffect(() => {
  fetch('/api/traces?limit=200')
    .then(res => res.json())
    .then(traces => {
      // Extract geo-coordinates from trace events (web sources have URLs → geo lookup)
      // Seed globe with historical nodes
      globeRef.current.seedPoints(tracesToGeoPoints(traces));
    });
}, []);
```

For events without real geo-coordinates (most web searches), distribute nodes across a world map using URL TLD mapping (`.co.uk` → London area, `.de` → Berlin area, etc.) or hash the URL to deterministic lat/lng. The visual effect: the globe always looks alive with your research history, even on first load.

**NEW: Recent Queries panel shows query text, not trace IDs.**

Right now the Recent Queries panel shows `trace:cf2037dd` for most entries. Only queries that came through `octane ask` show the actual query text. Fix: store the original query string in the trace metadata so all entries show human-readable text.

**NEW: Model panel memory bars should be proportional.**

The memory bars (23GB, 12GB, etc.) should be proportional to total system RAM (64GB). Right now they appear to be fixed-width. Scale them: `width = (model_ram / total_ram) * 100%`.

**NEW: Mobile-responsive layout.**

Mission Control needs to work on iPad and iPhone (accessed via LAN). The current 3-column layout breaks on small screens.

```
Desktop (>1200px):
┌──────────┬────────────┬──────────┐
│ Vitals   │   Globe    │  Models  │
├──────────┴────────────┴──────────┤
│ Live Events  │  Recent Queries   │
└──────────────┴───────────────────┘

Tablet (768-1200px):
┌──────────┬──────────┐
│ Vitals   │  Models  │
├──────────┴──────────┤
│       Globe         │
├──────────┬──────────┤
│ Events   │ Queries  │
└──────────┴──────────┘

Mobile (<768px):
┌─────────────────┐
│ Vitals (compact)│  ← horizontal bar, not vertical list
├─────────────────┤
│ Models (cards)  │
├─────────────────┤
│ Globe (hidden)  │  ← hide globe on mobile, too heavy
├─────────────────┤
│ Recent Queries  │
├─────────────────┤
│ Live Events     │
└─────────────────┘
```

Use Tailwind responsive breakpoints: `lg:grid-cols-3 md:grid-cols-2 grid-cols-1`. Hide globe on mobile with `hidden md:block`. The terminal page works naturally on mobile since xterm.js handles its own sizing via the fit addon.

### Files

```
octane/cli/search.py              # NEW
octane/cli/synthesize.py          # NEW
octane/cli/extract.py             # MODIFIED — --stdin
octane/cli/ask.py                 # MODIFIED — --deep N
octane/ui-frontend/src/pages/Dashboard.tsx    # Globe seed, responsive
octane/ui-frontend/src/components/Globe.tsx   # Historical data seeding
octane/ui-frontend/src/components/ModelPanel.tsx  # Proportional bars
octane/ui-frontend/src/index.css  # Mobile breakpoints
```

### Tests

`tests/unit/test_session40_pipes_ui.py` — 30 tests

---

## Session 41: iMessage + LAN Access + UI Enhancements (Round 2)

### Part A: iMessage Integration

**AppleScript bridge** (`octane/macos/applescript.py`):

```python
class AppleScriptBridge:
    async def send_imessage(self, to: str, message: str) -> bool:
        script = f'''
        tell application "Messages"
            send "{self._escape(message)}" to buddy "{to}" of \\
                (1st service whose service type = iMessage)
        end tell
        '''
        return await self._run_osascript(script)
    
    async def create_note(self, title: str, body: str, folder: str = "Octane") -> bool:
        # Create Apple Note with HTML body in Octane folder
    
    async def read_calendar(self, hours_ahead: int = 24) -> list[dict]:
        # Read upcoming calendar events
```

**iMessage Shadow:**

```python
class IMessageShadow:
    """Polls Messages.db for new messages, routes to ChatEngine."""
    
    async def on_message(self, sender: str, text: str):
        if sender not in self.approved_contacts:
            return
        response = await self.chat_engine.respond(text)
        await self.applescript.send_imessage(sender, response)
```

**iMessage as pipe endpoint:**

```bash
# Pipe any output to iMessage
octane search news "NVDA" --json | octane extract --stdin --json | octane synthesize --stdin --template briefing | octane macos imessage send --stdin --to "+1234567890"
```

**LAN access:**

```bash
octane config set lan_access true
octane config set lan_token "my-secret"
# Rebinds Bodega + Mission Control to 0.0.0.0
# Shows QR code with URL + token on octane ui start
```

**CLI:**

```bash
octane macos setup                    # Check permissions, create folders
octane macos imessage on              # Enable (prompts for approved contacts)
octane macos imessage off
octane macos imessage send "text" --to "+1234567890"
octane macos imessage send --stdin --to "+1234567890"
octane macos notes create "Title" --body "content"
octane macos notes create "Title" --stdin
octane macos status
```

### Part B: UI Enhancements (Round 2) — Data Visualizations

**Portfolio Charts page** (new page in nav: DASHBOARD | PORTFOLIO | TERMINAL):

```
┌─────────────────────────────────────────────────────────┐
│ PORTFOLIO OVERVIEW                            🔒 Touch ID│
├──────────────────────────┬──────────────────────────────┤
│                          │                              │
│  Allocation Donut        │  Sector Exposure Radar       │
│  ┌──────────────────┐    │  ┌──────────────────┐        │
│  │   ╭─────╮        │    │  │     Tech          │        │
│  │  ╱ NVDA  ╲       │    │  │   ╱    ╲          │        │
│  │ │  35%    │       │    │  │ Health──┼──Finance│        │
│  │  ╲ AAPL  ╱       │    │  │   ╲    ╱          │        │
│  │   ╰─────╯        │    │  │     Energy        │        │
│  └──────────────────┘    │  └──────────────────┘        │
│                          │                              │
├──────────────────────────┴──────────────────────────────┤
│                                                          │
│  Holdings Table (sortable by P&L, weight, sector)        │
│  ┌────────┬────────┬──────┬──────┬──────┬──────────┐    │
│  │ Ticker │ Shares │ Cost │ Curr │ P&L  │ Weight   │    │
│  ├────────┼────────┼──────┼──────┼──────┼──────────┤    │
│  │ NVDA   │ 50     │$165  │$177  │+$610 │ 35.2%    │    │
│  └────────┴────────┴──────┴──────┴──────┴──────────┘    │
│                                                          │
├──────────────────────────┬──────────────────────────────┤
│                          │                              │
│  Correlation Heatmap     │  Risk Distribution           │
│  ┌──────────────────┐    │  ┌──────────────────┐        │
│  │ NVDA AAPL VOO AMD│    │  │                  │        │
│  │ ███  ▓▓▓  ░░░ ███│    │  │   ╱╲             │        │
│  │ ▓▓▓  ███  ▓▓▓ ▓▓▓│    │  │  ╱  ╲            │        │
│  │ ░░░  ▓▓▓  ███ ░░░│    │  │ ╱    ╲           │        │
│  │ ███  ▓▓▓  ░░░ ███│    │  │╱______╲          │        │
│  └──────────────────┘    │  └──────────────────┘        │
│  (dark=high correlation)  │  (Monte Carlo distribution)  │
│                          │                              │
└──────────────────────────┴──────────────────────────────┘
```

**Chart types to implement (using Recharts or D3):**

| Chart | Data Source | What It Shows | Library |
|-------|-----------|---------------|---------|
| **Allocation Donut** | portfolio positions | Weight per holding, click to drill | Recharts PieChart |
| **Sector Radar** | sector_exposure catalyst | Portfolio exposure across GICS sectors | Recharts RadarChart |
| **Correlation Heatmap** | correlation_analysis catalyst | Pairwise correlation between holdings | D3 custom (color matrix) |
| **Risk Distribution** | Monte Carlo catalyst | Bell curve of projected returns | Recharts AreaChart |
| **Net Worth Timeline** | portfolio_snapshots table | Net worth over time | Recharts LineChart |
| **Dividend Bar** | dividend_tracker catalyst | Monthly projected dividend income | Recharts BarChart |
| **Research Clustering** | extracted_documents + pgVector | Topic clusters of your research | D3 force-directed (like the globe but 2D) |

**Research page enhancements:**

| Chart | What It Shows |
|-------|---------------|
| **Source Distribution Pie** | Breakdown of extractions by source type (arXiv, YouTube, web, PDF) |
| **Research Activity Histogram** | Extractions per day/week over time |
| **Trust Score Distribution** | Histogram of reliability scores across all stored content |
| **Dimension Spider/Radar** | For a given investigation, show coverage across dimensions |

**Implementation notes:**

- Recharts for standard charts (donut, bar, line, area, radar) — already available as a React dependency
- D3 for custom visualizations (correlation heatmap, clustering, spider graphs) — already bundled for the globe
- All chart data comes from API endpoints that query existing Postgres tables and catalysts
- Charts are components that can be mixed into any page
- Touch ID gate on portfolio charts (same as portfolio API endpoints)

### Files

```
octane/macos/__init__.py
octane/macos/applescript.py
octane/macos/imessage_shadow.py
octane/macos/permissions.py
octane/cli/macos.py
octane/ui-frontend/src/pages/Portfolio.tsx          # NEW page
octane/ui-frontend/src/components/charts/
├── AllocationDonut.tsx
├── SectorRadar.tsx
├── CorrelationHeatmap.tsx
├── RiskDistribution.tsx
├── NetWorthTimeline.tsx
├── DividendBar.tsx
├── ResearchClustering.tsx
├── SourceDistribution.tsx
├── TrustHistogram.tsx
└── DimensionSpider.tsx
octane/ui/routes/portfolio_api.py                   # Portfolio chart data endpoints
octane/ui/routes/charts.py                          # Research chart data endpoints
```

### Tests

`tests/unit/test_session41_imessage_charts.py` — 25 tests

---

## Session 42: eyeso Scripts — The Community Flywheel

### eyeso v0.1 — Intentionally Simple

eyeso is NOT a compiler. It's a ~250 line Python parser that reads a structured text file and executes Octane CLI commands via subprocess. The syntax is designed for shareability: anyone can read an eyeso script and understand what it does in 10 seconds.

**Syntax:**

```eyeso
# morning-briefing.eyeso
# Sends portfolio + news summary via iMessage every morning.
#
# Install:  octane script install morning-briefing
# Run:      octane script run morning-briefing
# Schedule: octane script schedule morning-briefing --every day --at 7:00

name: morning-briefing
description: Portfolio + news summary delivered via iMessage
author: rahul
version: 1.0

---

# Variables — user sets these on install
$PHONE = "+1234567890"
$TOPICS = "AI semiconductors, NVDA, portfolio"

# Steps — each step's output is available as {step_name}
portfolio: octane portfolio show --json
news: octane search news "$TOPICS" --json --limit 5
extracted: octane extract --stdin --json < {news}
briefing: octane synthesize --stdin --template briefing < {extracted}
send: octane macos imessage send --stdin --to "$PHONE" < {briefing}
```

**Syntax rules:**

```
# Comments start with #
# Header (before ---): name, description, author, version
# Variables: $NAME = "default value" — user overrides on install/run
# Steps: name: octane command
# References: {step_name} inserts previous step's output
# Stdin pipe: < {step_name} pipes as stdin
# @parallel — following steps run concurrently
# @sequential — following steps run in order (default)
```

**10 built-in scripts:**

| Script | Audience | What It Does |
|--------|----------|-------------|
| `morning-briefing.eyeso` | Everyone | Portfolio + news + calendar → iMessage |
| `arxiv-monitor.eyeso` | Researchers | Watch topic on arXiv, alert on new papers |
| `earnings-prep.eyeso` | Finance | Deep investigation 3 days before earnings |
| `weekly-portfolio.eyeso` | Finance | Snapshot + risk + rebalance suggestions |
| `news-digest.eyeso` | Everyone | Daily topic digest → Apple Notes |
| `competitor-watch.eyeso` | OSINT | Monitor competitor mentions |
| `price-alert.eyeso` | Finance | Multi-signal price monitor → iMessage |
| `literature-review.eyeso` | Academic | Weekly arXiv + YouTube scan |
| `deep-dive.eyeso` | Everyone | 12-dim investigation with citations |
| `daily-standup.eyeso` | Developers | Yesterday's research + today's calendar |

**CLI:**

```bash
octane script list                                        # Installed scripts
octane script install morning-briefing                    # From built-in examples
octane script install https://github.com/user/script.eyeso  # From URL
octane script run morning-briefing                        # Run now
octane script run morning-briefing --var PHONE="+1..."    # Override variables
octane script schedule morning-briefing --every day --at 7:00  # As Shadow
octane script unschedule morning-briefing
octane script show morning-briefing                       # View contents
octane script edit morning-briefing                       # Open in $EDITOR
octane script create "my-flow"                            # Scaffold new script
octane script validate my-flow.eyeso                      # Check syntax
```

**The interpreter:**

```python
class EyesoRunner:
    def __init__(self, script_path, var_overrides=None):
        self.metadata, self.variables, self.steps = self.parse(script_path)
        if var_overrides:
            self.variables.update(var_overrides)
    
    async def run(self):
        outputs = {}
        for step in self.steps:
            cmd = self.resolve(step.command, outputs)
            stdin_data = outputs.get(step.stdin_ref) if step.stdin_ref else None
            result = await self.execute(cmd, stdin=stdin_data)
            outputs[step.name] = result
    
    async def execute(self, cmd, stdin=None):
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate(input=stdin.encode() if stdin else None)
        return stdout.decode()
```

### Files

```
octane/eyeso/__init__.py
octane/eyeso/parser.py
octane/eyeso/runner.py
octane/eyeso/scheduler.py
octane/cli/script.py
octane/eyeso/examples/
├── morning-briefing.eyeso
├── arxiv-monitor.eyeso
├── earnings-prep.eyeso
├── weekly-portfolio.eyeso
├── news-digest.eyeso
├── competitor-watch.eyeso
├── price-alert.eyeso
├── literature-review.eyeso
├── deep-dive.eyeso
└── daily-standup.eyeso
```

### Tests

`tests/unit/test_session42_eyeso.py` — 30 tests

---

## Session 43: Open Source Prep + Launch Polish

### Open Source Boundary — Physical Separation

```bash
# Verify Bodega independence:
cd bodega-mlx-engine && pip install -e . && python -m bodega_mlx_engine.main --help
# Must work with zero octane imports

# Verify Octane depends correctly:
cd octane && pip install -e . && octane health
```

### setup.sh

```bash
#!/bin/bash
# 1. Check Apple Silicon (fail on Intel)
# 2. Homebrew, postgresql, redis, ffmpeg
# 3. Python venv + dependencies
# 4. Postgres init + migrations
# 5. Download default model (raptor-8b)
# 6. macOS permissions guidance
# 7. Create ~/Octane/inbox/
# 8. Install eyeso example scripts
# 9. Run fast test subset
# 10. Print first-command suggestion
```

### QUICKSTART.md

```markdown
# Octane — 5 Minutes to First Investigation

## Install
git clone https://github.com/srswti/octane && cd octane && bash setup.sh

## First Commands
octane ask "what's happening with NVDA?"
octane ask "NVDA valuation" --deep 6 --cite
octane portfolio import ~/Downloads/schwab.csv
octane portfolio analyze --deep
octane ui start    # Mission Control at localhost:44480

## eyeso Scripts
octane script run morning-briefing --var PHONE="+1234567890"
```

### Demo Assets

- `demo_portfolio.csv` — 10-position sample portfolio
- `demo.sh` — 30-second recording script
- 4 Reddit post drafts
- Mission Control screenshots (dashboard, globe, terminal, portfolio charts)
- License files: `bodega-mlx-engine/LICENSE` (MIT), `octane/LICENSE` (BSL 1.1)

### Final Integration Test

```bash
octane search news "NVDA" --json \
  | octane extract --stdin --json \
  | octane synthesize --stdin --template briefing \
  | octane macos imessage send --stdin --to "+1234567890"
# Verify: iMessage received on iPhone with NVDA briefing
```

### Tests

`tests/unit/test_session43_launch.py` — 15 tests

---

## UI Enhancement Summary (Across Sessions 40-41)

### Bug Fixes

| Bug | Fix | Session |
|-----|-----|---------|
| Globe blank on fresh load | Seed from `/api/traces` historical data on mount | 40 |
| Recent Queries shows trace IDs not query text | Store query string in trace metadata | 40 |
| Model memory bars not proportional | Scale width to `model_ram / total_ram` | 40 |

### New Pages

| Page | Nav Label | Content | Session |
|------|-----------|---------|---------|
| Portfolio | PORTFOLIO | Allocation donut, sector radar, holdings table, correlation heatmap, risk distribution, net worth timeline, dividend bar | 41 |

### New Charts

| Chart Type | Library | Page | Data Source |
|-----------|---------|------|-------------|
| Allocation Donut | Recharts PieChart | Portfolio | portfolio positions |
| Sector Radar | Recharts RadarChart | Portfolio | sector_exposure catalyst |
| Correlation Heatmap | D3 color matrix | Portfolio | correlation_analysis catalyst |
| Risk Distribution | Recharts AreaChart | Portfolio | Monte Carlo catalyst |
| Net Worth Timeline | Recharts LineChart | Portfolio | portfolio_snapshots |
| Dividend Income Bar | Recharts BarChart | Portfolio | dividend_tracker catalyst |
| Source Distribution Pie | Recharts PieChart | Dashboard | extracted_documents |
| Research Activity Histogram | Recharts BarChart | Dashboard | extracted_documents by date |
| Trust Score Histogram | Recharts BarChart | Dashboard | reliability scores |
| Dimension Spider | Recharts RadarChart | Research detail | investigation dimensions |
| Research Clustering | D3 force-directed | Research | topic similarity (pgVector if available) |

### Mobile Responsiveness

| Breakpoint | Layout | Globe | Charts |
|-----------|--------|-------|--------|
| Desktop (>1200px) | 3-column hero + 2-column bottom | Full 3D globe | Full size |
| Tablet (768-1200px) | 2-column hero + 2-column bottom | Smaller globe | Stacked |
| Mobile (<768px) | Single column stack | Hidden (too heavy) | Compact cards, swipeable |

Terminal page needs no mobile changes — xterm.js fit addon handles resizing. Portfolio charts stack vertically on mobile with swipe between chart types.

---

## Session Map

| Session | Title | Key Deliverables | Tests |
|---------|-------|-----------------|-------|
| 40 | Pipes + UI Round 1 | `octane search/synthesize`, `--stdin`, globe fix, mobile responsive, query text in traces | +30 |
| 41 | iMessage + UI Round 2 | AppleScript bridge, iMessage Shadow, LAN access, Portfolio page with 7 chart types, research charts | +25 |
| 42 | eyeso Scripts | Parser, runner, 10 examples, `octane script` CLI, scheduler | +30 |
| 43 | Launch Polish | setup.sh, QUICKSTART.md, OSS separation, Reddit posts, demo, licenses | +15 |

**Estimated total at Session 43: ~1,400 tests**

---

## Launch Day

```
Octane v1.0 — Session 43

38,000+ lines of Python
~1,400 tests
43 sessions
~100 CLI commands
5 agents · 13+ catalysts · 4 extraction pipelines · 3 model tiers
16 Postgres tables · 10 eyeso scripts · 11 chart types
Touch ID · Air-gap · Bodega-only · Apple Silicon
Mission Control UI (desktop + tablet + mobile)
iMessage · Apple Notes · Folder watching
Modular pipes: search | extract | synthesize
eyeso script community

"Bloomberg's depth. Your Mac's power. Your data never leaves."
```