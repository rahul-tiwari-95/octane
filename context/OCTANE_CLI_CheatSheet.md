# OCTANE CLI — Complete Cheat Sheet
### SRSWTI Research Labs | Session 36 | 2026-03-30

> **~90 commands** across 16 top-level + 16 sub-apps.  
> Activate venv first: `source .venv/bin/activate`

---

## Table of Contents

1. [Basics](#basics)  
2. [Ask & Chat](#ask--chat)  
3. [Power Commands (investigate/compare/chain)](#power-commands)  
4. [Research (Shadows background)](#research)  
5. [Extraction Pipeline](#extraction-pipeline)  
6. [Knowledge Recall](#knowledge-recall)  
7. [Tracing & Auditing](#tracing--auditing)  
8. [Data Store & Files](#data-store--files)  
9. [Background Monitors (watch)](#background-monitors)  
10. [Portfolio Management](#portfolio-management)  
11. [Workflow Templates](#workflow-templates)  
12. [Model Management](#model-management)  
13. [Daemon](#daemon)  
14. [Database](#database)  
15. [Preferences](#preferences)  
16. [Security: Vault & Airgap](#security-vault--airgap)  
17. [Mission Control Web UI](#mission-control-web-ui)  
18. [Chaining & Pipelining](#chaining--pipelining)  
19. [Shadows (perpetual tasks)](#shadows-perpetual-tasks)  
20. [.octane script files](#octane-script-files)  
21. [Analytics Dashboard](#analytics-dashboard)

---

## Basics

```bash
octane --help                  # Full CLI surface
octane version                 # Version, stack info, registered agents
octane health                  # RAM, CPU, GPU, loaded model, Bodega status
octane sysstat                 # Live system snapshot
octane agents                  # List all 5 agents and capabilities
```

---

## Ask & Chat

### Single-shot query
```bash
octane ask "What is the current price of AAPL?"
octane ask "Explain transformers" --verbose       # Show full DAG trace after answer
octane ask "Latest AI papers on reasoning" --deep  # Multi-round iterative web search
octane ask "What did I research yesterday?" --recall  # Infer from stored data only (Postgres/Redis)
octane ask "NVDA analysis" --monitor               # Live RAM/CPU overlay during query
```

### Deep mode explained
`--deep` triggers multi-round search with iterative query expansion.  
The agent keeps searching until novelty drops (n-gram convergence detection).

### Interactive chat
```bash
octane chat              # Multi-turn conversation
octane chat --verbose    # With debug logging
```

**Slash commands inside chat:**
| Command | Action |
|---------|--------|
| `/deep` | Toggle deep mode on/off |
| `/trace [id]` | Show Synapse trace for last response |
| `/history` | Print conversation history |
| `/clear` | Clear conversation context |
| `/exit` | End session |

### Session mode
```bash
octane session   # Chat + annotated trace replay when you type END
```

### Feedback
```bash
octane feedback thumbs_up                    # On last response
octane feedback thumbs_down abc123 --user me # On specific trace
```

---

## Power Commands

### `octane investigate` — multi-dimensional deep research
```bash
# Basic
octane investigate "attention mechanisms in LLMs"

# Deep: 12 research dimensions, arXiv + YouTube sources, with citations
octane investigate "attention mechanisms" --deep 12 --sources arxiv,youtube --cite

# With trust verification labels (CONFIRMED/LIKELY/UNVERIFIED)  
octane investigate "quantum computing 2026" --verify

# Quick shallow sweep (2 dimensions)
octane investigate "latest NVDA earnings" --max-dimensions 2

# No streaming (wait for full output)
octane investigate "nuclear fusion progress" --no-stream
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--deep N` | N research dimensions (default 8 if present, up to 12+) |
| `--max-dimensions N` | Cap dimensions (2–8) |
| `--sources arxiv,youtube,web` | Comma-separated source types |
| `--cite` | Inline citations + Sources section |
| `--verify` | Trust-level labels |
| `--stream/--no-stream` | Stream findings as they arrive |

### `octane compare` — structured comparison matrix
```bash
octane compare "NVDA vs AMD vs INTC"
octane compare "React vs Vue vs Svelte" --sources web --cite
octane compare "M4 Pro vs M4 Max vs M4 Ultra" --verify
```

### `octane chain` — multi-step pipeline with variable interpolation
```bash
# Simple two-step chain
octane chain "ask What is NVDA's revenue" "synthesize {prev} into a brief"

# With variables
octane chain \
  "ask What is {ticker}'s market cap" \
  "investigate {prev} growth trajectory" \
  --var ticker=AAPL

# Save chain definition for replay
octane chain \
  "ask {topic}" \
  "investigate {prev} --deep 4" \
  --var topic="AI regulation" \
  --save my_research_chain
```

**`{prev}`** — interpolates the output of the previous step.  
**`--var KEY=VALUE`** — repeatable, sets template variables.  
**`--save NAME`** — saves as `~/.octane/workflows/NAME.workflow.json`.

---

## Research

### Background research (powered by Shadows)
```bash
# Start a background research task that runs every 6 hours
octane research start "AI chip market trends"

# Custom interval and depth
octane research start "quantum computing" --every 4 --depth exhaustive

# Depths: shallow (2 angles), deep (4), exhaustive (8)
```

### Monitor and retrieve
```bash
octane research status                    # All active research tasks
octane research list                      # All tasks with finding counts and last-run
octane research log TASK_ID               # Progress log
octane research log TASK_ID --follow      # Stream in real time
octane research log TASK_ID --lines 100   # Last 100 entries
```

### Reports
```bash
octane research report TASK_ID                 # LLM-synthesized report
octane research report TASK_ID --raw           # Raw findings, no synthesis
octane research report TASK_ID --cycles 3      # Last 3 cycles only
octane research report TASK_ID --since 2026-03-01  # Filter by date
octane research report TASK_ID --export report.md  # Save to file
```

### Search and browse
```bash
octane research recall "transformer attention"  # Search stored findings
octane research recall "NVDA" --limit 20
octane research library                         # Browse all findings grouped by task
```

### Stop
```bash
octane research stop TASK_ID
```

---

## Extraction Pipeline

### Extract from any source
```bash
# YouTube transcript
octane extract run "https://youtube.com/watch?v=VIDEO_ID"

# arXiv paper
octane extract run "2401.12345"
octane extract run "https://arxiv.org/abs/2401.12345"

# PDF file
octane extract run ~/papers/paper.pdf

# EPUB book
octane extract run ~/books/book.epub

# Options
octane extract run URL --quality deep          # deep vs fast vs auto
octane extract run URL --type youtube          # Override source type detection
octane extract run URL --chunks                # Show individual text chunks
octane extract run URL --output ~/notes/extract.md  # Save to file
octane extract run URL --open                  # Open output folder in Finder
```

### Search sources
```bash
octane extract search-youtube "transformer architecture explained" --limit 10
octane extract search-arxiv "chain of thought reasoning" --limit 5
```

### Batch extraction
```bash
# Extract multiple URLs from a file (one URL per line, # comments and blanks skipped)
octane extract batch ~/urls.txt
```

### Pipeline chaining (search → extract)
```bash
# Print bare URLs/IDs to stdout (for piping)
octane extract search-youtube "attention mechanisms" --urls-only
octane extract search-arxiv "chain of thought" --urls-only

# Extract and persist ALL search results in one shot
octane extract search-youtube "attention mechanisms" --extract-all
octane extract search-arxiv "chain of thought" --extract-all

# Pipe pattern: search → save URLs → batch extract
octane extract search-arxiv "reasoning" --urls-only > papers.txt
octane extract batch papers.txt
```

### YouTube login (for age-restricted / premium content)
```bash
octane extract youtube-login
```

---

## Knowledge Recall
// RAHUL TODO — bugs fixed: recall now works across all tables. Try these:
### Search across all stored knowledge
```bash
# Search across all 5 sources (extractions, web pages, findings, files, artifacts)
octane recall search "transformer attention"
octane recall search "NVDA earnings" --limit 20

# Filter by source type
octane recall search "attention" --type youtube    # Only YouTube extractions
octane recall search "attention" --type arxiv      # Only arXiv papers
octane recall search "NVDA" --type finding         # Only research findings
octane recall search "config" --type file          # Only indexed files
octane recall search "analysis" --type artifact    # Only generated artifacts

# Verbose output (show content previews)
octane recall search "transformer" -v
```

### Knowledge base summary
```bash
octane recall stats   # Row counts and total words per source table
```

**Source types:** `youtube | arxiv | pdf | epub | web | finding | file | artifact`

---

## Tracing & Auditing

### Trace — technical event timeline
```bash
octane trace                       # List recent traces (latest first)
octane trace --list                # Same as above
octane trace abc123                # Inspect a specific trace (partial ID OK)
octane trace abc123 --verbose      # Expand URLs and content previews
```

### DAG — dry-run decomposition
```bash
octane dag "Compare React and Vue for a startup"
# Shows how the query would be decomposed into agent tasks (no execution)
```

### Audit — provenance chain for a stored finding
```bash
octane audit FINDING_ID                      # Auto-detect table
octane audit WEB_PAGE_ID --table web_pages   # Specify table
# Tables: auto | web_pages | research_findings_v2 | generated_artifacts
```

---

## Data Store & Files

### Store — browse all Postgres + Redis data
```bash
octane store stats                          # Row counts for every table + Redis memory
octane store pages                          # Browse fetched web pages
octane store pages "NVDA" --limit 50 --verbose  # Search with content preview
octane store findings "transformer"         # Search research findings
octane store artifacts "analysis"           # Search generated artifacts
octane store redis                          # Redis namespace breakdown
```

### Files — index and search local files
```bash
octane files add ~/projects/my-app          # Index folder recursively
octane files add ~/notes/report.md --project research  # Tag with project
octane files list                           # List indexed files
octane files list --project research        # Filter by project
octane files search "authentication flow"   # Semantic search
octane files stats                          # Indexing statistics
octane files reindex ~/projects/my-app/src  # Force re-index
```

### Projects
```bash
octane project list
octane project create "ML Research" --description "papers and experiments"
octane project show "ML Research"
octane project archive "ML Research"        # Archive
octane project archive "ML Research" --restore  # Unarchive
```

---


```bash
octane watch start AAPL                   # Monitor every 1 hour
octane watch start BTC --every 0.5        # Every 30 minutes
octane watch status                       # Show running monitors
octane watch latest AAPL                  # Latest stored quote
octane watch cancel AAPL                  # Stop monitoring AAPL
octane watch stop                         # Stop the background worker
```

---

## Portfolio Management

```bash
# Import broker CSV
octane portfolio import ~/exports/schwab.csv --broker Schwab
octane portfolio import positions.csv --broker Fidelity --account IRA --dry-run

# View positions
octane portfolio show
octane portfolio show --broker Schwab --prices  # With live prices (yfinance)

# Analysis
octane portfolio analyze                  # LLM-based analysis
octane portfolio analyze --deep           # Multi-agent deep analysis
octane portfolio risk                     # Concentration, sectors, top holdings
octane portfolio rebalance --target equal # Equal-weight rebalancing
octane portfolio rebalance --target "AAPL:30,GOOGL:30,MSFT:40" --investment 10000
```

### Dividends & Income
// RAHUL TODO — test dividend fetch (needs yfinance + internet)
```bash
octane portfolio dividends AAPL MSFT GOOGL          # Show dividend info (yield, schedule, income)
octane portfolio dividends AAPL MSFT --save          # Fetch and save dividend data to Postgres
```

### Tax Lots (FIFO/LIFO)
// RAHUL TODO — add a few lots then try lots-sell with FIFO vs LIFO
```bash
octane portfolio lots                                # All open tax lots
octane portfolio lots --ticker AAPL                   # Filter by ticker
octane portfolio lots-add AAPL 100 150.50 2024-03-15  # Add a lot (ticker shares cost date)
octane portfolio lots-add TSLA 50 180.00 2023-06-01 --broker Schwab --account IRA
octane portfolio lots-sell AAPL 30                    # Preview FIFO allocation
octane portfolio lots-sell AAPL 30 --method LIFO      # Preview LIFO allocation
octane portfolio lots-sell AAPL 30 --execute           # Execute and record the sale
```

### Tax-Loss Harvesting
// RAHUL TODO — import positions first, then run harvest
```bash
octane portfolio harvest                             # Scan positions for harvesting opportunities
octane portfolio harvest --min-loss 10.0              # Only show losses >= 10%
# Shows: unrealised loss, loss %, long/short term, wash sale warnings (30-day rule)
```

### Net Worth
// RAHUL TODO — try net-worth with --snapshot and --history
```bash
octane portfolio net-worth                           # Current net worth (equities + crypto)
octane portfolio net-worth --cash 50000              # Include cash balance
octane portfolio net-worth --snapshot                 # Save snapshot to Postgres
octane portfolio net-worth --history                  # Show timeline of past snapshots
```

### XIRR
```bash
octane portfolio xirr AAPL                           # Compute annualised IRR from tax lot cashflows
```

### Crypto
// RAHUL TODO — import a crypto CSV and try --price for live CoinGecko pricing
```bash
# Import from exchange CSV (auto-detects: Coinbase, Kraken, Binance, Gemini)
octane portfolio crypto import ~/exports/coinbase.csv
octane portfolio crypto import kraken.csv --exchange Kraken

# View crypto positions
octane portfolio crypto show                         # All positions
octane portfolio crypto show --exchange Coinbase      # Filter by exchange
octane portfolio crypto show --price                  # With live CoinGecko prices + P&L
```

Supported brokers: `Schwab | Fidelity | Vanguard | IBKR | Robinhood | Webull | ETRADE`  
Supported crypto exchanges: `Coinbase | Kraken | Binance | Gemini | Generic CSV`

---

## Workflow Templates

```bash
# Export any past query as a reusable workflow
octane workflow export TRACE_ID --name my_analysis --desc "weekly stock checkup"

# List saved workflows
octane workflow list

# Run a saved workflow
octane workflow run ~/.octane/workflows/my_analysis.workflow.json
octane workflow run my_analysis.workflow.json --query "AAPL vs TSLA" --verbose
octane workflow run template.json --var ticker=GOOGL --var period=Q1
```

---

## Model Management

```bash
octane model info                              # Currently loaded model + config
octane model reload-parser                     # Reload with default qwen3 parser
octane model reload-parser --parser harmony    # Reload with specific parser
```

---

## Daemon

```bash
octane daemon start                     # Start (auto topology)
octane daemon start --foreground        # Foreground mode (for debugging)
octane daemon start --topology power    # Topologies: auto | compact | balanced | power
octane daemon stop
octane daemon status                    # PID, uptime, queue depth, connections
octane daemon drain                     # Graceful shutdown (finish running tasks)
octane daemon watch                     # Live dashboard: queue table + log stream
octane daemon watch --interval 0.5 --log-lines 30  # Custom refresh
```

---

## Database

```bash
octane db migrate       # Apply pending schema migrations (idempotent)
octane db status        # Migration versions + row counts per table
octane db reset --yes   # ⚠️ DEV ONLY: drop all tables and re-create
```

---

## Preferences

```bash
octane pref show                                   # All preferences
octane pref set verbosity concise                  # concise | detailed
octane pref set expertise advanced                 # beginner | intermediate | advanced
octane pref set response_style bullets             # prose | bullets | code-first
octane pref set domains "finance, machine learning"
octane pref reset verbosity                        # Reset one
octane pref reset --yes                            # Reset all
```

---

## Security: Vault & Airgap

### Vault — Touch ID-protected encrypted secrets
```bash
octane vault create finance            # Create vault (AES key in Keychain)
octane vault status                    # List all vaults
octane vault write finance api_key sk-abc123   # Store secret (Touch ID)
octane vault read finance api_key              # Retrieve (Touch ID)
octane vault list-keys finance                 # List keys (Touch ID)
octane vault destroy finance --yes             # Permanently delete
```

Vault names: `finance | health | research | code`

### Airgap — network kill-switch
```bash
octane airgap on --reason "sensitive analysis"  # Block all outbound
octane airgap off                               # Restore network
octane airgap status                            # Current state
```

---

## Mission Control Web UI

```bash
octane ui start                          # Start on port 44480 (background)
octane ui start --foreground --dev       # Foreground + auto-reload
octane ui start --port 8080 --host 0.0.0.0
octane ui stop
octane ui status

# Auto-kills stale processes on occupied port
# Access at http://localhost:44480 or http://octane.local:44480
# Dev frontend at http://localhost:5173 (cd octane/ui-frontend && npm run dev)
```

---

## Chaining & Pipelining

### Shell pipelining (combine with Unix tools)
```bash
# Ask and pipe to clipboard
octane ask "Summarize NVDA Q4 earnings" | pbcopy

# Ask and save to file
octane ask "Write a Python web scraper" > scraper.py

# Ask with recall, then investigate deeper
octane ask "What stocks did I research?" --recall | xargs -I{} octane investigate "{}"

# Compare, then extract a paper mentioned in the output
octane compare "GPT-4 vs Claude vs Gemini" 2>&1 | tee comparison.md
```

### `octane chain` — built-in multi-step pipelines
```bash
# Step 1 asks, Step 2 uses {prev} (previous output) to investigate
octane chain \
  "ask What are the top 3 AI startups in 2026" \
  "investigate {prev} funding and technology"

# Three-step with variables
octane chain \
  "ask What is {company}'s latest product" \
  "compare {prev} vs competitors" \
  "investigate {prev} market impact --deep 4" \
  --var company="OpenAI"

# Save pipeline for later
octane chain \
  "ask {topic} overview" \
  "investigate {prev} --deep 6 --sources arxiv,web --cite" \
  --save deep_research

# Replay saved pipeline
octane workflow run ~/.octane/workflows/deep_research.workflow.json \
  --var topic="neuromorphic computing"
```

### Combining with `octane research` (long-running)
```bash
# Start background research, then recall findings in a future ask
octane research start "quantum error correction" --every 4 --depth exhaustive

# ... days later ...
octane ask "What have we learned about quantum error correction?" --recall
octane research report TASK_ID --export quantum_report.md
```

---

## Shadows (Perpetual Tasks)

Shadows is the background task engine. Commands that use Shadows:

| Command | What it runs in background |
|---------|---------------------------|
| `octane research start` | Runs research cycles every N hours |
| `octane watch start TICKER` | Polls price/news every N hours |

### How Shadows works:
1. Tasks persist in Redis with schedules
2. Worker process picks them up: `octane research worker` or managed by daemon
3. Tasks survive restarts — they're stored, not in-memory
4. Each cycle writes findings to Postgres

### Manual worker control:
```bash
# The daemon manages workers automatically, but you can also:
octane daemon start                 # Starts worker + daemon
octane research status              # See what Shadows tasks are running
octane watch status                 # See monitor tasks
```

---

## .octane Script Files

`.octane` files are script files that run multiple Octane commands, each in its own process.

### Example: `fetch_news.octane`

```octane
#!/usr/bin/env octane-run
# fetch_news.octane — Morning news research pipeline
# Run: octane run fetch_news.octane

# These run in parallel (each spawns its own process)
@parallel
ask "Top tech news today" --deep
ask "S&P 500 pre-market movers" --deep
investigate "AI industry developments today" --max-dimensions 3
extract run "https://arxiv.org/list/cs.AI/recent" --output ~/notes/arxiv_today.md

# These run after the parallel block completes
@sequential
research recall "market analysis" --limit 5
chain "ask Summarize today's findings from {date}" --var date=$(date +%Y-%m-%d)
```

### Script syntax:
```
# Comment lines start with #
@parallel    — subsequent commands run concurrently
@sequential  — subsequent commands run one after another (default)
@split       — in UI terminal, split view to show each command's output

# Variables
$DATE        — current date (YYYY-MM-DD)  
$HOME        — home directory
${VAR}       — environment variable
{prev}       — output of previous command (in @sequential blocks)
```

### Running .octane scripts:
```bash
# From terminal
chmod +x fetch_news.octane
./fetch_news.octane

# Or explicitly
octane run fetch_news.octane

# From Mission Control web terminal — @split directives create
# split panes so you see all parallel commands simultaneously
```

### More examples:

**`weekly_report.octane`**
```octane
#!/usr/bin/env octane-run
# Weekly portfolio + research summary

@sequential
portfolio show --prices
portfolio risk
research report TASK_ID --cycles 1 --export ~/reports/weekly_${DATE}.md
ask "Given this portfolio and research, what should I focus on?" --recall
```

**`deep_dive.octane`**
```octane
#!/usr/bin/env octane-run
# Deep research on a topic passed as $1

@parallel
investigate "$1" --deep 8 --sources arxiv,web --cite --verify
extract search-youtube "$1" --limit 5
extract search-arxiv "$1" --limit 5

@sequential
research start "$1" --depth exhaustive --every 12
```

Usage: `./deep_dive.octane "transformer architecture improvements"`

---

## Analytics Dashboard

// RAHUL TODO — run octane stats to see your knowledge base overview
```bash
octane stats   # Personal analytics: row counts, words, extraction breakdown,
               # trace files, local mirror files with disk usage
```

Shows:
- Row counts and total words for 6 Postgres tables
- Extraction breakdown by source type (youtube, arxiv, pdf, epub, web)
- Trace file count from `~/.octane/traces/`
- Local extraction mirror file count and size from `~/.octane/extractions/`

---

## Quick Reference Card

| Want to... | Command |
|------------|---------|
| Quick question | `octane ask "..."` |
| Research deep | `octane ask "..." --deep` |
| Use stored data only | `octane ask "..." --recall` |
| Multi-turn chat | `octane chat` |
| Compare things | `octane compare "A vs B vs C"` |
| Deep investigation | `octane investigate "topic" --deep 8` |
| Multi-step pipeline | `octane chain "step1" "step2" --var key=val` |
| Background research | `octane research start "topic" --every 6` |
| Check research | `octane research report TASK_ID` |
| Extract YouTube | `octane extract run "URL"` |
| Extract arXiv | `octane extract run "2401.12345"` |
| Monitor a stock | `octane watch start AAPL` |
| See what's stored | `octane store stats` |
| Search stored pages | `octane store pages "query"` |
| Check traces | `octane trace` |
| Inspect a trace | `octane trace TRACE_ID -v` |
| Web UI | `octane ui start --foreground --dev` |
| Import portfolio | `octane portfolio import file.csv --broker Schwab` |
| System health | `octane health` |
| Set preferences | `octane pref set verbosity concise` |
| Secure secrets | `octane vault write finance key value` |
| Go offline | `octane airgap on` |
| Search all knowledge | `octane recall search "query"` |
| Knowledge stats | `octane recall stats` |
| Analytics dashboard | `octane stats` |
| Batch extract URLs | `octane extract batch urls.txt` |
| Pipe search results | `octane extract search-arxiv "q" --urls-only` |
| Extract all results | `octane extract search-youtube "q" --extract-all` |

---

*Last updated: Session 36, 2026-03-30*
