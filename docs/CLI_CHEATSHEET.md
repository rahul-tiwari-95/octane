# Octane CLI Cheatsheet

All ~100 commands with examples. Run any command with `--help` for full options.

---

## Core Commands

```bash
octane health                          # System status (Redis, Postgres, BodegaOS)
octane version                         # Show Octane version
octane agents                          # List registered agents and status
```

---

## Ask & Chat

```bash
# Single-shot query
octane ask "What is NVDA's P/E ratio?"

# Deep research with N dimensions (1-16) and citations
octane ask "explain AI chip export restrictions" --deep 4 --cite

# Deep research with verification pass
octane ask "is RAG still relevant?" --deep 6 --cite --verify

# Quiet mode (no progress indicators, stdout clean for piping)
octane ask "summary of transformer architecture" --quiet

# Interactive conversation with full ChatEngine
octane chat

# Non-streaming (waits for full response)
octane chat --no-stream
```

---

## Research — Power Commands

```bash
# Investigate: decompose, research in parallel, synthesize, cite
octane investigate "future of autonomous vehicles" --deep 8 --cite

# Compare two things across N dimensions
octane compare "PyTorch vs JAX" --deep 4

# Chain: pipe output of one research into the next
octane chain "explain attention mechanisms" "how does it apply to vision models?"

# Check status of a running research task
octane research status <task-id>

# Get full report for a completed task
octane research report <task-id>

# List recent research tasks
octane research list
octane research list --limit 20
```

---

## Search

```bash
# Web search
octane search web "NVDA earnings Q4 2025"
octane search web "NVDA earnings" --json            # JSON output for piping
octane search web "NVDA earnings" --urls-only       # URLs only

# News
octane search news "AI regulation Europe"

# arXiv papers
octane search arxiv "speculative decoding"
octane search arxiv "RAG retrieval" --limit 5

# YouTube
octane search youtube "Andrej Karpathy transformers"
```

---

## Extract

```bash
# Extract from a URL
octane extract url "https://example.com/article"

# Extract from stdin (pipe from search)
octane search web "NVDA" --json | octane extract stdin --json

# Extract with JSON output
octane extract url "https://arxiv.org/abs/1706.03762" --json
```

---

## Synthesize

```bash
# Synthesize from piped extraction
octane search web "transformer architecture" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin

# Synthesize with a specific query focus
octane synthesize run --stdin --query "what are the key innovations?"

# Use a template
octane synthesize run --stdin --template briefing
octane synthesize run --stdin --template compare

# No streaming (wait for full output)
octane synthesize run --stdin --no-stream
```

---

## Full Pipe: Search → Extract → Synthesize

```bash
octane search arxiv "RAG techniques 2025" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "latest innovations in RAG"
```

---

## Recall — Knowledge Base Search

```bash
# Unified search across all stored knowledge
octane recall search "transformer attention"

# Search with more results
octane recall search "NVDA" --limit 20

# Search a specific source type
octane recall search "speculative decoding" --type arxiv

# System stats: how much has been researched
octane stats
octane stats --json
```

---

## Portfolio

```bash
# Import positions from Schwab CSV export
octane portfolio import positions.csv

# Show current holdings
octane portfolio show

# Risk analysis
octane portfolio risk

# Dividend analysis
octane portfolio dividends

# Tax lot analysis
octane portfolio tax-lots

# Full portfolio report (beta — needs BodegaOS running)
octane portfolio report

# Crypto holdings (beta)
octane portfolio crypto
```

---

## Watch — Live Market Monitoring

```bash
# Watch a ticker with live updates
octane watch NVDA
octane watch AAPL --interval 30

# Watch multiple tickers
octane watch NVDA AAPL MSFT

# Watch with alerts
octane watch NVDA --alert-above 150 --alert-below 120
```

---

## Workflow & DAG

```bash
# List available workflows
octane workflow list

# Run a workflow
octane workflow run research-and-brief --query "NVDA competitive position"

# DAG commands
octane dag list
octane dag run <dag-name>
octane dag status <run-id>
```

---

## Daemon

```bash
# Start Octane daemon
octane daemon start

# Check daemon status
octane daemon status

# Stop daemon
octane daemon stop

# View daemon logs
octane daemon logs
octane daemon logs --follow
```

---

## Mission Control UI

```bash
# Start the Mission Control web UI at localhost:44480
octane ui start

# Start on a specific port
octane ui start --port 8080

# Stop UI server
octane ui stop
```

---

## Security

```bash
# Vault — Touch ID protected secret storage
octane vault set finance api_key
octane vault get finance api_key
octane vault list
octane vault delete finance api_key

# Air-gap — kill all external network access
octane airgap on
octane airgap off
octane airgap status

# Audit log
octane audit log
octane audit log --limit 50
octane audit export > audit.json
```

---

## Files & Projects

```bash
# File inbox (auto-extracts documents dropped into ~/Octane/inbox/)
octane files list
octane files show <file-id>
octane files extract <file-path>

# Projects
octane project create "AI Chip Research"
octane project list
octane project switch "AI Chip Research"
octane project status
```

---

## System & Models

```bash
# System stats
octane power                    # power usage and thermal state
octane sysstat                  # CPU, RAM, GPU utilization

# Model management (via BodegaOS Sensors API)
octane model list               # list loaded models
octane model info <model-name>  # model details
```

---

## Preferences

```bash
octane pref list
octane pref set research.default_depth 6
octane pref set research.always_cite true
octane pref get research.default_depth
```

---

## Trace & Debug

```bash
# View trace of last command
octane trace last

# View trace by ID
octane trace show <trace-id>

# List recent traces
octane trace list
```

---

## Database

```bash
# Run pending migrations
octane db migrate

# Database status
octane db status

# Backup database
octane db backup

# Store key/value
octane store set <key> <value>
octane store get <key>
octane store list
```
