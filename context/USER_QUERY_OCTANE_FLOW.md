# USER_QUERY_OCTANE_FLOW.md
# Project Octane — Detailed Query Flow Reference
# This document traces a real query through every component of the system.
# Use this to understand how data moves through Octane end-to-end.
# Last updated: 2026-02-16

---

## How to Read This Document

This traces a single user query from the moment it's typed to the final output. Every component that touches the data is shown, every Synapse event is logged, every decision is documented. If you're building a new agent or debugging a flow, this is your reference for "what should happen when."

---

## The Query

```
$ octane ask "What happened with NVIDIA today and write me a script to chart their stock performance this month"
```

**Why this query is a good reference:** It exercises multiple agents (Web + Code), requires parallel execution, triggers HIL (code execution is high-risk), creates multiple checkpoints, and demonstrates the self-healing Code Agent loop.

**User Profile (from P&L Agent):**
```json
{
  "user_id": "default",
  "expertise_level": "advanced",
  "preferred_verbosity": "concise",
  "domain_interests": ["finance", "coding"],
  "hil_level": "balanced",
  "auto_approve": {
    "web.search": true,
    "web.finance": true,
    "web.news": true,
    "code.execute": false
  }
}
```

---

## Step 1: User Input → CLI

```
Component:  octane/main.py (Typer CLI)
Action:     Receives raw query string from terminal
Creates:    correlation_id = "evt_7f3a"
            session_id = "sess_01"
Passes to:  OSA.Orchestrator.run(query, session_id)
```

The CLI is thin. It parses the command, generates IDs, and hands off to OSA. No business logic lives here.

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "query_received",
  "source": "cli",
  "target": "osa",
  "payload_summary": "What happened with NVIDIA today and write me a script...",
  "timestamp": "2026-02-16T10:00:00.000Z"
}
```

---

## Step 2: OSA — Guard (Parallel)

```
Component:  octane/osa/guard.py
Action:     Safety check on raw input — runs in PARALLEL with next steps
Model:      None (regex + rules). Phase 2+ adds small model semantic check.
Duration:   ~50ms
```

Guard spawns as a background task so it doesn't block the pipeline:
```python
guard_task = asyncio.create_task(self.guard.check(query))
```

**Checks performed:**
| Check | Result | Details |
|-------|--------|---------|
| Input length | ✅ PASS | 89 chars, under 10,000 limit |
| SQL injection patterns | ✅ PASS | No DROP, DELETE, UNION, etc. |
| Prompt injection patterns | ✅ PASS | No "ignore previous instructions" |
| Blacklisted terms | ✅ PASS | No blocked keywords |

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "guard_check",
  "source": "osa.guard",
  "target": "osa",
  "success": true,
  "latency_ms": 48,
  "payload_summary": "All checks passed: length, injection, blacklist"
}
```

---

## Step 3: OSA — P&L Profile Fetch

```
Component:  octane/agents/pnl/agent.py → preference_manager.py
Action:     Retrieve current user profile from Postgres
Model:      None (database query)
Duration:   ~10ms
```

OSA needs the profile BEFORE decomposition because the Decomposer uses it to tailor the plan (e.g., "user is advanced, prefers concise output, interested in finance").

```sql
SELECT key, value FROM user_preferences WHERE user_id = 'default';
```

**Returns:**
```json
{
  "expertise_level": "advanced",
  "preferred_verbosity": "concise",
  "domain_interests": ["finance", "coding"],
  "hil_level": "balanced",
  "auto_approve": { "web.search": true, "code.execute": false }
}
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "pnl_profile_fetch",
  "source": "osa",
  "target": "pnl",
  "latency_ms": 8,
  "success": true
}
```

---

## Step 4: OSA — Decomposer

```
Component:  octane/osa/decomposer.py
Action:     Analyze query → produce TaskDAG
Model:      BIG MODEL (30B on 64GB, 8B on 16GB)
Duration:   ~2-3 seconds
```

This is the most important LLM call in the pipeline. The Decomposer receives the raw query + user profile and reasons about what agents, sub-agents, and data sources are needed.

**LLM Prompt (simplified):**
```
You are the Octane Decomposer. Given a user query and their profile,
produce a task DAG that specifies which agents to call, in what order,
and what can run in parallel.

Available agents: web (sub: news, finance, search), code (sub: planner,
writer, executor, debugger, validator), memory (sub: read, write),
sysstat, pnl

User query: "What happened with NVIDIA today and write me a script
to chart their stock performance this month"

User profile: advanced, concise, interests: finance + coding

Output format: TaskDAG JSON
```

**Decomposer Reasoning (captured in Synapse):**
```
"This query has two distinct parts:
 1. 'What happened with NVIDIA today' → needs news + finance data
 2. 'Write me a script to chart stock performance this month'
    → needs timeseries data + code generation + execution

 Part 1 is informational (read-only, low risk).
 Part 2 involves code execution (high risk, needs HIL approval).

 Dependencies:
 - t1 (news) and t2 (market data) are independent → parallel
 - t3 (timeseries) is independent of t1/t2 → also parallel
 - t4 (code generation) depends on t3 (needs the data to chart)
 - t5 (synthesis) depends on t1, t2, and t4

 Parallel group: [t1, t2, t3]
 Sequential: t3 → t4 → t5 (but t5 also waits for t1, t2)"
```

**Output TaskDAG:**
```json
{
  "correlation_id": "evt_7f3a",
  "nodes": [
    {
      "id": "t1",
      "agent": "web",
      "sub_agent": "news",
      "input_query": "NVIDIA news today",
      "instruction": "",
      "depends_on": []
    },
    {
      "id": "t2",
      "agent": "web",
      "sub_agent": "finance",
      "input_query": "NVDA market data today",
      "instruction": "",
      "depends_on": []
    },
    {
      "id": "t3",
      "agent": "web",
      "sub_agent": "finance",
      "input_query": "NVDA timeseries 1 month daily",
      "instruction": "Return raw OHLCV data as JSON array",
      "depends_on": []
    },
    {
      "id": "t4",
      "agent": "code",
      "sub_agent": "full_pipeline",
      "input_query": "Write Python script to chart NVDA stock performance this month using the timeseries data",
      "instruction": "Use matplotlib. Save chart as nvda_chart.png. Include volume as secondary axis.",
      "depends_on": ["t3"]
    },
    {
      "id": "t5",
      "agent": "osa",
      "sub_agent": "synthesize",
      "input_query": "Combine news, market data, and chart into concise briefing for advanced user",
      "instruction": "Lead with price action, explain with news context, reference chart artifact",
      "depends_on": ["t1", "t2", "t4"]
    }
  ],
  "parallel_groups": [["t1", "t2", "t3"]],
  "reasoning": "Two-part query: informational (news+finance) + code generation (chart). Parallel data fetch, sequential code + synthesis."
}
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "decomposition",
  "source": "osa.decomposer",
  "target": "osa.router",
  "payload_summary": "5 tasks: 3 parallel [web.news, web.finance, web.finance], 1 sequential [code], 1 synthesis [osa]",
  "reasoning": "Two-part query: informational + code generation...",
  "model_used": "SRSWTI/axe-turbo-31b",
  "tokens_in": 340,
  "tokens_out": 280,
  "latency_ms": 2800
}
```

---

## Step 5: OSA — Checkpoint #1 (Plan)

```
Component:  octane/osa/checkpoint_manager.py
Action:     Snapshot the pipeline state after decomposition
Model:      None (pure data operation)
Duration:   ~5ms
```

This is the "fresh start" checkpoint. If anything goes catastrophically wrong later, we can always revert to this point and re-plan.

**Checkpoint:**
```json
{
  "id": "cp_001",
  "correlation_id": "evt_7f3a",
  "checkpoint_type": "plan",
  "timestamp": "2026-02-16T10:00:03.000Z",
  "dag": "<the TaskDAG above>",
  "completed_tasks": [],
  "pending_tasks": ["t1", "t2", "t3", "t4", "t5"],
  "accumulated_results": {},
  "decisions": [],
  "approved_decisions": []
}
```

---

## Step 6: OSA — Policy Engine + Decision Ledger

```
Component:  octane/osa/policy.py
Action:     Assess risk level and confidence for each task
Model:      None (deterministic rules)
Duration:   ~1ms
```

Policy Engine walks through each TaskNode and assigns a risk level based on deterministic rules:

**Rules applied:**
```
- web.* (any sub-agent) → LOW risk (read-only API calls)
- code.full_pipeline → HIGH risk (involves code execution)
- osa.synthesize → LOW risk (read-only, internal)
- confidence = 1.0 for exact API matches, lower for ambiguous routing
```

**Decision Ledger:**

| # | Task | Risk | Confidence | Status | Reason |
|---|------|------|------------|--------|--------|
| D1 | t1: Web.News "NVIDIA news today" | LOW | 0.95 | `auto_approved` | Read-only, user allows web.news |
| D2 | t2: Web.Finance "NVDA market data" | LOW | 0.95 | `auto_approved` | Read-only, user allows web.finance |
| D3 | t3: Web.Finance "NVDA timeseries" | LOW | 0.93 | `auto_approved` | Read-only, user allows web.finance |
| D4 | t4: Code.FullPipeline "chart script" | HIGH | 0.80 | `pending` | Code execution. User has `code.execute: false`. **NEEDS HUMAN REVIEW.** |
| D5 | t5: OSA.Synthesize "combine results" | LOW | 0.90 | `auto_approved` | Internal synthesis, read-only |

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "policy_assessment",
  "source": "osa.policy",
  "target": "osa.hil",
  "payload_summary": "5 decisions: 4 auto-approved, 1 pending (HIGH risk code execution)",
  "latency_ms": 1
}
```

---

## Step 7: OSA — HIL Manager

```
Component:  octane/osa/hil_manager.py
Action:     Present pending decisions to user, collect response
Model:      None (CLI interaction via Rich)
Duration:   ~5 seconds (waiting for human input)
```

HIL Manager is smart about timing. Decision #4 (code execution) depends on t3 (timeseries data), which hasn't been fetched yet. So HIL Manager makes an optimization:

> "I'll dispatch the parallel group [t1, t2, t3] immediately (all auto-approved)
> and present Decision #4 to the user WHILE the data is being fetched.
> This overlaps human thinking time with network I/O."

**What the user sees in terminal:**

```
┌──────────────────────────────────────────────────────────────────┐
│  🔵  OCTANE — Processing: "What happened with NVIDIA today..."   │
│                                                                   │
│  ✅ 4 decisions auto-approved:                                    │
│     • Search NVIDIA news (News API)                               │
│     • Fetch NVDA market data (Finance API)                        │
│     • Fetch NVDA 1-month timeseries (Finance API)                 │
│     • Synthesize final briefing                                   │
│                                                                   │
│  ⚠️  1 decision needs your review:                                │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  DECISION #4 [HIGH RISK]                                   │   │
│  │                                                             │   │
│  │  Action:  Generate Python script + execute in sandbox       │   │
│  │  Purpose: Chart NVDA stock performance (last month)         │   │
│  │  Method:  matplotlib line chart with volume bars            │   │
│  │  Output:  nvda_chart.png saved to sandbox                   │   │
│  │                                                             │   │
│  │  Packages to install: matplotlib                            │   │
│  │  Sandbox: isolated venv in /tmp/octane-sandboxes/           │   │
│  │  Timeout: 30 seconds                                        │   │
│  │  Network: disabled after package install                    │   │
│  │                                                             │   │
│  │  Confidence: 80%                                            │   │
│  │  Why uncertain: "Code generation from natural language       │   │
│  │   may require iteration. Self-healing loop will retry       │   │
│  │   up to 3 times if execution fails."                        │   │
│  │                                                             │   │
│  │  [1] ✅ Approve   [2] ✏️ Modify   [3] ❌ Skip (data only)   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ⏳ Meanwhile: fetching news and financial data in background...  │
└──────────────────────────────────────────────────────────────────┘

Your choice [1/2/3]: 1
```

**User presses: 1 (Approve)**

Decision #4 updated:
```json
{
  "id": "D4",
  "status": "human_approved",
  "human_feedback": "",
  "timestamp": "2026-02-16T10:00:08.000Z"
}
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "hil_review_complete",
  "source": "osa.hil",
  "target": "osa.router",
  "payload_summary": "Decision D4 approved by user. Pipeline continues.",
  "latency_ms": 5200
}
```

---

## Step 8: Parallel Agent Execution (t1, t2, t3)

These were dispatched while HIL was waiting. All three run concurrently via `asyncio.gather()`.

### t1: Web Agent → News

```
Component:  octane/agents/web/agent.py → query_strategist.py → fetcher.py → synthesizer.py
Duration:   ~1200ms total
```

**Sub-agent: Query Strategist (6B grunt model)**
```
Input:  "NVIDIA news today"
Output: [
  { "api": "news", "endpoint": "/api/v1/news/search", "params": { "q": "NVIDIA", "period": "1d" } },
  { "api": "news", "endpoint": "/api/v1/news/search", "params": { "q": "NVDA earnings semiconductor", "period": "1d" } }
]
```
*Generates two search variations to cast a wider net.*

**Sub-agent: Fetcher (no model, pure HTTP)**
```
Request 1: GET http://localhost:8032/api/v1/news/search?q=NVIDIA&period=1d
Response:  5 articles (export restrictions, GTC keynote, analyst reports)

Request 2: GET http://localhost:8032/api/v1/news/search?q=NVDA+earnings+semiconductor&period=1d
Response:  3 articles (sector analysis, Morgan Stanley upgrade, supply chain)

Combined:  8 articles (deduplicated by URL)
```

**Sub-agent: Synthesizer (14B worker model)**
```
Input:  8 raw articles with titles, sources, dates, summaries
Prompt: "Synthesize these 8 news articles about NVIDIA into 3 key
         stories. Be concise. Include source attribution."

Output: "Three key stories today: (1) Renewed US-China export
         restriction talks weighing on chip sector — Reuters reports
         new restrictions under discussion. (2) Jensen Huang GTC
         keynote announced for March — potential product reveals
         expected. (3) Morgan Stanley upgrades NVDA citing sustained
         datacenter demand through 2027."
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "agent_egress",
  "source": "web.news",
  "target": "osa",
  "latency_ms": 1180,
  "tokens_in": 1200,
  "tokens_out": 85,
  "model_used": "SRSWTI/bodega-raptor-8b-mxfp4",
  "success": true,
  "payload_summary": "3 key NVIDIA stories synthesized from 8 articles"
}
```

---

### t2: Web Agent → Finance (Market Data)

```
Component:  octane/agents/web/agent.py → query_strategist.py → fetcher.py → synthesizer.py
Duration:   ~400ms total
```

**Sub-agent: Query Strategist**
```
Input:  "NVDA market data today"
Output: [{ "api": "finance", "endpoint": "/api/v1/finance/market/NVDA", "params": {} }]
```
*Simple query — one API call is sufficient.*

**Sub-agent: Fetcher**
```
Request: GET http://localhost:8030/api/v1/finance/market/NVDA
Response: {
  "ticker": "NVDA",
  "price": 142.50,
  "change": -4.72,
  "change_percent": -3.2,
  "volume": 45200000,
  "avg_volume": 22100000,
  "market_cap": "3.48T",
  "pe_ratio": 62.1,
  "day_high": 147.80,
  "day_low": 141.20,
  "prev_close": 147.22
}
```

**Sub-agent: Synthesizer**
```
Passes through structured data — no summarization needed for numeric data.
Adds: volume_ratio = 45.2M / 22.1M = 2.05x ("2x average volume")
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "agent_egress",
  "source": "web.finance",
  "target": "osa",
  "latency_ms": 380,
  "success": true,
  "payload_summary": "NVDA $142.50 (-3.2%) volume 2x avg"
}
```

---

### t3: Web Agent → Finance (Timeseries)

```
Component:  octane/agents/web/agent.py → query_strategist.py → fetcher.py → synthesizer.py
Duration:   ~600ms total
```

**Sub-agent: Query Strategist**
```
Input:  "NVDA timeseries 1 month daily"
Output: [{
  "api": "finance",
  "endpoint": "/api/v1/finance/timeseries/NVDA",
  "params": { "period": "1mo", "interval": "1d" }
}]
```

**Sub-agent: Fetcher**
```
Request: GET http://localhost:8030/api/v1/finance/timeseries/NVDA?period=1mo&interval=1d
Response: [
  { "date": "2026-01-16", "open": 148.20, "high": 150.10, "low": 147.30, "close": 149.80, "volume": 21500000 },
  { "date": "2026-01-17", "open": 149.50, "high": 152.40, "low": 149.00, "close": 151.20, "volume": 24300000 },
  ... (22 trading days total)
  { "date": "2026-02-16", "open": 145.30, "high": 147.80, "low": 141.20, "close": 142.50, "volume": 45200000 }
]
```

**Sub-agent: Synthesizer**
```
Passes through raw timeseries — Code Agent needs the raw OHLCV array.
Does NOT summarize. Adds metadata: { "data_points": 22, "date_range": "2026-01-16 to 2026-02-16" }
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "agent_egress",
  "source": "web.finance.timeseries",
  "target": "osa",
  "latency_ms": 580,
  "success": true,
  "payload_summary": "22 days OHLCV data for NVDA (2026-01-16 to 2026-02-16)"
}
```

---

**Parallel group complete.** All three tasks finished while the user was reviewing HIL Decision #4. OSA now has results for t1, t2, t3 in `accumulated_results`.

---

## Step 9: OSA — Checkpoint #2 (Post-Parallel, Pre-Code)

```
Component:  octane/osa/checkpoint_manager.py
Action:     Snapshot state after all parallel tasks complete, before high-risk code execution
Duration:   ~5ms
```

**Checkpoint:**
```json
{
  "id": "cp_002",
  "correlation_id": "evt_7f3a",
  "checkpoint_type": "post_execution",
  "timestamp": "2026-02-16T10:00:09.000Z",
  "completed_tasks": ["t1", "t2", "t3"],
  "pending_tasks": ["t4", "t5"],
  "accumulated_results": {
    "t1": { "news_synthesis": "Three key stories today: (1) Renewed US-China..." },
    "t2": { "ticker": "NVDA", "price": 142.50, "change_percent": -3.2, "..." : "..." },
    "t3": [ { "date": "2026-01-16", "close": 149.80, "..." : "..." }, "...22 records..." ]
  },
  "decisions": ["D1: auto", "D2: auto", "D3: auto", "D4: human_approved", "D5: auto"]
}
```

**Why this checkpoint matters:** If Code Agent fails catastrophically (crashes, infinite loop, corrupts sandbox), we revert here. All web data is preserved. We could skip code generation entirely and just output the news + finance data, or retry with a different code strategy.

---

## Step 10: Code Agent Execution (t4)

```
Component:  octane/agents/code/agent.py → planner.py → writer.py → executor.py → validator.py
Duration:   ~8500ms total
This is the most complex agent execution in this pipeline.
```

### Sub-agent: Planner (Big Model)

```
Model:     30B brain (or 8B on lower RAM)
Duration:  ~1500ms
```

```
Input:  "Write Python script to chart NVDA stock performance this month"
Context: { timeseries_data: <22 days OHLCV>, instruction: "Use matplotlib. Save as nvda_chart.png. Include volume." }

Output (code specification):
{
  "language": "python",
  "requirements": ["matplotlib"],
  "description": "Line chart of NVDA closing prices with volume bar overlay",
  "approach": [
    "Parse timeseries JSON data",
    "Extract dates, closing prices, and volumes",
    "Create dual-axis plot: line chart (price) + bar chart (volume)",
    "Format x-axis as dates, add labels and title",
    "Save as nvda_chart.png at 150 DPI"
  ],
  "expected_output": "Chart saved as nvda_chart.png",
  "expected_artifacts": ["nvda_chart.png"]
}
```

### Sub-agent: Writer (Code Model)

```
Model:     axe-turbo-1b or bodega-solomon-9b (code-specialized)
Duration:  ~2000ms
```

```python
# Generated code:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime

# Timeseries data injected from Web Agent t3 results
data = json.loads('''[
  {"date": "2026-01-16", "open": 148.20, "high": 150.10, "low": 147.30, "close": 149.80, "volume": 21500000},
  {"date": "2026-01-17", "open": 149.50, "high": 152.40, "low": 149.00, "close": 151.20, "volume": 24300000},
  ...
  {"date": "2026-02-16", "open": 145.30, "high": 147.80, "low": 141.20, "close": 142.50, "volume": 45200000}
]''')

dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
closes = [d['close'] for d in data]
volumes = [d['volume'] for d in data]

fig, ax1 = plt.subplots(figsize=(12, 6))

# Price line
ax1.plot(dates, closes, 'b-', linewidth=2, label='Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)

# Volume bars
ax2 = ax1.twinx()
ax2.bar(dates, [v / 1_000_000 for v in volumes], alpha=0.3, color='gray', width=0.8, label='Volume (M)')
ax2.set_ylabel('Volume (millions)', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

plt.title('NVDA Stock Performance — Last Month')
fig.tight_layout()
plt.savefig('nvda_chart.png', dpi=150, bbox_inches='tight')
print("Chart saved as nvda_chart.png")
```

### Sub-agent: Executor (No Model — Pure Infrastructure)

```
Duration:  ~4500ms (dominated by venv creation + pip install)
```

**Execution sequence:**
```
1. mkdir /tmp/octane-sandboxes/sandbox_evt_7f3a/
2. python -m venv /tmp/octane-sandboxes/sandbox_evt_7f3a/venv
3. source venv/bin/activate
4. pip install matplotlib  (this is the slow part: ~3 seconds)
5. Write script to: /tmp/octane-sandboxes/sandbox_evt_7f3a/chart_nvda.py
6. Execute: timeout 30 python chart_nvda.py
7. Capture results:
   {
     "stdout": "Chart saved as nvda_chart.png",
     "stderr": "",
     "exit_code": 0,
     "artifacts": [
       "/tmp/octane-sandboxes/sandbox_evt_7f3a/nvda_chart.png"
     ],
     "execution_time_ms": 1200
   }
```

### Sub-agent: Validator (Deterministic + Small Model)

```
Duration:  ~200ms
```

**Validation checks:**
| Check | Result | Details |
|-------|--------|---------|
| Exit code == 0 | ✅ PASS | Process completed normally |
| stdout matches expected | ✅ PASS | Contains "Chart saved as nvda_chart.png" |
| Artifact exists | ✅ PASS | nvda_chart.png is 184KB (valid file) |
| No errors in stderr | ✅ PASS | stderr is empty |
| File size sanity | ✅ PASS | 184KB is reasonable for a chart PNG |

```json
{
  "passed": true,
  "checks": {
    "exit_code": true,
    "stdout_match": true,
    "artifact_exists": true,
    "no_stderr": true,
    "file_size_sane": true
  }
}
```

**Validator passed on first attempt — Debugger was NOT invoked.**

### What If Validator Had Failed?

If the script had produced an error (e.g., `ModuleNotFoundError: No module named 'matplotlib.dates'`), the flow would have been:

```
Validator: { passed: false, error: "ModuleNotFoundError" }
    ↓
Debugger (14B worker model):
  Input: { code: "...", stderr: "ModuleNotFoundError...", spec: "..." }
  Analysis: "matplotlib.dates is part of matplotlib. The import is correct
             but the package may not be fully installed. Try: pip install
             matplotlib --force-reinstall"
  Output: { fixed_code: "...", fix_description: "Added explicit import check" }
    ↓
Executor: runs fixed code
    ↓
Validator: re-checks
    ↓
(max 3 retries before giving up)
```

### Code Agent Final Result

```json
{
  "agent_name": "code",
  "success": true,
  "result": {
    "code": "import matplotlib.pyplot as plt...",
    "output": "Chart saved as nvda_chart.png",
    "artifacts": ["/tmp/octane-sandboxes/sandbox_evt_7f3a/nvda_chart.png"],
    "attempts": 1,
    "validation": { "passed": true }
  },
  "latency_ms": 8500,
  "metadata": {
    "model_used": "SRSWTI/axe-turbo-1b",
    "sandbox_path": "/tmp/octane-sandboxes/sandbox_evt_7f3a/",
    "packages_installed": ["matplotlib"]
  }
}
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "agent_egress",
  "source": "code",
  "target": "osa",
  "latency_ms": 8500,
  "success": true,
  "payload_summary": "Chart generated: nvda_chart.png (184KB). 1 attempt, all validations passed.",
  "metadata": { "attempts": 1, "artifacts": ["nvda_chart.png"] }
}
```

---

## Step 11: OSA — Checkpoint #3 (Pre-Synthesis)

```
Component:  octane/osa/checkpoint_manager.py
Action:     Final checkpoint before synthesis — all agent work is done
Duration:   ~5ms
```

**Checkpoint:**
```json
{
  "id": "cp_003",
  "correlation_id": "evt_7f3a",
  "checkpoint_type": "pre_synthesis",
  "timestamp": "2026-02-16T10:00:17.500Z",
  "completed_tasks": ["t1", "t2", "t3", "t4"],
  "pending_tasks": ["t5"],
  "accumulated_results": {
    "t1": { "news_synthesis": "Three key stories..." },
    "t2": { "ticker": "NVDA", "price": 142.50, "..." : "..." },
    "t3": [ "...22 OHLCV records..." ],
    "t4": { "code": "...", "artifacts": ["nvda_chart.png"], "attempts": 1 }
  }
}
```

---

## Step 12: OSA — Evaluator + Final Synthesis (t5)

```
Component:  octane/osa/evaluator.py
Action:     Review all results, synthesize final output
Model:      BIG MODEL (30B on 64GB, 8B on 16GB)
Duration:   ~3 seconds
```

**Evaluator receives:**
- t1 result: News synthesis (3 key stories)
- t2 result: Market data (NVDA $142.50, -3.2%, 2x volume)
- t3 result: Raw timeseries (22 days)
- t4 result: Generated chart + code + validation status
- User profile: advanced, concise

**Evaluator LLM Prompt:**
```
You are the Octane Evaluator. Synthesize these results into a
final response for the user.

User query: "What happened with NVIDIA today and write me a
script to chart their stock performance this month"

User profile: advanced engineer, prefers concise output.

Available data:
- News: [t1 synthesis]
- Market: [t2 structured data]
- Chart: Successfully generated at nvda_chart.png, first attempt
- Timeseries: 22 days of data, range $138-$152

Instructions: Lead with price action, explain with news, reference chart.
Keep it concise for advanced user.
```

**Evaluator Quality Gate (before outputting):**
| Check | Result |
|-------|--------|
| Addresses both parts of query (news + chart)? | ✅ Yes |
| Consistent with user's verbosity preference? | ✅ Concise |
| All factual claims grounded in retrieved data? | ✅ Price, volume, news from APIs |
| Chart artifact referenced and accessible? | ✅ Path included |
| No hallucinated information? | ✅ All data from t1/t2/t3 |

**Final Output:**
```
NVDA $142.50 (-3.2%) on 2x average volume.

Key drivers today:
• Renewed US-China export restriction talks weighing on chip
  sector broadly — NVDA down alongside AMD and AVGO
• Partially offset by Morgan Stanley upgrade citing sustained
  datacenter demand into 2027
• Jensen Huang GTC keynote announced for next month — potential
  catalyst

Chart generated → nvda_chart.png
Monthly trend shows NVDA range-bound $138-$152. Today's drop
brings it to the lower end of the range on elevated volume.

📊 View: /tmp/octane-sandboxes/sandbox_evt_7f3a/nvda_chart.png
📝 Code: /tmp/octane-sandboxes/sandbox_evt_7f3a/chart_nvda.py
```

**Synapse Event:**
```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "synthesis_complete",
  "source": "osa.evaluator",
  "target": "cli",
  "latency_ms": 2900,
  "tokens_in": 1800,
  "tokens_out": 120,
  "model_used": "SRSWTI/axe-turbo-31b",
  "success": true
}
```

---

## Step 13: Output to User + Post-Processing

```
Component:  octane/main.py (CLI) + P&L Agent + Memory Agent
Action:     Display output, record feedback signals, persist useful data
```

### CLI Display

The output is rendered via Rich with formatting:
- Price data in colored text (red for negative)
- Bullet points for key drivers
- File paths as clickable links (in supported terminals)
- Chart preview hint

### P&L Agent — Feedback Recording (Async, Non-Blocking)

```
Component:  octane/agents/pnl/feedback_learner.py
```

P&L records this interaction for future learning:
```json
{
  "correlation_id": "evt_7f3a",
  "query_type": "finance + code_generation",
  "agents_used": ["web.news", "web.finance", "code"],
  "hil_decisions": { "code.execute": "human_approved" },
  "timestamp": "2026-02-16T10:00:20.500Z"
}
```

P&L then waits for feedback signals:
- **Explicit:** User types `octane feedback 👍` or `octane feedback 👎`
- **Implicit:** If user asks a follow-up question within 60 seconds, that's a positive engagement signal
- **Implicit:** If user runs `octane ask` with a completely different topic, that's neutral (query was answered)

Over time, P&L learns:
- "User frequently asks finance + code combination queries"
- "User always approves code execution for chart generation"
- → Eventually auto-approves code.execute for chart-type tasks

### Memory Agent — Selective Persistence (Async, Non-Blocking)

```
Component:  octane/agents/memory/writer.py
```

Memory Writer evaluates what's worth storing:

| Data | Tier | TTL | Reason |
|------|------|-----|--------|
| Full synthesis output | Hot (Redis) | 4 hours | Enables follow-up: "What about AMD?" |
| Session context (query + agents used) | Hot (Redis) | 4 hours | Enables: "Compare that to last week" |
| NVDA market snapshot | Warm (Postgres) | Permanent | Longitudinal tracking: "NVDA price history" |
| Chart artifact path | Warm (Postgres) | 7 days | Reference: "Show me that chart again" |
| Raw timeseries data | NOT stored | — | Can be re-fetched cheaply; not worth storage |
| News articles | NOT stored | — | Ephemeral; tomorrow's news will differ |

```sql
-- Hot cache write
SET octane:session:sess_01:latest_nvda <serialized output> EX 14400

-- Warm tier write
INSERT INTO memory_chunks (content, metadata, tier, created_at)
VALUES ('NVDA snapshot 2026-02-16: $142.50 -3.2%',
        '{"ticker": "NVDA", "price": 142.50, "type": "market_snapshot"}',
        'warm', NOW());
```

---

## Step 14: Pipeline Complete — Synapse Final Event

```json
{
  "correlation_id": "evt_7f3a",
  "event_type": "pipeline_complete",
  "source": "osa",
  "target": "cli",
  "timestamp": "2026-02-16T10:00:20.500Z",

  "total_latency_ms": 20500,
  "total_latency_breakdown": {
    "guard": 48,
    "pnl_fetch": 8,
    "decomposer": 2800,
    "checkpoint_1": 5,
    "policy_engine": 1,
    "hil_review": 5200,
    "web_parallel": 1180,
    "checkpoint_2": 5,
    "code_agent": 8500,
    "checkpoint_3": 5,
    "evaluator": 2900,
    "memory_write": 50,
    "pnl_record": 10
  },

  "agents_used": ["web.news", "web.finance", "code", "osa.evaluator"],
  "models_used": [
    "SRSWTI/axe-turbo-31b (decomposer, evaluator)",
    "SRSWTI/bodega-raptor-8b-mxfp4 (web.synthesizer)",
    "SRSWTI/bodega-raptor-0.9b (query_strategist)",
    "SRSWTI/axe-turbo-1b (code.writer)"
  ],

  "tokens_in": 3540,
  "tokens_out": 570,
  "tasks_completed": 5,
  "tasks_failed": 0,
  "checkpoints_created": 3,
  "hil_reviews": 1,
  "hil_approved": 1,
  "hil_declined": 0,

  "artifacts": [
    "/tmp/octane-sandboxes/sandbox_evt_7f3a/nvda_chart.png",
    "/tmp/octane-sandboxes/sandbox_evt_7f3a/chart_nvda.py"
  ]
}
```

---

## Step 15: Follow-Up Query (Demonstrating Memory)

```
$ octane ask "Now do the same for AMD"
```

### What happens differently:

**Memory Agent (hot cache) hit:**
OSA checks Redis before decomposition:
```
GET octane:session:sess_01:latest_nvda → HIT
```
OSA now knows: "User just analyzed NVDA. 'The same' means: news + market data + chart. Replace NVDA → AMD."

**Decomposer is faster:**
Instead of reasoning from scratch, Decomposer recognizes the pattern:
```
"Same pipeline as evt_7f3a. Replace NVDA with AMD.
 Reuse structure: [web.news(AMD), web.finance(AMD),
 web.timeseries(AMD), code.chart(AMD), osa.synthesize]"
```
Decomposition takes ~1s instead of ~3s.

**HIL is skipped for code execution:**
P&L learned from the previous interaction that user approves code execution for chart generation. If `hil_level` is "balanced" and this is the same pattern, HIL Manager auto-approves Decision #4.

**Pipeline completes faster:** ~12s instead of ~20s (no HIL wait, faster decomposition).

### Second follow-up: "Compare them side by side"

**Memory Agent serves both from hot cache:**
```
GET octane:session:sess_01:latest_nvda → NVDA data
GET octane:session:sess_01:latest_amd → AMD data
```

**No new web fetches needed.** Code Agent generates a comparison chart using both datasets. Total time: ~5s (just code generation + execution).

---

## Resource Usage Summary (64GB M1 Max)

| Component | RAM Usage | Notes |
|-----------|-----------|-------|
| 30B brain model | ~18GB | Loaded once at startup by SysStat |
| 8B worker model | ~5GB | Loaded at startup |
| 0.9B grunt model | ~600MB | Loaded at startup |
| Code model (axe-turbo-1b) | ~700MB | Loaded at startup or swapped in |
| Redis | ~100MB | Shadows + hot cache |
| PostgreSQL | ~200MB | Warm tier + pgVector |
| Python processes | ~500MB | Octane + agents |
| Sandbox venvs | ~200MB per sandbox | Temporary, cleaned up |
| **Total** | **~25GB** | Leaves ~39GB headroom on 64GB |

---

## Viewing the Trace

After the query completes, you can inspect the full trace:

```bash
$ octane trace evt_7f3a

╔══════════════════════════════════════════════════════════════╗
║  OCTANE TRACE — evt_7f3a                                     ║
║  Query: "What happened with NVIDIA today and write me..."    ║
║  Total: 20.5s | Tasks: 5 | Models: 4 | Checkpoints: 3      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  00.000s  ▶ query_received         cli → osa                 ║
║  00.048s  ✓ guard_check            osa.guard (PASS)          ║
║  00.056s  ✓ pnl_profile_fetch      osa → pnl (8ms)          ║
║  02.856s  ✓ decomposition          osa.decomposer (2800ms)   ║
║           │  "5 tasks, 3 parallel, 2 sequential"             ║
║  02.861s  ⬛ checkpoint_1           plan                      ║
║  02.862s  ✓ policy_assessment      4 auto, 1 pending         ║
║  08.062s  ✓ hil_review             D4 approved (5200ms)      ║
║  03.100s  ┬ parallel_start         [t1, t2, t3]              ║
║  03.480s  │✓ web.finance           t2 NVDA market (380ms)    ║
║  03.680s  │✓ web.finance.ts        t3 NVDA timeseries (580ms)║
║  04.280s  │✓ web.news              t1 NVIDIA news (1180ms)   ║
║  08.100s  ┴ parallel_complete      3/3 succeeded              ║
║  08.105s  ⬛ checkpoint_2           post_execution             ║
║  16.605s  ✓ code                   t4 chart (8500ms, 1 try)  ║
║  16.610s  ⬛ checkpoint_3           pre_synthesis              ║
║  19.510s  ✓ synthesis              t5 evaluator (2900ms)      ║
║  19.560s  ✓ memory_write           hot + warm tiers           ║
║  19.570s  ✓ pnl_record             feedback signal recorded   ║
║  20.500s  ▶ pipeline_complete      SUCCESS                    ║
║                                                              ║
║  Artifacts: nvda_chart.png (184KB), chart_nvda.py            ║
╚══════════════════════════════════════════════════════════════╝
```

---

## What the Alternative Flows Would Look Like

### If user had DECLINED Decision #4 (code execution):

```
User presses: [3] ❌ Skip (data only)
    ↓
Decision D4: status = "human_declined", feedback = "Just give me the data"
    ↓
OSA reverts to Checkpoint #2 (post-parallel, all web data preserved)
    ↓
Decomposer re-plans: removes t4 (code), modifies t5 (synthesis without chart)
    ↓
New DAG: just t5 synthesis using t1 + t2 data
    ↓
Output: news + market data, no chart. Total time: ~8s instead of 20s.
```

### If user had MODIFIED Decision #4:

```
User presses: [2] ✏️ Modify
User types: "Use plotly instead of matplotlib, and make it interactive HTML"
    ↓
Decision D4: status = "human_modified", feedback = "Use plotly, output HTML"
    ↓
Code Agent receives modified instruction
Planner outputs: { requirements: ["plotly"], output: "nvda_chart.html" }
Writer generates plotly code instead of matplotlib
    ↓
Output includes interactive HTML chart instead of static PNG
```

### If Code Agent had FAILED all 3 retries:

```
Attempt 1: ImportError → Debugger fixes import → retry
Attempt 2: ValueError (bad data parse) → Debugger fixes parsing → retry
Attempt 3: matplotlib rendering error → Debugger can't fix
    ↓
Code Agent returns: { success: false, error: "Max retries exhausted", attempts: 3 }
    ↓
OSA.Evaluator receives partial results:
  - t1 ✅ (news), t2 ✅ (market data), t3 ✅ (timeseries), t4 ❌ (code failed)
    ↓
Evaluator synthesizes what's available:
  "Here's the NVIDIA briefing. I attempted to generate a chart but
   encountered technical issues after 3 attempts. The timeseries data
   is available if you'd like to chart it manually."
    ↓
Graceful degradation — user still gets 80% of the answer.
```

### If a Web Agent Fetcher had FAILED (Bodega API down):

```
Fetcher: GET localhost:8032/api/v1/news/search → ConnectionError
    ↓
Synapse event: { event_type: "agent_error", source: "web.news" }
    ↓
Web Agent returns: { success: false, error: "News API unreachable" }
    ↓
OSA receives: t1 ❌, t2 ✅, t3 ✅
    ↓
OSA.Evaluator synthesizes with available data:
  "NVDA $142.50 (-3.2%) on elevated volume. Chart generated.
   Note: news data was unavailable — unable to provide context
   for today's price movement."
    ↓
Graceful degradation — user gets market data + chart without news.
```
