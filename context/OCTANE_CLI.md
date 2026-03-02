# OCTANE_CLI_v2.md
# Six Power Commands — Using the Full Stack
# SRSWTI Research Labs | 2026-03-01

---

## Why These Commands

`octane ask` is a single-shot query. It routes to one or two agents and returns a response. That's useful but it's like using a Formula 1 car to drive to the grocery store.

These six commands use the full engine: multi-tier BodegaRouter, parallel agent dispatch, domain pipelines, catalysts, memory recall, Code Agent analysis, cross-referencing — all composed into workflows that solve real problems.

Each command follows the same Unix principle: **a clear verb that does one powerful thing.**

---

## 1. octane investigate

**The "throw everything at it" command.**

Unlike `ask` (one-shot) or `research` (background), `investigate` is a synchronous deep-dive that decomposes your question into multiple dimensions, researches each in parallel, cross-references findings, and produces a structured report. All in one session.

### CLI

```bash
octane investigate "Is NVDA overvalued at current levels?"
octane investigate "Should I move from React to Svelte for my next project?"
octane investigate "What's the real state of AI in drug discovery?"

# Options
octane investigate "query" --depth exhaustive    # 8 dimensions (default: deep = 4-6)
octane investigate "query" --export report.md    # save to file
octane investigate "query" --verbose             # show dimension-by-dimension trace
```

### What Happens

```
User: "Is NVDA overvalued at current levels?"

┌─────────────────────────────────────────────────────────┐
│ DimensionPlanner (REASON tier)                          │
│                                                         │
│ "This is a valuation question. I need to investigate    │
│  from multiple independent angles to give a thorough    │
│  answer. Let me identify the dimensions:"               │
│                                                         │
│  1. Fundamentals  — P/E, P/S, margins, growth rate      │
│  2. Earnings      — last 4 quarters, beats/misses       │
│  3. Consensus     — analyst targets, ratings, upgrades   │
│  4. Peers         — AMD, INTC, AVGO same metrics         │
│  5. Technicals    — RSI, moving averages, volume trend   │
│  6. Macro         — AI capex cycle, datacenter demand    │
│  7. Risks         — competition, regulation, overbuilt   │
│  8. Memory        — any prior NVDA research/context      │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 1 (parallel) ────────────────────────────────────┐
│  Web.Finance  → NVDA P/E P/S margins revenue growth    │
│  Web.Finance  → AMD INTC AVGO same metrics             │
│  Web.Search   → NVDA earnings Q1-Q4 2025 results       │
│  Web.News     → NVDA analyst ratings price targets      │
│  Web.News     → AI datacenter capex spending 2026       │
│  Web.Search   → NVDA risks competition market concerns  │
│  Memory.Read  → prior NVDA research findings            │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 2 ───────────────────────────────────────────────┐
│  Code.Catalyst → technical_indicators(NVDA)            │
│  Code.Catalyst → valuation_comparison(NVDA, AMD, ...)  │
│  Code.Analyze  → cross-reference claims across sources │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 3 ───────────────────────────────────────────────┐
│  Evaluator (REASON tier) → structured multi-section    │
│  report with verdict, confidence, and evidence trail   │
└─────────────────────────────────────────────────────────┘
```

### Output Format

```
🔍 Investigation: Is NVDA overvalued at current levels?

  8 dimensions analyzed | 14 sources | 3 catalysts | Memory: 2 prior findings

  ── Fundamentals ──────────────────────────────────────
  P/E: 55.2x (sector median: 28.4x). P/S: 30.1x. Revenue growth: 94% YoY.
  Gross margin: 76%. Operating margin: 62%. FCF margin: 48%.

  ── Peer Comparison ───────────────────────────────────
  ┌──────────┬───────┬───────┬────────┬──────────┐
  │ Ticker   │ P/E   │ P/S   │ Growth │ Margin   │
  ├──────────┼───────┼───────┼────────┼──────────┤
  │ NVDA     │ 55.2  │ 30.1  │ 94%    │ 76%      │
  │ AMD      │ 42.8  │ 11.2  │ 18%    │ 52%      │
  │ INTC     │ 18.3  │ 2.1   │ -8%    │ 41%      │
  │ AVGO     │ 38.5  │ 18.6  │ 44%    │ 74%      │
  └──────────┴───────┴───────┴────────┴──────────┘

  ── Technicals ────────────────────────────────────────
  RSI: 32 (oversold). Below 50-day MA. Volume declining.
  Support at $165, resistance at $195.

  ── Verdict ───────────────────────────────────────────
  NVDA trades at a premium to peers but its growth rate (94%)
  justifies a higher multiple. Current RSI suggests short-term
  oversold. At current levels, it's fairly valued to slightly
  expensive IF growth sustains. Key risk: AI capex cycle slowing.

  Confidence: MEDIUM-HIGH (6 of 8 dimensions have strong data)

  Sources: 14 unique URLs
  Run 'octane investigate "NVDA" --export nvda-report.md' to save
```

### Key Differences from `octane ask`

| | `ask` | `investigate` |
|---|---|---|
| Dimensions | 1-2 (single template) | 4-8 (multi-dimensional) |
| Code Agent | Optional | Always runs (cross-reference + catalysts) |
| Memory | Not checked | Always checked for prior context |
| Output | Prose response | Structured multi-section report |
| Peer comparison | Not included | Automatic for entities with peers |
| Duration | 10-30s | 60-180s |

---

## 2. octane compare

**Structured multi-dimensional comparison with quantitative analysis.**

### CLI

```bash
octane compare "NVDA vs AMD"
octane compare "React vs Svelte vs Vue"
octane compare "Fidelity vs Schwab for beginner investor"
octane compare "renting vs buying in Boston"

# Options
octane compare "A vs B" --dimensions 8     # more comparison axes
octane compare "A vs B" --export comp.md
octane compare "A vs B" --quantitative     # force numerical analysis where possible
```

### What Happens

```
User: "NVDA vs AMD"

┌─────────────────────────────────────────────────────────┐
│ ComparisonPlanner (REASON tier)                         │
│                                                         │
│ Items: NVDA, AMD                                        │
│ Type: financial/stock comparison                        │
│ Dimensions:                                             │
│   1. Financials (P/E, revenue, margins, growth)         │
│   2. Product portfolio (GPUs, data center, AI chips)    │
│   3. Market position (share, partnerships, moat)        │
│   4. Recent performance (stock returns, earnings)       │
│   5. Forward outlook (analyst targets, pipeline)        │
│   6. Risks (competition, concentration, macro)          │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 1 (parallel — BOTH items researched independently) ┐
│                                                           │
│  NVDA side:                    AMD side:                  │
│  Web.Finance → NVDA data       Web.Finance → AMD data    │
│  Web.Search → NVDA products    Web.Search → AMD products │
│  Web.News → NVDA outlook       Web.News → AMD outlook    │
│                                                           │
└───────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 2 ───────────────────────────────────────────────┐
│  Code.Analyze → build comparison matrix                │
│  Code.Catalyst → valuation_comparison(NVDA, AMD)       │
│  Code.Catalyst → return_calculator(NVDA, AMD, 1yr)     │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 3 ───────────────────────────────────────────────┐
│  Evaluator → side-by-side report with explicit         │
│  tradeoff analysis and use-case recommendations        │
└─────────────────────────────────────────────────────────┘
```

### Output Format

```
⚖ Comparison: NVDA vs AMD

  6 dimensions | 12 sources | 2 catalysts

  ┌──────────────────┬────────────────────┬────────────────────┐
  │ Dimension        │ NVDA               │ AMD                │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Price            │ $177.19            │ $118.42            │
  │ P/E              │ 55.2x              │ 42.8x              │
  │ Revenue Growth   │ 94% YoY ✓         │ 18% YoY            │
  │ Gross Margin     │ 76% ✓             │ 52%                │
  │ AI Market Share  │ ~80% datacenter ✓ │ ~15% datacenter    │
  │ 1yr Return       │ +142%             │ +28%                │
  │ RSI              │ 32 (oversold)     │ 58 (neutral)       │
  │ Analyst Targets  │ $220 (+24%)       │ $145 (+22%)        │
  └──────────────────┴────────────────────┴────────────────────┘

  ── Tradeoff Analysis ─────────────────────────────────
  NVDA wins on: growth, margins, AI dominance, analyst conviction
  AMD wins on: valuation (cheaper P/E), diversification (CPU+GPU), less concentration risk

  ── When to Pick Which ────────────────────────────────
  NVDA: if you believe AI infrastructure spending continues to accelerate
  AMD: if you want AI exposure with a margin of safety on valuation
```

---

## 3. octane monitor

**Compound multi-signal surveillance. Not just price — everything.**

### CLI

```bash
octane monitor "NVDA"
octane monitor "NVDA" --signals price,news,earnings,sentiment,insider
octane monitor "Bitcoin" --signals price,news,sentiment --every 2h
octane monitor "my portfolio" --signals drift,news,dividends

# Manage monitors
octane monitor list
octane monitor pause <id>
octane monitor stop <id>
octane monitor alerts           # show all pending alerts
```

### What Happens

```
User: octane monitor "NVDA" --signals price,news,earnings,sentiment

Creates a compound Shadows perpetual task with 4 sub-monitors:

┌─ Price Monitor (every 1h) ─────────────────────────────┐
│  Web.Finance → NVDA current price, volume, change      │
│  Code.Catalyst → technical_indicators(NVDA)            │
│  Alert if: daily change > ±5%, RSI < 30 or > 70,      │
│            volume > 2x average, price crosses MA        │
└─────────────────────────────────────────────────────────┘

┌─ News Monitor (every 4h) ──────────────────────────────┐
│  Web.News → "NVDA Nvidia news"                         │
│  Web.Search → "NVDA analyst upgrade downgrade"         │
│  Memory → dedup against previously seen articles       │
│  Alert if: major news (earnings, lawsuit, product      │
│            launch, executive change, rating change)     │
└─────────────────────────────────────────────────────────┘

┌─ Earnings Monitor (daily) ─────────────────────────────┐
│  Web.Search → "NVDA earnings date next quarter"        │
│  Alert if: earnings date within 14 days                │
│  Alert if: pre-earnings analyst estimate revisions     │
└─────────────────────────────────────────────────────────┘

┌─ Sentiment Monitor (every 6h) ─────────────────────────┐
│  Web.Search → "NVDA sentiment Reddit HN Twitter"       │
│  Code.Analyze → sentiment score from aggregated text   │
│  Memory → compare to previous sentiment reading        │
│  Alert if: sentiment shift > 20% in either direction   │
└─────────────────────────────────────────────────────────┘

Cross-signal alerts:
  → Price drop >3% AND negative news → HIGH ALERT
  → Sentiment shift negative AND earnings within 14 days → MEDIUM ALERT
  → All signals stable → silent log, no notification
```

### Alert Display

```bash
$ octane monitor alerts

  🔴 HIGH — NVDA Multi-Signal Alert (2h ago)
     Price: -4.16% ($177.19) + RSI oversold (32)
     News: "Nvidia faces new export restrictions on AI chips to China"
     Recommendation: Review position. Multiple bearish signals aligned.

  🟡 MEDIUM — NVDA Earnings Approaching (12h ago)
     Earnings date: March 15, 2026 (14 days away)
     Analyst estimates revised down 3% in past week
     Recommendation: Consider position sizing before earnings.

  🟢 LOW — AAPL Dividend Payment (1d ago)
     Quarterly dividend: $0.25/share
     Note: Auto-reinvested if DRIP enabled.
```

---

## 4. octane plan

**Goal-oriented action planning. Not a query — a commitment.**

### CLI

```bash
octane plan "Build a $100K portfolio over 5 years starting with $500/month"
octane plan "Transition from frontend engineer to ML engineer in 6 months"
octane plan "Launch a newsletter with 1000 subscribers in 3 months"
octane plan "Get into marathon shape from couch potato by October"

# Options
octane plan "goal" --timeline 6m         # explicit timeline
octane plan "goal" --constraints "budget: $200/month, time: 2h/day"
octane plan "goal" --export plan.md
octane plan "goal" --monitor             # auto-create monitoring after plan
```

### What Happens

```
User: "Build a $100K portfolio over 5 years starting with $500/month"

┌─────────────────────────────────────────────────────────┐
│ GoalAnalyzer (REASON tier)                              │
│                                                         │
│ Goal type: financial / wealth building                  │
│ Timeline: 5 years                                       │
│ Constraint: $500/month                                  │
│ Target: $100K                                           │
│                                                         │
│ Research needed:                                        │
│  1. Is $100K in 5yr achievable at $500/mo?              │
│  2. What return rate is needed? Is it realistic?        │
│  3. Best vehicles for this timeline + amount            │
│  4. Tax optimization (IRA, 401k, taxable)               │
│  5. Historical scenarios — what worked?                  │
│  6. Risk scenarios — what could go wrong?               │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 1 (parallel research) ──────────────────────────┐
│  Web.Search  → investment vehicles $500/month          │
│  Web.Search  → IRA vs taxable account comparison       │
│  Web.Finance → VOO VTI SCHD historical returns 5yr    │
│  Web.News    → market outlook next 5 years             │
│  Memory      → user's risk tolerance, existing prefs   │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 2 (Code Agent analysis) ────────────────────────┐
│  Code.Catalyst → portfolio_projection                  │
│    input: $500/mo, 5yr, various allocations            │
│    output: which allocation hits $100K?                │
│                                                         │
│  Code.Analyze → milestone calculator                   │
│    output: month-by-month targets                      │
│    Year 1: $6,240 · Year 2: $13,100 · ...             │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Wave 3 (Synthesis) ──────────────────────────────────┐
│  Evaluator → structured action plan with milestones   │
│  Memory.Write → store plan for future reference       │
└─────────────────────────────────────────────────────────┘
```

### Output Format

```
📋 Plan: $100K Portfolio in 5 Years

  Feasibility: ✅ ACHIEVABLE
  Required return: 7.2% annualized (historical S&P avg: 10.5%)
  Confidence: HIGH — below historical average required rate

  ── Phase 1: Foundation (Months 1-3) ──────────────────
  □ Open Fidelity Roth IRA (tax-free growth for 5yr horizon)
  □ Set up $500/month automatic investment
  □ Initial allocation:
      40% VTI · 25% VOO · 20% SCHD · 15% QQQ
  □ Milestone: $1,530 by month 3

  ── Phase 2: Accumulation (Months 4-24) ───────────────
  □ Continue $500/month, no changes
  □ Rebalance quarterly if any position drifts >5% from target
  □ Milestone: $13,100 by month 24

  ── Phase 3: Growth (Months 25-48) ────────────────────
  □ If income increases, raise contribution to $600-700/month
  □ Consider adding international exposure (VXUS) at 10%
  □ Milestone: $55,800 by month 48

  ── Phase 4: Final Push (Months 49-60) ────────────────
  □ Review allocation — shift 5% from QQQ to SCHD for stability
  □ Target: $100,000 by month 60
  □ If behind target: increase monthly contribution, not risk

  ── Risk Scenarios ────────────────────────────────────
  Bear case (5% return): $72,800 at 60 months
  Base case (7.2% return): $100,000 at 60 months
  Bull case (10% return): $118,400 at 60 months

  Set up monitoring? Run:
    octane monitor "my portfolio" --signals drift,dividends
```

---

## 5. octane replay

**Re-run a past analysis with fresh data. See what changed.**

### CLI

```bash
octane replay <trace-id>
octane replay <trace-id> --diff          # highlight changes from original
octane replay <trace-id> --export delta.md

# List replayable traces
octane replay list
```

### What Happens

```
User: octane replay abc12345 --diff

1. Read trace abc12345 from Synapse
2. Extract the original DAG structure and query
3. Re-run the same DAG with current data
4. Diff the outputs:
   - Which facts changed?
   - Which prices moved?
   - Any new information?
5. Produce a delta report

Output:

🔄 Replay: abc12345 — "NVDA valuation analysis"
   Original run: 2026-02-22 | This run: 2026-03-01

  ── What Changed ──────────────────────────────────────
  Price:    $185.40 → $177.19 (-4.4%)
  RSI:     58 → 32 (moved from neutral to oversold)
  Analyst: unchanged (consensus $220 target)
  News:    NEW — export restriction concerns emerged
  P/E:     57.8 → 55.2 (compressed with price)

  ── What Didn't Change ────────────────────────────────
  Revenue growth: still 94% YoY
  Gross margin: still 76%
  AI market share: still ~80% datacenter

  ── Updated Verdict ───────────────────────────────────
  Original: "fairly valued to slightly expensive"
  Updated:  "approaching attractive levels, RSI oversold,
             but new regulatory risk emerged"
```

### How It Works Under the Hood

The replay system reads the original trace's DAG structure (which agents, what queries, what templates) and reconstructs the same TaskDAG. It then dispatches through the normal Orchestrator pipeline but forces the same decomposition. The Evaluator receives both the original output (from memory) and the new output, and produces a diff-aware synthesis.

The key constraint: replay preserves the DAG shape but refreshes the data. Same agents, same dimensions, fresh results.

---

## 6. octane chain

**Explicit multi-step pipeline from the command line. eyeso before eyeso exists.**

### CLI

```bash
# Basic chain — each step feeds into the next
octane chain \
  "fetch finance NVDA AAPL MSFT" \
  "analyze technical {prev}" \
  "synthesize report {all}"

# Named steps with explicit references
octane chain \
  "prices: fetch finance NVDA AMD AVGO" \
  "news: fetch news AI chip stocks earnings" \
  "tech: analyze technical {prices}" \
  "report: synthesize investment-brief {prices} {news} {tech}"

# Save as workflow for reuse
octane chain --save "weekly-check" \
  "fetch finance {{tickers}}" \
  "analyze technical {prev}" \
  "fetch news {{tickers}} latest" \
  "synthesize brief {all}"

# Then run the saved chain
octane workflow run weekly-check --var tickers="NVDA AAPL MSFT"
```

### Step Syntax

Each step in a chain is a string with the format:

```
[name:] command subcommand [arguments] [{reference}]
```

References:
- `{prev}` — output of the previous step
- `{step_name}` — output of a named step
- `{all}` — all prior step outputs combined
- `{{variable}}` — template variable (for `--save` mode)

### What Happens

```
User: octane chain \
        "prices: fetch finance NVDA AAPL MSFT" \
        "tech: analyze technical {prices}" \
        "news: fetch news AI stocks latest" \
        "synthesize investment-brief {all}"

Step 1 — "prices: fetch finance NVDA AAPL MSFT"
  → WebAgent.fetch_finance(["NVDA", "AAPL", "MSFT"])
  → Returns: price data for all three tickers
  → Stored as: chain.prices

Step 2 — "tech: analyze technical {prices}"
  → CodeAgent with technical_indicators catalyst
  → Input: chain.prices
  → Returns: RSI, MA, volume analysis for each ticker
  → Stored as: chain.tech

Step 3 — "news: fetch news AI stocks latest"
  → WebAgent.fetch_news("AI stocks latest")
  → Returns: synthesized news summary
  → Stored as: chain.news

Step 4 — "synthesize investment-brief {all}"
  → Evaluator receives: chain.prices + chain.tech + chain.news
  → Produces: cohesive investment brief combining all data
  → Output to terminal
```

### Chain Display

```
🔗 Chain: 4 steps

  [1/4] prices: fetch finance NVDA AAPL MSFT ............ ✅ (2.1s)
  [2/4] tech: analyze technical .......................... ✅ (4.3s)
  [3/4] news: fetch news AI stocks latest ................ ✅ (3.8s)
  [4/4] synthesize investment-brief ...................... ✅ (6.2s)

  Total: 16.4s | 4 agents used | 3 data sources

  🔥 Result:
  [synthesized investment brief output]
```

---

## Implementation Priority

| Command | Complexity | Uses | Session |
|---------|-----------|------|---------|
| `investigate` | Medium — new DimensionPlanner + multi-wave DAG | Everything: all tiers, parallel agents, catalysts, memory | 22 |
| `compare` | Medium — variant of investigate with two-sided structure | Web.Finance + Code.Catalyst + Evaluator | 22 |
| `chain` | Low-medium — step parser + sequential/named execution | Existing agent calls composed explicitly | 22 |
| `plan` | Medium — GoalAnalyzer + milestone Code Agent | Research + Code.Catalyst + Memory + Evaluator | 23 |
| `monitor` | Medium-high — compound Shadows task with signal routing | Shadows + Web.Finance + Web.News + Code + Memory + Alerts | 23 |
| `replay` | Medium — trace reader + DAG reconstruction + diff synthesis | Synapse traces + Orchestrator + Evaluator diff mode | 24 |

---

## How These Commands Relate to eyeso

These CLI commands are the building blocks that eyeso scripts compose. Every `octane investigate`, `octane compare`, `octane chain` becomes an eyeso verb:

```eyeso
# eyeso script using the power commands
result = investigate "NVDA valuation" depth=exhaustive
comparison = compare "NVDA vs AMD"
plan = plan "build $100K portfolio" timeline=5y

monitor "NVDA" signals=[price, news, sentiment]
monitor "AMD" signals=[price, news]

every morning:
    brief = investigate "overnight market changes" depth=shallow
    $brief → notify
```

The CLI commands prove the capabilities. eyeso composes them.