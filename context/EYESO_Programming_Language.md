# EYESO Programming Language

**eyeso** — the purest form of Octane.

iso-octane (2,2,4-trimethylpentane) is the gold standard of fuel purity. eyeso is the gold standard of AI workflow composition. You write what you want to happen. Octane figures out how.

---

## Philosophy

eyeso follows four principles borrowed from Unix and adapted for AI workflows:

1. **Do one thing well.** Each eyeso command maps to one Octane capability. `fetch`, `research`, `analyze`, `filter`, `synthesize` — each does exactly one thing.

2. **Compose through pipes.** The `→` operator passes output from one command as input to the next. Small commands chained together solve complex problems.

3. **Text is the universal interface.** Every command produces structured data (text, tables, lists). Every command accepts structured data. No type system to fight.

4. **Make the common case effortless.** `research "NVDA"` should just work. No boilerplate, no imports, no configuration preamble. Sensible defaults everywhere. Power when you need it, invisible when you don't.

---

## Quick Start

```eyeso
# hello.eyeso — your first eyeso script

ask "what is the current price of NVDA?"
```

```bash
$ octane eyeso run hello.eyeso
NVDA is currently trading at $177.19, down 4.16% today...
```

That's it. One line. Octane handles decomposition, routing, agent dispatch, synthesis.

Now something real:

```eyeso
# morning-check.eyeso

tickers = ["NVDA", "AAPL", "MSFT", "VOO"]

parallel for t in $tickers:
    prices.$t = fetch finance $t

for t in $tickers:
    if $prices.$t.change_pct < -3:
        alert "$t down ${prices.$t.change_pct}% — check position"

print "Morning check complete"
```

```bash
$ octane eyeso run morning-check.eyeso
⚠ NVDA down -4.16% — check position
Morning check complete
```

---

## Execution Modes

eyeso scripts run in two modes with the same syntax.

### Interactive Mode (Interpreted)

```bash
$ octane eyeso
eyeso> ticker = "NVDA"
eyeso> data = fetch finance $ticker
eyeso> print $data.price
177.19
eyeso> print $data.change_pct
-4.16
eyeso> if $data.change_pct < -5: alert "Big drop on $ticker"
eyeso>
```

Line by line. See results immediately. Great for exploration and prototyping.

### Script Mode (Interpreted)

```bash
$ octane eyeso run morning-check.eyeso
```

Runs the script top to bottom. `parallel` blocks execute concurrently. Errors halt execution unless caught with `try/catch`.

### Deploy Mode (Compiled)

```bash
# Preview the compiled DAG
$ octane eyeso compile morning-check.eyeso --preview
DAG: 4 parallel fetch nodes → 1 filter pass → 1 alert gate
Estimated runtime: ~15 seconds

# Deploy as a scheduled background task
$ octane eyeso deploy morning-check.eyeso --every 8h --name "morning-check"
Deployed: morning-check (every 8h, next run 06:00 UTC)

# Check deployed scripts
$ octane eyeso jobs
  ID        Name            Every    Last Run     Status
  a3f8c1    morning-check   8h       2h ago       ● active

# Stop a deployed script
$ octane eyeso stop morning-check
```

The compiler analyzes your script, builds an optimized DAG (parallel where possible, sequential where dependencies exist), and registers it as a Shadows perpetual task.

---

## Language Reference

### Variables

```eyeso
# Assignment
name = "NVDA"
amount = 500
tickers = ["NVDA", "AAPL", "MSFT"]
config = {depth: "deep", every: 6}

# Variable substitution
print "Checking $name"
print "Price is ${data.price}"

# Destructuring from command output
{price, volume, change_pct} = fetch finance "NVDA"
```

Variables are dynamically typed. Numbers, strings, lists, and dicts.

### Commands

Commands are the core of eyeso. Each maps to an Octane capability.

```eyeso
# Ask — run a full OSA pipeline (decompose → route → dispatch → synthesize)
result = ask "what should I invest in with $500?"

# Fetch — direct agent calls (no decomposition, no synthesis)
data = fetch finance "NVDA"
news = fetch news "AI developments"
results = fetch search "best ETFs 2026"
page = fetch url "https://example.com/article"

# Research — long-running background research
research start "AI drug discovery" depth=deep every=12h
research stop <task-id>
findings = research report <task-id>
tasks = research list

# Analyze — Code Agent execution
chart = analyze price-chart $data
projection = analyze portfolio-projection {initial: 500, monthly: 100}
indicators = analyze technical $data
score = analyze match-score $job $profile

# Synthesize — LLM synthesis from data
summary = synthesize $findings
report = synthesize briefing $portfolio $news $research
draft = synthesize article $research tone="conversational"

# Filter — data filtering and transformation
high_confidence = filter $findings where confidence == "multi_source"
big_movers = filter $prices where abs(change_pct) > 3
recent = filter $jobs where days_ago < 7

# Memory — read and write to Octane's memory
profile = recall "my portfolio allocation"
remember "portfolio" $allocation

# Export — save to file
$report → export "report.md"
$chart → export "chart.png"
$data → export "prices.csv"
```

### The Pipe Operator: →

The `→` operator passes the output of one command as input to the next. This is eyeso's composition mechanism.

```eyeso
# Simple pipe
fetch finance "NVDA" → analyze price-chart → export "nvda.png"

# Multi-step pipeline
fetch search "AI drug discovery breakthroughs" \
    → filter where word_count > 100 \
    → synthesize report \
    → export "research.md"

# Named intermediate results
data = fetch finance "NVDA"
$data → analyze technical → export "indicators.png"
$data → analyze price-chart → export "price.png"
```

Pipes chain left to right. Each step receives the previous step's output as its first argument.

### Parallel Execution

```eyeso
# Parallel block — all lines run concurrently
parallel:
    macro = fetch news "market conditions"
    prices = fetch finance "VOO"
    research = fetch search "index fund performance 2026"

# All three complete before execution continues
print "Got macro, prices, and research"

# Parallel for — each iteration runs concurrently
parallel for ticker in ["NVDA", "AAPL", "MSFT", "GOOGL"]:
    data.$ticker = fetch finance $ticker

# After the parallel for, data.NVDA, data.AAPL etc. are all populated
```

In interpreted mode, `parallel` uses asyncio.gather under the hood.
In compiled mode, `parallel` becomes a single DAG wave with multiple nodes.

### Conditionals

```eyeso
if $price < 150:
    alert "Buy signal"
elif $price > 200:
    alert "Take profit"
else:
    log "Hold steady"

# Inline conditional
action = "buy" if $price < 150 else "hold"
```

### Loops

```eyeso
# For-each
for ticker in $tickers:
    data = fetch finance $ticker
    print "$ticker: ${data.price}"

# Parallel for-each (each iteration concurrent)
parallel for ticker in $tickers:
    data.$ticker = fetch finance $ticker

# While (interpreted mode only — not available in compiled/deployed scripts)
while true:
    data = fetch finance "NVDA"
    if $data.change_pct < -5:
        alert "Drop detected"
        break
    wait 5m
```

### Error Handling

```eyeso
try:
    data = fetch finance "INVALID_TICKER"
    chart = analyze price-chart $data
catch err:
    log "Failed: $err"
    data = fetch finance "SPY"  # fallback

# Per-command error suppression
data = fetch finance "MAYBE_VALID" ?? null
if $data == null:
    print "Ticker not found"
```

The `??` operator provides a default value if the command fails. Like Kotlin's `?:` or C#'s `??`.

### Scheduling Blocks

```eyeso
# Run a block on a schedule (compiled mode only)
every 6h:
    data = fetch finance "NVDA"
    if $data.change_pct < -5:
        alert "NVDA down big"

# Named schedule
every morning as "portfolio-check":
    portfolio = recall "my portfolio"
    parallel for t in $portfolio.tickers:
        prices.$t = fetch finance $t
    drift = analyze portfolio-drift $portfolio $prices
    if len($drift.alerts) > 0:
        synthesize briefing $drift → notify

# Schedule aliases
# every morning  = every day at 06:00 local
# every evening  = every day at 18:00 local
# every hour     = every 1h
# every 6h       = every 6 hours
# every monday   = every week on Monday at 09:00
```

### Built-in Functions

```eyeso
# Output
print "message"              # stdout
log "message"                # structlog (silent in terminal, visible in traces)
alert "message"              # high-priority notification

# Data
len($list)                   # length of list or string
keys($dict)                  # keys of a dict
values($dict)                # values of a dict
type($var)                   # "string", "number", "list", "dict", "null"
abs($number)                 # absolute value
round($number, 2)            # round to N decimals

# String
upper($str)                  # uppercase
lower($str)                  # lowercase
contains($str, "substr")     # boolean
split($str, ",")             # split to list
join($list, ", ")            # join list to string

# Time
now()                        # current datetime
today()                      # current date
days_ago($datetime)          # days since datetime
wait 5m                      # pause execution (interpreted mode)
wait 30s
wait 2h

# Collections
sort $list by field desc     # sort a list of dicts
first $list                  # first element
last $list                   # last element
unique $list                 # deduplicate
count $list where condition  # count matching elements
sum $list.field              # sum a field across list
avg $list.field              # average a field
```

### Comments

```eyeso
# Single line comment

## Section header (rendered in --preview output)

# Multi-line comments use consecutive # lines
# There is no block comment syntax.
# This is intentional — eyeso scripts should be short enough
# that you don't need to comment out large blocks.
```

---

## Complete Examples

### Example 1: Investment Research Pipeline

```eyeso
## Investment Analysis for a Beginner
# Run: octane eyeso run invest.eyeso --var budget=500

budget = $args.budget ?? 500

# Gather intelligence in parallel
parallel:
    macro = fetch news "stock market outlook 2026"
    vehicles = fetch search "best investments for $${budget} beginner"
    funds = fetch finance "VOO VTI SCHD QQQ"

# Build projection
projection = analyze portfolio-projection {
    initial: $budget,
    monthly: 100,
    tickers: ["VOO", "VTI", "SCHD", "QQQ"],
    allocations: [0.25, 0.40, 0.20, 0.15]
}

# Synthesize personalized plan
plan = synthesize investment-plan $macro $vehicles $funds $projection

$plan → export "investment-plan.md"
print "Plan saved to investment-plan.md"
```

### Example 2: Deep Research with Quality Control

```eyeso
## Research Agent
# Deploy: octane eyeso deploy research.eyeso --every 12h --var topic="AI drug discovery"

topic = $args.topic

parallel:
    breakthroughs = fetch search "$topic breakthroughs 2026"
    companies = fetch search "$topic companies pipeline"
    failures = fetch search "$topic failures limitations"
    papers = fetch search "$topic research papers studies"

# Quality gate — skip if results are thin
total_words = sum([$breakthroughs, $companies, $failures, $papers].word_count)
if $total_words < 200:
    log "Insufficient data this cycle — skipping"
    return

# Cross-reference and synthesize
findings = synthesize research-report {
    sources: [$breakthroughs, $companies, $failures, $papers],
    format: "structured",
    confidence_levels: true
}

# Store findings with topic tag
remember "research:$topic" $findings

# Only alert on high-confidence multi-source findings
important = filter $findings where confidence == "multi_source"
if len($important) > 0:
    alert "New confirmed findings for $topic: ${len($important)} items"

$findings → export "research/${topic}_${today()}.md"
```

### Example 3: Morning Briefing

```eyeso
## Morning Briefing — runs daily at 6 AM
# Deploy: octane eyeso deploy briefing.eyeso --every morning --name "morning-briefing"

portfolio = recall "my portfolio"
research_projects = research list where status == "active"

# Portfolio check
parallel for t in $portfolio.tickers:
    prices.$t = fetch finance $t

drift = analyze portfolio-drift $portfolio $prices
portfolio_summary = synthesize portfolio-status $drift

# Research updates
parallel for project in $research_projects:
    updates.$project.id = research report $project.id --since yesterday

# News digest
interests = recall "my interests" ?? ["AI", "tech", "finance"]
parallel for topic in $interests:
    news.$topic = fetch news "$topic latest"

# Compile briefing
briefing = synthesize morning-briefing {
    portfolio: $portfolio_summary,
    research: $updates,
    news: $news,
    date: today()
}

$briefing → export "briefings/${today()}.md"
$briefing → notify

print "☀️ Briefing ready"
```

### Example 4: Career Monitor

```eyeso
## Career Monitor — scans for matching jobs
# Deploy: octane eyeso deploy career.eyeso --every 6h

profile = recall "my career profile"

# Search multiple angles
parallel:
    jobs_role = fetch search "${profile.target_role} remote ${profile.salary}+"
    jobs_company = fetch search "${profile.industries} engineering jobs"
    jobs_stack = fetch search "${profile.skills} senior staff principal engineer"

# Combine and deduplicate
all_jobs = unique [$jobs_role, $jobs_company, $jobs_stack] by url
previous = recall "career:seen_jobs" ?? []

new_jobs = filter $all_jobs where url not in $previous.urls

# Score each new job
for job in $new_jobs:
    job.score = analyze match-score $job $profile

# Filter and rank
good_matches = filter $new_jobs where score >= 60
good_matches = sort $good_matches by score desc

# Store and alert
remember "career:seen_jobs" $all_jobs
remember "career:latest_matches" $good_matches

if len($good_matches) > 0:
    summary = synthesize career-matches $good_matches $profile
    alert "${len($good_matches)} new job matches found"
    $summary → export "career/matches_${today()}.md"
```

### Example 5: Content Creation Pipeline

```eyeso
## Newsletter Writer
# Run: octane eyeso run newsletter.eyeso --var topic="Claude Code"

topic = $args.topic

# Deep research
parallel:
    articles = fetch search "$topic review analysis"
    news = fetch news "$topic latest"
    discussion = fetch search "$topic developer opinions HN Reddit"
    competitors = fetch search "$topic vs alternatives comparison"

# Fact-check key claims
facts = analyze fact-check [$articles, $news]
verified = filter $facts where status == "verified"
flagged = filter $facts where status == "unverified"

# Generate outline
outline = synthesize outline {
    topic: $topic,
    research: [$articles, $news, $discussion, $competitors],
    tone: "conversational but informed",
    audience: "tech professionals"
}

print "Outline:"
print $outline
print ""
print "Proceed with draft? (Ctrl+C to edit outline first)"
wait 5s

# Write draft
draft = synthesize article {
    outline: $outline,
    facts: $verified,
    flagged_claims: $flagged,
    style: "newsletter"
}

$draft → export "newsletter/${topic}_draft.md"

print "Draft saved. ${len($verified)} facts verified, ${len($flagged)} flagged."
```

---

## Command Line Interface

```bash
# Run a script
octane eyeso run script.eyeso
octane eyeso run script.eyeso --var ticker=NVDA --var budget=500

# Interactive REPL
octane eyeso

# Compile and preview (shows DAG, no execution)
octane eyeso compile script.eyeso --preview

# Deploy as scheduled background task
octane eyeso deploy script.eyeso --every 6h --name "my-task"
octane eyeso deploy script.eyeso --every morning

# Manage deployed scripts
octane eyeso jobs                  # list all deployed scripts
octane eyeso logs <name>           # view execution logs
octane eyeso logs <name> --follow  # stream live logs
octane eyeso stop <name>           # stop a deployed script
octane eyeso restart <name>        # restart a stopped script

# Validate syntax without running
octane eyeso check script.eyeso

# Format/lint
octane eyeso fmt script.eyeso
```

---

## Script Arguments

Scripts accept arguments via `--var` flags, accessible through `$args`:

```eyeso
# portfolio.eyeso
budget = $args.budget ?? 1000
risk = $args.risk ?? "moderate"
tickers = $args.tickers ?? ["VOO", "VTI"]

print "Analyzing $risk portfolio with $${budget}"
```

```bash
octane eyeso run portfolio.eyeso --var budget=5000 --var risk=aggressive
```

The `??` operator provides defaults when arguments aren't supplied.

---

## How eyeso Maps to Octane

Every eyeso command compiles to an Octane operation:

| eyeso | Octane | What Happens |
|-------|--------|-------------|
| `ask "query"` | `Orchestrator.run_stream()` | Full OSA pipeline: decompose → route → dispatch → synthesize |
| `fetch finance X` | `WebAgent._fetch_finance()` | Direct Bodega Finance API call |
| `fetch news X` | `WebAgent._fetch_news()` | Bodega News API + Synthesizer |
| `fetch search X` | `WebAgent._fetch_search()` | Bodega Search API + Synthesizer |
| `fetch url X` | `ContentExtractor.extract_url()` | trafilatura/Playwright extraction |
| `research start` | `ResearchStore.start_task()` | Shadows perpetual research cycle |
| `research report` | `ResearchSynthesizer.generate()` | Rolling synthesis from findings |
| `analyze X` | `CodeAgent.execute()` + catalyst | Catalyst match or LLM code generation |
| `synthesize X` | `Evaluator.evaluate()` | LLM synthesis via BodegaRouter REASON tier |
| `filter X` | Python list comprehension | In-process data filtering |
| `recall X` | `MemoryAgent.recall()` | Redis → Postgres → pgVector waterfall |
| `remember X` | `MemoryAgent.store()` | Write to memory system |
| `export X` | File write | Save to ~/octane_output/ |
| `alert X` | Notification queue | Shown on next `octane health` |
| `parallel:` | `asyncio.gather()` / DAG wave | Concurrent execution |
| `every X:` | Shadows perpetual task | Compiled to scheduled DAG |
| `→` | Output piping | Previous result becomes next input |

---

## Design Decisions

**Why not just use Python?**
Python can do everything eyeso does. But Python requires you to think about imports, async/await, error handling boilerplate, Octane client initialization, and data serialization. eyeso lets you think about the problem. `fetch finance "NVDA"` is one line. The equivalent Python is 8 lines of setup + await + error handling.

**Why not just use bash?**
Bash composes CLI commands well but has no concept of structured data, parallel execution is clunky, and error handling is fragile. eyeso understands that `fetch finance "NVDA"` returns a dict with `.price`, `.volume`, `.change_pct` fields. Bash sees everything as strings.

**Why its own syntax?**
eyeso's syntax is designed around AI workflow patterns that don't exist in other languages. `parallel for`, `→` pipe to export, `every morning:` scheduling blocks, `?? default` fallback operator, `filter X where Y` — these are first-class constructs because they're the things you do most often with Octane. A DSL embedded in YAML would be verbose. A Python library would require boilerplate. eyeso is purpose-built.

**Why two execution modes?**
Interactive mode (interpreted) is for exploration: "what does NVDA look like right now?" Deploy mode (compiled) is for automation: "check my portfolio every morning and alert me." Same script works in both modes. You prototype interactively, then deploy when it works. No rewriting.

---

## Implementation Architecture

```
script.eyeso
    │
    ▼
┌──────────┐
│  Lexer   │  Tokenizes .eyeso source into token stream
└────┬─────┘
     ▼
┌──────────┐
│  Parser  │  Builds AST (Abstract Syntax Tree) from tokens
└────┬─────┘
     ▼
┌──────────┐
│ Analyzer │  Resolves variables, validates commands, checks types
└────┬─────┘
     │
     ├──── interpreted ────▶ ┌─────────────┐
     │                       │ Interpreter  │  Executes AST nodes sequentially
     │                       │              │  parallel blocks → asyncio.gather
     │                       └─────────────┘
     │
     └──── compiled ───────▶ ┌─────────────┐
                             │  Compiler   │  Transforms AST → TaskDAG
                             │              │  parallel → DAG waves
                             │              │  every → Shadows schedule
                             └──────┬──────┘
                                    ▼
                             ┌─────────────┐
                             │ Orchestrator │  Executes DAG via Octane OSA
                             └─────────────┘
```

### Files

```
octane/eyeso/
├── __init__.py
├── lexer.py          # Tokenizer: source → tokens
├── parser.py         # Parser: tokens → AST
├── ast_nodes.py      # AST node definitions
├── analyzer.py       # Semantic analysis + validation
├── interpreter.py    # Line-by-line execution (interactive + script mode)
├── compiler.py       # AST → TaskDAG (deploy mode)
├── runtime.py        # Built-in functions, variable scope, I/O
├── repl.py           # Interactive REPL (octane eyeso)
└── cli.py            # CLI commands (run, compile, deploy, jobs, stop)
```

---

## Versioning

eyeso follows semantic versioning independently from Octane.

**v0.1** (Session 23): Variables, commands, pipes, parallel blocks, for-each, if/else, try/catch, `??` operator. Interpreter only. No deploy mode.

**v0.2** (Session 24+): Compiler + deploy mode. `every` scheduling blocks. `--preview` DAG visualization. Shadows integration.

**v1.0** (future): Stable syntax. Community script sharing. `octane eyeso install` from marketplace. Full test coverage. Language server for editor support.