# OCTANE_APPLICATIONS_EVERYDAY_USE.md
# Project Octane — 5 Real-World Application Flows
# This document shows what people can actually DO with Octane every day.
# Each flow is traced through the full agent architecture.
# Last updated: 2026-02-22

---

## The 5 Flows

| # | Flow | Who It's For | What It Does |
|---|------|-------------|--------------|
| 1 | **Personal Wealth Manager** | Anyone with $10 to $10M | Grows your portfolio with personalized strategy, risk management, and continuous monitoring |
| 2 | **Deep Research Agent** | Students, researchers, analysts, curious people | Investigates any topic across multiple sources, builds a knowledge base, produces structured reports |
| 3 | **Career Autopilot** | Job seekers, career changers, freelancers | Monitors opportunities, tailors your resume per role, preps you for interviews, tracks applications |
| 4 | **Personal Health & Fitness Intelligence** | Anyone who cares about their health | Tracks nutrition, designs workout plans, monitors trends, adapts based on your progress and preferences |
| 5 | **Content Creation Studio** | Writers, YouTubers, podcasters, marketers | Researches topics, generates outlines, writes drafts, validates facts, manages your content pipeline |

---

# Flow 1: Personal Wealth Manager

## The Vision

You tell Octane: "I have $500 to invest. I'm 28, I can tolerate moderate risk, and I want to build long-term wealth." Octane doesn't just give you generic advice — it builds you a personalized investment thesis, monitors it daily, alerts you when action is needed, and learns your risk tolerance from your actual behavior over time.

Or you say: "I have $10,000 and I need to make a quick return in the next 3 months." Octane shifts strategy entirely — focuses on momentum plays, earnings catalysts, and short-term technical setups, while continuously monitoring your positions and flagging exit signals.

This isn't a robo-advisor black box. You see every decision, every reasoning trace, every data source. You approve or reject trades. Octane learns from your choices.

## Setup

```bash
$ octane pref set domain_interests '["finance", "investing"]'
$ octane pref set risk_tolerance moderate        # conservative | moderate | aggressive
$ octane pref set investment_horizon long_term   # short_term | medium_term | long_term
$ octane pref set portfolio_size 500
$ octane pref set investment_experience beginner  # beginner | intermediate | advanced
```

## Flow A: Initial Portfolio Strategy

### User Query
```bash
$ octane ask "I have $500 to start investing. I'm 28, moderate risk tolerance, 
  long-term horizon. What should I do?"
```

### Step-by-Step Octane Flow

```
═══════════════════════════════════════════════════════════
STEP 1: OSA — DECOMPOSER
═══════════════════════════════════════════════════════════

Decomposer (big model) reasons:
  "This is an investment strategy query. The user is a beginner with
   $500 and a long-term horizon. I need to:
   
   1. Research current market conditions (macro overview)
   2. Research best investment vehicles for small accounts ($500)
   3. Research low-cost index funds and ETFs accessible at $500
   4. Get specific fund data (expense ratios, performance, minimums)
   5. Build a portfolio allocation recommendation
   6. Generate a concrete action plan with specific tickers
   
   The user is a beginner — I should explain concepts clearly but
   not be patronizing (P&L says expertise_level: beginner but
   domain_interest: finance, so they're learning)."

TaskDAG:
  t1: Web.News → "current market conditions macro outlook 2026"
  t2: Web.Search → "best investments for $500 beginner long term 2026"
  t3: Web.Finance → "VOO VTI SCHD QQQ performance expense ratio"
  t4: Web.Search → "fractional shares brokerages for beginners 2026"
  t5: Code.Analyze → "Build portfolio allocation model for $500 
       moderate risk long-term, calculate projected growth"
  t6: OSA.Synthesize → "Create personalized investment plan"

  Parallel: [t1, t2, t3, t4]
  Sequential: [t3, t4] → t5 → t6


═══════════════════════════════════════════════════════════
STEP 2: POLICY ENGINE + HIL
═══════════════════════════════════════════════════════════

Decision Ledger:
  D1-D4: [LOW risk] Web searches → auto_approved
  D5: [MEDIUM risk, 0.75 confidence] Code execution for 
      financial modeling
      → HIL presents:
        "I'd like to run a portfolio growth projection model.
         It will calculate expected returns based on historical
         data for the ETFs I'm recommending.
         [✅ Approve] [❌ Skip — just give recommendations]"
      → User approves

  D6: [LOW risk] Synthesis → auto_approved


═══════════════════════════════════════════════════════════
STEP 3: PARALLEL WEB RESEARCH
═══════════════════════════════════════════════════════════

Web Agent (t1 — Macro Overview):
  Query Strategist generates:
    - "US stock market outlook 2026"
    - "Fed interest rate forecast 2026"
    - "S&P 500 valuation levels current"
  Fetcher → Bodega News API + Search API
  Synthesizer: "Markets are moderately valued. Fed holding rates
    steady. Broad market indices up 8% YTD. Bond yields at 4.2%.
    Most analysts recommend dollar-cost averaging for new investors."

Web Agent (t2 — Beginner Investment Vehicles):
  Query Strategist generates:
    - "best way to invest $500 for beginners 2026"
    - "index fund vs ETF for small accounts"
    - "micro-investing platforms comparison"
  Fetcher → Bodega Search API (Beru)
  Synthesizer: "For $500, low-cost broad market ETFs are ideal.
    Fractional shares available on most platforms. Key options:
    total market (VTI), S&P 500 (VOO), dividend growth (SCHD),
    growth tilt (QQQ). No minimums with fractional shares."

Web Agent (t3 — Specific Fund Data):
  Fetcher → Bodega Finance API:
    GET /api/v1/finance/market/VOO
    GET /api/v1/finance/market/VTI
    GET /api/v1/finance/market/SCHD
    GET /api/v1/finance/market/QQQ
    GET /api/v1/finance/timeseries/VOO?period=5y&interval=1mo
    GET /api/v1/finance/timeseries/VTI?period=5y&interval=1mo
  Returns: current prices, expense ratios, 1yr/3yr/5yr returns,
    dividend yields, monthly price history for modeling

Web Agent (t4 — Brokerage Research):
  Fetcher → Bodega Search API
  Synthesizer: "Fidelity, Schwab, and Robinhood all offer
    fractional shares with no commissions. Fidelity has the
    best research tools for beginners."


═══════════════════════════════════════════════════════════
STEP 4: CODE AGENT — PORTFOLIO MODELING
═══════════════════════════════════════════════════════════

Code Agent receives fund data from t3 + research from t2.

Planner: "Build a Python script that:
  1. Takes $500 initial investment
  2. Models 3 portfolio allocations (conservative, moderate, aggressive)
  3. Uses historical returns from VOO, VTI, SCHD, QQQ
  4. Projects growth over 10, 20, 30 years
  5. Assumes $100/month additional contributions
  6. Accounts for expense ratios
  7. Shows range of outcomes (10th, 50th, 90th percentile)
  8. Outputs a summary table"

Writer generates Monte Carlo simulation script:
  - Uses historical monthly returns from t3 timeseries data
  - Runs 1000 simulations per allocation
  - Calculates compound growth with DCA

Executor runs in sandbox. Output:

  ┌──────────────────────────────────────────────────────────┐
  │ Portfolio Projections: $500 initial + $100/month          │
  │ Risk: Moderate | Horizon: Long-term                       │
  │                                                           │
  │ MODERATE PORTFOLIO (recommended):                         │
  │   40% VTI | 25% VOO | 20% SCHD | 15% QQQ                │
  │   Expense ratio: 0.05% weighted avg                       │
  │                                                           │
  │   10 years:  $18,200 — $24,500 — $33,100  (10/50/90th)   │
  │   20 years:  $52,400 — $82,300 — $131,000 (10/50/90th)   │
  │   30 years:  $142,000 — $268,000 — $512,000(10/50/90th)  │
  │                                                           │
  │ Your $500 + $100/month could grow to ~$268K in 30 years   │
  │ at historical average returns.                            │
  └──────────────────────────────────────────────────────────┘

Validator confirms: calculations match historical return ranges,
  expense ratios correctly applied, Monte Carlo distribution
  is reasonable.


═══════════════════════════════════════════════════════════
STEP 5: SYNTHESIS + PERSONALIZED OUTPUT
═══════════════════════════════════════════════════════════

Evaluator assembles everything and generates:

  "Here's your personalized investment plan:

   Starting point: $500 + $100/month contributions
   Strategy: Long-term wealth building via low-cost index ETFs
   Risk level: Moderate

   Recommended portfolio:
     40% VTI (Total US Market)    — $200
     25% VOO (S&P 500)            — $125
     20% SCHD (Dividend Growth)   — $100
     15% QQQ (Nasdaq/Growth)      — $75

   Why this mix: VTI gives you the entire US market. VOO
   concentrates on large caps for stability. SCHD adds dividend
   income that compounds over time. QQQ gives growth exposure
   without going all-in on tech.

   Total annual cost: ~$0.25 per $500 invested (0.05% avg)

   Projected outcomes with $100/month:
     10 years: ~$24,500 (median)
     20 years: ~$82,300 (median)
     30 years: ~$268,000 (median)

   Next steps:
   1. Open a Fidelity account (best research tools for beginners)
   2. Buy fractional shares in the allocation above
   3. Set up automatic $100/month investment
   4. Tell me to monitor this portfolio — I'll track it daily

   📊 Full projection model: /tmp/octane-sandboxes/.../projection.py
   📈 Growth chart: /tmp/octane-sandboxes/.../growth_chart.png"


═══════════════════════════════════════════════════════════
STEP 6: MEMORY + P&L LEARNING
═══════════════════════════════════════════════════════════

Memory Agent stores:
  Hot (Redis): Full recommendation + portfolio allocation
  Warm (Postgres): {
    portfolio: { VOO: 25%, VTI: 40%, SCHD: 20%, QQQ: 15% },
    initial_investment: 500,
    monthly_contribution: 100,
    risk_tolerance: "moderate",
    horizon: "long_term",
    created: "2026-02-22"
  }

P&L records: user engaged with finance + code modeling workflow.
```

## Flow B: Daily Portfolio Monitoring (Perpetual Task)

Once the user sets up monitoring, this runs automatically every morning:

```bash
$ octane ask "Monitor my portfolio daily and alert me if anything needs attention"
```

**OSA registers a Shadows Perpetual Task:**

```python
async def portfolio_monitor(
    perpetual: Perpetual = Perpetual(every=timedelta(hours=8))
) -> None:
    # Runs 3x daily: morning, midday, closing
```

**Each execution:**
```
1. Memory Agent → retrieve portfolio allocation from warm tier
2. Web Agent.Finance → fetch current prices for VOO, VTI, SCHD, QQQ
3. Web Agent.News → search for news affecting held positions
4. Code Agent → calculate:
   - Daily P&L
   - Portfolio drift from target allocation (rebalancing signal)
   - Volatility spike detection
   - Comparison to benchmark (S&P 500)
5. Policy Engine evaluates alerts:
   - Position down >5% in a day → HIGH alert
   - Portfolio drift >10% from target → MEDIUM alert (suggest rebalance)
   - Dividend payment detected → LOW info
   - Major news affecting holdings → MEDIUM alert
6. If alerts triggered → notify user via CLI next time they open octane
7. If no alerts → log silently to Memory for trend tracking
```

**What the user sees when they open their terminal:**

```
$ octane health

🔥 Octane v0.1.0 | RAM: 24.1/64.0 GB | CPU: 12%

📊 Portfolio Alert (2 notifications since last check):

  ⚠️ MEDIUM — Portfolio drift detected
     QQQ is now 22% of your portfolio (target: 15%)
     due to QQQ outperforming other holdings by 8% this month.
     Recommendation: Sell $34 QQQ, buy $20 VTI + $14 SCHD
     [Review rebalance plan →]

  ℹ️ LOW — SCHD dividend payment
     $3.42 dividend received. Reinvested automatically if DRIP enabled.

  Overall: Portfolio $612.40 (+22.5% since inception)
```

## Flow C: Quick Return Strategy (Different User Profile)

```bash
$ octane pref set risk_tolerance aggressive
$ octane pref set investment_horizon short_term
$ octane ask "I have $10,000 and I need maximum returns in 90 days. What are my options?"
```

**Octane's approach changes completely** because P&L profile is different:

```
Decomposer reasoning:
  "Short-term aggressive strategy. This is higher risk. I need:
   1. Current market momentum data (which sectors are trending)
   2. Upcoming earnings catalysts in the next 90 days
   3. Technical analysis signals (RSI, moving averages)
   4. Options strategies for leveraged plays
   5. CRITICAL: Clear risk warnings — user could lose money fast
   
   Guard check: This is legitimate financial research, not manipulation.
   But I should emphasize risk disclosure prominently."

Web Agent research:
  - Sector momentum: AI/datacenter still leading, energy rotating in
  - Earnings in next 90 days: NVDA (March), AAPL (April), MSFT (April)
  - Technical signals: NVDA oversold (RSI 32), potential bounce candidate
  - Options: covered calls on high-IV names for income, bull call spreads

Code Agent:
  - Backtests momentum strategy on last 2 years of data
  - Calculates expected return distribution for aggressive allocation
  - Models worst-case scenarios (max drawdown analysis)

Output includes prominent risk section:
  "⚠️ IMPORTANT: Short-term aggressive strategies carry significant
   risk. Historical backtests show this approach can lose 15-30%
   in adverse conditions. Only invest money you can afford to lose.
   
   That said, here are 3 strategies ranked by risk/reward..."
```

---

# Flow 2: Deep Research Agent

## The Vision

You're writing a paper, preparing a presentation, investigating a topic, or just deeply curious about something. You tell Octane: "Research the impact of AI on pharmaceutical drug discovery." Octane doesn't just Google it — it conducts a multi-source investigation, cross-references findings, identifies contradictions, builds a structured knowledge base, and produces a report you can actually cite.

Over time, as you research related topics, Octane's Memory Agent connects the dots. Ask about "AI in protein folding" next week, and Octane already has context from your drug discovery research.

## User Query

```bash
$ octane ask "I'm writing a paper on how AI is transforming pharmaceutical drug 
  discovery. Do a deep investigation — find recent breakthroughs, key companies, 
  success stories, failures, and the current state of the field. I need this to 
  be thorough and citable."
```

## The Full Flow

```
═══════════════════════════════════════════════════════════
DECOMPOSER
═══════════════════════════════════════════════════════════

"This is a deep research query. The user needs thoroughness and
 citability. I need to cast a wide net across multiple source types:

 1. Recent news and press releases (last 6 months)
 2. Academic/scientific sources (research papers, reviews)
 3. Company-specific information (key players, pipelines)
 4. Regulatory data (FDA approvals involving AI)
 5. Failure cases (drug candidates that AI predicted would work but didn't)
 6. Industry analysis and market data
 
 Then I need to:
 7. Cross-reference and validate claims across sources
 8. Structure findings into a citable report
 9. Store everything in Memory for future research sessions"

TaskDAG (9 tasks, 6 parallel):

  Parallel group 1 — Data gathering:
    t1: Web.Search → "AI drug discovery breakthroughs 2025 2026"
    t2: Web.Search → "AI pharmaceutical companies pipeline results"
    t3: Web.Search → "AI drug discovery clinical trial failures limitations"
    t4: Web.Search → "FDA approved drugs developed using AI"
    t5: Web.Search → "AI protein folding AlphaFold drug design papers"
    t6: Web.Finance → "AI drug discovery companies stock performance market cap"
         (Recursion, Insilico Medicine, AbCellera, Relay Therapeutics, etc.)

  Sequential:
    t7: Code.Validate → cross-reference claims, identify contradictions,
        verify company claims against actual trial data
    t8: OSA.Synthesize → structured report with sections
    t9: Memory.Write → store full research corpus with tags


═══════════════════════════════════════════════════════════
PARALLEL RESEARCH EXECUTION
═══════════════════════════════════════════════════════════

Each Web Agent call goes through the full sub-agent pipeline:

Web Agent (t1 — Breakthroughs):
  Query Strategist generates 3 search variations:
    - "AI drug discovery breakthrough 2025 2026"
    - "machine learning pharmaceutical new drug approval"
    - "generative AI molecule design clinical trial success"
  
  Fetcher hits Bodega Search API with each variation.
  Gets ~30 results across all three searches.
  
  Browser Agent (Playwright) activated for 2 URLs that require
  JS rendering (Nature.com article, Science.org paper).
  
  Synthesizer processes all results:
    "Key breakthroughs identified:
     1. Insilico Medicine's INS018-055 — first fully AI-designed
        drug to reach Phase II trials for idiopathic pulmonary fibrosis
     2. Recursion's REC-994 — AI-identified repurposed drug for
        cerebral cavernous malformation, Phase II
     3. AbSci's de novo antibody design platform generating novel
        therapeutic antibodies in 6 weeks vs 6 months
     [... 8 more breakthroughs with source URLs]"

Web Agent (t2 — Companies):
  Synthesizer output:
    "Key companies and their pipelines:
     - Recursion Pharmaceuticals (RXRX): 50+ programs, proprietary
       BioHive supercomputer, partnered with Roche ($150M deal)
     - Insilico Medicine: end-to-end AI platform, 30+ pipeline assets
     - Isomorphic Labs (Google DeepMind spinoff): AlphaFold-based
       drug design, partnered with Eli Lilly and Novartis
     - AbCellera (ABCL): AI antibody discovery, 170+ programs
     [... with market cap, funding, key partnerships]"

Web Agent (t3 — Failures and Limitations):
  Synthesizer output:
    "Notable limitations and failures:
     1. BenevolentAI's atopic dermatitis drug (BEN-2293) failed
        Phase IIa — AI predicted efficacy but clinical reality differed
     2. Exscientia's EXS-21546 discontinued after Phase I — safety
        concerns not predicted by computational models
     3. General limitation: AI excels at target identification and
        molecule design but still poor at predicting ADMET properties
        and off-target effects in human biology
     4. Data quality problem: AI models trained on biased datasets
        (mostly Caucasian populations) may not generalize"

Web Agent (t4 — FDA Approvals):
  Specific regulatory data with dates and approval numbers.

Web Agent (t5 — Papers):
  Academic sources with DOIs, author names, publication dates.

Web Agent (t6 — Financial Data):
  Bodega Finance API returns: stock performance, market caps,
  funding rounds for all AI pharma companies mentioned.


═══════════════════════════════════════════════════════════
CODE AGENT — CROSS-REFERENCE AND VALIDATION
═══════════════════════════════════════════════════════════

Code Agent (t7) receives ALL web results and:

1. Extracts all factual claims with source attribution
2. Cross-references: "Insilico's INS018-055 in Phase II"
   → Confirmed by 3 independent sources (Reuters, FierceBiotech, Nature)
3. Identifies contradictions:
   "Source A says Recursion has 50+ programs, Source B says 30+.
    Most recent source (company Q4 report) confirms 50+."
4. Verifies company claims against actual ClinicalTrials.gov data
5. Flags unverifiable claims:
   "AbSci's claim of '6 weeks vs 6 months' — only source is 
    company press release. Flagged as unverified."
6. Generates a credibility score per claim (multi-source confirmed,
   single source, company-only claim)

Output: structured JSON of validated claims with confidence levels.


═══════════════════════════════════════════════════════════
SYNTHESIS — STRUCTURED REPORT
═══════════════════════════════════════════════════════════

Evaluator produces a structured report:

  "# AI in Pharmaceutical Drug Discovery: Current State (2026)

   ## Executive Summary
   AI-driven drug discovery has moved from theoretical promise to
   clinical reality, with 15+ AI-designed drugs now in clinical trials...

   ## Key Breakthroughs (2024-2026)
   [Validated claims with source citations]

   ## Major Players and Their Pipelines
   [Company profiles with financial data from t6]

   ## Success Stories
   [Phase II+ drugs with confirmed clinical data]

   ## Notable Failures and Limitations
   [Honest assessment of where AI falls short]

   ## Industry Outlook
   [Market projections, expert opinions, trend analysis]

   ## Sources
   [All URLs, paper DOIs, with credibility scores]
   
   ⚠️ Claims marked with [*] are single-source or company-reported
   and should be independently verified."


═══════════════════════════════════════════════════════════
MEMORY — KNOWLEDGE BASE
═══════════════════════════════════════════════════════════

Memory Agent stores:
  Cold tier (pgVector): All research chunks with embeddings
    Tags: ["AI", "pharma", "drug_discovery", "research"]
  Warm tier (Postgres): Structured company profiles, financial data
  Hot tier (Redis): Full report for immediate follow-up

Future benefit: When user researches "AI in protein engineering"
  next month, Memory surfaces relevant prior research automatically.
  Knowledge compounds over time.
```

## Follow-Up Interactions

```bash
$ octane ask "Focus on Insilico Medicine specifically — give me everything"
# → Memory retrieves prior context. Web Agent does targeted deep dive.
# → No re-research on general landscape. Builds on existing knowledge.

$ octane ask "Compare the AI approaches of Recursion vs Isomorphic Labs"
# → Memory has both company profiles. Only fetches recent news updates.
# → Code Agent generates comparison matrix.

$ octane ask "Draft an introduction paragraph for my paper using this research"
# → Memory serves the full research corpus.
# → Synthesis generates academic-tone paragraph with citations.
# → P&L knows user wants citable, formal writing.
```

---

# Flow 3: Career Autopilot

## The Vision

Job searching is one of the most time-consuming, emotionally draining activities people go through. You're refreshing LinkedIn 50 times a day, tailoring resumes for each application, researching companies before interviews, and trying to negotiate salary — all while potentially still working full-time.

Octane automates the grunt work. You tell it your target role, salary range, and preferences. It monitors job boards continuously, matches opportunities to your profile, tailors your resume for each application, researches companies before interviews, and even helps you prep for specific interview formats. All locally — your career data never leaves your machine.

## Setup

```bash
$ octane ask "I'm a senior frontend engineer with 7 years of experience in 
  React and TypeScript. I want to move into a principal engineer or engineering 
  manager role. Target salary: $250K+. Preferred: remote or SF Bay Area. 
  Industries I like: AI/ML tooling, developer tools, fintech."

# Octane stores this in P&L + Memory and sets up monitoring
```

## Flow A: Continuous Opportunity Monitoring

**Shadows Perpetual Task — runs every 6 hours:**

```
═══════════════════════════════════════════════════════════
PERPETUAL: JOB MONITOR
═══════════════════════════════════════════════════════════

Every 6 hours:

1. Memory Agent → retrieve user career profile
   { role: "principal engineer / eng manager",
     experience: "7yr React/TS",
     salary: "$250K+",
     location: "remote / SF Bay",
     industries: ["AI/ML tooling", "dev tools", "fintech"] }

2. Web Agent → search multiple sources:
   - "principal engineer React remote $250K+"
   - "engineering manager frontend developer tools SF Bay Area"
   - "senior staff engineer AI ML tooling company"
   
   Browser Agent (Playwright) → scrapes:
   - LinkedIn job search (login handled by user via HIL first time)
   - levels.fyi for salary benchmarking
   - Company career pages for target companies

3. Code Agent → scoring and matching:
   For each job found, calculate match score:
   - Role title match (principal/staff/EM): 0-25 points
   - Tech stack overlap (React, TS, frontend): 0-25 points
   - Salary range overlap: 0-20 points
   - Location match: 0-15 points
   - Industry match: 0-15 points
   
   Rank all jobs by total score. Filter: score >= 60/100.

4. Memory Agent → check against previously seen jobs
   - Deduplicate: skip jobs already surfaced
   - Track: mark jobs as "new", "still open", "closed"

5. If new high-scoring jobs found → queue notification

User opens Octane next morning:

  $ octane health

  💼 Career Monitor: 3 new matches since yesterday

  🟢 STRONG MATCH (92/100)
     Principal Engineer, Frontend Platform
     Vercel — Remote (US)
     $280K-$340K | React, TypeScript, Next.js
     Posted: 2 hours ago
     Why: Perfect role + stack match. Salary exceeds target.
     Company is in your "dev tools" interest area.
     [📝 Tailor resume] [🔍 Research company] [⏭ Skip]

  🟢 STRONG MATCH (85/100)
     Engineering Manager, Web Platform
     Anthropic — SF / Remote
     $300K-$380K | React, TypeScript, team leadership
     Posted: 12 hours ago
     Why: EM role at AI company. Strong salary. Your AI/ML interest.
     [📝 Tailor resume] [🔍 Research company] [⏭ Skip]

  🟡 GOOD MATCH (71/100)
     Staff Frontend Engineer
     Stripe — SF
     $250K-$310K | React, design systems
     Posted: 1 day ago
     Why: Fintech match. Slightly below principal level but
          Stripe promotes internally. Worth considering.
     [📝 Tailor resume] [🔍 Research company] [⏭ Skip]
```

## Flow B: One-Click Resume Tailoring

User clicks `[📝 Tailor resume]` for the Vercel role:

```
═══════════════════════════════════════════════════════════
RESUME TAILORING PIPELINE
═══════════════════════════════════════════════════════════

1. Memory Agent → retrieve user's base resume (stored in warm tier)

2. Web Agent → fetch full job description for Vercel Principal Engineer
   Browser Agent scrapes the full JD via Playwright

3. Code Agent → analyze JD and extract:
   - Required skills: React, TypeScript, Next.js, performance optimization
   - Preferred skills: design systems, A/B testing, CI/CD
   - Key phrases: "developer experience", "platform engineering",
     "cross-functional leadership", "technical vision"
   - Culture signals: "open source", "community", "craft"

4. OSA.Evaluator (big model) → rewrite resume sections:
   - Reorders experience bullets to lead with most relevant work
   - Adjusts language to mirror JD phrases naturally
   - Emphasizes: platform engineering, DX improvements, Next.js if applicable
   - Adds quantified impacts: "Improved build times by 60%"
   - Adjusts summary to target principal-level narrative
   
   HIL Checkpoint:
     "I've tailored your resume for Vercel's Principal Engineer role.
      Key changes:
      • Moved design system work to top of experience
      • Added Next.js projects from your side work
      • Reframed team lead experience as technical leadership
      • Emphasized DX and platform impact metrics
      
      [✅ Use this version] [✏️ Edit before saving] [👀 Show diff]"

5. Memory Agent → store tailored version
   Tag: "resume:vercel:principal_engineer:2026-02-22"

6. Output: tailored resume as both text preview and downloadable document
```

## Flow C: Interview Prep

```bash
$ octane ask "I have an interview with Vercel for the Principal Engineer role 
  next Tuesday. Prep me."
```

```
OSA decomposes into comprehensive prep pipeline:

1. Web Agent → company deep dive:
   - Vercel's recent product launches, funding, leadership
   - Glassdoor interview experiences for similar roles
   - Vercel's tech blog (what they care about technically)
   - Recent conference talks by Vercel engineers
   - Competitors and market position (Netlify, Cloudflare Pages)

2. Web Agent → role-specific research:
   - "Principal engineer interview questions frontend"
   - "System design interview frontend platform"
   - Common Vercel interview format (phone screen → system design → 
     culture fit → team match)

3. Code Agent → generate practice problems:
   - System design: "Design a global edge deployment platform"
   - Technical deep dive: "Optimize React hydration for streaming SSR"
   - Leadership scenario: "How would you migrate a monolith to micro-frontends
     across 5 teams?"

4. Memory Agent → retrieve user's resume + Vercel JD from prior session

5. Synthesis → comprehensive prep package:
   
   "INTERVIEW PREP: Vercel Principal Engineer
   
    Company Brief:
    [2-paragraph summary of Vercel's current state]
    
    What They're Looking For:
    [Mapped from JD + Glassdoor + tech blog analysis]
    
    Likely Interview Format:
    [Based on Glassdoor reports: 5 rounds...]
    
    Practice Questions (10):
    [Technical, system design, behavioral, leadership]
    
    Your Strongest Talking Points:
    [Mapped from YOUR resume to THEIR needs]
    
    Questions to Ask Them:
    [5 thoughtful questions based on company research]
    
    Salary Negotiation Data:
    [levels.fyi data for Vercel principal eng: $280-$380K]
    [Your leverage points and negotiation strategy]"
```

---

# Flow 4: Personal Health & Fitness Intelligence

## The Vision

You want to get healthier, but you're overwhelmed by conflicting information. Should you do keto or Mediterranean? HIIT or strength training? How much protein do you actually need? Every fitness influencer says something different.

Octane becomes your personal health analyst. You tell it your goals, current stats, and constraints. It builds evidence-based plans, adapts them based on your progress, and cuts through the noise with actual scientific research — not bro-science.

All health data stays on YOUR machine. No cloud fitness app selling your data to insurance companies.

## Setup

```bash
$ octane ask "I'm setting up my health profile. I'm 32, male, 185 lbs, 5'11. 
  I want to lose 15 lbs of fat while building muscle. I can work out 4 days 
  per week, 45 minutes each. I have access to a full gym. No injuries. 
  I eat out 3-4 times per week. Budget for supplements: $50/month."
```

## Flow A: Personalized Program Design

```
═══════════════════════════════════════════════════════════
DECOMPOSER
═══════════════════════════════════════════════════════════

"Health and fitness program design. I need:
 1. Evidence-based research on body recomposition (fat loss + muscle gain)
 2. Workout program design for 4x/week hypertrophy + fat loss
 3. Nutrition plan with caloric targets and macro splits
 4. Supplement recommendations within $50/month budget
 5. Progress tracking metrics and timeline expectations
 
 This involves health information — Guard should ensure I include
 appropriate disclaimers. I should cite scientific sources, not
 fitness influencer content."

TaskDAG:
  Parallel:
    t1: Web.Search → "body recomposition research evidence based 2025 2026"
    t2: Web.Search → "4 day workout split hypertrophy fat loss program"
    t3: Web.Search → "protein intake body recomposition 185 lbs male"
    t4: Web.Search → "evidence based supplements fat loss muscle gain"
  
  Sequential:
    t5: Code.Calculate → TDEE, macros, caloric targets, weekly plan
    t6: OSA.Synthesize → complete program


═══════════════════════════════════════════════════════════
WEB RESEARCH (PARALLEL)
═══════════════════════════════════════════════════════════

Web Agent prioritizes scientific sources over fitness blogs:

t1 — Body Recomposition Research:
  Query Strategist prioritizes: pubmed, examine.com, systematic reviews
  Synthesizer: "Research supports body recomposition is achievable for
    intermediate lifters at a moderate caloric deficit (300-500 cal).
    Key factors: high protein (1g/lb), progressive overload, adequate
    sleep. Timeline: expect 1-2 lbs fat loss per week while maintaining
    or slowly gaining muscle. Source: Barakat et al. (2020) systematic review."

t2 — Workout Programming:
  Synthesizer: "Evidence supports Upper/Lower split 4x/week for
    concurrent hypertrophy and fat loss. Key principles:
    - Compound movements first (squat, bench, deadlift, OHP, rows)
    - 3-4 sets of 8-12 reps for hypertrophy
    - Progressive overload (add weight or reps weekly)
    - 45-min sessions: 4 compounds + 2 accessories"

t3 — Protein Research:
  Synthesizer: "Meta-analysis by Morton et al. (2018): 0.73g/lb is
    the upper threshold for muscle protein synthesis. For recomp,
    1g/lb (185g daily) provides safety margin. Spread across 4 meals."

t4 — Supplements:
  Synthesizer: "Evidence-based tier list within $50/month:
    Tier 1 (strong evidence): Creatine monohydrate ($10), Vitamin D ($8)
    Tier 2 (moderate evidence): Whey protein ($25), Magnesium ($7)
    Total: $50/month. Everything else has weak or no evidence."


═══════════════════════════════════════════════════════════
CODE AGENT — PERSONALIZED CALCULATIONS
═══════════════════════════════════════════════════════════

Code Agent generates personalized nutrition script:

Input: age=32, weight=185, height=71in, sex=male, activity=4x/week
Output:
  BMR (Mifflin-St Jeor): 1,892 cal
  TDEE (moderate activity): 2,932 cal
  Target (300 cal deficit): 2,632 cal/day
  
  Macro split (high protein recomp):
    Protein: 185g (740 cal, 28%)
    Fat: 73g (657 cal, 25%)
    Carbs: 309g (1,235 cal, 47%)
  
  Per meal (4 meals/day):
    Protein: 46g | Fat: 18g | Carbs: 77g | Total: 658 cal
  
  Projected timeline:
    Fat loss: 1.5 lbs/week → 10 weeks to lose 15 lbs
    Expected weight at 10 weeks: ~172 lbs (accounting for muscle gain)
    Realistic range: 170-175 lbs


═══════════════════════════════════════════════════════════
SYNTHESIS — COMPLETE PROGRAM
═══════════════════════════════════════════════════════════

"YOUR PERSONALIZED PROGRAM

 ⚠️ Disclaimer: This is not medical advice. Consult a healthcare
 provider before starting any new exercise or nutrition program.

 NUTRITION (2,632 cal/day):
   Protein: 185g | Fat: 73g | Carbs: 309g
   
   Eating out strategy (3-4x/week):
   - Prioritize protein at every restaurant meal
   - Most restaurants: grilled chicken/fish + rice + vegetables
   - Ask for dressings/sauces on the side
   - One meal out can be 'flexible' — don't stress about macros

 WORKOUT (4x/week, 45 min):
   Monday — Upper Strength
     Bench Press 4×6-8
     Barbell Row 4×6-8
     Overhead Press 3×8-10
     Lat Pulldown 3×10-12
     Bicep Curl 2×12-15
     Tricep Pushdown 2×12-15

   Tuesday — Lower Strength
     Squat 4×6-8
     Romanian Deadlift 3×8-10
     Leg Press 3×10-12
     Walking Lunges 3×12/leg
     Calf Raises 3×15-20

   Thursday — Upper Hypertrophy
     Incline Dumbbell Press 4×10-12
     Cable Row 4×10-12
     Lateral Raises 3×12-15
     Face Pulls 3×15-20
     Hammer Curl 3×10-12
     Overhead Tricep Extension 3×10-12

   Friday — Lower Hypertrophy
     Front Squat 3×8-10
     Hip Thrust 4×10-12
     Leg Curl 3×10-12
     Leg Extension 3×12-15
     Calf Raises 3×15-20

 SUPPLEMENTS ($50/month):
   Creatine Monohydrate: 5g/day ($10)
   Whey Protein: 1 scoop post-workout ($25)
   Vitamin D3: 2000 IU/day ($8)
   Magnesium Glycinate: 200mg before bed ($7)

 TRACKING:
   Tell me to track your progress weekly. I'll ask for:
   - Weight (same time, same conditions)
   - Key lift numbers (bench, squat, deadlift)
   - How you feel (energy, sleep, mood)
   And I'll adjust the program based on your trajectory."
```

## Flow B: Weekly Check-in and Adaptation

```bash
$ octane ask "Weekly check-in: weight 183, bench went from 185 to 190, 
  squat 225 to 230. Sleep has been bad this week — averaging 5.5 hours. 
  Energy is low."
```

```
Memory Agent → retrieves prior weeks' data + program
Code Agent → plots weight trend, lift progression, calculates rates
P&L → knows user's communication style from past interactions

Evaluator synthesizes:

  "WEEK 3 CHECK-IN
   
   ✅ Weight: 183 lbs (-2 lbs from start). On track for ~1 lb/week.
   ✅ Bench: 190 (+5). Good progression.
   ✅ Squat: 230 (+5). Solid.
   
   🚨 Sleep: 5.5 hours is a problem.
      Research shows <6 hours sleep reduces muscle protein synthesis
      by 18% and increases cortisol (promotes fat storage). This is
      actively working against your goals.
      
      Priority adjustment: Sleep > nutrition > training right now.
      
      Recommendations:
      - Magnesium glycinate before bed (you already have this)
      - No caffeine after 2 PM
      - Phone in another room by 10 PM
      - If you can only train 3x this week, that's fine — skip Friday
      
      Your training and nutrition are dialed in. Sleep is the
      bottleneck. Fix that and everything accelerates.
   
   📊 Progress chart: [shows weight trajectory + projected path]"
```

---

# Flow 5: Content Creation Studio

## The Vision

You're a content creator — maybe a blogger, YouTuber, newsletter writer, or social media marketer. Every week you need ideas, research, outlines, drafts, fact-checking, and scheduling. Octane handles the entire pipeline from "I need to write about something" to "here's a polished draft with verified facts."

## Setup

```bash
$ octane pref set content_type newsletter        # blog | newsletter | youtube | podcast
$ octane pref set audience "tech professionals"
$ octane pref set tone "conversational but informed"
$ octane pref set publish_schedule weekly
$ octane pref set topics '["AI", "software engineering", "developer tools", "career growth"]'
```

## Flow A: Topic Discovery + Research

```bash
$ octane ask "I need to write my weekly newsletter. What should I write about 
  this week? Give me 5 topic ideas with research for each."
```

```
═══════════════════════════════════════════════════════════
DECOMPOSER
═══════════════════════════════════════════════════════════

"Content ideation query. I need to:
 1. Find trending topics in the user's areas (AI, software eng, dev tools)
 2. Check what major publications covered this week
 3. Cross-reference with what the user has written before (Memory)
    to avoid repetition
 4. Score topics by: timeliness, audience interest, uniqueness
 5. For each topic, do preliminary research so the user can 
    choose informed"

TaskDAG:
  Parallel — trend scanning:
    t1: Web.News → "AI news this week 2026"
    t2: Web.News → "software engineering trends this week"
    t3: Web.News → "developer tools launches updates this week"
    t4: Web.Search → "Hacker News top stories this week"
    t5: Web.Search → "most discussed tech topics Twitter/X this week"
  
  Context:
    t6: Memory.Read → "user's past newsletter topics" (avoid repeats)
    t7: Memory.Read → "user's audience engagement data" (what resonated)
  
  Sequential:
    t8: Code.Score → rank topics by timeliness × uniqueness × audience fit
    t9: OSA.Synthesize → 5 topic pitches with preliminary research


═══════════════════════════════════════════════════════════
RESEARCH + SCORING
═══════════════════════════════════════════════════════════

Web Agent scans all sources. Code Agent scores and ranks.

Memory Agent provides:
  Past topics: ["AI code generation tools", "React Server Components",
    "The future of TypeScript", "DevOps is dead, long live platform eng"]
  High engagement: topics with "practical how-to" angle performed 2x
    better than "opinion/prediction" pieces

Code Agent scoring:
  For each potential topic:
    Timeliness score (0-30): How current? Breaking this week?
    Uniqueness score (0-30): Has the user covered this before?
      Have competitors covered it to death?
    Audience fit (0-20): Does it match "tech professionals" audience?
    Practical angle (0-20): Can this be a "how-to" or actionable piece?
      (bonus: user's audience prefers practical content)


═══════════════════════════════════════════════════════════
OUTPUT — 5 TOPIC PITCHES
═══════════════════════════════════════════════════════════

"Here are 5 topic ideas for this week, ranked by potential:

 1. ⭐ SCORE: 92/100
    'Claude Code and the Rise of Terminal-First AI Development'
    Hook: Anthropic launched Claude Code this month — a command-line
    AI coding tool. This signals a shift from IDE plugins to
    terminal-native AI. Your audience of engineers will care.
    
    Preliminary research:
    - 3 articles on Claude Code launch
    - Comparison data: Claude Code vs Cursor vs Copilot
    - Developer sentiment from HN (mixed but curious)
    Angle: 'I tried Claude Code for a week. Here's what changed.'
    Estimated write time: 3 hours with Octane research assist

 2. ⭐ SCORE: 87/100
    'The Hidden Cost of AI-Generated Code: Technical Debt at Scale'
    Hook: Multiple companies are reporting increases in code review
    time and subtle bugs from AI-generated code. Counternarrative
    to the 'AI makes developers 10x faster' hype.
    
    Preliminary research:
    - GitClear study on AI code quality trends
    - 2 engineering blog posts on AI code review challenges
    - Stack Overflow developer survey data
    Angle: Practical guide to AI code review workflows
    Estimated write time: 4 hours

 3. ⭐ SCORE: 81/100
    [...]

 4. ⭐ SCORE: 74/100
    [...]

 5. ⭐ SCORE: 68/100
    [...]

 Which topic do you want to develop? I'll do deep research and
 generate a full outline."
```

## Flow B: Deep Dive + Draft

```bash
$ octane ask "Let's go with topic 1 — Claude Code. Do deep research and 
  write me a full first draft."
```

```
═══════════════════════════════════════════════════════════
DEEP RESEARCH PIPELINE
═══════════════════════════════════════════════════════════

Web Agent does targeted deep research:
  - Full Claude Code documentation and feature set
  - Hands-on review articles (5+ sources)
  - Developer forum discussions (HN, Reddit, X)
  - Competitor comparison data (Cursor, Copilot, Windsurf)
  - Usage statistics if available
  - Anthropic's blog post and positioning

Browser Agent (Playwright) scrapes:
  - Claude Code docs (JS-rendered site)
  - Specific HN discussion threads

Code Agent:
  - Validates feature comparison claims across sources
  - Generates comparison table data
  - Fact-checks any statistics mentioned in sources

Memory Agent:
  - Checks if user has covered Cursor or Copilot before
  - Retrieves user's writing style from past newsletters
  - Stores all research for this topic


═══════════════════════════════════════════════════════════
OUTLINE GENERATION
═══════════════════════════════════════════════════════════

OSA.Evaluator (big model) generates outline:

  "OUTLINE: Claude Code and Terminal-First AI Development

   I. Hook (2 paragraphs)
      - 'I've been using AI coding tools since Copilot launched in 2021.
         This week I tried something different.'
      - Set up the shift: IDE plugins → terminal-native AI

   II. What is Claude Code? (3 paragraphs)
       - What it does (command-line AI agent)
       - How it's different from Copilot/Cursor (agent vs autocomplete)
       - Key capabilities: file editing, terminal commands, git workflow

   III. My Experience (4 paragraphs) — NEEDS USER INPUT
        - [placeholder: user's actual experience needed]
        - What worked well
        - What didn't work
        - The 'aha moment'

   IV. The Bigger Trend: Terminal-First AI (3 paragraphs)
       - Why terminal is having a renaissance
       - Developer workflow is moving back to CLI
       - Implications for IDE vendors

   V. Practical Comparison (table + 2 paragraphs)
      - Claude Code vs Cursor vs Copilot: features, pricing, approach
      - When to use which

   VI. What This Means For You (2 paragraphs)
       - Actionable takeaway for the reader
       - Try this specific workflow this week

   [✅ Approve outline] [✏️ Modify sections] [🔄 Different angle]"

User: ✅ Approve

═══════════════════════════════════════════════════════════
DRAFT GENERATION
═══════════════════════════════════════════════════════════

Evaluator writes full draft section by section, using:
  - Research data for factual claims
  - User's past writing style (from P&L + Memory)
  - User's "conversational but informed" tone preference
  - Comparison table from Code Agent validation

HIL Checkpoint at Section III:
  "Section III is about YOUR personal experience with Claude Code.
   I can write a template based on common experiences, but it'll
   be much better if you give me your actual experience.
   
   [✏️ I'll write my experience — give me the template]
   [⏭ Skip personal section — make it more general]
   [📝 Here's my experience: ___]"

User provides their experience. Draft continues.

Output:
  "Full draft: 1,847 words
   Sections: 6 (all complete)
   Facts verified: 12/12 claims checked against sources
   Reading time: ~8 minutes
   
   📄 Draft: /tmp/octane-content/claude-code-newsletter-v1.md
   
   Suggested edits:
   - Paragraph 4 could be punchier — consider starting with the quote
   - The comparison table might need a 'pricing' column
   - Your closing CTA could be more specific
   
   [✏️ Edit in place] [📤 Export to Google Docs] [🔄 Revise section ___]"
```

## Flow C: Fact-Checking and Final Polish

```bash
$ octane ask "Fact-check this draft and make sure every claim is accurate"
```

```
Code Agent scans the draft and extracts all factual claims:

  Claim: "Claude Code was released in February 2026"
  → Verified: ✅ (Anthropic blog post, multiple news sources)

  Claim: "Cursor has over 1 million active developers"
  → Verified: ⚠️ (Source is a Cursor press release from Q3 2025.
     Current number may be higher. Suggest: "over 1 million as of late 2025")

  Claim: "Terminal-based workflows reduce context switching by 40%"
  → NOT VERIFIED: ❌ (No credible source found for this statistic.
     Suggest removing or rephrasing as qualitative observation)

  [... checks all 12 claims]

  "3 claims verified ✅, 1 needs update ⚠️, 1 should be removed ❌
   Updated draft with corrections applied.
   
   All changes tracked — review diff?
   [👀 Show changes] [✅ Accept all] [✏️ Review one by one]"
```

---

## Summary: What Makes These Flows Powerful

Every flow demonstrates the same Octane principles:

**Composable agents:** The same Web Agent, Code Agent, Memory Agent, and OSA serve ALL five flows. No custom code per application — just different TaskDAGs.

**Memory compounds:** Research from your drug discovery investigation helps your newsletter about AI in healthcare. Your portfolio monitoring data feeds into your financial newsletter. Knowledge doesn't stay in silos.

**P&L personalization:** Octane writes differently for a finance newsletter vs an interview prep document vs a workout plan. It learns your tone, your detail preferences, your communication style.

**HIL keeps you in control:** Octane doesn't execute trades, send applications, or publish content without your approval. It does the research and preparation, you make the decisions.

**Everything is local:** Your financial data, health metrics, career information, and unpublished content never leave your machine. No cloud service is selling your portfolio strategy to hedge funds or your health data to insurance companies.

**Workflows are shareable:** Someone builds a "Earnings Analysis" workflow template. You import it and it works with YOUR data, YOUR preferences, YOUR local models. The workflow is portable, the data is private.

```bash
# Anyone can import and use a community workflow:
$ octane workflow install community/earnings-analysis.json
$ octane workflow run earnings-analysis --ticker NVDA

# Or create their own:
$ octane workflow export my-morning-briefing.json
$ octane workflow publish my-morning-briefing.json --public
```

---

## The Common Thread

All five flows follow the same Octane pattern:

```
User intent
  → OSA decomposes into agent tasks
  → Web Agent gathers external intelligence
  → Code Agent validates, calculates, and generates artifacts
  → Memory Agent stores and retrieves context
  → P&L personalizes the experience
  → SysStat ensures resources are available
  → HIL keeps the human in control of critical decisions
  → Output is actionable, cited, and personalized
  → System learns from every interaction
```

The power isn't in any single agent — it's in their composition. The same primitives that monitor your stock portfolio also research your newsletter topics, prep your interviews, and design your workout program. You just describe what you want, and Octane figures out which agents to chain together.

That's what makes it an operating system, not an app.
