# Octane Quickstart

Get from zero to your first AI research query in 5 minutes.

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS Tahoe 16.x
- Homebrew (`brew.sh`)

---

## Step 1 — Install BodegaOS Sensors

BodegaOS Sensors is the inference engine that powers Octane. It is a native macOS app, not a Python package.

```bash
bash scripts/install_sensors.sh
```

The script will:
1. Check your RAM and download the correct edition (Standard ≤32 GB, Pro >32 GB)
2. Save the `.dmg` to `~/Downloads`
3. Prompt you to drag it into Applications and launch it

After launching:
- Find the **Bodega Inference Engine** toggle
- Turn it **ON**
- Wait until it turns **green** — that means inference is live at `localhost:44468`

---

## Step 2 — Install Octane

```bash
git clone https://github.com/srswti/octane
cd octane
bash setup.sh
```

`setup.sh` will install PostgreSQL, Redis, Python dependencies, run database migrations, and verify everything is healthy. It also checks whether BodegaOS Sensors is running and lets you know if it isn't.

---

## Step 3 — Activate and Verify

```bash
source .venv/bin/activate
octane health
```

Expected output:
```
  ✓  Redis         localhost:6379
  ✓  PostgreSQL    localhost:5432
  ✓  BodegaOS      localhost:44468  (2 model(s) loaded)
  ✓  Octane daemon ready
```

If BodegaOS shows as unreachable, ensure the Inference Engine toggle is green in the BodegaOS Sensors app.

---

## Step 4 — First Commands

```bash
# Quick test — instant response, no deep research
octane ask "hello"

# Deep research — decomposes into 4 dimensions, cites sources
octane ask "explain transformers" --deep 4 --cite

# Open Mission Control dashboard
octane ui start
# → http://localhost:44480

# Composable pipe: search → extract → synthesize
octane search arxiv "speculative decoding" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "key insights"
```

---

## What Just Happened?

When you ran `octane ask "explain transformers" --deep 4`:

1. **IntentGate** classified your query as a deep research request
2. **DimensionPlanner** broke it into 4 independent research angles
3. **Web Agent** searched web, arXiv, and YouTube in parallel
4. **Evaluator** scored and deduped findings with trust weights
5. **Synthesizer** asked BodegaOS to produce a cited report
6. Everything was stored in your local Postgres knowledge base

Next time you ask about transformers, Octane already knows what you've researched.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `BodegaOS: unreachable` | Open BodegaOS Sensors app, enable Inference Engine toggle |
| `Redis: not responding` | `brew services start redis` |
| `PostgreSQL: not ready` | `brew services start postgresql@16` |
| `command not found: octane` | `source .venv/bin/activate` |
| setup.sh fails on brew | Install Homebrew first: `brew.sh` |

---

## Next Steps

- [CLI_CHEATSHEET.md](CLI_CHEATSHEET.md) — all ~100 commands with examples
- [ARCHITECTURE.md](ARCHITECTURE.md) — how Octane works internally
- [USE_CASE_RESEARCH.md](USE_CASE_RESEARCH.md) — deep research workflows
- [BODEGA.md](BODEGA.md) — BodegaOS Sensors details
