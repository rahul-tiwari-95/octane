# Use Case: Deep Research

Octane as a private research terminal — from a question to a cited, cross-referenced report stored permanently in your local knowledge base.

---

## The Core Idea

Octane doesn't just answer questions — it researches them. The difference:

- A chatbot gives you whatever the model was trained on
- Octane goes and gets current information, extracts it, trust-scores it, synthesizes it, and stores it

Every research session builds your local knowledge base. Three months from now, you can ask `octane recall search "AI export restrictions"` and get everything you've ever researched on that topic, instantly.

---

## Basic Research

```bash
# Deep research, 4 dimensions, with citations
octane ask "what are the latest developments in speculative decoding?" --deep 4 --cite
```

What happens:
1. **DimensionPlanner** breaks this into 4 angles (e.g., *academic papers*, *production implementations*, *benchmark comparisons*, *industry adoption*)
2. Each angle is researched in parallel by the Web Agent
3. Results are scored and deduplicated
4. BodegaOS synthesizes a cited report
5. Everything is stored in your Postgres knowledge base

---

## Graduated Depth

```bash
# Quick — single-dimension, fast answer
octane ask "what is CUDA?" --deep 1

# Standard — 4 angles, good coverage
octane ask "how does Flash Attention work?" --deep 4 --cite

# Thorough — 8 angles, broad cross-referencing
octane ask "state of the art in multimodal models 2025" --deep 8 --cite

# Maximum — 12+ angles, full domain sweep
octane investigate "complete landscape of AI inference optimization" --deep 12 --cite --verify
```

The `--verify` flag adds a second pass that cross-checks key claims across sources.

---

## Investigating a Topic Over Time

```bash
# First session: establish baseline
octane investigate "NVDA competitive position in AI chips" --deep 6 --cite

# One week later: research has evolved, new findings will be added
octane investigate "NVDA competitive position in AI chips" --deep 6 --cite

# See everything you've ever learned about NVDA
octane recall search "NVDA"

# Full stats on your knowledge base
octane stats
```

Octane uses content-hash deduplication — the same article won't be stored twice, but new articles will be added and the synthesis will reflect the latest understanding.

---

## Comparing and Contrasting

```bash
# Compare two approaches across multiple dimensions
octane compare "PyTorch vs JAX for research" --deep 4

# Compare companies
octane compare "NVDA vs AMD in AI inference market" --deep 6 --cite

# Compare technologies
octane compare "RAG vs fine-tuning for domain adaptation" --deep 4 --cite
```

---

## Source-Specific Research

```bash
# Research only from arXiv papers
octane search arxiv "mixture of experts scaling" --limit 8 --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "key findings on MoE scaling"

# Research from YouTube (transcripts extracted)
octane search youtube "Andrej Karpathy neural networks" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "key teaching points"

# Research with web + synthesize
octane search web "AI agent frameworks 2025" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "which frameworks are production-ready?"
```

---

## Chaining Research

```bash
# Chain: first question feeds context into second
octane chain \
  "explain the attention mechanism in transformers" \
  "how does this apply to vision transformers specifically?"
```

---

## Saving Findings to Projects

```bash
# Create a research project
octane project create "Q2 2026 AI Infrastructure Research"

# Switch to it
octane project switch "Q2 2026 AI Infrastructure Research"

# All subsequent research is tagged to this project
octane investigate "inference hardware landscape" --deep 6 --cite

# Review project status
octane project status
```

---

## Recall — Mining Your Knowledge Base

After researching for a while, your knowledge base becomes valuable:

```bash
# Search everything you've learned
octane recall search "transformer"
octane recall search "NVDA earnings"
octane recall search "speculative decoding" --type arxiv

# Full stats
octane stats
```

Output example:
```
Knowledge Base Stats
──────────────────────────────────────────
  Research tasks:     234
  Findings stored:  4,821
  Extractions:      1,203
  Memory nodes:       891
  Unique sources:     567
  Total stored:     ~2.4 GB
──────────────────────────────────────────
```

---

## OSINT Workflows

See [USE_CASE_OSINT.md](USE_CASE_OSINT.md) for using Octane's research pipeline for open source intelligence gathering.

---

## Tips

- **Start with `--deep 4`** for most research. Increase if coverage feels shallow.
- **Always use `--cite`** if you'll share or act on the output — trust scores matter.
- **`octane recall search`** before re-researching — you may already have the answer.
- **Research while you sleep** — run a deep investigation before bed, read the report in the morning.
