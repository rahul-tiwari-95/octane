# Use Case: Local AI Power User

For users who want maximum control over local inference on Apple Silicon — model selection, performance tuning, composable pipelines, and air-gapped operation.

---

## Why Octane Over Raw Ollama/LMStudio?

| | Ollama / LMStudio | Octane + BodegaOS |
|-|-------------------|-------------------|
| Inference | Single model at a time | Multi-model registry, simultaneous |
| Routing | Manual model selection | Auto-routing by task complexity (FAST/MID/REASON) |
| Memory | Stateless | Persistent knowledge base (Postgres + pgVector) |
| Tools | None | Web search, extraction, arXiv, YouTube, finance |
| Pipelines | Manual | Composable CLI pipes (search → extract → synthesize) |
| Background | None | Shadows (perpetual research cycles, file watchers) |
| UI | Basic | Mission Control with globe, vitals, event stream |
| Speculative decoding | No | Yes (BodegaOS) |
| Continuous batching | No | Yes (BodegaOS) |

---

## BodegaOS Sensors — The Engine

BodegaOS Sensors is not Ollama. It is SRSWTI's purpose-built MLX inference server for Apple Silicon:

- **Speculative decoding**: Draft model pre-populates tokens for the main model, then verifies. Dramatically faster for long outputs.
- **Continuous batching**: Multiple concurrent inference requests share GPU time efficiently.
- **Structured outputs**: Guaranteed JSON schema conformance — the model cannot produce malformed JSON.
- **Prompt caching**: Shared prefix caching across requests.
- **Multi-model registry**: Hot-swap between models without reloading.
- **Metal**: Direct Apple GPU access, not CUDA compatibility shim.

```bash
# Check what's loaded
octane model list

# See active inference stats
octane health
```

---

## Model Tiers — Auto-Routing

Octane's **BodegaRouter** selects the right model automatically:

```bash
# FAST tier: instant classification, greetings, lookups (<1s)
octane ask "hello"
octane ask "what time is it in Tokyo?"

# MID tier: extraction, summarization, structured outputs (2-5s)
octane extract url "https://arxiv.org/abs/1706.03762"

# REASON tier: deep synthesis, multi-step, citations (10-30s)
octane investigate "scaling laws for language models" --deep 6 --cite
```

You don't configure this — the router decides based on the task graph.

---

## Composable Pipes — Unix Philosophy for AI

Every command reads stdin, writes stdout in JSON:

```bash
# Search → Extract → Synthesize
octane search arxiv "mixture of experts" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "key findings"

# Search multiple sources, combine
{ octane search web "NVDA" --json; octane search news "NVDA" --json; } \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "latest developments"

# Extract into recall
octane extract url "https://arxiv.org/abs/2307.09288" --json | jq .

# Portfolio data to JSON for external tools
octane portfolio show --json | jq '.holdings[] | select(.value > 10000)'
```

---

## Air-Gap Mode — Full Offline Operation

```bash
# Block all external network traffic
octane airgap on
octane airgap status    # confirm active

# Work entirely with local knowledge and local inference
octane recall search "anything you've previously researched"
octane ask "question based on local knowledge"

# Re-enable
octane airgap off
```

Air-gap is implemented at the OS network layer — it's not just a flag, it actually kills external connections.

---

## Security Vault — Touch ID Protected Secrets

```bash
# Store secrets with Touch ID authentication
octane vault set finance schwab_api_key
octane vault set llm openai_key    # if you want optional cloud fallback

# Retrieve (requires Touch ID)
octane vault get finance schwab_api_key

# List namespaces
octane vault list
```

Secrets are encrypted at rest using AES-256 and only decryptable with your biometric.

---

## Persistent Knowledge — Gets Smarter Over Time

```bash
# After weeks of research, your knowledge base is valuable
octane stats

# Mine it
octane recall search "any topic you've researched"
octane recall search "transformer" --type arxiv --limit 50
```

Unlike a chat session, this knowledge persists. The more you use Octane, the better it answers questions from memory.

---

## Daemon + Shadows — Background Intelligence

```bash
# Start daemon (keeps agents warm, runs Shadows)
octane daemon start

# Drop a PDF in the inbox — auto-extracted while you work on something else
cp ~/Downloads/paper.pdf ~/Octane/inbox/

# The FileWatcher Shadow extracts it in the background
# Check it was processed:
octane files list
```

---

## Mission Control — Live System View

```bash
octane ui start
```

Open `http://localhost:44480`:
- Watch GPU utilization as models process requests
- See live inference events ("Synthesizing finding 3/8...")
- Monitor memory pressure as model weights compete for unified memory
- Run terminal commands from the browser

---

## Performance Tips

- **More RAM = better models**: 16 GB is the minimum, 32 GB unlocks larger contexts, 64 GB+ allows simultaneous full-quality models
- **Close memory-intensive apps**: Chrome + Electron apps eat unified memory that models need
- **Pro edition**: If you have >32 GB RAM, `install_sensors.sh` automatically downloads BodegaOS Sensors Pro with larger model support
- **Model warm-up**: First request after boot is slower as models load into memory; subsequent requests are fast

---

## Trace and Debug

```bash
# See what happened on the last request
octane trace last

# Full trace with timing
octane trace last --verbose

# View engine-level detail
octane trace show <trace-id>
```

---

## For r/LocalLLaMA Users

If you're used to running models manually with Ollama or llama.cpp:

1. **You don't configure models** — BodegaOS Sensors manages them
2. **You don't write prompts** — Octane's agent pipeline handles prompt construction
3. **You don't manage context** — Memory Agent handles what to include
4. **The output is structured** — findings are stored, recalled, and built upon

The inference layer is opinionated so the application layer can be powerful.
