# BodegaOS Sensors

BodegaOS Sensors is the inference engine that powers Octane. It is a native macOS app developed by SRSWTI Research Labs, distributed separately from Octane.

---

## What It Is

BodegaOS Sensors is a high-performance MLX inference server for Apple Silicon. It:

- Runs multiple language models simultaneously using Apple's Metal GPU stack
- Serves a local HTTP API at `localhost:44468`
- Uses speculative decoding and continuous batching for low latency
- Supports structured JSON outputs with guaranteed schema conformance
- Includes a built-in RAG pipeline and prompt caching
- Auto-selects the optimal model tier for each request

Octane sends all AI inference requests to BodegaOS Sensors. Without it running, `octane ask`, `octane investigate`, and `octane chat` will not work. Data commands (`octane recall`, `octane extract`, `octane portfolio show`) work offline.

---

## Install

BodegaOS Sensors is a native macOS app distributed as a `.dmg`. It is **not** a Python package.

```bash
bash scripts/install_sensors.sh
```

The installer script will:
1. Detect your Mac's RAM
2. Download the correct edition:
   - **Standard** — for Macs with ≤32 GB unified memory
   - **Pro** — for Macs with >32 GB unified memory (M2 Ultra, M3 Max/Ultra, M4 Max/Ultra, etc.)
3. Save the `.dmg` to `~/Downloads`
4. Walk you through installation

### After the DMG installs

1. Open **BodegaOS Sensors** from Applications
2. Find the **Bodega Inference Engine** toggle
3. Turn it **ON**
4. Wait until the toggle turns **green** — inference is live

You should see a green status indicator in your macOS menu bar when the engine is running.

---

## Verify

```bash
curl http://localhost:44468/health
```

Expected response:
```json
{
  "status": "ok",
  "models_detail": [...]
}
```

Or from Octane:
```bash
octane health
```

---

## System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Chip | Apple Silicon M1 | M2/M3/M4 recommended for larger models |
| macOS | Tahoe 16.x | Required by BodegaOS Sensors |
| RAM | 8 GB | 16 GB+ recommended; >32 GB unlocks Pro edition |

---

## Model Tiers

Octane's **BodegaRouter** automatically routes requests to the right tier:

| Tier | Purpose | Example Tasks |
|------|---------|---------------|
| `FAST` | Sub-second responses | Intent classification, greetings, quick lookups |
| `MID` | Balanced speed/quality | Extraction, summarization, structured outputs |
| `REASON` | Maximum quality | Deep research synthesis, multi-step reasoning, cited reports |

You don't configure this manually — Octane decides based on task complexity.

---

## BodegaOS Sensors vs. Octane

| | BodegaOS Sensors | Octane |
|-|-----------------|--------|
| Role | Inference engine | Intelligence OS |
| What it does | Runs models on your GPU | Research, analysis, knowledge accumulation |
| Port | 44468 | 44480 (Mission Control UI) |
| Install | DMG app | `bash setup.sh` |
| Without it | Octane can't do AI inference | BodegaOS has no agent layer |

They are separate products designed to work together.

---

## Troubleshooting

**Green toggle but octane health shows unreachable:**
- Wait 10-15 seconds after enabling the toggle — model loading takes time
- Try `curl http://localhost:44468/health` directly

**Download failed in install_sensors.sh:**
- Check your internet connection
- Ensure `curl` is available: `which curl`

**"This Mac doesn't meet requirements":**
- Requires Apple Silicon and macOS Tahoe 16.x
- Intel Macs are not supported

**Models are slow:**
- More RAM = larger, faster models
- Close memory-intensive apps to free unified memory
- Pro edition (>32 GB) enables larger model weights
