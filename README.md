# 🔥 Octane

**Local-first agentic operating system for Apple Silicon.**

Octane is a hierarchical agent system modeled after biological nervous systems. It runs entirely on your machine using MLX-based inference via the Bodega Inference Engine.

## Architecture

```
User → CLI → OSA (Brain) → Agents (Organs) → Tools → External Services
                ↕
           Synapse (Nervous System) + Shadows (Neural Bus)
```

### Agents
- **Web Agent** — Internet data retrieval (search, finance, news)
- **Code Agent** — Code generation, execution, self-healing
- **Memory Agent** — Three-tier memory (Redis → Postgres → pgVector)
- **SysStat Agent** — System monitoring, model management
- **P&L Agent** — User personalization and learning

## Quick Start

```bash
# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Check system health
octane health

# Ask a question
octane ask "What is happening with NVIDIA?"

# List agents
octane agents

# Show version
octane version
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check octane/
```

## Project Status

**Session 1 Complete:** Project scaffold, CLI, SysStat Agent with live Bodega health check.

See `context/OCTANE_4_PHASE_PLAN_v2.md` for the full build plan.
See `context/OCTANE_AI_IDE_v2.md` for AI IDE agent guidelines.
See `SessionHistory/` for session-by-session build log.
