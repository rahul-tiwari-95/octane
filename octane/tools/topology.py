"""Model topology definitions for Octane.

A *topology* maps ModelTier → :class:`ModelConfig` — a full set of Bodega
``load-model`` parameters (model_path, context_length, max_concurrency,
prompt_cache_size, speculative decoding, parsers, etc.).

Three named topologies cover the range of Apple Silicon configs:

    compact   — 8–12 GB RAM  (lean: max_concurrency=1, 8 K ctx on REASON,
                               no speculative decoding)
    balanced  — 16–24 GB RAM (default: max_concurrency=2 for FAST/MID,
                               speculative decoding on REASON via qwen3-0.6B)
    power     — 32 GB+ RAM   (max_concurrency=4/2, three distinct models,
                               num_draft_tokens=5 for REASON)

Usage::

    from octane.tools.topology import get_topology, ModelTier

    topo   = get_topology("auto")                  # auto-detects from system RAM
    cfg    = topo.resolve_config(ModelTier.REASON) # → ModelConfig
    params = cfg.to_load_params()                  # → dict for load-model API
    model  = topo.resolve(ModelTier.FAST)          # → "bodega-raptor-90M" (str)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# ── Tier enum ─────────────────────────────────────────────────────────────────


class ModelTier(str, enum.Enum):
    """Inference cost/capability tiers.

    FAST   — Tiny model (~90M params).  Sub-second latency.  Good for:
             keyword extraction, query routing, short classifications.
    MID    — Medium model (~1B params).  Fast.  Good for: chunk
             summarization, structured data extraction.
    REASON — Large model (~8B params).  Deep reasoning.  Good for: full
             synthesis, evaluation, complex analysis.
    EMBED  — Local embedding model (sentence-transformers).  Not routed
             through Bodega — handled in-process.
    """

    FAST = "fast"
    MID = "mid"
    REASON = "reason"
    EMBED = "embed"


# ── ModelConfig ───────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Full Bodega ``load-model`` parameters for one model tier.

    Every field maps 1-to-1 to a ``POST /v1/admin/load-model`` request body
    parameter.  ``to_load_params()`` serialises only the non-None / non-zero
    optional fields so the payload stays minimal.

    Args:
        model_id:          Alias used in inference API calls (e.g. ``"bodega-raptor-8b"``).
        model_path:        HuggingFace repo ID or absolute local path.
        model_type:        Bodega model type — ``"lm"``, ``"embeddings"``, etc.
        context_length:    KV-cache context window in tokens.
        max_concurrency:   Parallel requests the model handler accepts.
        prompt_cache_size: Number of prompt-cache slots (0 = disabled).
                           Each slot eliminates prefill cost for a recurring
                           system-prompt prefix.
        reasoning_parser:  Parser for ``<think>`` extraction (e.g. ``"qwen3"``).
        tool_call_parser:  Parser for tool-call extraction.
        draft_model_path:  HuggingFace path of the speculative-decoding draft
                           model.  Must share a tokenizer with *model_path*.
        num_draft_tokens:  Number of draft tokens per speculation step.
                           0 = speculative decoding disabled.
    """

    model_id: str
    model_path: str
    model_type: str = "lm"
    context_length: int = 32768
    max_concurrency: int = 1
    prompt_cache_size: int = 10
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    draft_model_path: str | None = None
    num_draft_tokens: int = 0

    def to_load_params(self) -> dict[str, Any]:
        """Return a dict suitable for ``POST /v1/admin/load-model``.

        Only includes optional fields that carry a non-None / non-zero value
        so the Bodega payload remains clean and minimal.
        """
        params: dict[str, Any] = {
            "model_path": self.model_path,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "context_length": self.context_length,
            "max_concurrency": self.max_concurrency,
            "prompt_cache_size": self.prompt_cache_size,
        }
        if self.reasoning_parser is not None:
            params["reasoning_parser"] = self.reasoning_parser
        if self.tool_call_parser is not None:
            params["tool_call_parser"] = self.tool_call_parser
        if self.draft_model_path is not None:
            params["draft_model_path"] = self.draft_model_path
        if self.num_draft_tokens > 0:
            params["num_draft_tokens"] = self.num_draft_tokens
        return params


# ── Topology dataclass ────────────────────────────────────────────────────────


@dataclass
class Topology:
    """Maps each ModelTier to a full :class:`ModelConfig`.

    Args:
        name:   Human-readable topology name (e.g. "balanced").
        models: Mapping of ModelTier → ModelConfig with full Bodega load params.
    """

    name: str
    models: dict[ModelTier, ModelConfig] = field(default_factory=dict)

    def resolve_config(self, tier: ModelTier) -> ModelConfig:
        """Return the full :class:`ModelConfig` for *tier*.

        Falls back in order: FAST → REASON when the exact tier is not present.
        Raises ``ValueError`` only if no fallback exists.
        """
        if tier in self.models:
            return self.models[tier]
        # Fallback chain — prefer cheap before expensive
        for fallback in (ModelTier.FAST, ModelTier.REASON):
            if fallback in self.models:
                return self.models[fallback]
        raise ValueError(
            f"No model configured for tier {tier!r} in topology '{self.name}' "
            f"and no fallback available."
        )

    def resolve(self, tier: ModelTier) -> str:
        """Return the Bodega model_id string for *tier*.

        Convenience wrapper around :meth:`resolve_config`.  Equivalent to
        ``topology.resolve_config(tier).model_id``.
        """
        return self.resolve_config(tier).model_id


# ── Named topologies ──────────────────────────────────────────────────────────
#
# Speculative decoding draft model — shared tokenizer with bodega-raptor-8b:
_DRAFT_MODEL = "Qwen/Qwen3-0.6B-MLX-8bit"

TOPOLOGIES: dict[str, Topology] = {
    # ── compact ───────────────────────────────────────────────────────────────
    # M1/M2 8–12 GB.
    # Strategy: survive on minimal VRAM.
    #   • max_concurrency=1 everywhere — no RAM for parallel KV-caches
    #   • REASON context_length=8192 — halves KV-cache footprint (~600 MB saved)
    #   • prompt_cache_size=5 — small cache; saves prefill on short system prompts
    #   • NO speculative decoding — draft model (~624 MB) would bust the budget
    "compact": Topology(
        name="compact",
        models={
            ModelTier.FAST: ModelConfig(
                model_id="bodega-raptor-90M",
                model_path="SRSWTI/bodega-raptor-90m",
                context_length=32768,
                max_concurrency=1,
                prompt_cache_size=5,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
            ),
            ModelTier.MID: ModelConfig(
                model_id="bodega-raptor-90M",
                model_path="SRSWTI/bodega-raptor-90m",
                context_length=32768,
                max_concurrency=1,
                prompt_cache_size=5,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
            ),
            ModelTier.REASON: ModelConfig(
                model_id="bodega-raptor-8b",
                model_path="SRSWTI/bodega-raptor-8b-mxfp4",
                context_length=8192,        # ← compact: halved to save ~600 MB
                max_concurrency=1,
                prompt_cache_size=5,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
                # draft_model_path=None     ← speculative decoding OFF (no RAM budget)
            ),
        },
    ),
    # ── balanced (default) ────────────────────────────────────────────────────
    # M1/M2 Pro 16–24 GB.
    # Strategy: full quality with meaningful parallelism.
    #   • max_concurrency=2 for FAST/MID — handles concurrent web-search chunks
    #   • REASON context_length=32768 — full window
    #   • prompt_cache_size=15 — covers evaluator + decomposer static prefixes
    #   • Speculative decoding on REASON: qwen3-0.6B drafts 3 tokens per step
    #     → expect 1.5–2× tokens/s uplift on the 8B model
    "balanced": Topology(
        name="balanced",
        models={
            ModelTier.FAST: ModelConfig(
                model_id="bodega-raptor-90M",
                model_path="SRSWTI/bodega-raptor-90m",
                context_length=32768,
                max_concurrency=2,          # ← balanced: parallel chunk processing
                prompt_cache_size=15,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
            ),
            ModelTier.MID: ModelConfig(
                model_id="bodega-raptor-90M",
                model_path="SRSWTI/bodega-raptor-90m",
                context_length=32768,
                max_concurrency=2,
                prompt_cache_size=15,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
            ),
            ModelTier.REASON: ModelConfig(
                model_id="bodega-raptor-8b",
                model_path="SRSWTI/bodega-raptor-8b-mxfp4",
                context_length=32768,
                max_concurrency=1,
                prompt_cache_size=15,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
                draft_model_path=_DRAFT_MODEL,  # ← speculative decoding ON
                num_draft_tokens=3,
            ),
        },
    ),
    # ── power ─────────────────────────────────────────────────────────────────
    # M2/M3 Max/Ultra 32 GB+.
    # Strategy: maximum throughput, three distinct models per tier.
    #   • max_concurrency=4 for FAST, 2 for MID/REASON
    #   • MID upgrades to bodega-raptor-1b for richer chunk summaries
    #   • prompt_cache_size=25 — large cache covers all static prefixes
    #   • Speculative decoding with 5 draft tokens (vs 3 in balanced)
    "power": Topology(
        name="power",
        models={
            ModelTier.FAST: ModelConfig(
                model_id="bodega-raptor-90M",
                model_path="SRSWTI/bodega-raptor-90m",
                context_length=32768,
                max_concurrency=4,          # ← power: high-throughput routing
                prompt_cache_size=25,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
            ),
            ModelTier.MID: ModelConfig(
                model_id="bodega-raptor-1b",
                model_path="SRSWTI/bodega-raptor-0.9b",
                context_length=32768,
                max_concurrency=2,
                prompt_cache_size=25,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
            ),
            ModelTier.REASON: ModelConfig(
                model_id="bodega-raptor-8b",
                model_path="SRSWTI/bodega-raptor-8b-mxfp4",
                context_length=32768,
                max_concurrency=2,          # ← power: parallel deep reasoning
                prompt_cache_size=25,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3",
                draft_model_path=_DRAFT_MODEL,  # ← speculative decoding ON
                num_draft_tokens=5,             # ← power: 5 draft tokens (vs 3)
            ),
        },
    ),
}

DEFAULT_TOPOLOGY = "balanced"

# ── Auto-detection ────────────────────────────────────────────────────────────


def detect_topology() -> str:
    """Detect the appropriate topology name from available system RAM.

    Returns one of ``'compact'``, ``'balanced'``, ``'power'``.
    Falls back to ``DEFAULT_TOPOLOGY`` (``"balanced"``) if *psutil* is not
    installed.
    """
    try:
        import psutil  # type: ignore

        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 32:
            return "power"
        elif ram_gb >= 16:
            return "balanced"
        else:
            return "compact"
    except ImportError:
        return DEFAULT_TOPOLOGY


def get_topology(name: str = "auto") -> Topology:
    """Return a :class:`Topology` by name.

    ``"auto"`` triggers :func:`detect_topology` to pick from system RAM.

    Raises:
        ValueError: If *name* is not ``"auto"`` and not in :data:`TOPOLOGIES`.
    """
    if name == "auto":
        name = detect_topology()
    if name not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{name}'. "
            f"Valid values: {sorted(TOPOLOGIES)} + 'auto'."
        )
    return TOPOLOGIES[name]
