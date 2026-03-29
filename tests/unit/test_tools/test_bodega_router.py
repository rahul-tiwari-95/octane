"""Tests for BodegaRouter, ModelConfig, and topology definitions (Session 20C).

Covers:
    - ModelTier enum values
    - ModelConfig.to_load_params() — baseline, optional fields, speculative
    - Topology.resolve_config() — exact match and fallback chain
    - Topology.resolve() — still returns str (backward-compat)
    - Named topology configurations (compact / balanced / power)
      · compact  : max_concurrency=1, context_length=8192 on REASON, no speculative
      · balanced : max_concurrency=2 on FAST, speculative decoding on REASON
      · power    : max_concurrency=4 on FAST, 3 distinct models, num_draft_tokens=5
    - detect_topology() — returns a valid name
    - get_topology() — "auto", named, invalid
    - BodegaRouter.resolve_model_id()
    - BodegaRouter.resolve_config() — returns ModelConfig
    - BodegaRouter.chat() — delegates with correct model_id
    - BodegaRouter.chat_simple() — returns text, uses tier
    - BodegaRouter.health() / close() — delegation
    - ModelManager.ensure_topology_loaded() — deduplication, load params, error handling
"""

from __future__ import annotations

import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from octane.tools.topology import (
    DEFAULT_TOPOLOGY,
    TOPOLOGIES,
    ModelConfig,
    ModelTier,
    Topology,
    detect_topology,
    get_topology,
)
from octane.tools.bodega_router import BodegaRouter, LoadedModel
from octane.agents.sysstat.model_manager import ModelManager


# ── ModelTier ─────────────────────────────────────────────────────────────────


def test_model_tier_fast_value():
    assert ModelTier.FAST == "fast"


def test_model_tier_mid_value():
    assert ModelTier.MID == "mid"


def test_model_tier_reason_value():
    assert ModelTier.REASON == "reason"


def test_model_tier_embed_value():
    assert ModelTier.EMBED == "embed"


# ── ModelConfig.to_load_params() ──────────────────────────────────────────────


def test_model_config_to_load_params_baseline():
    """to_load_params() includes all mandatory fields."""
    cfg = ModelConfig(
        model_id="my-model",
        model_path="SRSWTI/my-model",
        context_length=16384,
        max_concurrency=2,
        prompt_cache_size=10,
    )
    params = cfg.to_load_params()
    assert params["model_path"] == "SRSWTI/my-model"
    assert params["model_id"] == "my-model"
    assert params["model_type"] == "lm"
    assert params["context_length"] == 16384
    assert params["max_concurrency"] == 2
    assert params["prompt_cache_size"] == 10


def test_model_config_to_load_params_omits_none_optional():
    """Optional None fields (reasoning_parser, draft_model_path) are excluded."""
    cfg = ModelConfig(model_id="x", model_path="repo/x")
    params = cfg.to_load_params()
    assert "reasoning_parser" not in params
    assert "tool_call_parser" not in params
    assert "draft_model_path" not in params
    assert "num_draft_tokens" not in params


def test_model_config_to_load_params_includes_parsers_when_set():
    cfg = ModelConfig(
        model_id="x",
        model_path="repo/x",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3",
    )
    params = cfg.to_load_params()
    assert params["reasoning_parser"] == "qwen3"
    assert params["tool_call_parser"] == "qwen3"


def test_model_config_to_load_params_speculative_decoding():
    """draft_model_path and num_draft_tokens appear when speculative decoding is on."""
    cfg = ModelConfig(
        model_id="raptor-8b",
        model_path="SRSWTI/bodega-raptor-8b-mxfp4",
        draft_model_path="Qwen/Qwen3-0.6B-MLX-8bit",
        num_draft_tokens=3,
    )
    params = cfg.to_load_params()
    assert params["draft_model_path"] == "Qwen/Qwen3-0.6B-MLX-8bit"
    assert params["num_draft_tokens"] == 3


def test_model_config_to_load_params_zero_draft_tokens_omitted():
    """num_draft_tokens=0 (default) must NOT appear in the payload."""
    cfg = ModelConfig(model_id="x", model_path="repo/x", num_draft_tokens=0)
    params = cfg.to_load_params()
    assert "num_draft_tokens" not in params


# ── Topology.resolve_config() and resolve() ───────────────────────────────────


def test_topology_resolve_config_exact_match():
    fast_cfg = ModelConfig(model_id="fast-model", model_path="repo/fast")
    reason_cfg = ModelConfig(model_id="reason-model", model_path="repo/reason")
    topo = Topology(
        name="test",
        models={ModelTier.FAST: fast_cfg, ModelTier.REASON: reason_cfg},
    )
    assert topo.resolve_config(ModelTier.FAST) is fast_cfg
    assert topo.resolve_config(ModelTier.REASON) is reason_cfg


def test_topology_resolve_config_mid_fallback_to_fast():
    """MID not configured → falls back to FAST config."""
    fast_cfg = ModelConfig(model_id="fast-model", model_path="repo/fast")
    topo = Topology(
        name="test",
        models={ModelTier.FAST: fast_cfg, ModelTier.REASON: ModelConfig(model_id="r", model_path="r")},
    )
    assert topo.resolve_config(ModelTier.MID) is fast_cfg


def test_topology_resolve_returns_model_id_string():
    """resolve() is a backward-compat str shortcut over resolve_config()."""
    cfg = ModelConfig(model_id="bodega-raptor-8b", model_path="SRSWTI/bodega-raptor-8b-mxfp4")
    topo = Topology(name="test", models={ModelTier.REASON: cfg})
    assert topo.resolve(ModelTier.REASON) == "bodega-raptor-8b"


def test_topology_resolve_config_raises_when_no_fallback():
    """EMBED not configured and no FAST/REASON → raises."""
    topo = Topology(name="empty", models={})
    with pytest.raises(ValueError, match="No model configured"):
        topo.resolve_config(ModelTier.FAST)


# ── Named topologies — model IDs ──────────────────────────────────────────────


def test_topology_compact_has_all_inference_tiers():
    topo = TOPOLOGIES["compact"]
    assert ModelTier.FAST in topo.models
    assert ModelTier.MID in topo.models
    assert ModelTier.REASON in topo.models


def test_topology_compact_fast_model_id():
    assert TOPOLOGIES["compact"].models[ModelTier.FAST].model_id == "bodega-raptor-90M"


def test_topology_compact_reason_model_id():
    assert TOPOLOGIES["compact"].models[ModelTier.REASON].model_id == "bodega-raptor-8b"


def test_topology_balanced_has_all_inference_tiers():
    topo = TOPOLOGIES["balanced"]
    assert ModelTier.FAST in topo.models
    assert ModelTier.MID in topo.models
    assert ModelTier.REASON in topo.models


def test_topology_power_mid_is_raptor_8b():
    """Power topology uses bodega-raptor-8b-mxfp4 for MID — 3-tier structure on 64 GB+."""
    cfg = TOPOLOGIES["power"].models[ModelTier.MID]
    assert cfg.model_id == "bodega-raptor-8b"
    assert "bodega-raptor-8b-mxfp4" in cfg.model_path


def test_all_topologies_cover_fast_mid_reason():
    for name, topo in TOPOLOGIES.items():
        for tier in (ModelTier.FAST, ModelTier.MID, ModelTier.REASON):
            resolved = topo.resolve(tier)
            assert isinstance(resolved, str) and resolved, (
                f"Topology '{name}' returned empty model_id for tier {tier}"
            )


# ── Named topologies — compact is genuinely lean ──────────────────────────────


def test_compact_reason_uses_8k_context():
    """compact saves ~600 MB by halving the REASON context window."""
    cfg = TOPOLOGIES["compact"].models[ModelTier.REASON]
    assert cfg.context_length == 8192


def test_compact_max_concurrency_is_1():
    """compact uses max_concurrency=1 on all tiers — no spare RAM for parallel KV."""
    for tier in (ModelTier.FAST, ModelTier.MID, ModelTier.REASON):
        cfg = TOPOLOGIES["compact"].models[tier]
        assert cfg.max_concurrency == 1, f"compact.{tier.value}.max_concurrency should be 1"


def test_compact_has_no_speculative_decoding():
    """compact omits draft model — qwen3-0.6B (~624 MB) would bust the budget."""
    cfg = TOPOLOGIES["compact"].models[ModelTier.REASON]
    assert cfg.draft_model_path is None
    assert cfg.num_draft_tokens == 0


def test_compact_prompt_cache_size_is_small():
    """compact keeps prompt_cache_size=5 (tiny footprint)."""
    for tier in (ModelTier.FAST, ModelTier.MID, ModelTier.REASON):
        cfg = TOPOLOGIES["compact"].models[tier]
        assert cfg.prompt_cache_size == 5


# ── Named topologies — balanced has real concurrency + speculative ─────────────


def test_balanced_fast_max_concurrency_is_2():
    """balanced allows 2 parallel requests on the FAST tier."""
    assert TOPOLOGIES["balanced"].models[ModelTier.FAST].max_concurrency == 2


def test_balanced_mid_max_concurrency_is_2():
    assert TOPOLOGIES["balanced"].models[ModelTier.MID].max_concurrency == 2


def test_balanced_reason_max_concurrency_is_1():
    """balanced keeps REASON at 1 — only one large model KV-cache fits."""
    assert TOPOLOGIES["balanced"].models[ModelTier.REASON].max_concurrency == 1


def test_balanced_reason_has_speculative_decoding():
    """balanced enables speculative decoding on REASON via qwen3-0.6B draft."""
    cfg = TOPOLOGIES["balanced"].models[ModelTier.REASON]
    assert cfg.draft_model_path == "Qwen/Qwen3-0.6B-MLX-8bit"
    assert cfg.num_draft_tokens == 3


def test_balanced_reason_full_context():
    """balanced uses full 32K context on REASON (unlike compact's 8K)."""
    assert TOPOLOGIES["balanced"].models[ModelTier.REASON].context_length == 32768


def test_balanced_prompt_cache_size():
    """balanced uses prompt_cache_size=15 — covers static evaluator/decomposer prefixes."""
    for tier in (ModelTier.FAST, ModelTier.MID, ModelTier.REASON):
        cfg = TOPOLOGIES["balanced"].models[tier]
        assert cfg.prompt_cache_size == 15


# ── Named topologies — power has highest throughput ───────────────────────────


def test_power_fast_max_concurrency_is_4():
    """power routes up to 8 parallel classification/extraction requests (extreme for 64GB M1 Max)."""
    assert TOPOLOGIES["power"].models[ModelTier.FAST].max_concurrency == 8


def test_power_reason_max_concurrency_is_2():
    """power REASON uses axe-stealth-37b with max_concurrency=2 (heavy model)."""
    assert TOPOLOGIES["power"].models[ModelTier.REASON].max_concurrency == 2


def test_power_reason_is_axe_stealth_37b():
    """power REASON uses axe-stealth-37b (multimodal, 128K context).

    The 37b model provides the highest quality reasoning for chat.
    No speculative decoding — no shared-tokenizer draft model yet.
    """
    cfg = TOPOLOGIES["power"].models[ModelTier.REASON]
    assert cfg.model_id == "axe-stealth-37b"
    assert cfg.model_type == "multimodal"
    assert cfg.context_length == 131072
    assert cfg.draft_model_path is None
    assert cfg.continuous_batching is True


def test_power_fast_has_continuous_batching():
    """power FAST model enables continuous batching for high-throughput routing."""
    cfg = TOPOLOGIES["power"].models[ModelTier.FAST]
    assert cfg.continuous_batching is True
    assert cfg.cb_completion_batch_size > 0


def test_power_prompt_cache_size():
    """power uses prompt_cache_size=25."""
    for tier in (ModelTier.FAST, ModelTier.MID, ModelTier.REASON):
        cfg = TOPOLOGIES["power"].models[tier]
        assert cfg.prompt_cache_size == 25


def test_compact_balanced_power_are_differentiated():
    """The three topologies must NOT be identical."""
    compact = TOPOLOGIES["compact"]
    balanced = TOPOLOGIES["balanced"]
    power = TOPOLOGIES["power"]

    # compact vs balanced: different max_concurrency on FAST
    assert compact.models[ModelTier.FAST].max_concurrency != balanced.models[ModelTier.FAST].max_concurrency
    # compact vs balanced: different context_length on REASON
    assert compact.models[ModelTier.REASON].context_length != balanced.models[ModelTier.REASON].context_length
    # balanced vs power: different MID model
    assert balanced.models[ModelTier.MID].model_id != power.models[ModelTier.MID].model_id
    # balanced vs power: different REASON model entirely
    assert balanced.models[ModelTier.REASON].model_id != power.models[ModelTier.REASON].model_id


# ── to_load_params integration — topology configs produce valid payloads ───────


def test_balanced_reason_to_load_params_has_speculative():
    params = TOPOLOGIES["balanced"].models[ModelTier.REASON].to_load_params()
    assert "draft_model_path" in params
    assert "num_draft_tokens" in params
    assert params["num_draft_tokens"] == 3


def test_compact_reason_to_load_params_no_speculative():
    params = TOPOLOGIES["compact"].models[ModelTier.REASON].to_load_params()
    assert "draft_model_path" not in params
    assert "num_draft_tokens" not in params


def test_all_topology_load_params_have_required_fields():
    required = {"model_path", "model_id", "model_type", "context_length", "max_concurrency"}
    for topo_name, topo in TOPOLOGIES.items():
        for tier, cfg in topo.models.items():
            params = cfg.to_load_params()
            missing = required - params.keys()
            assert not missing, (
                f"Topology '{topo_name}' tier '{tier.value}' missing params: {missing}"
            )


# ── detect_topology() ─────────────────────────────────────────────────────────


def test_detect_topology_returns_valid_name():
    result = detect_topology()
    assert result in TOPOLOGIES


def test_detect_topology_compact_on_low_ram():
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.total = 8 * (1024**3)  # 8 GB
        result = detect_topology()
    assert result == "compact"


def test_detect_topology_balanced_on_mid_ram():
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.total = 16 * (1024**3)  # 16 GB
        result = detect_topology()
    assert result == "balanced"


def test_detect_topology_power_on_high_ram():
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.total = 64 * (1024**3)  # 64 GB
        result = detect_topology()
    assert result == "power"


def test_detect_topology_falls_back_when_psutil_missing():
    import sys
    original = sys.modules.get("psutil")
    sys.modules["psutil"] = None  # type: ignore[assignment]
    try:
        result = detect_topology()
    finally:
        if original is None:
            del sys.modules["psutil"]
        else:
            sys.modules["psutil"] = original
    assert result == DEFAULT_TOPOLOGY


# ── get_topology() ────────────────────────────────────────────────────────────


def test_get_topology_by_name():
    topo = get_topology("balanced")
    assert topo.name == "balanced"


def test_get_topology_auto_returns_valid_topology():
    topo = get_topology("auto")
    assert topo.name in TOPOLOGIES


def test_get_topology_invalid_raises():
    with pytest.raises(ValueError, match="Unknown topology"):
        get_topology("turbo")


# ── BodegaRouter ──────────────────────────────────────────────────────────────


def _make_router(topology_name: str = "balanced") -> BodegaRouter:
    """Return a BodegaRouter with a named topology (no HTTP calls)."""
    return BodegaRouter(topology=topology_name)


def test_router_resolve_model_id_fast():
    router = _make_router("balanced")
    assert router.resolve_model_id(ModelTier.FAST) == "bodega-raptor-90M"


def test_router_resolve_model_id_reason():
    router = _make_router("balanced")
    assert router.resolve_model_id(ModelTier.REASON) == "bodega-raptor-8b"


def test_router_resolve_config_returns_model_config():
    router = _make_router("balanced")
    cfg = router.resolve_config(ModelTier.REASON)
    assert isinstance(cfg, ModelConfig)
    assert cfg.model_id == "bodega-raptor-8b"


def test_router_resolve_config_balanced_reason_has_speculative():
    router = _make_router("balanced")
    cfg = router.resolve_config(ModelTier.REASON)
    assert cfg.draft_model_path == "Qwen/Qwen3-0.6B-MLX-8bit"
    assert cfg.num_draft_tokens == 3


def test_router_resolve_config_compact_reason_no_speculative():
    router = _make_router("compact")
    cfg = router.resolve_config(ModelTier.REASON)
    assert cfg.draft_model_path is None
    assert cfg.num_draft_tokens == 0


def test_router_topology_property():
    router = _make_router("compact")
    assert router.topology.name == "compact"


@pytest.mark.asyncio
async def test_router_chat_passes_model_id_to_client():
    """chat() must forward the tier's model_id to the inner BodegaInferenceClient."""
    router = _make_router("balanced")
    # Pre-populate loaded model cache so we don't hit real Bodega in unit tests.
    # The topology model_id is the canonical short alias used when Bodega loads
    # models with an explicit model_id param.  Direct-match path is tested here.
    router._loaded_models = {
        "bodega-raptor-90M": LoadedModel(model_id="bodega-raptor-90M", model_type="lm", context_length=4096),
        "bodega-raptor-8b": LoadedModel(model_id="bodega-raptor-8b", model_type="lm", context_length=32768),
    }
    router._loaded_models_ts = time.monotonic()
    expected_model = "bodega-raptor-90M"

    mock_response = {
        "choices": [{"message": {"content": "pong"}}],
        "usage": {},
    }
    router._client.chat = AsyncMock(return_value=mock_response)

    await router.chat([{"role": "user", "content": "ping"}], tier=ModelTier.FAST)

    router._client.chat.assert_awaited_once()
    call_kwargs = router._client.chat.call_args.kwargs
    assert call_kwargs["model"] == expected_model


@pytest.mark.asyncio
async def test_router_chat_simple_returns_text():
    router = _make_router("balanced")
    # Pre-populate loaded model cache — avoids network call to Bodega
    router._loaded_models = {
        "bodega-raptor-90M": LoadedModel(model_id="bodega-raptor-90M", model_type="lm", context_length=4096),
    }
    router._loaded_models_ts = time.monotonic()
    mock_response = {"choices": [{"message": {"content": "NVDA"}}], "usage": {}}
    router._client.chat = AsyncMock(return_value=mock_response)

    result = await router.chat_simple("what is the ticker for Nvidia?", tier=ModelTier.FAST)
    assert result == "NVDA"


@pytest.mark.asyncio
async def test_router_chat_simple_uses_reason_tier_by_default():
    router = _make_router("balanced")
    # Pre-populate loaded model cache — avoids network call to Bodega
    router._loaded_models = {
        "bodega-raptor-90M": LoadedModel(model_id="bodega-raptor-90M", model_type="lm", context_length=4096),
        "bodega-raptor-8b": LoadedModel(model_id="bodega-raptor-8b", model_type="lm", context_length=32768),
    }
    router._loaded_models_ts = time.monotonic()
    mock_response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    router._client.chat = AsyncMock(return_value=mock_response)

    await router.chat_simple("hello")

    call_kwargs = router._client.chat.call_args.kwargs
    assert call_kwargs["model"] == "bodega-raptor-8b"  # REASON default


@pytest.mark.asyncio
async def test_router_health_delegates_to_inner_client():
    router = _make_router()
    router._client.health = AsyncMock(return_value={"status": "ok"})

    result = await router.health()
    assert result == {"status": "ok"}
    router._client.health.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_close_delegates_to_inner_client():
    router = _make_router()
    router._client.close = AsyncMock()

    await router.close()
    router._client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_accepts_topology_object():
    """BodegaRouter can accept a Topology instance directly."""
    topo = get_topology("power")
    router = BodegaRouter(topology=topo)
    assert router.topology.name == "power"
    assert router.resolve_model_id(ModelTier.MID) == "bodega-raptor-8b"


# ── ModelManager.ensure_topology_loaded() ────────────────────────────────────


def _make_manager() -> ModelManager:
    """Return a ModelManager with a mocked BodegaInferenceClient."""
    mock_bodega = MagicMock()
    mock_bodega.load_model = AsyncMock(return_value={"status": "loaded"})
    mock_bodega.current_model = AsyncMock()
    mock_bodega.list_models = AsyncMock()
    mock_bodega.unload_model = AsyncMock()
    return ModelManager(bodega=mock_bodega)


@pytest.mark.asyncio
async def test_ensure_topology_loaded_calls_load_for_unique_models():
    """Loads distinct models once each; deduplicates FAST=MID in compact."""
    manager = _make_manager()
    topo = get_topology("compact")
    results = await manager.ensure_topology_loaded(topo)

    # compact: FAST and MID share bodega-raptor-90M → only loaded once
    # REASON is bodega-raptor-8b → loaded once
    # CLASSIFY is bodega-vertex-4b → loaded once
    # Total = 3 load calls
    assert manager.bodega.load_model.await_count == 3


@pytest.mark.asyncio
async def test_ensure_topology_loaded_dedup_status():
    """The duplicate tier must report status='skipped' with reason='duplicate_model_id'."""
    manager = _make_manager()
    topo = get_topology("compact")
    results = await manager.ensure_topology_loaded(topo)

    # In compact, MID shares model_id with FAST → skipped
    skipped_tiers = [k for k, v in results.items() if v["status"] == "skipped"]
    assert len(skipped_tiers) == 1


@pytest.mark.asyncio
async def test_ensure_topology_loaded_power_loads_four_models():
    """power has 4 distinct models — FAST(90M), MID(8b), REASON(37b), CLASSIFY(4b)."""
    manager = _make_manager()
    topo = get_topology("power")
    results = await manager.ensure_topology_loaded(topo)

    assert manager.bodega.load_model.await_count == 4
    loaded_tiers = [k for k, v in results.items() if v["status"] == "loaded"]
    assert len(loaded_tiers) == 4
    skipped_tiers = [k for k, v in results.items() if v["status"] == "skipped"]
    assert len(skipped_tiers) == 0


@pytest.mark.asyncio
async def test_ensure_topology_loaded_passes_full_params():
    """ensure_topology_loaded passes max_concurrency, prompt_cache_size to load_model."""
    manager = _make_manager()
    topo = get_topology("balanced")
    await manager.ensure_topology_loaded(topo)

    # Check that at least one call included max_concurrency and prompt_cache_size
    all_kwargs = [call.kwargs for call in manager.bodega.load_model.call_args_list]
    assert any("max_concurrency" in kw for kw in all_kwargs)
    assert any("prompt_cache_size" in kw for kw in all_kwargs)


@pytest.mark.asyncio
async def test_ensure_topology_loaded_passes_speculative_params():
    """ensure_topology_loaded passes draft_model_path and num_draft_tokens for REASON in balanced."""
    manager = _make_manager()
    topo = get_topology("balanced")
    await manager.ensure_topology_loaded(topo)

    # Find the REASON tier load call (bodega-raptor-8b)
    reason_calls = [
        call.kwargs
        for call in manager.bodega.load_model.call_args_list
        if call.kwargs.get("model_id") == "bodega-raptor-8b"
    ]
    assert reason_calls, "bodega-raptor-8b was never loaded"
    reason_kwargs = reason_calls[0]
    assert reason_kwargs["draft_model_path"] == "Qwen/Qwen3-0.6B-MLX-8bit"
    assert reason_kwargs["num_draft_tokens"] == 3


@pytest.mark.asyncio
async def test_ensure_topology_loaded_compact_no_speculative_in_params():
    """compact REASON load call must NOT include draft_model_path."""
    manager = _make_manager()
    topo = get_topology("compact")
    await manager.ensure_topology_loaded(topo)

    reason_calls = [
        call.kwargs
        for call in manager.bodega.load_model.call_args_list
        if call.kwargs.get("model_id") == "bodega-raptor-8b"
    ]
    assert reason_calls
    assert "draft_model_path" not in reason_calls[0]
    assert "num_draft_tokens" not in reason_calls[0]


@pytest.mark.asyncio
async def test_ensure_topology_loaded_handles_load_error_gracefully():
    """A failed load does not propagate — status='failed' is returned instead."""
    manager = _make_manager()
    manager.bodega.load_model = AsyncMock(side_effect=RuntimeError("connection refused"))
    topo = get_topology("power")

    results = await manager.ensure_topology_loaded(topo)

    failed = [k for k, v in results.items() if v["status"] == "failed"]
    # All 4 tiers in power should fail (FAST, MID, REASON, CLASSIFY)
    assert len(failed) == 4


@pytest.mark.asyncio
async def test_ensure_topology_loaded_returns_metadata():
    """Loaded tiers include max_concurrency and speculative_decoding in result dict."""
    manager = _make_manager()
    topo = get_topology("balanced")
    results = await manager.ensure_topology_loaded(topo)

    for tier_name, result in results.items():
        if result["status"] == "loaded":
            assert "max_concurrency" in result
            assert "prompt_cache_size" in result
            assert "speculative_decoding" in result


# ── Session 33: LoadedModel, TTL cache, context-length clamping, model-gone ──


class TestLoadedModel:
    """Tests for the LoadedModel dataclass."""

    def test_loaded_model_basic_creation(self):
        m = LoadedModel(model_id="test-model", model_type="lm", context_length=32768)
        assert m.model_id == "test-model"
        assert m.model_type == "lm"
        assert m.context_length == 32768
        assert m.status == "running"
        assert m.memory_mb == 0.0

    def test_loaded_model_with_all_fields(self):
        m = LoadedModel(
            model_id="axe-stealth-37b",
            model_type="multimodal",
            context_length=131072,
            status="running",
            memory_mb=37000.0,
        )
        assert m.model_type == "multimodal"
        assert m.context_length == 131072
        assert m.memory_mb == 37000.0

    def test_loaded_model_defaults(self):
        m = LoadedModel(model_id="x", model_type="lm", context_length=4096)
        assert m.status == "running"
        assert m.memory_mb == 0.0


class TestCacheValidity:
    """Tests for _cache_is_valid() TTL behaviour."""

    def test_cache_invalid_when_no_models(self):
        router = _make_router("balanced")
        router._loaded_models = None
        assert router._cache_is_valid() is False

    def test_cache_valid_when_fresh(self):
        router = _make_router("balanced")
        router._loaded_models = {"x": LoadedModel(model_id="x", model_type="lm", context_length=4096)}
        router._loaded_models_ts = time.monotonic()
        assert router._cache_is_valid() is True

    def test_cache_invalid_when_expired(self):
        router = _make_router("balanced")
        router._loaded_models = {"x": LoadedModel(model_id="x", model_type="lm", context_length=4096)}
        # Set timestamp 120s in the past (TTL is 60s)
        router._loaded_models_ts = time.monotonic() - 120.0
        assert router._cache_is_valid() is False

    def test_cache_invalid_at_default_timestamp(self):
        """Default _loaded_models_ts=0.0 means cache is always expired."""
        router = _make_router("balanced")
        router._loaded_models = {"x": LoadedModel(model_id="x", model_type="lm", context_length=4096)}
        # Don't set _loaded_models_ts — stays at 0.0
        assert router._cache_is_valid() is False

    def test_invalidate_model_cache_clears_both(self):
        router = _make_router("balanced")
        router._loaded_models = {"x": LoadedModel(model_id="x", model_type="lm", context_length=4096)}
        router._loaded_models_ts = time.monotonic()
        assert router._cache_is_valid() is True

        router.invalidate_model_cache()
        assert router._loaded_models is None
        assert router._loaded_models_ts == 0.0
        assert router._cache_is_valid() is False


class TestContextLengthClamping:
    """Tests for context-length aware max_tokens clamping in chat()."""

    @pytest.mark.asyncio
    async def test_chat_clamps_max_tokens_to_model_context(self):
        """When max_tokens exceeds available window, it should be clamped."""
        router = _make_router("balanced")
        # Small context model — only 2048 tokens
        router._loaded_models = {
            "small-model": LoadedModel(model_id="small-model", model_type="lm", context_length=2048),
        }
        router._loaded_models_ts = time.monotonic()
        mock_response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        router._client.chat = AsyncMock(return_value=mock_response)

        # Input is ~100 chars → ~25 tokens.  Available = 2048 - 25 - 128 = 1895
        # Requested max_tokens=4096 should be clamped
        await router.chat(
            [{"role": "user", "content": "x" * 100}],
            tier=ModelTier.FAST,
            max_tokens=4096,
        )

        call_kwargs = router._client.chat.call_args.kwargs
        assert call_kwargs["max_tokens"] < 4096
        assert call_kwargs["max_tokens"] <= 2048

    @pytest.mark.asyncio
    async def test_chat_no_clamp_when_tokens_fit(self):
        """When max_tokens fits in context window, it passes through unchanged."""
        router = _make_router("balanced")
        router._loaded_models = {
            "big-model": LoadedModel(model_id="big-model", model_type="lm", context_length=131072),
        }
        router._loaded_models_ts = time.monotonic()
        mock_response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        router._client.chat = AsyncMock(return_value=mock_response)

        await router.chat(
            [{"role": "user", "content": "short prompt"}],
            tier=ModelTier.FAST,
            max_tokens=2048,
        )

        call_kwargs = router._client.chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_chat_clamp_floor_is_256(self):
        """Even with a tiny context model, max_tokens never goes below 256."""
        router = _make_router("balanced")
        # Absurdly small context
        router._loaded_models = {
            "tiny": LoadedModel(model_id="tiny", model_type="lm", context_length=256),
        }
        router._loaded_models_ts = time.monotonic()
        mock_response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        router._client.chat = AsyncMock(return_value=mock_response)

        await router.chat(
            [{"role": "user", "content": "a" * 2000}],
            tier=ModelTier.FAST,
            max_tokens=4096,
        )

        call_kwargs = router._client.chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256


class TestModelGoneRetry:
    """Tests for model-gone retry (invalidate + re-resolve on 404/400)."""

    @pytest.mark.asyncio
    async def test_chat_retries_on_404_with_different_model(self):
        """When chat() gets 404 (model unloaded), it invalidates cache and retries."""
        router = _make_router("balanced")
        router._loaded_models = {
            "model-a": LoadedModel(model_id="model-a", model_type="lm", context_length=32768),
        }
        router._loaded_models_ts = time.monotonic()

        # First call raises 404
        mock_404 = MagicMock()
        mock_404.status_code = 404
        error = httpx.HTTPStatusError("not found", request=MagicMock(), response=mock_404)

        ok_response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        router._client.chat = AsyncMock(side_effect=[error, ok_response])

        # After invalidation, the admin endpoint returns model-b instead.
        mock_admin_resp = MagicMock()
        mock_admin_resp.status_code = 200
        mock_admin_resp.raise_for_status = MagicMock()
        mock_admin_resp.json.return_value = {
            "data": [
                {"id": "model-b", "type": "lm", "context_length": 16384, "status": "running", "memory": {}},
            ],
            "total": 1,
        }
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_admin_resp)
        router._client._get_client = AsyncMock(return_value=mock_http_client)

        result = await router.chat(
            [{"role": "user", "content": "hello"}],
            tier=ModelTier.FAST,
        )

        assert result == ok_response
        # Should have been called twice — once for 404, once for retry
        assert router._client.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_chat_reraises_non_404_errors(self):
        """Non-404/400 HTTP errors are not retried — they propagate."""
        router = _make_router("balanced")
        router._loaded_models = {
            "model-a": LoadedModel(model_id="model-a", model_type="lm", context_length=32768),
        }
        router._loaded_models_ts = time.monotonic()

        mock_500 = MagicMock()
        mock_500.status_code = 500
        error = httpx.HTTPStatusError("server error", request=MagicMock(), response=mock_500)
        router._client.chat = AsyncMock(side_effect=error)

        with pytest.raises(httpx.HTTPStatusError):
            await router.chat(
                [{"role": "user", "content": "hello"}],
                tier=ModelTier.FAST,
            )


class TestModelContextLengthHelper:
    """Tests for get_model_context_length() and loaded_models_summary()."""

    @pytest.mark.asyncio
    async def test_get_model_context_length_known_model(self):
        router = _make_router("balanced")
        router._loaded_models = {
            "my-model": LoadedModel(model_id="my-model", model_type="lm", context_length=65536),
        }
        router._loaded_models_ts = time.monotonic()

        ctx = await router.get_model_context_length("my-model")
        assert ctx == 65536

    @pytest.mark.asyncio
    async def test_get_model_context_length_unknown_model_returns_fallback(self):
        router = _make_router("balanced")
        router._loaded_models = {}
        router._loaded_models_ts = time.monotonic()

        ctx = await router.get_model_context_length("nonexistent")
        assert ctx == 4096

    @pytest.mark.asyncio
    async def test_loaded_models_summary_returns_list_of_dicts(self):
        router = _make_router("balanced")
        router._loaded_models = {
            "model-a": LoadedModel(
                model_id="model-a", model_type="lm",
                context_length=32768, memory_mb=5500.0,
            ),
            "model-b": LoadedModel(
                model_id="model-b", model_type="multimodal",
                context_length=131072, memory_mb=37000.0,
            ),
        }
        router._loaded_models_ts = time.monotonic()

        summary = await router.loaded_models_summary()
        assert len(summary) == 2
        ids = {s["id"] for s in summary}
        assert ids == {"model-a", "model-b"}
        for s in summary:
            assert "type" in s
            assert "context_length" in s
            assert "memory_mb" in s

    @pytest.mark.asyncio
    async def test_loaded_models_summary_empty_when_no_models(self):
        router = _make_router("balanced")
        router._loaded_models = {}
        router._loaded_models_ts = time.monotonic()

        summary = await router.loaded_models_summary()
        assert summary == []


class TestFetchLoadedModels:
    """Tests for _fetch_loaded_models() TTL caching and /v1/admin/loaded-models parsing."""

    @pytest.mark.asyncio
    async def test_fetch_returns_cached_when_valid(self):
        """If cache is within TTL, no HTTP call is made."""
        router = _make_router("balanced")
        cached = {
            "m1": LoadedModel(model_id="m1", model_type="lm", context_length=4096),
        }
        router._loaded_models = cached
        router._loaded_models_ts = time.monotonic()

        result = await router._fetch_loaded_models()
        assert result is cached

    @pytest.mark.asyncio
    async def test_fetch_parses_admin_endpoint_response(self):
        """_fetch_loaded_models() parses /v1/admin/loaded-models JSON correctly."""
        router = _make_router("balanced")
        # Cache expired
        router._loaded_models = None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "axe-stealth-37b",
                    "type": "multimodal",
                    "context_length": 131072,
                    "status": "running",
                    "memory": {"total_mb": 37000.0},
                },
                {
                    "id": "bodega-vertex-4b",
                    "type": "lm",
                    "context_length": 16384,
                    "status": "running",
                    "memory": {"total_mb": 5500.0},
                },
            ],
            "total": 2,
        }

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        router._client._get_client = AsyncMock(return_value=mock_http_client)

        result = await router._fetch_loaded_models()

        assert len(result) == 2
        assert "axe-stealth-37b" in result
        assert result["axe-stealth-37b"].model_type == "multimodal"
        assert result["axe-stealth-37b"].context_length == 131072
        assert result["axe-stealth-37b"].memory_mb == 37000.0
        assert result["bodega-vertex-4b"].model_type == "lm"

    @pytest.mark.asyncio
    async def test_fetch_skips_non_running_models(self):
        """Models with status != 'running' are excluded."""
        router = _make_router("balanced")
        router._loaded_models = None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "running-model", "type": "lm", "context_length": 4096, "status": "running", "memory": {}},
                {"id": "loading-model", "type": "lm", "context_length": 4096, "status": "loading", "memory": {}},
            ],
            "total": 2,
        }

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        router._client._get_client = AsyncMock(return_value=mock_http_client)

        result = await router._fetch_loaded_models()
        assert "running-model" in result
        assert "loading-model" not in result

    @pytest.mark.asyncio
    async def test_fetch_updates_timestamp_on_success(self):
        """After a successful fetch, _loaded_models_ts is updated."""
        router = _make_router("balanced")
        router._loaded_models = None
        router._loaded_models_ts = 0.0

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [], "total": 0}

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        router._client._get_client = AsyncMock(return_value=mock_http_client)

        before = time.monotonic()
        await router._fetch_loaded_models()
        after = time.monotonic()

        assert router._loaded_models_ts >= before
        assert router._loaded_models_ts <= after


class TestTierFallbackWithLoadedModel:
    """Tests for tier-aware fallback using LoadedModel.model_type."""

    @pytest.mark.asyncio
    async def test_reason_prefers_lm_over_multimodal(self):
        """REASON tier should prefer lm model over multimodal when both available."""
        router = _make_router("balanced")
        router._loaded_models = {
            "multimodal-37b": LoadedModel(model_id="multimodal-37b", model_type="multimodal", context_length=131072),
            "lm-4b": LoadedModel(model_id="lm-4b", model_type="lm", context_length=16384),
        }
        router._loaded_models_ts = time.monotonic()

        model = await router._resolve_available_model(ModelTier.REASON, auto_load=False)
        # REASON prefers lm (iteration order may vary, but lm is checked first)
        assert model == "lm-4b"

    @pytest.mark.asyncio
    async def test_reason_falls_back_to_multimodal_when_no_lm(self):
        """REASON falls back to multimodal if no lm models are loaded."""
        router = _make_router("balanced")
        router._loaded_models = {
            "vision-model": LoadedModel(model_id="vision-model", model_type="multimodal", context_length=131072),
        }
        router._loaded_models_ts = time.monotonic()

        model = await router._resolve_available_model(ModelTier.REASON, auto_load=False)
        assert model == "vision-model"

    @pytest.mark.asyncio
    async def test_fast_prefers_lm_over_multimodal(self):
        """FAST tier should prefer lm (small, fast) over multimodal (large)."""
        router = _make_router("balanced")
        router._loaded_models = {
            "big-multimodal": LoadedModel(model_id="big-multimodal", model_type="multimodal", context_length=131072),
            "small-lm": LoadedModel(model_id="small-lm", model_type="lm", context_length=4096),
        }
        router._loaded_models_ts = time.monotonic()

        model = await router._resolve_available_model(ModelTier.FAST, auto_load=False)
        assert model == "small-lm"
