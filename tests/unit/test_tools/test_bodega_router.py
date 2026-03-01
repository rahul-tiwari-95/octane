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
from octane.tools.bodega_router import BodegaRouter
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


def test_topology_power_mid_is_1b():
    """Power topology upgrades MID to bodega-raptor-1b."""
    assert TOPOLOGIES["power"].models[ModelTier.MID].model_id == "bodega-raptor-1b"


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
    """power routes up to 4 parallel classification/extraction requests."""
    assert TOPOLOGIES["power"].models[ModelTier.FAST].max_concurrency == 4


def test_power_reason_max_concurrency_is_2():
    """power allows 2 parallel deep-reasoning requests."""
    assert TOPOLOGIES["power"].models[ModelTier.REASON].max_concurrency == 2


def test_power_reason_num_draft_tokens_is_5():
    """power uses 5 draft tokens (vs balanced's 3) for more speculative gains."""
    cfg = TOPOLOGIES["power"].models[ModelTier.REASON]
    assert cfg.num_draft_tokens == 5
    assert cfg.draft_model_path == "Qwen/Qwen3-0.6B-MLX-8bit"


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
    # balanced vs power: different num_draft_tokens on REASON
    assert balanced.models[ModelTier.REASON].num_draft_tokens != power.models[ModelTier.REASON].num_draft_tokens


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
    mock_response = {"choices": [{"message": {"content": "NVDA"}}], "usage": {}}
    router._client.chat = AsyncMock(return_value=mock_response)

    result = await router.chat_simple("what is the ticker for Nvidia?", tier=ModelTier.FAST)
    assert result == "NVDA"


@pytest.mark.asyncio
async def test_router_chat_simple_uses_reason_tier_by_default():
    router = _make_router("balanced")
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
    assert router.resolve_model_id(ModelTier.MID) == "bodega-raptor-1b"


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
    # Total = 2 load calls
    assert manager.bodega.load_model.await_count == 2


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
async def test_ensure_topology_loaded_power_loads_three_models():
    """power has 3 distinct models — all 3 should be loaded."""
    manager = _make_manager()
    topo = get_topology("power")
    results = await manager.ensure_topology_loaded(topo)

    assert manager.bodega.load_model.await_count == 3
    loaded_tiers = [k for k, v in results.items() if v["status"] == "loaded"]
    assert len(loaded_tiers) == 3


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
    # All 3 tiers in power should fail
    assert len(failed) == 3


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
