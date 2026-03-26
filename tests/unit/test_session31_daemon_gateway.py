"""Tests for the Daemon InferenceProxy — backpressure gateway to Bodega.

Tests verify:
    - Per-model semaphore registration and slot management
    - Concurrency limiting (max_concurrency enforced)
    - Queue position tracking (waiting count)
    - CLASSIFY model routing
    - Pressure reporting (idle/nominal/busy/full/saturated)
    - Snapshot serialization
    - Timeout on slot acquisition
    - Unknown model graceful passthrough
    - Unregister model
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from octane.daemon.inference_proxy import InferenceProxy, ModelSlot


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_proxy() -> InferenceProxy:
    """Build an InferenceProxy with a mock BodegaInferenceClient."""
    bodega = MagicMock()
    bodega.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })
    bodega.chat_simple = AsyncMock(return_value="classified: web_search")
    proxy = InferenceProxy(bodega)
    return proxy


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_model_creates_slot():
    proxy = _make_proxy()
    proxy.register_model("test-model", max_concurrency=3)
    slot = proxy._slots["test-model"]
    assert slot.max_concurrency == 3
    assert slot.active == 0
    assert slot.waiting == 0
    assert slot.available == 3


def test_register_duplicate_model_is_noop():
    proxy = _make_proxy()
    proxy.register_model("test-model", max_concurrency=3)
    proxy.register_model("test-model", max_concurrency=99)  # Should NOT overwrite
    assert proxy._slots["test-model"].max_concurrency == 3


def test_unregister_model():
    proxy = _make_proxy()
    proxy.register_model("test-model", max_concurrency=2)
    proxy.unregister_model("test-model")
    assert "test-model" not in proxy._slots


def test_unregister_unknown_is_noop():
    proxy = _make_proxy()
    proxy.unregister_model("nonexistent")  # Should not crash


# ── Slot acquisition ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_slot_basic_acquire_release():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=2)

    async with proxy.slot("m1") as pos:
        slot = proxy._slots["m1"]
        assert slot.active == 1
        assert pos >= 0

    # After release, active should be 0
    assert proxy._slots["m1"].active == 0
    assert proxy._slots["m1"].total_served == 1


@pytest.mark.asyncio
async def test_slot_concurrency_limit():
    """Only max_concurrency tasks should be active simultaneously."""
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=2)

    active_count = 0
    max_active = 0

    async def worker():
        nonlocal active_count, max_active
        async with proxy.slot("m1"):
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)  # Hold slot briefly
            active_count -= 1

    # Run 5 workers — max 2 should be active at any time
    await asyncio.gather(*(worker() for _ in range(5)))

    assert max_active <= 2
    assert proxy._slots["m1"].total_served == 5


@pytest.mark.asyncio
async def test_slot_timeout():
    """TimeoutError raised when no slot frees within timeout."""
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=1)

    # Hold the slot
    async with proxy.slot("m1"):
        with pytest.raises(asyncio.TimeoutError):
            async with proxy.slot("m1", timeout=0.1):
                pass  # Should never reach here


@pytest.mark.asyncio
async def test_slot_unknown_model_passthrough():
    """Unknown model_id yields position 0 without blocking."""
    proxy = _make_proxy()
    async with proxy.slot("unknown-model") as pos:
        assert pos == 0


# ── Proxied inference methods ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_acquires_slot():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=1)

    result = await proxy.chat(
        messages=[{"role": "user", "content": "test"}],
        model="m1",
    )
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert proxy._slots["m1"].total_served == 1
    proxy.bodega.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_simple_acquires_slot():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=2)

    text = await proxy.chat_simple(prompt="hello", model="m1")
    assert text == "classified: web_search"
    assert proxy._slots["m1"].total_served == 1


@pytest.mark.asyncio
async def test_classify_uses_classify_model():
    proxy = _make_proxy()
    proxy.register_model("bodega-vertex-4b", max_concurrency=2)
    proxy.classify_model = "bodega-vertex-4b"

    result = await proxy.classify(prompt="web_search or memory_recall?")
    assert result == "classified: web_search"
    assert proxy._slots["bodega-vertex-4b"].total_served == 1


@pytest.mark.asyncio
async def test_classify_falls_back_to_current():
    """When no CLASSIFY model is set, falls back to 'current'."""
    proxy = _make_proxy()
    # No classify model registered, no register_model call
    result = await proxy.classify(prompt="test")
    assert result == "classified: web_search"
    proxy.bodega.chat_simple.assert_awaited_once()


# ── Pressure reporting ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pressure_idle():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=4)
    report = proxy.pressure_report()
    assert report["m1"] == "idle"


@pytest.mark.asyncio
async def test_pressure_nominal():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=4)

    async with proxy.slot("m1"):
        report = proxy.pressure_report()
        assert report["m1"] == "nominal"  # 1/4 = 0.25 < 0.5


@pytest.mark.asyncio
async def test_pressure_busy():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=4)

    async with proxy.slot("m1"):
        async with proxy.slot("m1"):
            async with proxy.slot("m1"):
                report = proxy.pressure_report()
                assert report["m1"] == "busy"  # 3/4 = 0.75


@pytest.mark.asyncio
async def test_pressure_full():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=2)

    async with proxy.slot("m1"):
        async with proxy.slot("m1"):
            report = proxy.pressure_report()
            assert report["m1"] == "full"  # 2/2, no waiters


# ── Snapshot ──────────────────────────────────────────────────────────────────

def test_snapshot_structure():
    proxy = _make_proxy()
    proxy.register_model("m1", max_concurrency=2)
    proxy.register_model("m2", max_concurrency=4)
    proxy.classify_model = "m2"

    snap = proxy.snapshot()
    assert snap["classify_model"] == "m2"
    assert snap["total_registered"] == 2
    assert "m1" in snap["models"]
    assert "m2" in snap["models"]
    assert snap["models"]["m1"]["max_concurrency"] == 2
    assert snap["models"]["m2"]["max_concurrency"] == 4


# ── ModelSlot basics ──────────────────────────────────────────────────────────

def test_model_slot_available():
    slot = ModelSlot(
        model_id="test",
        max_concurrency=3,
        semaphore=asyncio.Semaphore(3),
        active=1,
    )
    assert slot.available == 2


def test_model_slot_snapshot():
    slot = ModelSlot(
        model_id="test",
        max_concurrency=3,
        semaphore=asyncio.Semaphore(3),
        active=1,
        waiting=2,
        total_served=10,
        total_waited_ms=500.0,
    )
    snap = slot.snapshot()
    assert snap["model_id"] == "test"
    assert snap["max_concurrency"] == 3
    assert snap["active"] == 1
    assert snap["waiting"] == 2
    assert snap["available"] == 2
    assert snap["total_served"] == 10
    assert snap["avg_wait_ms"] == 50.0


# ── Topology CLASSIFY tier ───────────────────────────────────────────────────

def test_topology_classify_tier_exists_in_all_topologies():
    from octane.tools.topology import TOPOLOGIES, ModelTier

    for topo_name, topo in TOPOLOGIES.items():
        assert ModelTier.CLASSIFY in topo.models, (
            f"Topology '{topo_name}' is missing CLASSIFY tier"
        )
        cfg = topo.models[ModelTier.CLASSIFY]
        assert cfg.model_id == "bodega-vertex-4b"
        assert cfg.model_path == "srswti/bodega-vertex-4b"
        assert cfg.max_concurrency >= 1


def test_topology_classify_never_idle_unloaded():
    from octane.daemon.model_manager import IDLE_POLICIES

    for topo_name, policy in IDLE_POLICIES.items():
        assert policy.classify_idle_sec == 0.0, (
            f"CLASSIFY should never auto-unload in '{topo_name}'"
        )


# ── Model alias resolution (Session 32 bug fix) ──────────────────────────────

def test_register_model_with_path_alias():
    """register_model with model_path creates a case-insensitive alias."""
    proxy = _make_proxy()
    proxy.register_model(
        "bodega-raptor-90M", max_concurrency=8,
        model_path="SRSWTI/bodega-raptor-90m",
    )
    # Exact match
    assert proxy._resolve("bodega-raptor-90M") == "bodega-raptor-90M"
    # Case-insensitive short ID
    assert proxy._resolve("bodega-raptor-90m") == "bodega-raptor-90M"
    # Full model_path (as Bodega returns in API)
    assert proxy._resolve("SRSWTI/bodega-raptor-90m") == "bodega-raptor-90M"
    # Full model_path uppercase
    assert proxy._resolve("srswti/bodega-raptor-90m") == "bodega-raptor-90M"
    # Unknown
    assert proxy._resolve("nonexistent") is None


@pytest.mark.asyncio
async def test_slot_with_path_alias():
    """slot() resolves model_path alias to the registered slot."""
    proxy = _make_proxy()
    proxy.register_model(
        "bodega-raptor-90M", max_concurrency=4,
        model_path="SRSWTI/bodega-raptor-90m",
    )
    # Use the Bodega-format path — should match the registered slot.
    async with proxy.slot("srswti/bodega-raptor-90m") as pos:
        assert pos >= 0
        assert proxy._slots["bodega-raptor-90M"].active == 1

    assert proxy._slots["bodega-raptor-90M"].total_served == 1


@pytest.mark.asyncio
async def test_chat_with_path_alias():
    """chat() through proxy with Bodega model_path format."""
    proxy = _make_proxy()
    proxy.register_model(
        "bodega-raptor-8b", max_concurrency=6,
        model_path="srswti/bodega-raptor-8b-mxfp4",
    )
    result = await proxy.chat(
        messages=[{"role": "user", "content": "test"}],
        model="srswti/bodega-raptor-8b-mxfp4",  # Bodega path format
    )
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert proxy._slots["bodega-raptor-8b"].total_served == 1


def test_unregister_cleans_aliases():
    """unregister_model removes both the slot and all aliases."""
    proxy = _make_proxy()
    proxy.register_model(
        "bodega-raptor-8b", max_concurrency=6,
        model_path="srswti/bodega-raptor-8b-mxfp4",
    )
    assert proxy._resolve("srswti/bodega-raptor-8b-mxfp4") == "bodega-raptor-8b"

    proxy.unregister_model("bodega-raptor-8b")
    assert "bodega-raptor-8b" not in proxy._slots
    assert proxy._resolve("srswti/bodega-raptor-8b-mxfp4") is None
    assert proxy._resolve("bodega-raptor-8b") is None


# ── Self-referential daemon guard (Session 32 bug fix) ────────────────────────

def test_daemon_check_skips_when_we_are_daemon(tmp_path, monkeypatch):
    """BodegaInferenceClient._check_daemon() returns False when our PID matches daemon PID."""
    from octane.tools.bodega_inference import BodegaInferenceClient
    import os

    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(str(os.getpid()))

    monkeypatch.setattr(
        "octane.daemon.client.get_pid_path",
        lambda: pid_file,
    )
    monkeypatch.setattr(
        "octane.daemon.client.is_daemon_running",
        lambda: True,
    )

    client = BodegaInferenceClient.__new__(BodegaInferenceClient)
    client._daemon_checked = False
    client._daemon_available = False

    assert client._check_daemon() is False
    assert client._daemon_available is False


def test_daemon_check_allows_when_different_pid(tmp_path, monkeypatch):
    """_check_daemon() returns True when daemon is running but we're not it."""
    from octane.tools.bodega_inference import BodegaInferenceClient
    import os

    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(str(os.getpid() + 99999))  # Different PID

    monkeypatch.setattr(
        "octane.daemon.client.get_pid_path",
        lambda: pid_file,
    )
    monkeypatch.setattr(
        "octane.daemon.client.is_daemon_running",
        lambda: True,
    )

    client = BodegaInferenceClient.__new__(BodegaInferenceClient)
    client._daemon_checked = False
    client._daemon_available = False

    assert client._check_daemon() is True
    assert client._daemon_available is True
