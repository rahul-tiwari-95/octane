"""Model Manager — aggressive model lifecycle for the Octane Daemon.

Controls when models are loaded and unloaded from Bodega.  The key insight:
on a 16 GB machine, the 8B REASON model (~5 GB VRAM) should NOT stay loaded
when nobody's using it.  On a 64 GB machine, keep everything loaded.

Idle detection runs as a background asyncio task.  When a model's idle time
exceeds its threshold, the manager unloads it via Bodega's admin API.  When
a request needs that tier, the manager reloads it on demand.

Memory budgets per topology:
    compact  (8-12 GB):  Unload REASON after 2 min idle. MID shares FAST model.
    balanced (16-24 GB): Unload REASON after 5 min idle. MID shares FAST model.
    power    (32+ GB):   Never auto-unload. All tiers stay resident.

The manager integrates with DaemonState to track loaded models and with
the PoolManager for Bodega HTTP client access.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

from octane.daemon.state import DaemonState, LoadedModel

logger = structlog.get_logger().bind(component="model_manager")


# ── Idle Thresholds (seconds) ─────────────────────────────────────────────────

@dataclass(frozen=True)
class IdlePolicy:
    """Per-topology idle thresholds for each tier.

    0 = never auto-unload (model stays resident).
    """

    fast_idle_sec: float = 0.0       # FAST: always resident (tiny model)
    mid_idle_sec: float = 0.0        # MID: always resident (often same as FAST)
    reason_idle_sec: float = 300.0   # REASON: 5 min default


IDLE_POLICIES: dict[str, IdlePolicy] = {
    "compact": IdlePolicy(
        fast_idle_sec=0.0,         # 90M — always loaded, negligible memory
        mid_idle_sec=0.0,          # Same model as FAST on compact
        reason_idle_sec=120.0,     # 8B — unload after 2 min (tight RAM)
    ),
    "balanced": IdlePolicy(
        fast_idle_sec=0.0,
        mid_idle_sec=0.0,
        reason_idle_sec=300.0,     # 8B — unload after 5 min
    ),
    "power": IdlePolicy(
        fast_idle_sec=0.0,
        mid_idle_sec=0.0,
        reason_idle_sec=0.0,       # Never unload — 64 GB can hold everything
    ),
}


# ── Estimated Model Memory (MB) ──────────────────────────────────────────────

MODEL_MEMORY_ESTIMATES: dict[str, float] = {
    "bodega-raptor-90M": 180.0,      # ~180 MB (tiny)
    "bodega-raptor-1b": 900.0,       # ~900 MB (0.9B params)
    "bodega-raptor-8b": 5000.0,      # ~5 GB (8B params, 4-bit quant)
}


# ── Model Manager ────────────────────────────────────────────────────────────


class ModelManager:
    """Manages model lifecycle — loading, unloading, idle detection.

    The manager runs a background task that periodically checks for idle
    models and unloads them.  When a model is needed, ensure_loaded()
    loads it on demand.

    Args:
        state:          DaemonState for model registry.
        topology_name:  Which topology determines idle thresholds.
        check_interval: How often to run idle checks (seconds).
        bodega_client:  BodegaInferenceClient for load/unload API calls.
                        If None, operates in dry-run mode (state tracking only).
    """

    def __init__(
        self,
        state: DaemonState,
        topology_name: str = "balanced",
        check_interval: float = 30.0,
        bodega_client: Any = None,
    ) -> None:
        self.state = state
        self.topology_name = topology_name
        self.check_interval = check_interval
        self.bodega = bodega_client
        self.policy = IDLE_POLICIES.get(topology_name, IDLE_POLICIES["balanced"])
        self._idle_task: asyncio.Task | None = None
        self._running = False

        # Stats
        self.total_loads: int = 0
        self.total_unloads: int = 0

    def get_idle_threshold(self, tier: str) -> float:
        """Get idle threshold for a model tier.

        Returns 0.0 if the model should never be auto-unloaded.
        """
        tier_lower = tier.lower()
        if tier_lower == "fast":
            return self.policy.fast_idle_sec
        elif tier_lower == "mid":
            return self.policy.mid_idle_sec
        elif tier_lower == "reason":
            return self.policy.reason_idle_sec
        return 0.0  # Unknown tier → never unload

    async def start(self) -> None:
        """Start the background idle-detection loop."""
        if self._running:
            return
        self._running = True
        self._idle_task = asyncio.create_task(self._idle_loop())
        logger.info(
            "model_manager_started",
            topology=self.topology_name,
            reason_idle_sec=self.policy.reason_idle_sec,
        )

    async def stop(self) -> None:
        """Stop the background idle-detection loop."""
        self._running = False
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        self._idle_task = None
        logger.info("model_manager_stopped")

    async def register_loaded(
        self,
        model_id: str,
        tier: str,
        estimated_memory_mb: float | None = None,
    ) -> None:
        """Register a model as loaded (called on daemon startup or after load).

        Args:
            model_id:  Bodega model ID (e.g. "bodega-raptor-8b").
            tier:      Model tier ("fast", "mid", "reason").
            estimated_memory_mb: Override memory estimate.
        """
        mem = estimated_memory_mb
        if mem is None:
            mem = MODEL_MEMORY_ESTIMATES.get(model_id, 0.0)

        model = LoadedModel(
            model_id=model_id,
            tier=tier,
            estimated_memory_mb=mem,
        )
        await self.state.register_model(model)
        self.total_loads += 1
        logger.info(
            "model_registered",
            model_id=model_id,
            tier=tier,
            memory_mb=mem,
        )

    async def record_usage(self, model_id: str) -> None:
        """Record that a model was used (resets idle timer)."""
        await self.state.touch_model(model_id)

    async def ensure_loaded(self, model_id: str, tier: str) -> bool:
        """Ensure a model is loaded. Loads it on demand if needed.

        Returns True if the model is (or was successfully) loaded.
        Returns False if loading failed (no Bodega client, or API error).
        """
        existing = await self.state.get_model(model_id)
        if existing is not None:
            # Already loaded — just touch it
            await self.record_usage(model_id)
            return True

        # Need to load — requires Bodega client
        if self.bodega is None:
            logger.warning("model_load_skipped_no_client", model_id=model_id)
            return False

        return await self._load_model(model_id, tier)

    async def _load_model(self, model_id: str, tier: str) -> bool:
        """Load a model via Bodega admin API."""
        try:
            from octane.tools.topology import get_topology, ModelTier

            tier_map = {"fast": ModelTier.FAST, "mid": ModelTier.MID, "reason": ModelTier.REASON}
            model_tier = tier_map.get(tier.lower())
            if model_tier is None:
                logger.error("unknown_tier", tier=tier)
                return False

            topo = get_topology(self.topology_name)
            config = topo.resolve_config(model_tier)
            params = config.to_load_params()

            logger.info("model_loading", model_id=model_id, tier=tier)
            success = await self.bodega.load_model(params)

            if success:
                await self.register_loaded(model_id, tier)
                return True
            else:
                logger.error("model_load_failed", model_id=model_id)
                return False

        except Exception as exc:
            logger.error("model_load_error", model_id=model_id, error=str(exc))
            return False

    async def _unload_model(self, model_id: str) -> bool:
        """Unload a model via Bodega admin API."""
        if self.bodega is None:
            logger.warning("model_unload_skipped_no_client", model_id=model_id)
            # Still remove from state registry
            await self.state.unregister_model(model_id)
            self.total_unloads += 1
            return True

        try:
            logger.info("model_unloading", model_id=model_id)
            success = await self.bodega.unload_model(model_id)
            if success:
                await self.state.unregister_model(model_id)
                self.total_unloads += 1
                logger.info("model_unloaded", model_id=model_id)
                return True
            else:
                logger.error("model_unload_failed", model_id=model_id)
                return False
        except Exception as exc:
            logger.error("model_unload_error", model_id=model_id, error=str(exc))
            return False

    async def check_idle(self) -> list[str]:
        """Check all loaded models for idle timeout. Unloads idle models.

        Returns list of model IDs that were unloaded.
        """
        unloaded: list[str] = []
        models = await self.state.get_all_models()

        for model in models:
            threshold = self.get_idle_threshold(model.tier)
            if threshold <= 0:
                continue  # Never auto-unload this tier

            if model.idle_seconds >= threshold:
                logger.info(
                    "model_idle_timeout",
                    model_id=model.model_id,
                    tier=model.tier,
                    idle_seconds=round(model.idle_seconds, 1),
                    threshold=threshold,
                )
                success = await self._unload_model(model.model_id)
                if success:
                    unloaded.append(model.model_id)

        return unloaded

    async def _idle_loop(self) -> None:
        """Background task: periodically check for idle models."""
        try:
            while self._running:
                await asyncio.sleep(self.check_interval)
                if not self._running:
                    break
                try:
                    unloaded = await self.check_idle()
                    if unloaded:
                        logger.info(
                            "idle_sweep_complete",
                            unloaded=unloaded,
                            remaining=len(await self.state.get_all_models()),
                        )
                except Exception as exc:
                    logger.error("idle_check_error", error=str(exc))
        except asyncio.CancelledError:
            pass

    def snapshot(self) -> dict[str, Any]:
        """Manager state for monitoring."""
        return {
            "topology": self.topology_name,
            "policy": {
                "fast_idle_sec": self.policy.fast_idle_sec,
                "mid_idle_sec": self.policy.mid_idle_sec,
                "reason_idle_sec": self.policy.reason_idle_sec,
            },
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "running": self._running,
        }
