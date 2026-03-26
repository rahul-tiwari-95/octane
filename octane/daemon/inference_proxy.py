"""Inference Proxy — backpressure-aware gateway to Bodega.

The daemon is the SOLE gateway to Bodega.  Every inference request —
from the CLI, from OSA, from agents — enters through this proxy.

Engineering model:
    Bodega runs on a single port (localhost:44468) with a multi-model
    registry.  Each loaded model has a max_concurrency parameter that
    limits how many parallel requests Bodega will run on that model.

    The InferenceProxy enforces the SAME concurrency limits on the
    client side via per-model ``asyncio.Semaphore``s.  This prevents
    request pile-up at the Bodega level — callers wait in an ordered
    queue local to the daemon instead of hammering the HTTP endpoint.

    Think of it as a runway controller at an airport:
        - Each model is a runway with N landing slots.
        - The proxy is the tower that holds planes (requests) in a
          holding pattern when all slots are occupied.
        - Position-in-queue feedback lets callers show "⏳ queued (2 ahead)".

Dedicated CLASSIFY model:
    bodega-vertex-4b is loaded on daemon start and reserved for internal
    operations: query classification, routing, dimension planning, judging.
    User-facing inference NEVER competes with these small, fast operations
    because vertex-4b has its own semaphore slot.

Usage:
    proxy = InferenceProxy(bodega_client)
    proxy.register_model("bodega-raptor-90M", max_concurrency=8)
    proxy.register_model("axe-stealth-37b", max_concurrency=4)
    proxy.register_model("bodega-vertex-4b", max_concurrency=2)

    # Acquire a slot (waits if busy), then call Bodega:
    async with proxy.slot("axe-stealth-37b") as position:
        result = await bodega.chat(messages, model="axe-stealth-37b")
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

logger = structlog.get_logger().bind(component="inference_proxy")


# ── Slot metrics ──────────────────────────────────────────────────────────────

@dataclass
class ModelSlot:
    """Semaphore + metrics for a single model."""

    model_id: str
    max_concurrency: int
    semaphore: asyncio.Semaphore
    active: int = 0
    waiting: int = 0
    total_served: int = 0
    total_waited_ms: float = 0.0

    @property
    def available(self) -> int:
        return self.max_concurrency - self.active

    def snapshot(self) -> dict[str, Any]:
        avg_wait = (
            round(self.total_waited_ms / self.total_served, 1)
            if self.total_served
            else 0.0
        )
        return {
            "model_id": self.model_id,
            "max_concurrency": self.max_concurrency,
            "active": self.active,
            "waiting": self.waiting,
            "available": self.available,
            "total_served": self.total_served,
            "avg_wait_ms": avg_wait,
        }


# ── InferenceProxy ────────────────────────────────────────────────────────────

class InferenceProxy:
    """Backpressure-aware gateway to Bodega.

    The proxy holds per-model semaphores and routes requests through them.
    When all slots for a model are occupied, callers block until a slot
    frees up.  Queue position is available for real-time UX feedback.
    """

    def __init__(self, bodega_client: Any) -> None:
        self.bodega = bodega_client
        self._slots: dict[str, ModelSlot] = {}
        # alias → canonical model_id (case-insensitive path/id lookup)
        self._aliases: dict[str, str] = {}
        # Classify model ID — set by lifecycle after loading vertex-4b.
        self.classify_model: str | None = None

    def _resolve(self, model_id: str) -> str | None:
        """Resolve any model identifier to its canonical registered name.

        Handles: exact match, case-insensitive match, and full model_path
        (e.g. 'srswti/bodega-raptor-90m' → 'bodega-raptor-90M').
        """
        if model_id in self._slots:
            return model_id
        return self._aliases.get(model_id.lower())

    def register_model(
        self,
        model_id: str,
        max_concurrency: int,
        model_path: str | None = None,
    ) -> None:
        """Register a model and create its concurrency semaphore.

        Also registers aliases (case-insensitive model_id and model_path)
        so callers using Bodega's full path format are matched correctly.
        """
        if model_id in self._slots:
            logger.debug("model_already_registered", model_id=model_id)
            return
        self._slots[model_id] = ModelSlot(
            model_id=model_id,
            max_concurrency=max_concurrency,
            semaphore=asyncio.Semaphore(max_concurrency),
        )
        # Register aliases for flexible lookup.
        self._aliases[model_id.lower()] = model_id
        if model_path:
            self._aliases[model_path.lower()] = model_id
        logger.info(
            "proxy_model_registered",
            model_id=model_id,
            max_concurrency=max_concurrency,
        )

    def unregister_model(self, model_id: str) -> None:
        """Unregister a model (after Bodega unload)."""
        canonical = self._resolve(model_id) or model_id
        self._slots.pop(canonical, None)
        # Remove all aliases pointing to this model.
        self._aliases = {
            alias: cid for alias, cid in self._aliases.items()
            if cid != canonical
        }
        logger.info("proxy_model_unregistered", model_id=canonical)

    @asynccontextmanager
    async def slot(
        self,
        model_id: str,
        timeout: float = 300.0,
    ) -> AsyncIterator[int]:
        """Acquire an inference slot for *model_id*.

        Context manager: holds a semaphore permit for the duration of
        the ``async with`` block.  Callers run their Bodega request
        inside the block.

        Args:
            model_id: Bodega model to target.
            timeout:  Max seconds to wait for a free slot.

        Yields:
            The caller's position in the wait queue (0 = no wait).

        Raises:
            asyncio.TimeoutError: If no slot freed within *timeout*.
            KeyError: If *model_id* is not registered.
        """
        canonical = self._resolve(model_id)
        ms = self._slots.get(canonical) if canonical else None
        if ms is None:
            # Unknown model — pass through without gating (graceful).
            logger.warning("proxy_unknown_model", model_id=model_id)
            yield 0
            return

        ms.waiting += 1
        position = ms.waiting
        t0 = time.monotonic()

        if position > 0 and ms.available == 0:
            logger.info(
                "proxy_queued",
                model_id=model_id,
                position=position,
                active=ms.active,
            )

        try:
            await asyncio.wait_for(ms.semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            ms.waiting -= 1
            logger.warning(
                "proxy_slot_timeout",
                model_id=model_id,
                waited_sec=round(time.monotonic() - t0, 1),
            )
            raise

        ms.waiting -= 1
        ms.active += 1
        waited_ms = (time.monotonic() - t0) * 1000
        ms.total_waited_ms += waited_ms
        ms.total_served += 1

        if waited_ms > 100:
            logger.info(
                "proxy_slot_acquired",
                model_id=model_id,
                waited_ms=round(waited_ms, 1),
            )

        try:
            yield position
        finally:
            ms.active -= 1
            ms.semaphore.release()

    # ── Convenience inference methods ─────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "current",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Proxied chat completion — acquires a slot, then calls Bodega."""
        async with self.slot(model, timeout=timeout):
            return await self.bodega.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )

    async def chat_simple(
        self,
        prompt: str,
        system: str = "",
        model: str = "current",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 300.0,
    ) -> str:
        """Proxied simple chat — slot-guarded."""
        async with self.slot(model, timeout=timeout):
            return await self.bodega.chat_simple(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def classify(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Route a classification request through the CLASSIFY model.

        Uses the daemon-exclusive vertex-4b model.  Falls back to
        'current' if no classify model is registered.
        """
        model = self.classify_model or "current"
        async with self.slot(model, timeout=30.0):
            return await self.bodega.chat_simple(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    # ── Status / monitoring ───────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Full proxy state for daemon status/watch."""
        return {
            "classify_model": self.classify_model,
            "models": {
                mid: ms.snapshot() for mid, ms in self._slots.items()
            },
            "total_registered": len(self._slots),
        }

    def pressure_report(self) -> dict[str, str]:
        """Human-readable pressure per model: idle / nominal / busy / saturated."""
        report: dict[str, str] = {}
        for mid, ms in self._slots.items():
            ratio = ms.active / ms.max_concurrency if ms.max_concurrency else 0
            if ms.active == 0:
                label = "idle"
            elif ratio < 0.5:
                label = "nominal"
            elif ratio < 1.0:
                label = "busy"
            else:
                label = "saturated" if ms.waiting > 0 else "full"
            report[mid] = label
        return report
