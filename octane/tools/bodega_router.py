"""BodegaRouter — tier-aware routing layer over BodegaInferenceClient.

Agents use BodegaRouter instead of BodegaInferenceClient so that every
inference call is automatically directed to the model best suited for its
complexity / latency requirements.

Quick-start::

    from octane.tools.bodega_router import BodegaRouter
    from octane.tools.topology import ModelTier

    router = BodegaRouter()                           # topology="auto"
    text   = await router.chat_simple(
        "Extract the ticker symbol from: buy NVDA now",
        tier=ModelTier.FAST,
    )

Tier semantics (matched to live Bodega models):
    FAST   — bodega-raptor-90M  — keyword extraction, routing, classification
    MID    — bodega-raptor-90M  — chunk summarization (upgrades to 1B in power)
    REASON — bodega-raptor-8b   — deep analysis, synthesis, evaluation
    EMBED  — handled in-process; not routed through Bodega

The router is API-compatible with BodegaInferenceClient for all inference
methods, with an extra optional ``tier=`` parameter on each.  Admin methods
(load_model / unload_model) are intentionally omitted — those stay on the
raw BodegaInferenceClient used exclusively by SysStat.ModelManager.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from octane.tools.bodega_inference import BodegaInferenceClient
from octane.tools.topology import ModelConfig, ModelTier, Topology, get_topology

logger = structlog.get_logger().bind(component="bodega_router")


class BodegaRouter:
    """Tier-aware routing wrapper around :class:`BodegaInferenceClient`.

    Args:
        topology: Topology name (``'auto'``, ``'compact'``, ``'balanced'``,
                  ``'power'``) **or** a pre-built :class:`Topology` object.
                  Defaults to ``'auto'`` which auto-detects from system RAM.
        base_url: Bodega Inference Engine base URL.  Falls back to
                  ``settings.bodega_inference_url``.
        timeout:  Default HTTP read timeout in seconds.
    """

    def __init__(
        self,
        topology: str | Topology = "auto",
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._client = BodegaInferenceClient(base_url=base_url, timeout=timeout)
        self._topology: Topology = (
            topology if isinstance(topology, Topology) else get_topology(topology)
        )
        logger.debug("bodega_router_init", topology=self._topology.name)

    # ── Topology helpers ──────────────────────────────────────────────────────

    def resolve_config(self, tier: ModelTier) -> ModelConfig:
        """Return the full :class:`ModelConfig` for *tier* under the active topology.

        Use this when you need more than just the model_id — e.g. to inspect
        ``max_concurrency``, ``prompt_cache_size``, or ``draft_model_path``
        that were used when loading the model.
        """
        return self._topology.resolve_config(tier)

    def resolve_model_id(self, tier: ModelTier) -> str:
        """Return the Bodega model_id for *tier* under the active topology."""
        return self._topology.resolve(tier)

    @property
    def topology(self) -> Topology:
        """The active :class:`Topology`."""
        return self._topology

    # ── Inference ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier = ModelTier.REASON,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Routed chat completion.

        Resolves the tier to a model_id and passes it in the Bodega request
        so the engine routes to the correct queued model.

        Args:
            messages:    OpenAI-format message list.
            tier:        ModelTier for this call.  Defaults to REASON.
            temperature: Sampling temperature.
            max_tokens:  Max output tokens.

        Returns:
            Full Bodega API response dict (OpenAI-compatible).
        """
        model_id = self.resolve_model_id(tier)
        logger.debug("router_chat", tier=tier.value, model_id=model_id)
        return await self._client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_id,
        )

    async def chat_simple(
        self,
        prompt: str,
        system: str = "",
        tier: ModelTier = ModelTier.REASON,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Routed convenience chat — sends *prompt*, returns plain text.

        Args:
            prompt:      User prompt string.
            system:      Optional system message.
            tier:        ModelTier for this call.
            temperature: Sampling temperature.
            max_tokens:  Max output tokens.

        Returns:
            Generated text string (content only, no metadata).
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = await self.chat(
            messages=messages,
            tier=tier,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        prompt: str,
        system: str = "",
        tier: ModelTier = ModelTier.REASON,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        total_timeout: float = 90.0,
    ) -> AsyncIterator[str]:
        """Routed streaming chat — yields text chunks as they arrive.

        Uses SSE (OpenAI-compatible).  Each yielded value is a raw text
        fragment — callers print / accumulate immediately.

        Uses a no-keepalive httpx client per stream to avoid drain blocking
        on cancellation (same pattern as BodegaInferenceClient.chat_stream).

        Args:
            prompt:        User prompt.
            system:        Optional system message.
            tier:          ModelTier to route to.
            temperature:   Sampling temperature.
            max_tokens:    Max tokens to generate.
            total_timeout: Read timeout between SSE chunks.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model_id = self.resolve_model_id(tier)
        logger.debug("router_chat_stream", tier=tier.value, model_id=model_id)

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(
            base_url=self._client.base_url,
            timeout=httpx.Timeout(
                connect=5.0, read=total_timeout, write=10.0, pool=5.0
            ),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        ) as stream_client:
            async with stream_client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ── Health & info — delegate through ─────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Delegate to inner client health check."""
        return await self._client.health()

    async def current_model(self) -> dict[str, Any]:
        """Delegate to inner client current_model."""
        return await self._client.current_model()

    async def list_models(self) -> dict[str, Any]:
        """Delegate to inner client list_models."""
        return await self._client.list_models()

    async def queue_stats(self) -> dict[str, Any]:
        """Delegate to inner client queue_stats."""
        return await self._client.queue_stats()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
