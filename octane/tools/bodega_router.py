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

Tier semantics (power topology — 32 GB+ RAM):
    FAST   — bodega-raptor-90M  (lm)       — routing, extraction, classification
    MID    — bodega-vertex-4b   (lm)       — chunk summarization (falls back to 90M)
    REASON — axe-stealth-37b   (multimodal) — final-answer synthesis, deep analysis
    EMBED  — handled in-process; not routed through Bodega

Key behaviours:
  • Tier-aware fallback: REASON prefers multimodal (large) > lm (small).
    FAST/MID prefer lm (small 90M) over multimodal (large 37b).
  • Auto-load: if the preferred model is not loaded Bodega is asked to load
    it on demand.  This is transparent to all callers.
  • wait_for_server(): blocks with live terminal feedback until Bodega is
    reachable.  Called from pre_flight() when the engine is offline.
  • gather_completions(): fires N chat requests concurrently via
    asyncio.gather, routing each to the correct model tier.  Both the 90M
    and the 37B can receive concurrent requests and handle them in parallel
    thanks to their CB-enabled max_concurrency settings.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from octane.tools.bodega_inference import BodegaInferenceClient
from octane.tools.topology import ModelConfig, ModelTier, Topology, get_topology

logger = structlog.get_logger().bind(component="bodega_router")


# ── Loaded model info ─────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    """Runtime info about a model loaded in Bodega."""
    model_id: str
    model_type: str          # "lm" or "multimodal"
    context_length: int      # max context window
    status: str = "running"
    memory_mb: float = 0.0   # total memory usage


# How long to cache the loaded-models list before re-querying.
_MODEL_CACHE_TTL_S = 60.0


class BodegaRouter:
    """Tier-aware routing wrapper around :class:`BodegaInferenceClient`.

    Dynamically queries Bodega for loaded models and falls back gracefully
    if the preferred tier model isn't available.  Can auto-load models on
    demand and waits for the Bodega server to come online if it is offline.

    Model discovery uses ``/v1/admin/loaded-models`` with a TTL cache
    (default 60 s) so that runtime model swaps (load/unload via external
    scripts) are detected without manual intervention.

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
        timeout: float = 600.0,
    ) -> None:
        self._client = BodegaInferenceClient(base_url=base_url, timeout=timeout)
        self._topology: Topology = (
            topology if isinstance(topology, Topology) else get_topology(topology)
        )
        # Cache — {model_id: LoadedModel}.  Invalidated on TTL expiry or manual call.
        self._loaded_models: dict[str, LoadedModel] | None = None
        self._loaded_models_ts: float = 0.0  # monotonic timestamp of last fetch
        logger.debug("bodega_router_init", topology=self._topology.name)

    # ── Server health & wait ──────────────────────────────────────────────────

    async def wait_for_server(
        self,
        poll_interval: float = 3.0,
        max_attempts: int | None = None,
    ) -> None:
        """Block until Bodega is reachable, printing live status to stderr.

        Called from pre_flight() when the initial health check fails.  The
        user sees a clear message to start their inference engine and Octane
        keeps polling until it comes up — no action required beyond starting
        the engine.

        Args:
            poll_interval: Seconds between health check attempts.
            max_attempts:  Stop after this many attempts (None = infinite).
        """
        attempt = 0
        print(
            "\n[Octane] Bodega inference engine not reachable at "
            f"{self._client.base_url}\n"
            "         Start the engine and Octane will pick it up automatically.\n",
            file=sys.stderr,
            flush=True,
        )
        while True:
            attempt += 1
            if max_attempts is not None and attempt > max_attempts:
                return
            try:
                health = await self._client.health()
                if health.get("status") == "ok":
                    print(
                        f"[Octane] Bodega is online (attempt {attempt}). "
                        f"Models: {health.get('model_id', 'none')}\n",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._loaded_models = None  # force fresh discovery
                    return
            except Exception:
                pass
            print(
                f"[Octane] Waiting for Bodega... (attempt {attempt})"
                f"  Retry in {poll_interval:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(poll_interval)

    # ── Dynamic model discovery ───────────────────────────────────────────────

    def _cache_is_valid(self) -> bool:
        """Return True if cached model list is still within TTL."""
        if self._loaded_models is None:
            return False
        return (time.monotonic() - self._loaded_models_ts) < _MODEL_CACHE_TTL_S

    async def _fetch_loaded_models(self, *, wait_if_down: bool = False) -> dict[str, LoadedModel]:
        """Fetch currently loaded models from ``/v1/admin/loaded-models``.

        Uses a TTL cache (default 60 s) so runtime model swaps are detected
        automatically without manual invalidation.

        Args:
            wait_if_down: If True and Bodega is unreachable, call
                          wait_for_server() before giving up.
        """
        if self._cache_is_valid():
            return self._loaded_models  # type: ignore[return-value]

        async def _do_fetch() -> dict[str, LoadedModel]:
            client = await self._client._get_client()
            response = await client.get("/v1/admin/loaded-models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            result: dict[str, LoadedModel] = {}
            for m in models:
                if m.get("status") != "running":
                    continue
                mem = m.get("memory", {})
                result[m["id"]] = LoadedModel(
                    model_id=m["id"],
                    model_type=m.get("type", "lm"),
                    context_length=m.get("context_length", 4096),
                    status=m.get("status", "running"),
                    memory_mb=mem.get("total_mb", 0.0),
                )
            return result

        try:
            self._loaded_models = await _do_fetch()
            self._loaded_models_ts = time.monotonic()
            logger.debug(
                "bodega_models_discovered",
                models=[
                    {"id": m.model_id, "type": m.model_type, "ctx": m.context_length}
                    for m in self._loaded_models.values()
                ],
            )
            return self._loaded_models
        except Exception as exc:
            if wait_if_down:
                await self.wait_for_server()
                try:
                    self._loaded_models = await _do_fetch()
                    self._loaded_models_ts = time.monotonic()
                    return self._loaded_models
                except Exception:
                    pass
            # Fall back to /health models_detail as last resort
            try:
                resp = await self._client.health()
                if resp.get("status") == "ok":
                    models_detail = resp.get("models_detail", [])
                    self._loaded_models = {
                        m["id"]: LoadedModel(
                            model_id=m["id"],
                            model_type=m.get("type", "lm"),
                            context_length=m.get("context_length", 4096),
                        )
                        for m in models_detail
                        if m.get("status") == "running"
                    }
                    self._loaded_models_ts = time.monotonic()
                    return self._loaded_models
            except Exception:
                pass
            logger.warning("bodega_model_discovery_failed", error=str(exc))
            return {}

    def invalidate_model_cache(self) -> None:
        """Clear cached model list — call after load/unload operations."""
        self._loaded_models = None
        self._loaded_models_ts = 0.0

    # ── Auto-load ─────────────────────────────────────────────────────────────

    async def _auto_load_tier(self, tier: ModelTier) -> bool:
        """Load the topology-configured model for *tier* via Bodega admin API.

        Returns True if the load succeeded, False otherwise.
        """
        try:
            config = self._topology.resolve_config(tier)
            params = config.to_load_params()
            logger.info(
                "bodega_auto_loading_model",
                tier=tier.value,
                model_id=config.model_id,
                model_path=config.model_path,
            )
            print(
                f"[Octane] Auto-loading {config.model_id} ({config.model_path}) "
                f"for {tier.value} tier...",
                file=sys.stderr,
                flush=True,
            )
            client = await self._client._get_client()
            resp = await client.post("/v1/admin/load-model", json=params)
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") == "loaded":
                self.invalidate_model_cache()
                print(
                    f"[Octane] {config.model_id} loaded successfully.",
                    file=sys.stderr,
                    flush=True,
                )
                return True
            logger.warning("bodega_auto_load_unexpected", result=result)
            return False
        except Exception as exc:
            logger.warning("bodega_auto_load_failed", tier=tier.value, error=str(exc))
            return False

    # ── Tier-aware model resolution ───────────────────────────────────────────

    async def _resolve_available_model(
        self,
        tier: ModelTier,
        *,
        auto_load: bool = True,
    ) -> str | None:
        """Resolve a tier to an actually-loaded model ID, or None.

        Matching strategy (tried in order):
          1. Direct match   — topology model_id == loaded model ID
          2. Path match     — topology model_path.lower() == loaded ID.lower()
                             handles Bodega registering models by full HF path
          3. Tier-aware fallback:
               ALL tiers  → prefer lm over multimodal (multimodal CB not yet
                            supported by Bodega; avoids routing to vision models
                            for text-only synthesis).
          4. Auto-load — load the topology model from Bodega admin API if a
             type-based fallback was not found.  Skipped for REASON when any
             lm model is already available (prevents downloading a large model
             when a capable lm is ready).
        """
        loaded = await self._fetch_loaded_models()

        if not loaded:
            if auto_load:
                loaded_ok = await self._auto_load_tier(tier)
                if loaded_ok:
                    loaded = await self._fetch_loaded_models()
            if not loaded:
                return None

        preferred_cfg = self._topology.resolve_config(tier)
        preferred_id   = preferred_cfg.model_id
        preferred_path = preferred_cfg.model_path

        # 1. Direct match
        if preferred_id in loaded:
            return preferred_id

        # 2. Path-based match (case-insensitive)
        preferred_path_lower = preferred_path.lower()
        for loaded_id in loaded:
            if loaded_id.lower() == preferred_path_lower:
                logger.debug(
                    "router_path_match",
                    tier=tier.value,
                    preferred=preferred_id,
                    matched=loaded_id,
                )
                return loaded_id

        # 3. Tier-aware fallback (no exact or path match)
        # All tiers prefer lm over multimodal — Bodega continuous batching for
        # multimodal is not yet supported, and lm models handle text synthesis
        # (the primary use-case for REASON) equally well.
        if tier == ModelTier.REASON:
            for model_type_pref in ("lm", "multimodal"):
                for loaded_id, info in loaded.items():
                    if info.model_type == model_type_pref:
                        logger.info(
                            "bodega_tier_fallback",
                            tier=tier.value,
                            preferred=preferred_id,
                            using=loaded_id,
                            ctx=info.context_length,
                            reason=f"type={model_type_pref} fallback",
                        )
                        return loaded_id
        else:
            # FAST / MID: prefer speed — lm (90M) first, then multimodal.
            #
            # Exception: for MID, if the topology-configured model is distinct
            # from what's loaded (e.g. 8b configured but only 90m loaded), try
            # auto-loading it before falling back to the 90m.  FAST always falls
            # back immediately because its topology model IS the 90m.
            if tier == ModelTier.MID and auto_load:
                loaded_ok = await self._auto_load_tier(tier)
                if loaded_ok:
                    loaded = await self._fetch_loaded_models()
                    if preferred_id in loaded:
                        return preferred_id
                    for loaded_id in loaded:
                        if loaded_id.lower() == preferred_path_lower:
                            return loaded_id
                # auto-load failed or model still not found — fall through to
                # type-based fallback so the query still completes
                logger.warning(
                    "bodega_mid_autoload_failed_falling_back",
                    preferred=preferred_id,
                )

            for model_type_pref in ("lm", "multimodal"):
                for loaded_id, info in loaded.items():
                    if info.model_type == model_type_pref:
                        logger.info(
                            "bodega_tier_fallback",
                            tier=tier.value,
                            preferred=preferred_id,
                            using=loaded_id,
                            ctx=info.context_length,
                            reason=f"type={model_type_pref} fallback",
                        )
                        return loaded_id

        # 4. Auto-load: model not found and no type-based fallback found.
        #    Skip auto-load for REASON if ANY lm model is loaded — we never want
        #    to trigger a large model download when a capable lm is available.
        if tier == ModelTier.REASON and any(
            i.model_type == "lm" for i in loaded.values()
        ):
            return None
        if auto_load and tier != ModelTier.MID:
            loaded_ok = await self._auto_load_tier(tier)
            if loaded_ok:
                # Retry steps 1 & 2 with fresh model list
                loaded = await self._fetch_loaded_models()
                if preferred_id in loaded:
                    return preferred_id
                for loaded_id in loaded:
                    if loaded_id.lower() == preferred_path_lower:
                        return loaded_id

        return None

    # ── Chat premium mode ─────────────────────────────────────────────────────

    async def prepare_for_chat(
        self,
        *,
        on_status: Any | None = None,
    ) -> dict[str, Any]:
        """Prepare the inference engine for chat premium mode.

        Unloads all models except REASON, then ensures the REASON-tier
        model is loaded.  Returns a status dict with details.

        Args:
            on_status: Optional callable(msg: str) for progress messages.
        """
        def _emit(msg: str) -> None:
            if on_status:
                on_status(msg)

        result: dict[str, Any] = {
            "unloaded": [],
            "loaded_model": None,
            "already_ready": False,
        }

        reason_cfg = self._topology.resolve_config(ModelTier.REASON)
        reason_id = reason_cfg.model_id
        reason_path = reason_cfg.model_path

        loaded = await self._fetch_loaded_models()
        if not loaded:
            _emit("No models currently loaded")

        # Check if REASON is already the only model loaded
        loaded_ids = set(loaded.keys())
        reason_already = reason_id in loaded_ids or any(
            lid.lower() == reason_path.lower() for lid in loaded_ids
        )

        non_reason = [
            mid for mid in loaded_ids
            if mid != reason_id and mid.lower() != reason_path.lower()
        ]

        if reason_already and not non_reason:
            result["already_ready"] = True
            result["loaded_model"] = reason_id
            _emit(f"{reason_id} already loaded exclusively")
            return result

        # Unload all non-REASON models
        for model_id in non_reason:
            _emit(f"Unloading {model_id}...")
            try:
                await self._client.unload_model_by_id(model_id)
                result["unloaded"].append(model_id)
                logger.info("chat_unloaded_model", model_id=model_id)
            except Exception as exc:
                logger.warning("chat_unload_failed", model_id=model_id, error=str(exc))

        self.invalidate_model_cache()

        # Load REASON if not already loaded
        if not reason_already:
            _emit(f"Loading {reason_id}...")
            try:
                params = reason_cfg.to_load_params()
                client = await self._client._get_client()
                resp = await client.post("/v1/admin/load-model", json=params)
                resp.raise_for_status()
                load_result = resp.json()
                if load_result.get("status") == "loaded":
                    self.invalidate_model_cache()
                    result["loaded_model"] = reason_id
                    _emit(f"{reason_id} loaded")
                else:
                    logger.warning("chat_reason_load_unexpected", result=load_result)
            except Exception as exc:
                logger.warning("chat_reason_load_failed", error=str(exc))
        else:
            result["loaded_model"] = reason_id

        return result

    # ── Topology helpers ──────────────────────────────────────────────────────

    def resolve_config(self, tier: ModelTier) -> ModelConfig:
        """Return the full :class:`ModelConfig` for *tier* under the active topology."""
        return self._topology.resolve_config(tier)

    def resolve_model_id(self, tier: ModelTier) -> str:
        """Return the Bodega model_id for *tier* under the active topology."""
        return self._topology.resolve(tier)

    @property
    def topology(self) -> Topology:
        """The active :class:`Topology`."""
        return self._topology

    # ── Model introspection ───────────────────────────────────────────────────

    async def get_model_context_length(self, model_id: str) -> int:
        """Return the context_length for a loaded model, or 4096 as fallback."""
        loaded = await self._fetch_loaded_models()
        info = loaded.get(model_id)
        return info.context_length if info else 4096

    async def loaded_models_summary(self) -> list[dict[str, Any]]:
        """Return a summary of all currently loaded models (for CLI / status)."""
        loaded = await self._fetch_loaded_models()
        return [
            {
                "id": info.model_id,
                "type": info.model_type,
                "context_length": info.context_length,
                "memory_mb": info.memory_mb,
            }
            for info in loaded.values()
        ]

    # ── Inference ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier = ModelTier.REASON,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Routed chat completion.

        Dynamically resolves the tier to an actually-loaded model.
        Falls back to any available model if the preferred one isn't loaded.
        Auto-loads the model if it isn't present.

        If the resolved model has a smaller context window than the request
        needs, ``max_tokens`` is clamped to fit.

        Args:
            messages:    OpenAI-format message list.
            tier:        ModelTier for this call.  Defaults to REASON.
            temperature: Sampling temperature.
            max_tokens:  Max output tokens.

        Returns:
            Full Bodega API response dict (OpenAI-compatible).

        Raises:
            RuntimeError: If no models are loaded or Bodega is unreachable.
        """
        model_id = await self._resolve_available_model(tier)

        if model_id is None:
            raise RuntimeError(
                f"No model available for tier {tier.value}. "
                "Ensure Bodega is running: curl http://localhost:44468/health"
            )

        # Clamp max_tokens to model's context window (leave room for input)
        ctx = await self.get_model_context_length(model_id)
        input_estimate = sum(len(m.get("content", "")) // 4 for m in messages)
        available_tokens = max(ctx - input_estimate - 128, 256)
        if max_tokens > available_tokens:
            logger.debug(
                "router_clamp_max_tokens",
                model=model_id,
                ctx=ctx,
                input_est=input_estimate,
                requested=max_tokens,
                clamped=available_tokens,
            )
            max_tokens = available_tokens

        logger.debug("router_chat", tier=tier.value, model_id=model_id)
        try:
            return await self._client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model_id,
            )
        except (httpx.HTTPStatusError,) as exc:
            # Model may have been unloaded between resolve and request.
            # Invalidate cache and retry once with fresh discovery.
            if getattr(exc, 'response', None) is not None and exc.response.status_code in (404, 400):
                logger.info("router_model_gone_retrying", model=model_id, tier=tier.value)
                self.invalidate_model_cache()
                retry_model = await self._resolve_available_model(tier)
                if retry_model and retry_model != model_id:
                    return await self._client.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model=retry_model,
                    )
            raise

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

        Dynamically resolves the tier to an actually-loaded model.
        Falls back to any available model if the preferred one isn't loaded.

        Args:
            prompt:        User prompt.
            system:        Optional system message.
            tier:          ModelTier to route to.
            temperature:   Sampling temperature.
            max_tokens:    Max tokens to generate.
            total_timeout: Read timeout between SSE chunks.

        Raises:
            RuntimeError: If no models are loaded or Bodega is unreachable.
        """
        model_id = await self._resolve_available_model(tier)

        if model_id is None:
            raise RuntimeError(
                f"No model available for tier {tier.value}. "
                "Ensure Bodega is running: curl http://localhost:44468/health"
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Clamp max_tokens to model context window
        ctx = await self.get_model_context_length(model_id)
        input_estimate = sum(len(m.get("content", "")) // 4 for m in messages)
        available_tokens = max(ctx - input_estimate - 128, 256)
        if max_tokens > available_tokens:
            max_tokens = available_tokens

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

    # ── Concurrent batch inference ────────────────────────────────────────────

    async def gather_completions(
        self,
        requests: list[dict[str, Any]],
    ) -> list[str | BaseException]:
        """Fire N chat requests concurrently, routing each to the right model.

        Both the 90M (FAST) and 37B (REASON) can receive concurrent requests
        because they are loaded with max_concurrency>1 and continuous batching
        enabled.  This method lets the caller saturate both models in parallel.

        Args:
            requests: List of request dicts, each with keys:
                        ``prompt``   (str, required)
                        ``tier``     (ModelTier, default REASON)
                        ``system``   (str, default "")
                        ``temperature`` (float, default 0.7)
                        ``max_tokens``  (int, default 1024)

        Returns:
            List of results in the same order as *requests*.
            Each entry is either the generated text (str) or an Exception
            if that individual request failed (other requests still succeed).

        Example::

            results = await router.gather_completions([
                {"prompt": "Summarise NVDA earnings", "tier": ModelTier.FAST},
                {"prompt": "Write a full thesis on NVDA vs AMD", "tier": ModelTier.REASON},
                {"prompt": "What sector is MSFT in?", "tier": ModelTier.FAST},
            ])
        """
        tasks = [
            self.chat_simple(
                prompt=req["prompt"],
                system=req.get("system", ""),
                tier=req.get("tier", ModelTier.REASON),
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 1024),
            )
            for req in requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, (req, res) in enumerate(zip(requests, results)):
            if isinstance(res, BaseException):
                logger.warning(
                    "gather_completion_failed",
                    index=i,
                    tier=req.get("tier", ModelTier.REASON),
                    error=str(res),
                )
        return list(results)

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
