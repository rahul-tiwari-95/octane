"""Async client for Bodega Inference Engine (localhost:44468).

OpenAI-compatible API. Used by agents for LLM inference.
Model loading/unloading is done ONLY by SysStat.ModelManager.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from octane.config import settings

logger = structlog.get_logger().bind(component="bodega_inference")


class BodegaInferenceClient:
    """Async client for the local Bodega Inference Engine.

    Provides chat completion, health checks, and model info.
    Admin endpoints (load/unload) are exposed but should ONLY be called
    by SysStat.ModelManager.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or settings.bodega_inference_url).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ---- Inference ----

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: str = "current",
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format messages [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            model: Model identifier (usually "current" for whatever is loaded)

        Returns:
            Full API response dict
        """
        client = await self._get_client()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        # If the model was loaded with reasoning_parser="qwen3", Bodega extracts
        # the <think> block into message["reasoning_content"] automatically.
        # Log it at DEBUG and strip it so callers always receive clean content.
        message = result.get("choices", [{}])[0].get("message", {})
        if reasoning := message.get("reasoning_content"):
            logger.debug(
                "model_reasoning_content",
                trace=reasoning[:400],
                model=model,
            )

        logger.debug(
            "chat_completion",
            model=model,
            messages_count=len(messages),
            usage=result.get("usage"),
        )
        return result

    async def chat_simple(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Convenience: send a simple prompt, get back just the text."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return result["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        total_timeout: float = 90.0,
    ) -> AsyncIterator[str]:
        """Streaming chat: yields text chunks as they arrive from Bodega.

        Uses the OpenAI-compatible SSE format (data: {...} lines).
        Each yielded value is a raw text fragment — callers print them
        immediately to give the user a real-time typing effect.

        Args:
            total_timeout: Passed to httpx as the read-timeout between SSE
                lines.  NOTE: callers (e.g. Orchestrator.run_stream) apply a
                per-__anext__() asyncio.wait_for() guard as the primary
                wall-clock cap, because asyncio.timeout() inside a suspended
                async generator is unreliable on Python 3.13.

        Usage:
            async for chunk in bodega.chat_stream(prompt, system=sys):
                print(chunk, end="", flush=True)
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "current",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        # Use a fresh, no-keepalive client for each stream request.
        # The shared pool client forces httpcore to DRAIN the remaining response
        # body before returning the connection to the pool.  If Bodega is still
        # generating when the caller cancels (e.g. via asyncio.wait_for on
        # __anext__()), that drain blocks indefinitely — preventing the
        # cancelled task from finishing and therefore blocking wait_for itself.
        # A no-keepalive client closes the TCP socket immediately on cleanup.
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=5.0, read=total_timeout, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        ) as stream_client:
            async with stream_client.stream("POST", "/v1/chat/completions", json=payload) as response:
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

    # ---- Health & Info ----

    async def health(self) -> dict[str, Any]:
        """Check server health."""
        client = await self._get_client()
        try:
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e)}

    async def current_model(self) -> dict[str, Any]:
        """Get the currently loaded model info."""
        client = await self._get_client()
        try:
            response = await client.get("/v1/admin/current-model")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}

    async def list_models(self) -> dict[str, Any]:
        """List all cached/available models."""
        client = await self._get_client()
        try:
            response = await client.get("/v1/models")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}

    async def queue_stats(self) -> dict[str, Any]:
        """Get queue performance metrics."""
        client = await self._get_client()
        try:
            response = await client.get("/v1/queue/stats")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}

    # ---- Admin (SysStat.ModelManager ONLY) ----

    async def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        """Load a model. ONLY called by SysStat.ModelManager.

        Args:
            model_path: HuggingFace model path or local path
            **kwargs: lora_paths, lora_scales, tool_call_parser, etc.
        """
        client = await self._get_client()
        payload = {"model_path": model_path, **kwargs}
        response = await client.post("/v1/admin/load-model", json=payload)
        response.raise_for_status()
        return response.json()

    async def unload_model(self) -> dict[str, Any]:
        """Unload the current model. ONLY called by SysStat.ModelManager."""
        client = await self._get_client()
        response = await client.post("/v1/admin/unload-model")
        response.raise_for_status()
        return response.json()
