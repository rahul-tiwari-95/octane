"""Async client for Bodega Inference Engine (localhost:44468).

OpenAI-compatible API. Used by agents for LLM inference.
Model loading/unloading is done ONLY by SysStat.ModelManager.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from octane.config import settings

logger = structlog.get_logger().bind(component="bodega_inference")

# Pre-compiled patterns for stripping think-block content.
# Reasoning models (axe-stealth-37b, qwen3, etc.) emit <think>...</think>
# inline in the content field when loaded WITHOUT reasoning_parser.
# Stripping here keeps ALL callers clean — dimension_planner, evaluator,
# synthesizer, decomposer, etc.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
# Unclosed block — model was cut off mid-think; strip from opening tag to end.
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str, log_trace: bool = True) -> str:
    """Remove <think>...</think> blocks from a model response.

    When a reasoning model is loaded WITHOUT reasoning_parser, Bodega
    returns the full thinking chain inside the content field.  This
    function strips it before any caller ever sees it, so no component
    needs to handle think-blocks individually.

    If log_trace=True (default) the first 400 chars of the stripped
    thinking chain are logged at DEBUG for observability.
    """
    # Capture the think block for debug logging before removing it.
    think_match = _THINK_RE.search(text) or _THINKING_RE.search(text)
    if think_match and log_trace:
        matched = think_match.group(0)
        # Extract inner content between the outer tags robustly.
        open_end = matched.find(">")
        close_start = matched.rfind("<")
        inner = matched[open_end + 1 : close_start].strip() if open_end >= 0 else matched
        logger.debug("model_think_stripped", trace=inner[:400])

    text = _THINK_RE.sub("", text)
    text = _THINKING_RE.sub("", text)
    # Unclosed opening tag (generation cut off mid-think)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


class BodegaInferenceClient:
    """Async client for the local Bodega Inference Engine.

    Provides chat completion, health checks, and model info.
    Admin endpoints (load/unload) are exposed but should ONLY be called
    by SysStat.ModelManager.

    When the Octane daemon is running, inference requests are routed
    through the daemon's InferenceProxy (per-model semaphore backpressure).
    When the daemon is not running, requests go directly to Bodega HTTP.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = (base_url or settings.bodega_inference_url).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # Daemon routing: cached check to avoid stat() on every call.
        self._daemon_checked: bool = False
        self._daemon_available: bool = False

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

    def _check_daemon(self) -> bool:
        """Check if daemon is running (cached per-client lifetime).

        Returns False if we ARE the daemon process — prevents recursive
        socket connections when the daemon's internal OSA pipeline creates
        its own BodegaInferenceClient instances.
        """
        if not self._daemon_checked:
            try:
                from octane.daemon.client import is_daemon_running, get_pid_path
                # If our PID matches the daemon PID, we ARE the daemon.
                # Skip the IPC route — go directly to Bodega HTTP.
                pid_path = get_pid_path()
                if pid_path.exists():
                    try:
                        daemon_pid = int(pid_path.read_text().strip())
                        if daemon_pid == os.getpid():
                            self._daemon_available = False
                            self._daemon_checked = True
                            return False
                    except (ValueError, OSError):
                        pass
                self._daemon_available = is_daemon_running()
            except Exception:
                self._daemon_available = False
            self._daemon_checked = True
        return self._daemon_available

    async def _daemon_infer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any] | None:
        """Try to route an inference request through the daemon.

        Returns the Bodega response dict if daemon handled it, None if
        daemon is not available (caller should fall back to direct HTTP).

        The socket timeout is set to (self.timeout + 30s) so the daemon
        always has time to complete the Bodega HTTP call before the
        client gives up waiting on the socket.

        Benefits of daemon routing:
            - Per-model semaphore backpressure (5 terminals don't swamp Bodega)
            - Priority scheduling (interactive P0 ahead of background P3)
            - Metrics: active/waiting/avg_wait_ms per model
            - Model lifecycle coordination (idle unload, ensure_loaded)
        """
        if not self._check_daemon():
            return None

        try:
            from octane.daemon.client import DaemonClient

            client = DaemonClient()
            if not await client.connect(timeout=5.0):
                return None

            try:
                result = await client.request(
                    "infer",
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": self.timeout,
                    },
                    timeout=self.timeout + 30.0,
                )
            finally:
                await client.close()

            if result.get("status") == "ok":
                return result.get("data", {})

            # Daemon returned an error — log and fall through to direct HTTP.
            error = result.get("error", "unknown daemon error")
            logger.warning("daemon_infer_fallback", error=error, model=model)
            return None

        except Exception as exc:
            logger.debug("daemon_infer_unavailable", error=str(exc))
            return None

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
        # Try daemon route first — backpressure-aware gateway.
        daemon_result = await self._daemon_infer(messages, model, temperature, max_tokens)
        if daemon_result is not None:
            return daemon_result

        # Direct HTTP fall-through (daemon not running or unavailable).
        client = await self._get_client()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Disable chain-of-thought for all Qwen3-family models.
            # Has no effect on models without thinking support.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        # If the model was loaded with reasoning_parser="qwen3", Bodega extracts
        # the <think> block into message["reasoning_content"] automatically.
        # Log it at DEBUG.
        message = result.get("choices", [{}])[0].get("message", {})
        if reasoning := message.get("reasoning_content"):
            logger.debug(
                "model_reasoning_content",
                trace=reasoning[:400],
                model=model,
            )
        elif message.get("content") and "<think>" in message["content"]:
            # Model loaded WITHOUT reasoning_parser: think block is in content.
            # Strip it here so every caller gets clean text.
            message["content"] = _strip_think_tags(message["content"])

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

        content = result["choices"][0]["message"]["content"]
        # Belt-and-suspenders: strip any residual think tags that weren't
        # caught in chat() (e.g. if caller bypasses chat() directly).
        return _strip_think_tags(content, log_trace=False) if "<think>" in content else content

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

                # Stateful think-block filter.
                # Reasoning models emit <think>...</think> as raw token stream.
                # We suppress all tokens from <think> to </think> inclusive so
                # the caller's terminal never shows the thinking chain.
                _in_think = False
                _think_buf: list[str] = []   # accumulates partial </think> tag

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
                        if not text:
                            continue

                        # ── think-block suppression ────────────────────────
                        if _in_think:
                            _think_buf.append(text)
                            combined = "".join(_think_buf)
                            close_idx = combined.lower().find("</think>")
                            if close_idx >= 0:
                                _in_think = False
                                _think_buf = []
                                remainder = combined[close_idx + 8:]
                                # Strip any leading whitespace/newline after </think>
                                remainder = remainder.lstrip("\n ")
                                if remainder:
                                    yield remainder
                            continue

                        if "<think>" in text.lower():
                            # Think block starting mid-chunk
                            before, _, after = text.lower().partition("<think>")
                            # Emit content before <think>
                            pre = text[: len(before)]
                            if pre:
                                yield pre
                            # Check if </think> is in the same chunk
                            rest = text[len(before) + 7:]
                            close_idx = rest.lower().find("</think>")
                            if close_idx >= 0:
                                remainder = rest[close_idx + 8:].lstrip("\n ")
                                if remainder:
                                    yield remainder
                            else:
                                _in_think = True
                                _think_buf = [rest]
                            continue
                        # ── end think-block suppression ────────────────────

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
        """Get info about the currently loaded model(s).

        Uses /v1/admin/loaded-models (the multi-model registry endpoint).
        Normalizes to a stable shape so all callers work unchanged:

            {"loaded": True,  "model_path": "SRSWTI/...", "context_length": 32768,
             "total_loaded": 1, "all_models": ["id"], "model_info": {...}}

            {"loaded": False, "model_info": {}}           # no models
            {"error": "..."}                              # HTTP / network error
        """
        client = await self._get_client()
        try:
            response = await client.get("/v1/admin/loaded-models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            if not models:
                return {"loaded": False, "model_info": {}}
            # Use the first entry as the "primary" model
            first = models[0]
            model_id = first.get("id", "unknown")
            ctx = first.get("context_length")
            model_info = {
                "model_path": model_id,
                "model_id": model_id,
                "model_type": first.get("model_type", "lm"),
                "context_length": ctx,
                "reasoning_parser": first.get("reasoning_parser"),
                "status": first.get("status", "unknown"),
            }
            return {
                "loaded": True,
                "model_path": model_id,        # top-level for display
                "context_length": ctx,         # top-level for display
                "total_loaded": len(models),
                "all_models": [m.get("id") for m in models],
                "model_info": model_info,      # nested for ModelManager compat
            }
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
