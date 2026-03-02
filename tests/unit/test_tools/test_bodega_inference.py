"""Cancellation and fresh-client tests for BodegaInferenceClient.

These tests validate the Bug 9 fix: chat_stream() uses a fresh no-keepalive
httpx.AsyncClient per stream, ensuring that cancellation of the caller
(via asyncio.wait_for or task.cancel()) does not block in connection-pool drain.

All tests are pure unit tests — no real HTTP connections made.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. chat_stream uses fresh no-keepalive client
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_creates_new_client_per_call():
    """Each chat_stream() call should create a fresh httpx.AsyncClient."""
    from octane.tools.bodega_inference import BodegaInferenceClient

    client = BodegaInferenceClient(base_url="http://localhost:44468")
    client_instances: list = []

    original_cls = None

    class TrackingAsyncClient:
        """Records construction args and mocks out the HTTP layer."""

        def __init__(self, **kwargs):
            client_instances.append(kwargs)
            self._limits = kwargs.get("limits")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, **kwargs):
            return _FakeStreamContext()

    class _FakeStreamContext:
        async def __aenter__(self):
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}

            async def fake_aiter_lines():
                yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
                yield "data: [DONE]"

            mock_resp.aiter_lines = fake_aiter_lines
            return mock_resp

        async def __aexit__(self, *args):
            pass

    import httpx

    with patch("httpx.AsyncClient", TrackingAsyncClient):
        # Call chat_stream twice
        chunks_1 = []
        async for chunk in client.chat_stream("prompt 1"):
            chunks_1.append(chunk)

        chunks_2 = []
        async for chunk in client.chat_stream("prompt 2"):
            chunks_2.append(chunk)

    # Each call should have created a new client instance
    assert len(client_instances) == 2, (
        f"Expected 2 fresh clients (one per stream call), got {len(client_instances)}"
    )


@pytest.mark.asyncio
async def test_chat_stream_client_has_no_keepalive():
    """The fresh client must be created with max_keepalive_connections=0."""
    import httpx
    from octane.tools.bodega_inference import BodegaInferenceClient

    client = BodegaInferenceClient(base_url="http://localhost:44468")
    captured_limits: list = []

    class CapturingClient:
        def __init__(self, **kwargs):
            limits = kwargs.get("limits")
            if limits:
                captured_limits.append(limits)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, **kwargs):
            return _NullStreamContext()

    class _NullStreamContext:
        async def __aenter__(self):
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}

            async def fake_aiter_lines():
                yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
                yield "data: [DONE]"

            mock_resp.aiter_lines = fake_aiter_lines
            return mock_resp

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient", CapturingClient):
        async for _ in client.chat_stream("test prompt"):
            pass

    assert captured_limits, "httpx.AsyncClient was not constructed with a limits argument"
    limits = captured_limits[0]
    assert limits.max_keepalive_connections == 0, (
        f"Expected max_keepalive_connections=0, got {limits.max_keepalive_connections}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cancellation of chat_stream completes promptly
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_cancellation_does_not_hang():
    """Cancelling a task that is consuming chat_stream should complete within 1 s."""
    import httpx
    from octane.tools.bodega_inference import BodegaInferenceClient

    client = BodegaInferenceClient(base_url="http://localhost:44468")

    class SlowStreamContext:
        """Simulates a stream that yields very slowly (like a still-generating model)."""

        def __init__(self, **kwargs):
            pass  # accepts base_url, timeout, limits kwargs from BodegaInferenceClient

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, **kwargs):
            return _SlowBody()

    class _SlowBody:
        async def __aenter__(self):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}

            async def slow_aiter_lines():
                yield 'data: {"choices":[{"delta":{"content":"first chunk"}}]}'
                await asyncio.sleep(9999)   # hangs here
                yield 'data: {"choices":[{"delta":{"content":"second chunk"}}]}'
                yield "data: [DONE]"

            mock_resp.aiter_lines = slow_aiter_lines
            return mock_resp

        async def __aexit__(self, *args):
            # With max_keepalive_connections=0, no pool drain — exits instantly
            pass

    chunks_received = []

    async def consume():
        async for chunk in client.chat_stream("test"):
            chunks_received.append(chunk)

    with patch("httpx.AsyncClient", SlowStreamContext):
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # let one chunk arrive
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # Should have received at least the first chunk before cancellation
    assert len(chunks_received) >= 1
    assert chunks_received[0] == "first chunk"


# ─────────────────────────────────────────────────────────────────────────────
# 4. current_model() uses /v1/admin/loaded-models (BUG-21-8)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_current_model_uses_loaded_models_endpoint():
    """current_model() must call /v1/admin/loaded-models, not /v1/admin/current-model."""
    from octane.tools.bodega_inference import BodegaInferenceClient

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "object": "list",
        "data": [
            {"id": "bodega-raptor-8b", "status": "initialized"}
        ],
        "total": 1,
    })

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = BodegaInferenceClient.__new__(BodegaInferenceClient)
    client._get_client = AsyncMock(return_value=mock_client)

    result = await client.current_model()

    # Verify correct endpoint called
    mock_client.get.assert_called_once_with("/v1/admin/loaded-models")
    # Verify normalized response
    assert result["loaded"] is True
    assert result["model_path"] == "bodega-raptor-8b"
    assert result["model_info"]["model_path"] == "bodega-raptor-8b"
    assert result["total_loaded"] == 1


@pytest.mark.asyncio
async def test_current_model_no_models_loaded():
    """current_model() returns loaded=False when loaded-models data is empty."""
    from octane.tools.bodega_inference import BodegaInferenceClient

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"object": "list", "data": [], "total": 0})

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = BodegaInferenceClient.__new__(BodegaInferenceClient)
    client._get_client = AsyncMock(return_value=mock_client)

    result = await client.current_model()

    assert result["loaded"] is False
    assert result["model_info"] == {}


@pytest.mark.asyncio
async def test_current_model_multi_model_registry():
    """current_model() exposes all_models list and total_loaded for multi-model Bodega."""
    from octane.tools.bodega_inference import BodegaInferenceClient

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "object": "list",
        "data": [
            {"id": "bodega-raptor-8b", "status": "initialized"},
            {"id": "bodega-solomon-9b", "status": "initialized"},
        ],
        "total": 2,
    })

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = BodegaInferenceClient.__new__(BodegaInferenceClient)
    client._get_client = AsyncMock(return_value=mock_client)

    result = await client.current_model()

    assert result["loaded"] is True
    assert result["total_loaded"] == 2
    assert result["all_models"] == ["bodega-raptor-8b", "bodega-solomon-9b"]
    # Primary model is first entry
    assert result["model_path"] == "bodega-raptor-8b"
