"""Unit-test conftest — MockBodega, shared fixtures, and async helpers.

All fixtures here are available to every test under tests/unit/ without import.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEventBus


# ─────────────────────────────────────────────────────────────────────────────
# MockBodega — drop-in replacement for BodegaInferenceClient
# ─────────────────────────────────────────────────────────────────────────────

class MockBodega:
    """Configurable fake BodegaInferenceClient for unit tests.

    Args:
        chat_delay:       Seconds to sleep before chat_simple returns.
        stream_delay:     Seconds to sleep before EACH token in chat_stream.
        stream_chunks:    List of strings yielded by chat_stream (default: ["mock chunk"]).
        chat_response:    String returned by chat_simple (default: "mock response").
        raises:           If set, all methods raise this exception.
        stream_raises:    If set, chat_stream raises this after the first delay.
        health_response:  Dict returned by health() (default: {"status": "ok"}).
    """

    def __init__(
        self,
        *,
        chat_delay: float = 0.0,
        stream_delay: float = 0.0,
        stream_chunks: list[str] | None = None,
        chat_response: str = "mock response",
        raises: Exception | None = None,
        stream_raises: Exception | None = None,
        health_response: dict | None = None,
    ) -> None:
        self.chat_delay = chat_delay
        self.stream_delay = stream_delay
        self.stream_chunks = stream_chunks or ["mock chunk"]
        self.chat_response = chat_response
        self.raises = raises
        self.stream_raises = stream_raises
        self.health_response = health_response or {"status": "ok"}
        # Call counters for assertion
        self.chat_simple_calls: int = 0
        self.chat_stream_calls: int = 0

    async def health(self) -> dict:
        if self.raises:
            raise self.raises
        return self.health_response

    async def current_model(self) -> dict:
        return {"model": "mock-model", "model_path": "mock/path"}

    async def chat_simple(self, prompt: str = "", system: str = "", **kwargs) -> str:
        self.chat_simple_calls += 1
        if self.raises:
            raise self.raises
        if self.chat_delay > 0:
            await asyncio.sleep(self.chat_delay)
        return self.chat_response

    async def chat_stream(
        self, prompt: str = "", system: str = "", **kwargs
    ) -> AsyncIterator[str]:
        self.chat_stream_calls += 1
        if self.raises:
            raise self.raises
        if self.stream_delay > 0:
            await asyncio.sleep(self.stream_delay)
        if self.stream_raises:
            raise self.stream_raises
        for chunk in self.stream_chunks:
            yield chunk

    async def close(self) -> None:
        pass


class MockBodegaSlowStream(MockBodega):
    """MockBodega variant that yields tokens one-by-one with a per-token delay.

    Useful for testing streaming timeout behaviour without a real 120s wait —
    tests monkeypatch the _EVAL_CHUNK_TIMEOUT constant to a small value.
    """

    def __init__(self, token_delay: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.token_delay = token_delay

    async def chat_stream(self, prompt="", system="", **kwargs) -> AsyncIterator[str]:
        self.chat_stream_calls += 1
        for chunk in self.stream_chunks:
            if self.token_delay > 0:
                await asyncio.sleep(self.token_delay)
            yield chunk


class MockBodegaHanging(MockBodega):
    """MockBodega whose chat_stream never yields — hangs forever.

    Simulates a Bodega that starts generating but produces no output within the
    timeout window.  Use with monkeypatched _EVAL_CHUNK_TIMEOUT for fast tests.
    """

    async def chat_stream(self, prompt="", system="", **kwargs) -> AsyncIterator[str]:
        self.chat_stream_calls += 1
        await asyncio.sleep(9999)   # effectively forever
        yield "never reached"       # keeps it an async generator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synapse() -> SynapseEventBus:
    return SynapseEventBus(persist=False)


def make_request(query: str, metadata: dict | None = None) -> AgentRequest:
    return AgentRequest(
        query=query,
        session_id="test",
        source="test",
        metadata=metadata or {},
    )


async def collect_stream(gen: AsyncIterator[str]) -> list[str]:
    """Drain an async-generator and return all chunks as a list."""
    chunks: list[str] = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synapse():
    """A fresh in-memory SynapseEventBus for each test."""
    return SynapseEventBus(persist=False)


@pytest.fixture
def mock_bodega():
    """A basic MockBodega with instant responses."""
    return MockBodega()


@pytest.fixture
def mock_bodega_slow():
    """A MockBodega with 0.05 s per-token delay (fast enough for unit tests)."""
    return MockBodegaSlowStream(token_delay=0.05, stream_chunks=["slow ", "token "])


@pytest.fixture
def mock_bodega_hanging():
    """A MockBodega whose stream never yields."""
    return MockBodegaHanging()


@pytest.fixture
def mock_bodega_failing():
    """A MockBodega that always raises RuntimeError."""
    return MockBodega(raises=RuntimeError("Bodega offline"))
