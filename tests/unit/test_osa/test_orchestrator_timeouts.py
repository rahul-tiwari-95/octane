"""Timeout and cancellation tests for the OSA Orchestrator eval-stream path.

These tests validate the two-layer safety introduced in Session 18C:
  1. Per-chunk timeout: asyncio.wait_for(_eval_gen.__anext__(), _EVAL_CHUNK_TIMEOUT)
  2. Cleanup safety:    asyncio.wait_for(_eval_gen.aclose(), _EVAL_ACLOSE_TIMEOUT)

All tests use monkeypatched timeout constants (0.1 s instead of 120 s) so the
suite runs in milliseconds rather than minutes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import octane.osa.orchestrator as _orch_mod
from octane.models.dag import TaskDAG, TaskNode
from octane.models.schemas import AgentResponse
from octane.models.synapse import SynapseEventBus


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_synapse():
    return SynapseEventBus(persist=False)


def _fake_agent_response(text="data from agent"):
    return AgentResponse(agent="web", success=True, output=text)


def _single_node_dag(instruction="fetch NVDA price"):
    node = TaskNode(agent="web", instruction=instruction, metadata={"sub_agent": "finance"})
    return TaskDAG(
        nodes=[node],
        reasoning="single web task",
        original_query=instruction,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Evaluator stream: normal completion
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_stream_normal_yields_all_chunks(monkeypatch):
    """Happy path: evaluator yields 3 chunks, all are collected without timeout."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 2.0)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 1.0)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    async def fast_eval_stream(query, results, **kw):
        yield "chunk one "
        yield "chunk two "
        yield "chunk three"

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("NVDA is $189"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=fast_eval_stream),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        chunks = []
        async for chunk in osa.run_stream("NVDA price", session_id="test"):
            chunks.append(chunk)

    assert "".join(chunks) == "chunk one chunk two chunk three"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-chunk timeout: hanging stream falls back to raw concatenation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_chunk_timeout_yields_fallback_from_agent_output(monkeypatch):
    """When the eval generator stalls past the chunk timeout, raw agent output is returned."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.1)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    async def hanging_eval_stream(query, results, **kw):
        await asyncio.sleep(9999)
        yield "never"

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("NVDA is at $189.82 today"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=hanging_eval_stream),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        chunks = []
        async for chunk in osa.run_stream("NVDA price", session_id="test"):
            chunks.append(chunk)

    # Fallback should contain the raw agent output
    combined = "".join(chunks)
    assert "NVDA is at $189.82 today" in combined


@pytest.mark.asyncio
async def test_eval_chunk_timeout_fires_and_does_not_hang(monkeypatch):
    """run_stream must complete within a short wall-clock window even if eval hangs."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.1)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    async def hanging_eval_stream(query, results, **kw):
        await asyncio.sleep(9999)
        yield "never"

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("some data"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=hanging_eval_stream),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        # Should complete well within 2 seconds (0.05 + 0.1 s timeouts + overhead)
        result = await asyncio.wait_for(
            _collect_stream(osa.run_stream("test", session_id="test")),
            timeout=2.0,
        )

    assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Partial chunks: timeout after some chunks already yielded
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_chunk_timeout_preserves_partial_output(monkeypatch):
    """When timeout fires after some chunks, the partial output is NOT overwritten."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.1)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    async def slow_eval_stream(query, results, **kw):
        yield "partial answer "
        await asyncio.sleep(9999)   # hangs after first chunk
        yield "never reached"

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("raw data"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=slow_eval_stream),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        chunks = await asyncio.wait_for(
            _collect_stream(osa.run_stream("test", session_id="test")),
            timeout=2.0,
        )

    combined = "".join(chunks)
    # The already-yielded "partial answer" must be preserved
    assert "partial answer" in combined
    # Raw concatenation fallback should NOT appear (partial was already yielded)
    assert "raw data" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# 4. aclose() timeout: hung cleanup does not block stream completion
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_aclose_timeout_does_not_block_stream_return(monkeypatch):
    """Even if _eval_gen.aclose() hangs, the stream must complete within aclose timeout."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.1)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    class HungCloseGen:
        """Async generator that hangs inside aclose()."""

        def __init__(self):
            self._chunks = ["one ", "two"]
            self._idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._idx]
            self._idx += 1
            return chunk

        async def aclose(self):
            await asyncio.sleep(9999)   # simulates hung connection-pool drain

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("data"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", return_value=HungCloseGen()),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        chunks = await asyncio.wait_for(
            _collect_stream(osa.run_stream("test", session_id="test")),
            timeout=2.0,  # must finish in ≤ _EVAL_ACLOSE_TIMEOUT + overhead
        )

    assert "one " in "".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Guard block: no eval stream is even started
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_guard_block_short_circuits_before_eval(monkeypatch):
    """If Guard blocks the query, run_stream should yield the block message and stop."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    eval_called = []

    async def spy_eval_stream(query, results, **kw):
        eval_called.append(True)
        yield "should not reach"

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": False, "reason": "unsafe content"}),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=spy_eval_stream),
    ):
        chunks = await _collect_stream(osa.run_stream("harmful query", session_id="test"))

    assert not eval_called, "evaluate_stream must not be called when guard blocks"
    combined = "".join(chunks)
    assert combined  # some output is returned (the block message)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Egress event is emitted even after timeout fallback
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_egress_event_emitted_after_timeout_fallback(monkeypatch):
    """A Synapse egress event must be emitted even when the chunk timeout fires."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.1)

    synapse = make_synapse()
    osa = _orch_mod.Orchestrator(synapse)
    osa._preflight_done = True

    dag = _single_node_dag()

    async def hanging_eval_stream(query, results, **kw):
        await asyncio.sleep(9999)
        yield "never"

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("raw result"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=hanging_eval_stream),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        await asyncio.wait_for(
            _collect_stream(osa.run_stream("test", session_id="test")),
            timeout=2.0,
        )

    egress_events = [e for e in synapse._events if e.event_type == "egress"]
    assert egress_events, "Egress event must be emitted even after timeout fallback"


# ─────────────────────────────────────────────────────────────────────────────
# 7. run_from_dag: same timeout guard is active
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_from_dag_timeout_fallback(monkeypatch):
    """run_from_dag must also honour _EVAL_CHUNK_TIMEOUT and fall back gracefully."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 0.05)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.1)

    synapse = make_synapse()
    osa = _orch_mod.Orchestrator(synapse)
    osa._preflight_done = True

    dag = _single_node_dag()

    async def hanging_eval_stream(query, results, **kw):
        await asyncio.sleep(9999)
        yield "never"

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("raw from agent"))

    with (
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=hanging_eval_stream),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        chunks = await asyncio.wait_for(
            _collect_stream(osa.run_from_dag(dag, "test query")),
            timeout=2.0,
        )

    combined = "".join(chunks)
    assert "raw from agent" in combined


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluator evaluate_stream: StopAsyncIteration on first call (empty output)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_stream_empty_generator_completes_cleanly(monkeypatch):
    """An eval generator that immediately raises StopAsyncIteration produces empty output."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 2.0)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    async def empty_eval_stream(query, results, **kw):
        return
        yield  # make it an async generator

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("agent output"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=empty_eval_stream),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        # Should complete without error
        chunks = await _collect_stream(osa.run_stream("test", session_id="test"))

    # Empty evaluator output = empty chunks list (no fallback needed; partial list is empty)
    # The egress event is still emitted
    synapse_events = list(osa.synapse._events)
    assert any(e.event_type == "egress" for e in synapse_events)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Agent failure: run_stream completes even when all agents fail
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_agents_fail_stream_still_completes(monkeypatch):
    """When all dispatched agents return success=False, stream must still complete."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 2.0)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    async def eval_with_failures(query, results, **kw):
        yield "Unable to retrieve data at this time."

    failed_resp = AgentResponse(agent="web", success=False, output="")
    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=failed_resp)

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", side_effect=eval_with_failures),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        chunks = await _collect_stream(osa.run_stream("test", session_id="test"))

    assert "Unable to retrieve" in "".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Cancellation: cancelling run_stream does not leak the eval generator
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_stream_cancellation_closes_eval_gen(monkeypatch):
    """Cancelling run_stream while it is yielding should close the eval generator."""
    monkeypatch.setattr(_orch_mod, "_EVAL_CHUNK_TIMEOUT", 2.0)
    monkeypatch.setattr(_orch_mod, "_EVAL_ACLOSE_TIMEOUT", 0.5)

    osa = _orch_mod.Orchestrator(make_synapse())
    osa._preflight_done = True

    dag = _single_node_dag()

    gen_closed = []

    class TrackingGen:
        """Async generator that records whether aclose() was called."""

        async def __anext__(self):
            await asyncio.sleep(0.02)
            return "chunk"

        def __aiter__(self):
            return self

        async def aclose(self):
            gen_closed.append(True)

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=_fake_agent_response("data"))

    with (
        patch.object(osa.guard, "check_input", return_value={"safe": True}),
        patch.object(osa.decomposer, "decompose", return_value=dag),
        patch.object(osa.router, "get_agent", return_value=agent_mock),
        patch.object(osa.evaluator, "evaluate_stream", return_value=TrackingGen()),
        patch.object(osa.policy, "check_query_length", return_value=None),
        patch.object(osa, "_get_memory_agent", return_value=None),
    ):
        task = asyncio.create_task(
            _collect_stream(osa.run_stream("test", session_id="test"))
        )
        await asyncio.sleep(0.06)   # let one chunk through
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # aclose() must have been called (via the finally block)
    assert gen_closed, "eval generator aclose() was not called after task cancellation"


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_stream(gen) -> list[str]:
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks
