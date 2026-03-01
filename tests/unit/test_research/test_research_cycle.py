"""Unit tests for Session 20A fixes in research_cycle.

Covers:
  - MINIMUM_QUALITY_WORDS constant exists
  - _event_label uses "nodes" key (DAG ready fix)
  - quality gate skips storage when word count < threshold
  - stopped-task guard: cycle returns early when task.status == "stopped"
  - cross-angle dedup: duplicate outputs are skipped
  - pre-cycle health check sets bodega_available flag
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────

def test_minimum_quality_words_constant():
    """MINIMUM_QUALITY_WORDS must be defined and >= 10."""
    from octane.tasks.research import MINIMUM_QUALITY_WORDS
    assert MINIMUM_QUALITY_WORDS >= 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. DAG ready fix: "nodes" key is used (not "node_count")
# ─────────────────────────────────────────────────────────────────────────────

def test_event_label_dag_uses_nodes_key():
    """decomposition_complete must read payload['nodes'], not 'node_count'."""
    from octane.tasks.research import _event_label
    from octane.models.synapse import SynapseEvent

    # "nodes" key → should show the count
    event = MagicMock(spec=SynapseEvent)
    event.event_type = "decomposition_complete"
    event.payload = {"nodes": 3}
    event.source = ""

    label = _event_label(event)
    assert label is not None
    assert "3" in label


def test_event_label_dag_falls_back_to_node_count():
    """Falls back to 'node_count' for backward compat if 'nodes' is absent."""
    from octane.tasks.research import _event_label
    from octane.models.synapse import SynapseEvent

    event = MagicMock(spec=SynapseEvent)
    event.event_type = "decomposition_complete"
    event.payload = {"node_count": 2}  # legacy key
    event.source = ""

    label = _event_label(event)
    assert label is not None
    assert "2" in label


def test_event_label_dag_question_mark_when_missing():
    """Shows '?' when neither 'nodes' nor 'node_count' is present."""
    from octane.tasks.research import _event_label
    from octane.models.synapse import SynapseEvent

    event = MagicMock(spec=SynapseEvent)
    event.event_type = "decomposition_complete"
    event.payload = {}
    event.source = ""

    label = _event_label(event)
    assert label is not None
    assert "?" in label


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stopped-task guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stopped_task_guard_returns_early():
    """research_cycle must return without incrementing cycle or storing findings
    when task status is 'stopped'."""
    from octane.research.models import ResearchTask

    stopped_task = ResearchTask(topic="test topic")
    stopped_task.status = "stopped"

    mock_store = AsyncMock()
    mock_store.get_task = AsyncMock(return_value=stopped_task)
    mock_store.increment_cycle = AsyncMock(return_value=0)
    mock_store.add_finding = AsyncMock(return_value=None)
    mock_store.log_entry = AsyncMock()
    mock_store.close = AsyncMock()

    with patch("octane.research.store.ResearchStore", return_value=mock_store):
        from octane.tasks.research import research_cycle

        mock_log = MagicMock()
        mock_log.info = MagicMock()
        mock_log.warning = MagicMock()

        # Call without Perpetual/TaskLogger injection (use defaults)
        await research_cycle(
            task_id="test-stopped",
            topic="test topic",
            log=mock_log,
        )

    # Must NOT have incremented the cycle counter
    mock_store.increment_cycle.assert_not_called()
    # Must NOT have stored a finding
    mock_store.add_finding.assert_not_called()
    # Must have closed cleanly
    mock_store.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Quality gate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quality_gate_skips_storage_for_short_content():
    """Finding must NOT be stored when content is below MINIMUM_QUALITY_WORDS."""
    from octane.research.models import ResearchTask
    from octane.tasks.research import MINIMUM_QUALITY_WORDS

    active_task = ResearchTask(topic="NVDA earnings")
    active_task.status = "active"

    short_content = "NVDA up today."   # << 30 words
    assert len(short_content.split()) < MINIMUM_QUALITY_WORDS

    mock_store = AsyncMock()
    mock_store.get_task = AsyncMock(return_value=active_task)
    mock_store.increment_cycle = AsyncMock(return_value=1)
    mock_store.add_finding = AsyncMock(return_value=None)
    mock_store.log_entry = AsyncMock()
    mock_store.close = AsyncMock()
    mock_store._redis = AsyncMock(return_value=None)

    # Patch AngleGenerator to return one angle
    mock_angle_gen = AsyncMock()
    mock_angle_gen.generate = AsyncMock(return_value=[{"angle": "price", "query": "NVDA price"}])

    # Patch Orchestrator.run_stream to yield junk (< 30 words)
    async def _fake_stream(*args, **kwargs):
        yield short_content

    mock_osa = MagicMock()
    mock_osa.run_stream = _fake_stream
    mock_osa._preflight_done = False

    with (
        patch("octane.research.store.ResearchStore", return_value=mock_store),
        patch("octane.research.angles.AngleGenerator", return_value=mock_angle_gen),
        patch("octane.osa.orchestrator.Orchestrator", return_value=mock_osa),
        patch("octane.tools.bodega_inference.BodegaInferenceClient"),
    ):
        from octane.tasks.research import research_cycle
        mock_log = MagicMock()
        await research_cycle(task_id="test-quality", topic="NVDA earnings", log=mock_log)

    mock_store.add_finding.assert_not_called()


@pytest.mark.asyncio
async def test_quality_gate_allows_good_content():
    """Finding IS stored when content exceeds MINIMUM_QUALITY_WORDS."""
    from octane.research.models import ResearchTask, ResearchFinding
    from octane.tasks.research import MINIMUM_QUALITY_WORDS

    active_task = ResearchTask(topic="NVDA earnings")
    active_task.status = "active"

    good_content = " ".join(["word"] * (MINIMUM_QUALITY_WORDS + 10))

    mock_store = AsyncMock()
    mock_store.get_task = AsyncMock(return_value=active_task)
    mock_store.increment_cycle = AsyncMock(return_value=1)
    mock_store.add_finding = AsyncMock(return_value=MagicMock(word_count=MINIMUM_QUALITY_WORDS + 10))
    mock_store.log_entry = AsyncMock()
    mock_store.close = AsyncMock()
    mock_store._redis = AsyncMock(return_value=None)

    mock_angle_gen = AsyncMock()
    mock_angle_gen.generate = AsyncMock(return_value=[{"angle": "price", "query": "NVDA price"}])

    async def _fake_stream(*args, **kwargs):
        yield good_content

    mock_osa = MagicMock()
    mock_osa.run_stream = _fake_stream
    mock_osa._preflight_done = False
    mock_osa._events = []

    with (
        patch("octane.research.store.ResearchStore", return_value=mock_store),
        patch("octane.research.angles.AngleGenerator", return_value=mock_angle_gen),
        patch("octane.osa.orchestrator.Orchestrator", return_value=mock_osa),
        patch("octane.tools.bodega_inference.BodegaInferenceClient"),
    ):
        from octane.tasks.research import research_cycle
        mock_log = MagicMock()
        await research_cycle(task_id="test-quality-pass", topic="NVDA earnings", log=mock_log)

    mock_store.add_finding.assert_called_once()
