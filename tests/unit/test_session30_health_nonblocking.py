"""
Session 30 — health/sysstat non-blocking tests.

Verifies that:
1. _health() renders the System panel even when Bodega times out.
2. _sysstat() renders the System panel even when Bodega times out.
3. A normal (fast) Bodega response is displayed correctly.
4. Bodega offline (ConnectionRefusedError) is handled gracefully.

Strategy
--------
* Deferred imports inside _health()/_sysstat() are patched at their SOURCE
  module, not at octane.cli.health (which never holds those names at module
  scope).
* We replace octane.cli.health.console with a real rich.console.Console that
  writes to a StringIO buffer so we can assert on rendered text.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console


# ── helpers ──────────────────────────────────────────────────────────────────

_SNAPSHOT = {
    "ram_used_gb": 24.5,
    "ram_total_gb": 48.0,
    "ram_available_gb": 23.5,
    "ram_percent": 51,
    "cpu_percent": 12.3,
    "cpu_count": 10,
}

_GOOD_MODEL = {
    "loaded": True,
    "model_path": "/models/mlx-community/gemma-3-12b-it-4bit",
    "all_models": ["/models/mlx-community/gemma-3-12b-it-4bit"],
    "context_length": 8192,
}

_GOOD_HEALTH = {"status": "ok"}


def _make_monitor_mock():
    m = MagicMock()
    m.snapshot.return_value = _SNAPSHOT
    return m


def _make_bodega_mock(*, timeout_on_model: bool = False):
    b = AsyncMock()
    b.close = AsyncMock()
    if timeout_on_model:
        async def _slow():
            await asyncio.sleep(999)
        b.current_model.side_effect = _slow
        b.health.side_effect = _slow
    else:
        b.current_model.return_value = dict(_GOOD_MODEL)
        b.health.return_value = dict(_GOOD_HEALTH)
    return b


def _make_topo_mock():
    t = MagicMock()
    t.resolve_config.return_value = MagicMock(
        model_id="gemma-3-4b",
        model_path="/models/gemma-3-4b",
    )
    return t


def _capture_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, no_color=True)
    return con, buf


# ── _health() tests ───────────────────────────────────────────────────────────

class TestHealthNonBlocking:

    @pytest.mark.asyncio
    async def test_health_renders_when_bodega_times_out(self):
        from octane.cli.health import _health
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient",
                  return_value=_make_bodega_mock(timeout_on_model=True)),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.tools.topology.detect_topology", return_value="default"),
            patch("octane.tools.topology.get_topology", return_value=_make_topo_mock()),
            patch("octane.cli.health.console", con),
        ):
            await _health()
        output = buf.getvalue()
        assert "busy" in output.lower() or "inference" in output.lower(), (
            f"Expected 'busy'/'inference' in output.\n---\n{output}"
        )

    @pytest.mark.asyncio
    async def test_health_renders_normally_when_bodega_ok(self):
        from octane.cli.health import _health
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient",
                  return_value=_make_bodega_mock()),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 3.0),
            patch("octane.tools.topology.detect_topology", return_value="default"),
            patch("octane.tools.topology.get_topology", return_value=_make_topo_mock()),
            patch("octane.cli.health.console", con),
        ):
            await _health()
        output = buf.getvalue()
        assert "gemma" in output.lower(), f"Expected model name in output.\n---\n{output}"

    @pytest.mark.asyncio
    async def test_health_handles_bodega_offline(self):
        from octane.cli.health import _health
        bodega = _make_bodega_mock()
        bodega.current_model.side_effect = ConnectionRefusedError("no server")
        bodega.health.side_effect = ConnectionRefusedError("no server")
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient", return_value=bodega),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.tools.topology.detect_topology", return_value="default"),
            patch("octane.tools.topology.get_topology", return_value=_make_topo_mock()),
            patch("octane.cli.health.console", con),
        ):
            await _health()
        output = buf.getvalue()
        assert "offline" in output.lower(), f"Expected 'offline' in output.\n---\n{output}"

    @pytest.mark.asyncio
    async def test_health_system_panel_always_uses_psutil(self):
        from octane.cli.health import _health
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient",
                  return_value=_make_bodega_mock(timeout_on_model=True)),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.tools.topology.detect_topology", return_value="default"),
            patch("octane.tools.topology.get_topology", return_value=_make_topo_mock()),
            patch("octane.cli.health.console", con),
        ):
            await _health()
        output = buf.getvalue()
        assert "24.5" in output and "48.0" in output, (
            f"Expected RAM values 24.5/48.0 in output.\n---\n{output}"
        )

    @pytest.mark.asyncio
    async def test_health_bodega_close_always_called(self):
        from octane.cli.health import _health
        bodega = _make_bodega_mock(timeout_on_model=True)
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient", return_value=bodega),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.tools.topology.detect_topology", return_value="default"),
            patch("octane.tools.topology.get_topology", return_value=_make_topo_mock()),
            patch("octane.cli.health.console", con),
        ):
            await _health()
        bodega.close.assert_awaited_once()


# ── _sysstat() tests ──────────────────────────────────────────────────────────

class TestSysStatNonBlocking:

    @pytest.mark.asyncio
    async def test_sysstat_shows_busy_on_timeout(self):
        from octane.cli.health import _sysstat
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient",
                  return_value=_make_bodega_mock(timeout_on_model=True)),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.cli.health.console", con),
        ):
            await _sysstat()
        output = buf.getvalue()
        assert "busy" in output.lower() or "inference" in output.lower(), (
            f"Expected 'busy'/'inference'.\n---\n{output}"
        )

    @pytest.mark.asyncio
    async def test_sysstat_shows_model_when_bodega_ok(self):
        from octane.cli.health import _sysstat
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient",
                  return_value=_make_bodega_mock()),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 3.0),
            patch("octane.cli.health.console", con),
        ):
            await _sysstat()
        output = buf.getvalue()
        assert "gemma" in output.lower(), f"Expected model name.\n---\n{output}"

    @pytest.mark.asyncio
    async def test_sysstat_system_panel_always_shown(self):
        from octane.cli.health import _sysstat
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient",
                  return_value=_make_bodega_mock(timeout_on_model=True)),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.cli.health.console", con),
        ):
            await _sysstat()
        output = buf.getvalue()
        assert "24.5" in output and "48.0" in output, (
            f"Expected RAM values.\n---\n{output}"
        )

    @pytest.mark.asyncio
    async def test_sysstat_handles_offline(self):
        from octane.cli.health import _sysstat
        bodega = _make_bodega_mock()
        bodega.current_model.side_effect = ConnectionRefusedError("no server")
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient", return_value=bodega),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.cli.health.console", con),
        ):
            await _sysstat()
        output = buf.getvalue()
        assert "offline" in output.lower(), f"Expected 'offline'.\n---\n{output}"

    @pytest.mark.asyncio
    async def test_sysstat_close_always_called(self):
        from octane.cli.health import _sysstat
        bodega = _make_bodega_mock(timeout_on_model=True)
        con, buf = _capture_console()
        with (
            patch("octane.agents.sysstat.monitor.Monitor", return_value=_make_monitor_mock()),
            patch("octane.tools.bodega_inference.BodegaInferenceClient", return_value=bodega),
            patch("octane.cli.health._BODEGA_PROBE_TIMEOUT", 0.05),
            patch("octane.cli.health.console", con),
        ):
            await _sysstat()
        bodega.close.assert_awaited_once()
