"""Session 30 — --recall mode for octane ask.

Tests cover:
    - CLI option registered and help text present
    - _ask_recall returns early when no stored data matches
    - _ask_recall builds correct context from injected pg rows and streams
      tokens from a mocked BodegaRouter.chat_stream
    - Postgres unavailability is handled gracefully (no crash)
    - Redis unavailability is handled gracefully (no crash)

All tests are pure-unit — no network, no Bodega, no Postgres, no Redis.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


async def _agen(*items):
    """Yield items as an async generator."""
    for item in items:
        yield item


# ─────────────────────────────────────────────────────────────────────────────
# CLI registration
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallCLIRegistered:
    """--recall flag must be visible in octane ask --help."""

    def test_recall_flag_in_help(self):
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0, result.output
        assert "--recall" in result.output or "-r" in result.output

    def test_recall_is_boolean_flag(self):
        """Invoking with --recall should not require an argument."""
        import inspect
        from octane.cli.ask import ask
        sig = inspect.signature(ask)
        assert "recall" in sig.parameters
        param = sig.parameters["recall"]
        assert param.default is not inspect.Parameter.empty


# ─────────────────────────────────────────────────────────────────────────────
# _ask_recall — no stored data
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallNoData:
    """When Postgres returns zero rows, _ask_recall prints guidance and exits."""

    def test_no_data_prints_guidance(self):
        """_ask_recall must print guidance when no rows match the query."""
        from octane.cli.ask import _ask_recall

        mock_pg = MagicMock()
        mock_pg.available = True
        mock_pg.connect = AsyncMock()
        mock_pg.close = AsyncMock()
        mock_pg.fetch = AsyncMock(return_value=[])

        # Redis: no active keys
        mock_r = AsyncMock()
        mock_r.smembers = AsyncMock(return_value=set())
        mock_r.aclose = AsyncMock()

        output_lines: list[str] = []

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.tools.bodega_router.BodegaRouter"),
            patch("redis.asyncio.from_url", return_value=mock_r),
            patch("octane.cli.ask.console") as mock_console,
            patch("octane.cli.ask.err_console") as mock_err_console,
        ):
            mock_console.print = lambda *a, **kw: output_lines.append(str(a))
            mock_err_console.print = lambda *a, **kw: output_lines.append(str(a))
            mock_console.status = MagicMock().__enter__ = MagicMock(return_value=MagicMock())
            _run(_ask_recall("NVDA earnings"))

        combined = " ".join(output_lines)
        assert "No stored data" in combined or "octane research start" in combined


# ─────────────────────────────────────────────────────────────────────────────
# _ask_recall — with data
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallWithData:
    """When Postgres has rows, _ask_recall streams synthesis from REASON tier."""

    def _make_pg(self, findings=None, pages=None):
        from datetime import datetime, timezone
        mock_pg = MagicMock()
        mock_pg.available = True
        mock_pg.connect = AsyncMock()
        mock_pg.close = AsyncMock()

        _findings = findings or [
            {
                "topic": "NVDA earnings Q1 2026",
                "content": "NVDA beat estimates by 12%.",
                "cycle_num": 1,
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ]
        _pages = pages or [
            {
                "title": "NVDA Q1 Results",
                "url": "https://example.com/nvda",
                "content": "Revenue hit $44B.",
                "fetched_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            }
        ]

        async def _fetch(sql, *args):
            if "research_findings" in sql:
                return _findings
            if "web_pages" in sql:
                return _pages
            return []

        mock_pg.fetch = _fetch
        return mock_pg

    def test_streams_tokens_from_reason_tier(self):
        """_ask_recall must call chat_stream with ModelTier.REASON."""
        from octane.cli.ask import _ask_recall
        from octane.tools.topology import ModelTier

        mock_pg = self._make_pg()

        mock_r = AsyncMock()
        mock_r.smembers = AsyncMock(return_value=set())
        mock_r.aclose = AsyncMock()

        called_tier = {}

        async def _fake_stream(prompt, system, tier, max_tokens):
            called_tier["tier"] = tier
            yield "NVDA "
            yield "earnings "
            yield "strong."

        mock_router_instance = MagicMock()
        mock_router_instance._client = MagicMock()
        mock_router_instance._client.health = AsyncMock(return_value={"status": "ok"})
        mock_router_instance.chat_stream = _fake_stream

        mock_router_cls = MagicMock(return_value=mock_router_instance)

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.tools.bodega_router.BodegaRouter", mock_router_cls),
            patch("redis.asyncio.from_url", return_value=mock_r),
            patch("octane.tools.topology.detect_topology", return_value="power"),
            patch("octane.tools.topology.get_topology", return_value=MagicMock()),
            patch("octane.cli.ask.console") as mock_console,
        ):
            mock_console.print = MagicMock()
            mock_console.status = MagicMock()
            mock_console.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_console.status.return_value.__exit__ = MagicMock(return_value=False)
            _run(_ask_recall("NVDA earnings"))

        assert called_tier.get("tier") == ModelTier.REASON

    def test_pg_unavailable_does_not_crash(self):
        """If Postgres is down, _ask_recall handles gracefully."""
        from octane.cli.ask import _ask_recall

        mock_pg = MagicMock()
        mock_pg.available = False
        mock_pg.connect = AsyncMock()
        mock_pg.close = AsyncMock()

        mock_r = AsyncMock()
        mock_r.smembers = AsyncMock(return_value=set())
        mock_r.aclose = AsyncMock()

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.tools.bodega_router.BodegaRouter"),
            patch("redis.asyncio.from_url", return_value=mock_r),
            patch("octane.cli.ask.console") as mock_console,
        ):
            mock_console.print = MagicMock()
            mock_console.status = MagicMock()
            mock_console.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_console.status.return_value.__exit__ = MagicMock(return_value=False)
            # Should not raise
            _run(_ask_recall("anything"))

    def test_redis_unavailable_does_not_crash(self):
        """If Redis is down, _ask_recall falls back gracefully."""
        from octane.cli.ask import _ask_recall

        mock_pg = self._make_pg()

        async def _fake_stream(prompt, system, tier, max_tokens):
            yield "ok"

        mock_router_instance = MagicMock()
        mock_router_instance._client = MagicMock()
        mock_router_instance._client.health = AsyncMock(return_value={"status": "ok"})
        mock_router_instance.chat_stream = _fake_stream

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.tools.bodega_router.BodegaRouter", return_value=mock_router_instance),
            patch("redis.asyncio.from_url", side_effect=ConnectionError("redis down")),
            patch("octane.tools.topology.detect_topology", return_value="power"),
            patch("octane.tools.topology.get_topology", return_value=MagicMock()),
            patch("octane.cli.ask.console") as mock_console,
        ):
            mock_console.print = MagicMock()
            mock_console.status = MagicMock()
            mock_console.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_console.status.return_value.__exit__ = MagicMock(return_value=False)
            # Should not raise
            _run(_ask_recall("NVDA earnings"))


# ─────────────────────────────────────────────────────────────────────────────
# Context budget
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallContextBudget:
    """_MAX_RECALL_CHARS constant must be at least 8000 chars."""

    def test_max_recall_chars_is_large_enough(self):
        from octane.cli.ask import _MAX_RECALL_CHARS
        assert _MAX_RECALL_CHARS >= 8_000
