"""Unit tests for PortfolioStore (mocked Postgres, no real DB)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octane.portfolio.models import Position, Portfolio
from octane.portfolio.store import _row_to_position


# ── _row_to_position helper ───────────────────────────────────────────────────

class TestRowToPosition:
    def _make_row(self, **overrides):
        defaults = {
            "id": 1,
            "ticker": "NVDA",
            "quantity": 100.0,
            "avg_cost": 650.0,
            "currency": "USD",
            "broker": "Schwab",
            "account_id": "IRA",
            "sector": "Technology",
            "asset_class": "equity",
            "notes": "",
            "project_id": None,
        }
        defaults.update(overrides)
        return defaults

    def test_basic_mapping(self):
        row = self._make_row()
        pos = _row_to_position(row)
        assert pos.ticker == "NVDA"
        assert pos.quantity == 100.0
        assert pos.avg_cost == 650.0

    def test_broker_mapped(self):
        pos = _row_to_position(self._make_row(broker="Fidelity"))
        assert pos.broker == "Fidelity"

    def test_sector_mapped(self):
        pos = _row_to_position(self._make_row(sector="Healthcare"))
        assert pos.sector == "Healthcare"

    def test_missing_broker_defaults_empty(self):
        row = self._make_row()
        del row["broker"]
        pos = _row_to_position(row)
        assert pos.broker == ""


# ── PortfolioStore with mocked PgClient ──────────────────────────────────────

@pytest.fixture
def mock_pg():
    pg = MagicMock()
    pg.available = True
    pg.connect = AsyncMock()
    pg.close = AsyncMock()
    pg.fetchrow = AsyncMock(return_value={"id": 42})
    pg.fetch = AsyncMock(return_value=[])
    pg.fetchval = AsyncMock(return_value=None)
    pg.execute = AsyncMock(return_value=1)
    return pg


@pytest.fixture
def store(mock_pg):
    from octane.portfolio.store import PortfolioStore
    s = PortfolioStore.__new__(PortfolioStore)
    s._pg = mock_pg
    return s


class TestPortfolioStoreUpsert:
    @pytest.mark.asyncio
    async def test_upsert_returns_id(self, store, mock_pg):
        pos = Position(ticker="NVDA", quantity=10, avg_cost=650.0, broker="Schwab")
        mock_pg.fetchrow = AsyncMock(return_value={"id": 42})
        result = await store.upsert_position(pos)
        assert result == 42

    @pytest.mark.asyncio
    async def test_upsert_calls_fetchrow(self, store, mock_pg):
        pos = Position(ticker="AAPL", quantity=5, avg_cost=180.0)
        await store.upsert_position(pos)
        mock_pg.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_many_count(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value={"id": 1})
        positions = [
            Position("NVDA", 10, 650.0),
            Position("AAPL", 5, 180.0),
            Position("SPY", 20, 450.0),
        ]
        count = await store.upsert_many(positions)
        assert count == 3

    @pytest.mark.asyncio
    async def test_unavailable_pg_raises(self, store, mock_pg):
        mock_pg.available = False
        with pytest.raises(RuntimeError, match="not available"):
            await store.upsert_position(Position("NVDA", 10, 650.0))


class TestPortfolioStoreRead:
    @pytest.mark.asyncio
    async def test_list_returns_empty_when_no_rows(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[])
        positions = await store.list_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_list_maps_rows(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[{
            "id": 1, "ticker": "NVDA", "quantity": 10.0, "avg_cost": 650.0,
            "currency": "USD", "broker": "Schwab", "account_id": "",
            "sector": "Technology", "asset_class": "equity", "notes": "", "project_id": None,
        }])
        positions = await store.list_positions()
        assert len(positions) == 1
        assert positions[0].ticker == "NVDA"

    @pytest.mark.asyncio
    async def test_get_position_none_when_missing(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value=None)
        result = await store.get_position("NVDA")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_tickers(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[{"ticker": "NVDA"}, {"ticker": "AAPL"}])
        tickers = await store.get_tickers()
        assert "NVDA" in tickers

    @pytest.mark.asyncio
    async def test_unavailable_pg_returns_empty_list(self, store, mock_pg):
        mock_pg.available = False
        positions = await store.list_positions()
        assert positions == []


class TestPortfolioStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_calls_execute(self, store, mock_pg):
        mock_pg.execute = AsyncMock(return_value=1)
        count = await store.delete_position("NVDA")
        mock_pg.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_calls_execute(self, store, mock_pg):
        mock_pg.execute = AsyncMock(return_value=5)
        await store.clear()
        mock_pg.execute.assert_called_once()
