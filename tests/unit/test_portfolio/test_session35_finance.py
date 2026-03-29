"""Unit tests for Session 35 — Advanced Finance.

Tests cover:
    - TaxLot model (FIFO/LIFO, remaining_shares, is_long_term)
    - Dividend model (annual_income_per_share)
    - CryptoPosition model (cost_basis, uppercase)
    - NetWorthSnapshot model
    - Financial calculations: XIRR, Sharpe ratio, tax-loss harvesting, dividend income
    - Crypto CSV parsers (Coinbase, Kraken, Binance, Gemini, Generic)
    - CoinGecko ID mapping
    - Store row converters
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octane.portfolio.models import (
    CryptoPosition,
    Dividend,
    NetWorthSnapshot,
    Position,
    TaxLot,
)
from octane.portfolio.finance import (
    HarvestCandidate,
    annual_dividend_income,
    find_harvest_candidates,
    sharpe_ratio,
    xirr,
)
from octane.portfolio.crypto import (
    detect_exchange,
    parse_crypto_csv_text,
    _COINGECKO_IDS,
)
from octane.portfolio.store import (
    _row_to_tax_lot,
    _row_to_dividend,
    _row_to_snapshot,
    _row_to_crypto,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TaxLot Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaxLotModel:
    def test_ticker_uppercased(self):
        lot = TaxLot(ticker="nvda", shares=10, cost_per_share=100.0)
        assert lot.ticker == "NVDA"

    def test_remaining_shares(self):
        lot = TaxLot(ticker="AAPL", shares=100, cost_per_share=150.0, sold_shares=25)
        assert lot.remaining_shares == 75.0

    def test_cost_basis(self):
        lot = TaxLot(ticker="AAPL", shares=100, cost_per_share=150.0, sold_shares=50)
        assert lot.cost_basis == 7500.0  # 50 remaining * $150

    def test_is_long_term_old_purchase(self):
        old_date = datetime.date.today() - datetime.timedelta(days=400)
        lot = TaxLot(ticker="MSFT", shares=10, cost_per_share=300.0, purchase_date=old_date)
        assert lot.is_long_term is True

    def test_is_short_term_recent(self):
        recent = datetime.date.today() - datetime.timedelta(days=30)
        lot = TaxLot(ticker="MSFT", shares=10, cost_per_share=300.0, purchase_date=recent)
        assert lot.is_long_term is False

    def test_to_dict_has_all_keys(self):
        lot = TaxLot(ticker="goog", shares=5, cost_per_share=175.0)
        d = lot.to_dict()
        assert d["ticker"] == "GOOG"
        assert d["shares"] == 5
        assert "remaining_shares" in d
        assert "is_long_term" in d
        assert "cost_basis" in d

    def test_zero_sold_means_full_remaining(self):
        lot = TaxLot(ticker="TSLA", shares=50, cost_per_share=200.0)
        assert lot.remaining_shares == 50.0


# ═══════════════════════════════════════════════════════════════════════════════
# Dividend Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDividendModel:
    def test_ticker_uppercased(self):
        d = Dividend(ticker="aapl", amount=0.96)
        assert d.ticker == "AAPL"

    def test_annual_income_quarterly(self):
        d = Dividend(ticker="AAPL", amount=0.96, frequency="quarterly")
        assert d.annual_income_per_share == 3.84

    def test_annual_income_monthly(self):
        d = Dividend(ticker="O", amount=0.25, frequency="monthly")
        assert d.annual_income_per_share == 3.0

    def test_annual_income_annual(self):
        d = Dividend(ticker="SPY", amount=6.0, frequency="annual")
        assert d.annual_income_per_share == 6.0

    def test_to_dict(self):
        d = Dividend(ticker="VZ", amount=0.665, div_yield=0.065)
        dd = d.to_dict()
        assert dd["ticker"] == "VZ"
        assert dd["amount"] == 0.665
        assert dd["div_yield"] == 0.065


# ═══════════════════════════════════════════════════════════════════════════════
# CryptoPosition Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCryptoPositionModel:
    def test_coin_uppercased(self):
        cp = CryptoPosition(coin="btc", quantity=1.5, cost_per_coin=30000)
        assert cp.coin == "BTC"

    def test_cost_basis(self):
        cp = CryptoPosition(coin="ETH", quantity=10, cost_per_coin=2000)
        assert cp.cost_basis == 20000.0

    def test_to_dict(self):
        cp = CryptoPosition(coin="SOL", quantity=50, cost_per_coin=100, exchange="Coinbase")
        d = cp.to_dict()
        assert d["coin"] == "SOL"
        assert d["cost_basis"] == 5000.0
        assert d["exchange"] == "Coinbase"


# ═══════════════════════════════════════════════════════════════════════════════
# NetWorthSnapshot Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetWorthSnapshotModel:
    def test_defaults(self):
        snap = NetWorthSnapshot()
        assert snap.total_value == 0.0
        assert snap.snapshot_date == datetime.date.today()

    def test_to_dict(self):
        snap = NetWorthSnapshot(total_value=100000, equities_value=80000, crypto_value=20000)
        d = snap.to_dict()
        assert d["total_value"] == 100000
        assert "snapshot_date" in d


# ═══════════════════════════════════════════════════════════════════════════════
# XIRR Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestXIRR:
    def test_simple_doubling(self):
        """$1000 invested, $2000 returned 1 year later → ~100% return."""
        flows = [
            (datetime.date(2023, 1, 1), -1000),
            (datetime.date(2024, 1, 1), 2000),
        ]
        rate = xirr(flows)
        assert rate is not None
        assert abs(rate - 1.0) < 0.01  # ~100%

    def test_zero_return(self):
        """$1000 invested, $1000 returned → ~0% return."""
        flows = [
            (datetime.date(2023, 1, 1), -1000),
            (datetime.date(2024, 1, 1), 1000),
        ]
        rate = xirr(flows)
        assert rate is not None
        assert abs(rate) < 0.01

    def test_loss(self):
        """$1000 invested, $500 returned → negative return."""
        flows = [
            (datetime.date(2023, 1, 1), -1000),
            (datetime.date(2024, 1, 1), 500),
        ]
        rate = xirr(flows)
        assert rate is not None
        assert rate < 0

    def test_multiple_cashflows(self):
        """Multiple investments and a final value."""
        flows = [
            (datetime.date(2022, 1, 1), -5000),
            (datetime.date(2022, 7, 1), -3000),
            (datetime.date(2023, 1, 1), -2000),
            (datetime.date(2024, 1, 1), 12000),
        ]
        rate = xirr(flows)
        assert rate is not None
        assert rate > 0  # Should be positive (gained money)

    def test_too_few_cashflows(self):
        assert xirr([(datetime.date(2023, 1, 1), -1000)]) is None

    def test_empty_cashflows(self):
        assert xirr([]) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Sharpe Ratio Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSharpeRatio:
    def test_positive_returns(self):
        # Slightly varied positive returns to avoid zero std dev
        returns = [0.005 + (i % 5) * 0.001 for i in range(100)]
        sr = sharpe_ratio(returns)
        assert sr is not None
        assert sr > 0

    def test_negative_returns(self):
        returns = [-0.005 - (i % 5) * 0.001 for i in range(100)]
        sr = sharpe_ratio(returns)
        assert sr is not None
        assert sr < 0

    def test_zero_volatility(self):
        # Same return every day → zero std → None
        returns = [0.005] * 100
        sr = sharpe_ratio(returns)
        assert sr is None

    def test_too_few_returns(self):
        assert sharpe_ratio([0.01]) is None
        assert sharpe_ratio([]) is None

    def test_high_sharpe(self):
        # Very consistent, moderately positive excess returns
        returns = [0.01 + (i % 3) * 0.001 for i in range(252)]
        sr = sharpe_ratio(returns, risk_free_annual=0.05)
        assert sr is not None
        assert sr > 3.0  # Very high Sharpe for consistent returns


# ═══════════════════════════════════════════════════════════════════════════════
# Tax-Loss Harvesting Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaxLossHarvesting:
    def test_finds_losers(self):
        positions = [
            {"ticker": "NVDA", "quantity": 10, "avg_cost": 900.0},
            {"ticker": "AAPL", "quantity": 20, "avg_cost": 200.0},
        ]
        prices = {"NVDA": 700.0, "AAPL": 220.0}
        candidates = find_harvest_candidates(positions, prices, min_loss_pct=5.0)
        assert len(candidates) == 1
        assert candidates[0].ticker == "NVDA"
        assert candidates[0].unrealised_loss < 0

    def test_no_losses_returns_empty(self):
        positions = [{"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0}]
        prices = {"AAPL": 200.0}
        assert find_harvest_candidates(positions, prices) == []

    def test_below_threshold_excluded(self):
        positions = [{"ticker": "MSFT", "quantity": 10, "avg_cost": 100.0}]
        prices = {"MSFT": 98.0}  # 2% loss — below 5% threshold
        assert find_harvest_candidates(positions, prices, min_loss_pct=5.0) == []

    def test_wash_sale_flag(self):
        positions = [{"ticker": "TSLA", "quantity": 5, "avg_cost": 300.0}]
        prices = {"TSLA": 200.0}
        recent_sales = [{"ticker": "TSLA", "sale_date": datetime.date.today().isoformat()}]
        candidates = find_harvest_candidates(positions, prices, recent_sales)
        assert len(candidates) == 1
        assert candidates[0].wash_sale_risk is True

    def test_old_sale_no_wash_risk(self):
        positions = [{"ticker": "TSLA", "quantity": 5, "avg_cost": 300.0}]
        prices = {"TSLA": 200.0}
        old = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
        recent_sales = [{"ticker": "TSLA", "sale_date": old}]
        candidates = find_harvest_candidates(positions, prices, recent_sales)
        assert len(candidates) == 1
        assert candidates[0].wash_sale_risk is False

    def test_missing_price_skipped(self):
        positions = [{"ticker": "XYZ", "quantity": 10, "avg_cost": 100.0}]
        prices = {}
        assert find_harvest_candidates(positions, prices) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Dividend Income Calculator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnualDividendIncome:
    def test_basic_calculation(self):
        holdings = [{"ticker": "AAPL", "quantity": 100}]
        div_info = {"AAPL": {"dividendRate": 3.84, "dividendYield": 0.005}}
        result = annual_dividend_income(holdings, div_info)
        assert result["total_annual_income"] == 384.0
        assert len(result["breakdown"]) == 1

    def test_multiple_tickers(self):
        holdings = [
            {"ticker": "AAPL", "quantity": 100},
            {"ticker": "MSFT", "quantity": 50},
        ]
        div_info = {
            "AAPL": {"dividendRate": 3.84, "dividendYield": 0.005},
            "MSFT": {"dividendRate": 3.0, "dividendYield": 0.008},
        }
        result = annual_dividend_income(holdings, div_info)
        assert result["total_annual_income"] == 384.0 + 150.0

    def test_no_dividends(self):
        holdings = [{"ticker": "TSLA", "quantity": 10}]
        div_info = {}
        result = annual_dividend_income(holdings, div_info)
        assert result["total_annual_income"] == 0.0

    def test_zero_rate(self):
        holdings = [{"ticker": "AMZN", "quantity": 10}]
        div_info = {"AMZN": {"dividendRate": 0, "dividendYield": 0}}
        result = annual_dividend_income(holdings, div_info)
        assert result["total_annual_income"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Crypto Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectExchange:
    def test_coinbase(self):
        assert detect_exchange(["Timestamp", "Transaction Type", "Asset", "Quantity Transacted", "Spot Price at Transaction"]) == "Coinbase"

    def test_kraken(self):
        assert detect_exchange(["txid", "asset", "amount", "fee", "type"]) == "Kraken"

    def test_binance(self):
        assert detect_exchange(["Date", "Coin", "Amount", "Price"]) == "Binance"

    def test_gemini(self):
        assert detect_exchange(["Date", "Symbol", "Amount", "Price (USD)"]) == "Gemini"

    def test_unknown(self):
        assert detect_exchange(["foo", "bar", "baz"]) == "Generic"


COINBASE_CSV = """\
Timestamp,Transaction Type,Asset,Quantity Transacted,Spot Price at Transaction,Subtotal
2024-01-15,Buy,BTC,0.5,42000.00,21000.00
2024-02-20,Buy,BTC,0.3,44000.00,13200.00
2024-03-10,Buy,ETH,5.0,3000.00,15000.00
2024-04-15,Sell,BTC,0.1,45000.00,4500.00
"""


class TestCoinbaseParser:
    def test_aggregates_buys(self):
        positions = parse_crypto_csv_text(COINBASE_CSV)
        btc = next((p for p in positions if p.coin == "BTC"), None)
        assert btc is not None
        assert btc.quantity == 0.8  # 0.5 + 0.3 (sells excluded)

    def test_eth_parsed(self):
        positions = parse_crypto_csv_text(COINBASE_CSV)
        eth = next((p for p in positions if p.coin == "ETH"), None)
        assert eth is not None
        assert eth.quantity == 5.0

    def test_sells_excluded(self):
        """Sell transactions should not increase holdings."""
        positions = parse_crypto_csv_text(COINBASE_CSV)
        btc = next((p for p in positions if p.coin == "BTC"), None)
        # Only buys: 0.5 + 0.3 = 0.8 (not minus 0.1 for sell)
        assert btc is not None
        assert btc.quantity == 0.8

    def test_exchange_detected(self):
        positions = parse_crypto_csv_text(COINBASE_CSV)
        assert all(p.exchange == "Coinbase" for p in positions)

    def test_cost_per_coin_weighted(self):
        positions = parse_crypto_csv_text(COINBASE_CSV)
        btc = next(p for p in positions if p.coin == "BTC")
        # Weighted avg: (0.5*42000 + 0.3*44000) / 0.8 = (21000+13200)/0.8 = 42750
        expected_avg = (0.5 * 42000 + 0.3 * 44000) / 0.8
        assert abs(btc.cost_per_coin - expected_avg) < 1.0


BINANCE_CSV = """\
Date,Coin,Amount,Price
2024-01-01,SOL,100,95.50
2024-02-01,SOL,50,120.00
2024-03-01,ADA,10000,0.55
"""


class TestBinanceParser:
    def test_parses_coins(self):
        positions = parse_crypto_csv_text(BINANCE_CSV, exchange="Binance")
        coins = {p.coin for p in positions}
        assert "SOL" in coins
        assert "ADA" in coins

    def test_sol_aggregated(self):
        positions = parse_crypto_csv_text(BINANCE_CSV, exchange="Binance")
        sol = next(p for p in positions if p.coin == "SOL")
        assert sol.quantity == 150.0

    def test_ada_cost(self):
        positions = parse_crypto_csv_text(BINANCE_CSV, exchange="Binance")
        ada = next(p for p in positions if p.coin == "ADA")
        assert ada.cost_per_coin == 0.55


GEMINI_CSV = """\
Date,Symbol,Amount,Price (USD)
2024-06-01,BTCUSD,0.1,65000.00
2024-06-15,ETHUSD,2.0,3500.00
"""


class TestGeminiParser:
    def test_strips_usd_suffix(self):
        positions = parse_crypto_csv_text(GEMINI_CSV, exchange="Gemini")
        coins = {p.coin for p in positions}
        assert "BTC" in coins
        assert "ETH" in coins

    def test_btc_quantity(self):
        positions = parse_crypto_csv_text(GEMINI_CSV, exchange="Gemini")
        btc = next(p for p in positions if p.coin == "BTC")
        assert btc.quantity == 0.1


GENERIC_CRYPTO_CSV = """\
Coin,Quantity,Price
DOGE,50000,0.15
LINK,200,12.50
"""


class TestGenericCryptoParser:
    def test_parses_generic(self):
        positions = parse_crypto_csv_text(GENERIC_CRYPTO_CSV)
        assert len(positions) == 2

    def test_doge_quantity(self):
        positions = parse_crypto_csv_text(GENERIC_CRYPTO_CSV)
        doge = next(p for p in positions if p.coin == "DOGE")
        assert doge.quantity == 50000.0

    def test_empty_csv(self):
        positions = parse_crypto_csv_text("")
        assert positions == []


# ═══════════════════════════════════════════════════════════════════════════════
# CoinGecko ID Mapping Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoinGeckoMapping:
    def test_btc_maps_to_bitcoin(self):
        assert _COINGECKO_IDS["BTC"] == "bitcoin"

    def test_eth_maps_to_ethereum(self):
        assert _COINGECKO_IDS["ETH"] == "ethereum"

    def test_sol_maps_to_solana(self):
        assert _COINGECKO_IDS["SOL"] == "solana"

    def test_all_values_are_strings(self):
        for k, v in _COINGECKO_IDS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Store Row Converter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowConverters:
    def test_row_to_tax_lot(self):
        row = {
            "id": 1, "ticker": "NVDA", "shares": 10.0, "cost_per_share": 650.0,
            "purchase_date": datetime.date(2023, 6, 15), "broker": "Schwab",
            "account_id": "IRA", "sold_shares": 2.0, "notes": "", "position_id": 5,
        }
        lot = _row_to_tax_lot(row)
        assert lot.ticker == "NVDA"
        assert lot.remaining_shares == 8.0
        assert lot.purchase_date == datetime.date(2023, 6, 15)

    def test_row_to_dividend(self):
        row = {
            "id": 1, "ticker": "AAPL", "amount": 0.96, "ex_date": datetime.date(2024, 2, 9),
            "pay_date": None, "frequency": "quarterly", "div_yield": 0.005,
            "payout_ratio": 0.15, "growth_rate": 0.05, "source": "yfinance",
        }
        div = _row_to_dividend(row)
        assert div.ticker == "AAPL"
        assert div.amount == 0.96
        assert div.ex_date == datetime.date(2024, 2, 9)

    def test_row_to_snapshot(self):
        row = {
            "id": 1, "snapshot_date": datetime.date(2024, 6, 1),
            "total_value": 150000.0, "equities_value": 120000.0,
            "crypto_value": 30000.0, "cash_value": 0.0,
            "position_count": 15, "notes": "",
        }
        snap = _row_to_snapshot(row)
        assert snap.total_value == 150000.0
        assert snap.position_count == 15

    def test_row_to_crypto(self):
        row = {
            "id": 1, "coin": "BTC", "quantity": 1.5, "cost_per_coin": 40000.0,
            "exchange": "Coinbase", "wallet_address": "", "notes": "",
        }
        cp = _row_to_crypto(row)
        assert cp.coin == "BTC"
        assert cp.quantity == 1.5
        assert cp.cost_basis == 60000.0

    def test_row_to_tax_lot_string_date(self):
        row = {
            "id": 2, "ticker": "AAPL", "shares": 50.0, "cost_per_share": 150.0,
            "purchase_date": "2023-01-15", "broker": "", "account_id": "",
            "sold_shares": 0.0, "notes": "", "position_id": None,
        }
        lot = _row_to_tax_lot(row)
        assert lot.purchase_date == datetime.date(2023, 1, 15)


# ═══════════════════════════════════════════════════════════════════════════════
# TaxLotStore Tests (mocked Postgres)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_pg():
    pg = MagicMock()
    pg.available = True
    pg.connect = AsyncMock()
    pg.close = AsyncMock()
    pg.fetchrow = AsyncMock(return_value={"id": 1})
    pg.fetch = AsyncMock(return_value=[])
    pg.execute = AsyncMock(return_value=1)
    return pg


@pytest.fixture
def tax_lot_store(mock_pg):
    from octane.portfolio.store import TaxLotStore
    s = TaxLotStore.__new__(TaxLotStore)
    s._pg = mock_pg
    return s


class TestTaxLotStore:
    @pytest.mark.asyncio
    async def test_add_lot_returns_id(self, tax_lot_store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value={"id": 99})
        lot = TaxLot(ticker="NVDA", shares=10, cost_per_share=650.0)
        result = await tax_lot_store.add_lot(lot)
        assert result == 99

    @pytest.mark.asyncio
    async def test_list_lots_empty(self, tax_lot_store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[])
        lots = await tax_lot_store.list_lots()
        assert lots == []

    @pytest.mark.asyncio
    async def test_sell_shares_fifo(self, tax_lot_store, mock_pg):
        """FIFO should sell oldest lot first."""
        mock_pg.fetch = AsyncMock(return_value=[
            {
                "id": 1, "ticker": "AAPL", "shares": 50.0, "cost_per_share": 100.0,
                "purchase_date": datetime.date(2022, 1, 1), "broker": "", "account_id": "",
                "sold_shares": 0.0, "notes": "", "position_id": None,
            },
            {
                "id": 2, "ticker": "AAPL", "shares": 50.0, "cost_per_share": 200.0,
                "purchase_date": datetime.date(2024, 1, 1), "broker": "", "account_id": "",
                "sold_shares": 0.0, "notes": "", "position_id": None,
            },
        ])
        allocs = await tax_lot_store.sell_shares("AAPL", 30, method="FIFO")
        assert len(allocs) == 1
        assert allocs[0]["lot_id"] == 1  # Oldest first
        assert allocs[0]["shares_sold"] == 30.0

    @pytest.mark.asyncio
    async def test_sell_shares_lifo(self, tax_lot_store, mock_pg):
        """LIFO should sell newest lot first."""
        mock_pg.fetch = AsyncMock(return_value=[
            {
                "id": 1, "ticker": "AAPL", "shares": 50.0, "cost_per_share": 100.0,
                "purchase_date": datetime.date(2022, 1, 1), "broker": "", "account_id": "",
                "sold_shares": 0.0, "notes": "", "position_id": None,
            },
            {
                "id": 2, "ticker": "AAPL", "shares": 50.0, "cost_per_share": 200.0,
                "purchase_date": datetime.date(2024, 1, 1), "broker": "", "account_id": "",
                "sold_shares": 0.0, "notes": "", "position_id": None,
            },
        ])
        allocs = await tax_lot_store.sell_shares("AAPL", 30, method="LIFO")
        assert len(allocs) == 1
        assert allocs[0]["lot_id"] == 2  # Newest first
        assert allocs[0]["shares_sold"] == 30.0

    @pytest.mark.asyncio
    async def test_sell_spans_multiple_lots(self, tax_lot_store, mock_pg):
        """Selling more than one lot's worth should span multiple lots."""
        mock_pg.fetch = AsyncMock(return_value=[
            {
                "id": 1, "ticker": "AAPL", "shares": 20.0, "cost_per_share": 100.0,
                "purchase_date": datetime.date(2022, 1, 1), "broker": "", "account_id": "",
                "sold_shares": 0.0, "notes": "", "position_id": None,
            },
            {
                "id": 2, "ticker": "AAPL", "shares": 30.0, "cost_per_share": 200.0,
                "purchase_date": datetime.date(2023, 1, 1), "broker": "", "account_id": "",
                "sold_shares": 0.0, "notes": "", "position_id": None,
            },
        ])
        allocs = await tax_lot_store.sell_shares("AAPL", 40, method="FIFO")
        assert len(allocs) == 2
        assert allocs[0]["shares_sold"] == 20.0  # Entire first lot
        assert allocs[1]["shares_sold"] == 20.0  # Partial second lot

    @pytest.mark.asyncio
    async def test_record_sale_calls_execute(self, tax_lot_store, mock_pg):
        await tax_lot_store.record_sale(1, 10.0)
        mock_pg.execute.assert_called_once()
