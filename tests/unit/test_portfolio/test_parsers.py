"""Unit tests for broker CSV parsers (no Postgres, no network)."""

from __future__ import annotations

import pytest
from octane.portfolio.parsers import (
    detect_broker,
    parse_csv_text,
    _clean_num,
    _col,
)
from octane.portfolio.models import Position


# ── _clean_num ────────────────────────────────────────────────────────────────

class TestCleanNum:
    def test_plain_float(self):
        assert _clean_num("123.45") == 123.45

    def test_dollar_sign(self):
        assert _clean_num("$1,234.56") == 1234.56

    def test_commas(self):
        assert _clean_num("10,000") == 10000.0

    def test_empty_string(self):
        assert _clean_num("") == 0.0

    def test_none_value(self):
        assert _clean_num(None) == 0.0  # type: ignore[arg-type]

    def test_text_value(self):
        assert _clean_num("N/A") == 0.0


# ── detect_broker ─────────────────────────────────────────────────────────────

class TestDetectBroker:
    def test_schwab(self):
        assert detect_broker(["Symbol", "Quantity", "Price", "Cost Basis Per Share"]) == "Schwab"

    def test_fidelity(self):
        assert detect_broker(["Symbol", "Quantity", "Last Price", "Average Cost Basis"]) == "Fidelity"

    def test_vanguard(self):
        assert detect_broker(["Ticker Symbol", "Shares", "Share Price"]) == "Vanguard"

    def test_robinhood(self):
        assert detect_broker(["Symbol", "Quantity", "Average Cost"]) == "Robinhood"

    def test_webull(self):
        assert detect_broker(["Ticker", "Total Qty", "Avg Cost"]) == "Webull"

    def test_ibkr(self):
        assert detect_broker(["Symbol", "Position", "Cost Basis Price"]) == "IBKR"

    def test_unknown_falls_back_to_generic(self):
        assert detect_broker(["Name", "Shares", "Value"]) == "Generic"


# ── parse_csv_text — Schwab ───────────────────────────────────────────────────

SCHWAB_CSV = """\
Symbol,Quantity,Price,Cost Basis Per Share
NVDA,100,875.50,"$650.00"
AAPL,50,"$189.00","$155.00"
--Total,,,$0.00
"""

class TestSchwebParser:
    def test_parses_two_positions(self):
        positions = parse_csv_text(SCHWAB_CSV)
        assert len(positions) == 2

    def test_ticker_uppercase(self):
        positions = parse_csv_text(SCHWAB_CSV)
        assert positions[0].ticker == "NVDA"
        assert positions[1].ticker == "AAPL"

    def test_quantity(self):
        positions = parse_csv_text(SCHWAB_CSV)
        assert positions[0].quantity == 100.0
        assert positions[1].quantity == 50.0

    def test_avg_cost(self):
        positions = parse_csv_text(SCHWAB_CSV)
        assert positions[0].avg_cost == 650.0
        assert positions[1].avg_cost == 155.0

    def test_broker_label(self):
        positions = parse_csv_text(SCHWAB_CSV)
        assert all(p.broker == "Schwab" for p in positions)

    def test_total_row_skipped(self):
        positions = parse_csv_text(SCHWAB_CSV)
        tickers = {p.ticker for p in positions}
        assert "--TOTAL" not in tickers

    def test_account_id_passed_through(self):
        positions = parse_csv_text(SCHWAB_CSV, account_id="IRA-1234")
        assert all(p.account_id == "IRA-1234" for p in positions)


# ── Fidelity ─────────────────────────────────────────────────────────────────

FIDELITY_CSV = """\
Symbol,Description,Quantity,Last Price,Average Cost Basis,Current Value
MSFT,Microsoft Corp,25,$415.00,$320.00,$10375.00
TSLA,Tesla Inc,10,$175.00,$200.00,$1750.00
"""

class TestFidelityParser:
    def test_parses_correct_positions(self):
        positions = parse_csv_text(FIDELITY_CSV)
        assert len(positions) == 2

    def test_msft_cost(self):
        positions = parse_csv_text(FIDELITY_CSV)
        msft = next(p for p in positions if p.ticker == "MSFT")
        assert msft.avg_cost == 320.0

    def test_broker_is_fidelity(self):
        positions = parse_csv_text(FIDELITY_CSV)
        assert all(p.broker == "Fidelity" for p in positions)


# ── Robinhood ─────────────────────────────────────────────────────────────────

ROBINHOOD_CSV = """\
Symbol,Quantity,Average Cost,Equity
GOOGL,5,140.00,700.00
SPY,20,450.00,9000.00
"""

class TestRobinhoodParser:
    def test_two_positions(self):
        assert len(parse_csv_text(ROBINHOOD_CSV)) == 2

    def test_spy_qty(self):
        positions = parse_csv_text(ROBINHOOD_CSV)
        spy = next(p for p in positions if p.ticker == "SPY")
        assert spy.quantity == 20.0

    def test_broker_label(self):
        positions = parse_csv_text(ROBINHOOD_CSV)
        assert all(p.broker == "Robinhood" for p in positions)


# ── Vanguard ──────────────────────────────────────────────────────────────────

VANGUARD_CSV = """\
Ticker Symbol,Shares,Share Price,Average Cost,Market Value
VTI,100,225.00,200.00,22500.00
VXUS,50,55.00,48.00,2750.00
"""

class TestVanguardParser:
    def test_two_positions(self):
        assert len(parse_csv_text(VANGUARD_CSV)) == 2

    def test_vti_cost(self):
        positions = parse_csv_text(VANGUARD_CSV)
        vti = next(p for p in positions if p.ticker == "VTI")
        assert vti.avg_cost == 200.0
        assert vti.broker == "Vanguard"


# ── Generic fallback ──────────────────────────────────────────────────────────

GENERIC_CSV = """\
Ticker,Qty,Cost
AMZN,3,185.50
META,8,490.00
"""

class TestGenericParser:
    def test_fallback_parses(self):
        positions = parse_csv_text(GENERIC_CSV)
        assert len(positions) == 2

    def test_broker_is_generic(self):
        assert all(p.broker == "Generic" for p in parse_csv_text(GENERIC_CSV))


# ── Position model ────────────────────────────────────────────────────────────

class TestPositionModel:
    def test_ticker_normalised(self):
        p = Position(ticker="nvda", quantity=10, avg_cost=500.0)
        assert p.ticker == "NVDA"

    def test_cost_basis(self):
        p = Position(ticker="NVDA", quantity=10, avg_cost=500.0)
        assert p.cost_basis == 5000.0

    def test_to_dict_keys(self):
        p = Position(ticker="NVDA", quantity=10, avg_cost=500.0, broker="Schwab")
        d = p.to_dict()
        assert "ticker" in d and "quantity" in d and "broker" in d

    def test_portfolio_cost_basis(self):
        from octane.portfolio.models import Portfolio
        pf = Portfolio(positions=[
            Position("NVDA", 10, 500.0),
            Position("AAPL", 5, 200.0),
        ])
        assert pf.total_cost_basis == 6000.0

    def test_portfolio_tickers(self):
        from octane.portfolio.models import Portfolio
        pf = Portfolio(positions=[Position("NVDA", 1, 1.0), Position("AAPL", 1, 1.0)])
        assert set(pf.tickers) == {"NVDA", "AAPL"}
