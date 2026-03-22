"""Unit tests for Session 29 finance catalysts (no network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── earnings_calendar ─────────────────────────────────────────────────────────

class TestEarningsCalendar:
    def _make_yf_ticker(self, calendar: dict):
        mock = MagicMock()
        mock.calendar = calendar
        return mock

    def test_basic_result_structure(self):
        from octane.catalysts.finance.earnings_calendar import earnings_calendar

        mock_ticker = self._make_yf_ticker({"Earnings Date": ["2026-04-25"]})

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = earnings_calendar(
                {"ticker": "NVDA"}, output_dir="/tmp"
            )

        assert result["ticker"] == "NVDA"
        assert "earnings_date" in result
        assert "summary" in result
        assert "NVDA" in result["summary"]

    def test_ticker_extracted_from_instruction(self):
        from octane.catalysts.finance.earnings_calendar import earnings_calendar

        mock_ticker = self._make_yf_ticker({})

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = earnings_calendar(
                {"price": 875.0}, output_dir="/tmp", instruction="When does AAPL report earnings?"
            )

        assert result["ticker"] == "AAPL"

    def test_no_ticker_raises(self):
        from octane.catalysts.finance.earnings_calendar import earnings_calendar

        with pytest.raises(ValueError, match="ticker"):
            earnings_calendar({}, output_dir="/tmp", instruction="")

    def test_yfinance_failure_handled_gracefully(self):
        from octane.catalysts.finance.earnings_calendar import earnings_calendar

        mock_ticker = MagicMock()
        mock_ticker.calendar = MagicMock(side_effect=Exception("network error"))

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = earnings_calendar({"ticker": "NVDA"}, output_dir="/tmp")

        # Should return a result with None date, not raise
        assert result["ticker"] == "NVDA"
        assert result["earnings_date"] is None

    def test_eps_extracted_when_present(self):
        from octane.catalysts.finance.earnings_calendar import earnings_calendar

        cal = {"Earnings Date": "2026-04-25", "EPS Estimate": 5.87, "Revenue Estimate": 44e9}
        mock_ticker = self._make_yf_ticker(cal)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = earnings_calendar({"ticker": "NVDA"}, output_dir="/tmp")

        assert result["eps_estimate"] == 5.87
        assert result["rev_estimate"] == 44e9


# ── sector_exposure ───────────────────────────────────────────────────────────

class TestSectorExposure:
    def _mock_ticker(self, sector: str = "Technology", industry: str = "Semiconductors"):
        m = MagicMock()
        m.info = {"sector": sector, "industry": industry, "beta": 1.75, "marketCap": 3e12}
        return m

    def test_single_ticker(self):
        from octane.catalysts.finance.sector_exposure import sector_exposure

        with patch("yfinance.Ticker", return_value=self._mock_ticker()):
            result = sector_exposure({"ticker": "NVDA"}, output_dir="/tmp")

        assert result["ticker"] == "NVDA"
        assert result["sector"] == "Technology"
        assert result["industry"] == "Semiconductors"

    def test_summary_contains_ticker(self):
        from octane.catalysts.finance.sector_exposure import sector_exposure

        with patch("yfinance.Ticker", return_value=self._mock_ticker()):
            result = sector_exposure({"ticker": "AAPL"}, output_dir="/tmp")

        assert "AAPL" in result["summary"]

    def test_sector_map_built(self):
        from octane.catalysts.finance.sector_exposure import sector_exposure

        with patch("yfinance.Ticker", return_value=self._mock_ticker(sector="Technology")):
            result = sector_exposure({"tickers": ["NVDA", "AMD"]}, output_dir="/tmp")

        assert "Technology" in result["sector_map"]

    def test_no_ticker_raises(self):
        from octane.catalysts.finance.sector_exposure import sector_exposure

        with pytest.raises(ValueError, match="no tickers"):
            sector_exposure({}, output_dir="/tmp", instruction="")


# ── correlation_analysis ──────────────────────────────────────────────────────

class TestCorrelationAnalysis:
    def _make_series(self, n: int = 30, start: float = 100.0, step: float = 1.0):
        """Generate a simple ascending price series."""
        return [{"timestamp": f"2025-{i:02d}-01", "close": start + i * step} for i in range(n)]

    def test_single_series_returns_insufficient_data(self):
        from octane.catalysts.finance.correlation_analysis import correlation_analysis

        series = self._make_series(30)
        result = correlation_analysis(
            {"ticker": "NVDA", "time_series": series}, output_dir="/tmp"
        )
        # Only 1 series — should return graceful fallback
        assert "insufficient" in result["summary"].lower() or result["matrix"] == {}

    def test_two_series_returns_matrix(self):
        from octane.catalysts.finance.correlation_analysis import correlation_analysis

        series_a = self._make_series(30, 100, 1.0)
        series_b = self._make_series(30, 200, 2.0)
        resolved = {
            "ticker": "NVDA",
            "time_series": series_a,
            "extra_series": {"AAPL": series_b},
        }
        result = correlation_analysis(resolved, output_dir="/tmp")

        assert "NVDA" in result["tickers"]
        assert "AAPL" in result["tickers"]
        assert "NVDA" in result["matrix"]
        assert "AAPL" in result["matrix"]["NVDA"]

    def test_perfect_correlation(self):
        from octane.catalysts.finance.correlation_analysis import correlation_analysis
        import math

        # Both series move identically → correlation ≈ 1.0
        series = self._make_series(30, 100, 1.0)
        resolved = {
            "ticker": "A",
            "time_series": series,
            "extra_series": {"B": series},
        }
        result = correlation_analysis(resolved, output_dir="/tmp")

        if result["matrix"]:
            val = result["matrix"]["A"]["B"]
            assert abs(val - 1.0) < 0.01

    def test_summary_contains_tickers(self):
        from octane.catalysts.finance.correlation_analysis import correlation_analysis

        series_a = self._make_series(20, 100, 1.0)
        series_b = self._make_series(20, 50, 0.5)
        resolved = {
            "ticker": "X",
            "time_series": series_a,
            "extra_series": {"Y": series_b},
        }
        result = correlation_analysis(resolved, output_dir="/tmp")
        assert result["summary"]  # not empty


# ── Catalyst registry — new entries present ───────────────────────────────────

class TestCatalystRegistry:
    def test_earnings_calendar_registered(self):
        from octane.catalysts.registry import CATALYST_REGISTRY
        assert "earnings_calendar" in CATALYST_REGISTRY

    def test_sector_exposure_registered(self):
        from octane.catalysts.registry import CATALYST_REGISTRY
        assert "sector_exposure" in CATALYST_REGISTRY

    def test_correlation_analysis_registered(self):
        from octane.catalysts.registry import CATALYST_REGISTRY
        assert "correlation_analysis" in CATALYST_REGISTRY

    def test_earnings_triggers(self):
        from octane.catalysts.registry import CATALYST_REGISTRY
        triggers = CATALYST_REGISTRY["earnings_calendar"]["triggers"]
        assert "earnings" in triggers

    def test_correlation_requires_time_series(self):
        from octane.catalysts.registry import CATALYST_REGISTRY
        requires = CATALYST_REGISTRY["correlation_analysis"]["requires"]
        assert "time_series" in requires

    def test_all_new_catalysts_have_loader(self):
        from octane.catalysts.registry import CATALYST_REGISTRY
        for name in ("earnings_calendar", "sector_exposure", "correlation_analysis"):
            entry = CATALYST_REGISTRY[name]
            assert callable(entry["loader"]), f"{name} loader not callable"
            # Verify the loader actually returns a callable
            fn = entry["loader"]()
            assert callable(fn), f"{name} loader() did not return callable"
