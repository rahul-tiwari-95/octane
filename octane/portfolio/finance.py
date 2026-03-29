"""Financial calculations — XIRR, Sharpe ratio, dividend analysis, tax-loss harvesting.

Pure functions with no I/O deps so they can be tested without Postgres or network.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Optional


# ── XIRR (Extended Internal Rate of Return) ──────────────────────────────────

def xirr(
    cashflows: list[tuple[datetime.date, float]],
    guess: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> Optional[float]:
    """Compute XIRR using Newton's method.

    Args:
        cashflows: list of (date, amount) — negative = outflow, positive = inflow.
        guess: initial rate estimate.
        max_iter: max Newton iterations.
        tol: convergence tolerance.

    Returns:
        Annualised rate as a decimal (e.g. 0.12 = 12%), or None if no convergence.
    """
    if len(cashflows) < 2:
        return None

    # Sort by date
    flows = sorted(cashflows, key=lambda x: x[0])
    d0 = flows[0][0]

    def _npv(rate: float) -> float:
        return sum(
            amt / (1.0 + rate) ** ((d - d0).days / 365.0)
            for d, amt in flows
        )

    def _npv_deriv(rate: float) -> float:
        return sum(
            -((d - d0).days / 365.0) * amt / (1.0 + rate) ** ((d - d0).days / 365.0 + 1)
            for d, amt in flows
        )

    rate = guess
    for _ in range(max_iter):
        npv = _npv(rate)
        deriv = _npv_deriv(rate)
        if abs(deriv) < 1e-14:
            break
        new_rate = rate - npv / deriv
        if abs(new_rate - rate) < tol:
            return round(new_rate, 6)
        rate = new_rate

    # Fallback: if Newton didn't converge, try bisection
    return _xirr_bisect(flows, d0)


def _xirr_bisect(
    flows: list[tuple[datetime.date, float]],
    d0: datetime.date,
    lo: float = -0.99,
    hi: float = 10.0,
    tol: float = 1e-6,
    max_iter: int = 300,
) -> Optional[float]:
    """Bisection fallback for XIRR."""
    def _npv(rate: float) -> float:
        return sum(
            amt / (1.0 + rate) ** ((d - d0).days / 365.0)
            for d, amt in flows
        )

    f_lo = _npv(lo)
    f_hi = _npv(hi)
    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = _npv(mid)
        if abs(f_mid) < tol:
            return round(mid, 6)
        if f_mid * f_lo < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    return round((lo + hi) / 2.0, 6)


# ── Sharpe Ratio ─────────────────────────────────────────────────────────────

def sharpe_ratio(
    daily_returns: list[float],
    risk_free_annual: float = 0.05,
    trading_days: int = 252,
) -> Optional[float]:
    """Compute annualised Sharpe ratio from daily returns.

    Args:
        daily_returns: list of daily return fractions (e.g. 0.01 = 1%).
        risk_free_annual: annualised risk-free rate (default 5% ~ T-bills).
        trading_days: annualisation factor.

    Returns:
        Annualised Sharpe ratio, or None if insufficient data.
    """
    if len(daily_returns) < 2:
        return None

    rf_daily = (1 + risk_free_annual) ** (1 / trading_days) - 1
    excess = [r - rf_daily for r in daily_returns]

    mean_excess = sum(excess) / len(excess)
    variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev < 1e-14:
        return None

    return round((mean_excess / std_dev) * math.sqrt(trading_days), 4)


# ── Tax-Loss Harvesting ─────────────────────────────────────────────────────

@dataclass
class HarvestCandidate:
    """A position eligible for tax-loss harvesting."""
    ticker: str
    shares: float
    cost_basis: float
    current_value: float
    unrealised_loss: float
    loss_pct: float
    is_long_term: bool
    wash_sale_risk: bool  # True if sold the same ticker in last 30 days


def find_harvest_candidates(
    positions: list[dict],
    prices: dict[str, float],
    recent_sales: list[dict] | None = None,
    min_loss_pct: float = 5.0,
) -> list[HarvestCandidate]:
    """Scan positions for tax-loss harvesting opportunities.

    Args:
        positions: list of dicts with 'ticker', 'quantity', 'avg_cost' keys.
        prices: {ticker: current_price} map.
        recent_sales: list of dicts with 'ticker', 'sale_date' keys (for wash sale check).
        min_loss_pct: minimum loss % to consider for harvesting.

    Returns:
        List of HarvestCandidate sorted by largest absolute loss first.
    """
    if recent_sales is None:
        recent_sales = []

    today = datetime.date.today()
    wash_tickers: set[str] = set()
    for sale in recent_sales:
        sale_date = sale.get("sale_date")
        if isinstance(sale_date, str):
            sale_date = datetime.date.fromisoformat(sale_date)
        if sale_date and (today - sale_date).days <= 30:
            wash_tickers.add(sale["ticker"].upper())

    candidates: list[HarvestCandidate] = []
    for pos in positions:
        ticker = pos["ticker"].upper()
        qty = float(pos["quantity"])
        avg_cost = float(pos["avg_cost"])
        price = prices.get(ticker)
        if price is None or qty <= 0:
            continue

        cost_basis = qty * avg_cost
        current_value = qty * price
        loss = current_value - cost_basis

        if loss >= 0:
            continue  # No loss — skip

        loss_pct = abs(loss / cost_basis * 100) if cost_basis > 0 else 0.0
        if loss_pct < min_loss_pct:
            continue

        # Rough long-term check: use purchase_date if available
        purchase_date = pos.get("purchase_date")
        is_long = False
        if purchase_date:
            if isinstance(purchase_date, str):
                purchase_date = datetime.date.fromisoformat(purchase_date)
            is_long = (today - purchase_date).days > 365

        candidates.append(HarvestCandidate(
            ticker=ticker,
            shares=qty,
            cost_basis=round(cost_basis, 2),
            current_value=round(current_value, 2),
            unrealised_loss=round(loss, 2),
            loss_pct=round(loss_pct, 2),
            is_long_term=is_long,
            wash_sale_risk=ticker in wash_tickers,
        ))

    candidates.sort(key=lambda c: c.unrealised_loss)  # Most negative first
    return candidates


# ── Dividend Analytics ───────────────────────────────────────────────────────

def annual_dividend_income(
    holdings: list[dict],
    dividend_info: dict[str, dict],
) -> dict:
    """Estimate annual dividend income from current holdings.

    Args:
        holdings: list of dicts with 'ticker', 'quantity'.
        dividend_info: {ticker: {dividendRate, dividendYield, exDividendDate, ...}} from yfinance.

    Returns:
        Dict with per-ticker income and total.
    """
    breakdown: list[dict] = []
    total = 0.0

    for h in holdings:
        ticker = h["ticker"].upper()
        qty = float(h["quantity"])
        info = dividend_info.get(ticker, {})
        rate = float(info.get("dividendRate", 0) or 0)  # Annual dividend per share
        yld = float(info.get("dividendYield", 0) or 0)
        ex_date = info.get("exDividendDate")

        annual = round(rate * qty, 2)
        total += annual
        breakdown.append({
            "ticker": ticker,
            "shares": qty,
            "dividend_rate": rate,
            "dividend_yield": round(yld * 100, 2) if yld else 0.0,
            "annual_income": annual,
            "ex_date": ex_date,
        })

    breakdown.sort(key=lambda x: -x["annual_income"])
    return {"breakdown": breakdown, "total_annual_income": round(total, 2)}
