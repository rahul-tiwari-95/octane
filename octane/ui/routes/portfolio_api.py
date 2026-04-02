"""octane/ui/routes/portfolio_api.py — Portfolio chart data API.

Endpoints that power the Mission Control Portfolio page.
All data comes from Postgres (portfolio_positions, net_worth_snapshots,
dividends) and optional yfinance lookups for live prices.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("octane.ui.portfolio_api")

router = APIRouter(tags=["portfolio"])


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_positions() -> list[dict]:
    """Fetch all portfolio positions from Postgres."""
    try:
        from octane.tools.pg_client import PgClient
        from octane.portfolio.store import PortfolioStore

        pg = PgClient()
        store = PortfolioStore(pg)
        positions = await store.list_positions()
        return [p.to_dict() for p in positions]
    except Exception as exc:
        logger.warning("Cannot fetch positions: %s", exc)
        return []


async def _enrich_with_prices(positions: list[dict]) -> list[dict]:
    """Add current_price, market_value, pnl to positions via yfinance."""
    if not positions:
        return positions

    try:
        import yfinance as yf

        tickers = list({p["ticker"] for p in positions if p.get("ticker")})
        if not tickers:
            return positions

        data = yf.download(tickers, period="1d", progress=False)
        if data.empty:
            return positions

        price_map: dict[str, float] = {}
        if len(tickers) == 1:
            close_col = data.get("Close")
            if close_col is not None and not close_col.empty:
                price_map[tickers[0]] = float(close_col.iloc[-1])
        else:
            close_data = data.get("Close")
            if close_data is not None:
                for t in tickers:
                    if t in close_data.columns:
                        val = close_data[t].dropna()
                        if not val.empty:
                            price_map[t] = float(val.iloc[-1])

        for p in positions:
            price = price_map.get(p["ticker"])
            if price is not None:
                p["current_price"] = round(price, 2)
                qty = p.get("quantity", 0)
                avg = p.get("avg_cost", 0)
                p["market_value"] = round(qty * price, 2)
                p["cost_basis"] = round(qty * avg, 2)
                p["pnl"] = round(qty * (price - avg), 2)
                p["pnl_pct"] = round((price - avg) / avg * 100, 2) if avg else 0
    except Exception as exc:
        logger.warning("yfinance enrichment failed: %s", exc)

    return positions


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/portfolio/positions")
async def get_positions():
    """All portfolio positions with optional live price enrichment."""
    positions = await _get_positions()
    positions = await _enrich_with_prices(positions)

    total_value = sum(p.get("market_value", 0) for p in positions)
    total_cost = sum(p.get("cost_basis", 0) for p in positions)

    # Compute weights
    for p in positions:
        mv = p.get("market_value", 0)
        p["weight"] = round(mv / total_value * 100, 2) if total_value else 0

    return {
        "positions": positions,
        "total_market_value": round(total_value, 2),
        "total_cost_basis": round(total_cost, 2),
        "total_pnl": round(total_value - total_cost, 2),
        "count": len(positions),
    }


@router.get("/portfolio/allocation")
async def get_allocation():
    """Allocation data for donut chart — ticker → weight."""
    positions = await _get_positions()
    positions = await _enrich_with_prices(positions)

    total = sum(p.get("market_value", p.get("cost_basis", 0)) for p in positions)
    if total == 0:
        return {"allocations": [], "total": 0}

    allocations = []
    for p in positions:
        mv = p.get("market_value", p.get("cost_basis", 0))
        allocations.append({
            "ticker": p["ticker"],
            "value": round(mv, 2),
            "weight": round(mv / total * 100, 2),
            "sector": p.get("sector", "Unknown"),
        })

    allocations.sort(key=lambda x: x["value"], reverse=True)
    return {"allocations": allocations, "total": round(total, 2)}


@router.get("/portfolio/sectors")
async def get_sectors():
    """Sector exposure data for radar chart."""
    positions = await _get_positions()
    positions = await _enrich_with_prices(positions)

    total = sum(p.get("market_value", p.get("cost_basis", 0)) for p in positions)
    sector_map: dict[str, float] = {}

    for p in positions:
        sector = p.get("sector") or "Unknown"
        mv = p.get("market_value", p.get("cost_basis", 0))
        sector_map[sector] = sector_map.get(sector, 0) + mv

    sectors = [
        {
            "sector": s,
            "value": round(v, 2),
            "weight": round(v / total * 100, 2) if total else 0,
        }
        for s, v in sorted(sector_map.items(), key=lambda x: x[1], reverse=True)
    ]

    return {"sectors": sectors, "total": round(total, 2)}


@router.get("/portfolio/net-worth")
async def get_net_worth():
    """Net worth timeline for line chart."""
    try:
        from octane.tools.pg_client import PgClient
        from octane.portfolio.store import NetWorthStore

        pg = PgClient()
        store = NetWorthStore(pg)
        snapshots = await store.list_snapshots(limit=365)

        points = [
            {
                "date": s.date.isoformat() if hasattr(s.date, "isoformat") else str(s.date),
                "total": round(s.total, 2),
                "invested": round(s.invested, 2) if hasattr(s, "invested") else None,
            }
            for s in snapshots
        ]

        return {"snapshots": points, "count": len(points)}
    except Exception as exc:
        logger.warning("net_worth fetch failed: %s", exc)
        return {"snapshots": [], "count": 0}


@router.get("/portfolio/dividends")
async def get_dividends():
    """Dividend data for bar chart — monthly projected income."""
    try:
        from octane.tools.pg_client import PgClient
        from octane.portfolio.store import DividendStore

        pg = PgClient()
        store = DividendStore(pg)
        divs = await store.list_dividends()

        # Group by month
        monthly: dict[str, float] = {}
        for d in divs:
            key = f"{d.ex_date.year}-{d.ex_date.month:02d}" if hasattr(d.ex_date, "year") else str(d.ex_date)[:7]
            monthly[key] = monthly.get(key, 0) + (d.amount * d.shares if hasattr(d, "shares") else d.amount)

        bars = [
            {"month": m, "income": round(v, 2)}
            for m, v in sorted(monthly.items())
        ]

        return {"dividends": bars, "total_annual": round(sum(v for v in monthly.values()), 2)}
    except Exception as exc:
        logger.warning("dividends fetch failed: %s", exc)
        return {"dividends": [], "total_annual": 0}


@router.get("/portfolio/correlation")
async def get_correlation():
    """Correlation matrix for heatmap."""
    positions = await _get_positions()
    tickers = list({p["ticker"] for p in positions})[:10]

    if len(tickers) < 2:
        return {"tickers": tickers, "matrix": {}, "message": "Need ≥2 tickers"}

    try:
        import yfinance as yf
        import numpy as np

        data = yf.download(tickers, period="1y", progress=False)
        if data.empty:
            return {"tickers": tickers, "matrix": {}}

        close = data["Close"] if len(tickers) > 1 else data[["Close"]]
        returns = close.pct_change().dropna()

        if returns.empty or len(returns) < 5:
            return {"tickers": tickers, "matrix": {}}

        corr = returns.corr()
        matrix: dict[str, dict[str, float]] = {}
        for t1 in corr.index:
            matrix[str(t1)] = {}
            for t2 in corr.columns:
                matrix[str(t1)][str(t2)] = round(float(corr.loc[t1, t2]), 4)

        return {"tickers": [str(t) for t in corr.index], "matrix": matrix}
    except Exception as exc:
        logger.warning("correlation fetch failed: %s", exc)
        return {"tickers": tickers, "matrix": {}}
