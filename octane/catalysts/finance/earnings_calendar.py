"""Catalyst F6 — earnings_calendar.

Fetches upcoming earnings dates and analyst estimates for one or more
tickers using yfinance.  Triggered by queries containing "earnings",
"report date", "when does X report", etc.

Input (resolved_data keys from finance agent):
    ticker / symbol : str — the equity to look up
    tickers         : list[str] — optional multi-ticker mode

Output dict:
    ticker         : str
    earnings_date  : str | None  (ISO format)
    eps_estimate   : float | None
    rev_estimate   : float | None
    summary        : str
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.earnings_calendar")


def earnings_calendar(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Fetch upcoming earnings calendar for the ticker in resolved_data."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("earnings_calendar catalyst requires yfinance") from exc

    ticker = _extract_ticker(resolved_data, instruction)
    if not ticker:
        raise ValueError("earnings_calendar: cannot determine ticker from resolved data")

    t = yf.Ticker(ticker)

    # yfinance 0.2+ stores earnings dates in `earnings_dates`
    # Fall back gracefully if the attribute or data is missing
    next_date: str | None = None
    eps_est: float | None = None
    rev_est: float | None = None

    try:
        cal = t.calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earnings_date")
            if raw:
                if hasattr(raw, "__iter__") and not isinstance(raw, str):
                    raw = list(raw)[0]
                next_date = str(raw)[:10]
            eps_raw = cal.get("EPS Estimate") or cal.get("eps_estimate")
            if eps_raw is not None:
                try:
                    eps_est = float(eps_raw)
                except (TypeError, ValueError):
                    pass
            rev_raw = cal.get("Revenue Estimate") or cal.get("revenue_estimate")
            if rev_raw is not None:
                try:
                    rev_est = float(rev_raw)
                except (TypeError, ValueError):
                    pass
    except Exception as exc:
        logger.warning("earnings_calendar_fetch_failed", ticker=ticker, error=str(exc))

    # Build summary
    lines = [f"EARNINGS CALENDAR — {ticker}"]
    if next_date:
        lines.append(f"  Next Report:      {next_date}")
    else:
        lines.append("  Next Report:      Not yet announced")
    if eps_est is not None:
        lines.append(f"  EPS Estimate:     ${eps_est:.2f}")
    if rev_est is not None:
        b = rev_est / 1e9
        lines.append(f"  Revenue Estimate: ${b:.2f}B")

    summary = "\n".join(lines)
    logger.info("earnings_calendar_complete", ticker=ticker, next_date=next_date)

    return {
        "ticker":        ticker,
        "earnings_date": next_date,
        "eps_estimate":  eps_est,
        "rev_estimate":  rev_est,
        "summary":       summary,
    }


def _extract_ticker(resolved_data: dict[str, Any], instruction: str) -> str:
    # Direct keys
    for key in ("ticker", "symbol", "Ticker", "Symbol"):
        val = resolved_data.get(key)
        if val and isinstance(val, str):
            return val.upper()

    # Scan nested dicts (e.g. market_data sub-dict)
    for v in resolved_data.values():
        if isinstance(v, dict):
            for key in ("ticker", "symbol"):
                val = v.get(key)
                if val and isinstance(val, str):
                    return val.upper()

    # Last resort: pick first ALL-CAPS word from instruction
    import re
    match = re.search(r"\b([A-Z]{1,5})\b", instruction)
    return match.group(1) if match else ""
