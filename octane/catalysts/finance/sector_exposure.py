"""Catalyst F7 — sector_exposure.

Resolves the sector and industry for a ticker using yfinance and returns
a structured breakdown.  When multiple tickers are in resolved_data, it
aggregates them into a sector allocation map.

Input (resolved_data keys):
    ticker / symbol : str — single ticker
    tickers         : list[str] — optional multi-ticker mode

Output dict:
    ticker          : str
    sector          : str
    industry        : str
    sector_map      : dict[sector, list[ticker]] — multi-ticker breakdown
    summary         : str
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.sector_exposure")


def sector_exposure(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Retrieve sector + industry classification for one or more tickers."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("sector_exposure catalyst requires yfinance") from exc

    # Collect tickers to analyse
    tickers = _extract_tickers(resolved_data, instruction)
    if not tickers:
        raise ValueError("sector_exposure: no tickers found in resolved data")

    results: list[dict] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            results.append({
                "ticker":   ticker,
                "sector":   info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "beta":     info.get("beta"),
            })
        except Exception as exc:
            logger.warning("sector_exposure_fetch_failed", ticker=ticker, error=str(exc))
            results.append({"ticker": ticker, "sector": "Unknown", "industry": "Unknown"})

    # Build sector map
    sector_map: dict[str, list[str]] = {}
    for r in results:
        sector_map.setdefault(r["sector"], []).append(r["ticker"])

    # Build human-readable summary
    lines: list[str] = []
    if len(results) == 1:
        r = results[0]
        lines.append(f"SECTOR EXPOSURE — {r['ticker']}")
        lines.append(f"  Sector:   {r['sector']}")
        lines.append(f"  Industry: {r['industry']}")
        if r.get("beta") is not None:
            lines.append(f"  Beta:     {r['beta']:.2f}")
    else:
        lines.append(f"SECTOR EXPOSURE — {len(results)} TICKERS")
        for sector, ts in sorted(sector_map.items()):
            lines.append(f"  {sector:30s} [{', '.join(ts)}]")

    summary = "\n".join(lines)
    primary = results[0] if results else {}

    logger.info("sector_exposure_complete", tickers=tickers, sectors=list(sector_map.keys()))

    return {
        "ticker":     primary.get("ticker", ""),
        "sector":     primary.get("sector", "Unknown"),
        "industry":   primary.get("industry", "Unknown"),
        "beta":       primary.get("beta"),
        "sector_map": sector_map,
        "details":    results,
        "summary":    summary,
    }


def _extract_tickers(resolved_data: dict[str, Any], instruction: str) -> list[str]:
    # Explicit list
    ts = resolved_data.get("tickers")
    if isinstance(ts, list) and ts:
        return [str(t).upper() for t in ts]

    # Single ticker keys
    for key in ("ticker", "symbol", "Ticker", "Symbol"):
        val = resolved_data.get(key)
        if val and isinstance(val, str):
            return [val.upper()]

    # Nested
    for v in resolved_data.values():
        if isinstance(v, dict):
            for key in ("ticker", "symbol"):
                val = v.get(key)
                if val and isinstance(val, str):
                    return [val.upper()]

    # Parse instruction: extract all plausible ticker words (2–5 uppercase letters)
    import re
    found = re.findall(r"\b([A-Z]{1,5})\b", instruction)
    # Filter common words that aren't tickers
    _STOP = {"AND", "OR", "FOR", "THE", "IN", "ON", "AT", "BY", "VS", "IS", "A"}
    found = [t for t in found if t not in _STOP]
    return found[:5] if found else []
