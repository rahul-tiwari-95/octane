"""Catalyst F8 — correlation_analysis.

Computes a pairwise correlation matrix from the time-series data already
fetched by the finance agent.  Works on `time_series` lists of OHLCV dicts.
No network calls — pure computation on what the upstream agent already returned.

When only one ticker's time-series is present, fetches additional tickers
from the instruction text via yfinance.

Input (resolved_data keys):
    time_series     : list[dict] with 'timestamp' and 'close'
    ticker / symbol : str — primary ticker
    extra_series    : dict[ticker, list[dict]] — optional additional series

Output dict:
    tickers         : list[str]
    matrix          : dict[ticker, dict[ticker, float]] — correlation matrix
    summary         : str
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.correlation_analysis")


def correlation_analysis(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Compute pairwise return correlations for the tickers in context."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("correlation_analysis catalyst requires numpy") from exc

    primary_ticker = _get_ticker(resolved_data, instruction)
    primary_series = resolved_data.get("time_series", [])

    # Build {ticker: list[float]} return series map
    series_map: dict[str, list[float]] = {}

    if primary_series and primary_ticker:
        closes = _extract_closes(primary_series)
        if closes:
            series_map[primary_ticker] = _pct_returns(closes)

    # Merge any `extra_series` already in resolved_data
    extra = resolved_data.get("extra_series")
    if isinstance(extra, dict):
        for t, ts in extra.items():
            closes = _extract_closes(ts)
            if closes:
                series_map[t.upper()] = _pct_returns(closes)

    # If we only have one or zero series, fetch more from instruction / yfinance
    tickers_in_instruction = _parse_instruction_tickers(instruction)
    if len(series_map) < 2 and tickers_in_instruction:
        _fetch_additional(series_map, tickers_in_instruction, primary_ticker)

    if len(series_map) < 2:
        # Not enough data — return graceful fallback
        logger.warning("correlation_analysis_insufficient_data", tickers=list(series_map.keys()))
        return {
            "tickers": list(series_map.keys()),
            "matrix":  {},
            "summary": "CORRELATION: insufficient data (need ≥2 tickers with price history)",
        }

    # Align series lengths
    min_len = min(len(v) for v in series_map.values())
    aligned  = {t: v[-min_len:] for t, v in series_map.items()}
    tickers  = sorted(aligned.keys())

    # Build numpy matrix
    mat = np.array([aligned[t] for t in tickers])
    corr = np.corrcoef(mat)

    # Convert to nested dict
    matrix: dict[str, dict[str, float]] = {}
    for i, t1 in enumerate(tickers):
        matrix[t1] = {}
        for j, t2 in enumerate(tickers):
            matrix[t1][t2] = round(float(corr[i, j]), 4)

    # Human-readable summary
    lines = [f"CORRELATION MATRIX — {', '.join(tickers)}  (period: {min_len} trading days)"]
    for t1 in tickers:
        for t2 in tickers:
            if t1 >= t2:
                continue
            val = matrix[t1][t2]
            label = _corr_label(val)
            lines.append(f"  {t1} / {t2}: {val:+.4f}  [{label}]")

    summary = "\n".join(lines)
    logger.info("correlation_analysis_complete", tickers=tickers, points=min_len)

    return {
        "tickers": tickers,
        "matrix":  matrix,
        "summary": summary,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_closes(series: list[dict]) -> list[float]:
    closes = []
    for row in series:
        val = row.get("close") or row.get("Close") or row.get("adjClose")
        if val is not None:
            try:
                closes.append(float(val))
            except (TypeError, ValueError):
                pass
    return closes


def _pct_returns(closes: list[float]) -> list[float]:
    if len(closes) < 2:
        return []
    return [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]


def _get_ticker(resolved_data: dict[str, Any], instruction: str) -> str:
    for key in ("ticker", "symbol", "Ticker", "Symbol"):
        val = resolved_data.get(key)
        if val and isinstance(val, str):
            return val.upper()
    import re
    match = re.search(r"\b([A-Z]{1,5})\b", instruction)
    return match.group(1) if match else ""


def _parse_instruction_tickers(instruction: str) -> list[str]:
    import re
    _STOP = {"AND", "OR", "FOR", "THE", "IN", "ON", "AT", "VS", "COMPARE", "CORRELATION"}
    found = re.findall(r"\b([A-Z]{1,5})\b", instruction)
    return [t for t in found if t not in _STOP][:6]


def _fetch_additional(
    series_map: dict[str, list[float]],
    tickers: list[str],
    primary: str,
) -> None:
    """Fetch 1-year daily closes from yfinance for tickers not yet in series_map."""
    try:
        import yfinance as yf
    except ImportError:
        return

    for ticker in tickers:
        if ticker in series_map or ticker == primary:
            continue
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty:
                continue
            closes = [float(c) for c in hist["Close"].tolist()]
            rets   = _pct_returns(closes)
            if rets:
                series_map[ticker] = rets
        except Exception as exc:
            logger.warning("correlation_yfinance_fetch_failed", ticker=ticker, error=str(exc))


def _corr_label(val: float) -> str:
    abs_val = abs(val)
    if abs_val >= 0.9:
        return "very high"
    if abs_val >= 0.7:
        return "high"
    if abs_val >= 0.5:
        return "moderate"
    if abs_val >= 0.3:
        return "low"
    return "negligible"
