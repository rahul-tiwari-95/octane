"""Broker CSV parsers for portfolio import.

Each broker exports a slightly different CSV format.  This module auto-detects
the broker by inspecting column headers and routes to the right parser.

Supported brokers:
    Schwab    — "Symbol", "Quantity", "Price", "Cost Basis Per Share"
    Fidelity  — "Symbol", "Quantity", "Last Price", "Average Cost Basis"
    Vanguard  — "Ticker Symbol", "Shares", "Share Price", "Average Cost"
    IBKR      — "Symbol", "Position", "Cost Basis Price"
    Robinhood — "Symbol", "Quantity", "Average Cost"
    Webull    — "Ticker", "Total Qty", "Avg Cost"
    ETRADE    — "Symbol", "Quantity", "Price Paid"
    Generic   — fallback: looks for any column spellings of ticker/qty/cost

Usage::

    from octane.portfolio.parsers import parse_csv
    positions = parse_csv("/path/to/export.csv")

Each returned Position has quantity, avg_cost, and broker name filled in.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Callable

import structlog

from octane.portfolio.models import Position

logger = structlog.get_logger().bind(component="portfolio.parsers")

# ── Format detection ─────────────────────────────────────────────────────────

# Maps broker name → tuple of required header substrings (case-insensitive)
_FORMAT_SIGNATURES: dict[str, tuple[str, ...]] = {
    "Fidelity":  ("symbol", "quantity", "average cost basis"),
    "Schwab":    ("symbol", "quantity", "cost basis per share"),
    "Vanguard":  ("ticker symbol", "shares"),
    "IBKR":      ("symbol", "position", "cost basis price"),
    "Robinhood": ("symbol", "quantity", "average cost"),
    "Webull":    ("ticker", "total qty", "avg cost"),
    "ETRADE":    ("symbol", "quantity", "price paid"),
}


def _normalise_headers(raw: list[str]) -> list[str]:
    return [h.strip().lower() for h in raw]


def detect_broker(headers: list[str]) -> str:
    """Return broker name or 'Generic' if unrecognised."""
    norm = _normalise_headers(headers)
    header_str = " | ".join(norm)

    for broker, sigs in _FORMAT_SIGNATURES.items():
        if all(s in header_str for s in sigs):
            return broker

    return "Generic"


# ── Per-broker parsers ────────────────────────────────────────────────────────

def _clean_num(val: str) -> float:
    """Strip $, commas, whitespace and return float.  Returns 0.0 on failure."""
    cleaned = re.sub(r"[$,\s]", "", val or "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def _col(row: dict, *candidates: str) -> str:
    """Return the first matching column value (case-insensitive)."""
    norm = {k.lower(): v for k, v in row.items()}
    for c in candidates:
        v = norm.get(c.lower())
        if v is not None:
            return str(v).strip()
    return ""


def _parse_schwab(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Symbol")
        if not ticker or ticker.startswith("--") or ticker.lower() in ("symbol", "total"):
            continue
        qty    = _clean_num(_col(row, "Quantity"))
        cost   = _clean_num(_col(row, "Cost Basis Per Share", "Average Cost", "Price"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="Schwab"))
    return positions


def _parse_fidelity(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Symbol")
        if not ticker or ticker.startswith("Account") or ticker.lower() in ("symbol",):
            continue
        qty  = _clean_num(_col(row, "Quantity"))
        cost = _clean_num(_col(row, "Average Cost Basis", "Cost Basis Per Share", "Last Price"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="Fidelity"))
    return positions


def _parse_vanguard(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Ticker Symbol", "Ticker")
        if not ticker or ticker.startswith("--"):
            continue
        qty  = _clean_num(_col(row, "Shares", "Quantity"))
        cost = _clean_num(_col(row, "Average Cost", "Share Price"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="Vanguard"))
    return positions


def _parse_ibkr(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Symbol")
        if not ticker or ticker.lower() in ("symbol",):
            continue
        qty  = _clean_num(_col(row, "Position", "Quantity"))
        cost = _clean_num(_col(row, "Cost Basis Price", "Average Cost"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="IBKR"))
    return positions


def _parse_robinhood(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Symbol")
        if not ticker or ticker.lower() in ("symbol",):
            continue
        qty  = _clean_num(_col(row, "Quantity"))
        cost = _clean_num(_col(row, "Average Cost", "Average Buy Price"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="Robinhood"))
    return positions


def _parse_webull(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Ticker", "Symbol")
        if not ticker:
            continue
        qty  = _clean_num(_col(row, "Total Qty", "Quantity"))
        cost = _clean_num(_col(row, "Avg Cost", "Average Cost"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="Webull"))
    return positions


def _parse_etrade(rows: list[dict]) -> list[Position]:
    positions = []
    for row in rows:
        ticker = _col(row, "Symbol")
        if not ticker or ticker.lower() in ("symbol",):
            continue
        qty  = _clean_num(_col(row, "Quantity"))
        cost = _clean_num(_col(row, "Price Paid", "Average Cost"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="ETRADE"))
    return positions


def _parse_generic(rows: list[dict]) -> list[Position]:
    """Fallback: look for plausibly-named columns."""
    positions = []
    for row in rows:
        ticker = _col(row, "ticker", "symbol", "stock", "security")
        if not ticker:
            continue
        qty    = _clean_num(_col(row, "quantity", "qty", "shares", "position"))
        cost   = _clean_num(_col(row, "avg_cost", "average cost", "cost", "price", "unit cost"))
        if qty == 0:
            continue
        positions.append(Position(ticker=ticker, quantity=qty, avg_cost=cost, broker="Generic"))
    return positions


_PARSERS: dict[str, Callable[[list[dict]], list[Position]]] = {
    "Schwab":    _parse_schwab,
    "Fidelity":  _parse_fidelity,
    "Vanguard":  _parse_vanguard,
    "IBKR":      _parse_ibkr,
    "Robinhood": _parse_robinhood,
    "Webull":    _parse_webull,
    "ETRADE":    _parse_etrade,
    "Generic":   _parse_generic,
}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_csv(
    path: str | Path,
    broker: str | None = None,
    account_id: str = "",
) -> list[Position]:
    """Parse a broker CSV export and return a list of Position objects.

    Args:
        path:       Path to the CSV file.
        broker:     Override broker detection (uses auto-detect if None).
        account_id: Optional account identifier to tag all positions with.

    Returns:
        List of Position objects.  Empty list if no valid rows found.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file cannot be parsed as CSV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Portfolio CSV not found: {p}")

    raw = p.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM (Excel exports)
    reader = csv.DictReader(io.StringIO(raw))

    if reader.fieldnames is None:
        raise ValueError(f"CSV has no headers: {p}")

    rows = list(reader)
    detected = broker or detect_broker(list(reader.fieldnames))
    parser = _PARSERS.get(detected, _parse_generic)

    positions = parser(rows)

    if account_id:
        for pos in positions:
            pos.account_id = account_id

    logger.info(
        "portfolio_csv_parsed",
        broker=detected,
        file=str(p.name),
        positions=len(positions),
    )
    return positions


def parse_csv_text(
    text: str,
    broker: str | None = None,
    account_id: str = "",
) -> list[Position]:
    """Parse CSV from a string (useful for tests or in-memory data)."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return []

    rows = list(reader)
    detected = broker or detect_broker(list(reader.fieldnames))
    parser = _PARSERS.get(detected, _parse_generic)
    positions = parser(rows)

    if account_id:
        for pos in positions:
            pos.account_id = account_id

    return positions
