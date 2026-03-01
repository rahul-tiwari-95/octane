"""Catalyst F2 — return_calculator.

Computes absolute and percentage return given a buy price, quantity,
and current market price. Accepts a lump-sum initial investment
as an alternative to (buy_price × quantity).

Input (resolved_data keys from finance agent):
    price / regularMarketPrice  : current price (float)
    ticker / symbol             : str (optional)

The instruction text is parsed for buy_price, quantity, and initial investment.

Output dict:
    ticker          : str
    buy_price       : float
    current_price   : float
    quantity        : float
    initial_value   : float
    current_value   : float
    absolute_return : float
    pct_return      : float
    summary         : str
"""

from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.return_calculator")

# Patterns for extracting numbers from the instruction text
_PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"\b([\d,]+(?:\.\d+)?)\b")


def return_calculator(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Calculate return on investment from upstream market data + instruction parameters."""

    current_price = _extract_current_price(resolved_data)
    if current_price is None:
        raise ValueError("return_calculator: no current price found in upstream data")

    ticker = _extract_ticker(resolved_data)

    # Parse parameters from the instruction text
    buy_price, quantity, initial_investment = _parse_instruction(instruction, current_price)

    if initial_investment and not buy_price:
        # e.g. "I invested $500 in NVDA" — infer quantity
        quantity = initial_investment / buy_price if buy_price else initial_investment / current_price
        buy_price = buy_price or current_price  # can't compute return without buy_price

    if not buy_price or buy_price <= 0:
        buy_price = current_price  # fallback: assume bought at current (0% return)

    quantity = quantity or 1.0
    initial_value = round(buy_price * quantity, 2)
    current_value = round(current_price * quantity, 2)
    absolute_return = round(current_value - initial_value, 2)
    pct_return = round((absolute_return / initial_value) * 100, 2) if initial_value else 0.0

    sign = "+" if absolute_return >= 0 else ""
    emoji = "📈" if absolute_return >= 0 else "📉"

    summary = (
        f"{emoji} {ticker} Return Calculator\n"
        f"  Bought: {quantity:,.2f} shares @ ${buy_price:,.2f} = ${initial_value:,.2f}\n"
        f"  Current: ${current_price:,.2f}/share → ${current_value:,.2f}\n"
        f"  Return: {sign}${absolute_return:,.2f} ({sign}{pct_return:.2f}%)"
    )

    logger.info(
        "return_calculated",
        ticker=ticker,
        buy_price=buy_price,
        current_price=current_price,
        pct_return=pct_return,
    )

    return {
        "ticker": ticker,
        "buy_price": buy_price,
        "current_price": current_price,
        "quantity": quantity,
        "initial_value": initial_value,
        "current_value": current_value,
        "absolute_return": absolute_return,
        "pct_return": pct_return,
        "summary": summary,
        "artifacts": [],
    }


def _extract_current_price(data: dict[str, Any]) -> float | None:
    for key in ("price", "regularMarketPrice", "currentPrice", "close"):
        v = data.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    # Search nested
    for dep in data.get("_upstream", {}).values():
        if isinstance(dep, dict):
            for key in ("price", "regularMarketPrice", "currentPrice", "close"):
                v = dep.get(key)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
    return None


def _extract_ticker(data: dict[str, Any]) -> str:
    for key in ("ticker", "symbol"):
        v = data.get(key)
        if v:
            return str(v).upper()
    for dep in data.get("_upstream", {}).values():
        if isinstance(dep, dict):
            for key in ("symbol", "ticker"):
                v = dep.get(key)
                if v:
                    return str(v).upper()
    return "ASSET"


def _parse_instruction(text: str, current_price: float) -> tuple[float, float, float]:
    """Extract (buy_price, quantity, initial_investment) from the query text.

    Returns floats; 0.0 means 'not found'.
    """
    buy_price = 0.0
    quantity = 0.0
    initial_investment = 0.0

    if not text:
        return buy_price, quantity, initial_investment

    text_lower = text.lower()

    # Dollar amounts
    dollar_matches = [float(m.replace(",", "")) for m in _PRICE_RE.findall(text)]

    # Look for "at $X" or "paid $X" or "buy price $X" — requires explicit $ sign to avoid
    # matching quantity in patterns like "bought 10 shares at $500"
    buy_match = re.search(r"(?:at|paid|buy\s+price)\s+\$\s*([\d,]+(?:\.\d+)?)", text_lower)
    if not buy_match:
        # Fallback: "bought $X" or "purchased $X" with explicit dollar sign
        buy_match = re.search(r"(?:bought?|purchased?)\s+\$\s*([\d,]+(?:\.\d+)?)", text_lower)
    if buy_match:
        buy_price = float(buy_match.group(1).replace(",", ""))

    # Look for initial investment: "invested $X" or "put in $X" or "$X into"
    inv_match = re.search(r"(?:invested?|invest|put\s+in|initial(?:ly)?)\s+\$?\s*([\d,]+(?:\.\d+)?)", text_lower)
    if inv_match:
        initial_investment = float(inv_match.group(1).replace(",", ""))
    elif dollar_matches and not buy_price:
        # Largest dollar amount is likely the investment
        initial_investment = max(dollar_matches)

    # Look for quantity: "X shares" or "X units"
    qty_match = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:shares?|units?|stocks?)", text_lower)
    if qty_match:
        quantity = float(qty_match.group(1).replace(",", ""))
    elif initial_investment and buy_price:
        quantity = initial_investment / buy_price
    elif initial_investment:
        quantity = initial_investment / current_price

    return buy_price, quantity, initial_investment
