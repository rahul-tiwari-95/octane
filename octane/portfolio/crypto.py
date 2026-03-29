"""Crypto exchange CSV parsers and CoinGecko price fetcher.

Supported exchanges:
    Coinbase  — "Timestamp", "Transaction Type", "Asset", "Quantity Transacted", "Spot Price at Transaction"
    Kraken    — "asset", "amount", "fee", "type"
    Binance   — "Coin", "Amount", "Price"
    Gemini    — "Symbol", "Amount", "Price (USD)"
    Generic   — fallback for any CSV with coin/quantity/cost columns

Usage::

    from octane.portfolio.crypto import parse_crypto_csv, fetch_crypto_prices
    positions = parse_crypto_csv("/path/to/export.csv")
    prices = fetch_crypto_prices(["BTC", "ETH", "SOL"])
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import structlog

from octane.portfolio.models import CryptoPosition

logger = structlog.get_logger().bind(component="portfolio.crypto")

# ── Format detection ─────────────────────────────────────────────────────────

_EXCHANGE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "Coinbase": ("asset", "quantity transacted", "spot price"),
    "Kraken":   ("asset", "amount", "fee", "type"),
    "Binance":  ("coin", "amount", "price"),
    "Gemini":   ("symbol", "amount", "price (usd)"),
}


def detect_exchange(headers: list[str]) -> str:
    lower = [h.lower().strip() for h in headers]
    for exchange, sigs in _EXCHANGE_SIGNATURES.items():
        if all(any(sig in col for col in lower) for sig in sigs):
            return exchange
    return "Generic"


def _clean_num(val) -> float:
    if val is None:
        return 0.0
    s = str(val).strip().replace(",", "").replace("$", "").replace("+", "")
    if not s or s in ("N/A", "-", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _col(row: dict, *candidates: str) -> str:
    """Find a column value by trying multiple candidate names."""
    lower_row = {k.lower().strip(): v for k, v in row.items()}
    for c in candidates:
        if c.lower() in lower_row:
            return str(lower_row[c.lower()])
    return ""


def _normalise_headers(raw: list[str]) -> list[str]:
    return [h.strip() for h in raw]


# ── Exchange parsers ─────────────────────────────────────────────────────────

def _parse_coinbase(rows: list[dict], exchange: str) -> list[CryptoPosition]:
    """Coinbase: aggregate buys by asset."""
    holdings: dict[str, dict] = {}  # coin -> {qty, total_cost}
    for row in rows:
        tx_type = _col(row, "transaction type").lower()
        if tx_type not in ("buy", "receive", "staking income", "rewards income", "advanced trade buy"):
            continue
        coin = _col(row, "asset").upper().strip()
        qty = _clean_num(_col(row, "quantity transacted"))
        spot = _clean_num(_col(row, "spot price at transaction", "spot price"))
        if not coin or qty <= 0:
            continue
        if coin not in holdings:
            holdings[coin] = {"qty": 0.0, "total_cost": 0.0}
        holdings[coin]["qty"] += qty
        holdings[coin]["total_cost"] += qty * spot

    return [
        CryptoPosition(
            coin=coin,
            quantity=round(d["qty"], 8),
            cost_per_coin=round(d["total_cost"] / d["qty"], 4) if d["qty"] > 0 else 0.0,
            exchange=exchange,
        )
        for coin, d in holdings.items()
        if d["qty"] > 0
    ]


def _parse_kraken(rows: list[dict], exchange: str) -> list[CryptoPosition]:
    holdings: dict[str, dict] = {}
    for row in rows:
        tx_type = _col(row, "type").lower()
        if tx_type not in ("buy", "receive", "deposit", "staking"):
            continue
        coin = _col(row, "asset").upper().strip()
        # Kraken uses codes like XXBT for BTC, XETH for ETH
        coin = coin.replace("XXBT", "BTC").replace("XETH", "ETH")
        if coin.startswith("X") and len(coin) == 4:
            coin = coin[1:]
        if coin.startswith("Z") and len(coin) == 4:
            continue  # Fiat (ZUSD etc)
        qty = abs(_clean_num(_col(row, "amount")))
        cost = _clean_num(_col(row, "cost", "price"))
        if not coin or qty <= 0:
            continue
        if coin not in holdings:
            holdings[coin] = {"qty": 0.0, "total_cost": 0.0}
        holdings[coin]["qty"] += qty
        holdings[coin]["total_cost"] += cost if cost > 0 else 0.0

    return [
        CryptoPosition(
            coin=coin,
            quantity=round(d["qty"], 8),
            cost_per_coin=round(d["total_cost"] / d["qty"], 4) if d["qty"] > 0 and d["total_cost"] > 0 else 0.0,
            exchange=exchange,
        )
        for coin, d in holdings.items()
        if d["qty"] > 0
    ]


def _parse_binance(rows: list[dict], exchange: str) -> list[CryptoPosition]:
    holdings: dict[str, dict] = {}
    for row in rows:
        coin = _col(row, "coin", "asset").upper().strip()
        qty = _clean_num(_col(row, "amount", "quantity"))
        price = _clean_num(_col(row, "price", "avg price"))
        if not coin or qty <= 0:
            continue
        if coin in ("USD", "USDT", "USDC", "BUSD"):
            continue
        if coin not in holdings:
            holdings[coin] = {"qty": 0.0, "total_cost": 0.0}
        holdings[coin]["qty"] += qty
        holdings[coin]["total_cost"] += qty * price

    return [
        CryptoPosition(
            coin=coin,
            quantity=round(d["qty"], 8),
            cost_per_coin=round(d["total_cost"] / d["qty"], 4) if d["qty"] > 0 else 0.0,
            exchange=exchange,
        )
        for coin, d in holdings.items()
        if d["qty"] > 0
    ]


def _parse_gemini(rows: list[dict], exchange: str) -> list[CryptoPosition]:
    holdings: dict[str, dict] = {}
    for row in rows:
        coin = _col(row, "symbol", "asset").upper().strip()
        qty = _clean_num(_col(row, "amount", "quantity"))
        price = _clean_num(_col(row, "price (usd)", "price"))
        if not coin or qty <= 0:
            continue
        # Gemini uses pairs like BTCUSD — strip USD suffix
        for suffix in ("USD", "GBP", "EUR"):
            if coin.endswith(suffix) and len(coin) > len(suffix):
                coin = coin[:-len(suffix)]
        if coin not in holdings:
            holdings[coin] = {"qty": 0.0, "total_cost": 0.0}
        holdings[coin]["qty"] += qty
        holdings[coin]["total_cost"] += qty * price

    return [
        CryptoPosition(
            coin=coin,
            quantity=round(d["qty"], 8),
            cost_per_coin=round(d["total_cost"] / d["qty"], 4) if d["qty"] > 0 else 0.0,
            exchange=exchange,
        )
        for coin, d in holdings.items()
        if d["qty"] > 0
    ]


def _parse_generic_crypto(rows: list[dict], exchange: str) -> list[CryptoPosition]:
    holdings: dict[str, dict] = {}
    for row in rows:
        coin = _col(row, "coin", "asset", "symbol", "currency", "ticker").upper().strip()
        qty = _clean_num(_col(row, "quantity", "amount", "qty", "balance"))
        cost = _clean_num(_col(row, "cost", "price", "avg_cost", "cost_per_coin", "avg price"))
        if not coin or qty <= 0:
            continue
        if coin in ("USD", "USDT", "USDC", "BUSD", "DAI"):
            continue
        if coin not in holdings:
            holdings[coin] = {"qty": 0.0, "total_cost": 0.0}
        holdings[coin]["qty"] += qty
        holdings[coin]["total_cost"] += qty * cost

    return [
        CryptoPosition(
            coin=coin,
            quantity=round(d["qty"], 8),
            cost_per_coin=round(d["total_cost"] / d["qty"], 4) if d["qty"] > 0 else 0.0,
            exchange=exchange or "Unknown",
        )
        for coin, d in holdings.items()
        if d["qty"] > 0
    ]


_EXCHANGE_PARSERS: dict[str, callable] = {
    "Coinbase": _parse_coinbase,
    "Kraken":   _parse_kraken,
    "Binance":  _parse_binance,
    "Gemini":   _parse_gemini,
    "Generic":  _parse_generic_crypto,
}


# ── Public API ───────────────────────────────────────────────────────────────

def parse_crypto_csv(
    path: str,
    exchange: str | None = None,
) -> list[CryptoPosition]:
    """Parse a crypto exchange CSV export.

    Args:
        path: filesystem path to CSV file.
        exchange: override exchange name; auto-detected if None.

    Returns:
        List of CryptoPosition (aggregated by coin).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    text = p.read_text(encoding="utf-8-sig")
    return parse_crypto_csv_text(text, exchange=exchange)


def parse_crypto_csv_text(
    text: str,
    exchange: str | None = None,
) -> list[CryptoPosition]:
    """Parse crypto CSV from an in-memory string."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return []

    headers = _normalise_headers(list(reader.fieldnames))
    detected = exchange or detect_exchange(headers)
    logger.info("crypto_parse", exchange=detected, headers=headers[:5])

    rows = list(reader)
    if not rows:
        return []

    parser = _EXCHANGE_PARSERS.get(detected, _parse_generic_crypto)
    positions = parser(rows, detected)
    logger.info("crypto_parsed", exchange=detected, positions=len(positions))
    return positions


# ── CoinGecko Price Fetcher ──────────────────────────────────────────────────

# Map common coin symbols to CoinGecko IDs
_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "NEAR": "near",
    "FTM": "fantom",
    "ALGO": "algorand",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "XLM": "stellar",
    "ETC": "ethereum-classic",
    "FIL": "filecoin",
    "ICP": "internet-computer",
    "HBAR": "hedera-hashgraph",
    "VET": "vechain",
    "SAND": "the-sandbox",
    "MANA": "decentraland",
    "APE": "apecoin",
    "AAVE": "aave",
    "MKR": "maker",
    "CRV": "curve-dao-token",
    "COMP": "compound-governance-token",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "SEI": "sei-network",
    "TIA": "celestia",
    "JUP": "jupiter-exchange-solana",
    "RENDER": "render-token",
    "INJ": "injective-protocol",
    "PEPE": "pepe",
    "WIF": "dogwifcoin",
    "BONK": "bonk",
}


def fetch_crypto_prices(coins: list[str]) -> dict[str, float]:
    """Fetch current USD prices from CoinGecko (free API, no key needed).

    Args:
        coins: list of coin symbols (e.g. ["BTC", "ETH", "SOL"]).

    Returns:
        Dict of {symbol: price_usd}. Best-effort — missing coins omitted.
    """
    try:
        import urllib.request
        import json

        # Map symbols to CoinGecko IDs
        ids_to_symbols: dict[str, str] = {}
        for coin in coins:
            cg_id = _COINGECKO_IDS.get(coin.upper(), coin.lower())
            ids_to_symbols[cg_id] = coin.upper()

        if not ids_to_symbols:
            return {}

        # CoinGecko simple/price endpoint (free, rate-limited to ~10-30 req/min)
        id_list = ",".join(ids_to_symbols.keys())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={id_list}&vs_currencies=usd"

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        result: dict[str, float] = {}
        for cg_id, symbol in ids_to_symbols.items():
            if cg_id in data and "usd" in data[cg_id]:
                result[symbol] = float(data[cg_id]["usd"])

        return result

    except Exception as exc:
        logger.warning("coingecko_fetch_failed", error=str(exc))
        return {}
