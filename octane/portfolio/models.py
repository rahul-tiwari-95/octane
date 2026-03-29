"""Portfolio domain models.

The core data model for a portfolio position.  Intentionally plain dataclasses —
no ORM, no heavy deps — so they are safe to use in tests and catalysts alike.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    """A single holding in a brokerage account.

    ``ticker`` is normalised to UPPERCASE on construction.
    """

    ticker: str
    quantity: float
    avg_cost: float
    currency: str = "USD"
    broker: str = ""
    account_id: str = ""
    sector: str = ""
    asset_class: str = "equity"
    notes: str = ""
    project_id: Optional[int] = None
    # Database-generated fields (None until persisted)
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

    @property
    def market_value(self) -> Optional[float]:
        """Current market value — only available if current_price is set externally."""
        if self.current_price is None:  # type: ignore[attr-defined]
            return None
        return round(self.quantity * self.current_price, 2)  # type: ignore[attr-defined]

    @property
    def cost_basis(self) -> float:
        return round(self.quantity * self.avg_cost, 2)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "currency": self.currency,
            "broker": self.broker,
            "account_id": self.account_id,
            "sector": self.sector,
            "asset_class": self.asset_class,
            "notes": self.notes,
            "project_id": self.project_id,
        }


@dataclass
class Portfolio:
    """A collection of positions with aggregate stats."""

    positions: list[Position] = field(default_factory=list)
    name: str = "default"

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @property
    def total_cost_basis(self) -> float:
        return round(sum(p.cost_basis for p in self.positions), 2)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def by_ticker(self, ticker: str) -> Optional[Position]:
        t = ticker.upper()
        return next((p for p in self.positions if p.ticker == t), None)


# ── Tax Lot ──────────────────────────────────────────────────────────────────

@dataclass
class TaxLot:
    """An individual purchase lot for cost basis tracking (FIFO/LIFO/SpecID)."""

    ticker: str
    shares: float
    cost_per_share: float
    purchase_date: datetime.date = field(default_factory=datetime.date.today)
    broker: str = ""
    account_id: str = ""
    sold_shares: float = 0.0
    notes: str = ""
    position_id: Optional[int] = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

    @property
    def remaining_shares(self) -> float:
        return round(self.shares - self.sold_shares, 6)

    @property
    def cost_basis(self) -> float:
        return round(self.remaining_shares * self.cost_per_share, 2)

    @property
    def is_long_term(self) -> bool:
        """Held > 1 year from purchase date."""
        delta = datetime.date.today() - self.purchase_date
        return delta.days > 365

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "cost_per_share": self.cost_per_share,
            "purchase_date": self.purchase_date.isoformat(),
            "broker": self.broker,
            "account_id": self.account_id,
            "sold_shares": self.sold_shares,
            "remaining_shares": self.remaining_shares,
            "cost_basis": self.cost_basis,
            "is_long_term": self.is_long_term,
        }


# ── Dividend ─────────────────────────────────────────────────────────────────

@dataclass
class Dividend:
    """Dividend record for income tracking."""

    ticker: str
    amount: float = 0.0
    ex_date: Optional[datetime.date] = None
    pay_date: Optional[datetime.date] = None
    frequency: str = "quarterly"   # monthly|quarterly|semi-annual|annual
    div_yield: float = 0.0
    payout_ratio: float = 0.0
    growth_rate: float = 0.0       # YoY dividend growth %
    source: str = "yfinance"
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

    @property
    def annual_income_per_share(self) -> float:
        freq_map = {"monthly": 12, "quarterly": 4, "semi-annual": 2, "annual": 1}
        multiplier = freq_map.get(self.frequency, 4)
        return round(self.amount * multiplier, 4)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "amount": self.amount,
            "ex_date": self.ex_date.isoformat() if self.ex_date else None,
            "pay_date": self.pay_date.isoformat() if self.pay_date else None,
            "frequency": self.frequency,
            "div_yield": self.div_yield,
            "payout_ratio": self.payout_ratio,
            "growth_rate": self.growth_rate,
        }


# ── Net Worth Snapshot ───────────────────────────────────────────────────────

@dataclass
class NetWorthSnapshot:
    """Point-in-time portfolio value snapshot for timeline tracking."""

    snapshot_date: datetime.date = field(default_factory=datetime.date.today)
    total_value: float = 0.0
    equities_value: float = 0.0
    crypto_value: float = 0.0
    cash_value: float = 0.0
    position_count: int = 0
    notes: str = ""
    id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "snapshot_date": self.snapshot_date.isoformat(),
            "total_value": self.total_value,
            "equities_value": self.equities_value,
            "crypto_value": self.crypto_value,
            "cash_value": self.cash_value,
            "position_count": self.position_count,
        }


# ── Crypto Position ─────────────────────────────────────────────────────────

@dataclass
class CryptoPosition:
    """A cryptocurrency holding."""

    coin: str
    quantity: float = 0.0
    cost_per_coin: float = 0.0
    exchange: str = ""
    wallet_address: str = ""
    notes: str = ""
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.coin = self.coin.upper().strip()

    @property
    def cost_basis(self) -> float:
        return round(self.quantity * self.cost_per_coin, 2)

    def to_dict(self) -> dict:
        return {
            "coin": self.coin,
            "quantity": self.quantity,
            "cost_per_coin": self.cost_per_coin,
            "cost_basis": self.cost_basis,
            "exchange": self.exchange,
        }
