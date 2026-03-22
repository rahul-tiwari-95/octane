"""Portfolio domain models.

The core data model for a portfolio position.  Intentionally plain dataclasses —
no ORM, no heavy deps — so they are safe to use in tests and catalysts alike.
"""

from __future__ import annotations

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
