"""PortfolioStore — async Postgres persistence for portfolio positions.

All methods are async and use PgClient (asyncpg under the hood).
Positions are upserted by (ticker, broker, account_id) so re-importing
the same CSV is safe — it updates quantities and costs in place.
"""

from __future__ import annotations

import datetime

import structlog

from octane.portfolio.models import (
    CryptoPosition,
    Dividend,
    NetWorthSnapshot,
    Position,
    TaxLot,
)

logger = structlog.get_logger().bind(component="portfolio.store")


class PortfolioStore:
    """Async Postgres store for portfolio positions."""

    def __init__(self) -> None:
        from octane.tools.pg_client import PgClient
        self._pg = PgClient()

    async def connect(self) -> None:
        await self._pg.connect()

    async def close(self) -> None:
        await self._pg.close()

    # ── Write ─────────────────────────────────────────────────────────────

    async def upsert_position(self, pos: Position) -> int:
        """Insert or update a position.  Returns the row id."""
        if not self._pg.available:
            raise RuntimeError("PortfolioStore: Postgres not available")

        row = await self._pg.fetchrow(
            """
            INSERT INTO portfolio_positions
                (ticker, quantity, avg_cost, currency, broker, account_id,
                 sector, asset_class, notes, project_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (ticker, broker, account_id)
            DO UPDATE SET
                quantity    = EXCLUDED.quantity,
                avg_cost    = EXCLUDED.avg_cost,
                sector      = EXCLUDED.sector,
                asset_class = EXCLUDED.asset_class,
                notes       = EXCLUDED.notes,
                updated_at  = NOW()
            RETURNING id
            """,
            pos.ticker,
            pos.quantity,
            pos.avg_cost,
            pos.currency,
            pos.broker,
            pos.account_id,
            pos.sector,
            pos.asset_class,
            pos.notes,
            pos.project_id,
        )
        return row["id"] if row else -1

    async def upsert_many(self, positions: list[Position]) -> int:
        """Bulk upsert positions.  Returns count of rows affected."""
        count = 0
        for pos in positions:
            await self.upsert_position(pos)
            count += 1
        logger.info("portfolio_upsert_many", count=count)
        return count

    async def delete_position(self, ticker: str, broker: str = "", account_id: str = "") -> int:
        """Delete a position by ticker (optionally scoped to broker/account).
        Returns number of rows deleted."""
        if broker and account_id:
            return await self._pg.execute(
                "DELETE FROM portfolio_positions WHERE ticker=$1 AND broker=$2 AND account_id=$3",
                ticker.upper(), broker, account_id,
            )
        elif broker:
            return await self._pg.execute(
                "DELETE FROM portfolio_positions WHERE ticker=$1 AND broker=$2",
                ticker.upper(), broker,
            )
        else:
            return await self._pg.execute(
                "DELETE FROM portfolio_positions WHERE ticker=$1",
                ticker.upper(),
            )

    async def clear(self, project_id: int | None = None) -> int:
        """Delete all positions (optionally scoped to a project)."""
        if project_id is not None:
            return await self._pg.execute(
                "DELETE FROM portfolio_positions WHERE project_id=$1", project_id
            )
        return await self._pg.execute("DELETE FROM portfolio_positions")

    # ── Read ──────────────────────────────────────────────────────────────

    async def list_positions(
        self,
        project_id: int | None = None,
        broker: str | None = None,
    ) -> list[Position]:
        """Return all positions, optionally filtered."""
        if not self._pg.available:
            return []

        clauses = []
        params: list = []
        if project_id is not None:
            clauses.append(f"project_id = ${len(params)+1}")
            params.append(project_id)
        if broker:
            clauses.append(f"broker = ${len(params)+1}")
            params.append(broker)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await self._pg.fetch(
            f"SELECT * FROM portfolio_positions {where} ORDER BY ticker",
            *params,
        )
        return [_row_to_position(r) for r in rows]

    async def get_position(
        self, ticker: str, broker: str = "", account_id: str = ""
    ) -> Position | None:
        """Fetch a single position by ticker (+ optional broker/account scope)."""
        if not self._pg.available:
            return None

        t = ticker.upper()
        if broker and account_id:
            row = await self._pg.fetchrow(
                "SELECT * FROM portfolio_positions WHERE ticker=$1 AND broker=$2 AND account_id=$3",
                t, broker, account_id,
            )
        elif broker:
            row = await self._pg.fetchrow(
                "SELECT * FROM portfolio_positions WHERE ticker=$1 AND broker=$2 LIMIT 1",
                t, broker,
            )
        else:
            row = await self._pg.fetchrow(
                "SELECT * FROM portfolio_positions WHERE ticker=$1 LIMIT 1", t
            )
        return _row_to_position(row) if row else None

    async def get_tickers(self, project_id: int | None = None) -> list[str]:
        """Return distinct tickers in the portfolio."""
        if not self._pg.available:
            return []
        where = "WHERE project_id=$1" if project_id is not None else ""
        params = [project_id] if project_id is not None else []
        rows = await self._pg.fetch(
            f"SELECT DISTINCT ticker FROM portfolio_positions {where} ORDER BY ticker", *params
        )
        return [r["ticker"] for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_position(row) -> Position:
    return Position(
        id=row["id"],
        ticker=row["ticker"],
        quantity=float(row["quantity"]),
        avg_cost=float(row["avg_cost"]),
        currency=row["currency"],
        broker=row.get("broker", ""),
        account_id=row.get("account_id", ""),
        sector=row.get("sector", ""),
        asset_class=row.get("asset_class", "equity"),
        notes=row.get("notes", ""),
        project_id=row.get("project_id"),
    )


def _row_to_tax_lot(row) -> TaxLot:
    pd = row.get("purchase_date")
    if pd and not isinstance(pd, datetime.date):
        pd = datetime.date.fromisoformat(str(pd))
    return TaxLot(
        id=row["id"],
        ticker=row["ticker"],
        shares=float(row["shares"]),
        cost_per_share=float(row["cost_per_share"]),
        purchase_date=pd or datetime.date.today(),
        broker=row.get("broker", ""),
        account_id=row.get("account_id", ""),
        sold_shares=float(row.get("sold_shares", 0)),
        notes=row.get("notes", ""),
        position_id=row.get("position_id"),
    )


def _row_to_dividend(row) -> Dividend:
    def _to_date(v):
        if v is None:
            return None
        if isinstance(v, datetime.date):
            return v
        return datetime.date.fromisoformat(str(v))

    return Dividend(
        id=row["id"],
        ticker=row["ticker"],
        amount=float(row.get("amount", 0)),
        ex_date=_to_date(row.get("ex_date")),
        pay_date=_to_date(row.get("pay_date")),
        frequency=row.get("frequency", "quarterly"),
        div_yield=float(row.get("div_yield", 0)),
        payout_ratio=float(row.get("payout_ratio", 0)),
        growth_rate=float(row.get("growth_rate", 0)),
        source=row.get("source", "yfinance"),
    )


def _row_to_snapshot(row) -> NetWorthSnapshot:
    sd = row.get("snapshot_date")
    if sd and not isinstance(sd, datetime.date):
        sd = datetime.date.fromisoformat(str(sd))
    return NetWorthSnapshot(
        id=row["id"],
        snapshot_date=sd or datetime.date.today(),
        total_value=float(row.get("total_value", 0)),
        equities_value=float(row.get("equities_value", 0)),
        crypto_value=float(row.get("crypto_value", 0)),
        cash_value=float(row.get("cash_value", 0)),
        position_count=int(row.get("position_count", 0)),
        notes=row.get("notes", ""),
    )


def _row_to_crypto(row) -> CryptoPosition:
    return CryptoPosition(
        id=row["id"],
        coin=row["coin"],
        quantity=float(row.get("quantity", 0)),
        cost_per_coin=float(row.get("cost_per_coin", 0)),
        exchange=row.get("exchange", ""),
        wallet_address=row.get("wallet_address", ""),
        notes=row.get("notes", ""),
    )


# ── Tax Lot Store ────────────────────────────────────────────────────────────

class TaxLotStore:
    """Async Postgres store for tax lots."""

    def __init__(self) -> None:
        from octane.tools.pg_client import PgClient
        self._pg = PgClient()

    async def connect(self) -> None:
        await self._pg.connect()

    async def close(self) -> None:
        await self._pg.close()

    async def add_lot(self, lot: TaxLot) -> int:
        if not self._pg.available:
            raise RuntimeError("TaxLotStore: Postgres not available")
        row = await self._pg.fetchrow(
            """
            INSERT INTO tax_lots
                (position_id, ticker, shares, cost_per_share, purchase_date,
                 broker, account_id, sold_shares, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            lot.position_id, lot.ticker, lot.shares, lot.cost_per_share,
            lot.purchase_date, lot.broker, lot.account_id,
            lot.sold_shares, lot.notes,
        )
        return row["id"] if row else -1

    async def list_lots(self, ticker: str | None = None) -> list[TaxLot]:
        if not self._pg.available:
            return []
        if ticker:
            rows = await self._pg.fetch(
                "SELECT * FROM tax_lots WHERE ticker=$1 ORDER BY purchase_date",
                ticker.upper(),
            )
        else:
            rows = await self._pg.fetch(
                "SELECT * FROM tax_lots ORDER BY ticker, purchase_date"
            )
        return [_row_to_tax_lot(r) for r in rows]

    async def sell_shares(
        self, ticker: str, shares_to_sell: float, method: str = "FIFO"
    ) -> list[dict]:
        """Allocate shares to sell across lots using FIFO or LIFO.

        Returns list of dicts: [{lot_id, shares_sold, cost_per_share, gain}]
        Does NOT update the DB — caller commits if satisfied.
        """
        lots = await self.list_lots(ticker)
        open_lots = [lt for lt in lots if lt.remaining_shares > 0]

        if method.upper() == "LIFO":
            open_lots.sort(key=lambda lt: lt.purchase_date, reverse=True)
        # FIFO is default — already sorted by purchase_date ASC

        remaining = shares_to_sell
        allocations: list[dict] = []
        for lt in open_lots:
            if remaining <= 0:
                break
            take = min(remaining, lt.remaining_shares)
            allocations.append({
                "lot_id": lt.id,
                "ticker": lt.ticker,
                "shares_sold": round(take, 6),
                "cost_per_share": lt.cost_per_share,
                "purchase_date": lt.purchase_date.isoformat(),
                "is_long_term": lt.is_long_term,
            })
            remaining -= take

        return allocations

    async def record_sale(self, lot_id: int, shares_sold: float) -> None:
        """Increment sold_shares on a lot after a confirmed sale."""
        if not self._pg.available:
            return
        await self._pg.execute(
            "UPDATE tax_lots SET sold_shares = sold_shares + $1 WHERE id = $2",
            shares_sold, lot_id,
        )


# ── Dividend Store ───────────────────────────────────────────────────────────

class DividendStore:
    """Async Postgres store for dividend records."""

    def __init__(self) -> None:
        from octane.tools.pg_client import PgClient
        self._pg = PgClient()

    async def connect(self) -> None:
        await self._pg.connect()

    async def close(self) -> None:
        await self._pg.close()

    async def upsert_dividend(self, div: Dividend) -> int:
        if not self._pg.available:
            raise RuntimeError("DividendStore: Postgres not available")
        row = await self._pg.fetchrow(
            """
            INSERT INTO dividends
                (ticker, amount, ex_date, pay_date, frequency,
                 div_yield, payout_ratio, growth_rate, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            div.ticker, div.amount, div.ex_date, div.pay_date,
            div.frequency, div.div_yield, div.payout_ratio,
            div.growth_rate, div.source,
        )
        return row["id"] if row else -1

    async def list_dividends(self, ticker: str | None = None) -> list[Dividend]:
        if not self._pg.available:
            return []
        if ticker:
            rows = await self._pg.fetch(
                "SELECT * FROM dividends WHERE ticker=$1 ORDER BY ex_date DESC",
                ticker.upper(),
            )
        else:
            rows = await self._pg.fetch(
                "SELECT * FROM dividends ORDER BY ticker, ex_date DESC"
            )
        return [_row_to_dividend(r) for r in rows]


# ── Net Worth Store ──────────────────────────────────────────────────────────

class NetWorthStore:
    """Async Postgres store for net worth snapshots."""

    def __init__(self) -> None:
        from octane.tools.pg_client import PgClient
        self._pg = PgClient()

    async def connect(self) -> None:
        await self._pg.connect()

    async def close(self) -> None:
        await self._pg.close()

    async def save_snapshot(self, snap: NetWorthSnapshot) -> int:
        if not self._pg.available:
            raise RuntimeError("NetWorthStore: Postgres not available")
        row = await self._pg.fetchrow(
            """
            INSERT INTO net_worth_snapshots
                (snapshot_date, total_value, equities_value, crypto_value,
                 cash_value, position_count, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (snapshot_date)
            DO UPDATE SET
                total_value    = EXCLUDED.total_value,
                equities_value = EXCLUDED.equities_value,
                crypto_value   = EXCLUDED.crypto_value,
                cash_value     = EXCLUDED.cash_value,
                position_count = EXCLUDED.position_count,
                notes          = EXCLUDED.notes
            RETURNING id
            """,
            snap.snapshot_date, snap.total_value, snap.equities_value,
            snap.crypto_value, snap.cash_value, snap.position_count, snap.notes,
        )
        return row["id"] if row else -1

    async def list_snapshots(self, limit: int = 365) -> list[NetWorthSnapshot]:
        if not self._pg.available:
            return []
        rows = await self._pg.fetch(
            "SELECT * FROM net_worth_snapshots ORDER BY snapshot_date DESC LIMIT $1",
            limit,
        )
        return [_row_to_snapshot(r) for r in rows]


# ── Crypto Store ─────────────────────────────────────────────────────────────

class CryptoStore:
    """Async Postgres store for crypto positions."""

    def __init__(self) -> None:
        from octane.tools.pg_client import PgClient
        self._pg = PgClient()

    async def connect(self) -> None:
        await self._pg.connect()

    async def close(self) -> None:
        await self._pg.close()

    async def upsert_position(self, pos: CryptoPosition) -> int:
        if not self._pg.available:
            raise RuntimeError("CryptoStore: Postgres not available")
        row = await self._pg.fetchrow(
            """
            INSERT INTO crypto_positions
                (coin, quantity, cost_per_coin, exchange, wallet_address, notes)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (coin, exchange)
            DO UPDATE SET
                quantity      = EXCLUDED.quantity,
                cost_per_coin = EXCLUDED.cost_per_coin,
                wallet_address = EXCLUDED.wallet_address,
                notes         = EXCLUDED.notes,
                updated_at    = NOW()
            RETURNING id
            """,
            pos.coin, pos.quantity, pos.cost_per_coin,
            pos.exchange, pos.wallet_address, pos.notes,
        )
        return row["id"] if row else -1

    async def upsert_many(self, positions: list[CryptoPosition]) -> int:
        count = 0
        for pos in positions:
            await self.upsert_position(pos)
            count += 1
        return count

    async def list_positions(self, exchange: str | None = None) -> list[CryptoPosition]:
        if not self._pg.available:
            return []
        if exchange:
            rows = await self._pg.fetch(
                "SELECT * FROM crypto_positions WHERE exchange=$1 ORDER BY coin",
                exchange,
            )
        else:
            rows = await self._pg.fetch(
                "SELECT * FROM crypto_positions ORDER BY coin"
            )
        return [_row_to_crypto(r) for r in rows]

    async def get_coins(self) -> list[str]:
        if not self._pg.available:
            return []
        rows = await self._pg.fetch(
            "SELECT DISTINCT coin FROM crypto_positions ORDER BY coin"
        )
        return [r["coin"] for r in rows]
