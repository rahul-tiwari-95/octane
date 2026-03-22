"""PortfolioStore — async Postgres persistence for portfolio positions.

All methods are async and use PgClient (asyncpg under the hood).
Positions are upserted by (ticker, broker, account_id) so re-importing
the same CSV is safe — it updates quantities and costs in place.
"""

from __future__ import annotations

import structlog

from octane.portfolio.models import Position

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
