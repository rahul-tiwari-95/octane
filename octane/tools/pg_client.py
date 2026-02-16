"""PostgreSQL + pgVector client (stub for Phase 1).

Full implementation in Phase 2 when Memory Agent is deepened.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="pg_client")


class PgClient:
    """Async PostgreSQL client using asyncpg.

    Phase 1: Stub. Methods exist but log warnings.
    Phase 2: Full implementation with asyncpg + pgVector.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn
        self._pool = None

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        logger.warning("pg_client_stub", msg="PostgreSQL client is a stub in Phase 1")

    async def close(self) -> None:
        """Close the connection pool."""
        pass

    async def execute(self, query: str, *args: object) -> None:
        """Execute a query (stub)."""
        logger.warning("pg_execute_stub", query=query)

    async def fetch(self, query: str, *args: object) -> list[dict]:
        """Fetch rows (stub)."""
        logger.warning("pg_fetch_stub", query=query)
        return []
