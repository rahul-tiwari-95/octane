"""PostgreSQL + pgVector client.

Database:  octane
Host:      localhost:5432
User:      postgres  (or set DATABASE_URL / POSTGRES_URL in .env)

Connection string: postgresql://postgres@localhost:5432/octane
                   or set POSTGRES_URL in .env

Schema auto-created on first connect:

    memory_chunks  — warm tier: structured per-session text storage
    memory_embeddings — cold tier: pgVector semantic search (optional,
                        requires CREATE EXTENSION vector to be pre-installed)

Graceful degradation:
    - If Postgres is unreachable: log warning, return None/[] for all reads/writes.
      The MemoryAgent falls back to Redis-only mode silently.
    - If pgVector extension is absent: memory_embeddings table is skipped.
      Warm tier (plain Postgres) still works.
    - NEVER raises an exception that crashes the caller.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="pg_client")

# ── Schema DDL ───────────────────────────────────────────────────────────────

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS memory_chunks (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    slot        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    query       TEXT        NOT NULL DEFAULT '',
    agent_used  TEXT        NOT NULL DEFAULT '',
    metadata    JSONB       NOT NULL DEFAULT '{}',
    access_count INTEGER    NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_memory_slot    ON memory_chunks (slot);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_chunks (session_id);
CREATE INDEX IF NOT EXISTS idx_memory_session_slot ON memory_chunks (session_id, slot);
"""

_DDL_EMBEDDINGS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id         SERIAL  PRIMARY KEY,
    chunk_id   INTEGER REFERENCES memory_chunks(id) ON DELETE CASCADE,
    embedding  vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_embedding_vec
    ON memory_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class PgClient:
    """Async PostgreSQL client using asyncpg.

    Usage:
        pg = PgClient()
        await pg.connect()   # creates pool + runs DDL
        rows = await pg.fetch("SELECT ...", arg1, arg2)
        await pg.close()

    All methods are safe to call even when Postgres is unavailable —
    they return None / [] and log a warning instead of raising.
    """

    def __init__(self, dsn: str | None = None) -> None:
        if dsn is None:
            from octane.config import settings
            dsn = settings.postgres_url
        self.dsn = dsn
        self._pool = None
        self.available = False   # True once pool is live
        self.vector_enabled = False  # True once pgVector extension confirmed

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Create connection pool and ensure schema exists.

        Returns True if connected successfully, False if Postgres unavailable.
        Safe to call multiple times (idempotent).
        """
        if self._pool is not None:
            return self.available

        try:
            import asyncpg  # type: ignore
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=1,
                max_size=5,
                command_timeout=10,
            )
            self.available = True
            logger.info("pg_connected", dsn=self._redacted_dsn())
            await self._ensure_schema()
            return True
        except Exception as exc:
            logger.warning(
                "pg_unavailable",
                error=str(exc),
                hint="Memory will use Redis-only mode. Start Postgres to enable warm tier.",
            )
            self._pool = None
            self.available = False
            return False

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None
            self.available = False

    # ── Core query methods ─────────────────────────────────────────────────

    async def execute(self, query: str, *args: object) -> bool:
        """Run a DML query (INSERT / UPDATE / DELETE).

        Returns True on success, False if unavailable or error.
        """
        if not self.available or self._pool is None:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query, *args)
            return True
        except Exception as exc:
            logger.warning("pg_execute_error", error=str(exc), query=query[:80])
            return False

    async def fetch(self, query: str, *args: object) -> list[dict]:
        """Fetch multiple rows. Returns [] if unavailable."""
        if not self.available or self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("pg_fetch_error", error=str(exc), query=query[:80])
            return []

    async def fetchrow(self, query: str, *args: object) -> dict | None:
        """Fetch a single row. Returns None if unavailable or not found."""
        if not self.available or self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
        except Exception as exc:
            logger.warning("pg_fetchrow_error", error=str(exc), query=query[:80])
            return None

    async def fetchval(self, query: str, *args: object) -> object:
        """Fetch a single scalar value. Returns None if unavailable."""
        if not self.available or self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        except Exception as exc:
            logger.warning("pg_fetchval_error", error=str(exc), query=query[:80])
            return None

    # ── Schema management ──────────────────────────────────────────────────

    async def _ensure_schema(self) -> None:
        """Create tables if they don't exist. Safe to call repeatedly."""
        if not self.available or self._pool is None:
            return

        # ── Session 17: memory tables ──────────────────────────────────────
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(_DDL_CHUNKS)
                logger.info("pg_schema_ready", table="memory_chunks")
        except Exception as exc:
            logger.warning("pg_schema_error", error=str(exc))

        # pgVector is optional — try but don't fail if extension not installed
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(_DDL_EMBEDDINGS)
            self.vector_enabled = True
            logger.info("pg_schema_ready", table="memory_embeddings", vector=True)
        except Exception:
            logger.info("pg_vector_unavailable", hint="pgVector extension not installed — cold tier disabled")
            self.vector_enabled = False

        # ── Session 18A: structured storage schema ─────────────────────────
        try:
            import os
            schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
            with open(schema_path) as f:
                ddl = f.read()
            async with self._pool.acquire() as conn:
                await conn.execute(ddl)
            logger.info("pg_schema_ready", table="structured_storage_18a")
        except Exception as exc:
            logger.warning("pg_schema_18a_error", error=str(exc))

    def _redacted_dsn(self) -> str:
        """Log-safe DSN (hides password if present)."""
        import re
        return re.sub(r":([^@/]+)@", ":***@", self.dsn)
