"""MigrationRunner — idempotent Postgres schema migrations for Octane.

Applies ``octane/tools/schema.sql`` against the live database.  Tracks
applied versions in ``schema_migrations`` so it is safe to call repeatedly
(e.g. at every worker startup or from ``octane db migrate``).

Design principles:
    - Idempotent: every statement uses ``IF NOT EXISTS`` / ``IF NOT EXISTS``.
    - Transactional: the entire schema file runs inside a single transaction;
      either all changes land or none do.
    - Version-tracked: a sentinel row in ``schema_migrations`` records the
      migration version so repeated runs are no-ops.
    - Zero-downtime safe: never drops or alters existing columns.

Usage::

    runner = MigrationRunner()
    result = await runner.migrate()
    # result.applied  → True if new migration was applied
    # result.version  → migration version string
    # result.tables   → list of table names now in the schema

    status = await runner.status()
    # status.applied_versions  → list[str]
    # status.missing_versions  → list[str] (pending)
    # status.table_counts      → dict[table, row_count]
"""

from __future__ import annotations

import hashlib
import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="migrations")

# Current schema version — bump this when schema.sql changes substantially
SCHEMA_VERSION = "20A"

# Path to schema SQL file (relative to this file)
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class MigrationResult:
    applied: bool
    version: str
    tables: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class MigrationStatus:
    applied_versions: list[str] = field(default_factory=list)
    pending_versions: list[str] = field(default_factory=list)
    table_counts: dict[str, int] = field(default_factory=dict)
    pg_available: bool = False


class MigrationRunner:
    """Applies and tracks Postgres schema migrations.

    Args:
        dsn: asyncpg connection string.  Defaults to ``settings.postgres_url``.
    """

    # All known migration versions in order
    VERSIONS: list[str] = ["20A"]

    def __init__(self, dsn: str | None = None) -> None:
        if dsn is None:
            from octane.config import settings
            dsn = settings.postgres_url
        self._dsn = dsn

    # ── Public API ────────────────────────────────────────────────────────

    async def migrate(self) -> MigrationResult:
        """Apply all pending migrations.  Safe to call repeatedly.

        Returns:
            MigrationResult with applied=True if any new migration ran.
        """
        conn = await self._connect()
        if conn is None:
            return MigrationResult(applied=False, version="", error="Postgres unavailable")

        try:
            return await self._apply_all(conn)
        finally:
            await conn.close()

    async def status(self) -> MigrationStatus:
        """Return current migration state and per-table row counts."""
        conn = await self._connect()
        if conn is None:
            return MigrationStatus(pg_available=False)

        try:
            return await self._collect_status(conn)
        finally:
            await conn.close()

    async def reset(self) -> bool:
        """Drop all octane tables and re-apply schema.  DEV ONLY.

        Returns True on success.
        """
        conn = await self._connect()
        if conn is None:
            logger.error("migration_reset_no_pg")
            return False

        try:
            # Drop known tables in dependency order
            _DROP_ORDER = [
                "content_tags", "tags",
                "embeddings", "memory_embeddings", "memory_chunks",
                "research_findings_v2", "research_findings",
                "generated_artifacts", "portfolio_positions",
                "tracked_jobs", "user_files", "web_pages",
                "projects", "schema_migrations",
            ]
            async with conn.transaction():
                for tbl in _DROP_ORDER:
                    await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            logger.warning("migration_reset_dropped_all_tables")
            result = await self._apply_all(conn)
            return result.applied
        except Exception as exc:
            logger.error("migration_reset_failed", error=str(exc))
            return False
        finally:
            await conn.close()

    # ── Internals ─────────────────────────────────────────────────────────

    async def _connect(self):
        """Return a live asyncpg connection or None."""
        try:
            import asyncpg  # type: ignore
            return await asyncpg.connect(self._dsn)
        except Exception as exc:
            logger.warning("migration_connect_failed", error=str(exc))
            return None

    async def _apply_all(self, conn) -> MigrationResult:
        """Ensure schema_migrations exists, then apply any pending versions."""
        # Bootstrap: create tracking table first (outside version tracking)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     TEXT        PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Check which versions are already applied
        applied = set(
            r["version"]
            for r in await conn.fetch("SELECT version FROM schema_migrations")
        )

        pending = [v for v in self.VERSIONS if v not in applied]
        if not pending:
            tables = await self._list_tables(conn)
            logger.debug("migration_already_current", version=self.VERSIONS[-1])
            return MigrationResult(applied=False, version=self.VERSIONS[-1], tables=tables)

        # Read schema SQL
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")

        # Apply within a single transaction
        try:
            async with conn.transaction():
                # Split on statement boundaries, execute individually so
                # asyncpg can handle multi-statement files
                for stmt in _split_sql(sql):
                    if stmt.strip():
                        await conn.execute(stmt)
                # Record all newly applied versions
                for v in pending:
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1) "
                        "ON CONFLICT DO NOTHING",
                        v,
                    )
        except Exception as exc:
            logger.error("migration_apply_failed", error=str(exc))
            return MigrationResult(applied=False, version="", error=str(exc))

        tables = await self._list_tables(conn)
        logger.info(
            "migration_applied",
            versions=pending,
            tables=len(tables),
        )
        return MigrationResult(applied=True, version=pending[-1], tables=tables)

    async def _collect_status(self, conn) -> MigrationStatus:
        """Gather applied versions and per-table row counts."""
        # Check if schema_migrations exists yet
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name   = 'schema_migrations'
            )
            """
        )
        applied: list[str] = []
        if exists:
            rows = await conn.fetch(
                "SELECT version FROM schema_migrations ORDER BY applied_at"
            )
            applied = [r["version"] for r in rows]

        pending = [v for v in self.VERSIONS if v not in applied]

        # Collect all public tables
        table_rows = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )
        table_counts: dict[str, int] = {}
        for row in table_rows:
            tbl = row["table_name"]
            try:
                cnt = await conn.fetchval(f"SELECT COUNT(*) FROM {tbl}")
                table_counts[tbl] = int(cnt)
            except Exception:
                table_counts[tbl] = -1

        return MigrationStatus(
            applied_versions=applied,
            pending_versions=pending,
            table_counts=table_counts,
            pg_available=True,
        )

    @staticmethod
    async def _list_tables(conn) -> list[str]:
        rows = await conn.fetch(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )
        return [r["table_name"] for r in rows]


# ── SQL splitter ──────────────────────────────────────────────────────────────

def _split_sql(sql: str) -> list[str]:
    """Split a SQL file into individual statements, ignoring comments.

    Handles:
    - ``--`` line comments
    - ``/* */`` block comments
    - Dollar-quoted strings ($$...$$) that may span multiple lines
    - Standard semicolon delimiters
    """
    import re

    # Strip line comments while preserving newlines for error reporting
    sql = re.sub(r"--[^\n]*", "", sql)
    # Strip block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

    statements = []
    current: list[str] = []
    in_dollar_quote = False
    dollar_tag = ""

    i = 0
    while i < len(sql):
        # Detect start/end of dollar-quoted string
        if not in_dollar_quote:
            m = re.match(r"\$([^$]*)\$", sql[i:])
            if m:
                dollar_tag = m.group(0)
                in_dollar_quote = True
                current.append(dollar_tag)
                i += len(dollar_tag)
                continue
        else:
            if sql[i:].startswith(dollar_tag):
                current.append(dollar_tag)
                i += len(dollar_tag)
                in_dollar_quote = False
                continue

        ch = sql[i]
        if ch == ";" and not in_dollar_quote:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt + ";")
            current = []
        else:
            current.append(ch)
        i += 1

    # Trailing statement without semicolon
    last = "".join(current).strip()
    if last:
        statements.append(last)

    return statements
