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
SCHEMA_VERSION = "36A"

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
    VERSIONS: list[str] = ["20A", "28A", "29A", "35A", "36A"]

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

        # ── Step 1: try to enable pgvector OUTSIDE the transaction.
        # CREATE EXTENSION cannot run inside a transaction block, and if the
        # extension is not installed on the system it raises an error that
        # would otherwise roll back all subsequent DDL.  We handle it here so
        # the rest of the schema always applies regardless.
        vector_available = False
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            vector_available = True
        except Exception as ext_exc:
            logger.warning(
                "pgvector_extension_unavailable",
                hint="Install pgvector (brew install pgvector) to enable semantic search",
                error=str(ext_exc),
            )

        # Strip the extension statement from the SQL so it is not re-executed
        # inside the transaction below (avoids "cannot run inside transaction" errors).
        # If vector is unavailable also strip any statement that references the
        # vector type (i.e. the embeddings table and its indexes) — those use
        # vector(384) columns which require the extension to be loaded.
        def _keep_stmt(s: str) -> bool:
            su = s.strip().upper()
            if not su:
                return False
            if "CREATE EXTENSION" in su:
                return False
            # When vector is unavailable, skip the embeddings table AND all
            # indexes/statements that reference it (e.g. idx_embeddings_source
            # has no VECTOR keyword but the table won't exist).
            if not vector_available and ("VECTOR" in su or "EMBEDDINGS" in su):
                return False
            return True

        cleaned_stmts = [s for s in _split_sql(sql) if _keep_stmt(s)]

        # ── Step 2: incremental migrations (run after the baseline schema) ──
        # 28A: add provenance JSONB column for chain-of-custody tracking
        _MIGRATION_28A = """
            ALTER TABLE web_pages
                ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}';
            ALTER TABLE research_findings_v2
                ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}';
            ALTER TABLE generated_artifacts
                ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}';
            CREATE INDEX IF NOT EXISTS idx_web_pages_provenance
                ON web_pages USING GIN (provenance);
            CREATE INDEX IF NOT EXISTS idx_rfv2_provenance
                ON research_findings_v2 USING GIN (provenance);
        """

        # 29A: add broker/account detail and sector classification to portfolio_positions
        _MIGRATION_29A = """
            ALTER TABLE portfolio_positions
                ADD COLUMN IF NOT EXISTS broker      TEXT NOT NULL DEFAULT '';
            ALTER TABLE portfolio_positions
                ADD COLUMN IF NOT EXISTS account_id  TEXT NOT NULL DEFAULT '';
            ALTER TABLE portfolio_positions
                ADD COLUMN IF NOT EXISTS sector      TEXT NOT NULL DEFAULT '';
            ALTER TABLE portfolio_positions
                ADD COLUMN IF NOT EXISTS asset_class TEXT NOT NULL DEFAULT 'equity';
            CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_upsert
                ON portfolio_positions (ticker, broker, account_id);
        """

        # 35A: add tax_lots, dividends, net_worth_snapshots, crypto_positions
        _MIGRATION_35A = """
            CREATE TABLE IF NOT EXISTS tax_lots (
                id              SERIAL      PRIMARY KEY,
                position_id     INTEGER     REFERENCES portfolio_positions(id) ON DELETE CASCADE,
                ticker          TEXT        NOT NULL,
                shares          REAL        NOT NULL DEFAULT 0,
                cost_per_share  REAL        NOT NULL DEFAULT 0,
                purchase_date   DATE        NOT NULL DEFAULT CURRENT_DATE,
                broker          TEXT        NOT NULL DEFAULT '',
                account_id      TEXT        NOT NULL DEFAULT '',
                sold_shares     REAL        NOT NULL DEFAULT 0,
                notes           TEXT        NOT NULL DEFAULT '',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_tax_lots_position ON tax_lots (position_id);
            CREATE INDEX IF NOT EXISTS idx_tax_lots_ticker   ON tax_lots (ticker);

            CREATE TABLE IF NOT EXISTS dividends (
                id              SERIAL      PRIMARY KEY,
                ticker          TEXT        NOT NULL,
                amount          REAL        NOT NULL DEFAULT 0,
                ex_date         DATE,
                pay_date        DATE,
                frequency       TEXT        NOT NULL DEFAULT 'quarterly',
                div_yield       REAL        NOT NULL DEFAULT 0,
                payout_ratio    REAL        NOT NULL DEFAULT 0,
                growth_rate     REAL        NOT NULL DEFAULT 0,
                source          TEXT        NOT NULL DEFAULT 'yfinance',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_dividends_ticker  ON dividends (ticker);
            CREATE INDEX IF NOT EXISTS idx_dividends_ex_date ON dividends (ex_date);

            CREATE TABLE IF NOT EXISTS net_worth_snapshots (
                id              SERIAL      PRIMARY KEY,
                snapshot_date   DATE        NOT NULL DEFAULT CURRENT_DATE,
                total_value     REAL        NOT NULL DEFAULT 0,
                equities_value  REAL        NOT NULL DEFAULT 0,
                crypto_value    REAL        NOT NULL DEFAULT 0,
                cash_value      REAL        NOT NULL DEFAULT 0,
                position_count  INTEGER     NOT NULL DEFAULT 0,
                notes           TEXT        NOT NULL DEFAULT '',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_nw_snapshots_date ON net_worth_snapshots (snapshot_date);

            CREATE TABLE IF NOT EXISTS crypto_positions (
                id              SERIAL      PRIMARY KEY,
                coin            TEXT        NOT NULL,
                quantity        REAL        NOT NULL DEFAULT 0,
                cost_per_coin   REAL        NOT NULL DEFAULT 0,
                exchange        TEXT        NOT NULL DEFAULT '',
                wallet_address  TEXT        NOT NULL DEFAULT '',
                notes           TEXT        NOT NULL DEFAULT '',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_crypto_coin ON crypto_positions (coin);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_crypto_upsert ON crypto_positions (coin, exchange);
        """

        # 36A: add extracted_documents table for rich extraction persistence
        _MIGRATION_36A = """
            CREATE TABLE IF NOT EXISTS extracted_documents (
                id                  SERIAL      PRIMARY KEY,
                project_id          INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
                source_type         TEXT        NOT NULL DEFAULT 'web',
                source_url          TEXT        NOT NULL,
                content_hash        TEXT        NOT NULL DEFAULT '',
                title               TEXT        NOT NULL DEFAULT '',
                author              TEXT        NOT NULL DEFAULT '',
                raw_text            TEXT        NOT NULL DEFAULT '',
                chunks              JSONB       NOT NULL DEFAULT '[]',
                total_words         INTEGER     NOT NULL DEFAULT 0,
                total_chunks        INTEGER     NOT NULL DEFAULT 0,
                extraction_method   TEXT        NOT NULL DEFAULT '',
                reliability_score   REAL        NOT NULL DEFAULT 0.5,
                metadata            JSONB       NOT NULL DEFAULT '{}',
                local_path          TEXT        NOT NULL DEFAULT '',
                extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_extracted_docs_hash ON extracted_documents (content_hash) WHERE content_hash != '';
            CREATE INDEX IF NOT EXISTS idx_extracted_docs_source   ON extracted_documents (source_type);
            CREATE INDEX IF NOT EXISTS idx_extracted_docs_url      ON extracted_documents (source_url);
            CREATE INDEX IF NOT EXISTS idx_extracted_docs_project  ON extracted_documents (project_id);
            CREATE INDEX IF NOT EXISTS idx_extracted_docs_created  ON extracted_documents (extracted_at DESC);
        """

        _INCREMENTAL: dict[str, str] = {
            "28A": _MIGRATION_28A,
            "29A": _MIGRATION_29A,
            "35A": _MIGRATION_35A,
            "36A": _MIGRATION_36A,
        }

        # Apply within a single transaction.
        # Incremental migrations run BEFORE baseline indexes so that columns
        # added by migrations (e.g. broker in 29A) exist when the baseline
        # schema tries to CREATE INDEX on them.
        try:
            async with conn.transaction():
                # 1. Baseline CREATE TABLE statements (idempotent IF NOT EXISTS)
                #    Split into table-creation vs indexes: tables first, then
                #    incrementals, then baseline indexes.
                table_stmts = []
                index_stmts = []
                for stmt in cleaned_stmts:
                    if stmt.strip().upper().startswith("CREATE INDEX") or \
                       stmt.strip().upper().startswith("CREATE UNIQUE INDEX"):
                        index_stmts.append(stmt)
                    else:
                        table_stmts.append(stmt)

                for stmt in table_stmts:
                    await conn.execute(stmt)

                # 2. Incremental migrations (add columns, etc.)
                for v in pending:
                    if v in _INCREMENTAL:
                        for stmt in _split_sql(_INCREMENTAL[v]):
                            if stmt.strip():
                                await conn.execute(stmt)

                # 3. Baseline indexes (now columns from incrementals exist)
                for stmt in index_stmts:
                    await conn.execute(stmt)

                # 4. Record all newly applied versions
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
