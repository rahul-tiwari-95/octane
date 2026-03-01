"""Unit tests for MigrationRunner.

All tests mock asyncpg — no real Postgres connection needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from octane.tools.migrations import MigrationRunner, _split_sql, SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# 1. _split_sql utility
# ─────────────────────────────────────────────────────────────────────────────

def test_split_sql_basic():
    """Splits a two-statement SQL file correctly."""
    sql = "CREATE TABLE foo (id SERIAL);\nCREATE INDEX idx ON foo(id);"
    stmts = _split_sql(sql)
    assert len(stmts) == 2
    assert "foo" in stmts[0]
    assert "idx" in stmts[1]


def test_split_sql_strips_line_comments():
    """Line comments are stripped before splitting."""
    sql = "-- comment\nCREATE TABLE bar (id INT); -- inline\nSELECT 1;"
    stmts = _split_sql(sql)
    assert len(stmts) == 2
    assert "--" not in stmts[0]


def test_split_sql_empty_statements_ignored():
    """Empty statements (e.g. double semicolons) are not returned."""
    sql = "SELECT 1;;\nSELECT 2;"
    stmts = _split_sql(sql)
    assert len(stmts) == 2


def test_split_sql_no_trailing_garbage():
    """Trailing whitespace-only content after last `;` is not returned."""
    sql = "SELECT 1;\n\n   "
    stmts = _split_sql(sql)
    assert len(stmts) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCHEMA_VERSION
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_version_matches_versions_list():
    """SCHEMA_VERSION must be the last entry in MigrationRunner.VERSIONS."""
    assert SCHEMA_VERSION == MigrationRunner.VERSIONS[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 3. migrate() — no-op when already applied
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_migrate_noop_when_current():
    """migrate() returns applied=False when all versions already applied."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{"version": "20A"}])
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock()
    mock_conn.close = AsyncMock()

    # _list_tables
    mock_conn.fetch.side_effect = [
        [{"version": "20A"}],                        # schema_migrations query
        [{"table_name": "projects"}, {"table_name": "embeddings"}],  # _list_tables
    ]

    runner = MigrationRunner(dsn="postgresql://fake")
    with patch("asyncpg.connect", return_value=mock_conn):
        result = await runner.migrate()

    assert result.applied is False
    assert result.version == "20A"
    assert "projects" in result.tables


# ─────────────────────────────────────────────────────────────────────────────
# 4. migrate() — applies when pending
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_migrate_applies_when_pending():
    """migrate() returns applied=True when pending versions exist."""
    import asyncpg  # noqa: F401 — ensure importable

    # Simulate: no versions applied yet
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.close = AsyncMock()

    # transaction() context manager
    mock_tx = AsyncMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=mock_tx)

    call_count = 0

    async def _fetch(query, *args):
        nonlocal call_count
        call_count += 1
        if "schema_migrations" in query and call_count == 1:
            return []   # nothing applied yet
        # _list_tables
        return [{"table_name": "projects"}, {"table_name": "embeddings"},
                {"table_name": "schema_migrations"}]

    mock_conn.fetch = AsyncMock(side_effect=_fetch)

    runner = MigrationRunner(dsn="postgresql://fake")
    with patch("asyncpg.connect", return_value=mock_conn):
        result = await runner.migrate()

    assert result.applied is True
    assert result.version == "20A"
    assert len(result.tables) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. status() — pg unavailable
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_pg_unavailable():
    """status() returns pg_available=False when Postgres is unreachable."""
    import asyncpg

    runner = MigrationRunner(dsn="postgresql://fake")
    with patch("asyncpg.connect", side_effect=Exception("connection refused")):
        status = await runner.status()

    assert status.pg_available is False
    assert status.applied_versions == []
    assert status.table_counts == {}


# ─────────────────────────────────────────────────────────────────────────────
# 6. status() — returns table counts
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_returns_table_counts():
    """status() includes per-table row counts."""
    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()

    async def _fetchval(query, *args):
        if "information_schema.tables" in query and "schema_migrations" in query:
            return True   # table exists
        return 5   # row count

    mock_conn.fetchval = AsyncMock(side_effect=_fetchval)
    mock_conn.fetch = AsyncMock(side_effect=[
        [{"version": "20A"}],              # applied versions
        [{"table_name": "projects"}, {"table_name": "embeddings"}],  # table list
    ])

    runner = MigrationRunner(dsn="postgresql://fake")
    with patch("asyncpg.connect", return_value=mock_conn):
        status = await runner.status()

    assert status.pg_available is True
    assert "20A" in status.applied_versions
    assert "projects" in status.table_counts
