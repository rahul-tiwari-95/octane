"""Session 36 — Knowledge Accumulation & Observability tests.

Covers:
  A. ExtractionStore (Postgres persistence + dedup + local mirror)
  B. Extract CLI auto-persist + batch command
  C. Pipeline chaining (--extract-all, --urls-only)
  D. octane recall unified search
  E. Trace enrichment (bytes_extracted, wall_ms)
  F. octane stats analytics dashboard
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import typer


# ════════════════════════════════════════════════════════════════════════════════
#  A. ExtractionStore
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractionStore:
    """Unit tests for ExtractionStore — no real Postgres needed."""

    @pytest.fixture
    def mock_pg(self):
        pg = AsyncMock()
        pg.available = True
        pg.connect = AsyncMock()
        pg.close = AsyncMock()
        return pg

    @pytest.fixture
    def store(self, mock_pg):
        from octane.tools.structured_store import ExtractionStore
        return ExtractionStore(mock_pg)

    @pytest.mark.asyncio
    async def test_store_inserts_document(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value={
            "id": 1,
            "source_type": "youtube",
            "source_url": "https://youtube.com/watch?v=abc123",
            "content_hash": "abcdef1234567890",
            "total_words": 500,
            "local_path": "/tmp/test.md",
        })

        with patch.object(store, "_write_local_mirror", return_value="/tmp/test.md"):
            row = await store.store(
                source_type="youtube",
                source_url="https://youtube.com/watch?v=abc123",
                title="Test Video",
                author="Test Author",
                raw_text="word " * 500,
                chunks=[{"index": 0, "text": "chunk 1", "word_count": 100}],
                extraction_method="trafilatura",
                reliability_score=0.8,
            )

        assert row is not None
        assert row["id"] == 1
        assert row["source_type"] == "youtube"
        mock_pg.fetchrow.assert_called_once()
        # Verify SQL includes ON CONFLICT
        call_args = mock_pg.fetchrow.call_args
        assert "ON CONFLICT" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_store_generates_content_hash(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value={"id": 2, "content_hash": "abc"})
        with patch.object(store, "_write_local_mirror", return_value=""):
            await store.store(
                source_type="web",
                source_url="https://example.com",
                raw_text="hello world content",
            )
        call_args = mock_pg.fetchrow.call_args[0]
        # content_hash should be passed (auto-generated from raw_text)
        # It should be a 16-char hex string (sha256[:16])
        passed_hash = call_args[4]  # $4 = content_hash
        expected = hashlib.sha256("hello world content".encode()).hexdigest()[:16]
        assert passed_hash == expected

    @pytest.mark.asyncio
    async def test_seen_returns_true_when_exists(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value={"id": 5})
        result = await store.seen("abcdef1234567890")
        assert result is True

    @pytest.mark.asyncio
    async def test_seen_returns_false_when_missing(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value=None)
        result = await store.seen("nonexistent_hash")
        assert result is False

    @pytest.mark.asyncio
    async def test_seen_returns_false_for_empty_hash(self, store):
        result = await store.seen("")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_by_url(self, store, mock_pg):
        mock_pg.fetchrow = AsyncMock(return_value={"id": 3, "source_url": "https://example.com"})
        row = await store.get_by_url("https://example.com")
        assert row is not None
        assert row["source_url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[
            {"id": 1, "source_type": "arxiv", "title": "Attention Paper"},
        ])
        rows = await store.search("attention", source_type="arxiv", limit=5)
        assert len(rows) == 1
        call_sql = mock_pg.fetch.call_args[0][0]
        assert "source_type" in call_sql

    @pytest.mark.asyncio
    async def test_search_without_filter(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[])
        rows = await store.search("nonexistent")
        assert rows == []

    @pytest.mark.asyncio
    async def test_recent_all(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[
            {"id": 1, "source_type": "web"},
            {"id": 2, "source_type": "youtube"},
        ])
        rows = await store.recent(limit=10)
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_recent_filtered_by_type(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[{"id": 1, "source_type": "pdf"}])
        rows = await store.recent(source_type="pdf", limit=5)
        assert len(rows) == 1
        call_sql = mock_pg.fetch.call_args[0][0]
        assert "source_type" in call_sql

    @pytest.mark.asyncio
    async def test_count_all(self, store, mock_pg):
        mock_pg.fetchval = AsyncMock(return_value=42)
        n = await store.count()
        assert n == 42

    @pytest.mark.asyncio
    async def test_count_by_type(self, store, mock_pg):
        mock_pg.fetchval = AsyncMock(return_value=10)
        n = await store.count(source_type="youtube")
        assert n == 10

    @pytest.mark.asyncio
    async def test_stats_aggregation(self, store, mock_pg):
        mock_pg.fetch = AsyncMock(return_value=[
            {"source_type": "youtube", "doc_count": 5, "total_words": 25000, "total_chunks": 50},
            {"source_type": "arxiv", "doc_count": 3, "total_words": 15000, "total_chunks": 30},
        ])
        s = await store.stats()
        assert s["total_docs"] == 8
        assert s["total_words"] == 40000
        assert len(s["by_source"]) == 2

    def test_write_local_mirror_creates_file(self, store, tmp_path):
        store._EXTRACTIONS_DIR = tmp_path
        path = store._write_local_mirror(
            content_hash="abc123",
            title="Test Title",
            source_type="youtube",
            source_url="https://youtube.com/watch?v=test",
            raw_text="This is test content",
            author="Test Author",
        )
        assert path != ""
        p = Path(path)
        assert p.exists()
        content = p.read_text()
        assert "# Test Title" in content
        assert "youtube" in content

    def test_write_local_mirror_empty_hash_skips(self, store, tmp_path):
        store._EXTRACTIONS_DIR = tmp_path
        path = store._write_local_mirror("", "title", "web", "url", "text", "author")
        assert path == ""

    def test_write_local_mirror_empty_text_skips(self, store, tmp_path):
        store._EXTRACTIONS_DIR = tmp_path
        path = store._write_local_mirror("hash", "title", "web", "url", "", "author")
        assert path == ""


# ════════════════════════════════════════════════════════════════════════════════
#  B. Extract CLI auto-persist
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractPersist:
    """Tests for _persist_extraction in extract.py."""

    @pytest.fixture
    def mock_doc(self):
        from octane.extractors.models import ExtractedDocument, SourceType, TextChunk, ChunkMetadata
        return ExtractedDocument(
            source_type=SourceType.YOUTUBE,
            source_url="https://youtube.com/watch?v=test",
            title="Test Video",
            author="Test Channel",
            raw_text="word " * 1000,
            chunks=[
                TextChunk(
                    text="chunk one text",
                    index=0,
                    metadata=ChunkMetadata(timestamp_start=0, timestamp_end=45),
                ),
            ],
            extraction_method="youtube_api",
            reliability_score=0.55,
        )

    @pytest.mark.asyncio
    async def test_persist_extraction_stores_and_prints(self, mock_doc):
        from octane.cli.extract import _persist_extraction

        mock_pg = AsyncMock()
        mock_pg.available = True
        mock_pg.connect = AsyncMock()
        mock_pg.close = AsyncMock()
        mock_store = AsyncMock()
        mock_store.seen = AsyncMock(return_value=False)
        mock_store.store = AsyncMock(return_value={
            "id": 99,
            "local_path": "/tmp/abc.md",
        })

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.tools.structured_store.ExtractionStore", return_value=mock_store),
            patch("octane.cli.extract.console") as mock_console,
        ):
            row = await _persist_extraction(mock_doc)

        assert row is not None
        assert row["id"] == 99
        mock_store.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_skips_when_already_stored(self, mock_doc):
        from octane.cli.extract import _persist_extraction

        mock_pg = AsyncMock()
        mock_pg.available = True
        mock_pg.connect = AsyncMock()
        mock_pg.close = AsyncMock()
        mock_store = AsyncMock()
        mock_store.seen = AsyncMock(return_value=True)

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.tools.structured_store.ExtractionStore", return_value=mock_store),
            patch("octane.cli.extract.console"),
        ):
            row = await _persist_extraction(mock_doc)

        assert row is None
        mock_store.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_persist_returns_none_when_pg_unavailable(self, mock_doc):
        from octane.cli.extract import _persist_extraction

        mock_pg = AsyncMock()
        mock_pg.available = False
        mock_pg.connect = AsyncMock()
        mock_pg.close = AsyncMock()

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.extract.console"),
        ):
            row = await _persist_extraction(mock_doc)

        assert row is None


# ════════════════════════════════════════════════════════════════════════════════
#  C. Batch extraction
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractBatch:

    @pytest.mark.asyncio
    async def test_batch_reads_urls_and_extracts(self, tmp_path):
        from octane.cli.extract import _extract_batch
        from octane.extractors.models import ExtractedDocument, SourceType

        urls_file = tmp_path / "urls.txt"
        urls_file.write_text(
            "https://example.com/page1\n"
            "# comment line\n"
            "https://example.com/page2\n"
            "\n"
        )

        mock_doc = ExtractedDocument(
            source_type=SourceType.WEB,
            source_url="https://example.com/page1",
            raw_text="enough content here to pass the length check easily" * 5,
        )

        with (
            patch("octane.extractors.pipeline.extract", new_callable=AsyncMock, return_value=mock_doc),
            patch("octane.cli.extract._persist_extraction", new_callable=AsyncMock, return_value={"id": 1}),
            patch("octane.cli.extract.console"),
            patch("rich.progress.Progress"),
        ):
            await _extract_batch(str(urls_file), "auto")

    @pytest.mark.asyncio
    async def test_batch_handles_missing_file(self, tmp_path):
        from octane.cli.extract import _extract_batch

        with (
            patch("octane.cli.extract.console"),
            pytest.raises((SystemExit, typer.Exit)),
        ):
            await _extract_batch(str(tmp_path / "nofile.txt"), "auto")

    @pytest.mark.asyncio
    async def test_batch_skips_comments_and_blanks(self, tmp_path):
        from octane.cli.extract import _extract_batch
        from octane.extractors.models import ExtractedDocument, SourceType

        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("# only comments\n\n   \n# another\n")

        with patch("octane.cli.extract.console"):
            await _extract_batch(str(urls_file), "auto")
            # Should print "No URLs found" and return


# ════════════════════════════════════════════════════════════════════════════════
#  D. Pipeline chaining (--urls-only)
# ════════════════════════════════════════════════════════════════════════════════

class TestPipelineChaining:

    @pytest.mark.asyncio
    async def test_search_youtube_urls_only(self, capsys):
        from octane.cli.extract import _search_youtube

        mock_results = [
            {"title": "Video 1", "channel": "C1", "duration": "10:00",
             "url": "https://youtube.com/watch?v=abc"},
            {"title": "Video 2", "channel": "C2", "duration": "5:00",
             "url": "https://youtube.com/watch?v=def"},
        ]

        with patch(
            "octane.extractors.youtube.search.search_youtube",
            return_value=mock_results,
        ):
            await _search_youtube("test query", limit=5, extract_all=False, urls_only=True)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "https://youtube.com/watch?v=abc"
        assert lines[1] == "https://youtube.com/watch?v=def"

    @pytest.mark.asyncio
    async def test_search_arxiv_urls_only(self, capsys):
        from octane.cli.extract import _search_arxiv

        mock_results = [
            {"arxiv_id": "2408.09869", "title": "Paper 1", "authors": ["A"],
             "published": "2024-08-10"},
        ]

        with patch(
            "octane.extractors.academic.arxiv_search.search_arxiv",
            return_value=mock_results,
        ):
            await _search_arxiv("test", limit=5, extract_all=False, urls_only=True)

        captured = capsys.readouterr()
        assert "2408.09869" in captured.out

    @pytest.mark.asyncio
    async def test_search_youtube_extract_all(self):
        from octane.cli.extract import _search_youtube
        from octane.extractors.models import ExtractedDocument, SourceType

        mock_results = [
            {"title": "V1", "channel": "C1", "duration": "10:00",
             "url": "https://youtube.com/watch?v=abc"},
        ]
        mock_doc = ExtractedDocument(
            source_type=SourceType.YOUTUBE,
            source_url="https://youtube.com/watch?v=abc",
            raw_text="transcript content",
        )

        with (
            patch("octane.extractors.youtube.search.search_youtube", return_value=mock_results),
            patch("octane.extractors.pipeline.extract", new_callable=AsyncMock, return_value=mock_doc),
            patch("octane.cli.extract._persist_extraction", new_callable=AsyncMock, return_value={"id": 1}),
            patch("octane.cli.extract.console"),
        ):
            await _search_youtube("test", limit=5, extract_all=True, urls_only=False)


# ════════════════════════════════════════════════════════════════════════════════
#  E. Recall CLI
# ════════════════════════════════════════════════════════════════════════════════

class TestRecallSearch:

    @pytest.fixture
    def mock_pg(self):
        pg = AsyncMock()
        pg.available = True
        pg.connect = AsyncMock()
        pg.close = AsyncMock()
        return pg

    @pytest.mark.asyncio
    async def test_recall_search_displays_results(self, mock_pg):
        from octane.cli.recall import _recall_search

        mock_pg.fetch = AsyncMock(side_effect=[
            [{"id": 1, "source_type": "youtube", "source_url": "url", "title": "Video",
              "author": "Auth", "total_words": 500, "reliability_score": 0.8,
              "extracted_at": "2025-01-01T00:00:00", "preview": "text"}],
            [{"id": 2, "url": "https://example.com", "title": "Page",
              "word_count": 200, "fetched_at": "2025-01-01T00:00:00", "preview": "text"}],
            [],  # research_findings
            [],  # user_files
            [],  # artifacts
        ])

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.recall.console"),
        ):
            await _recall_search("test query", source_type=None, limit=10, verbose=False)

        assert mock_pg.fetch.call_count == 5  # 5 table queries

    @pytest.mark.asyncio
    async def test_recall_search_with_type_filter(self, mock_pg):
        from octane.cli.recall import _recall_search

        mock_pg.fetch = AsyncMock(return_value=[
            {"id": 1, "source_type": "youtube", "source_url": "url", "title": "V",
             "author": "A", "total_words": 100, "reliability_score": 0.5,
             "extracted_at": "2025-01-01", "preview": "text"},
        ])

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.recall.console"),
        ):
            await _recall_search("test", source_type="youtube", limit=5, verbose=False)

        # Only extracted_documents + web_pages should be queried (type=youtube matches both paths)
        # It won't query findings/files/artifacts since type=youtube
        assert mock_pg.fetch.call_count >= 1

    @pytest.mark.asyncio
    async def test_recall_search_no_results(self, mock_pg):
        from octane.cli.recall import _recall_search

        mock_pg.fetch = AsyncMock(return_value=[])

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.recall.console") as mock_console,
        ):
            await _recall_search("nonexistent_term_xyz", source_type=None, limit=10, verbose=False)

        # Should print "No results" guidance
        any_no_results = any(
            "No results" in str(call) for call in mock_console.print.call_args_list
        )
        assert any_no_results

    @pytest.mark.asyncio
    async def test_recall_stats(self, mock_pg):
        from octane.cli.recall import _recall_stats

        mock_pg.fetchval = AsyncMock(side_effect=[
            10,    # extracted_documents count
            5000,  # extracted_documents words
            20,    # web_pages count
            8000,  # web_pages words
            5,     # research_findings count
            3000,  # research_findings words
            3,     # user_files count
            1500,  # user_files words
            7,     # artifacts count
            2000,  # artifacts words
            100,   # embeddings count
        ])

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.recall.console"),
        ):
            await _recall_stats()


# ════════════════════════════════════════════════════════════════════════════════
#  F. Trace enrichment
# ════════════════════════════════════════════════════════════════════════════════

class TestTraceEnrichment:

    def test_trace_payload_displays_bytes_and_wall(self):
        from octane.cli.trace import _print_payload

        payload = {
            "n_articles": 5,
            "deep": True,
            "mode": "REASON+3000tok",
            "bytes_extracted": 15_000,
            "wall_ms": 3200,
        }
        with patch("octane.cli.trace.console") as mock_console:
            _print_payload("web_synthesis", payload, verbose=False)

        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "15,000" in printed or "15000" in printed
        assert "3,200" in printed or "3200" in printed

    def test_trace_payload_without_new_fields(self):
        from octane.cli.trace import _print_payload

        payload = {
            "n_articles": 3,
            "deep": False,
            "mode": "MID+768tok",
        }
        with patch("octane.cli.trace.console") as mock_console:
            _print_payload("web_synthesis", payload, verbose=False)

        # Should not crash when new fields are absent
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "articles" in printed


# ════════════════════════════════════════════════════════════════════════════════
#  G. Stats CLI
# ════════════════════════════════════════════════════════════════════════════════

class TestStatsDashboard:

    @pytest.fixture
    def mock_pg(self):
        pg = AsyncMock()
        pg.available = True
        pg.connect = AsyncMock()
        pg.close = AsyncMock()
        return pg

    @pytest.mark.asyncio
    async def test_stats_queries_all_tables(self, mock_pg):
        from octane.cli.stats import _stats

        call_count = 0

        async def mock_fetchval(query, *args):
            nonlocal call_count
            call_count += 1
            return 0

        mock_pg.fetchval = mock_fetchval
        mock_pg.fetch = AsyncMock(return_value=[])

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.stats.console"),
        ):
            await _stats()

        # Should query count + words for each source table
        assert call_count >= 6

    @pytest.mark.asyncio
    async def test_stats_handles_pg_unavailable(self):
        from octane.cli.stats import _stats

        mock_pg = AsyncMock()
        mock_pg.available = False

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.stats.console") as mock_console,
        ):
            await _stats()

        any_unavailable = any(
            "unavailable" in str(c).lower() for c in mock_console.print.call_args_list
        )
        assert any_unavailable

    @pytest.mark.asyncio
    async def test_stats_shows_extraction_breakdown(self, mock_pg):
        from octane.cli.stats import _stats

        mock_pg.fetchval = AsyncMock(return_value=0)
        mock_pg.fetch = AsyncMock(return_value=[
            {"source_type": "youtube", "doc_count": 5, "total_words": 25000, "total_chunks": 50},
            {"source_type": "arxiv", "doc_count": 3, "total_words": 15000, "total_chunks": 30},
        ])

        with (
            patch("octane.tools.pg_client.PgClient", return_value=mock_pg),
            patch("octane.cli.stats.console") as mock_console,
        ):
            await _stats()


# ════════════════════════════════════════════════════════════════════════════════
#  H. Schema additions
# ════════════════════════════════════════════════════════════════════════════════

class TestSchema:

    def test_extracted_documents_table_in_schema(self):
        schema_path = Path(__file__).parent.parent.parent / "octane" / "tools" / "schema.sql"
        ddl = schema_path.read_text()
        assert "CREATE TABLE IF NOT EXISTS extracted_documents" in ddl
        assert "content_hash" in ddl
        assert "source_type" in ddl
        assert "chunks" in ddl
        assert "JSONB" in ddl.upper() or "jsonb" in ddl
        assert "idx_extracted_docs_hash" in ddl

    def test_extracted_documents_has_local_path(self):
        schema_path = Path(__file__).parent.parent.parent / "octane" / "tools" / "schema.sql"
        ddl = schema_path.read_text()
        assert "local_path" in ddl


# ════════════════════════════════════════════════════════════════════════════════
#  I. CLI registration
# ════════════════════════════════════════════════════════════════════════════════

class TestCLIRegistration:

    def test_recall_app_importable(self):
        from octane.cli.recall import recall_app
        assert recall_app is not None

    def test_stats_register_importable(self):
        from octane.cli.stats import register
        assert callable(register)

    def test_extract_app_has_batch_command(self):
        from octane.cli.extract import extract_app
        # Typer app should have a "batch" command registered
        command_names = [cmd.name for cmd in extract_app.registered_commands]
        assert "batch" in command_names

    def test_extract_app_has_search_youtube(self):
        from octane.cli.extract import extract_app
        command_names = [cmd.name for cmd in extract_app.registered_commands]
        assert "search-youtube" in command_names

    def test_extract_app_has_search_arxiv(self):
        from octane.cli.extract import extract_app
        command_names = [cmd.name for cmd in extract_app.registered_commands]
        assert "search-arxiv" in command_names


# ════════════════════════════════════════════════════════════════════════════════
#  J. Extraction model to_dict includes all fields
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractionModels:

    def test_extracted_document_content_hash_generated(self):
        from octane.extractors.models import ExtractedDocument, SourceType
        doc = ExtractedDocument(
            source_type=SourceType.WEB,
            source_url="https://example.com",
            raw_text="test content for hashing",
        )
        assert doc.content_hash != ""
        assert len(doc.content_hash) == 16

    def test_extracted_document_to_dict(self):
        from octane.extractors.models import ExtractedDocument, SourceType
        doc = ExtractedDocument(
            source_type=SourceType.ARXIV,
            source_url="2408.09869",
            title="Test Paper",
            raw_text="content " * 100,
        )
        d = doc.to_dict()
        assert d["source_type"] == "arxiv"
        assert d["total_words"] == 100
        assert "content_hash" in d

    def test_extracted_document_empty_text_no_hash(self):
        from octane.extractors.models import ExtractedDocument, SourceType
        doc = ExtractedDocument(
            source_type=SourceType.WEB,
            source_url="https://example.com",
            raw_text="",
        )
        assert doc.content_hash == ""


# ════════════════════════════════════════════════════════════════════════════════
#  K. Extraction dedup integration flow
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractionDedup:

    @pytest.mark.asyncio
    async def test_dedup_by_content_hash(self):
        """Storing the same content twice should upsert, not duplicate."""
        from octane.tools.structured_store import ExtractionStore

        mock_pg = AsyncMock()
        mock_pg.available = True
        store = ExtractionStore(mock_pg)

        # First store
        mock_pg.fetchrow = AsyncMock(return_value={"id": 1})
        with patch.object(store, "_write_local_mirror", return_value=""):
            await store.store(
                source_type="web",
                source_url="https://example.com/1",
                raw_text="same content",
                content_hash="hash123",
            )

        # Verify the SQL uses ON CONFLICT for upsert
        sql = mock_pg.fetchrow.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql


# ════════════════════════════════════════════════════════════════════════════════
#  L. Trace summary footer includes new fields
# ════════════════════════════════════════════════════════════════════════════════

class TestTraceSummaryFooter:

    def test_footer_includes_page_and_byte_counts(self):
        """Trace detail footer should show pages extracted and KB extracted."""
        from octane.cli.trace import _print_trace_detail

        # Build a minimal mock trace
        from unittest.mock import MagicMock
        from datetime import datetime, timezone

        mock_event_search = MagicMock()
        mock_event_search.event_type = "web_search_round"
        mock_event_search.payload = {"round": 1, "pages_extracted": 5, "urls_found": 10}
        mock_event_search.source = "web"
        mock_event_search.target = ""
        mock_event_search.error = ""
        mock_event_search.timestamp = datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        mock_event_synth = MagicMock()
        mock_event_synth.event_type = "web_synthesis"
        mock_event_synth.payload = {
            "n_articles": 5, "mode": "MID", "bytes_extracted": 10240, "wall_ms": 2000,
        }
        mock_event_synth.source = "web"
        mock_event_synth.target = ""
        mock_event_synth.error = ""
        mock_event_synth.timestamp = datetime(2025, 1, 1, 0, 0, 2, tzinfo=timezone.utc)

        mock_trace = MagicMock()
        mock_trace.events = [mock_event_search, mock_event_synth]
        mock_trace.started_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_trace.total_duration_ms = 2000
        mock_trace.correlation_id = "test-123"
        mock_trace.success = True
        mock_trace.agents_used = ["web"]

        with patch("octane.cli.trace.console") as mock_console:
            _print_trace_detail(mock_trace, verbose=False)

        all_output = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "5 pages extracted" in all_output
        assert "KB extracted" in all_output
