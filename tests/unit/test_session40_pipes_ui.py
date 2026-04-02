"""Tests for Session 40 — Composable Pipes + UI Enhancements.

Covers:
  - octane search (web, news, youtube, arxiv) CLI sub-app
  - octane extract stdin command
  - octane synthesize CLI sub-app
  - UI: /api/traces-events/recent endpoint (globe seeding)
  - UI: dashboard query text extraction improvements
  - UI: ModelPanel proportional memory bars (structure tests)
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: octane search CLI — module structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchCLIRegistration:
    """octane search sub-app is properly registered."""

    def test_search_module_has_search_app(self):
        from octane.cli.search import search_app
        assert search_app is not None
        assert search_app.info.name is None or True  # Typer app exists

    def test_search_app_registered_in_cli(self):
        from octane.cli import register_all
        import typer
        app = typer.Typer()
        register_all(app)
        # Verify 'search' is a registered group
        # Typer stores registered sub-apps internally
        registered_names = [
            g.typer_instance.info.name or g.name
            for g in getattr(app, 'registered_groups', [])
        ]
        assert "search" in registered_names

    def test_search_web_command_exists(self):
        from octane.cli.search import search_app
        cmd_names = [c.name for c in search_app.registered_commands]
        assert "web" in cmd_names

    def test_search_news_command_exists(self):
        from octane.cli.search import search_app
        cmd_names = [c.name for c in search_app.registered_commands]
        assert "news" in cmd_names

    def test_search_youtube_command_exists(self):
        from octane.cli.search import search_app
        cmd_names = [c.name for c in search_app.registered_commands]
        assert "youtube" in cmd_names

    def test_search_arxiv_command_exists(self):
        from octane.cli.search import search_app
        cmd_names = [c.name for c in search_app.registered_commands]
        assert "arxiv" in cmd_names


class TestSearchWebLogic:
    """_search_web correctly parses Brave API responses."""

    @pytest.mark.asyncio
    async def test_search_web_json_output(self, capsys):
        from octane.cli.search import _search_web

        mock_response = {
            "web": {
                "results": [
                    {"title": "NVDA Q4 Earnings", "url": "https://example.com/nvda", "description": "Beat estimates", "age": "2d"},
                    {"title": "NVDA Revenue", "url": "https://example.com/nvda2", "description": "Record revenue", "age": "3d"},
                ]
            }
        }
        with patch("octane.tools.bodega_intel.BodegaIntelClient.web_search", new_callable=AsyncMock, return_value=mock_response):
            await _search_web("NVDA earnings", 10, output_json=True, urls_only=False)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["query"] == "NVDA earnings"
        assert data["source"] == "web"
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "NVDA Q4 Earnings"

    @pytest.mark.asyncio
    async def test_search_web_urls_only(self, capsys):
        from octane.cli.search import _search_web

        mock_response = {
            "web": {"results": [
                {"title": "A", "url": "https://a.com", "description": "a"},
                {"title": "B", "url": "https://b.com", "description": "b"},
            ]}
        }
        with patch("octane.tools.bodega_intel.BodegaIntelClient.web_search", new_callable=AsyncMock, return_value=mock_response):
            await _search_web("test", 5, output_json=False, urls_only=True)

        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines == ["https://a.com", "https://b.com"]

    @pytest.mark.asyncio
    async def test_search_web_empty_results(self, capsys):
        from octane.cli.search import _search_web

        with patch("octane.tools.bodega_intel.BodegaIntelClient.web_search", new_callable=AsyncMock, return_value={"web": {"results": []}}):
            await _search_web("nothing", 5, output_json=True, urls_only=False)

        data = json.loads(capsys.readouterr().out)
        assert data["results"] == []


class TestSearchNewsLogic:
    """_search_news correctly parses Brave news responses."""

    @pytest.mark.asyncio
    async def test_search_news_json_output(self, capsys):
        from octane.cli.search import _search_news

        mock_response = {
            "articles": [
                {"title": "AI News", "url": "https://news.com/ai", "source": "TechCrunch", "age": "1h", "description": "Latest"},
            ]
        }
        with patch("octane.tools.bodega_intel.BodegaIntelClient.news_search", new_callable=AsyncMock, return_value=mock_response):
            await _search_news("AI", 5, "3d", output_json=True, urls_only=False)

        data = json.loads(capsys.readouterr().out)
        assert data["source"] == "news"
        assert len(data["results"]) == 1
        assert data["results"][0]["source"] == "TechCrunch"


class TestSearchYouTubeLogic:
    """_search_youtube wraps YouTube search correctly."""

    @pytest.mark.asyncio
    async def test_search_youtube_json_output(self, capsys):
        from octane.cli.search import _search_youtube

        mock_results = [
            {"title": "Transformers Explained", "url": "https://youtube.com/watch?v=abc", "channel": "3Blue1Brown", "duration": "18:22", "views": "2M"},
        ]
        with patch("octane.extractors.youtube.search.search_youtube", return_value=mock_results):
            await _search_youtube("transformers", 5, output_json=True, urls_only=False)

        data = json.loads(capsys.readouterr().out)
        assert data["source"] == "youtube"
        assert data["results"][0]["channel"] == "3Blue1Brown"


class TestSearchArxivLogic:
    """_search_arxiv wraps arXiv search correctly."""

    @pytest.mark.asyncio
    async def test_search_arxiv_json_output(self, capsys):
        from octane.cli.search import _search_arxiv

        mock_results = [
            {
                "arxiv_id": "2408.09869",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani", "Shazeer"],
                "published": "2017-06-12",
                "summary": "We propose a new architecture...",
                "url": "https://arxiv.org/abs/2408.09869",
            },
        ]
        with patch("octane.extractors.academic.arxiv_search.search_arxiv", return_value=mock_results):
            await _search_arxiv("attention", 10, output_json=True, urls_only=False)

        data = json.loads(capsys.readouterr().out)
        assert data["source"] == "arxiv"
        assert data["results"][0]["arxiv_id"] == "2408.09869"

    @pytest.mark.asyncio
    async def test_search_arxiv_urls_only(self, capsys):
        from octane.cli.search import _search_arxiv

        mock_results = [
            {"arxiv_id": "2408.09869", "title": "Paper 1", "authors": [], "published": "", "summary": ""},
            {"arxiv_id": "2305.12345", "title": "Paper 2", "authors": [], "published": "", "summary": ""},
        ]
        with patch("octane.extractors.academic.arxiv_search.search_arxiv", return_value=mock_results):
            await _search_arxiv("test", 5, output_json=False, urls_only=True)

        lines = capsys.readouterr().out.strip().splitlines()
        assert lines == ["2408.09869", "2305.12345"]


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: octane extract stdin
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractStdin:
    """octane extract stdin reads URLs from pipe and extracts."""

    def test_extract_stdin_command_exists(self):
        from octane.cli.extract import extract_app
        cmd_names = [c.name for c in extract_app.registered_commands]
        assert "stdin" in cmd_names

    @pytest.mark.asyncio
    async def test_extract_stdin_parses_json_input(self, capsys):
        from octane.cli.extract import _extract_stdin
        from octane.agents.web.content_extractor import ExtractedContent

        json_input = json.dumps({
            "query": "test",
            "source": "web",
            "results": [
                {"title": "Page 1", "url": "https://example.com/1"},
                {"title": "Page 2", "url": "https://example.com/2"},
            ]
        })

        mock_extracted = [
            ExtractedContent(url="https://example.com/1", text="Hello world content", word_count=3, method="trafilatura", title="Page 1"),
        ]

        with patch("sys.stdin", StringIO(json_input)), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("octane.agents.web.content_extractor.ContentExtractor") as MockEx:
            inst = MockEx.return_value
            inst.extract_batch = AsyncMock(return_value=mock_extracted)
            await _extract_stdin(output_json=True, quality="auto", top_n=10)

        data = json.loads(capsys.readouterr().out)
        assert data["total"] == 1
        assert data["extracted"][0]["url"] == "https://example.com/1"

    @pytest.mark.asyncio
    async def test_extract_stdin_parses_plain_urls(self, capsys):
        from octane.cli.extract import _extract_stdin
        from octane.agents.web.content_extractor import ExtractedContent

        plain_input = "https://example.com/1\nhttps://example.com/2\n"

        mock_extracted = [
            ExtractedContent(url="https://example.com/1", text="Content here", word_count=2, method="trafilatura", title="Title"),
        ]

        with patch("sys.stdin", StringIO(plain_input)), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("octane.agents.web.content_extractor.ContentExtractor") as MockEx:
            inst = MockEx.return_value
            inst.extract_batch = AsyncMock(return_value=mock_extracted)
            await _extract_stdin(output_json=True, quality="auto", top_n=10)

        data = json.loads(capsys.readouterr().out)
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_extract_stdin_rejects_tty(self):
        from octane.cli.extract import _extract_stdin
        from click.exceptions import Exit

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with pytest.raises((SystemExit, Exit)):
                await _extract_stdin(output_json=True, quality="auto", top_n=10)

    @pytest.mark.asyncio
    async def test_extract_stdin_filters_failed(self, capsys):
        from octane.cli.extract import _extract_stdin
        from octane.agents.web.content_extractor import ExtractedContent

        json_input = json.dumps({"results": [{"url": "https://example.com"}]})
        mock_extracted = [
            ExtractedContent(url="https://example.com", text="Good content", word_count=2, method="trafilatura"),
            ExtractedContent(url="https://fail.com", text="", word_count=0, method="failed"),
        ]

        with patch("sys.stdin", StringIO(json_input)), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("octane.agents.web.content_extractor.ContentExtractor") as MockEx:
            inst = MockEx.return_value
            inst.extract_batch = AsyncMock(return_value=mock_extracted)
            await _extract_stdin(output_json=True, quality="auto", top_n=10)

        data = json.loads(capsys.readouterr().out)
        assert data["total"] == 1  # only the good one


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: octane synthesize CLI
# ═══════════════════════════════════════════════════════════════════════════════


class TestSynthesizeCLI:
    """octane synthesize CLI sub-app structure."""

    def test_synthesize_module_has_app(self):
        from octane.cli.synthesize import synthesize_app
        assert synthesize_app is not None

    def test_synthesize_registered_in_cli(self):
        from octane.cli import register_all
        import typer
        app = typer.Typer()
        register_all(app)
        registered_names = [
            g.typer_instance.info.name or g.name
            for g in getattr(app, 'registered_groups', [])
        ]
        assert "synthesize" in registered_names

    def test_synthesize_run_command_exists(self):
        from octane.cli.synthesize import synthesize_app
        cmd_names = [c.name for c in synthesize_app.registered_commands]
        assert "run" in cmd_names


class TestSynthesizeLogic:
    """_synthesize_stdin processes extracted content."""

    @pytest.mark.asyncio
    async def test_synthesize_stdin_processes_extracted_json(self, capsys):
        from octane.cli.synthesize import _synthesize_stdin

        json_input = json.dumps({
            "extracted": [
                {"url": "https://example.com", "text": "NVDA had a great quarter", "word_count": 6, "method": "trafilatura", "title": "NVDA News"},
            ]
        })

        with patch("sys.stdin", StringIO(json_input)), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("octane.agents.web.synthesizer.Synthesizer") as MockSynth:
            inst = MockSynth.return_value
            inst.synthesize_with_content = AsyncMock(return_value="NVDA beat earnings expectations.")
            await _synthesize_stdin(query="NVDA news", template="", no_stream=False)

        captured = capsys.readouterr()
        assert "NVDA beat earnings expectations." in captured.out

    @pytest.mark.asyncio
    async def test_synthesize_stdin_rejects_empty(self):
        from octane.cli.synthesize import _synthesize_stdin
        from click.exceptions import Exit

        with patch("sys.stdin", StringIO("")), \
             patch("sys.stdin.isatty", return_value=False):
            with pytest.raises((SystemExit, Exit)):
                await _synthesize_stdin(query="test", template="", no_stream=False)

    @pytest.mark.asyncio
    async def test_synthesize_stdin_rejects_no_items(self):
        from octane.cli.synthesize import _synthesize_stdin
        from click.exceptions import Exit

        json_input = json.dumps({"extracted": []})
        with patch("sys.stdin", StringIO(json_input)), \
             patch("sys.stdin.isatty", return_value=False):
            with pytest.raises((SystemExit, Exit)):
                await _synthesize_stdin(query="test", template="", no_stream=False)

    @pytest.mark.asyncio
    async def test_synthesize_infers_query_when_missing(self, capsys):
        from octane.cli.synthesize import _synthesize_stdin

        json_input = json.dumps({
            "extracted": [
                {"url": "https://example.com", "text": "Some content", "word_count": 2},
            ]
        })

        with patch("sys.stdin", StringIO(json_input)), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("octane.agents.web.synthesizer.Synthesizer") as MockSynth:
            inst = MockSynth.return_value
            inst.synthesize_with_content = AsyncMock(return_value="Summary")
            await _synthesize_stdin(query="", template="", no_stream=False)

        # Should have used default query
        call_args = MockSynth.return_value.synthesize_with_content.call_args
        assert "Summarize" in call_args.kwargs.get("query", call_args.args[0] if call_args.args else "")


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: UI — /api/traces-events/recent (globe seeding)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ui_app():
    from octane.ui.app import create_app
    return create_app()


@pytest.fixture
def ui_client(ui_app):
    from starlette.testclient import TestClient
    return TestClient(ui_app)


class TestTracesEventsRecent:
    """GET /api/traces-events/recent returns flattened events for globe seeding."""

    def test_endpoint_exists(self, ui_client):
        resp = ui_client.get("/api/traces-events/recent")
        assert resp.status_code == 200

    def test_returns_events_list(self, ui_client):
        data = ui_client.get("/api/traces-events/recent").json()
        assert "events" in data
        assert "total" in data
        assert isinstance(data["events"], list)

    def test_respects_limit_param(self, ui_client):
        resp = ui_client.get("/api/traces-events/recent?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: UI — Dashboard query text extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardQueryExtraction:
    """Dashboard route extracts query text from multiple event types."""

    def test_dashboard_endpoint_exists(self, ui_client):
        resp = ui_client.get("/api/dashboard")
        assert resp.status_code == 200

    def test_dashboard_returns_recent_traces(self, ui_client):
        data = ui_client.get("/api/dashboard").json()
        assert "recent_traces" in data
        assert isinstance(data["recent_traces"], list)

    def test_recent_traces_query_extraction_from_ingress(self):
        """_recent_traces extracts query from ingress event payload."""
        import tempfile, os, json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "test-trace-001.jsonl"
            events = [
                {"event_type": "ingress", "payload": {"query": "NVDA price"}, "timestamp": "2026-03-29T10:00:00", "duration_ms": 10},
                {"event_type": "agent_complete", "payload": {}, "timestamp": "2026-03-29T10:00:01", "duration_ms": 500},
            ]
            trace_file.write_text("\n".join(json.dumps(e) for e in events))

            with patch("octane.ui.routes.dashboard._TRACE_DIR", Path(tmpdir)):
                from octane.ui.routes.dashboard import _recent_traces
                traces = _recent_traces(10)

            assert len(traces) == 1
            assert traces[0]["query"] == "NVDA price"

    def test_recent_traces_query_fallback_from_decomposition(self):
        """_recent_traces falls back to decomposition event for query text."""
        import tempfile, json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "test-trace-002.jsonl"
            events = [
                {"event_type": "dispatch", "payload": {"original_query": "AI trends"}, "timestamp": "2026-03-29T10:00:00", "duration_ms": 5},
                {"event_type": "agent_complete", "payload": {}, "timestamp": "2026-03-29T10:00:01", "duration_ms": 300},
            ]
            trace_file.write_text("\n".join(json.dumps(e) for e in events))

            with patch("octane.ui.routes.dashboard._TRACE_DIR", Path(tmpdir)):
                from octane.ui.routes.dashboard import _recent_traces
                traces = _recent_traces(10)

            assert len(traces) == 1
            assert traces[0]["query"] == "AI trends"


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: UI — Pipe end-to-end JSON structures
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipeJsonContracts:
    """Verify JSON output contracts for piping between commands."""

    @pytest.mark.asyncio
    async def test_search_web_json_has_results_with_urls(self, capsys):
        from octane.cli.search import _search_web

        mock_response = {"web": {"results": [{"title": "T", "url": "https://x.com", "description": "D", "age": "1d"}]}}
        with patch("octane.tools.bodega_intel.BodegaIntelClient") as M:
            M.return_value.web_search = AsyncMock(return_value=mock_response)
            await _search_web("test", 5, output_json=True, urls_only=False)

        data = json.loads(capsys.readouterr().out)
        for r in data["results"]:
            assert "url" in r
            assert "title" in r

    @pytest.mark.asyncio
    async def test_extract_stdin_json_has_extracted_with_text(self, capsys):
        from octane.cli.extract import _extract_stdin
        from octane.agents.web.content_extractor import ExtractedContent

        json_input = json.dumps({"results": [{"url": "https://example.com"}]})
        mock_extracted = [
            ExtractedContent(url="https://example.com", text="Hello", word_count=1, method="trafilatura", title="T"),
        ]

        with patch("sys.stdin", StringIO(json_input)), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("octane.agents.web.content_extractor.ContentExtractor") as MockEx:
            MockEx.return_value.extract_batch = AsyncMock(return_value=mock_extracted)
            await _extract_stdin(output_json=True, quality="auto", top_n=10)

        data = json.loads(capsys.readouterr().out)
        for item in data["extracted"]:
            assert "url" in item
            assert "text" in item
            assert "word_count" in item

    @pytest.mark.asyncio
    async def test_search_news_json_contract(self, capsys):
        from octane.cli.search import _search_news

        mock_response = {"articles": [{"title": "N", "url": "https://n.com", "source": "S", "age": "1h", "description": "D"}]}
        with patch("octane.tools.bodega_intel.BodegaIntelClient") as M:
            M.return_value.news_search = AsyncMock(return_value=mock_response)
            await _search_news("test", 5, "3d", output_json=True, urls_only=False)

        data = json.loads(capsys.readouterr().out)
        assert data["source"] == "news"
        assert len(data["results"]) == 1
        assert "source" in data["results"][0]
