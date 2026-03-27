"""Tests for YouTube Playwright-first transcript extraction and debounce."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── _parse_timestamp ─────────────────────────────────────────────────────────

class TestParseTimestamp:
    """Tests for _parse_timestamp helper."""

    def test_mm_ss(self):
        from octane.extractors.youtube.transcript import _parse_timestamp
        assert _parse_timestamp("1:23") == 83.0

    def test_hh_mm_ss(self):
        from octane.extractors.youtube.transcript import _parse_timestamp
        assert _parse_timestamp("1:02:34") == 3754.0

    def test_zero(self):
        from octane.extractors.youtube.transcript import _parse_timestamp
        assert _parse_timestamp("0:00") == 0.0

    def test_invalid(self):
        from octane.extractors.youtube.transcript import _parse_timestamp
        assert _parse_timestamp("abc") == 0.0

    def test_empty(self):
        from octane.extractors.youtube.transcript import _parse_timestamp
        assert _parse_timestamp("") == 0.0


# ── get_transcript routing (Playwright-first) ────────────────────────────────

class TestGetTranscriptRouting:
    """Playwright (Tier 0) is always tried first; Tier 1+2 are fallbacks."""

    @patch("octane.extractors.youtube.transcript._tier0_playwright")
    @patch("octane.extractors.youtube.transcript._tier1_transcript_api")
    def test_playwright_tried_first(self, mock_t1, mock_t0):
        from octane.extractors.youtube.transcript import get_transcript

        mock_t0.return_value = [{"text": "from playwright", "start": 0.0, "duration": 1.0}]

        result = get_transcript("test123")
        mock_t0.assert_called_once_with("test123", "en")
        mock_t1.assert_not_called()
        assert result[0]["text"] == "from playwright"

    @patch("octane.extractors.youtube.transcript._tier0_playwright")
    @patch("octane.extractors.youtube.transcript._tier1_transcript_api")
    def test_falls_through_to_tier1(self, mock_t1, mock_t0):
        from octane.extractors.youtube.transcript import get_transcript

        mock_t0.return_value = None  # playwright failed
        mock_t1.return_value = [{"text": "from api", "start": 0.0, "duration": 1.0}]

        result = get_transcript("test123")
        mock_t0.assert_called_once()
        mock_t1.assert_called_once()
        assert result[0]["text"] == "from api"

    @patch("octane.extractors.youtube.transcript._tier0_playwright")
    @patch("octane.extractors.youtube.transcript._tier1_transcript_api")
    @patch("octane.extractors.youtube.transcript._tier2_ytdlp_no_cookies")
    def test_falls_through_to_tier2(self, mock_t2, mock_t1, mock_t0):
        from octane.extractors.youtube.transcript import get_transcript

        mock_t0.return_value = None
        mock_t1.return_value = None
        mock_t2.return_value = [{"text": "from ytdlp", "start": 0.0, "duration": 1.0}]

        result = get_transcript("test123")
        mock_t0.assert_called_once()
        mock_t1.assert_called_once()
        mock_t2.assert_called_once()
        assert result[0]["text"] == "from ytdlp"

    @patch("octane.extractors.youtube.transcript._tier0_playwright")
    @patch("octane.extractors.youtube.transcript._tier1_transcript_api")
    @patch("octane.extractors.youtube.transcript._tier2_ytdlp_no_cookies")
    def test_all_tiers_fail_returns_none(self, mock_t2, mock_t1, mock_t0):
        from octane.extractors.youtube.transcript import get_transcript

        mock_t0.return_value = None
        mock_t1.return_value = None
        mock_t2.return_value = None

        result = get_transcript("test123")
        assert result is None


# ── Tier 0: _playwright_extract_timedtext ─────────────────────────────────────

class TestPlaywrightTimedtext:
    """Tests for _playwright_extract_timedtext helper."""

    def test_parses_json3_events(self):
        from octane.extractors.youtube.transcript import _playwright_extract_timedtext

        # The implementation calls page.evaluate() twice:
        # 1st call: extract signed caption URL from ytInitialPlayerResponse
        # 2nd call: fetch(url) and return JSON
        caption_url = "https://www.youtube.com/api/timedtext?v=test123&lang=en&sig=abc"
        timedtext_json = {
            "events": [
                {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Hello "}]},
                {"tStartMs": 3000, "dDurationMs": 1500, "segs": [{"utf8": "World"}]},
            ]
        }

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=[caption_url, timedtext_json])

        segments = asyncio.run(_playwright_extract_timedtext(mock_page, "test123", "en"))
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello"
        assert segments[0]["start"] == 1.0
        assert segments[1]["text"] == "World"

    def test_returns_none_when_no_captions(self):
        from octane.extractors.youtube.transcript import _playwright_extract_timedtext

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)  # no captionTracks

        segments = asyncio.run(_playwright_extract_timedtext(mock_page, "test123", "en"))
        assert segments is None

    def test_returns_none_on_fetch_error(self):
        from octane.extractors.youtube.transcript import _playwright_extract_timedtext

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=[
            "https://example.com/timedtext?v=test123",  # caption URL found
            {"error": 429},  # fetch failed
        ])

        segments = asyncio.run(_playwright_extract_timedtext(mock_page, "test123", "en"))
        assert segments is None


# ── YouTube Login ─────────────────────────────────────────────────────────────

class TestYoutubeLogin:
    """Tests for the youtube_login function."""

    def test_cookie_path_is_valid(self):
        """Cookie file path should be properly configured."""
        from octane.extractors.youtube.transcript import _YOUTUBE_COOKIE_FILE
        assert isinstance(_YOUTUBE_COOKIE_FILE, Path)
        assert _YOUTUBE_COOKIE_FILE.name == "youtube.com.json"


# ── YouTube Debounce ──────────────────────────────────────────────────────────

class TestYoutubeDebounce:
    """Tests for the YouTube debounce in pipeline."""

    def test_debounce_spaces_calls(self):
        import octane.extractors.pipeline as pipeline

        # Reset state
        pipeline._youtube_lock = None
        pipeline._youtube_last_call = 0.0

        t0 = time.monotonic()

        async def _run():
            await pipeline._youtube_debounce()
            await pipeline._youtube_debounce()

        asyncio.run(_run())
        elapsed = time.monotonic() - t0
        # Second call should wait ~2s
        assert elapsed >= 1.5, f"Expected >=1.5s gap, got {elapsed:.2f}s"

    def test_debounce_no_wait_after_gap(self):
        import octane.extractors.pipeline as pipeline

        # Reset state — pretend last call was long ago
        pipeline._youtube_lock = None
        pipeline._youtube_last_call = 0.0

        t0 = time.monotonic()

        async def _run():
            await pipeline._youtube_debounce()

        asyncio.run(_run())
        elapsed = time.monotonic() - t0
        # First call should not wait
        assert elapsed < 0.5, f"Expected <0.5s, got {elapsed:.2f}s"


# ── ArXiv Debounce (regression) ──────────────────────────────────────────────

class TestArxivDebounceRegression:
    """Ensure arXiv debounce still works after refactor."""

    def test_debounce_spaces_calls(self):
        import octane.extractors.pipeline as pipeline

        pipeline._arxiv_lock = None
        pipeline._arxiv_last_call = 0.0

        t0 = time.monotonic()

        async def _run():
            await pipeline._arxiv_debounce()
            await pipeline._arxiv_debounce()

        asyncio.run(_run())
        elapsed = time.monotonic() - t0
        assert elapsed >= 3.0, f"Expected >=3.0s gap, got {elapsed:.2f}s"


# ── CLI youtube-login command ─────────────────────────────────────────────────

class TestYoutubeLoginCLI:
    """Test that the CLI command is registered."""

    def test_command_exists(self):
        from octane.cli.extract import extract_app
        command_names = [c.name for c in extract_app.registered_commands]
        assert "youtube-login" in command_names


# ── Cookie path constants ────────────────────────────────────────────────────

class TestCookieConstants:
    """Test cookie path configuration."""

    def test_cookie_dir_under_home(self):
        from octane.extractors.youtube.transcript import _COOKIE_DIR
        assert ".octane" in str(_COOKIE_DIR)
        assert "cookies" in str(_COOKIE_DIR)

    def test_youtube_cookie_file_path(self):
        from octane.extractors.youtube.transcript import _YOUTUBE_COOKIE_FILE
        assert _YOUTUBE_COOKIE_FILE.name == "youtube.com.json"

    def test_shared_with_browser_agent(self):
        """YouTube cookies use the same directory as BrowserAgent."""
        from octane.extractors.youtube.transcript import _COOKIE_DIR as yt_dir
        from octane.agents.web.browser import _COOKIE_DIR as browser_dir
        assert yt_dir == browser_dir


# ── Chrome cookie extraction ─────────────────────────────────────────────────

class TestBrowserCookieExtraction:
    """Tests for _extract_browser_cookies_for_playwright and helpers."""

    def test_detect_cookie_browser_returns_string_or_none(self):
        from octane.extractors.youtube.transcript import _detect_cookie_browser
        result = _detect_cookie_browser()
        assert result is None or isinstance(result, str)

    def test_get_cookie_browser_caches(self):
        import octane.extractors.youtube.transcript as t
        t._COOKIE_BROWSER = t._COOKIE_BROWSER_SENTINEL  # reset
        first = t._get_cookie_browser()
        second = t._get_cookie_browser()
        assert first == second  # same result, cached

    @patch("octane.extractors.youtube.transcript._get_cookie_browser", return_value=None)
    def test_extract_returns_empty_when_no_browser(self, _mock):
        import octane.extractors.youtube.transcript as t
        t._CACHED_PW_COOKIES = None
        result = t._extract_browser_cookies_for_playwright()
        assert result == []

    def test_chrome_epoch_conversion(self):
        """Chrome epoch timestamps (µs since 1601) are converted to Unix seconds."""
        import octane.extractors.youtube.transcript as t
        from unittest.mock import MagicMock
        import http.cookiejar

        t._CACHED_PW_COOKIES = None

        # Create a mock cookie with Chrome epoch timestamp
        cookie = http.cookiejar.Cookie(
            version=0, name="TEST", value="val",
            port=None, port_specified=False,
            domain=".youtube.com", domain_specified=True, domain_initial_dot=True,
            path="/", path_specified=True,
            secure=True, expires=13453533106401864,  # Chrome epoch
            discard=False, comment=None, comment_url=None, rest={},
        )

        mock_jar = [cookie]

        mock_ydl = MagicMock()
        mock_ydl.cookiejar = mock_jar
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)

        with patch("octane.extractors.youtube.transcript._get_cookie_browser", return_value="chrome"):
            with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
                result = t._extract_browser_cookies_for_playwright()

        assert len(result) == 1
        c = result[0]
        # Should be converted to Unix timestamp (around 2027)
        assert 1_700_000_000 < c["expires"] < 2_000_000_000
        assert c["secure"] is True
        assert c["domain"] == ".youtube.com"
