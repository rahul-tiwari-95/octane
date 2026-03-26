"""Comprehensive tests for Session 30 — Extraction Pipeline.

Covers: models, detect_source_type, classifier, VTT parser, chunker, pipeline.
Unit tests run without network; e2e tests marked @pytest.mark.e2e.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from octane.extractors.models import (
    ChunkMetadata,
    ExtractedDocument,
    RELIABILITY_MODIFIERS,
    SOURCE_RELIABILITY,
    SourceType,
    TaskState,
    TextChunk,
)
from octane.extractors.pipeline import detect_source_type
from octane.extractors.youtube.classifier import classify_video, _parse_duration
from octane.extractors.youtube.transcript import _parse_vtt, _vtt_time_to_seconds, transcript_to_text
from octane.extractors.chunker import chunk_text, chunk_transcript_by_time, chunk_epub_chapters


# ════════════════════════════════════════════════════════════════════════════════
#  1. Models
# ════════════════════════════════════════════════════════════════════════════════

class TestSourceType:
    def test_all_source_types_exist(self):
        assert SourceType.YOUTUBE.value == "youtube"
        assert SourceType.ARXIV.value == "arxiv"
        assert SourceType.PDF.value == "pdf"
        assert SourceType.EPUB.value == "epub"
        assert SourceType.WEB.value == "web"

    def test_source_type_from_string(self):
        assert SourceType("youtube") == SourceType.YOUTUBE
        assert SourceType("arxiv") == SourceType.ARXIV
        assert SourceType("pdf") == SourceType.PDF

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValueError):
            SourceType("unknown")


class TestTaskState:
    def test_all_states_exist(self):
        expected = {"pending", "searching", "extracting", "classifying",
                    "chunking", "complete", "failed", "interrupted", "spawned"}
        actual = {s.value for s in TaskState}
        assert actual == expected


class TestSourceReliability:
    def test_all_source_types_have_reliability(self):
        for st in SourceType:
            assert st in SOURCE_RELIABILITY, f"Missing reliability score for {st}"

    def test_arxiv_highest(self):
        assert SOURCE_RELIABILITY[SourceType.ARXIV] > SOURCE_RELIABILITY[SourceType.PDF]
        assert SOURCE_RELIABILITY[SourceType.ARXIV] > SOURCE_RELIABILITY[SourceType.YOUTUBE]
        assert SOURCE_RELIABILITY[SourceType.ARXIV] > SOURCE_RELIABILITY[SourceType.EPUB]

    def test_web_lowest(self):
        for st in SourceType:
            assert SOURCE_RELIABILITY[SourceType.WEB] <= SOURCE_RELIABILITY[st]

    def test_reliability_scores_in_range(self):
        for st, score in SOURCE_RELIABILITY.items():
            assert 0.0 <= score <= 1.0, f"{st}: {score} out of range"


class TestReliabilityModifiers:
    def test_peer_reviewed_positive(self):
        assert RELIABILITY_MODIFIERS["peer_reviewed"] > 0

    def test_music_content_negative(self):
        assert RELIABILITY_MODIFIERS["music_content"] < 0

    def test_all_modifiers_bounded(self):
        for key, val in RELIABILITY_MODIFIERS.items():
            assert -1.0 <= val <= 1.0, f"Modifier {key}={val} out of range"


class TestChunkMetadata:
    def test_defaults(self):
        m = ChunkMetadata()
        assert m.page is None
        assert m.chapter is None
        assert m.timestamp_start is None
        assert not m.is_table
        assert not m.is_junk

    def test_with_values(self):
        m = ChunkMetadata(page=3, chapter="Introduction", is_table=True)
        assert m.page == 3
        assert m.chapter == "Introduction"
        assert m.is_table is True


class TestTextChunk:
    def test_auto_counts(self):
        c = TextChunk(text="hello world foo bar")
        assert c.char_count == len("hello world foo bar")
        assert c.word_count == 4

    def test_explicit_counts_not_overwritten(self):
        c = TextChunk(text="hello", char_count=99, word_count=42)
        assert c.char_count == 99
        assert c.word_count == 42

    def test_empty_text(self):
        c = TextChunk(text="")
        assert c.word_count == 0
        assert c.char_count == 0

    def test_index_defaults_zero(self):
        c = TextChunk(text="test")
        assert c.index == 0


class TestExtractedDocument:
    def test_content_hash_auto_generated(self):
        doc = ExtractedDocument(
            source_type=SourceType.PDF,
            source_url="test.pdf",
            raw_text="some text content",
        )
        expected = hashlib.sha256("some text content".encode()).hexdigest()[:16]
        assert doc.content_hash == expected

    def test_empty_text_no_hash(self):
        doc = ExtractedDocument(
            source_type=SourceType.PDF,
            source_url="test.pdf",
        )
        assert doc.content_hash == ""

    def test_reliability_from_source_type(self):
        doc = ExtractedDocument(
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234.5678",
            raw_text="paper text",
        )
        assert doc.reliability_score == SOURCE_RELIABILITY[SourceType.ARXIV]

    def test_total_words(self):
        doc = ExtractedDocument(
            source_type=SourceType.WEB,
            source_url="https://example.com",
            raw_text="one two three four five",
        )
        assert doc.total_words == 5

    def test_total_chunks(self):
        doc = ExtractedDocument(
            source_type=SourceType.PDF,
            source_url="test.pdf",
            chunks=[TextChunk(text="a"), TextChunk(text="b"), TextChunk(text="c")],
        )
        assert doc.total_chunks == 3

    def test_to_dict_keys(self):
        doc = ExtractedDocument(
            source_type=SourceType.YOUTUBE,
            source_url="https://youtube.com",
            raw_text="hello",
            title="Test",
        )
        d = doc.to_dict()
        assert d["source_type"] == "youtube"
        assert d["title"] == "Test"
        assert d["total_words"] == 1
        assert "content_hash" in d
        assert "extracted_at" in d


# ════════════════════════════════════════════════════════════════════════════════
#  2. Source Detection
# ════════════════════════════════════════════════════════════════════════════════

class TestDetectSourceType:
    """Test detect_source_type with various URL and path patterns."""

    # YouTube
    @pytest.mark.parametrize("url,expected_id", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=XfpMkf4rD6E", "XfpMkf4rD6E"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://www.youtube.com/watch?v=dQw4w9WgXcQ&t=123", "dQw4w9WgXcQ"),
    ])
    def test_youtube_urls(self, url, expected_id):
        st, ident = detect_source_type(url)
        assert st == SourceType.YOUTUBE
        assert ident == expected_id

    # arXiv
    @pytest.mark.parametrize("source,expected_id", [
        ("1706.03762", "1706.03762"),
        ("2408.09869", "2408.09869"),
        ("2408.09869v2", "2408.09869v2"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/2408.09869", "2408.09869"),
    ])
    def test_arxiv_sources(self, source, expected_id):
        st, ident = detect_source_type(source)
        assert st == SourceType.ARXIV
        assert ident == expected_id

    # PDF
    def test_pdf_file(self):
        st, ident = detect_source_type("/path/to/paper.pdf")
        assert st == SourceType.PDF

    def test_pdf_file_uppercase(self):
        st, ident = detect_source_type("/path/to/paper.PDF")
        assert st == SourceType.PDF

    # EPUB
    def test_epub_file(self):
        st, ident = detect_source_type("/path/to/book.epub")
        assert st == SourceType.EPUB

    # Web fallback
    def test_web_url_fallback(self):
        st, _ = detect_source_type("https://example.com/article")
        assert st == SourceType.WEB

    def test_random_string_fallback(self):
        st, _ = detect_source_type("some random text")
        assert st == SourceType.WEB

    # Edge cases
    def test_whitespace_stripped(self):
        st, ident = detect_source_type("  1706.03762  ")
        assert st == SourceType.ARXIV
        assert ident == "1706.03762"


# ════════════════════════════════════════════════════════════════════════════════
#  3. YouTube Classifier
# ════════════════════════════════════════════════════════════════════════════════

class TestClassifier:
    def test_music_video_detected(self):
        meta = {"title": "Taylor Swift - Shake It Off (Official Music Video)", "channel": "TaylorSwiftVEVO"}
        label, conf = classify_video(meta)
        assert label == "music"
        assert conf > 0

    def test_lecture_detected_as_spoken(self):
        meta = {"title": "MIT 6.006 Introduction to Algorithms Lecture 1", "channel": "MIT OpenCourseWare"}
        label, _ = classify_video(meta)
        assert label == "spoken"

    def test_tutorial_detected_as_spoken(self):
        meta = {"title": "Python Tutorial for Beginners - How to Install Python", "channel": "TechWithTim"}
        label, _ = classify_video(meta)
        assert label == "spoken"

    def test_podcast_detected_as_spoken(self):
        meta = {"title": "Lex Fridman Podcast #100: Interview with Elon Musk", "channel": "Lex Fridman"}
        label, _ = classify_video(meta)
        assert label == "spoken"

    def test_remix_detected_as_music(self):
        meta = {"title": "Deep House Remix 2024", "channel": "DeepHouseRecords"}
        label, _ = classify_video(meta)
        assert label == "music"

    def test_empty_metadata_defaults_spoken(self):
        label, conf = classify_video({})
        assert label == "spoken"
        assert conf == 0.0

    def test_music_category_hint(self):
        meta = {"title": "Some Ambiguous Title", "channel": "Unknown", "category": "Music"}
        label, _ = classify_video(meta)
        assert label == "music"

    def test_education_category_hint(self):
        meta = {"title": "Some Ambiguous Title", "channel": "Unknown", "category": "Education"}
        label, _ = classify_video(meta)
        assert label == "spoken"

    def test_long_duration_favors_spoken(self):
        meta = {"title": "Something", "channel": "Unknown", "duration": "5400"}  # 90 min
        label, _ = classify_video(meta)
        assert label == "spoken"


class TestParseDuration:
    def test_seconds_string(self):
        assert _parse_duration("120") == 120.0

    def test_mm_ss(self):
        assert _parse_duration("3:42") == 222.0

    def test_hh_mm_ss(self):
        assert _parse_duration("1:30:00") == 5400.0

    def test_empty(self):
        assert _parse_duration("") is None

    def test_none(self):
        assert _parse_duration(None) is None


# ════════════════════════════════════════════════════════════════════════════════
#  4. VTT Parser
# ════════════════════════════════════════════════════════════════════════════════

class TestVTTParsing:
    SAMPLE_VTT = """\
WEBVTT

00:00:00.000 --> 00:00:05.000
Hello and welcome to the tutorial.

00:00:05.001 --> 00:00:10.000
Today we'll learn about transformers.

00:00:10.001 --> 00:00:15.000
They are a type of neural network.
"""

    def test_parse_vtt_basic(self):
        segments = _parse_vtt(self.SAMPLE_VTT)
        assert len(segments) == 3
        assert segments[0]["text"] == "Hello and welcome to the tutorial."
        assert segments[0]["start"] == 0.0
        assert segments[1]["text"] == "Today we'll learn about transformers."

    def test_parse_vtt_deduplication(self):
        vtt = """\
WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world

00:00:02.000 --> 00:00:07.000
Hello world

00:00:05.000 --> 00:00:10.000
Something new
"""
        segments = _parse_vtt(vtt)
        texts = [s["text"] for s in segments]
        assert texts.count("Hello world") == 1  # Deduplicated
        assert "Something new" in texts

    def test_parse_vtt_strips_html_tags(self):
        vtt = """\
WEBVTT

00:00:00.000 --> 00:00:05.000
<b>Bold text</b> and <i>italic</i>
"""
        segments = _parse_vtt(vtt)
        assert segments[0]["text"] == "Bold text and italic"

    def test_parse_vtt_strips_position_tags(self):
        vtt = """\
WEBVTT

00:00:00.000 --> 00:00:05.000
Hello <00:00:02.500>world <00:00:03.000>how are you
"""
        segments = _parse_vtt(vtt)
        assert "00:00:02.500" not in segments[0]["text"]
        assert "Hello" in segments[0]["text"]
        assert "world" in segments[0]["text"]

    def test_vtt_time_to_seconds(self):
        assert _vtt_time_to_seconds("00:00:00.000") == 0.0
        assert _vtt_time_to_seconds("00:01:30.500") == 90.5
        assert _vtt_time_to_seconds("01:30:00.000") == 5400.0


class TestTranscriptToText:
    def test_basic_join(self):
        segments = [
            {"text": "Hello", "start": 0.0},
            {"text": "world", "start": 1.0},
        ]
        assert transcript_to_text(segments) == "Hello world"

    def test_strips_whitespace(self):
        segments = [
            {"text": "  Hello  ", "start": 0.0},
            {"text": "  world  ", "start": 1.0},
        ]
        assert transcript_to_text(segments) == "Hello world"

    def test_skips_empty(self):
        segments = [
            {"text": "Hello", "start": 0.0},
            {"text": "", "start": 1.0},
            {"text": "world", "start": 2.0},
        ]
        assert transcript_to_text(segments) == "Hello world"


# ════════════════════════════════════════════════════════════════════════════════
#  5. Chunker
# ════════════════════════════════════════════════════════════════════════════════

class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", SourceType.PDF)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].index == 0

    def test_long_text_splits(self):
        text = "Word. " * 1000  # ~6000 chars
        chunks = chunk_text(text, SourceType.ARXIV)
        assert len(chunks) > 1
        # All chunks should have sequential indices
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_reliability_score_applied(self):
        chunks = chunk_text("Hello world.", SourceType.PDF, reliability_score=0.85)
        assert chunks[0].reliability_score == 0.85

    def test_arxiv_separators_include_headers(self):
        text = "# Section One\n\nParagraph one.\n\n## Section Two\n\nParagraph two."
        # Should not error
        chunks = chunk_text(text, SourceType.ARXIV)
        assert len(chunks) >= 1


class TestChunkTranscriptByTime:
    def test_basic_chunking(self):
        segments = [
            {"text": f"Segment {i}", "start": float(i * 10), "duration": 10.0}
            for i in range(10)
        ]
        chunks = chunk_transcript_by_time(segments, target_duration=45.0)
        assert len(chunks) >= 2
        assert chunks[0].metadata.timestamp_start == 0.0
        assert chunks[0].metadata.timestamp_end is not None

    def test_empty_segments(self):
        chunks = chunk_transcript_by_time([])
        assert chunks == []

    def test_single_segment(self):
        segments = [{"text": "Short video", "start": 0.0, "duration": 5.0}]
        chunks = chunk_transcript_by_time(segments)
        assert len(chunks) == 1
        assert chunks[0].text == "Short video"

    def test_indices_sequential(self):
        segments = [
            {"text": f"Word {i}", "start": float(i * 15), "duration": 15.0}
            for i in range(20)
        ]
        chunks = chunk_transcript_by_time(segments)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_reliability_applied(self):
        segments = [{"text": "test", "start": 0.0, "duration": 5.0}]
        chunks = chunk_transcript_by_time(segments, reliability_score=0.42)
        assert chunks[0].reliability_score == 0.42


class TestChunkEpubChapters:
    def test_small_chapters_not_split(self):
        chapters = [
            {"title": "Chapter 1", "text": "Short chapter content."},
            {"title": "Chapter 2", "text": "Another short chapter."},
        ]
        chunks = chunk_epub_chapters(chapters)
        assert len(chunks) == 2
        assert chunks[0].metadata.chapter == "Chapter 1"
        assert chunks[1].metadata.chapter == "Chapter 2"

    def test_large_chapter_split(self):
        chapters = [
            {"title": "Big Chapter", "text": "Word. " * 1000},
        ]
        chunks = chunk_epub_chapters(chapters)
        assert len(chunks) > 1
        # All chunks from same chapter
        for c in chunks:
            assert c.metadata.chapter == "Big Chapter"

    def test_empty_chapters_skipped(self):
        chapters = [
            {"title": "Empty", "text": ""},
            {"title": "Not Empty", "text": "Content here."},
        ]
        chunks = chunk_epub_chapters(chapters)
        assert len(chunks) == 1
        assert chunks[0].metadata.chapter == "Not Empty"

    def test_reliability_applied(self):
        chapters = [{"title": "Ch1", "text": "Content."}]
        chunks = chunk_epub_chapters(chapters, reliability_score=0.88)
        assert chunks[0].reliability_score == 0.88

    def test_indices_sequential_across_chapters(self):
        chapters = [
            {"title": "Ch1", "text": "Content one."},
            {"title": "Ch2", "text": "Content two."},
        ]
        chunks = chunk_epub_chapters(chapters)
        for i, c in enumerate(chunks):
            assert c.index == i


# ════════════════════════════════════════════════════════════════════════════════
#  6. Pipeline — detect_source_type + extract (mocked)
# ════════════════════════════════════════════════════════════════════════════════

class TestPipelineExtractMocked:
    """Test the pipeline extract() with mocked sub-extractors."""

    @pytest.mark.asyncio
    async def test_extract_youtube_mocked(self):
        """Mock YouTube extraction — no network."""
        mock_segments = [
            {"text": "Hello world", "start": 0.0, "duration": 5.0},
            {"text": "This is a test", "start": 5.0, "duration": 5.0},
        ]
        mock_meta = [{"video_id": "test123", "title": "Test Video", "channel": "TestChan", "duration": "60"}]

        with patch("octane.extractors.youtube.search.search_youtube", return_value=mock_meta), \
             patch("octane.extractors.youtube.transcript.get_transcript", return_value=mock_segments):
            from octane.extractors.pipeline import extract
            doc = await extract("https://www.youtube.com/watch?v=test1234567")

        assert doc.source_type == SourceType.YOUTUBE
        assert doc.title == "Test Video"
        assert doc.author == "TestChan"
        assert "Hello world" in doc.raw_text
        assert doc.total_chunks >= 1
        assert doc.extraction_method == "youtube-transcript-api"

    @pytest.mark.asyncio
    async def test_extract_youtube_no_transcript(self):
        """When no transcript available, returns empty doc."""
        mock_meta = [{"video_id": "test123", "title": "Silent", "channel": "Chan"}]

        with patch("octane.extractors.youtube.search.search_youtube", return_value=mock_meta), \
             patch("octane.extractors.youtube.transcript.get_transcript", return_value=None):
            from octane.extractors.pipeline import extract
            doc = await extract("https://www.youtube.com/watch?v=test1234567")

        assert doc.source_type == SourceType.YOUTUBE
        assert doc.raw_text == ""
        assert doc.reliability_score == 0.0

    @pytest.mark.asyncio
    async def test_extract_pdf_mocked(self, tmp_path):
        """Mock PDF extraction."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy")

        with patch("octane.extractors.pdf.extractor.extract_pdf", return_value=("Sample PDF text content.", "pymupdf4llm")):
            from octane.extractors.pipeline import extract
            doc = await extract(str(pdf_file))

        assert doc.source_type == SourceType.PDF
        assert doc.extraction_method == "pymupdf4llm"
        assert "Sample PDF text content" in doc.raw_text

    @pytest.mark.asyncio
    async def test_extract_unsupported_type(self):
        """Web source with no handler raises ValueError."""
        from octane.extractors.pipeline import extract
        with pytest.raises(ValueError, match="Unsupported source type"):
            await extract("https://example.com/random-page")


# ════════════════════════════════════════════════════════════════════════════════
#  7. End-to-End (network required — marked @e2e)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestE2EExtraction:
    """Live extraction tests. Only run with OCTANE_TEST_E2E=1."""

    @pytest.mark.asyncio
    async def test_youtube_live(self):
        from octane.extractors.pipeline import extract
        # Short famous video
        doc = await extract("https://www.youtube.com/watch?v=XfpMkf4rD6E")
        assert doc.source_type == SourceType.YOUTUBE
        assert doc.total_words > 100
        assert doc.total_chunks > 0
        assert doc.title != ""

    @pytest.mark.asyncio
    async def test_arxiv_live(self):
        from octane.extractors.pipeline import extract
        doc = await extract("1706.03762")  # Attention Is All You Need
        assert doc.source_type == SourceType.ARXIV
        assert doc.total_words > 1000
        assert "attention" in doc.raw_text.lower()
        assert doc.reliability_score >= 0.9

    def test_youtube_search_live(self):
        from octane.extractors.youtube.search import search_youtube
        results = search_youtube("attention is all you need", limit=3)
        assert len(results) >= 1
        assert "title" in results[0]
        assert "url" in results[0]

    def test_arxiv_search_live(self):
        from octane.extractors.academic.arxiv_search import search_arxiv
        results = search_arxiv("transformer neural network", max_results=3)
        assert len(results) >= 1
        assert "arxiv_id" in results[0]
        assert "title" in results[0]
