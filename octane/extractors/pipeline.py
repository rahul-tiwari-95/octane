"""Unified extraction pipeline — single entry point for all source types.

Usage:
    doc = await extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    doc = await extract("2408.09869")     # arXiv ID
    doc = await extract("/path/to/paper.pdf")
    doc = await extract("/path/to/book.epub")
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import structlog

from octane.extractors.models import (
    ExtractedDocument,
    SourceType,
    SOURCE_RELIABILITY,
    RELIABILITY_MODIFIERS,
)

logger = structlog.get_logger().bind(component="extractors.pipeline")

# Patterns for source detection
_YOUTUBE_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})"),
]
_ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def detect_source_type(source: str) -> tuple[SourceType, str]:
    """Detect source type and extract the canonical identifier.

    Returns:
        (source_type, identifier) — identifier is video_id, arxiv_id, or file path.
    """
    source = source.strip()

    # YouTube URL
    for pattern in _YOUTUBE_PATTERNS:
        m = pattern.search(source)
        if m:
            return SourceType.YOUTUBE, m.group(1)

    # arXiv URL
    m = _ARXIV_URL_PATTERN.search(source)
    if m:
        return SourceType.ARXIV, m.group(1)

    # arXiv ID (bare)
    m = _ARXIV_ID_PATTERN.match(source)
    if m:
        return SourceType.ARXIV, m.group(0)

    # File paths
    p = Path(source)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return SourceType.PDF, str(p)
    elif suffix == ".epub":
        return SourceType.EPUB, str(p)

    # Default to web
    return SourceType.WEB, source


async def extract(
    source: str,
    quality: str = "auto",
    source_type: SourceType | None = None,
) -> ExtractedDocument:
    """Extract content from any supported source.

    Args:
        source: URL, arXiv ID, or file path.
        quality: "fast", "deep", or "auto".
        source_type: Override auto-detection if needed.

    Returns:
        ExtractedDocument with raw_text, chunks, and metadata.
    """
    if source_type is None:
        detected_type, identifier = detect_source_type(source)
    else:
        detected_type = source_type
        _, identifier = detect_source_type(source)

    logger.info("extract_start", source=source[:80], type=detected_type.value, quality=quality)

    if detected_type == SourceType.YOUTUBE:
        return await _extract_youtube(identifier)
    elif detected_type == SourceType.ARXIV:
        return await _extract_arxiv(identifier, quality)
    elif detected_type == SourceType.PDF:
        return await _extract_pdf(identifier, quality)
    elif detected_type == SourceType.EPUB:
        return await _extract_epub(identifier)
    else:
        raise ValueError(f"Unsupported source type: {detected_type.value} for {source}")


async def _extract_youtube(video_id: str) -> ExtractedDocument:
    """Extract YouTube video transcript + metadata."""
    from octane.extractors.youtube.search import search_youtube
    from octane.extractors.youtube.transcript import get_transcript, transcript_to_text
    from octane.extractors.youtube.classifier import classify_video
    from octane.extractors.chunker import chunk_transcript_by_time

    loop = asyncio.get_event_loop()

    # Get metadata via search (for title/channel)
    results = await loop.run_in_executor(
        None, lambda: search_youtube(video_id, limit=1)
    )
    meta = results[0] if results else {"video_id": video_id, "title": "", "channel": ""}

    # Classify
    label, confidence = classify_video(meta)
    if label == "music" and confidence > 0.6:
        logger.info("youtube_music_detected", video_id=video_id, confidence=confidence)

    # Get transcript
    segments = await loop.run_in_executor(
        None, lambda: get_transcript(video_id)
    )

    if not segments:
        return ExtractedDocument(
            source_type=SourceType.YOUTUBE,
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            title=meta.get("title", ""),
            author=meta.get("channel", ""),
            extraction_method="no_transcript",
            reliability_score=0.0,
            metadata={"video_id": video_id, "classification": label},
        )

    raw_text = transcript_to_text(segments)

    # Compute reliability
    base = SOURCE_RELIABILITY[SourceType.YOUTUBE]
    if label == "music":
        base += RELIABILITY_MODIFIERS["music_content"]
    duration = meta.get("duration", "")
    try:
        dur_secs = float(duration) if duration.replace(".", "").isdigit() else 0
    except (ValueError, AttributeError):
        dur_secs = 0
    if dur_secs > 1200:
        base += RELIABILITY_MODIFIERS["long_form_lecture"]

    # Chunk by timestamp
    chunks = chunk_transcript_by_time(segments, reliability_score=max(0.0, min(1.0, base)))

    return ExtractedDocument(
        source_type=SourceType.YOUTUBE,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        title=meta.get("title", ""),
        author=meta.get("channel", ""),
        raw_text=raw_text,
        chunks=chunks,
        extraction_method="youtube-transcript-api",
        reliability_score=max(0.0, min(1.0, base)),
        metadata={
            "video_id": video_id,
            "classification": label,
            "classification_confidence": round(confidence, 2),
            "n_segments": len(segments),
            "duration": meta.get("duration", ""),
        },
    )


async def _extract_arxiv(arxiv_id: str, quality: str) -> ExtractedDocument:
    """Extract arXiv paper: search metadata + download PDF + extract text."""
    from octane.extractors.academic.arxiv_search import search_arxiv, download_arxiv_pdf
    from octane.extractors.pdf.extractor import extract_pdf
    from octane.extractors.chunker import chunk_text

    loop = asyncio.get_event_loop()

    # Get metadata
    papers = await loop.run_in_executor(
        None, lambda: search_arxiv(arxiv_id, max_results=1)
    )
    meta = papers[0] if papers else {}

    # Download PDF
    pdf_path = await loop.run_in_executor(
        None, lambda: download_arxiv_pdf(arxiv_id)
    )

    # Extract text
    raw_text, method = await loop.run_in_executor(
        None, lambda: extract_pdf(pdf_path, quality=quality)
    )

    # Compute reliability
    base = SOURCE_RELIABILITY[SourceType.ARXIV]
    if meta.get("journal_ref"):
        base += RELIABILITY_MODIFIERS["peer_reviewed"]

    # Chunk
    chunks = chunk_text(raw_text, SourceType.ARXIV, reliability_score=min(1.0, base))

    return ExtractedDocument(
        source_type=SourceType.ARXIV,
        source_url=meta.get("pdf_url", f"https://arxiv.org/abs/{arxiv_id}"),
        title=meta.get("title", arxiv_id),
        author=", ".join(meta.get("authors", [])),
        raw_text=raw_text,
        chunks=chunks,
        extraction_method=method,
        reliability_score=min(1.0, base),
        metadata={
            "arxiv_id": arxiv_id,
            "categories": meta.get("categories", []),
            "published": meta.get("published", ""),
            "summary": meta.get("summary", "")[:500],
            "journal_ref": meta.get("journal_ref", ""),
            "pdf_path": str(pdf_path),
        },
    )


async def _extract_pdf(path: str, quality: str) -> ExtractedDocument:
    """Extract text from a local PDF."""
    from octane.extractors.pdf.extractor import extract_pdf
    from octane.extractors.chunker import chunk_text

    loop = asyncio.get_event_loop()

    raw_text, method = await loop.run_in_executor(
        None, lambda: extract_pdf(path, quality=quality)
    )

    base = SOURCE_RELIABILITY[SourceType.PDF]
    chunks = chunk_text(raw_text, SourceType.PDF, reliability_score=base)

    return ExtractedDocument(
        source_type=SourceType.PDF,
        source_url=str(path),
        title=Path(path).stem,
        raw_text=raw_text,
        chunks=chunks,
        extraction_method=method,
        reliability_score=base,
        metadata={"path": str(path)},
    )


async def _extract_epub(path: str) -> ExtractedDocument:
    """Extract text from an EPUB file."""
    from octane.extractors.epub.extractor import extract_epub
    from octane.extractors.chunker import chunk_epub_chapters

    loop = asyncio.get_event_loop()

    raw_text, chapters = await loop.run_in_executor(
        None, lambda: extract_epub(path)
    )

    base = SOURCE_RELIABILITY[SourceType.EPUB]
    chunks = chunk_epub_chapters(chapters, reliability_score=base)

    return ExtractedDocument(
        source_type=SourceType.EPUB,
        source_url=str(path),
        title=Path(path).stem,
        raw_text=raw_text,
        chunks=chunks,
        extraction_method="ebooklib",
        reliability_score=base,
        metadata={"path": str(path), "n_chapters": len(chapters)},
    )
