"""Core data models for the extraction pipeline.

Every piece of extracted content carries source type, trust metadata,
extraction provenance, and cache identity from birth.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(Enum):
    YOUTUBE = "youtube"
    ARXIV = "arxiv"
    PDF = "pdf"
    EPUB = "epub"
    WEB = "web"


class TaskState(Enum):
    PENDING = "pending"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    CLASSIFYING = "classifying"
    CHUNKING = "chunking"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    SPAWNED_CHILDREN = "spawned"


# ── Trust & Reliability ───────────────────────────────────────────────

SOURCE_RELIABILITY: dict[SourceType, float] = {
    SourceType.ARXIV: 0.92,
    SourceType.PDF: 0.75,
    SourceType.EPUB: 0.70,
    SourceType.YOUTUBE: 0.55,
    SourceType.WEB: 0.40,
}

RELIABILITY_MODIFIERS: dict[str, float] = {
    "peer_reviewed": +0.10,
    "high_citation": +0.08,
    "verified_channel": +0.05,
    "long_form_lecture": +0.04,
    "auto_generated_captions": -0.05,
    "scanned_ocr": -0.08,
    "music_content": -0.20,
    "short_form": -0.10,
}


# ── Chunk-Level Model ─────────────────────────────────────────────────

@dataclass
class ChunkMetadata:
    page: int | None = None
    chapter: str | None = None
    section: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    is_table: bool = False
    is_code: bool = False
    is_junk: bool = False


@dataclass
class TextChunk:
    text: str
    index: int = 0
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    reliability_score: float = 0.5
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())


# ── Document-Level Model ──────────────────────────────────────────────

@dataclass
class ExtractedDocument:
    source_type: SourceType
    source_url: str
    title: str = ""
    author: str = ""
    raw_text: str = ""
    chunks: list[TextChunk] = field(default_factory=list)
    extraction_method: str = ""
    reliability_score: float = -1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    extracted_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.content_hash and self.raw_text:
            self.content_hash = hashlib.sha256(self.raw_text.encode()).hexdigest()[:16]
        if self.reliability_score < 0:
            self.reliability_score = SOURCE_RELIABILITY.get(self.source_type, 0.5)

    @property
    def total_words(self) -> int:
        return len(self.raw_text.split()) if self.raw_text else 0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "title": self.title,
            "author": self.author,
            "total_words": self.total_words,
            "total_chunks": self.total_chunks,
            "extraction_method": self.extraction_method,
            "reliability_score": round(self.reliability_score, 3),
            "content_hash": self.content_hash,
            "extracted_at": self.extracted_at,
            "metadata": self.metadata,
        }
