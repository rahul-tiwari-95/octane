"""Source-aware chunking strategies.

| Source  | Strategy                    | Size       | Overlap    |
|---------|-----------------------------|------------|------------|
| YouTube | Timestamp-based semantic    | 30-60s     | 5s         |
| arXiv   | Section-aware recursive     | 512 tokens | 50 tokens  |
| PDF     | Recursive character         | 512 tokens | 50 tokens  |
| EPUB    | Chapter-first, then recurse | 512 tokens | 50 tokens  |
"""

from __future__ import annotations

from octane.extractors.models import ChunkMetadata, SourceType, TextChunk


def chunk_text(
    text: str,
    source_type: SourceType = SourceType.PDF,
    reliability_score: float = 0.5,
) -> list[TextChunk]:
    """Chunk text using source-appropriate strategy.

    Uses langchain RecursiveCharacterTextSplitter with separators
    tuned per source type.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if source_type in (SourceType.ARXIV, SourceType.PDF):
        separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
    elif source_type == SourceType.EPUB:
        separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
    else:
        separators = ["\n\n", "\n", ". ", " "]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,        # ~512 tokens at ~4 chars/token
        chunk_overlap=200,      # ~50 tokens
        separators=separators,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    chunks: list[TextChunk] = []
    for i, chunk_text_str in enumerate(raw_chunks):
        chunks.append(TextChunk(
            text=chunk_text_str,
            index=i,
            metadata=ChunkMetadata(),
            reliability_score=reliability_score,
        ))
    return chunks


def chunk_transcript_by_time(
    segments: list[dict],
    target_duration: float = 45.0,
    reliability_score: float = 0.55,
) -> list[TextChunk]:
    """Chunk YouTube transcript segments by time windows.

    Groups segments into ~45-second windows, splitting at sentence
    boundaries when possible.
    """
    if not segments:
        return []

    chunks: list[TextChunk] = []
    current_texts: list[str] = []
    current_start = segments[0].get("start", 0.0)
    chunk_idx = 0

    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue

        current_texts.append(seg_text)
        elapsed = seg_start - current_start + seg.get("duration", 0.0)

        if elapsed >= target_duration:
            text = " ".join(current_texts)
            chunks.append(TextChunk(
                text=text,
                index=chunk_idx,
                metadata=ChunkMetadata(
                    timestamp_start=current_start,
                    timestamp_end=seg_start + seg.get("duration", 0.0),
                ),
                reliability_score=reliability_score,
            ))
            chunk_idx += 1
            current_texts = []
            current_start = seg_start + seg.get("duration", 0.0)

    # Flush remaining
    if current_texts:
        text = " ".join(current_texts)
        last_seg = segments[-1]
        chunks.append(TextChunk(
            text=text,
            index=chunk_idx,
            metadata=ChunkMetadata(
                timestamp_start=current_start,
                timestamp_end=last_seg.get("start", 0.0) + last_seg.get("duration", 0.0),
            ),
            reliability_score=reliability_score,
        ))

    return chunks


def chunk_epub_chapters(
    chapters: list[dict],
    reliability_score: float = 0.70,
) -> list[TextChunk]:
    """Chunk EPUB by chapter boundaries, then recursively split large chapters."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[TextChunk] = []
    global_idx = 0

    for chapter in chapters:
        title = chapter.get("title", "")
        text = chapter.get("text", "")
        if not text:
            continue

        if len(text) <= 2048:
            chunks.append(TextChunk(
                text=text,
                index=global_idx,
                metadata=ChunkMetadata(chapter=title),
                reliability_score=reliability_score,
            ))
            global_idx += 1
        else:
            sub_chunks = splitter.split_text(text)
            for sub in sub_chunks:
                chunks.append(TextChunk(
                    text=sub,
                    index=global_idx,
                    metadata=ChunkMetadata(chapter=title),
                    reliability_score=reliability_score,
                ))
                global_idx += 1

    return chunks
