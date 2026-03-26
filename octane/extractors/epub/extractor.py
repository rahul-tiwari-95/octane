"""EPUB text extraction via ebooklib + BeautifulSoup.

Preserves chapter structure as section boundaries for downstream chunking.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="extractors.epub.extractor")


def extract_epub(path: str | Path) -> tuple[str, list[dict]]:
    """Extract text from an EPUB file.

    Args:
        path: Path to the EPUB file.

    Returns:
        (full_text, chapters) — full text and list of chapter dicts
        [{title: str, text: str, index: int}].
    """
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPUB not found: {path}")

    book = epub.read_epub(str(path))

    chapters: list[dict] = []
    all_text_parts: list[str] = []
    idx = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if not text or len(text.strip()) < 20:
            continue

        # Try to extract chapter title from first heading
        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(strip=True) if heading else f"Section {idx + 1}"

        chapters.append({
            "title": title,
            "text": text,
            "index": idx,
        })
        all_text_parts.append(f"## {title}\n\n{text}")
        idx += 1

    full_text = "\n\n".join(all_text_parts)
    logger.debug("epub_extracted", path=str(path), chapters=len(chapters), words=len(full_text.split()))
    return full_text, chapters
