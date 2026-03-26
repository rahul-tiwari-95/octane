"""PDF text extraction — auto-routing between tiers.

Tier 1: pymupdf4llm  — fast structured Markdown (0.12s/page)
Tier 2: docling       — AI-powered deep understanding (1-5s/page, optional)

Auto-router checks text density: if pymupdf4llm returns <50 chars/page,
the PDF is likely scanned → route to docling OCR if available.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="extractors.pdf.extractor")


def extract_pdf(path: str | Path, quality: str = "auto") -> tuple[str, str]:
    """Extract text from a PDF file.

    Args:
        path: Path to the PDF file.
        quality: "fast" (pymupdf4llm only), "deep" (docling), or "auto" (smart routing).

    Returns:
        (text, method) — the extracted markdown text and the method used.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if quality == "fast":
        return _tier1_pymupdf(path), "pymupdf4llm"
    elif quality == "deep":
        return _tier2_docling(path), "docling"

    # Auto-route: try fast first, fall back to deep if low text density
    text = _tier1_pymupdf(path)
    n_pages = _count_pages(path)
    chars_per_page = len(text) / max(n_pages, 1)

    if chars_per_page < 50 and n_pages > 0:
        logger.info(
            "pdf_low_density_routing_to_deep",
            path=str(path),
            chars_per_page=round(chars_per_page, 1),
        )
        try:
            text = _tier2_docling(path)
            return text, "docling"
        except ImportError:
            logger.warning("docling_not_installed_using_pymupdf")
            return text, "pymupdf4llm"

    return text, "pymupdf4llm"


def _tier1_pymupdf(path: Path) -> str:
    """Fast Markdown extraction via pymupdf4llm."""
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(path))


def _tier2_docling(path: Path) -> str:
    """Deep AI-powered extraction via docling."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    except ImportError:
        raise ImportError(
            "docling is not installed. Install with: pip install docling"
        )


def _count_pages(path: Path) -> int:
    """Count pages in a PDF using pymupdf."""
    try:
        import pymupdf

        doc = pymupdf.open(str(path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0
