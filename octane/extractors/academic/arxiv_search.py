"""arXiv search and PDF download via the `arxiv` Python package.

Free, no API key. Rate limit: 1 req/3s (handled by the arxiv Client).
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="extractors.academic.arxiv_search")

# Default paper download directory
DEFAULT_PAPER_DIR = Path.home() / ".octane" / "cache" / "papers"


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for papers matching query.

    Returns list of dicts with: arxiv_id, title, authors, summary, published,
    pdf_url, categories, journal_ref.
    """
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "arxiv_id": paper.entry_id.split("/abs/")[-1],
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "published": paper.published.isoformat() if paper.published else "",
            "updated": paper.updated.isoformat() if paper.updated else "",
            "pdf_url": paper.pdf_url,
            "categories": paper.categories,
            "journal_ref": paper.journal_ref or "",
            "doi": paper.doi or "",
        })
    logger.debug("arxiv_search_ok", query=query[:60], n_results=len(results))
    return results


def download_arxiv_pdf(arxiv_id: str, output_dir: str | Path | None = None) -> Path:
    """Download a paper's PDF by arXiv ID.

    Args:
        arxiv_id: e.g. "2408.09869" or "2408.09869v2"
        output_dir: Directory to save to. Defaults to ~/.octane/cache/papers/

    Returns:
        Path to the downloaded PDF file.
    """
    import arxiv

    output_dir = Path(output_dir) if output_dir else DEFAULT_PAPER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    cached = list(output_dir.glob(f"{arxiv_id.replace('/', '_')}*.pdf"))
    if cached:
        logger.debug("arxiv_pdf_cached", arxiv_id=arxiv_id, path=str(cached[0]))
        return cached[0]

    client = arxiv.Client(delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))

    path = paper.download_pdf(dirpath=str(output_dir))
    logger.info("arxiv_pdf_downloaded", arxiv_id=arxiv_id, path=str(path))
    return Path(path)
