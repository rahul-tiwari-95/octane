"""ContentExtractor — full article text from URLs via trafilatura.

Sits between the Fetcher (URL discovery) and the Synthesizer (LLM summary).
Instead of passing 120-char snippets to the LLM, we pass up to 3 000 chars
of cleaned article text per source.

Extraction chain (tried in order):
  1. trafilatura.fetch_url()  — fast, handles ~80 % of sites (no JS required)
  2. Mark URL as method="failed" — caller can hand off to BrowserAgent

trafilatura calls are synchronous (urllib3 under the hood).
We wrap them with asyncio.to_thread() so the event loop is never blocked.

Graceful degradation:
  - If trafilatura not installed  → returns ExtractedContent(method="unavailable")
  - If a single URL fails         → ExtractedContent(method="failed", text="")
  - Never raises to the caller
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger().bind(component="web.content_extractor")

# ---------------------------------------------------------------------------
# trafilatura — optional dependency
# ---------------------------------------------------------------------------
try:
    import trafilatura  # type: ignore
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False
    logger.warning(
        "trafilatura_not_installed",
        note="ContentExtractor will return method='unavailable' for all URLs",
    )


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ExtractedContent:
    """Result of a single URL extraction attempt.

    Attributes:
        url:        The URL that was (or was attempted to be) fetched.
        text:       Cleaned article text, capped at max_chars_per_source.
        word_count: Approximate word count of *text*.
        method:     How the content was obtained:
                      "trafilatura" — successful trafilatura extraction
                      "browser"     — extracted by BrowserAgent (set by caller)
                      "failed"      — fetch/extraction returned nothing
                      "unavailable" — trafilatura is not installed
        error:      Short error message (empty on success).
    """
    url: str
    text: str
    word_count: int
    method: str          # "trafilatura" | "browser" | "failed" | "unavailable"
    error: str = field(default="")


# ---------------------------------------------------------------------------
# ContentExtractor
# ---------------------------------------------------------------------------

class ContentExtractor:
    """Extracts full article text from a list of URLs using trafilatura.

    Usage::

        extractor = ContentExtractor()
        result = await extractor.extract_url("https://reuters.com/…")
        batch  = await extractor.extract_batch(urls, top_n=5)

    Args:
        max_chars_per_source: Maximum characters to keep per article.
                              Keeps LLM prompts manageable (default 3 000).
        timeout:              Per-URL fetch timeout in seconds (default 15 s).
                              Both the HTTP fetch and the text-extraction steps
                              are each wrapped with ``asyncio.wait_for`` so a
                              single slow URL can never stall the whole batch.
    """

    def __init__(
        self,
        max_chars_per_source: int = 3_000,
        timeout: float = 15.0,
    ) -> None:
        self.max_chars_per_source = max_chars_per_source
        self.timeout = timeout
        self.available = _TRAFILATURA_AVAILABLE

    # ── public helpers ────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_chars: int) -> str:
        """Return at most *max_chars* characters, breaking at a word boundary.

        If the text already fits, it is returned unchanged.
        """
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            return truncated[:last_space]
        return truncated

    async def extract_url(
        self,
        url: str,
        max_chars: int | None = None,
    ) -> ExtractedContent:
        """Extract full text from a single URL.

        Returns an :class:`ExtractedContent` with:

        * ``method="trafilatura"``  — success.
        * ``method="failed"``       — fetch or extraction returned nothing.
        * ``method="unavailable"``  — trafilatura is not installed.

        Never raises.
        """
        cap = max_chars if max_chars is not None else self.max_chars_per_source

        if not _TRAFILATURA_AVAILABLE:
            logger.debug("trafilatura_unavailable", url=url)
            return ExtractedContent(url=url, text="", word_count=0, method="unavailable")

        try:
            # Wrap both synchronous trafilatura calls in threads so the event loop
            # is not blocked (trafilatura uses urllib3 internally).
            # Also enforce self.timeout so a slow/hung URL can't stall the whole
            # extraction batch indefinitely.
            html: str | None = await asyncio.wait_for(
                asyncio.to_thread(trafilatura.fetch_url, url),
                timeout=self.timeout,
            )
            if not html:
                logger.debug("trafilatura_fetch_empty", url=url)
                return ExtractedContent(
                    url=url, text="", word_count=0,
                    method="failed", error="fetch returned empty",
                )

            text: str | None = await asyncio.wait_for(
                asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    url=url,
                    include_comments=False,
                    include_tables=False,
                    favor_precision=True,
                    output_format="txt",
                ),
                timeout=5.0,
            )

            if not text or len(text.strip()) < 100:
                logger.debug("trafilatura_extract_too_short", url=url)
                return ExtractedContent(
                    url=url, text="", word_count=0,
                    method="failed", error="extracted text too short",
                )

            text = self._chunk_text(text.strip(), cap)
            word_count = len(text.split())
            logger.info("trafilatura_extracted", url=url, chars=len(text), words=word_count)
            return ExtractedContent(
                url=url, text=text, word_count=word_count, method="trafilatura",
            )

        except Exception as exc:
            logger.warning("trafilatura_error", url=url, error=str(exc))
            return ExtractedContent(
                url=url, text="", word_count=0, method="failed", error=str(exc),
            )

    async def extract_batch(
        self,
        urls: list[str],
        top_n: int = 5,
    ) -> list[ExtractedContent]:
        """Extract full text from the first *top_n* URLs concurrently.

        All extractions run simultaneously via ``asyncio.gather``.
        Returns the results sorted by ``word_count`` descending so the
        richest articles come first.
        """
        if not urls:
            return []

        target_urls = [u for u in urls if u][:top_n]
        tasks = [self.extract_url(url) for url in target_urls]
        results: list[ExtractedContent] = await asyncio.gather(*tasks)
        return sorted(results, key=lambda a: a.word_count, reverse=True)
