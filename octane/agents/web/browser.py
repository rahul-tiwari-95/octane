"""BrowserAgent — Playwright-powered scraper for JS-heavy and paywalled sites.

Extraction chain:
  1. Headless Chromium with stored cookies (if domain was visited before)
  2. Headless Chromium without cookies (fresh attempt)
  3. HEADED browser + human-assist prompt (interactive mode only)
     → User handles login / cookie consent / scrolling in the visible window
     → Cookies are saved for future headless attempts on this domain
  4. Fall through (return None) — caller stays on snippet-only mode

Cookie store: ~/.octane/browser/cookies/<domain>.json
  - Written via playwright context.storage_state()
  - Loaded on next headless attempt for the same domain

Non-interactive mode (background tasks, Session 16 research daemon):
  - Steps 1 and 2 only. Never opens a headed browser. Never blocks.
  - If both fail: returns None immediately.

Graceful degradation:
  - If playwright not installed: every method returns None
  - Never raises to the caller
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger().bind(component="web.browser")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    logger.warning("playwright_not_installed", note="BrowserAgent will return None for all scrapes")

# Where per-domain cookies are persisted
_COOKIE_DIR = Path.home() / ".octane" / "browser" / "cookies"

# Characters to keep from scraped DOM text
_MAX_CHARS = 3_000

# Seconds to wait for the page to load after navigation
_PAGE_TIMEOUT_MS = 15_000


def _cookie_path(domain: str) -> Path:
    """Return the cookie file path for a domain, e.g. nytimes.com.json"""
    safe_domain = domain.lstrip("www.").replace("/", "_")
    return _COOKIE_DIR / f"{safe_domain}.json"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


async def _page_to_text(page: Any) -> str | None:
    """Extract visible text content from a Playwright page."""
    try:
        text = await page.evaluate("() => document.body.innerText")
        if not text or len(text.strip()) < 100:
            return None
        return text.strip()[:_MAX_CHARS]
    except Exception as exc:
        logger.debug("page_to_text_error", error=str(exc))
        return None


class BrowserAgent:
    """Playwright-powered scraper. Used as fallback by ContentExtractor.

    Args:
        interactive: If True, may open a headed browser and ask the user for help
                     when headless extraction fails. Set False for background tasks.
    """

    def __init__(self, interactive: bool = True) -> None:
        self.interactive = interactive
        self.available = _PLAYWRIGHT_AVAILABLE

    async def scrape(self, url: str) -> str | None:
        """Scrape a URL. Returns cleaned text or None.

        Tries headless first (with and without cookies), then human-assist
        if interactive=True.
        """
        if not _PLAYWRIGHT_AVAILABLE:
            return None

        domain = _extract_domain(url)
        logger.info("browser_scrape_start", url=url, domain=domain, interactive=self.interactive)

        try:
            async with async_playwright() as pw:
                browser: Browser = await pw.chromium.launch(headless=True)

                # ── Attempt 1: headless with stored cookies ──────────────
                cookie_file = _cookie_path(domain)
                if cookie_file.exists():
                    text = await self._scrape_with_context(browser, url, cookie_file=cookie_file)
                    if text:
                        logger.info("browser_headless_cookie_success", url=url)
                        await browser.close()
                        return text
                    logger.debug("browser_headless_cookie_failed", url=url)

                # ── Attempt 2: headless without cookies ───────────────────
                text = await self._scrape_with_context(browser, url, cookie_file=None)
                if text:
                    logger.info("browser_headless_success", url=url)
                    await browser.close()
                    return text
                logger.debug("browser_headless_failed", url=url)

                await browser.close()

                # ── Attempt 3: human-assist (interactive mode only) ───────
                if not self.interactive:
                    logger.info("browser_non_interactive_skip", url=url)
                    return None

                return await self._human_assist(url, domain)

        except Exception as exc:
            logger.warning("browser_scrape_error", url=url, error=str(exc))
            return None

    async def _scrape_with_context(
        self,
        browser: "Browser",
        url: str,
        cookie_file: Path | None,
    ) -> str | None:
        """Launch a headless page, optionally loading stored cookies."""
        try:
            context_kwargs: dict[str, Any] = {
                "user_agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "viewport": {"width": 1280, "height": 800},
            }

            if cookie_file and cookie_file.exists():
                try:
                    with open(cookie_file) as f:
                        storage_state = json.load(f)
                    context_kwargs["storage_state"] = storage_state
                except Exception as exc:
                    logger.debug("cookie_load_error", path=str(cookie_file), error=str(exc))

            context: BrowserContext = await browser.new_context(**context_kwargs)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=5_000)
            text = await _page_to_text(page)
            await context.close()
            return text

        except Exception as exc:
            logger.debug("headless_context_error", url=url, error=str(exc))
            return None

    async def _human_assist(self, url: str, domain: str) -> str | None:
        """Open a headed (visible) browser and ask the user to interact.

        The user handles login, cookie consent, or scroll-to-load.
        After they press Enter in the terminal, we capture the DOM and
        persist the cookies for future headless access to this domain.

        asyncio.to_thread() wraps input() so the event loop is never blocked.
        """
        logger.info("browser_human_assist_start", url=url, domain=domain)

        try:
            async with async_playwright() as pw:
                browser: Browser = await pw.chromium.launch(
                    headless=False,
                    args=["--start-maximized"],
                )
                context: BrowserContext = await browser.new_context(no_viewport=True)
                page = await context.new_page()

                await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_TIMEOUT_MS)

                print(
                    f"\n🌐  Browser window opened for: {url}\n"
                    f"    Please log in / accept cookies / scroll to load content.\n"
                    f"    When the page shows the content you want, press Enter here."
                )

                # input() wrapped in to_thread — event loop stays alive while waiting
                await asyncio.to_thread(input, "    ➜  Press Enter when done: ")

                text = await _page_to_text(page)

                # Persist cookies/session for future headless access
                _COOKIE_DIR.mkdir(parents=True, exist_ok=True)
                cookie_file = _cookie_path(domain)
                try:
                    storage = await context.storage_state()
                    with open(cookie_file, "w") as f:
                        json.dump(storage, f, indent=2)
                    logger.info("browser_cookies_saved", domain=domain, path=str(cookie_file))
                    print(f"    ✅  Cookies saved — future access to {domain} will be headless.\n")
                except Exception as exc:
                    logger.warning("browser_cookie_save_error", error=str(exc))

                await context.close()
                await browser.close()

                if text:
                    logger.info("browser_human_assist_success", url=url, chars=len(text))
                    return text

                logger.warning("browser_human_assist_empty", url=url)
                return None

        except Exception as exc:
            logger.warning("browser_human_assist_error", url=url, error=str(exc))
            return None

    def clear_cookies(self, domain: str) -> bool:
        """Delete stored cookies for a domain. Returns True if file existed."""
        cookie_file = _cookie_path(domain)
        if cookie_file.exists():
            cookie_file.unlink()
            logger.info("browser_cookies_cleared", domain=domain)
            return True
        return False

    def list_saved_domains(self) -> list[str]:
        """Return list of domains with saved cookies."""
        if not _COOKIE_DIR.exists():
            return []
        return [f.stem for f in _COOKIE_DIR.glob("*.json")]
