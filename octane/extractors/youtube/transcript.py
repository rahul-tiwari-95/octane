"""YouTube transcript extraction — 3-tier fallback (Playwright-first).

Tier 0 (PRIMARY): Playwright headless browser with cookies.
        Cookies are sourced automatically from Chrome/Brave/Firefox via
        yt-dlp's ``cookiesfrombrowser`` — zero setup required.  Falls back
        to stored ``~/.octane/browser/cookies/youtube.com.json`` from
        ``octane extract youtube-login`` if available.
        A real browser session means YouTube cannot distinguish this from
        a normal user — bypasses IP blocks and 429 rate limits.

Tier 1: youtube-transcript-api (v1.2+ instance API) — fastest, lightest,
        but subject to IP bans for unauthenticated requests.

Tier 2: yt-dlp subtitle download without cookies — last resort fallback.

Optional one-time setup:  ``octane extract youtube-login``
  → Opens a headed Chromium window on youtube.com.
  → User logs into their Google account, then presses Enter.
  → Cookies are saved for future headless runs.
  (Usually not needed — Chrome cookies are extracted automatically.)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="extractors.youtube.transcript")

# Cookie store — same directory as the web BrowserAgent
_COOKIE_DIR = Path.home() / ".octane" / "browser" / "cookies"
_YOUTUBE_COOKIE_FILE = _COOKIE_DIR / "youtube.com.json"

# Browsers that yt-dlp can read cookies from (checked in order)
_BROWSER_COOKIE_CANDIDATES = ("chrome", "chromium", "brave", "firefox")

# ── Browser detection (cached) ───────────────────────────────────────────────

_COOKIE_BROWSER_SENTINEL = object()
_COOKIE_BROWSER: str | None | object = _COOKIE_BROWSER_SENTINEL


def _detect_cookie_browser() -> str | None:
    """Return the first installed browser name that yt-dlp can read cookies from."""
    if sys.platform == "darwin":
        browser_paths = {
            "chrome": Path("/Applications/Google Chrome.app"),
            "chromium": Path("/Applications/Chromium.app"),
            "brave": Path("/Applications/Brave Browser.app"),
            "firefox": Path("/Applications/Firefox.app"),
        }
        for name, path in browser_paths.items():
            if path.exists():
                return name
    else:
        return "chrome"
    return None


def _get_cookie_browser() -> str | None:
    global _COOKIE_BROWSER
    if _COOKIE_BROWSER is not _COOKIE_BROWSER_SENTINEL:
        return _COOKIE_BROWSER  # type: ignore[return-value]
    _COOKIE_BROWSER = _detect_cookie_browser()
    logger.debug("cookie_browser_detected", browser=_COOKIE_BROWSER)
    return _COOKIE_BROWSER


# ── Chrome cookie extraction for Playwright ──────────────────────────────────

# Cache: extracted once per process, reused across videos
_CACHED_PW_COOKIES: list[dict] | None = None


def _extract_browser_cookies_for_playwright() -> list[dict]:
    """Extract YouTube cookies from Chrome/Brave/Firefox via yt-dlp and convert
    to Playwright's ``add_cookies()`` format.

    Returns an empty list if no browser is detected or extraction fails.
    """
    global _CACHED_PW_COOKIES
    if _CACHED_PW_COOKIES is not None:
        return _CACHED_PW_COOKIES

    browser = _get_cookie_browser()
    if not browser:
        _CACHED_PW_COOKIES = []
        return _CACHED_PW_COOKIES

    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "cookiesfrombrowser": (browser,),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            jar = ydl.cookiejar
            pw_cookies: list[dict] = []
            for c in jar:
                if not c.domain:
                    continue
                # Only keep YouTube / Google cookies (auth + consent)
                if not any(d in c.domain for d in (".youtube.com", ".google.com", "youtube.com", "google.com")):
                    continue
                cookie: dict = {
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain,
                    "path": c.path or "/",
                }
                # Playwright only accepts expires=-1 (session) or positive unix timestamp.
                # yt-dlp may return Chrome-epoch microseconds (since 1601-01-01) —
                # detect via magnitude and convert to Unix seconds.
                if c.expires and c.expires > 0:
                    exp = c.expires
                    if exp > 13_000_000_000_000_000:  # Chrome epoch (µs since 1601)
                        exp = int(exp / 1_000_000) - 11_644_473_600
                    elif exp > 100_000_000_000:  # milliseconds
                        exp = int(exp / 1_000)
                    if exp > 0:
                        cookie["expires"] = exp
                if c.secure:
                    cookie["secure"] = True
                pw_cookies.append(cookie)

            logger.debug("browser_cookies_extracted", browser=browser, count=len(pw_cookies))
            _CACHED_PW_COOKIES = pw_cookies
            return pw_cookies

    except Exception as exc:
        logger.debug("browser_cookie_extraction_failed", browser=browser, error=str(exc))
        _CACHED_PW_COOKIES = []
        return _CACHED_PW_COOKIES


# ── Public API ───────────────────────────────────────────────────────────────


def get_transcript(video_id: str, lang: str = "en") -> list[dict] | None:
    """Get transcript using 3-tier fallback (Playwright-first).

    Returns list of segments: [{"text": str, "start": float, "duration": float}, ...]
    Returns None if all tiers fail.
    """
    # Tier 0 (PRIMARY): Playwright with cookies (auto-extracted from Chrome or stored)
    result = _tier0_playwright(video_id, lang)
    if result:
        return result

    # Tier 1: youtube-transcript-api (fast but often IP-blocked)
    result = _tier1_transcript_api(video_id, lang)
    if result:
        return result

    # Tier 2: yt-dlp without cookies (last resort)
    result = _tier2_ytdlp_no_cookies(video_id, lang)
    if result:
        return result

    logger.warning("all_transcript_tiers_failed", video_id=video_id)
    return None


def transcript_to_text(segments: list[dict]) -> str:
    """Join transcript segments into plain text."""
    return " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))


# ── YouTube Login (one-time setup) ───────────────────────────────────────────

_AUTH_COOKIE_NAMES = {"SID", "HSID", "SSID", "APISID", "SAPISID", "LOGIN_INFO"}


async def youtube_login() -> bool:
    """Extract YouTube/Google cookies so Tier 0 can use authenticated sessions.

    Strategy 1: Auto-extract cookies from Chrome/Brave/Firefox via yt-dlp's
                ``cookiesfrombrowser``.  No browser window needed.
    Strategy 2: Open YouTube using the user's real Chrome profile (persistent
                context) — reuses their existing Google session so Google
                won't block sign-in.  Requires Chrome to be closed.

    Saves cookies to ~/.octane/browser/cookies/youtube.com.json.
    Returns True on success.
    """
    _COOKIE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Strategy 1: Auto-extract from browser cookie jar ──────────────
    print("\n🔑  Attempting to auto-extract YouTube cookies from your browser…")
    # Force a fresh extraction (the process cache may be stale)
    global _CACHED_PW_COOKIES
    _CACHED_PW_COOKIES = None

    pw_cookies = _extract_browser_cookies_for_playwright()
    if pw_cookies and any(c["name"] in _AUTH_COOKIE_NAMES for c in pw_cookies):
        storage = {"cookies": pw_cookies, "origins": []}
        with open(_YOUTUBE_COOKIE_FILE, "w") as f:
            json.dump(storage, f, indent=2)
        logger.info("youtube_cookies_auto_extracted", path=str(_YOUTUBE_COOKIE_FILE), count=len(pw_cookies))
        print(f"    ✅  {len(pw_cookies)} cookies auto-extracted from your browser — no sign-in needed!\n")
        return True

    print("    ⚠  Auto-extraction didn't find auth cookies — trying Chrome profile…\n")

    # ── Strategy 2: Persistent context with real Chrome profile ───────
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("playwright_not_installed")
        print("    ❌  Playwright is not installed.  Run:  playwright install chromium\n")
        return False

    import sys
    if sys.platform == "darwin":
        chrome_profile = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    elif sys.platform == "linux":
        chrome_profile = Path.home() / ".config" / "google-chrome"
    else:
        chrome_profile = Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data"

    if not chrome_profile.exists():
        print(
            "    ❌  Chrome profile not found. Make sure Google Chrome is installed\n"
            "       and you've signed in to Google at least once.\n"
        )
        return False

    print(
        "🌐  Opening YouTube using your Chrome profile.\n"
        "    This reuses your existing Google login — no sign-in needed.\n\n"
        "    ⚠  Chrome must be CLOSED for this to work.\n"
        "       Please quit Chrome completely (Cmd+Q), then press Enter.\n"
    )

    import asyncio
    await asyncio.to_thread(input, "    ➜  Press Enter when Chrome is closed: ")

    try:
        async with async_playwright() as pw:
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=str(chrome_profile),
                headless=False,
                channel="chrome",
                no_viewport=True,
                args=["--start-maximized"],
            )

            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto("https://www.youtube.com", wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(3000)  # let cookies settle

            print("    🔍  Checking login status…")

            cookies = await context.cookies("https://www.youtube.com")
            google_cookies = await context.cookies("https://www.google.com")
            # Merge, preferring YouTube cookies on name collision
            seen = {c["name"] for c in cookies}
            all_cookies = cookies + [c for c in google_cookies if c["name"] not in seen]

            has_auth = any(c["name"] in _AUTH_COOKIE_NAMES for c in all_cookies)

            await context.close()

            if has_auth:
                storage = {"cookies": all_cookies, "origins": []}
                with open(_YOUTUBE_COOKIE_FILE, "w") as f:
                    json.dump(storage, f, indent=2)
                logger.info("youtube_cookies_saved", path=str(_YOUTUBE_COOKIE_FILE), count=len(all_cookies))
                print(f"    ✅  {len(all_cookies)} cookies saved from your Chrome profile!\n")
                return True
            else:
                print(
                    "    ⚠  You don't appear to be logged in to Google in Chrome.\n"
                    "       1. Open Chrome normally and sign in to your Google account.\n"
                    "       2. Visit youtube.com and confirm you see your profile icon.\n"
                    "       3. Close Chrome and run this command again.\n"
                )
                return False

    except Exception as exc:
        err = str(exc)
        if "SingletonLock" in err or "already running" in err.lower() or "lock" in err.lower():
            print(
                "\n    ❌  Chrome is still running. Please close ALL Chrome windows\n"
                "       (Cmd+Q on macOS), then try again.\n"
                "       Tip: Check Activity Monitor for 'Google Chrome' processes.\n"
            )
        else:
            print(f"\n    ❌  Failed to open Chrome profile: {err[:120]}\n")
            logger.warning("youtube_login_persistent_failed", error=err[:200])
        return False


# ── Tier 0 (PRIMARY): Playwright with cookies ────────────────────────────────


def _tier0_playwright(video_id: str, lang: str) -> list[dict] | None:
    """Fetch transcript via Playwright headless browser.

    Cookie sources (checked in order):
     1. Stored cookies from ``octane extract youtube-login``
     2. Auto-extracted from Chrome/Brave/Firefox via yt-dlp

    Uses a real Chromium browser — YouTube cannot distinguish from a normal
    user, so IP blocks and 429 rate limits are bypassed.
    """
    try:
        import asyncio

        return asyncio.run(_tier0_playwright_async(video_id, lang))
    except Exception as exc:
        logger.debug("transcript_tier0_failed", video_id=video_id, error=str(exc))
        return None


async def _tier0_playwright_async(video_id: str, lang: str) -> list[dict] | None:
    """Async implementation of Playwright transcript extraction."""
    from playwright.async_api import async_playwright

    url = f"https://www.youtube.com/watch?v={video_id}"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        # ── Build context with cookies ──
        context_kwargs: dict = {
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "viewport": {"width": 1280, "height": 800},
        }

        context = await browser.new_context(**context_kwargs)

        # Inject cookies — always use add_cookies() for reliable injection.
        # Source 1: stored session from youtube-login
        cookies_injected = False
        if _YOUTUBE_COOKIE_FILE.exists():
            try:
                with open(_YOUTUBE_COOKIE_FILE) as f:
                    stored = json.load(f)
                cookies = stored.get("cookies", [])
                if cookies:
                    # Ensure sameSite is set (required by Playwright)
                    for c in cookies:
                        if "sameSite" not in c:
                            c["sameSite"] = "Lax"
                    await context.add_cookies(cookies)
                    cookies_injected = True
                    logger.debug("playwright_loaded_stored_cookies", count=len(cookies))
            except Exception:
                pass

        # Source 2: auto-extract from Chrome if no stored session
        if not cookies_injected:
            pw_cookies = _extract_browser_cookies_for_playwright()
            if pw_cookies:
                for c in pw_cookies:
                    if "sameSite" not in c:
                        c["sameSite"] = "Lax"
                await context.add_cookies(pw_cookies)
                logger.debug("playwright_injected_browser_cookies", count=len(pw_cookies))

        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            # YouTube pages never reach "networkidle" — wait for player instead
            await page.wait_for_timeout(3000)

            # Strategy 1: Open the transcript panel and scrape from the DOM
            segments = await _playwright_extract_transcript(page)
            if segments:
                logger.debug("transcript_tier0_ok", video_id=video_id, segments=len(segments), method="panel")
                return segments

            # Strategy 2: Fetch YouTube's timedtext API through the browser context
            segments = await _playwright_extract_timedtext(page, video_id, lang)
            if segments:
                logger.debug("transcript_tier0_ok", video_id=video_id, segments=len(segments), method="timedtext")
                return segments

            return None
        finally:
            await context.close()
            await browser.close()


async def _playwright_extract_transcript(page) -> list[dict] | None:
    """Open YouTube's transcript panel and scrape segments from the DOM.

    Handles both the new (2025+) ``transcript-segment-view-model`` elements
    and the legacy ``ytd-transcript-segment-renderer`` elements.
    """
    try:
        # Click "...more" in the description to reveal the transcript button
        more_btn = page.locator("tp-yt-paper-button#expand")
        if await more_btn.count() > 0:
            await more_btn.first.click()
            await page.wait_for_timeout(1000)

        # Look for "Show transcript" button
        transcript_btn = page.locator('button:has-text("Show transcript")')
        if await transcript_btn.count() == 0:
            transcript_btn = page.get_by_role("button", name=re.compile(r"show transcript", re.IGNORECASE))
        if await transcript_btn.count() == 0:
            return None

        await transcript_btn.first.click()

        # Wait for actual segments to appear (the panel takes time to load)
        try:
            await page.wait_for_selector(
                'transcript-segment-view-model, ytd-transcript-segment-renderer',
                timeout=12000,
            )
        except Exception:
            # Panel may still be loading (spinner) — try a final short wait
            await page.wait_for_timeout(3000)

        # Extract segments via JS — handles both old and new YouTube DOM
        segments = await page.evaluate("""() => {
            const results = [];

            // New YouTube (2025+): transcript-segment-view-model elements
            const newSegs = document.querySelectorAll('transcript-segment-view-model');
            if (newSegs.length > 0) {
                for (const seg of newSegs) {
                    const tsEl = seg.querySelector(
                        '.ytwTranscriptSegmentViewModelTimestamp, '
                        + '.segment-timestamp'
                    );
                    const ts = tsEl ? tsEl.textContent.trim() : '';

                    // Remove timestamp and a11y label elements, keep only transcript text
                    const clone = seg.cloneNode(true);
                    const remove = clone.querySelectorAll(
                        '.ytwTranscriptSegmentViewModelTimestamp, '
                        + '.ytwTranscriptSegmentViewModelTimestampA11yLabel, '
                        + '.segment-timestamp'
                    );
                    for (const el of remove) el.remove();
                    let text = clone.textContent.trim().replace(/\\s+/g, ' ');

                    if (text) {
                        results.push({timestamp: ts, text: text});
                    }
                }
                return results;
            }

            // Legacy YouTube: ytd-transcript-segment-renderer elements
            const oldSegs = document.querySelectorAll('ytd-transcript-segment-renderer');
            for (const seg of oldSegs) {
                const tsEl = seg.querySelector('.segment-timestamp');
                const textEl = seg.querySelector('.segment-text');
                const ts = tsEl ? tsEl.textContent.trim() : '';
                const text = textEl ? textEl.textContent.trim() : '';
                if (text) {
                    results.push({timestamp: ts, text: text});
                }
            }
            return results;
        }""")

        if not segments:
            return None

        result = []
        for seg in segments:
            start = _parse_timestamp(seg.get("timestamp", "0:00"))
            result.append({
                "text": seg["text"],
                "start": start,
                "duration": 0.0,
            })

        return result if result else None

    except Exception as exc:
        logger.debug("playwright_transcript_panel_failed", error=str(exc))
        return None


async def _playwright_extract_timedtext(page, video_id: str, lang: str) -> list[dict] | None:
    """Extract transcript by fetching YouTube's signed timedtext URL from the page."""
    try:
        # Extract the signed captionTracks URL from ytInitialPlayerResponse
        # YouTube requires the full signed URL (with signature, key, expire params)
        caption_url = await page.evaluate(r"""(targetLang) => {
            if (typeof ytInitialPlayerResponse !== 'undefined') {
                const tracks = ytInitialPlayerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
                if (tracks && tracks.length > 0) {
                    for (const t of tracks) {
                        if (t.languageCode === targetLang) return t.baseUrl;
                    }
                    return tracks[0].baseUrl;
                }
            }
            return null;
        }""", lang)

        if not caption_url:
            logger.debug("playwright_no_caption_tracks", video_id=video_id)
            return None

        # Fetch timedtext via the page's own fetch() — uses the browser session
        # and cookies so YouTube treats it as a normal user request (avoids 429).
        sep = "&" if "?" in caption_url else "?"
        api_url = f"{caption_url}{sep}fmt=json3"

        data = await page.evaluate(r"""async (url) => {
            try {
                const resp = await fetch(url);
                if (!resp.ok) return {error: resp.status};
                return await resp.json();
            } catch(e) {
                return {error: e.message};
            }
        }""", api_url)

        if not data or "error" in data:
            logger.debug("playwright_timedtext_fetch_error", error=data.get("error") if data else "null")
            return None
        events = data.get("events", [])
        if not events:
            return None

        segments = []
        for event in events:
            segs = event.get("segs", [])
            text = "".join(s.get("utf8", "") for s in segs).strip()
            text = text.replace("\n", " ")
            if text:
                start_ms = event.get("tStartMs", 0)
                dur_ms = event.get("dDurationMs", 0)
                segments.append({
                    "text": text,
                    "start": start_ms / 1000.0,
                    "duration": dur_ms / 1000.0,
                })

        return segments if segments else None

    except Exception as exc:
        logger.debug("playwright_timedtext_failed", error=str(exc))
        return None


def _parse_timestamp(ts: str) -> float:
    """Parse a YouTube transcript timestamp like '1:23' or '1:02:34' to seconds."""
    parts = ts.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return 0.0


def _tier1_transcript_api(video_id: str, lang: str) -> list[dict] | None:
    """youtube-transcript-api v1.2+ instance-based API."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=[lang, "en"])
        segments = []
        for snippet in transcript:
            segments.append({
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
            })
        logger.debug("transcript_tier1_ok", video_id=video_id, segments=len(segments))
        return segments
    except Exception as exc:
        logger.debug("transcript_tier1_failed", video_id=video_id, error=str(exc))
        return None


def _tier2_ytdlp_no_cookies(video_id: str, lang: str) -> list[dict] | None:
    """Extract subtitles via yt-dlp without any cookies (last resort)."""
    try:
        import yt_dlp

        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
            "ignore_no_formats_error": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        segments = _extract_subs_from_info(info, lang)
        if segments:
            logger.debug("transcript_tier2_ok", video_id=video_id, segments=len(segments))
        return segments
    except Exception as exc:
        logger.debug("transcript_tier2_failed", video_id=video_id, error=str(exc))
        return None


def _extract_subs_from_info(info: dict, lang: str) -> list[dict] | None:
    """Extract subtitle segments from yt-dlp info dict."""
    subs = info.get("subtitles", {}).get(lang) or info.get("automatic_captions", {}).get(lang)
    if not subs:
        return None

    # yt-dlp returns subtitle data inline for some formats
    for sub_entry in subs:
        if sub_entry.get("ext") == "vtt" and sub_entry.get("data"):
            return _parse_vtt(sub_entry["data"])

    # If no inline data, get the URL and fetch
    for sub_entry in subs:
        if sub_entry.get("url"):
            import httpx

            resp = httpx.get(sub_entry["url"], timeout=15.0)
            if resp.status_code == 200:
                return _parse_vtt(resp.text)

    return None


def _parse_vtt(vtt_text: str) -> list[dict]:
    """Parse WebVTT subtitle text into segments. Deduplicates overlapping auto-subs."""
    timestamp_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
    )
    segments: list[dict] = []
    seen_texts: set[str] = set()

    lines = vtt_text.split("\n")
    i = 0
    while i < len(lines):
        match = timestamp_re.match(lines[i].strip())
        if match:
            start = _vtt_time_to_seconds(match.group(1))
            end = _vtt_time_to_seconds(match.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                # Strip VTT position tags like <00:01:23.456>
                clean = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", lines[i].strip())
                clean = re.sub(r"</?[^>]+>", "", clean)  # Strip HTML tags
                if clean:
                    text_lines.append(clean)
                i += 1
            text = " ".join(text_lines)
            if text and text not in seen_texts:
                seen_texts.add(text)
                segments.append({
                    "text": text,
                    "start": start,
                    "duration": end - start,
                })
        else:
            i += 1

    return segments


def _vtt_time_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s
