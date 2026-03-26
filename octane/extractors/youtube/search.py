"""YouTube search via scrapetube (primary) + yt-dlp (fallback)."""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="extractors.youtube.search")


def search_youtube(query: str, limit: int = 5) -> list[dict]:
    """Search YouTube using scrapetube's InnerTube API.

    Returns list of dicts with: video_id, title, channel, duration, view_count, url.
    Falls back to yt-dlp if scrapetube fails.
    """
    try:
        return _search_scrapetube(query, limit)
    except Exception as exc:
        logger.warning("scrapetube_failed_falling_back", error=str(exc))
        return _search_ytdlp(query, limit)


def _search_scrapetube(query: str, limit: int) -> list[dict]:
    import scrapetube

    videos = scrapetube.get_search(query, limit=limit, sort_by="relevance")
    results = []
    for v in videos:
        vid = v.get("videoId", "")
        if not vid:
            continue
        title = v.get("title", {})
        if isinstance(title, dict):
            title = title.get("runs", [{}])[0].get("text", "")

        # Duration comes as accessibility text like "3 minutes, 42 seconds"
        length_text = v.get("lengthText", {})
        if isinstance(length_text, dict):
            length_text = length_text.get("simpleText", "")

        # View count
        view_text = v.get("viewCountText", {})
        if isinstance(view_text, dict):
            view_text = view_text.get("simpleText", "")

        # Channel
        channel = ""
        owner = v.get("ownerText", {})
        if isinstance(owner, dict):
            runs = owner.get("runs", [])
            if runs:
                channel = runs[0].get("text", "")

        results.append({
            "video_id": vid,
            "title": str(title),
            "channel": channel,
            "duration": str(length_text),
            "view_count": str(view_text),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return results


def _search_ytdlp(query: str, limit: int) -> list[dict]:
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "default_search": f"ytsearch{limit}",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)

    results = []
    for entry in (info or {}).get("entries", []):
        if not entry:
            continue
        vid = entry.get("id", "")
        results.append({
            "video_id": vid,
            "title": entry.get("title", ""),
            "channel": entry.get("uploader", ""),
            "duration": str(entry.get("duration", "")),
            "view_count": str(entry.get("view_count", "")),
            "url": entry.get("url", f"https://www.youtube.com/watch?v={vid}"),
        })
    return results
