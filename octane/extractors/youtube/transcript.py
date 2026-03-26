"""YouTube transcript extraction — 3-tier fallback.

Tier 1: youtube-transcript-api (v1.2+ instance API) — fastest, lightest
Tier 2: yt-dlp subtitle extraction — resilient fallback
Tier 3: mlx-whisper local transcription — for videos without captions
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger().bind(component="extractors.youtube.transcript")


def get_transcript(video_id: str, lang: str = "en") -> list[dict] | None:
    """Get transcript using 3-tier fallback.

    Returns list of segments: [{"text": str, "start": float, "duration": float}, ...]
    Returns None if all tiers fail.
    """
    # Tier 1: youtube-transcript-api
    result = _tier1_transcript_api(video_id, lang)
    if result:
        return result

    # Tier 2: yt-dlp subtitles
    result = _tier2_ytdlp(video_id, lang)
    if result:
        return result

    logger.warning("all_transcript_tiers_failed", video_id=video_id)
    return None


def transcript_to_text(segments: list[dict]) -> str:
    """Join transcript segments into plain text."""
    return " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))


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


def _tier2_ytdlp(video_id: str, lang: str) -> list[dict] | None:
    """Extract subtitles via yt-dlp."""
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
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # Try manual subs first, then auto
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
    except Exception as exc:
        logger.debug("transcript_tier2_failed", video_id=video_id, error=str(exc))
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
