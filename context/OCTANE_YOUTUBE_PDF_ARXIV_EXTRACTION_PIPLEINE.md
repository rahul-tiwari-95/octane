# Octane Extraction Engine — W16 Task Architecture

**Stateful, recursive, interruptible content extraction for the Octane Daemon.**

> Target hardware: M1 Max 64GB, M5 Pro 64GB · No Google API keys · All local except YouTube runtime access  
> Integrates with: BodegaRouter (FAST/MID/REASON/EMBED), OSA Decomposer/Evaluator, ResearchSynthesizer, MSR  
> Last updated: March 2026 · v2 (Daemon-native rewrite)

---

## Table of Contents

1. [Design Philosophy: Why Not a Library](#1-design-philosophy-why-not-a-library)
2. [Core Data Model (Stateful + Trust-Scored)](#2-core-data-model-stateful--trust-scored)
3. [Daemon Task Lifecycle & Wave Model](#3-daemon-task-lifecycle--wave-model)
4. [Human-in-the-Loop Terminal (HIL)](#4-human-in-the-loop-terminal-hil)
5. [YouTube Search & Transcript Extraction](#5-youtube-search--transcript-extraction)
6. [arXiv & Academic Paper Search](#6-arxiv--academic-paper-search)
7. [PDF Text Extraction](#7-pdf-text-extraction)
8. [EPUB & Ebook Extraction](#8-epub--ebook-extraction)
9. [Chunking Strategies (Source-Aware)](#9-chunking-strategies-source-aware)
10. [Content Classification, Filtering & Junk Detection](#10-content-classification-filtering--junk-detection)
11. [Trust & Verification Scoring](#11-trust--verification-scoring)
12. [Recursive Citation Discovery](#12-recursive-citation-discovery)
13. [Cache Architecture (First-Class Citizen)](#13-cache-architecture-first-class-citizen)
14. [Privacy Hardening](#14-privacy-hardening)
15. [Unified Module Architecture](#15-unified-module-architecture)
16. [Dependency Matrix](#16-dependency-matrix)
17. [Gotchas & Operational Notes (2025–2026)](#17-gotchas--operational-notes-20252026)
18. [Session Roadmap (28–33)](#18-session-roadmap-2833)

---

## 1. Design Philosophy: Why Not a Library

The v1 extraction pipeline was a **standalone library** — stateless functions that fetch content and return text. That's fine for a script. It's wrong for Octane.

Octane's core loop is: **Search → Extract → Evaluate → Decompose → Search again.** The extraction layer isn't a one-shot function. It's a **daemon-managed task** that:

- **Persists state** across research "Waves" via `ResearchContext`
- **Triggers new tasks** based on what it finds (Paper A cites Paper B → auto-fetch B)
- **Accepts human steering** mid-flight without blocking (HIL terminal)
- **Scores every chunk** with source reliability before OSA touches it
- **Never re-fetches** what it already has (cache as first-class citizen)

The extraction logic itself (scrapetube, youtube-transcript-api, docling, etc.) is **unchanged from v1** — it's production-grade. What changes is the **execution context**: from function → task, from stateless → stateful, from linear → recursive.

```
┌─────────────────────────────────────────────────────────────────┐
│                      OCTANE DAEMON (PID 1)                      │
│                                                                 │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐ │
│  │  HIL Terminal │◄──►│ ResearchContext│◄──►│  ExtractionTask  │ │
│  │  (PTY/Socket) │    │    (State)     │    │  (BaseAgent)     │ │
│  └──────────────┘    └───────┬───────┘    └────────┬─────────┘ │
│                              │                     │            │
│                    ┌─────────▼─────────┐   ┌──────▼──────┐    │
│                    │   HashCache        │   │  OSA Layer   │    │
│                    │ ~/.octane/cache/   │   │  Decomposer  │    │
│                    └───────────────────┘   │  Evaluator   │    │
│                                            │  Synthesizer │    │
│                                            └─────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**One sentence**: The extraction engine is the intake valve of the W16. The Daemon is the crankshaft. The HIL terminal is the throttle. Connect them, and the engine breathes.

---

## 2. Core Data Model (Stateful + Trust-Scored)

Every piece of extracted content carries trust metadata, extraction provenance, and cache identity from birth.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib
import time

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
    INTERRUPTED = "interrupted"     # HIL pause
    SPAWNED_CHILDREN = "spawned"    # Recursive: spawned sub-tasks

# ─── Trust & Reliability ───────────────────────────────────────

SOURCE_RELIABILITY = {
    # Peer-reviewed / institutional
    SourceType.ARXIV: 0.92,
    # Structured, authored content
    SourceType.PDF: 0.75,           # Default for unknown PDFs
    SourceType.EPUB: 0.70,
    # Unverified human content
    SourceType.YOUTUBE: 0.55,
    SourceType.WEB: 0.40,
}

# Modifiers applied on top of base reliability
RELIABILITY_MODIFIERS = {
    "peer_reviewed": +0.10,         # arXiv with journal ref
    "high_citations": +0.08,        # Semantic Scholar citation count > 100
    "auto_generated_captions": -0.10,  # YouTube auto-captions (lower accuracy)
    "manual_captions": +0.05,       # Human-written subtitles
    "known_edu_channel": +0.10,     # Channels like 3Blue1Brown, MIT OCW
    "blog_or_opinion": -0.15,       # Detected opinion content
    "scanned_ocr": -0.08,          # OCR'd PDF (potential errors)
}

# ─── Chunk-Level Model ─────────────────────────────────────────

@dataclass
class ChunkMetadata:
    page: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    section_title: Optional[str] = None
    is_table: bool = False
    is_code: bool = False
    is_junk: bool = False           # Detected filler/ad content

@dataclass
class TextChunk:
    text: str
    index: int
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    reliability_score: float = 0.5  # Inherited from doc + modifiers

# ─── Document-Level Model ──────────────────────────────────────

@dataclass
class ExtractedDocument:
    source_type: SourceType
    source_url: str
    title: str
    authors: list[str] = field(default_factory=list)
    chunks: list[TextChunk] = field(default_factory=list)
    raw_text: str = ""
    language: str = "en"
    extraction_method: str = ""
    content_class: str = "spoken"

    # v2 additions
    reliability_score: float = 0.5
    cache_key: str = ""             # SHA-256 of source_url
    extracted_at: float = field(default_factory=time.time)
    wave: int = 0                   # Which research wave produced this
    parent_doc_id: Optional[str] = None  # If spawned by recursive citation
    cited_by: list[str] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)

    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.cache_key:
            self.cache_key = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
        if self.reliability_score == 0.5:
            self.reliability_score = SOURCE_RELIABILITY.get(self.source_type, 0.5)

# ─── Task & Research State ─────────────────────────────────────

@dataclass
class ExtractionTask:
    """A single unit of work managed by the Daemon."""
    task_id: str
    source: str                     # URL, file path, or search query
    source_type: SourceType
    state: TaskState = TaskState.PENDING
    wave: int = 0
    parent_task_id: Optional[str] = None
    quality: str = "auto"           # "fast" | "deep" | "precise" | "auto"
    max_results: int = 5
    results: list[ExtractedDocument] = field(default_factory=list)
    child_task_ids: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class ResearchContext:
    """
    Persistent state across all Waves of a research session.
    Saved to ~/.octane/research/<session_id>.json
    """
    session_id: str
    query: str                      # Original user query
    current_wave: int = 0
    max_waves: int = 3              # Default recursion depth
    tasks: list[ExtractionTask] = field(default_factory=list)
    documents: list[ExtractedDocument] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)      # Dedup
    seen_cache_keys: set[str] = field(default_factory=set)
    focus_filter: Optional[str] = None   # HIL steering (e.g., "skip music")
    paused: bool = False
    created_at: float = field(default_factory=time.time)

    def is_seen(self, url: str) -> bool:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        return key in self.seen_cache_keys

    def mark_seen(self, doc: ExtractedDocument):
        self.seen_urls.add(doc.source_url)
        self.seen_cache_keys.add(doc.cache_key)
        self.documents.append(doc)
```

---

## 3. Daemon Task Lifecycle & Wave Model

Research isn't a single fetch. It's a series of **Waves** — each wave discovers new sources from the previous wave's findings.

```
Wave 0 (User query)
  │
  ├─ Search YouTube: "attention mechanism transformers"  → 5 videos
  ├─ Search arXiv: "attention mechanism transformers"    → 10 papers
  │
  ▼ OSA Decomposer analyzes Wave 0 results
  │
Wave 1 (Auto-discovered)
  │
  ├─ Paper A cites "Vaswani et al. 2017"               → auto-fetch
  ├─ Paper B references "FlashAttention"                → auto-fetch
  ├─ YouTube video mentions "Yannic Kilcher explanation" → search + fetch
  │
  ▼ OSA Evaluator scores, Synthesizer drafts interim report
  │
Wave 2 (Deeper)
  │
  ├─ FlashAttention paper cites 3 more foundational works → auto-fetch
  │
  ▼ HIL: User says "enough depth, synthesize"
  │
  DONE → ResearchSynthesizer produces final output
```

### Task Executor (BaseAgent Integration)

```python
import asyncio
import uuid
from typing import Callable

class ExtractionAgent:
    """
    BaseAgent-compatible task executor for the Octane Daemon.
    Wraps extraction logic as a schedulable, interruptible task.
    """

    def __init__(self, daemon_state, cache, hil_channel):
        self.daemon = daemon_state
        self.cache = cache           # HashCache instance
        self.hil = hil_channel       # HIL terminal channel

    async def run_research(self, query: str, source_types: list[SourceType],
                           max_waves: int = 3, max_results: int = 5) -> ResearchContext:
        ctx = ResearchContext(
            session_id=str(uuid.uuid4())[:8],
            query=query,
            max_waves=max_waves,
        )

        for wave in range(max_waves):
            ctx.current_wave = wave

            # Check HIL for pause/steer
            if ctx.paused:
                await self._wait_for_resume(ctx)
            if self.hil.has_input():
                self._apply_hil_steering(ctx)

            # Generate tasks for this wave
            if wave == 0:
                tasks = self._create_initial_tasks(query, source_types, max_results)
            else:
                tasks = self._create_recursive_tasks(ctx)

            if not tasks:
                break  # Nothing new to discover

            # Execute tasks concurrently (respecting rate limits)
            results = await self._execute_wave(tasks, ctx)

            # Score and filter
            for doc in results:
                if ctx.is_seen(doc.source_url):
                    continue
                doc.wave = wave
                self._apply_trust_score(doc)
                self._apply_junk_filter(doc)
                ctx.mark_seen(doc)

            # Feed to OSA Decomposer for recursive discovery
            if wave < max_waves - 1:
                new_sources = await self._decompose_for_recursion(ctx)
                ctx.tasks.extend(new_sources)

        return ctx

    async def _execute_wave(self, tasks: list[ExtractionTask],
                            ctx: ResearchContext) -> list[ExtractedDocument]:
        """Execute all tasks in a wave with concurrency control."""
        semaphore = asyncio.Semaphore(5)
        results = []

        async def run_task(task: ExtractionTask):
            async with semaphore:
                # Cache check FIRST — never re-fetch
                cached = self.cache.get(task.source)
                if cached:
                    task.state = TaskState.COMPLETE
                    task.results = [cached]
                    return cached

                task.state = TaskState.EXTRACTING
                try:
                    docs = await self._extract(task)
                    task.state = TaskState.COMPLETE
                    task.results = docs
                    # Cache results
                    for doc in docs:
                        self.cache.put(doc)
                    return docs
                except Exception as e:
                    task.state = TaskState.FAILED
                    task.error = str(e)
                    return []

        gather_results = await asyncio.gather(
            *[run_task(t) for t in tasks], return_exceptions=True
        )
        for r in gather_results:
            if isinstance(r, list):
                results.extend(r)
            elif isinstance(r, ExtractedDocument):
                results.append(r)

        return results

    async def _extract(self, task: ExtractionTask) -> list[ExtractedDocument]:
        """Route to the correct extractor based on source type."""
        match task.source_type:
            case SourceType.YOUTUBE:
                return await self._extract_youtube(task)
            case SourceType.ARXIV:
                return await self._extract_arxiv(task)
            case SourceType.PDF:
                return await self._extract_pdf(task)
            case SourceType.EPUB:
                return await self._extract_epub(task)

    def _apply_hil_steering(self, ctx: ResearchContext):
        """Read HIL channel and update ResearchContext."""
        msg = self.hil.read()
        if not msg:
            return

        lower = msg.lower().strip()
        if lower in ("pause", "stop", "wait"):
            ctx.paused = True
        elif lower in ("resume", "go", "continue"):
            ctx.paused = False
        elif lower.startswith("focus "):
            ctx.focus_filter = lower[6:]  # e.g., "focus interviews"
        elif lower.startswith("skip "):
            ctx.focus_filter = f"NOT:{lower[5:]}"  # e.g., "skip music" → "NOT:music"
        elif lower in ("enough", "synthesize", "done"):
            ctx.max_waves = ctx.current_wave  # Stop after current wave
```

---

## 4. Human-in-the-Loop Terminal (HIL)

The killer differentiator. Users steer the research monster without killing it.

### Architecture: PTY + Unix Socket

Do **not** use `input()` — it blocks the event loop. Spawn a separate pseudo-terminal attached to the Daemon via Unix socket.

```python
import asyncio
import os
import socket
import threading
from pathlib import Path

HIL_SOCKET = Path.home() / ".octane" / "hil_socket"

class HILChannel:
    """
    Non-blocking Human-in-the-Loop channel.

    The Daemon reads from this. A separate CLI process (octane hil)
    connects and writes user commands.
    """

    def __init__(self):
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._server: socket.socket | None = None
        self._running = False

    def start(self):
        """Start the Unix socket listener in a background thread."""
        HIL_SOCKET.parent.mkdir(parents=True, exist_ok=True)
        if HIL_SOCKET.exists():
            HIL_SOCKET.unlink()

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(HIL_SOCKET))
        self._server.listen(1)
        self._server.settimeout(0.5)  # Non-blocking accept
        self._running = True

        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

    def _listen_loop(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
                data = conn.recv(4096).decode("utf-8").strip()
                if data:
                    with self._lock:
                        self._buffer.append(data)
                conn.close()
            except socket.timeout:
                continue

    def has_input(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    def read(self) -> str | None:
        with self._lock:
            return self._buffer.pop(0) if self._buffer else None

    def read_all(self) -> list[str]:
        with self._lock:
            msgs = list(self._buffer)
            self._buffer.clear()
            return msgs

    def stop(self):
        self._running = False
        if self._server:
            self._server.close()
        if HIL_SOCKET.exists():
            HIL_SOCKET.unlink()
```

### CLI Client (`octane hil`)

```python
#!/usr/bin/env python3
"""octane hil — connect to the running Daemon's HIL channel."""

import socket
import sys
from pathlib import Path

HIL_SOCKET = Path.home() / ".octane" / "hil_socket"

def send_command(cmd: str):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(HIL_SOCKET))
    sock.send(cmd.encode("utf-8"))
    sock.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_command(" ".join(sys.argv[1:]))
    else:
        # Interactive mode
        print("Octane HIL Terminal — type commands, Ctrl+C to exit")
        print("  Commands: pause | resume | focus <topic> | skip <type> | enough")
        while True:
            try:
                cmd = input("octane> ")
                if cmd.strip():
                    send_command(cmd.strip())
            except (KeyboardInterrupt, EOFError):
                break
```

### UX Contract

The HIL terminal does NOT show the full progress log (too noisy). It shows **actionable prompts** only. The Daemon pushes status summaries to a separate log stream:

```
octane> [Wave 0] Found 8 videos, 12 papers. 2 videos classified as music.
octane> [Wave 0] Extracting... 14/20 complete.
octane> skip music
octane> [Applied] Skipping music content in future waves.
octane> [Wave 1] Auto-discovered 6 cited papers. Fetching...
octane> focus flash attention
octane> [Applied] Prioritizing "flash attention" related content.
octane> enough
octane> [Synthesizing] 31 documents, 847 chunks. Running ResearchSynthesizer...
```

---

## 5. YouTube Search & Transcript Extraction

### 5.1 Search: `scrapetube` (primary) + `yt-dlp` (fallback)

**`scrapetube` v2.6.0** — lightest, fastest. Hits YouTube's InnerTube API directly. Only dependency: `requests`.

```python
import scrapetube

def search_youtube(query: str, limit: int = 5) -> list[dict]:
    videos = scrapetube.get_search(query, limit=limit, sort_by="relevance")
    results = []
    for v in videos:
        results.append({
            "video_id": v["videoId"],
            "title": v["title"]["runs"][0]["text"],
            "duration": v.get("lengthText", {}).get("simpleText", "N/A"),
            "views": v.get("viewCountText", {}).get("simpleText", "N/A"),
            "channel": v.get("ownerText", {}).get("runs", [{}])[0].get("text", ""),
            "url": f"https://www.youtube.com/watch?v={v['videoId']}",
        })
    return results
```

**`yt-dlp`** fallback (100K+ GitHub stars, weekly releases, most resilient to YouTube changes):

```python
import yt_dlp

def search_youtube_ytdlp(query: str, limit: int = 5) -> list[dict]:
    ydl_opts = {"quiet": True, "extract_flat": True, "default_search": f"ytsearch{limit}"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        return [
            {"video_id": e["id"], "title": e.get("title", ""), "url": e["url"]}
            for e in info.get("entries", [])
        ]
```

| Library | Mechanism | Speed | Maintained | Proxy support |
|---------|-----------|-------|-----------|---------------|
| **scrapetube** | InnerTube API POST | Fast | Active (v2.6) | Built-in |
| **yt-dlp** | InnerTube API | Slower (per-video) | Very active | Built-in |
| youtube-search-python | InnerTube API | Fast | **Archived** (June 2022) | No |
| Playwright | Browser render | Slowest | Framework only | Yes |

### 5.2 Transcript Extraction: Three-Tier Fallback

#### Tier 1: `youtube-transcript-api` v1.2.4 (fastest, lightest)

Single InnerTube POST → caption track URLs → timestamped transcript XML. One round-trip.

```python
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked
)

def get_transcript_fast(video_id: str, lang: str = "en") -> list[dict] | None:
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=[lang, "en"])
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None      # No captions → need Whisper
    except RequestBlocked:
        return "BLOCKED"  # Fall to yt-dlp
```

**Breaking change**: v1.0.0 (March 2025) switched to instance-based API. v1.2.0 **removed deprecated static methods**. Old `YouTubeTranscriptApi.get_transcript()` no longer exists.

#### Tier 2: `yt-dlp` subtitle extraction

```python
def get_transcript_ytdlp(video_id: str, lang: str = "en") -> list[dict] | None:
    opts = {
        "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": [lang], "subtitlesformat": "json3", "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
        return info.get("subtitles", {}).get(lang) or info.get("automatic_captions", {}).get(lang)
```

**Gotcha**: Auto-generated VTT has duplicate overlapping lines — needs dedup.

#### Tier 3: `mlx-whisper` local transcription

On Apple Silicon, `mlx-whisper` is 2–7× faster than all alternatives — leverages unified memory via MLX.

```python
import mlx_whisper

def transcribe_audio(audio_path: str) -> dict:
    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    )

def download_audio(video_id: str, output_dir: str = "/tmp/audio") -> str:
    opts = {
        "format": "m4a/bestaudio/best",
        "paths": {"home": output_dir},
        "outtmpl": {"default": "%(id)s.%(ext)s"},
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://youtube.com/watch?v={video_id}"])
    return f"{output_dir}/{video_id}.m4a"
```

**Whisper model comparison on Apple Silicon**:

| Model | Params | WER | Speed vs large | RAM | Pick |
|-------|--------|-----|----------------|-----|------|
| tiny | 39M | ~7.6% | ~10× | ~273 MB | Draft only |
| small | 244M | ~3.4% | ~4× | ~852 MB | RAM-limited |
| **turbo** | **809M** | **~2.5%** | **~8×** | **~2.3 GB** | **Default** |
| large-v3 | 1,550M | ~2.4% | 1× | ~3.9 GB | Max accuracy |

**Implementation speed** (M4 Pro 24GB):

| Implementation | Time | GPU | Notes |
|---------------|------|-----|-------|
| **mlx-whisper** | **1.02s** | Metal/MLX | Fastest |
| whisper.cpp (CoreML) | 1.23s | Metal+CoreML | Needs Xcode |
| faster-whisper | 6.96s | CPU only | CTranslate2 no MPS |
| openai-whisper | 5.37s | Buggy MPS | Many ops unimplemented |

On M1 Max 64GB: 12 min audio in ~13 seconds with turbo.

---

## 6. arXiv & Academic Paper Search

### 6.1 `arxiv` Python package (v2.4.1) — Primary

Free, no API key. Returns metadata + helpers for PDF/source download. Rate limit: 1 req/3s.

```python
import arxiv

def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for paper in client.results(search):
        results.append({
            "arxiv_id": paper.entry_id.split("/abs/")[-1],
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.isoformat(),
            "pdf_url": paper.pdf_url,
            "categories": paper.categories,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,  # For trust scoring
        })
    return results

def download_arxiv_pdf(arxiv_id: str, output_dir: str = "./papers") -> str:
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    return str(paper.download_pdf(dirpath=output_dir))
```

**Advanced queries**: `ti:attention AND cat:cs.CL`, `au:vaswani`, `abs:"retrieval augmented generation"`.

### 6.2 Semantic Scholar — Citation Graph & Recommendations

~200M papers across all fields. Free tier: 1 RPS with API key (free to apply). Provides citation counts, influence scores, TLDR abstracts, and — critically — the **citation graph** that powers recursive discovery.

```python
from semanticscholar import SemanticScholar

sch = SemanticScholar()  # Or SemanticScholar(api_key="your-key")

# Search
papers = sch.search_paper("retrieval augmented generation", limit=10)

# Get paper details by arXiv ID
paper = sch.get_paper("ArXiv:2408.09869")

# Citation graph — THE KEY for recursive discovery
refs = sch.get_paper_references("ArXiv:2408.09869")   # What this paper cites
cites = sch.get_paper_citations("ArXiv:2408.09869")   # What cites this paper

# Recommendations (find-more-like-this)
recs = sch.get_recommended_papers("ArXiv:2408.09869", limit=5)
```

### 6.3 Comparison

| Feature | arXiv API | Semantic Scholar |
|---------|-----------|------------------|
| Coverage | 2.4M+ (physics, CS, math) | ~200M (all fields) |
| API key | No | No (recommended) |
| PDF download | Built-in | No (links only) |
| Citation graph | No | **Yes** |
| Recommendations | No | **Yes** |
| TLDR abstracts | No | Yes |
| Best for | Search + download | Discovery + recursion |

**Decision**: Use `arxiv` for search + download. Use Semantic Scholar for citation-based recursive discovery (Section 12).

---

## 7. PDF Text Extraction

### 7.1 Three-Tier Auto-Routing

#### Tier 1: `pymupdf4llm` — Fast Structured Markdown (0.12s/page)

```python
import pymupdf4llm

def extract_pdf_fast(path: str) -> str:
    return pymupdf4llm.to_markdown(path, header=False, footer=False)
```

Best balance of speed + quality. No GPU. Built on MuPDF C engine.

#### Tier 2: `docling` — AI-Powered Deep Understanding (1–5s/page)

IBM's open-source framework. Computer vision models for layout, reading order, tables, formulas, code blocks. MIT license. MLX acceleration on Apple Silicon. Handles arXiv URLs directly.

```python
from docling.document_converter import DocumentConverter

def extract_pdf_deep(path: str) -> tuple[str, list]:
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document
    markdown = doc.export_to_markdown()
    tables = [t.export_to_dataframe(doc=doc) for t in doc.tables]
    return markdown, tables
```

#### Tier 3: `marker-pdf` — Highest Fidelity (11s/page)

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

def extract_pdf_precise(path: str) -> str:
    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    return converter(path).markdown
```

### 7.2 Auto-Router

```python
import pymupdf

def extract_pdf(path: str, quality: str = "auto") -> tuple[str, list[TextChunk]]:
    if quality == "fast":
        text = extract_pdf_fast(path)
    elif quality == "deep":
        text, _ = extract_pdf_deep(path)
    elif quality == "precise":
        text = extract_pdf_precise(path)
    else:
        doc = pymupdf.open(path)
        sample = doc[0].get_text() if doc.page_count > 0 else ""
        has_imgs = any(page.get_images() for page in doc)
        doc.close()

        is_scanned = len(sample.strip()) < 50 and has_imgs
        is_academic = any(kw in sample.lower() for kw in ["abstract", "references", "arxiv", "theorem"])

        if is_scanned or is_academic:
            text, _ = extract_pdf_deep(path)
        else:
            text = extract_pdf_fast(path)

    chunks = chunk_text(text, SourceType.PDF)
    return text, chunks
```

| Tool | Speed | Tables | Formulas | OCR | GPU | Best for |
|------|-------|--------|----------|-----|-----|----------|
| **pymupdf4llm** | 0.12s/pg | Good | Basic | Auto | No | 90% of PDFs |
| **docling** | 1–5s/pg | Excellent | Good | Yes | Optional (MLX) | Academic, complex layout |
| **marker-pdf** | 11s/pg | Excellent | Excellent | Yes | Recommended | Training data |

---

## 8. EPUB & Ebook Extraction

### `ebooklib` + BeautifulSoup (preserves chapter structure)

```python
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def extract_epub(path: str) -> tuple[str, list[TextChunk]]:
    book = epub.read_epub(path)
    title = book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else "Unknown"
    authors = [a[0] for a in book.get_metadata("DC", "creator")]

    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if isinstance(item, epub.EpubNav):
            continue
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        heading = soup.find(["h1", "h2", "h3"])
        section_title = heading.get_text(strip=True) if heading else item.get_name()
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        text = "\n\n".join(paragraphs)
        if text:
            chapters.append({"title": section_title, "text": text})

    all_text = "\n\n".join(ch["text"] for ch in chapters)
    chunks = [
        TextChunk(text=ch["text"], index=i,
                  metadata=ChunkMetadata(section_title=ch["title"], page=i + 1))
        for i, ch in enumerate(chapters)
    ]
    return all_text, chunks
```

**MOBI/AZW3**: Convert with Calibre's `ebook-convert book.mobi book.epub` first, then process with ebooklib.

**Alternative**: `docling` handles EPUB, DOCX, PPTX, HTML, images, audio — Swiss Army knife for anything ebooklib can't handle.

---

## 9. Chunking Strategies (Source-Aware)

### 9.1 Strategy Matrix

| Source | Strategy | Size | Overlap | Boundary Signals |
|--------|----------|------|---------|------------------|
| YouTube | Timestamp-based semantic | 30–60s windows | 5s | Sentence + silence gaps |
| arXiv | Section-aware recursive | 512 tokens | 50 tokens | `## headings`, `\n\n` |
| General PDF | Recursive character | 512 tokens | 50 tokens | `\n\n`, `\n`, `. ` |
| EPUB | Chapter-first, then recursive | 512 tokens | 50 tokens | Chapter boundaries, `<h1>`–`<h3>` |

### 9.2 Implementation

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def chunk_text(text: str, source_type: SourceType = SourceType.PDF) -> list[TextChunk]:
    if source_type in (SourceType.ARXIV, SourceType.PDF):
        headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split)
        header_chunks = header_splitter.split_text(text)
        recursive = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        final = recursive.split_documents(header_chunks)
        return [
            TextChunk(text=c.page_content, index=i,
                      metadata=ChunkMetadata(section_title=c.metadata.get("h1", c.metadata.get("h2", ""))))
            for i, c in enumerate(final)
        ]
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        splits = splitter.split_text(text)
        return [TextChunk(text=s, index=i) for i, s in enumerate(splits)]
```

### 9.3 YouTube: Timestamp-Aware Chunking

```python
def chunk_transcript_by_time(segments: list[dict], target_duration: float = 45.0) -> list[TextChunk]:
    chunks, current_texts, current_start = [], [], segments[0]["start"] if segments else 0

    for seg in segments:
        current_texts.append(seg["text"])
        elapsed = seg["start"] + seg.get("duration", 0) - current_start
        if elapsed >= target_duration:
            chunks.append(TextChunk(
                text=" ".join(current_texts), index=len(chunks),
                metadata=ChunkMetadata(start_time=current_start,
                                       end_time=seg["start"] + seg.get("duration", 0)),
            ))
            current_texts = current_texts[-3:]  # Keep last 3 segments as overlap
            current_start = seg["start"]

    if current_texts:
        chunks.append(TextChunk(text=" ".join(current_texts), index=len(chunks),
                                metadata=ChunkMetadata(start_time=current_start)))
    return chunks
```

### 9.4 Key Research (2025–2026)

- Recursive 512-token splitting placed **first at 69% accuracy** across 50 academic papers (Vecta Feb 2026 benchmark).
- Semantic chunking can gain up to 9% recall (Chroma research) but a NAACL 2025 paper found fixed 200-word chunks matched or beat semantic chunking — compute cost not always justified.
- **Practical default**: Recursive 512-token, 50-token overlap. Section-aware for academic papers. Semantic only if plateau demands it.

---

## 10. Content Classification, Filtering & Junk Detection

### 10.1 Music vs. Spoken (YouTube)

Metadata-only heuristic achieves ~85–90% accuracy:

```python
import re

MUSIC_TITLE_PATTERNS = [
    r"official\s*(music\s*)?video", r"official\s*audio", r"\blyrics?\b",
    r"\bft\.?\s", r"\bfeat\.?\s", r"music\s*video", r"\bremix\b", r"visualizer",
]
MUSIC_CHANNEL_PATTERNS = [r"vevo$", r"\bmusic\b", r"\brecords?\b"]

def classify_video_metadata(metadata: dict) -> tuple[str, float]:
    score = 0.0
    categories = [c.lower() for c in metadata.get("categories", [])]
    if "music" in categories: score += 3.0
    if "education" in categories: score -= 3.0
    title = metadata.get("title", "").lower()
    if any(re.search(p, title) for p in MUSIC_TITLE_PATTERNS): score += 2.0
    channel = metadata.get("channel", "").lower()
    if any(re.search(p, channel) for p in MUSIC_CHANNEL_PATTERNS): score += 2.0
    dur = metadata.get("duration", 0)
    if 120 <= dur <= 330: score += 0.6
    elif dur > 600: score -= 0.5
    if metadata.get("automatic_captions"): score -= 0.5

    label = "music" if score > 0 else "spoken"
    return label, min(abs(score) / 5.0, 1.0)
```

For ambiguous cases: `inaSpeechSegmenter` (CNN-based, won MIREX 2018) segments audio into speech/music/noise.

### 10.2 Junk Detection at Extraction Level

YouTube auto-transcripts are polluted with filler, ads, UI noise. Pre-filter **before** chunking to cut token count ~30% for OSA:

```python
JUNK_PATTERNS = [
    r"subscribe\s*(to\s*my\s*channel)?",
    r"hit\s*the\s*(bell|like)\s*(button|icon)",
    r"link\s*in\s*(the\s*)?description",
    r"welcome\s*(back\s*)?(to\s*(this|my)\s*(channel|video))",
    r"before\s*we\s*(get\s*started|begin).*sponsor",
    r"(use\s*)?code\s*\w+\s*(for|at)\s*checkout",
    r"thanks?\s*(for|to)\s*(watching|viewing)",
    r"don'?t\s*forget\s*to\s*(like|subscribe|share)",
    r"leave\s*a\s*comment\s*(below|down)",
    r"check\s*out\s*my\s*(patreon|merch|store)",
]

def is_junk_segment(text: str, duration: float = 0) -> bool:
    """Returns True if the segment is filler/ad content."""
    lower = text.lower().strip()

    # Too short to contain meaning
    if duration > 0 and duration < 2.0:
        return True
    if len(lower) < 10:
        return True

    # Pattern match
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, lower):
            return True

    return False

def filter_junk_segments(segments: list[dict]) -> list[dict]:
    """Remove junk segments from YouTube transcript."""
    return [
        seg for seg in segments
        if not is_junk_segment(seg.get("text", ""), seg.get("duration", 0))
    ]
```

### 10.3 PDF Content Classification

```python
def classify_pdf(path: str) -> str:
    """Returns: 'academic', 'technical', 'form', 'scanned', 'general'"""
    import pymupdf
    doc = pymupdf.open(path)
    sample = doc[0].get_text() if doc.page_count > 0 else ""
    has_imgs = bool(doc[0].get_images()) if doc.page_count > 0 else False
    doc.close()
    lower = sample.lower()
    if len(sample.strip()) < 50 and has_imgs: return "scanned"
    if any(kw in lower for kw in ["abstract", "arxiv", "theorem", "doi:", "references"]): return "academic"
    if any(kw in lower for kw in ["api", "function", "class ", "import "]): return "technical"
    return "general"
```

### 10.4 HIL-Driven Filtering

When the user says `skip music` or `focus interviews`, the `ResearchContext.focus_filter` is applied at search time:

```python
def apply_focus_filter(results: list[dict], focus: str | None) -> list[dict]:
    if not focus:
        return results

    if focus.startswith("NOT:"):
        exclude = focus[4:].lower()
        return [r for r in results
                if exclude not in r.get("title", "").lower()
                and exclude not in classify_video_metadata(r)[0]]
    else:
        return sorted(results,
                      key=lambda r: (focus.lower() in r.get("title", "").lower()),
                      reverse=True)
```

---

## 11. Trust & Verification Scoring

Extraction gives text. OSA needs **truth**. Every chunk carries a reliability score that the OSA Evaluator uses to weight findings during synthesis.

### 11.1 Scoring Logic

```python
def compute_reliability(doc: ExtractedDocument, semantic_scholar_data: dict | None = None) -> float:
    """
    Compute reliability score for a document.
    Base score from SourceType + modifiers from metadata signals.
    """
    score = SOURCE_RELIABILITY.get(doc.source_type, 0.5)

    # arXiv modifiers
    if doc.source_type == SourceType.ARXIV:
        if doc.extra.get("journal_ref"):
            score += RELIABILITY_MODIFIERS["peer_reviewed"]
        if semantic_scholar_data:
            citations = semantic_scholar_data.get("citationCount", 0)
            if citations > 100:
                score += RELIABILITY_MODIFIERS["high_citations"]
            elif citations > 500:
                score += RELIABILITY_MODIFIERS["high_citations"] * 1.5

    # YouTube modifiers
    elif doc.source_type == SourceType.YOUTUBE:
        if doc.extraction_method == "mlx-whisper":
            score += RELIABILITY_MODIFIERS["auto_generated_captions"]
        elif "manual" in doc.extraction_method:
            score += RELIABILITY_MODIFIERS["manual_captions"]

        channel = doc.extra.get("channel", "").lower()
        EDU_CHANNELS = {"3blue1brown", "mit opencourseware", "computerphile",
                        "two minute papers", "yannic kilcher", "andrej karpathy",
                        "sentdex", "fireship", "the coding train"}
        if channel in EDU_CHANNELS:
            score += RELIABILITY_MODIFIERS["known_edu_channel"]

    # PDF modifiers
    elif doc.source_type == SourceType.PDF:
        if doc.content_class == "scanned":
            score += RELIABILITY_MODIFIERS["scanned_ocr"]

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))
```

### 11.2 Propagation to Chunks

```python
def propagate_trust_to_chunks(doc: ExtractedDocument):
    """Push document-level reliability to each chunk, with per-chunk modifiers."""
    for chunk in doc.chunks:
        chunk.reliability_score = doc.reliability_score

        # Per-chunk penalty for junk
        if chunk.metadata.is_junk:
            chunk.reliability_score *= 0.1

        # Tables in academic papers are higher confidence than narrative
        if chunk.metadata.is_table and doc.source_type == SourceType.ARXIV:
            chunk.reliability_score = min(1.0, chunk.reliability_score + 0.05)
```

---

## 12. Recursive Citation Discovery

This is what makes the engine **self-sustaining**. Paper A's results spawn tasks for Paper B, C, D.

### 12.1 Reference Extraction from PDFs

After extracting a paper, parse the references section for arXiv IDs, DOIs, and titles:

```python
import re

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,]+")

def extract_references(text: str) -> list[dict]:
    """
    Extract cited paper identifiers from the references section.
    Works on Markdown output from docling/pymupdf4llm.
    """
    # Find references section
    refs_start = None
    for marker in ["## References", "## Bibliography", "# References"]:
        idx = text.find(marker)
        if idx != -1:
            refs_start = idx
            break
    if refs_start is None:
        return []

    refs_text = text[refs_start:]

    # Extract arXiv IDs
    arxiv_ids = list(set(ARXIV_ID_PATTERN.findall(refs_text)))

    # Extract DOIs
    dois = list(set(DOI_PATTERN.findall(refs_text)))

    results = []
    for aid, version in arxiv_ids:
        results.append({"type": "arxiv", "id": aid, "version": version or ""})
    for doi in dois:
        results.append({"type": "doi", "id": doi.rstrip(".")})

    return results
```

### 12.2 Citation Scoring (Which references are worth chasing?)

Not all 50 references in a paper are worth fetching. Use Semantic Scholar to rank by influence:

```python
async def rank_citations(references: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rank extracted references by citation count via Semantic Scholar.
    Returns top_k most-cited papers worth fetching in the next Wave.
    """
    sch = SemanticScholar()
    scored = []

    for ref in references:
        try:
            if ref["type"] == "arxiv":
                paper = sch.get_paper(f"ArXiv:{ref['id']}")
            elif ref["type"] == "doi":
                paper = sch.get_paper(f"DOI:{ref['id']}")
            else:
                continue

            if paper and paper.citationCount:
                scored.append({
                    "id": ref["id"],
                    "type": ref["type"],
                    "title": paper.title,
                    "citations": paper.citationCount,
                    "pdf_url": f"https://arxiv.org/pdf/{ref['id']}" if ref["type"] == "arxiv" else None,
                })
        except Exception:
            continue

    # Sort by citations descending, take top_k
    scored.sort(key=lambda x: x["citations"], reverse=True)
    return scored[:top_k]
```

### 12.3 Spawning Recursive Tasks

```python
async def _decompose_for_recursion(self, ctx: ResearchContext) -> list[ExtractionTask]:
    """
    OSA Decomposer role: analyze current wave's results,
    spawn new tasks for high-value undiscovered sources.
    """
    new_tasks = []

    for doc in ctx.documents:
        if doc.wave != ctx.current_wave:
            continue  # Only process current wave's findings
        if doc.source_type != SourceType.ARXIV:
            continue  # Only recurse on academic papers (for now)

        # Extract references
        refs = extract_references(doc.raw_text)
        if not refs:
            continue

        # Rank by citation importance
        ranked = await rank_citations(refs, top_k=3)

        for ref in ranked:
            url = ref.get("pdf_url", "")
            if not url or ctx.is_seen(url):
                continue

            task = ExtractionTask(
                task_id=str(uuid.uuid4())[:8],
                source=url,
                source_type=SourceType.ARXIV,
                wave=ctx.current_wave + 1,
                parent_task_id=doc.cache_key,
                quality="deep",
                max_results=1,
            )
            new_tasks.append(task)

    return new_tasks
```

---

## 13. Cache Architecture (First-Class Citizen)

Cache is not an optimization. It's **infrastructure**. Check cache before ANY network request.

```python
import json
import hashlib
from pathlib import Path

CACHE_DIR = Path.home() / ".octane" / "cache"

class HashCache:
    """
    Content-addressable cache for extracted documents and raw files.

    Structure:
        ~/.octane/cache/
        ├── docs/           # Serialized ExtractedDocument JSON
        │   ├── a1b2c3d4.json
        │   └── ...
        ├── pdfs/           # Downloaded PDF files
        │   ├── a1b2c3d4.pdf
        │   └── ...
        ├── audio/          # Downloaded audio files
        │   ├── e5f6g7h8.m4a
        │   └── ...
        └── manifest.json   # URL → cache_key mapping
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.docs_dir = cache_dir / "docs"
        self.pdfs_dir = cache_dir / "pdfs"
        self.audio_dir = cache_dir / "audio"
        self._manifest_path = cache_dir / "manifest.json"

        for d in [self.docs_dir, self.pdfs_dir, self.audio_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    @staticmethod
    def _hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    # ─── Document Cache ────────────────────────────────────

    def has(self, url: str) -> bool:
        key = self._hash(url)
        return (self.docs_dir / f"{key}.json").exists()

    def get(self, url: str) -> ExtractedDocument | None:
        key = self._hash(url)
        path = self.docs_dir / f"{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        # Deserialize back to ExtractedDocument
        return _dict_to_extracted_doc(data)

    def put(self, doc: ExtractedDocument):
        key = doc.cache_key or self._hash(doc.source_url)
        path = self.docs_dir / f"{key}.json"
        path.write_text(json.dumps(_extracted_doc_to_dict(doc), indent=2))
        self._manifest[doc.source_url] = key
        self._save_manifest()

    # ─── File Cache ────────────────────────────────────────

    def has_file(self, url: str, kind: str = "pdfs") -> bool:
        key = self._hash(url)
        d = self.cache_dir / kind
        return any(d.glob(f"{key}.*"))

    def get_file_path(self, url: str, kind: str = "pdfs") -> Path | None:
        key = self._hash(url)
        d = self.cache_dir / kind
        matches = list(d.glob(f"{key}.*"))
        return matches[0] if matches else None

    def put_file(self, url: str, file_path: Path, kind: str = "pdfs") -> Path:
        key = self._hash(url)
        ext = file_path.suffix
        dest = self.cache_dir / kind / f"{key}{ext}"
        if not dest.exists():
            import shutil
            shutil.copy2(file_path, dest)
        return dest

    # ─── Stats ─────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "documents": len(list(self.docs_dir.glob("*.json"))),
            "pdfs": len(list(self.pdfs_dir.glob("*"))),
            "audio": len(list(self.audio_dir.glob("*"))),
            "manifest_entries": len(self._manifest),
        }
```

### Cache-First Extraction Pattern

```python
async def _extract_with_cache(self, task: ExtractionTask) -> list[ExtractedDocument]:
    """ALWAYS check cache before network."""

    # 1. Check document cache
    cached = self.cache.get(task.source)
    if cached:
        return [cached]

    # 2. Check file cache (PDF/audio already downloaded?)
    if task.source_type in (SourceType.PDF, SourceType.ARXIV):
        cached_file = self.cache.get_file_path(task.source, "pdfs")
        if cached_file:
            text, chunks = extract_pdf(str(cached_file), quality=task.quality)
            doc = ExtractedDocument(source_type=task.source_type, source_url=task.source,
                                   title="", raw_text=text, chunks=chunks)
            self.cache.put(doc)
            return [doc]

    # 3. Network fetch (last resort)
    return await self._extract(task)
```

---

## 14. Privacy Hardening

### 14.1 User-Agent Rotation

Randomize per request pool. Never log UA strings in SynapseEvent.

```python
import random

UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36 Edg/131.0",
]

def get_random_ua() -> str:
    return random.choice(UA_POOL)

def get_session_with_ua():
    """Create a requests Session with randomized UA."""
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": get_random_ua()})
    return s
```

### 14.2 Log Scrubbing

```python
# NEVER log these fields in SynapseEvent or daemon logs:
SCRUB_FIELDS = {"user_agent", "proxy_url", "ip_address", "cookie", "session_token"}

def scrub_for_logging(data: dict) -> dict:
    """Remove privacy-sensitive fields before logging."""
    return {k: v for k, v in data.items() if k not in SCRUB_FIELDS}
```

### 14.3 No Telemetry Leaks

- scrapetube: Only sends InnerTube API POSTs. No cookies, no login state. Safe.
- youtube-transcript-api: Sends Innertube POSTs. Supports proxies. Safe.
- yt-dlp: Rotate UA per invocation. Disable `--geo-bypass` logging.
- arxiv package: Only hits `export.arxiv.org`. No auth, no tracking.
- Semantic Scholar: API key is per-developer, not per-user. Do not log it.

---

## 15. Unified Module Architecture

```
octane/
├── daemon/
│   ├── __init__.py
│   ├── daemon.py                  # Main Daemon process (PID 1)
│   ├── hil.py                     # HILChannel (Unix socket PTY)
│   └── task_scheduler.py          # Wave-based task scheduling
│
├── extractors/
│   ├── __init__.py
│   ├── config.py                  # Pydantic BaseSettings
│   ├── models.py                  # ExtractedDocument, TextChunk, ResearchContext, etc.
│   ├── agent.py                   # ExtractionAgent (BaseAgent wrapper)
│   ├── pipeline.py                # Unified extract() entry point
│   ├── chunker.py                 # Source-aware chunking
│   ├── classifier.py              # Content classification + junk detection
│   ├── trust.py                   # Reliability scoring
│   ├── cache.py                   # HashCache (first-class)
│   ├── privacy.py                 # UA rotation, log scrubbing
│   │
│   ├── youtube/
│   │   ├── __init__.py
│   │   ├── search.py              # scrapetube + yt-dlp fallback
│   │   ├── transcript.py          # 3-tier: transcript-api → yt-dlp → whisper
│   │   ├── whisper_local.py       # mlx-whisper
│   │   └── junk_filter.py         # YouTube filler/ad detection
│   │
│   ├── academic/
│   │   ├── __init__.py
│   │   ├── arxiv_search.py        # arxiv package wrapper
│   │   ├── semantic_scholar.py    # S2 API (citations, recs)
│   │   ├── citation_graph.py      # Recursive reference extraction + ranking
│   │   └── paper_download.py      # PDF download + cache
│   │
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── extractor.py           # Auto-routing: pymupdf4llm → docling → marker
│   │   └── table_extractor.py     # Docling table extraction
│   │
│   └── epub/
│       ├── __init__.py
│       └── extractor.py           # ebooklib + BeautifulSoup
│
├── osa/
│   ├── decomposer.py             # Analyzes chunks → emits new SearchTasks
│   ├── evaluator.py              # Uses reliability_score to weight findings
│   └── synthesizer.py            # ResearchSynthesizer (MSR output)
│
└── cli/
    ├── octane_hil.py             # `octane hil` — HIL terminal client
    ├── octane_investigate.py     # `octane investigate <query>`
    └── octane_extract.py         # `octane extract <url/path>`
```

### Unified Entry Point (now Task-aware)

```python
# octane/extractors/pipeline.py

async def extract(
    source: str,
    source_type: SourceType | None = None,
    quality: str = "auto",
    max_results: int = 5,
    recursive: bool = False,        # Enable citation recursion
    max_waves: int = 3,
    cache: HashCache | None = None,
    hil: HILChannel | None = None,
) -> ResearchContext | list[ExtractedDocument]:
    """
    Unified extraction interface.

    If recursive=True, returns a full ResearchContext with multi-wave results.
    If recursive=False, returns a flat list of ExtractedDocument (v1 compat).
    """
    if source_type is None:
        source_type = _detect_source_type(source)

    cache = cache or HashCache()
    hil = hil or HILChannel()  # No-op if not started

    if recursive:
        agent = ExtractionAgent(daemon_state=None, cache=cache, hil_channel=hil)
        return await agent.run_research(
            query=source,
            source_types=[source_type],
            max_waves=max_waves,
            max_results=max_results,
        )
    else:
        # Flat mode (v1 compat)
        task = ExtractionTask(
            task_id="single", source=source, source_type=source_type,
            quality=quality, max_results=max_results,
        )
        agent = ExtractionAgent(daemon_state=None, cache=cache, hil_channel=hil)
        return await agent._extract(task)
```

---

## 16. Dependency Matrix

### Core (always installed)

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `scrapetube` | 2.6.0 | YouTube search | ~50KB |
| `youtube-transcript-api` | ≥1.2.4 | Caption extraction | ~100KB |
| `yt-dlp` | ≥2026.3 | Audio/subtitle fallback | ~50MB+ |
| `arxiv` | 2.4.1 | arXiv API | Tiny |
| `pymupdf4llm` | latest | Fast PDF→Markdown | ~30MB |
| `ebooklib` | 0.20 | EPUB reading | Tiny |
| `beautifulsoup4` | latest | HTML parsing | Tiny |
| `tenacity` | latest | Retry logic | Tiny |

### Optional (per-feature)

| Package | Purpose | When needed |
|---------|---------|-------------|
| `mlx-whisper` | Local Whisper transcription | Videos without captions |
| `docling` ≥2.70 | Deep PDF extraction | Academic papers, complex layout |
| `marker-pdf` | Highest-fidelity PDF | Training data generation |
| `semanticscholar` 0.11 | Citation graph + recs | Recursive academic discovery |
| `inaSpeechSegmenter` | Audio music/speech detection | Ambiguous YouTube content |
| `langchain-text-splitters` | Section-aware chunking | Academic paper chunking |

### System

| Tool | Install |
|------|---------|
| `ffmpeg` | `brew install ffmpeg` |
| `deno` | `brew install deno` (required by yt-dlp 2025+) |
| `yt-dlp-ejs` | `pip install yt-dlp-ejs` |

---

## 17. Gotchas & Operational Notes (2025–2026)

### YouTube

1. **Cloud IP blocking** — #1 issue. YouTube blocks AWS/GCP/Azure. Works on your M5 Pro at home, fails on any cloud VM. Residential proxies required for cloud.
2. **yt-dlp requires Deno** — Late 2025 change. YouTube JS challenges. Install `yt-dlp-ejs`.
3. **youtube-transcript-api v1.2.0 broke the old API** — `get_transcript()` is gone. Use `YouTubeTranscriptApi().fetch()`.
4. **POT tokens** growing — Some videos require Proof of Origin Tokens. `PoTokenRequired` exception.
5. **Rate limit ~250 requests** on residential IPs.

### PDF

6. **PyMuPDF is now AGPL + commercial** — Check if AGPL affects distribution.
7. **docling dropped Python 3.9** in v2.70. Requires 3.10+.
8. **Scanned PDFs** return near-zero text from pymupdf4llm — auto-router detects and routes to docling OCR.

### arXiv

9. **Soft rate limit ~1 req/3s.** The `arxiv` package handles this.
10. **Max 30,000 results per query**, in slices of 2,000.

### General

11. **Cache everything.** Network is the bottleneck, not extraction.
12. **Unified output format** — every extractor produces `ExtractedDocument`. This is what makes it composable with OSA.

---

## 18. Session Roadmap (28–33)

| Session | Focus | Deliverable |
|---------|-------|-------------|
| **28** | **Daemon Integration** | Wrap `extract()` in `ExtractionAgent` (BaseAgent). `ResearchContext` state holding. Wire into existing Daemon process. |
| **29** | **Trust Scoring + Junk Filter** | `reliability_score` on `ExtractedDocument` and `TextChunk`. YouTube junk detection. Source modifiers. |
| **30** | **HIL Terminal** | `HILChannel` Unix socket. `octane hil` CLI client. Non-blocking steer commands (pause/resume/focus/skip/enough). |
| **31** | **Recursive Citation Graph** | Reference extraction from PDFs. Semantic Scholar citation ranking. Auto-spawn Wave N+1 tasks. |
| **32** | **HashCache + Dedup** | `~/.octane/cache/` for docs, PDFs, audio. Cache-first extraction pattern. Manifest-based dedup across sessions. |
| **33** | **Privacy Hardening + Polish** | UA rotation pool. Log scrubbing. No telemetry leaks. Integration test: full `octane investigate` end-to-end. |

---

## eyeso Integration (Updated)

```
# eyeso: recursive research pipeline

investigate "attention mechanisms in transformers"
  waves=3
  sources=[arxiv, youtube]
  quality="deep"

  wave 0:
    search arxiv "attention mechanism transformer" limit=10
    search youtube "attention mechanism explained" limit=5
    → extract → classify → score_trust → chunk → store

  wave 1..N:
    discover_citations from wave(N-1)
    → rank_by_influence top=3
    → extract → score_trust → chunk → store

  on_hil "enough":
    → synthesize collection="attention-research"
    → report format=markdown confidence_weighted=true
```

The engine is the same. The execution context is the difference. Function → Task. Stateless → Stateful. Linear → Recursive.

Capiche.