"""Structured storage layer — Session 18A.

Provides Python wrappers around the normalised Postgres schema defined in
``octane/tools/schema.sql``:

    ProjectStore    — CRUD for the ``projects`` table
    WebPageStore    — dedup-aware insert + search for ``web_pages``
    ArtifactStore   — register + query ``generated_artifacts``
    FileIndexer     — index local files/folders → ``user_files``
    EmbeddingEngine — embed text chunks + semantic search via pgVector

All classes accept an injected ``PgClient`` for testability and degrade
gracefully when Postgres is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="structured_store")

# ── Supported file extensions for FileIndexer ────────────────────────────────
_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml"}
_CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
                    ".sh", ".sql", ".html", ".css"}
_DOC_EXTENSIONS  = {".pdf", ".docx", ".doc"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

INDEXABLE_EXTENSIONS = _TEXT_EXTENSIONS | _CODE_EXTENSIONS | _DOC_EXTENSIONS | _IMAGE_EXTENSIONS

# Embedding dimension — must match the model used in EmbeddingEngine
EMBEDDING_DIM = 384

# Chunk size (tokens ≈ chars/4, we use words as a proxy)
CHUNK_WORDS = 200


# ══════════════════════════════════════════════════════════════════════════════
# ProjectStore
# ══════════════════════════════════════════════════════════════════════════════

class ProjectStore:
    """CRUD for the ``projects`` table."""

    def __init__(self, pg) -> None:
        self._pg = pg  # PgClient instance

    async def create(self, name: str, description: str = "") -> dict | None:
        """Create a project. Returns the new row or None on failure."""
        row = await self._pg.fetchrow(
            """
            INSERT INTO projects (name, description)
            VALUES ($1, $2)
            ON CONFLICT (name) DO UPDATE
                SET description = EXCLUDED.description,
                    updated_at  = NOW()
            RETURNING *
            """,
            name, description,
        )
        if row:
            logger.info("project_created", name=name, id=row.get("id"))
        return row

    async def get(self, name: str) -> dict | None:
        """Fetch a project by name."""
        return await self._pg.fetchrow(
            "SELECT * FROM projects WHERE name = $1", name
        )

    async def get_by_id(self, project_id: int) -> dict | None:
        """Fetch a project by primary key."""
        return await self._pg.fetchrow(
            "SELECT * FROM projects WHERE id = $1", project_id
        )

    async def list(self, include_archived: bool = False) -> list[dict]:
        """List all projects, newest first."""
        if include_archived:
            return await self._pg.fetch(
                "SELECT * FROM projects ORDER BY created_at DESC"
            )
        return await self._pg.fetch(
            "SELECT * FROM projects WHERE status != 'archived' ORDER BY created_at DESC"
        )

    async def archive(self, name: str) -> bool:
        """Soft-delete a project by setting status='archived'."""
        return await self._pg.execute(
            "UPDATE projects SET status='archived', updated_at=NOW() WHERE name=$1", name
        )

    async def delete(self, name: str) -> bool:
        """Hard-delete project and all cascade-linked rows."""
        return await self._pg.execute(
            "DELETE FROM projects WHERE name=$1", name
        )


# ══════════════════════════════════════════════════════════════════════════════
# WebPageStore
# ══════════════════════════════════════════════════════════════════════════════

def _url_hash(url: str) -> str:
    """SHA-256 of the normalised URL (lowercase, stripped trailing slash)."""
    normalised = url.lower().rstrip("/")
    return hashlib.sha256(normalised.encode()).hexdigest()


class WebPageStore:
    """Dedup-aware storage for fetched web pages."""

    def __init__(self, pg) -> None:
        self._pg = pg

    async def store(
        self,
        url: str,
        content: str,
        title: str = "",
        project_id: int | None = None,
        fetch_status: str = "ok",
    ) -> dict | None:
        """Insert a page, or update it if the URL was already seen.

        Returns the upserted row dict (with ``id`` field).
        Returns None if Postgres unavailable.
        """
        h = _url_hash(url)
        wc = len(content.split())
        row = await self._pg.fetchrow(
            """
            INSERT INTO web_pages
                (url, url_hash, title, content, word_count, fetch_status, project_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (url_hash) DO UPDATE
                SET content      = EXCLUDED.content,
                    title        = EXCLUDED.title,
                    word_count   = EXCLUDED.word_count,
                    fetch_status = EXCLUDED.fetch_status,
                    fetched_at   = NOW()
            RETURNING *
            """,
            url, h, title, content, wc, fetch_status, project_id,
        )
        if row:
            logger.info("web_page_stored", url=url[:80], words=wc)
        return row

    async def seen(self, url: str) -> bool:
        """Return True if this URL has already been stored (dedup check)."""
        h = _url_hash(url)
        row = await self._pg.fetchrow(
            "SELECT id FROM web_pages WHERE url_hash = $1", h
        )
        return row is not None

    async def get(self, url: str) -> dict | None:
        """Retrieve a stored page by URL."""
        h = _url_hash(url)
        return await self._pg.fetchrow(
            "SELECT * FROM web_pages WHERE url_hash = $1", h
        )

    async def recent(self, project_id: int | None = None, limit: int = 20) -> list[dict]:
        """Return the most recently fetched pages."""
        if project_id is not None:
            return await self._pg.fetch(
                "SELECT * FROM web_pages WHERE project_id=$1 ORDER BY fetched_at DESC LIMIT $2",
                project_id, limit,
            )
        return await self._pg.fetch(
            "SELECT * FROM web_pages ORDER BY fetched_at DESC LIMIT $1", limit
        )

    async def count(self, project_id: int | None = None) -> int:
        """Return total number of stored pages (optionally filtered by project)."""
        if project_id is not None:
            val = await self._pg.fetchval(
                "SELECT COUNT(*) FROM web_pages WHERE project_id=$1", project_id
            )
        else:
            val = await self._pg.fetchval("SELECT COUNT(*) FROM web_pages")
        return int(val or 0)


# ══════════════════════════════════════════════════════════════════════════════
# ArtifactStore
# ══════════════════════════════════════════════════════════════════════════════

class ArtifactStore:
    """Register and query generated artifacts (code, reports, charts)."""

    def __init__(self, pg) -> None:
        self._pg = pg

    async def register(
        self,
        content: str,
        artifact_type: str = "code",
        filename: str = "",
        language: str = "",
        description: str = "",
        session_id: str = "",
        project_id: int | None = None,
    ) -> dict | None:
        """Persist a generated artifact. Returns the inserted row."""
        row = await self._pg.fetchrow(
            """
            INSERT INTO generated_artifacts
                (project_id, session_id, artifact_type, filename,
                 content, language, description)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            project_id, session_id, artifact_type, filename,
            content, language, description,
        )
        if row:
            logger.info(
                "artifact_registered",
                type=artifact_type,
                filename=filename,
                id=row.get("id"),
            )
        return row

    async def query(
        self,
        session_id: str | None = None,
        artifact_type: str | None = None,
        project_id: int | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Fetch artifacts with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if session_id is not None:
            params.append(session_id)
            clauses.append(f"session_id = ${len(params)}")
        if artifact_type is not None:
            params.append(artifact_type)
            clauses.append(f"artifact_type = ${len(params)}")
        if project_id is not None:
            params.append(project_id)
            clauses.append(f"project_id = ${len(params)}")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        sql = f"SELECT * FROM generated_artifacts {where} ORDER BY created_at DESC LIMIT ${len(params)}"
        return await self._pg.fetch(sql, *params)

    async def get(self, artifact_id: int) -> dict | None:
        return await self._pg.fetchrow(
            "SELECT * FROM generated_artifacts WHERE id=$1", artifact_id
        )


# ══════════════════════════════════════════════════════════════════════════════
# FileIndexer
# ══════════════════════════════════════════════════════════════════════════════

def _file_hash(path: Path) -> str:
    """SHA-256 of file bytes (for change detection)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        pass
    return h.hexdigest()


def _extract_text(path: Path) -> str:
    """Extract plain text from a file. Returns '' on failure."""
    ext = path.suffix.lower()

    if ext in _TEXT_EXTENSIONS | _CODE_EXTENSIONS:
        try:
            return path.read_text(errors="replace")
        except OSError:
            return ""

    if ext == ".pdf":
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""

    if ext in {".docx", ".doc"}:
        try:
            import docx  # type: ignore
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    if ext in _IMAGE_EXTENSIONS:
        # No OCR — store path only, empty content
        return ""

    return ""


class FileIndexer:
    """Index local files and folders into the ``user_files`` table."""

    def __init__(self, pg, project_id: int | None = None) -> None:
        self._pg = pg
        self._project_id = project_id

    async def index_file(self, path: str | Path) -> dict | None:
        """Index a single file. Skips if unchanged (same hash)."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            logger.warning("file_not_found", path=str(p))
            return None

        ext = p.suffix.lower()
        if ext not in INDEXABLE_EXTENSIONS:
            logger.debug("file_skipped_unsupported_ext", path=str(p), ext=ext)
            return None

        fh = _file_hash(p)

        # Dedup: skip if already indexed with same hash
        existing = await self._pg.fetchrow(
            "SELECT id, file_hash FROM user_files WHERE path=$1", str(p)
        )
        if existing and existing.get("file_hash") == fh:
            logger.debug("file_unchanged", path=str(p))
            return dict(existing)

        content = _extract_text(p)
        wc = len(content.split())

        row = await self._pg.fetchrow(
            """
            INSERT INTO user_files
                (project_id, path, filename, extension, content, word_count, file_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (path) DO UPDATE
                SET content     = EXCLUDED.content,
                    word_count  = EXCLUDED.word_count,
                    file_hash   = EXCLUDED.file_hash,
                    indexed_at  = NOW()
            RETURNING *
            """,
            self._project_id, str(p), p.name, ext, content, wc, fh,
        )
        if row:
            logger.info("file_indexed", path=str(p), words=wc)
        return row

    async def index_folder(
        self,
        folder: str | Path,
        recursive: bool = True,
        max_files: int = 500,
    ) -> list[dict]:
        """Index all supported files in *folder*. Returns list of indexed rows."""
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            logger.warning("folder_not_found", path=str(root))
            return []

        pattern = "**/*" if recursive else "*"
        files = [
            f for f in root.glob(pattern)
            if f.is_file() and f.suffix.lower() in INDEXABLE_EXTENSIONS
        ][:max_files]

        results: list[dict] = []
        for f in files:
            row = await self.index_file(f)
            if row:
                results.append(row)

        logger.info("folder_indexed", path=str(root), files=len(results))
        return results

    async def reindex(self, path: str | Path) -> dict | None:
        """Force re-index regardless of hash."""
        p = Path(path).expanduser().resolve()
        # Delete existing record to force re-insert
        await self._pg.execute("DELETE FROM user_files WHERE path=$1", str(p))
        return await self.index_file(p)

    async def stats(self, project_id: int | None = None) -> dict:
        """Return indexing stats: total files, total words, breakdown by extension."""
        pid = project_id or self._project_id
        if pid is not None:
            rows = await self._pg.fetch(
                "SELECT extension, COUNT(*) as n, SUM(word_count) as words "
                "FROM user_files WHERE project_id=$1 GROUP BY extension ORDER BY n DESC",
                pid,
            )
            total_files = await self._pg.fetchval(
                "SELECT COUNT(*) FROM user_files WHERE project_id=$1", pid
            )
            total_words = await self._pg.fetchval(
                "SELECT COALESCE(SUM(word_count),0) FROM user_files WHERE project_id=$1", pid
            )
        else:
            rows = await self._pg.fetch(
                "SELECT extension, COUNT(*) as n, SUM(word_count) as words "
                "FROM user_files GROUP BY extension ORDER BY n DESC"
            )
            total_files = await self._pg.fetchval("SELECT COUNT(*) FROM user_files")
            total_words = await self._pg.fetchval(
                "SELECT COALESCE(SUM(word_count),0) FROM user_files"
            )
        return {
            "total_files": int(total_files or 0),
            "total_words": int(total_words or 0),
            "by_extension": [dict(r) for r in (rows or [])],
        }


# ══════════════════════════════════════════════════════════════════════════════
# EmbeddingEngine
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    overlap = chunk_words // 5  # 20% overlap
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_words])
        chunks.append(chunk)
        i += chunk_words - overlap
    return chunks


class EmbeddingEngine:
    """Embed text chunks and perform semantic search via pgVector.

    Uses the ``sentence-transformers`` library if available; falls back to
    zero-vectors when offline (tests still pass — just no semantic ranking).
    """

    def __init__(self, pg, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._pg = pg
        self._model_name = model_name
        self._model = None  # lazy-loaded

    def _load_model(self):
        """Lazily load the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self._model_name)
                logger.info("embedding_model_loaded", model=self._model_name)
            except Exception as exc:
                logger.warning(
                    "embedding_model_unavailable",
                    model=self._model_name,
                    error=str(exc),
                )
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts. Falls back to zero-vectors."""
        model = self._load_model()
        if model is None:
            return [[0.0] * EMBEDDING_DIM for _ in texts]
        try:
            vecs = model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        except Exception as exc:
            logger.warning("embed_failed", error=str(exc))
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    async def embed_and_store(
        self,
        source_type: str,
        source_id: int,
        text: str,
        chunk_words: int = CHUNK_WORDS,
    ) -> int:
        """Chunk *text*, embed each chunk, and store in ``embeddings`` table.

        Returns the number of chunks stored.
        """
        if not text.strip():
            return 0

        chunks = _chunk_text(text, chunk_words=chunk_words)
        if not chunks:
            return 0

        vectors = self._embed(chunks)

        # Delete existing embeddings for this source (re-embed on update)
        await self._pg.execute(
            "DELETE FROM embeddings WHERE source_type=$1 AND source_id=$2",
            source_type, source_id,
        )

        stored = 0
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            ok = await self._pg.execute(
                """
                INSERT INTO embeddings (source_type, source_id, chunk_index, chunk_text, embedding)
                VALUES ($1, $2, $3, $4, $5::vector)
                """,
                source_type, source_id, idx, chunk, vec,
            )
            if ok:
                stored += 1

        logger.info(
            "embeddings_stored",
            source_type=source_type,
            source_id=source_id,
            chunks=stored,
        )
        return stored

    async def semantic_search(
        self,
        query: str,
        source_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Find the most semantically similar chunks to *query*.

        Returns a list of dicts with keys:
            source_type, source_id, chunk_index, chunk_text, distance
        """
        vecs = self._embed([query])
        query_vec = vecs[0]

        source_filter = ""
        params: list[Any] = [query_vec, limit]
        if source_type is not None:
            source_filter = "WHERE source_type = $3"
            params.append(source_type)

        rows = await self._pg.fetch(
            f"""
            SELECT source_type, source_id, chunk_index, chunk_text,
                   embedding <=> $1::vector AS distance
            FROM embeddings
            {source_filter}
            ORDER BY distance ASC
            LIMIT $2
            """,
            *params,
        )
        return [dict(r) for r in rows]
