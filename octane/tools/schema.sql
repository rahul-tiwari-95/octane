-- ════════════════════════════════════════════════════════════════════════════
-- Octane Structured Storage Schema  (Session 18A)
-- Applied by PgClient._ensure_schema() on first connect.
-- ════════════════════════════════════════════════════════════════════════════

-- ── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── projects ─────────────────────────────────────────────────────────────────
-- Top-level container; every other table links back via project_id (nullable
-- means "global / no project").
CREATE TABLE IF NOT EXISTS projects (
    id          SERIAL      PRIMARY KEY,
    name        TEXT        NOT NULL UNIQUE,
    description TEXT        NOT NULL DEFAULT '',
    status      TEXT        NOT NULL DEFAULT 'active',   -- active | archived
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── web_pages ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS web_pages (
    id              SERIAL      PRIMARY KEY,
    project_id      INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
    url             TEXT        NOT NULL,
    url_hash        TEXT        NOT NULL UNIQUE,   -- SHA-256 of normalised URL
    title           TEXT        NOT NULL DEFAULT '',
    content         TEXT        NOT NULL DEFAULT '',
    word_count      INTEGER     NOT NULL DEFAULT 0,
    fetch_status    TEXT        NOT NULL DEFAULT 'ok',  -- ok | error | empty
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_web_pages_project  ON web_pages (project_id);
CREATE INDEX IF NOT EXISTS idx_web_pages_hash     ON web_pages (url_hash);
CREATE INDEX IF NOT EXISTS idx_web_pages_fetched  ON web_pages (fetched_at DESC);

-- ── user_files ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_files (
    id              SERIAL      PRIMARY KEY,
    project_id      INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
    path            TEXT        NOT NULL UNIQUE,   -- absolute path on disk
    filename        TEXT        NOT NULL,
    extension       TEXT        NOT NULL DEFAULT '',
    content         TEXT        NOT NULL DEFAULT '',
    word_count      INTEGER     NOT NULL DEFAULT 0,
    file_hash       TEXT        NOT NULL DEFAULT '',  -- SHA-256 of file bytes
    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_user_files_project   ON user_files (project_id);
CREATE INDEX IF NOT EXISTS idx_user_files_path      ON user_files (path);
CREATE INDEX IF NOT EXISTS idx_user_files_ext       ON user_files (extension);

-- ── generated_artifacts ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS generated_artifacts (
    id              SERIAL      PRIMARY KEY,
    project_id      INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
    session_id      TEXT        NOT NULL DEFAULT '',
    artifact_type   TEXT        NOT NULL DEFAULT 'code',  -- code | report | chart | other
    filename        TEXT        NOT NULL DEFAULT '',
    content         TEXT        NOT NULL,
    language        TEXT        NOT NULL DEFAULT '',
    description     TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_project    ON generated_artifacts (project_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session    ON generated_artifacts (session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type       ON generated_artifacts (artifact_type);

-- ── research_findings_v2 ──────────────────────────────────────────────────────
-- Replaces the single-table research_findings from Session 17 with richer schema.
CREATE TABLE IF NOT EXISTS research_findings_v2 (
    id              SERIAL      PRIMARY KEY,
    project_id      INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
    task_id         TEXT        NOT NULL,
    cycle_num       INTEGER     NOT NULL DEFAULT 0,
    topic           TEXT        NOT NULL,
    content         TEXT        NOT NULL,
    agents_used     TEXT[]      NOT NULL DEFAULT '{}',
    sources         TEXT[]      NOT NULL DEFAULT '{}',
    word_count      INTEGER     NOT NULL DEFAULT 0,
    depth           TEXT        NOT NULL DEFAULT 'shallow',  -- shallow|deep|exhaustive
    confidence      REAL        NOT NULL DEFAULT 0.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rfv2_task_id   ON research_findings_v2 (task_id);
CREATE INDEX IF NOT EXISTS idx_rfv2_project   ON research_findings_v2 (project_id);
CREATE INDEX IF NOT EXISTS idx_rfv2_created   ON research_findings_v2 (task_id, created_at DESC);

-- ── portfolio_positions ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id              SERIAL      PRIMARY KEY,
    project_id      INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
    ticker          TEXT        NOT NULL,
    quantity        REAL        NOT NULL DEFAULT 0,
    avg_cost        REAL        NOT NULL DEFAULT 0,
    currency        TEXT        NOT NULL DEFAULT 'USD',
    notes           TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_positions_project ON portfolio_positions (project_id);
CREATE INDEX IF NOT EXISTS idx_positions_ticker  ON portfolio_positions (ticker);

-- ── tracked_jobs ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tracked_jobs (
    id              SERIAL      PRIMARY KEY,
    project_id      INTEGER     REFERENCES projects(id) ON DELETE CASCADE,
    job_type        TEXT        NOT NULL DEFAULT 'research',  -- research|watch|scrape
    external_id     TEXT        NOT NULL DEFAULT '',
    status          TEXT        NOT NULL DEFAULT 'pending',
    result_summary  TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tracked_jobs_project ON tracked_jobs (project_id);
CREATE INDEX IF NOT EXISTS idx_tracked_jobs_status  ON tracked_jobs (status);

-- ── tags / content_tags ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tags (
    id      SERIAL  PRIMARY KEY,
    name    TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS content_tags (
    id              SERIAL      PRIMARY KEY,
    tag_id          INTEGER NOT NULL REFERENCES tags(id)         ON DELETE CASCADE,
    web_page_id     INTEGER          REFERENCES web_pages(id)    ON DELETE CASCADE,
    user_file_id    INTEGER          REFERENCES user_files(id)   ON DELETE CASCADE,
    artifact_id     INTEGER          REFERENCES generated_artifacts(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_content_tags_page     ON content_tags (web_page_id)  WHERE web_page_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_content_tags_file     ON content_tags (user_file_id) WHERE user_file_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_content_tags_artifact ON content_tags (artifact_id)  WHERE artifact_id IS NOT NULL;

-- ── schema_migrations ────────────────────────────────────────────────────────
-- Tracks which migration versions have been applied (idempotent runner).
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT        PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── embeddings ────────────────────────────────────────────────────────────────
-- 384-dim vectors (all-MiniLM-L6-v2 or nomic-embed-text).
-- source_type identifies which table the chunk came from.
CREATE TABLE IF NOT EXISTS embeddings (
    id              SERIAL      PRIMARY KEY,
    source_type     TEXT        NOT NULL,   -- web_page | user_file | artifact | memory_chunk | research_finding
    source_id       INTEGER     NOT NULL,
    chunk_index     INTEGER     NOT NULL DEFAULT 0,
    chunk_text      TEXT        NOT NULL,
    embedding       vector(384),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vec
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
