CREATE TABLE organization_settings (
    org_id TEXT PRIMARY KEY NOT NULL,
    chunk_size INTEGER NOT NULL DEFAULT 1075 CHECK (chunk_size > 0),
    should_llm_filter INTEGER NOT NULL DEFAULT 0 CHECK (should_llm_filter IN (0, 1)),
    filter_prompt TEXT,
    include_items TEXT,
    exclude_items TEXT,
    profile_buckets TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE file_blobs (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    sha256 BLOB NOT NULL,
    content_type TEXT,
    filename TEXT,
    byte_length INTEGER NOT NULL CHECK (byte_length >= 0),
    bytes BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (org_id, sha256),
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE content_sources (
    document_id TEXT PRIMARY KEY NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('text', 'url', 'file', 'conversation')),
    source_url TEXT,
    file_blob_id TEXT,
    content_type TEXT,
    extraction_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (extraction_status IN ('pending', 'extracting', 'done', 'failed')),
    extracted_content TEXT,
    extracted_metadata TEXT NOT NULL DEFAULT '{}',
    error_kind TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (file_blob_id) REFERENCES file_blobs(id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE download_tokens (
    token_hash BLOB PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    file_blob_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (file_blob_id) REFERENCES file_blobs(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE container_tag_merge_jobs (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    source_tags TEXT NOT NULL,
    target_tag TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE memory_forget_batches (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    container_tag TEXT NOT NULL,
    query TEXT NOT NULL,
    dry_run INTEGER NOT NULL CHECK (dry_run IN (0, 1)),
    candidate_count INTEGER NOT NULL DEFAULT 0 CHECK (candidate_count >= 0),
    forgotten_count INTEGER NOT NULL DEFAULT 0 CHECK (forgotten_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX file_blobs_org_created_idx ON file_blobs (org_id, created_at);
CREATE INDEX content_sources_extraction_idx ON content_sources (extraction_status, updated_at);
CREATE INDEX download_tokens_expiry_idx ON download_tokens (expires_at);
CREATE INDEX container_tag_merge_jobs_status_idx ON container_tag_merge_jobs (org_id, status, created_at);
CREATE INDEX memory_forget_batches_org_created_idx ON memory_forget_batches (org_id, container_tag, created_at);
