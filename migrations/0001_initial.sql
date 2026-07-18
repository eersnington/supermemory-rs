CREATE TABLE organizations (
    id TEXT PRIMARY KEY NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE TABLE documents (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    custom_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('unknown', 'queued', 'extracting', 'chunking', 'embedding', 'indexing', 'done', 'failed')),
    container_tags TEXT NOT NULL,
    entity_context TEXT,
    metadata TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('memory', 'superrag')),
    filepath TEXT,
    filter_by_metadata TEXT NOT NULL,
    dreaming TEXT NOT NULL CHECK (dreaming IN ('instant', 'dynamic')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id)
) STRICT;

CREATE TABLE jobs (
    id TEXT PRIMARY KEY NOT NULL,
    document_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'extracting', 'chunking', 'embedding', 'indexing', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    available_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX jobs_available_idx
    ON jobs (status, available_at);

CREATE INDEX documents_custom_id_idx
    ON documents (org_id, custom_id);

CREATE INDEX documents_content_hash_idx
    ON documents (org_id, content_hash);

CREATE UNIQUE INDEX jobs_one_active_document_idx
    ON jobs (document_id)
    WHERE status IN ('queued', 'extracting', 'chunking', 'embedding', 'indexing');
