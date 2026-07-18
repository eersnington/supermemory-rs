ALTER TABLE organizations ADD COLUMN name TEXT;
ALTER TABLE organizations ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}';

ALTER TABLE documents ADD COLUMN title TEXT;
ALTER TABLE documents ADD COLUMN summary TEXT;
ALTER TABLE documents ADD COLUMN document_type TEXT;
ALTER TABLE documents ADD COLUMN source TEXT;
ALTER TABLE documents ADD COLUMN url TEXT;
ALTER TABLE documents ADD COLUMN user_id TEXT;

ALTER TABLE memories ADD COLUMN source_count INTEGER NOT NULL DEFAULT 0 CHECK (source_count >= 0);

CREATE TABLE spaces (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    container_tag TEXT NOT NULL,
    entity_context TEXT,
    name TEXT,
    description TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    profile_buckets TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (org_id, container_tag),
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE api_keys (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT,
    key_hash BLOB NOT NULL,
    name TEXT,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX api_keys_hash_idx ON api_keys (key_hash, enabled);

CREATE TABLE legacy_imports (
    source_hash TEXT PRIMARY KEY NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    report TEXT NOT NULL
) STRICT;
