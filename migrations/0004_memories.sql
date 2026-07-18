CREATE TABLE memories (
    id TEXT PRIMARY KEY NOT NULL,
    org_id TEXT NOT NULL,
    container_tag TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    is_inferred INTEGER NOT NULL DEFAULT 0 CHECK (is_inferred IN (0, 1)),
    is_static INTEGER NOT NULL DEFAULT 0 CHECK (is_static IN (0, 1)),
    is_latest INTEGER NOT NULL DEFAULT 1 CHECK (is_latest IN (0, 1)),
    is_forgotten INTEGER NOT NULL DEFAULT 0 CHECK (is_forgotten IN (0, 1)),
    root_memory_id TEXT,
    parent_memory_id TEXT,
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    forget_after TEXT,
    forget_reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id),
    FOREIGN KEY (parent_memory_id) REFERENCES memories(id)
) STRICT;

CREATE TABLE memory_sources (
    memory_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, document_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE memory_relations (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('updates', 'extends', 'derives')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (parent_id, child_id, relation),
    FOREIGN KEY (parent_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (child_id) REFERENCES memories(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE memory_embeddings (
    memory_id TEXT PRIMARY KEY NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX memories_active_profile_idx
    ON memories (org_id, container_tag, is_latest, is_forgotten, is_static, updated_at);
CREATE INDEX memory_relations_child_idx ON memory_relations (child_id);
CREATE INDEX memory_sources_document_idx ON memory_sources (document_id);
CREATE INDEX memory_embeddings_active_idx ON memory_embeddings (model_id, dimensions, active);
