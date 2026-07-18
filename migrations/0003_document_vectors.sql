ALTER TABLE documents ADD COLUMN revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0);
ALTER TABLE jobs ADD COLUMN revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0);
ALTER TABLE document_chunks ADD COLUMN stable_id TEXT;

UPDATE document_chunks SET stable_id = 'legacy-' || id WHERE stable_id IS NULL;

CREATE UNIQUE INDEX document_chunks_stable_id_idx ON document_chunks (stable_id);
CREATE INDEX jobs_document_revision_idx ON jobs (document_id, revision);

CREATE TABLE chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX chunk_embeddings_model_idx ON chunk_embeddings (model_id, dimensions);
