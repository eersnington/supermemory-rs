DROP INDEX jobs_one_active_document_idx;

CREATE UNIQUE INDEX jobs_one_active_document_revision_idx
    ON jobs (document_id, revision)
    WHERE status IN ('queued', 'extracting', 'chunking', 'embedding', 'indexing');
