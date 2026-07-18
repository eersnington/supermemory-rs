CREATE TABLE document_chunks (
    id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    content TEXT NOT NULL,
    UNIQUE (document_id, ordinal),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) STRICT;

CREATE VIRTUAL TABLE document_chunks_fts USING fts5(
    content,
    content='document_chunks',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER document_chunks_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER document_chunks_delete AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER document_chunks_update AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO document_chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
