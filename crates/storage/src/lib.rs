//! `SQLite` persistence, migrations, and atomic document identity decisions.

use std::{path::Path, thread, time::Duration};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::Serialize;
use serde_json::{Map, Value};
use sha1::{Digest, Sha1};
use thiserror::Error;

const MIGRATIONS: &[(i64, &str)] = &[
    (1, include_str!("../../../migrations/0001_initial.sql")),
    (
        2,
        include_str!("../../../migrations/0002_searchable_chunks.sql"),
    ),
];
const BASE58: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const LOCAL_SLUG: &str = "local";

/// All values needed for a document create or compatibility upsert.
#[derive(Debug, Clone)]
pub struct UpsertDocument {
    pub content: String,
    pub custom_id: Option<String>,
    pub container_tags: Vec<String>,
    pub entity_context: Option<String>,
    pub metadata: Map<String, Value>,
    pub task_type: String,
    pub filepath: Option<String>,
    pub filter_by_metadata: Map<String, Value>,
    pub dreaming: String,
}

/// Result of an atomic identity/upsert decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpsertResult {
    pub id: String,
    pub status: String,
    pub enqueued: bool,
}

/// A claimed document job whose processing happens outside the database lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedJob {
    pub id: String,
    pub document_id: String,
    pub content: String,
}

/// A full-text search result.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub document_id: String,
    pub chunk: String,
    pub score: f64,
}

/// Stored representation returned by the HTTP API.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    pub id: String,
    pub content: String,
    pub custom_id: Option<String>,
    pub status: String,
    pub container_tags: Vec<String>,
    pub entity_context: Option<String>,
    pub metadata: Map<String, Value>,
    pub task_type: String,
    pub filepath: Option<String>,
    pub filter_by_metadata: Map<String, Value>,
    pub dreaming: String,
    pub created_at: String,
    pub updated_at: String,
}

struct StoredDocument {
    document: Document,
    content_hash: String,
}

/// An initialized application database connection.
pub struct Storage {
    connection: Connection,
    local_org_id: String,
}

impl Storage {
    /// Opens a database and applies all embedded migrations.
    ///
    /// # Errors
    /// Returns an error when the database cannot be opened, migrated, or seeded.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let mut connection = Connection::open(path).map_err(StorageError::Open)?;
        let local_org_id = initialize(&mut connection)?;
        Ok(Self {
            connection,
            local_org_id,
        })
    }

    /// Opens an isolated in-memory database and applies all migrations.
    ///
    /// # Errors
    /// Returns an error when the database cannot be migrated or seeded.
    pub fn in_memory() -> Result<Self, StorageError> {
        let mut connection = Connection::open_in_memory().map_err(StorageError::Open)?;
        let local_org_id = initialize(&mut connection)?;
        Ok(Self {
            connection,
            local_org_id,
        })
    }

    /// Returns the organization used for local unauthenticated requests.
    #[must_use]
    pub fn local_organization_id(&self) -> &str {
        &self.local_org_id
    }

    /// Sanitizes, identifies, updates, and if needed queues a document atomically.
    ///
    /// # Errors
    /// Returns an error when randomness, serialization, or the transaction fails.
    pub fn upsert_document(
        &mut self,
        mut input: UpsertDocument,
    ) -> Result<UpsertResult, StorageError> {
        input.content = sanitize_content(&input.content);
        let hash = content_hash(&input.content);
        let org_id = self.local_org_id.clone();
        let tx = self.connection.transaction().map_err(StorageError::Write)?;

        let existing = if let Some(custom_id) = input.custom_id.as_deref() {
            find_custom(&tx, &org_id, custom_id, &input.container_tags)?
        } else {
            find_duplicate(&tx, &org_id, &hash, &input.metadata, &input.container_tags)?
        };

        let result = if let Some(existing) = existing {
            apply_existing(&tx, &existing, &input, &hash)?
        } else {
            insert_new(&tx, &org_id, &input, &hash)?
        };
        tx.commit().map_err(StorageError::Write)?;
        Ok(result)
    }

    /// Finds by internal ID first, then custom ID, within the local organization.
    ///
    /// # Errors
    /// Returns an error when the query fails or stored JSON is malformed.
    pub fn find_document(&self, identifier: &str) -> Result<Option<Document>, StorageError> {
        if let Some(found) = query_one(&self.connection, "id = ?2", &self.local_org_id, identifier)?
        {
            return Ok(Some(found.document));
        }
        Ok(query_one(
            &self.connection,
            "custom_id = ?2",
            &self.local_org_id,
            identifier,
        )?
        .map(|found| found.document))
    }

    /// Atomically claims the oldest available document job.
    ///
    /// # Errors
    /// Returns an error if the claim transaction cannot be read, written, or committed.
    pub fn claim_job(&mut self) -> Result<Option<ClaimedJob>, StorageError> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(StorageError::Write)?;
        let job = tx
            .query_row(
                "SELECT jobs.id, documents.id, documents.content FROM jobs JOIN documents ON documents.id=jobs.document_id WHERE jobs.status='queued' AND jobs.available_at <= CURRENT_TIMESTAMP ORDER BY jobs.created_at, jobs.rowid LIMIT 1",
                [],
                |row| Ok(ClaimedJob { id: row.get(0)?, document_id: row.get(1)?, content: row.get(2)? }),
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(job) = job else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(None);
        };
        tx.execute(
            "UPDATE jobs SET status='extracting', attempts=attempts+1, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND status='queued'",
            [&job.id],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='extracting', updated_at=CURRENT_TIMESTAMP WHERE id=?1",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(Some(job))
    }

    /// Replaces a document's searchable chunks and completes its claimed job atomically.
    ///
    /// # Errors
    /// Returns an error if searchable content or lifecycle state cannot be persisted.
    pub fn complete_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[String],
    ) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id=?1",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        for (ordinal, content) in chunks.iter().enumerate() {
            tx.execute(
                "INSERT INTO document_chunks (document_id, ordinal, content) VALUES (?1, ?2, ?3)",
                params![job.document_id, ordinal, content],
            )
            .map_err(StorageError::Write)?;
        }
        tx.execute(
            "UPDATE jobs SET status='done', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND status='extracting'",
            [&job.id],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }

    /// Marks a claimed job and its document failed while preserving prior indexed content.
    ///
    /// # Errors
    /// Returns an error if the failure state cannot be persisted atomically.
    pub fn fail_job(&mut self, job: &ClaimedJob, error: &str) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE jobs SET status='failed', last_error=?2, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
            params![job.id, error],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?1",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }

    /// Deletes a claimed document whose extracted content produced no chunks.
    ///
    /// # Errors
    /// Returns an error if the document cleanup transaction cannot be committed.
    pub fn delete_empty_document(&mut self, job: &ClaimedJob) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        tx.execute(
            "DELETE FROM documents WHERE id=?1 AND status='extracting'",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }

    /// Searches completed local documents using `SQLite` FTS5 relevance.
    ///
    /// # Errors
    /// Returns an error if the search query cannot be executed or decoded.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, StorageError> {
        let query = format!("\"{}\"", query.replace('"', "\"\""));
        let mut statement = self.connection.prepare(
            "SELECT documents.id, document_chunks.content, -bm25(document_chunks_fts) FROM document_chunks_fts JOIN document_chunks ON document_chunks.id=document_chunks_fts.rowid JOIN documents ON documents.id=document_chunks.document_id WHERE document_chunks_fts MATCH ?1 AND documents.org_id=?2 AND documents.status='done' ORDER BY bm25(document_chunks_fts), document_chunks.ordinal LIMIT ?3",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(params![query, self.local_org_id, limit], |row| {
                Ok(SearchHit {
                    document_id: row.get(0)?,
                    chunk: row.get(1)?,
                    score: row.get(2)?,
                })
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Returns the current migration version.
    ///
    /// # Errors
    /// Returns an error when the migration ledger cannot be read.
    pub fn migration_version(&self) -> Result<i64, StorageError> {
        self.connection
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .map_err(StorageError::Read)
    }
}

fn apply_existing(
    tx: &rusqlite::Transaction<'_>,
    existing: &StoredDocument,
    input: &UpsertDocument,
    hash: &str,
) -> Result<UpsertResult, StorageError> {
    let id = existing.document.id.clone();
    let status = existing.document.status.as_str();
    let active = matches!(
        status,
        "unknown" | "queued" | "extracting" | "chunking" | "embedding" | "indexing"
    );
    if active {
        return Ok(UpsertResult {
            id,
            status: status.to_owned(),
            enqueued: false,
        });
    }

    if status == "failed" {
        update_full(
            tx,
            &id,
            input,
            hash,
            &merge_metadata(&existing.document.metadata, &input.metadata),
            "queued",
        )?;
        let enqueued = enqueue_unless_active(tx, &id)?;
        return Ok(UpsertResult {
            id,
            status: "queued".to_owned(),
            enqueued,
        });
    }

    if existing.content_hash == hash {
        if metadata_equivalent(&existing.document.metadata, &input.metadata) {
            return Ok(UpsertResult {
                id,
                status: "done".to_owned(),
                enqueued: false,
            });
        }
        let metadata = merge_metadata(&existing.document.metadata, &input.metadata);
        tx.execute(
            "UPDATE documents SET metadata = ?2, updated_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id, json(&metadata)?],
        )
        .map_err(StorageError::Write)?;
        return Ok(UpsertResult {
            id,
            status: "done".to_owned(),
            enqueued: false,
        });
    }

    let metadata = merge_metadata(&existing.document.metadata, &input.metadata);
    update_full(tx, &id, input, hash, &metadata, "queued")?;
    let enqueued = enqueue_unless_active(tx, &id)?;
    Ok(UpsertResult {
        id,
        status: "queued".to_owned(),
        enqueued,
    })
}

fn insert_new(
    tx: &rusqlite::Transaction<'_>,
    org_id: &str,
    input: &UpsertDocument,
    hash: &str,
) -> Result<UpsertResult, StorageError> {
    let id = generate_id()?;
    tx.execute(
        "INSERT INTO documents (id, org_id, content, content_hash, custom_id, status, container_tags, entity_context, metadata, task_type, filepath, filter_by_metadata, dreaming) VALUES (?1, ?2, ?3, ?4, ?5, 'queued', ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![id, org_id, input.content, hash, input.custom_id, json(&input.container_tags)?, input.entity_context, json(&input.metadata)?, input.task_type, input.filepath, json(&input.filter_by_metadata)?, input.dreaming],
    ).map_err(StorageError::Write)?;
    enqueue_unless_active(tx, &id)?;
    Ok(UpsertResult {
        id,
        status: "queued".to_owned(),
        enqueued: true,
    })
}

fn update_full(
    tx: &rusqlite::Transaction<'_>,
    id: &str,
    input: &UpsertDocument,
    hash: &str,
    metadata: &Map<String, Value>,
    status: &str,
) -> Result<(), StorageError> {
    tx.execute(
        "UPDATE documents SET content=?2, content_hash=?3, custom_id=?4, status=?5, container_tags=?6, entity_context=?7, metadata=?8, task_type=?9, filepath=?10, filter_by_metadata=?11, dreaming=?12, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
        params![id, input.content, hash, input.custom_id, status, json(&input.container_tags)?, input.entity_context, json(metadata)?, input.task_type, input.filepath, json(&input.filter_by_metadata)?, input.dreaming],
    ).map_err(StorageError::Write)?;
    Ok(())
}

fn enqueue_unless_active(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<bool, StorageError> {
    let active: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM jobs WHERE document_id=?1 AND status IN ('queued','extracting','chunking','embedding','indexing'))",
        [document_id], |row| row.get(0),
    ).map_err(StorageError::Read)?;
    if active {
        return Ok(false);
    }
    tx.execute(
        "INSERT INTO jobs (id, document_id, kind, status) VALUES (?1, ?2, 'document', 'queued')",
        params![generate_id()?, document_id],
    )
    .map_err(StorageError::Write)?;
    Ok(true)
}

fn find_custom(
    connection: &Connection,
    org_id: &str,
    custom_id: &str,
    tags: &[String],
) -> Result<Option<StoredDocument>, StorageError> {
    let tags = json(tags)?;
    let mut statement = connection.prepare(&format!("{SELECT_DOCUMENT} WHERE org_id=?1 AND custom_id=?2 AND container_tags=?3 ORDER BY rowid LIMIT 1")).map_err(StorageError::Read)?;
    read_optional(statement.query_row(params![org_id, custom_id, tags], read_document))
}

fn find_duplicate(
    connection: &Connection,
    org_id: &str,
    hash: &str,
    metadata: &Map<String, Value>,
    tags: &[String],
) -> Result<Option<StoredDocument>, StorageError> {
    let wanted_tags = normalized_tags(tags);
    let mut statement = connection
        .prepare(&format!(
            "{SELECT_DOCUMENT} WHERE org_id=?1 AND content_hash=?2 AND status='done' ORDER BY rowid"
        ))
        .map_err(StorageError::Read)?;
    let rows = statement
        .query_map(params![org_id, hash], read_document)
        .map_err(StorageError::Read)?;
    for row in rows {
        let candidate = row.map_err(StorageError::Read)?;
        if normalized_tags(&candidate.document.container_tags) == wanted_tags
            && metadata_equivalent(&candidate.document.metadata, metadata)
        {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

const SELECT_DOCUMENT: &str = "SELECT id, content, content_hash, custom_id, status, container_tags, entity_context, metadata, task_type, filepath, filter_by_metadata, dreaming, created_at, updated_at FROM documents";

fn query_one(
    connection: &Connection,
    predicate: &str,
    org_id: &str,
    identifier: &str,
) -> Result<Option<StoredDocument>, StorageError> {
    let mut statement = connection
        .prepare(&format!(
            "{SELECT_DOCUMENT} WHERE org_id=?1 AND {predicate} ORDER BY rowid LIMIT 1"
        ))
        .map_err(StorageError::Read)?;
    read_optional(statement.query_row(params![org_id, identifier], read_document))
}

fn read_optional(
    result: rusqlite::Result<StoredDocument>,
) -> Result<Option<StoredDocument>, StorageError> {
    result.optional().map_err(StorageError::Read)
}

fn read_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredDocument> {
    let tags: String = row.get(5)?;
    let metadata: String = row.get(7)?;
    let filter: String = row.get(10)?;
    Ok(StoredDocument {
        document: Document {
            id: row.get(0)?,
            content: row.get(1)?,
            custom_id: row.get(3)?,
            status: row.get(4)?,
            container_tags: parse_json(&tags, 5)?,
            entity_context: row.get(6)?,
            metadata: parse_json(&metadata, 7)?,
            task_type: row.get(8)?,
            filepath: row.get(9)?,
            filter_by_metadata: parse_json(&filter, 10)?,
            dreaming: row.get(11)?,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
        },
        content_hash: row.get(2)?,
    })
}

fn parse_json<T: serde::de::DeserializeOwned>(value: &str, column: usize) -> rusqlite::Result<T> {
    serde_json::from_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}

fn json(value: &(impl Serialize + ?Sized)) -> Result<String, StorageError> {
    serde_json::to_string(value).map_err(StorageError::Serialize)
}

fn merge_metadata(
    existing: &Map<String, Value>,
    incoming: &Map<String, Value>,
) -> Map<String, Value> {
    let mut merged = existing.clone();
    for (key, value) in incoming {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

fn metadata_equivalent(left: &Map<String, Value>, right: &Map<String, Value>) -> bool {
    fn comparable(source: &Map<String, Value>) -> Map<String, Value> {
        source
            .iter()
            .filter(|(key, _)| !key.starts_with("sm_") && key.as_str() != "commonQuestions")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
    serde_json::to_string(&comparable(left)).ok() == serde_json::to_string(&comparable(right)).ok()
}

fn normalized_tags(tags: &[String]) -> Vec<&str> {
    let mut result: Vec<_> = tags
        .iter()
        .map(String::as_str)
        .filter(|tag| !tag.trim().is_empty())
        .collect();
    result.sort_unstable();
    result
}

/// Applies v0.0.5 content sanitization.
#[must_use]
pub fn sanitize_content(content: &str) -> String {
    content.trim().chars().filter(|character| !matches!(*character as u32, 0x00..=0x08 | 0x0b | 0x0c | 0x0e..=0x1f | 0x7f)).collect::<String>().trim().to_owned()
}

/// Computes the lowercase SHA-1 hash of UTF-8 content.
#[must_use]
pub fn content_hash(content: &str) -> String {
    format!("{:x}", Sha1::digest(content.as_bytes()))
}

/// Generates an unbiased 22-character Bitcoin Base58 identifier from OS randomness.
///
/// # Errors
/// Returns an error when the operating system random source is unavailable.
pub fn generate_id() -> Result<String, StorageError> {
    let mut id = String::with_capacity(22);
    let mut random = [0_u8; 32];
    while id.len() < 22 {
        getrandom::fill(&mut random).map_err(StorageError::Random)?;
        for byte in random.into_iter().filter(|byte| *byte < 232) {
            id.push(char::from(BASE58[usize::from(byte % 58)]));
            if id.len() == 22 {
                break;
            }
        }
    }
    Ok(id)
}

fn initialize(connection: &mut Connection) -> Result<String, StorageError> {
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(StorageError::Configure)?;
    connection
        .execute_batch("PRAGMA foreign_keys=ON;")
        .map_err(StorageError::Configure)?;
    configure_wal(connection)?;
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(StorageError::Migrate)?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY NOT NULL, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP) STRICT;").map_err(StorageError::Migrate)?;
    let current: i64 = tx
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |row| row.get(0),
        )
        .map_err(StorageError::Migrate)?;
    for &(version, sql) in MIGRATIONS.iter().filter(|(version, _)| *version > current) {
        tx.execute_batch(sql).map_err(StorageError::Migrate)?;
        tx.execute(
            "INSERT INTO schema_migrations (version) VALUES (?1)",
            [version],
        )
        .map_err(StorageError::Migrate)?;
    }
    validate_columns(&tx, "organizations", &["id", "slug", "created_at"])?;
    validate_columns(
        &tx,
        "documents",
        &[
            "id",
            "org_id",
            "content",
            "content_hash",
            "custom_id",
            "status",
            "container_tags",
            "entity_context",
            "metadata",
            "task_type",
            "filepath",
            "filter_by_metadata",
            "dreaming",
            "created_at",
            "updated_at",
        ],
    )?;
    validate_columns(
        &tx,
        "document_chunks",
        &["id", "document_id", "ordinal", "content"],
    )?;
    validate_columns(
        &tx,
        "jobs",
        &[
            "id",
            "document_id",
            "kind",
            "status",
            "attempts",
            "available_at",
            "last_error",
            "created_at",
            "updated_at",
        ],
    )?;
    let org_id = tx
        .query_row(
            "SELECT id FROM organizations WHERE slug=?1",
            [LOCAL_SLUG],
            |row| row.get(0),
        )
        .optional()
        .map_err(StorageError::Migrate)?;
    let org_id = if let Some(id) = org_id {
        id
    } else {
        let id = generate_id()?;
        tx.execute(
            "INSERT INTO organizations (id, slug) VALUES (?1, ?2)",
            params![id, LOCAL_SLUG],
        )
        .map_err(StorageError::Migrate)?;
        id
    };
    tx.execute(
        "UPDATE jobs SET status='queued', available_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE status IN ('extracting','chunking','embedding','indexing')",
        [],
    )
    .map_err(StorageError::Migrate)?;
    tx.execute(
        "UPDATE documents SET status='queued', updated_at=CURRENT_TIMESTAMP WHERE id IN (SELECT document_id FROM jobs WHERE status='queued') AND status IN ('extracting','chunking','embedding','indexing')",
        [],
    )
    .map_err(StorageError::Migrate)?;
    tx.commit().map_err(StorageError::Migrate)?;
    Ok(org_id)
}

fn configure_wal(connection: &Connection) -> Result<(), StorageError> {
    const RETRIES: usize = 5;
    for attempt in 0..=RETRIES {
        match connection.execute_batch("PRAGMA journal_mode=WAL;") {
            Ok(()) => return Ok(()),
            Err(error) if attempt < RETRIES && is_lock_error(&error) => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(StorageError::Configure(error)),
        }
    }
    unreachable!("bounded WAL configuration loop always returns")
}

fn is_lock_error(error: &rusqlite::Error) -> bool {
    matches!(
        error.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
    )
}

fn validate_columns(
    connection: &Connection,
    table: &str,
    expected: &[&str],
) -> Result<(), StorageError> {
    let mut statement = connection
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(StorageError::Migrate)?;
    let actual = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(StorageError::Migrate)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(StorageError::Migrate)?;
    if actual
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
    {
        Ok(())
    } else {
        Err(StorageError::MalformedSchema {
            table: table.to_owned(),
            expected: expected.join(", "),
            actual: actual.join(", "),
        })
    }
}

/// Failure while opening or using application storage.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("failed to open the SQLite database: {0}")]
    Open(#[source] rusqlite::Error),
    #[error("failed to configure SQLite; the database was not modified: {0}")]
    Configure(#[source] rusqlite::Error),
    #[error("failed to apply SQLite migrations; the migration was rolled back: {0}")]
    Migrate(#[source] rusqlite::Error),
    #[error("failed to write SQLite data; the transaction was rolled back: {0}")]
    Write(#[source] rusqlite::Error),
    #[error("failed to read SQLite data: {0}")]
    Read(#[source] rusqlite::Error),
    #[error("failed to serialize document data: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("the operating system random source failed; no identifier was generated: {0}")]
    Random(getrandom::Error),
    #[error(
        "existing table {table} has malformed columns; expected [{expected}], found [{actual}]; repair or recreate this unreleased schema-v1 database"
    )]
    MalformedSchema {
        table: String,
        expected: String,
        actual: String,
    },
}
