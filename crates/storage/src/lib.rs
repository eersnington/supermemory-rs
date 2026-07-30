//! `SQLite` persistence, migrations, and atomic document identity decisions.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use rusqlite::{
    Connection, OptionalExtension, TransactionBehavior, functions::FunctionFlags, params,
};
use serde::Serialize;
use serde_json::{Map, Value, json};
use sha1::{Digest, Sha1};
use sha2::Sha256;
use thiserror::Error;

const MIGRATIONS: &[(i64, &str)] = &[
    (1, include_str!("../../../migrations/0001_initial.sql")),
    (
        2,
        include_str!("../../../migrations/0002_searchable_chunks.sql"),
    ),
    (
        3,
        include_str!("../../../migrations/0003_document_vectors.sql"),
    ),
    (4, include_str!("../../../migrations/0004_memories.sql")),
    (
        5,
        include_str!("../../../migrations/0005_legacy_compatibility.sql"),
    ),
    (
        6,
        include_str!("../../../migrations/0006_memory_job_state.sql"),
    ),
    (
        7,
        include_str!("../../../migrations/0007_revision_scoped_active_jobs.sql"),
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
    pub revision: i64,
    pub organization_id: String,
    pub container_tag: String,
    pub document_date: Option<String>,
}

/// A durable provider-extraction job claimed independently of document indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedMemoryJob {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub revision: i64,
    pub organization_id: String,
    pub container_tag: String,
    pub document_date: Option<String>,
    pub extraction_result: Option<String>,
    pub attempts: i64,
}

/// Existing organization memory supplied as extraction context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistingMemory {
    pub id: String,
    pub content: String,
}

/// A chunk and its normalized embedding ready for atomic publication.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddedChunk<'a> {
    pub content: &'a str,
    pub vector: &'a [f32],
}

/// A full-text search result.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub id: String,
    pub document_id: String,
    pub chunk: String,
    pub score: f64,
    pub position: usize,
    pub custom_id: Option<String>,
    pub metadata: Map<String, Value>,
    pub filepath: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub document_content: String,
}

/// Organization-local document constraints applied before semantic ranking.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub organization_id: Option<String>,
    pub container_tags: Vec<String>,
    pub document_id: Option<String>,
    pub filepath: Option<String>,
    pub filters: Option<FilterExpression>,
}

/// Boolean metadata filter tree used by V3 and V4 search.
#[derive(Debug, Clone)]
pub enum FilterExpression {
    And(Vec<FilterExpression>),
    Or(Vec<FilterExpression>),
    Condition(FilterCondition),
}

/// One metadata comparison in a search filter.
#[derive(Debug, Clone)]
pub struct FilterCondition {
    pub key: String,
    pub value: String,
    pub kind: FilterKind,
    pub numeric_operator: NumericOperator,
    pub negate: bool,
    pub ignore_case: bool,
}

/// Comparison behavior for a metadata condition.
#[derive(Debug, Clone, Copy)]
pub enum FilterKind {
    Metadata,
    Numeric,
    ArrayContains,
    StringContains,
}

/// Numeric comparison operator.
#[derive(Debug, Clone, Copy)]
pub enum NumericOperator {
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
}

/// Parent relation supplied to deterministic memory reconciliation.
#[derive(Debug, Clone)]
pub struct MemoryParent {
    pub memory_id: String,
    pub relation: String,
}

/// Validated model proposal and embedding ready for reconciliation.
#[derive(Debug, Clone)]
pub struct MemoryProposal {
    pub temporary_id: String,
    pub content: String,
    pub is_inferred: bool,
    pub is_static: bool,
    pub metadata: Map<String, Value>,
    pub parents: Vec<MemoryParent>,
    pub forget_after: Option<String>,
    pub forget_reason: Option<String>,
    pub vector: Vec<f32>,
}

/// Canonical memory returned by reconciliation and profile queries.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
#[expect(
    clippy::struct_excessive_bools,
    reason = "canonical persisted memory flags"
)]
pub struct MemoryRecord {
    pub id: String,
    pub memory: String,
    pub metadata: Map<String, Value>,
    pub is_inferred: bool,
    pub is_static: bool,
    pub is_latest: bool,
    pub is_forgotten: bool,
    pub root_memory_id: Option<String>,
    pub parent_memory_id: Option<String>,
    pub version: i64,
    pub forget_after: Option<String>,
    pub forget_reason: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Exact semantic memory search result.
#[derive(Debug, Clone)]
pub struct MemorySearchHit {
    pub record: MemoryRecord,
    pub similarity: f64,
    pub parents: Vec<MemoryRelationHit>,
    pub children: Vec<MemoryRelationHit>,
    pub related: Vec<MemoryRelationHit>,
    pub documents: Vec<MemorySourceDocument>,
}

/// One memory connected through the lineage graph.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryRelationHit {
    pub relation: String,
    pub version: i64,
    pub memory: String,
    pub metadata: Map<String, Value>,
    pub updated_at: String,
}

/// Source document attached to a memory.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySourceDocument {
    pub id: String,
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub document_type: Option<String>,
    pub metadata: Map<String, Value>,
    pub summary: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Counts from an idempotent legacy JSONL import.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LegacyImportReport {
    pub organizations: usize,
    pub spaces: usize,
    pub documents: usize,
    pub chunks: usize,
    pub memories: usize,
    pub relations: usize,
    pub sources: usize,
    pub api_keys: usize,
}

/// Imported API-key hash paired with its organization, when known.
pub type ApiKeyIdentity = ([u8; 32], Option<String>);

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
    database_path: Option<PathBuf>,
}

impl Storage {
    /// Opens a database and applies all embedded migrations.
    ///
    /// # Errors
    /// Returns an error when the database cannot be opened, migrated, or seeded.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let database_path = path.as_ref().to_path_buf();
        let mut connection = Connection::open(&database_path).map_err(StorageError::Open)?;
        let local_org_id = initialize(&mut connection)?;
        Ok(Self {
            connection,
            local_org_id,
            database_path: Some(database_path),
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
            database_path: None,
        })
    }

    /// Opens an independent read connection without rerunning migrations or job recovery.
    ///
    /// # Errors
    /// Returns an error for in-memory storage or when the read connection cannot be configured.
    pub fn fork(&self) -> Result<Self, StorageError> {
        let path = self
            .database_path
            .as_ref()
            .ok_or(StorageError::CannotForkInMemory)?;
        let connection = Connection::open(path).map_err(StorageError::Open)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(StorageError::Configure)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA query_only=ON;")
            .map_err(StorageError::Configure)?;
        register_vector_functions(&connection)?;
        Ok(Self {
            connection,
            local_org_id: self.local_org_id.clone(),
            database_path: Some(path.clone()),
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
    pub fn upsert_document(&mut self, input: UpsertDocument) -> Result<UpsertResult, StorageError> {
        let org_id = self.local_org_id.clone();
        self.upsert_document_for(&org_id, input)
    }

    /// Applies document identity and upsert rules in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when identity resolution or persistence fails.
    pub fn upsert_document_for(
        &mut self,
        org_id: &str,
        mut input: UpsertDocument,
    ) -> Result<UpsertResult, StorageError> {
        input.content = sanitize_content(&input.content);
        let hash = content_hash(&input.content);
        let org_id = org_id.to_owned();
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
        self.find_document_for(&self.local_org_id, identifier)
    }

    /// Finds a document by internal then custom ID in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored data cannot be read.
    pub fn find_document_for(
        &self,
        org_id: &str,
        identifier: &str,
    ) -> Result<Option<Document>, StorageError> {
        if let Some(found) = query_one(&self.connection, "id = ?2", org_id, identifier)? {
            return Ok(Some(found.document));
        }
        Ok(
            query_one(&self.connection, "custom_id = ?2", org_id, identifier)?
                .map(|found| found.document),
        )
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
                "SELECT jobs.id, documents.id, documents.content, jobs.revision, documents.org_id, documents.container_tags, documents.metadata FROM jobs JOIN documents ON documents.id=jobs.document_id WHERE jobs.kind='document' AND jobs.status='queued' AND jobs.revision=documents.revision AND jobs.available_at <= CURRENT_TIMESTAMP ORDER BY jobs.created_at, jobs.rowid LIMIT 1",
                [],
                |row| {
                    let tags: Vec<String> = parse_json(&row.get::<_, String>(5)?, 5)?;
                    let metadata: Map<String, Value> = parse_json(&row.get::<_, String>(6)?, 6)?;
                    Ok(ClaimedJob {
                        id: row.get(0)?,
                        document_id: row.get(1)?,
                        content: row.get(2)?,
                        revision: row.get(3)?,
                        organization_id: row.get(4)?,
                        container_tag: tags.into_iter().next().unwrap_or_else(|| "sm_project_default".to_owned()),
                        document_date: metadata.get("date").and_then(Value::as_str).map(str::to_owned),
                    })
                },
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(job) = job else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(None);
        };
        tx.execute(
            "UPDATE jobs SET status='extracting', attempts=attempts+1, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='queued'",
            params![job.id, job.revision],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='extracting', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.document_id, job.revision],
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
        let embedded: Vec<_> = chunks
            .iter()
            .map(|content| EmbeddedChunk {
                content,
                vector: &[],
            })
            .collect();
        self.publish_job(job, &embedded, None, false)
    }

    /// Atomically publishes chunks and normalized vectors for a claimed revision.
    ///
    /// # Errors
    /// Returns an error when vector validation fails or the revision became stale.
    pub fn complete_embedded_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        model_id: &str,
        dimensions: usize,
    ) -> Result<(), StorageError> {
        if dimensions == 0 {
            return Err(StorageError::InvalidVectorDimensions { dimensions });
        }
        for chunk in chunks {
            validate_vector(chunk.vector, dimensions)?;
        }
        self.publish_job(job, chunks, Some((model_id, dimensions)), false)
    }

    /// Publishes chunks and schedules durable memory extraction in the same transaction.
    ///
    /// # Errors
    /// Returns an error for invalid vectors, stale revisions, or failed persistence.
    pub fn complete_embedded_job_with_memory_extraction(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        model_id: &str,
        dimensions: usize,
    ) -> Result<(), StorageError> {
        if dimensions != 768 {
            return Err(StorageError::InvalidVectorDimensions { dimensions });
        }
        for chunk in chunks {
            validate_vector(chunk.vector, dimensions)?;
        }
        self.publish_job(job, chunks, Some((model_id, dimensions)), true)
    }

    fn publish_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        embedding: Option<(&str, usize)>,
        schedule_memory_extraction: bool,
    ) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let current_revision: Option<i64> = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1",
                [&job.document_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        if current_revision != Some(job.revision) {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: current_revision,
            });
        }
        let mut existing = HashMap::new();
        {
            let mut statement = tx
                .prepare(
                    "SELECT ordinal, content, stable_id FROM document_chunks WHERE document_id=?1",
                )
                .map_err(StorageError::Read)?;
            let rows = statement
                .query_map([&job.document_id], |row| {
                    Ok((
                        row.get::<_, usize>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(StorageError::Read)?;
            for row in rows {
                let (ordinal, content, stable_id) = row.map_err(StorageError::Read)?;
                existing.insert((ordinal, content), stable_id);
            }
        }
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id=?1",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        for (ordinal, chunk) in chunks.iter().enumerate() {
            let stable_id = existing
                .remove(&(ordinal, chunk.content.to_owned()))
                .map_or_else(generate_id, Ok)?;
            tx.execute(
                "INSERT INTO document_chunks (document_id, ordinal, content, stable_id) VALUES (?1, ?2, ?3, ?4)",
                params![job.document_id, ordinal, chunk.content, stable_id],
            )
            .map_err(StorageError::Write)?;
            if let Some((model_id, dimensions)) = embedding {
                let chunk_id = tx.last_insert_rowid();
                tx.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, model_id, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                    params![chunk_id, model_id, dimensions, vector_bytes(chunk.vector)],
                )
                .map_err(StorageError::Write)?;
            }
        }
        tx.execute(
            "UPDATE jobs SET status='done', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status IN ('extracting','chunking','embedding','indexing')",
            params![job.id, job.revision],
        )
        .map_err(StorageError::Write)?;
        if schedule_memory_extraction {
            tx.execute(
                "INSERT INTO jobs (id, document_id, kind, status, revision) VALUES (?1, ?2, 'memory', 'queued', ?3)",
                params![generate_id()?, job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
            tx.execute(
                "UPDATE documents SET status='indexing', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
        } else {
            tx.execute(
                "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Advances a claimed job and its current document revision to a processing stage.
    ///
    /// # Errors
    /// Returns an error for an unsupported stage, stale revision, or failed transaction.
    pub fn mark_job_stage(&mut self, job: &ClaimedJob, stage: &str) -> Result<(), StorageError> {
        if !matches!(stage, "chunking" | "embedding" | "indexing") {
            return Err(StorageError::InvalidJobStage(stage.to_owned()));
        }
        let expected = match stage {
            "chunking" => "extracting",
            "embedding" => "chunking",
            "indexing" => "embedding",
            _ => return Err(StorageError::InvalidJobStage(stage.to_owned())),
        };
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let jobs = tx
            .execute(
                "UPDATE jobs SET status=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status=?4",
                params![job.id, job.revision, stage, expected],
            )
            .map_err(StorageError::Write)?;
        let documents = tx
            .execute(
                "UPDATE documents SET status=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision, stage],
            )
            .map_err(StorageError::Write)?;
        if jobs != 1 || documents != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: None,
            });
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Marks a claimed job and its document failed while preserving prior indexed content.
    ///
    /// # Errors
    /// Returns an error if the failure state cannot be persisted atomically.
    pub fn fail_job(&mut self, job: &ClaimedJob, error: &str) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE jobs SET status='failed', last_error=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.id, job.revision, error],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.document_id, job.revision],
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
            "DELETE FROM documents WHERE id=?1 AND revision=?2 AND status IN ('extracting','chunking')",
            params![job.document_id, job.revision],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }

    /// Claims one provider-extraction job and increments its durable attempt count.
    ///
    /// # Errors
    /// Returns an error when the claim transaction cannot be completed.
    pub fn claim_memory_job(&mut self) -> Result<Option<ClaimedMemoryJob>, StorageError> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(StorageError::Write)?;
        let job = tx
            .query_row(
                "SELECT jobs.id, documents.id, documents.content, jobs.revision, documents.org_id, documents.container_tags, documents.metadata, jobs.extraction_result, jobs.attempts FROM jobs JOIN documents ON documents.id=jobs.document_id WHERE jobs.kind='memory' AND jobs.status='queued' AND jobs.revision=documents.revision AND jobs.available_at<=CURRENT_TIMESTAMP ORDER BY jobs.created_at, jobs.rowid LIMIT 1",
                [],
                |row| {
                    let tags: Vec<String> = parse_json(&row.get::<_, String>(5)?, 5)?;
                    let metadata: Map<String, Value> = parse_json(&row.get::<_, String>(6)?, 6)?;
                    Ok(ClaimedMemoryJob {
                        id: row.get(0)?,
                        document_id: row.get(1)?,
                        content: row.get(2)?,
                        revision: row.get(3)?,
                        organization_id: row.get(4)?,
                        container_tag: tags.into_iter().next().unwrap_or_else(|| "sm_project_default".to_owned()),
                        document_date: metadata.get("date").and_then(Value::as_str).map(str::to_owned),
                        extraction_result: row.get(7)?,
                        attempts: row.get::<_, i64>(8)?.saturating_add(1),
                    })
                },
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(job) = job else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(None);
        };
        let updated = tx
            .execute(
                "UPDATE jobs SET status='extracting', attempts=attempts+1, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='queued'",
                params![job.id, job.revision],
            )
            .map_err(StorageError::Write)?;
        if updated != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id,
                claimed: job.revision,
                current: None,
            });
        }
        tx.commit().map_err(StorageError::Write)?;
        Ok(Some(job))
    }

    /// Returns current active memories used as provider extraction context.
    ///
    /// # Errors
    /// Returns an error when stored memories cannot be read.
    pub fn existing_memories_for_extraction(
        &self,
        org_id: &str,
        container_tag: &str,
    ) -> Result<Vec<ExistingMemory>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_latest=1 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after)>CURRENT_TIMESTAMP) ORDER BY updated_at DESC, id LIMIT 100",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(params![org_id, container_tag], |row| {
                Ok(ExistingMemory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                })
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Persists provider output separately from retry error text.
    ///
    /// # Errors
    /// Returns an error when the revision is stale or the write fails.
    pub fn cache_memory_extraction(
        &mut self,
        job: &ClaimedMemoryJob,
        result: &str,
    ) -> Result<(), StorageError> {
        let updated = self.connection.execute(
            "UPDATE jobs SET extraction_result=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='extracting' AND EXISTS(SELECT 1 FROM documents WHERE id=jobs.document_id AND revision=jobs.revision)",
            params![job.id, job.revision, result],
        ).map_err(StorageError::Write)?;
        if updated == 1 {
            Ok(())
        } else {
            Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: None,
            })
        }
    }

    /// Marks a memory job complete only while its claimed document revision is current.
    ///
    /// # Errors
    /// Returns an error when the completion transaction fails.
    pub fn complete_memory_job(&mut self, job: &ClaimedMemoryJob) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let current: Option<i64> = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1",
                [&job.document_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        if current != Some(job.revision) {
            tx.execute(
                "UPDATE jobs SET status='failed', last_error_kind='stale_revision', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.id, job.revision],
            )
            .map_err(StorageError::Write)?;
            tx.commit().map_err(StorageError::Write)?;
            return Ok(());
        }
        let jobs = tx.execute(
            "UPDATE jobs SET status='done', last_error_kind=NULL, last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='extracting'",
            params![job.id, job.revision],
        ).map_err(StorageError::Write)?;
        let documents = tx.execute(
            "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.document_id, job.revision],
        ).map_err(StorageError::Write)?;
        if jobs != 1 || documents != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current,
            });
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Persists a bounded retry or terminal memory-extraction failure.
    ///
    /// # Errors
    /// Returns an error when the revision is stale or retry state cannot be committed.
    pub fn retry_memory_job(
        &mut self,
        job: &ClaimedMemoryJob,
        error_kind: &str,
        error: &str,
        retry_delay: Option<Duration>,
    ) -> Result<(), StorageError> {
        const MAX_ATTEMPTS: i64 = 4;
        let terminal = retry_delay.is_none() || job.attempts >= MAX_ATTEMPTS;
        let delay_seconds = retry_delay
            .map(|delay| delay.as_secs().max(1))
            .unwrap_or_default();
        let modifier = format!("+{delay_seconds} seconds");
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let status = if terminal { "failed" } else { "queued" };
        let updated = tx.execute(
            "UPDATE jobs SET status=?3, last_error_kind=?4, last_error=?5, available_at=datetime(CURRENT_TIMESTAMP, ?6), updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='extracting'",
            params![job.id, job.revision, status, error_kind, error, modifier],
        ).map_err(StorageError::Write)?;
        if updated != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: None,
            });
        }
        if terminal {
            tx.execute(
                "UPDATE documents SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Searches completed local documents using `SQLite` FTS5 relevance.
    ///
    /// # Errors
    /// Returns an error if the search query cannot be executed or decoded.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, StorageError> {
        self.search_for(&self.local_org_id, query, limit)
    }

    /// Lexically searches one explicit organization.
    ///
    /// # Errors
    /// Returns an error if the FTS query cannot be executed.
    pub fn search_for(
        &self,
        org_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchHit>, StorageError> {
        let query = format!("\"{}\"", query.replace('"', "\"\""));
        let mut statement = self.connection.prepare(
            "SELECT document_chunks.stable_id, documents.id, document_chunks.content, -bm25(document_chunks_fts), document_chunks.ordinal, documents.custom_id, documents.metadata, documents.filepath, documents.created_at, documents.updated_at, documents.content FROM document_chunks_fts JOIN document_chunks ON document_chunks.id=document_chunks_fts.rowid JOIN documents ON documents.id=document_chunks.document_id WHERE document_chunks_fts MATCH ?1 AND documents.org_id=?2 AND documents.status='done' ORDER BY bm25(document_chunks_fts), document_chunks.ordinal LIMIT ?3",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(params![query, org_id, limit], |row| {
                Ok(SearchHit {
                    id: row.get(0)?,
                    document_id: row.get(1)?,
                    chunk: row.get(2)?,
                    score: row.get(3)?,
                    position: row.get(4)?,
                    custom_id: row.get(5)?,
                    metadata: parse_json(&row.get::<_, String>(6)?, 6)?,
                    filepath: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                    document_content: row.get(10)?,
                })
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Searches current chunks by exact cosine similarity over normalized vectors.
    ///
    /// # Errors
    /// Returns an error for malformed query/stored vectors or a failed database read.
    pub fn search_semantic(
        &self,
        query: &[f32],
        model_id: &str,
        limit: usize,
        threshold: f32,
        options: &SearchOptions,
    ) -> Result<Vec<SearchHit>, StorageError> {
        validate_vector(query, query.len())?;
        let dimensions = query.len();
        let organization_id = options
            .organization_id
            .as_deref()
            .unwrap_or(&self.local_org_id);
        let query_bytes = vector_bytes(query);
        let mut statement = self.connection.prepare(
            "SELECT document_chunks.stable_id, documents.id, document_chunks.content, cosine_similarity(chunk_embeddings.vector, ?1, ?2) AS score, document_chunks.ordinal, documents.custom_id, documents.metadata, documents.filepath, documents.created_at, documents.updated_at, documents.container_tags, documents.content FROM chunk_embeddings JOIN document_chunks ON document_chunks.id=chunk_embeddings.chunk_id JOIN documents ON documents.id=document_chunks.document_id WHERE chunk_embeddings.model_id=?3 AND chunk_embeddings.dimensions=?2 AND documents.org_id=?4 AND documents.status='done' AND (?5 IS NULL OR documents.id=?5 OR documents.custom_id=?5) AND cosine_similarity(chunk_embeddings.vector, ?1, ?2)>=?6 ORDER BY score DESC, documents.id, document_chunks.ordinal, document_chunks.stable_id",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(
                params![
                    query_bytes,
                    dimensions,
                    model_id,
                    organization_id,
                    options.document_id,
                    threshold
                ],
                |row| {
                    Ok((
                        SearchHit {
                            id: row.get(0)?,
                            document_id: row.get(1)?,
                            chunk: row.get(2)?,
                            score: row.get(3)?,
                            position: row.get(4)?,
                            custom_id: row.get(5)?,
                            metadata: parse_json(&row.get::<_, String>(6)?, 6)?,
                            filepath: row.get(7)?,
                            created_at: row.get(8)?,
                            updated_at: row.get(9)?,
                            document_content: row.get(11)?,
                        },
                        row.get::<_, String>(10)?,
                    ))
                },
            )
            .map_err(StorageError::Read)?;
        let mut hits = Vec::new();
        for row in rows {
            let (hit, container_tags) = row.map_err(StorageError::Read)?;
            let container_tags: Vec<String> = serde_json::from_str(&container_tags)
                .map_err(StorageError::DeserializeSearchData)?;
            if matches_search_options(
                &hit.document_id,
                hit.custom_id.as_deref(),
                &container_tags,
                hit.filepath.as_deref(),
                &hit.metadata,
                options,
            ) {
                hits.push(hit);
                if hits.len() == limit {
                    break;
                }
            }
        }
        Ok(hits)
    }

    /// Reconciles extracted memories, lineage, sources, and vectors in one transaction.
    ///
    /// # Errors
    /// Returns an error if proposals are malformed or persistence cannot commit atomically.
    pub fn reconcile_memories(
        &mut self,
        document_id: &str,
        container_tag: &str,
        proposals: &[MemoryProposal],
        model_id: &str,
        dimensions: usize,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let org_id = self.local_org_id.clone();
        self.reconcile_memories_for(
            &org_id,
            document_id,
            None,
            None,
            container_tag,
            proposals,
            model_id,
            dimensions,
        )
    }

    /// Reconciles extracted memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error if proposals or persistence are invalid.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "revision-gated proposal publication and completion share one transaction"
    )]
    pub fn reconcile_memories_for(
        &mut self,
        org_id: &str,
        document_id: &str,
        expected_revision: Option<i64>,
        completion_job_id: Option<&str>,
        container_tag: &str,
        proposals: &[MemoryProposal],
        model_id: &str,
        dimensions: usize,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        for proposal in proposals {
            validate_vector(&proposal.vector, dimensions)?;
            if proposal.temporary_id.is_empty() || proposal.content.trim().is_empty() {
                return Err(StorageError::InvalidMemoryProposal);
            }
        }
        let org_id = org_id.to_owned();
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let source_revision: Option<i64> = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1 AND org_id=?2",
                params![document_id, org_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(source_revision) = source_revision else {
            return Err(StorageError::UnknownMemorySource(document_id.to_owned()));
        };
        if let Some(expected_revision) = expected_revision
            && expected_revision != source_revision
        {
            return Err(StorageError::StaleRevision {
                document_id: document_id.to_owned(),
                claimed: expected_revision,
                current: Some(source_revision),
            });
        }
        let mut temporary_ids = HashMap::new();
        let mut record_ids = Vec::new();
        for proposal in proposals {
            if let Some(existing) =
                find_exact_memory(&tx, &org_id, container_tag, &proposal.content)?
            {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_sources (memory_id, document_id) VALUES (?1, ?2)",
                    params![existing.id, document_id],
                )
                .map_err(StorageError::Write)?;
                temporary_ids.insert(proposal.temporary_id.clone(), existing.id.clone());
                record_ids.push(existing.id);
                continue;
            }
            let parents = resolve_memory_parents(
                &tx,
                &org_id,
                container_tag,
                &proposal.parents,
                &temporary_ids,
            )?;
            let primary = parents.first();
            let id = format!("mem_{}", generate_id()?);
            let root_memory_id = primary.map(|parent| {
                parent
                    .root_memory_id
                    .clone()
                    .unwrap_or_else(|| parent.id.clone())
            });
            let parent_memory_id = primary.map(|parent| parent.id.clone());
            let version = primary.map_or(1, |parent| parent.version + 1);
            let is_inferred =
                proposal.is_inferred || parents.iter().any(|parent| parent.relation == "derives");
            let forget_after = valid_future_datetime(&tx, proposal.forget_after.as_deref())?;
            tx.execute(
                "INSERT INTO memories (id, org_id, container_tag, content, metadata, is_inferred, is_static, root_memory_id, parent_memory_id, version, forget_after, forget_reason) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![id, org_id, container_tag, proposal.content, json(&proposal.metadata)?, is_inferred, proposal.is_static, root_memory_id, parent_memory_id, version, forget_after, proposal.forget_reason],
            ).map_err(StorageError::Write)?;
            tx.execute(
                "INSERT INTO memory_sources (memory_id, document_id) VALUES (?1, ?2)",
                params![id, document_id],
            )
            .map_err(StorageError::Write)?;
            for parent in &parents {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_relations (parent_id, child_id, relation) VALUES (?1, ?2, ?3)",
                    params![parent.id, id, parent.relation],
                )
                .map_err(StorageError::Write)?;
                if parent.relation == "updates" {
                    tx.execute(
                        "UPDATE memories SET is_latest=0, is_static=0, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
                        [&parent.id],
                    )
                    .map_err(StorageError::Write)?;
                    tx.execute(
                        "UPDATE memory_embeddings SET active=0 WHERE memory_id=?1",
                        [&parent.id],
                    )
                    .map_err(StorageError::Write)?;
                }
            }
            tx.execute(
                "INSERT INTO memory_embeddings (memory_id, model_id, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                params![id, model_id, dimensions, vector_bytes(&proposal.vector)],
            )
            .map_err(StorageError::Write)?;
            temporary_ids.insert(proposal.temporary_id.clone(), id.clone());
            record_ids.push(id);
        }
        let records = record_ids
            .iter()
            .map(|id| read_memory_by_id(&tx, id))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(job_id) = completion_job_id {
            let jobs = tx.execute(
                "UPDATE jobs SET status='done', last_error_kind=NULL, last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND document_id=?2 AND revision=?3 AND status='extracting'",
                params![job_id, document_id, source_revision],
            ).map_err(StorageError::Write)?;
            let documents = tx.execute(
                "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![document_id, source_revision],
            ).map_err(StorageError::Write)?;
            if jobs != 1 || documents != 1 {
                return Err(StorageError::StaleRevision {
                    document_id: document_id.to_owned(),
                    claimed: source_revision,
                    current: None,
                });
            }
        }
        tx.commit().map_err(StorageError::Write)?;
        Ok(records)
    }

    /// Returns active static profile memories in recovered profile order.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn static_profile(&self, container_tag: &str) -> Result<Vec<MemoryRecord>, StorageError> {
        self.static_profile_for(&self.local_org_id, container_tag)
    }

    /// Returns static profile memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn static_profile_for(
        &self,
        org_id: &str,
        container_tag: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_static=1 AND is_latest=1 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP) ORDER BY (SELECT COUNT(*) FROM memory_sources WHERE memory_id=memories.id) DESC, updated_at DESC LIMIT 300",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(params![org_id, container_tag], read_memory)
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        Ok(deduplicate_memories(
            rows,
            100,
            &std::collections::HashSet::new(),
        ))
    }

    /// Returns active dynamic profile memories ordered by recency.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn dynamic_profile(
        &self,
        container_tag: &str,
        static_memories: &[MemoryRecord],
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.dynamic_profile_for(&self.local_org_id, container_tag, static_memories)
    }

    /// Returns dynamic profile memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn dynamic_profile_for(
        &self,
        org_id: &str,
        container_tag: &str,
        static_memories: &[MemoryRecord],
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_static=0 AND is_latest=1 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP) ORDER BY updated_at DESC LIMIT 300",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(params![org_id, container_tag], read_memory)
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        let excluded = static_memories
            .iter()
            .map(|memory| normalized_memory(&memory.memory))
            .collect();
        Ok(deduplicate_memories(rows, 100, &excluded))
    }

    /// Returns memories assigned to a profile bucket.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn bucket_profile(
        &self,
        container_tag: &str,
        bucket: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.bucket_profile_for(&self.local_org_id, container_tag, bucket)
    }

    /// Returns bucket memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn bucket_profile_for(
        &self,
        org_id: &str,
        container_tag: &str,
        bucket: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let static_memories = self.static_profile_for(org_id, container_tag)?;
        let dynamic = self.dynamic_profile_for(org_id, container_tag, &static_memories)?;
        Ok(static_memories
            .into_iter()
            .chain(dynamic)
            .filter(|memory| {
                memory
                    .metadata
                    .get("buckets")
                    .and_then(Value::as_array)
                    .is_some_and(|buckets| {
                        buckets.iter().any(|value| value.as_str() == Some(bucket))
                    })
            })
            .take(100)
            .collect())
    }

    /// Soft-forgets one active memory while preserving graph and source history.
    ///
    /// # Errors
    /// Returns an error if the memory is absent or the transition cannot commit.
    pub fn forget_memory(
        &mut self,
        id: Option<&str>,
        content: Option<&str>,
        container_tag: &str,
        reason: Option<&str>,
    ) -> Result<String, StorageError> {
        let org_id = self.local_org_id.clone();
        self.forget_memory_for(&org_id, id, content, container_tag, reason)
    }

    /// Soft-forgets one active memory in an explicit organization.
    ///
    /// # Errors
    /// Returns an error if the memory is absent or persistence fails.
    pub fn forget_memory_for(
        &mut self,
        org_id: &str,
        id: Option<&str>,
        content: Option<&str>,
        container_tag: &str,
        reason: Option<&str>,
    ) -> Result<String, StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let memory_id: Option<String> = if let Some(id) = id {
            tx.query_row(
                "SELECT id FROM memories WHERE id=?1 AND org_id=?2 AND container_tag=?3 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP)",
                params![id, org_id, container_tag],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?
        } else if let Some(content) = content {
            tx.query_row(
                "SELECT id FROM memories WHERE content=?1 AND org_id=?2 AND container_tag=?3 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP) ORDER BY rowid LIMIT 1",
                params![content, org_id, container_tag],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?
        } else {
            None
        };
        let memory_id = memory_id.ok_or(StorageError::MemoryNotFound)?;
        tx.execute(
            "UPDATE memories SET is_forgotten=1, is_latest=0, is_static=0, forget_after=CURRENT_TIMESTAMP, forget_reason=?2, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
            params![memory_id, reason.unwrap_or("user_requested")],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE memory_embeddings SET active=0 WHERE memory_id=?1",
            [&memory_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(memory_id)
    }

    /// Searches canonical memories with exact cosine scoring.
    ///
    /// # Errors
    /// Returns an error for malformed vectors or stored metadata.
    pub fn search_memories(
        &self,
        query: &[f32],
        model_id: &str,
        container_tag: &str,
        limit: usize,
        threshold: f32,
        include_forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        self.search_memories_for(
            &self.local_org_id,
            query,
            model_id,
            container_tag,
            limit,
            threshold,
            include_forgotten,
        )
    }

    /// Searches memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error for malformed stored vectors or metadata.
    #[expect(clippy::too_many_arguments, reason = "search contract parameters")]
    pub fn search_memories_for(
        &self,
        org_id: &str,
        query: &[f32],
        model_id: &str,
        container_tag: &str,
        limit: usize,
        threshold: f32,
        include_forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        validate_vector(query, query.len())?;
        let query_bytes = vector_bytes(query);
        let mut statement = self.connection.prepare(
            "SELECT memories.id, memories.content, memories.metadata, memories.is_inferred, memories.is_static, memories.is_latest, memories.is_forgotten, memories.root_memory_id, memories.parent_memory_id, memories.version, memories.forget_after, memories.forget_reason, memories.created_at, memories.updated_at, cosine_similarity(memory_embeddings.vector, ?1, ?2) AS similarity FROM memory_embeddings JOIN memories ON memories.id=memory_embeddings.memory_id WHERE memories.org_id=?3 AND memories.container_tag=?4 AND memory_embeddings.model_id=?5 AND memory_embeddings.dimensions=?2 AND (memory_embeddings.active=1 OR ?6=1) AND (memories.is_latest=1 OR ?6=1) AND (memories.is_forgotten=0 OR ?6=1) AND (memories.forget_after IS NULL OR datetime(memories.forget_after)>CURRENT_TIMESTAMP OR ?6=1) AND cosine_similarity(memory_embeddings.vector, ?1, ?2)>=?7 ORDER BY similarity DESC, memories.id LIMIT ?8",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(
                params![
                    query_bytes,
                    query.len(),
                    org_id,
                    container_tag,
                    model_id,
                    include_forgotten,
                    threshold,
                    limit
                ],
                |row| {
                    Ok(MemorySearchHit {
                        record: read_memory(row)?,
                        similarity: row.get(14)?,
                        parents: Vec::new(),
                        children: Vec::new(),
                        related: Vec::new(),
                        documents: Vec::new(),
                    })
                },
            )
            .map_err(StorageError::Read)?;
        let mut hits = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        hydrate_memory_hits(&self.connection, &mut hits)?;
        Ok(hits)
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

    /// Imports a stable JSONL export from the read-only legacy sidecar.
    ///
    /// Existing IDs are retained and duplicate rows make the operation idempotent.
    ///
    /// # Errors
    /// Returns an error and rolls back the complete import if parsing or validation fails.
    pub fn import_legacy_export(
        &mut self,
        path: &Path,
    ) -> Result<LegacyImportReport, StorageError> {
        let file = std::fs::File::open(path).map_err(|source| StorageError::ReadLegacyExport {
            path: path.to_path_buf(),
            source,
        })?;
        let mut tables: HashMap<String, Vec<Map<String, Value>>> = HashMap::new();
        let mut complete = false;
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|source| StorageError::ReadLegacyExport {
                path: path.to_path_buf(),
                source,
            })?;
            let value: Value = serde_json::from_str(&line).map_err(|source| {
                StorageError::MalformedLegacyExport {
                    line: index + 1,
                    source,
                }
            })?;
            match value.get("type").and_then(Value::as_str) {
                Some("row") => {
                    let table = required_string(&value, "table")?.to_owned();
                    let row = value
                        .get("row")
                        .and_then(Value::as_object)
                        .cloned()
                        .ok_or(StorageError::InvalidLegacyRow { line: index + 1 })?;
                    tables.entry(table).or_default().push(row);
                }
                Some("complete") => complete = true,
                Some("manifest" | "table") => {}
                _ => return Err(StorageError::InvalidLegacyRow { line: index + 1 }),
            }
        }
        if !complete {
            return Err(StorageError::IncompleteLegacyExport);
        }
        self.import_legacy_tables(&tables)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "foreign-key import order is kept visible in one transaction"
    )]
    fn import_legacy_tables(
        &mut self,
        tables: &HashMap<String, Vec<Map<String, Value>>>,
    ) -> Result<LegacyImportReport, StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let mut report = LegacyImportReport::default();
        for row in table_rows(tables, "organization") {
            let id = field(row, "id")?;
            let slug = field(row, "slug")?;
            tx.execute(
                "UPDATE organizations SET slug=slug || '-rs-' || substr(id, 1, 6) WHERE slug=?1 AND id<>?2 AND id=?3 AND NOT EXISTS(SELECT 1 FROM documents WHERE org_id=organizations.id)",
                params![slug, id, self.local_org_id],
            ).map_err(StorageError::Write)?;
            report.organizations += tx.execute(
                "INSERT OR IGNORE INTO organizations (id, slug, created_at, name, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, slug, field(row, "created_at")?, optional_field(row, "name"), json_field(row, "metadata", &json!({}))?],
            ).map_err(StorageError::Write)?;
        }
        let mut spaces = HashMap::new();
        for row in table_rows(tables, "space") {
            let id = field(row, "id")?.to_owned();
            let container_tag = field(row, "container_tag")?.to_owned();
            spaces.insert(id.clone(), container_tag.clone());
            report.spaces += tx.execute(
                "INSERT OR IGNORE INTO spaces (id, org_id, container_tag, entity_context, name, description, metadata, profile_buckets, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![id, field(row, "org_id")?, container_tag, optional_field(row, "entity_context"), optional_field(row, "name"), optional_field(row, "description"), json_field(row, "metadata", &json!({}))?, json_field(row, "profile_buckets", &json!([]))?, field(row, "created_at")?, field(row, "updated_at")?],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "document") {
            let mut metadata = object_field(row, "metadata")?;
            copy_optional_metadata(row, &mut metadata, "title");
            copy_optional_metadata(row, &mut metadata, "summary");
            copy_optional_metadata(row, &mut metadata, "type");
            let status = compatible_document_status(optional_field(row, "status"));
            let task_type = match optional_field(row, "task_type") {
                Some("superrag") => "superrag",
                _ => "memory",
            };
            report.documents += tx.execute(
                "INSERT OR IGNORE INTO documents (id, org_id, content, content_hash, custom_id, status, container_tags, entity_context, metadata, task_type, filepath, filter_by_metadata, dreaming, created_at, updated_at, title, summary, document_type, source, url, user_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9, ?10, '{}', 'dynamic', ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
                params![field(row, "id")?, field(row, "org_id")?, optional_field(row, "content").unwrap_or_default(), field(row, "content_hash")?, optional_field(row, "custom_id"), status, json_field(row, "container_tags", &json!(["sm_project_default"]))?, json(&metadata)?, task_type, optional_field(row, "filepath"), field(row, "created_at")?, field(row, "updated_at")?, optional_field(row, "title"), optional_field(row, "summary"), optional_field(row, "type"), optional_field(row, "source"), optional_field(row, "url"), optional_field(row, "user_id")],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "chunk") {
            let changed = tx.execute(
                "INSERT OR IGNORE INTO document_chunks (document_id, ordinal, content, stable_id) VALUES (?1, ?2, ?3, ?4)",
                params![field(row, "document_id")?, integer_field(row, "position")?, field(row, "content")?, field(row, "id")?],
            ).map_err(StorageError::Write)?;
            report.chunks += changed;
            if changed == 1 {
                if let Some(vector) = vector_field(row, "embedding")? {
                    let chunk_id = tx.last_insert_rowid();
                    tx.execute(
                        "INSERT INTO chunk_embeddings (chunk_id, model_id, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                        params![chunk_id, optional_field(row, "embedding_model").unwrap_or("legacy"), vector.len(), vector_bytes(&vector)],
                    ).map_err(StorageError::Write)?;
                }
            }
        }
        let mut pending_parents = Vec::new();
        for row in table_rows(tables, "memory_entry") {
            let id = field(row, "id")?.to_owned();
            let container_tag = optional_field(row, "space_id")
                .and_then(|space| spaces.get(space))
                .cloned()
                .unwrap_or_else(|| "sm_project_default".to_owned());
            let mut metadata = object_field(row, "metadata")?;
            metadata.insert(
                "buckets".to_owned(),
                value_field(row, "buckets")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            );
            report.memories += tx.execute(
                "INSERT OR IGNORE INTO memories (id, org_id, container_tag, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at, source_count) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![id, field(row, "org_id")?, container_tag, field(row, "memory")?, json(&metadata)?, bool_field(row, "is_inference"), bool_field(row, "is_static"), bool_field(row, "is_latest"), bool_field(row, "is_forgotten"), optional_field(row, "root_memory_id"), integer_field(row, "version")?.max(1), optional_field(row, "forget_after"), optional_field(row, "forget_reason"), field(row, "created_at")?, field(row, "updated_at")?, integer_field(row, "source_count")?.max(0)],
            ).map_err(StorageError::Write)?;
            if let Some(parent) = optional_field(row, "parent_memory_id") {
                pending_parents.push((id.clone(), parent.to_owned()));
            }
            if let Some(vector) = vector_field(row, "memory_embedding")? {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_embeddings (memory_id, model_id, dimensions, vector, active) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![id, optional_field(row, "memory_embedding_model").unwrap_or("legacy"), vector.len(), vector_bytes(&vector), bool_field(row, "is_latest") && !bool_field(row, "is_forgotten")],
                ).map_err(StorageError::Write)?;
            }
        }
        for (id, parent) in pending_parents {
            tx.execute(
                "UPDATE memories SET parent_memory_id=?2 WHERE id=?1 AND EXISTS(SELECT 1 FROM memories WHERE id=?2)",
                params![id, parent],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "memory_relations") {
            let relation = compatible_relation(field(row, "relation")?);
            report.relations += tx.execute(
                "INSERT OR IGNORE INTO memory_relations (parent_id, child_id, relation) SELECT ?1, ?2, ?3 WHERE EXISTS(SELECT 1 FROM memories WHERE id=?1) AND EXISTS(SELECT 1 FROM memories WHERE id=?2)",
                params![field(row, "from_id")?, field(row, "to_id")?, relation],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "memory_document_source") {
            report.sources += tx.execute(
                "INSERT OR IGNORE INTO memory_sources (memory_id, document_id, created_at) SELECT ?1, ?2, ?3 WHERE EXISTS(SELECT 1 FROM memories WHERE id=?1) AND EXISTS(SELECT 1 FROM documents WHERE id=?2)",
                params![field(row, "memory_entry_id")?, field(row, "document_id")?, field(row, "added_at")?],
            ).map_err(StorageError::Write)?;
        }
        let member_organizations = table_rows(tables, "member")
            .iter()
            .filter_map(|row| {
                optional_field(row, "user_id")
                    .zip(optional_field(row, "organization_id"))
                    .map(|(user, org)| (user.to_owned(), org.to_owned()))
            })
            .collect::<HashMap<_, _>>();
        let imported_organizations = table_rows(tables, "organization")
            .iter()
            .filter_map(|row| optional_field(row, "id"))
            .collect::<std::collections::HashSet<_>>();
        for row in table_rows(tables, "apikey") {
            let Some(key) = optional_field(row, "key") else {
                continue;
            };
            let organization_id = optional_field(row, "reference_id")
                .and_then(|reference| member_organizations.get(reference))
                .filter(|organization| imported_organizations.contains(organization.as_str()));
            report.api_keys += tx.execute(
                "INSERT OR IGNORE INTO api_keys (id, org_id, key_hash, name, enabled, expires_at, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![field(row, "id")?, organization_id, Sha256::digest(key.as_bytes()).as_slice(), optional_field(row, "name"), bool_field(row, "enabled"), optional_field(row, "expires_at"), field(row, "created_at")?, optional_field(row, "updated_at")],
            ).map_err(StorageError::Write)?;
        }
        tx.commit().map_err(StorageError::Write)?;
        Ok(report)
    }

    /// Returns active imported API-key hashes for authentication.
    ///
    /// # Errors
    /// Returns an error if the key store cannot be read.
    pub fn api_key_hashes(&self) -> Result<Vec<[u8; 32]>, StorageError> {
        Ok(self
            .api_key_identities()?
            .into_iter()
            .map(|(hash, _)| hash)
            .collect())
    }

    /// Returns active imported API-key hashes with organization identity.
    ///
    /// # Errors
    /// Returns an error if a key hash is malformed or cannot be read.
    pub fn api_key_identities(&self) -> Result<Vec<ApiKeyIdentity>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT key_hash, org_id FROM api_keys WHERE enabled=1 AND (expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)",
        ).map_err(StorageError::Read)?;
        statement
            .query_map([], |row| {
                let bytes: Vec<u8> = row.get(0)?;
                let hash = bytes.try_into().map_err(|bytes: Vec<u8>| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Blob,
                        format!("API key hash has {} bytes, expected 32", bytes.len()).into(),
                    )
                })?;
                Ok((hash, row.get(1)?))
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Returns whether a legacy snapshot fingerprint was fully imported.
    ///
    /// # Errors
    /// Returns an error if the migration ledger cannot be read.
    pub fn has_legacy_import(&self, source_hash: &str) -> Result<bool, StorageError> {
        self.connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM legacy_imports WHERE source_hash=?1)",
                [source_hash],
                |row| row.get(0),
            )
            .map_err(StorageError::Read)
    }

    /// Records a completed, validated legacy import fingerprint.
    ///
    /// # Errors
    /// Returns an error if the completion marker cannot be persisted.
    pub fn record_legacy_import(
        &mut self,
        source_hash: &str,
        report: &LegacyImportReport,
    ) -> Result<(), StorageError> {
        self.connection
            .execute(
                "INSERT OR IGNORE INTO legacy_imports (source_hash, report) VALUES (?1, ?2)",
                params![source_hash, format!("{report:?}")],
            )
            .map(|_| ())
            .map_err(StorageError::Write)
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
    let active: bool = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs WHERE document_id=?1 AND revision=(SELECT revision FROM documents WHERE id=?1) AND kind='document' AND status IN ('queued','extracting','chunking','embedding','indexing'))",
            [id.as_str()],
            |row| row.get(0),
        )
        .map_err(StorageError::Read)?;
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
        "UPDATE documents SET content=?2, content_hash=?3, custom_id=?4, status=?5, container_tags=?6, entity_context=?7, metadata=?8, task_type=?9, filepath=?10, filter_by_metadata=?11, dreaming=?12, revision=revision+1, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
        params![id, input.content, hash, input.custom_id, status, json(&input.container_tags)?, input.entity_context, json(metadata)?, input.task_type, input.filepath, json(&input.filter_by_metadata)?, input.dreaming],
    ).map_err(StorageError::Write)?;
    Ok(())
}

fn enqueue_unless_active(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<bool, StorageError> {
    let active: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM jobs WHERE document_id=?1 AND revision=(SELECT revision FROM documents WHERE id=?1) AND kind='document' AND status IN ('queued','extracting','chunking','embedding','indexing'))",
        [document_id], |row| row.get(0),
    ).map_err(StorageError::Read)?;
    if active {
        return Ok(false);
    }
    let revision: i64 = tx
        .query_row(
            "SELECT revision FROM documents WHERE id=?1",
            [document_id],
            |row| row.get(0),
        )
        .map_err(StorageError::Read)?;
    tx.execute(
        "INSERT INTO jobs (id, document_id, kind, status, revision) VALUES (?1, ?2, 'document', 'queued', ?3)",
        params![generate_id()?, document_id, revision],
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

fn table_rows<'a>(
    tables: &'a HashMap<String, Vec<Map<String, Value>>>,
    table: &str,
) -> &'a [Map<String, Value>] {
    tables.get(table).map_or(&[], Vec::as_slice)
}

fn value_field<'a>(row: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    row.get(key).filter(|value| !value.is_null())
}

fn optional_field<'a>(row: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    value_field(row, key).and_then(Value::as_str)
}

fn field<'a>(row: &'a Map<String, Value>, key: &str) -> Result<&'a str, StorageError> {
    optional_field(row, key).ok_or_else(|| StorageError::MissingLegacyField(key.to_owned()))
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, StorageError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| StorageError::MissingLegacyField(key.to_owned()))
}

fn integer_field(row: &Map<String, Value>, key: &str) -> Result<i64, StorageError> {
    value_field(row, key)
        .and_then(|value| {
            value.as_i64().or_else(|| {
                value
                    .get("$bigint")
                    .and_then(Value::as_str)
                    .and_then(|value| value.parse().ok())
            })
        })
        .ok_or_else(|| StorageError::MissingLegacyField(key.to_owned()))
}

fn bool_field(row: &Map<String, Value>, key: &str) -> bool {
    value_field(row, key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn object_field(row: &Map<String, Value>, key: &str) -> Result<Map<String, Value>, StorageError> {
    match value_field(row, key) {
        None => Ok(Map::new()),
        Some(Value::Object(value)) => Ok(value.clone()),
        Some(Value::String(value)) => {
            serde_json::from_str(value).map_err(StorageError::DeserializeSearchData)
        }
        Some(_) => Err(StorageError::InvalidLegacyField(key.to_owned())),
    }
}

fn json_field(
    row: &Map<String, Value>,
    key: &str,
    default: &Value,
) -> Result<String, StorageError> {
    json(value_field(row, key).unwrap_or(default))
}

fn copy_optional_metadata(row: &Map<String, Value>, metadata: &mut Map<String, Value>, key: &str) {
    if let Some(value) = value_field(row, key) {
        metadata.insert(key.to_owned(), value.clone());
    }
}

fn vector_field(row: &Map<String, Value>, key: &str) -> Result<Option<Vec<f32>>, StorageError> {
    let Some(value) = value_field(row, key) else {
        return Ok(None);
    };
    let values = if let Some(values) = value.as_array() {
        values.clone()
    } else if let Some(values) = value.get("$vector").and_then(Value::as_array) {
        values.clone()
    } else if let Some(value) = value.as_str() {
        serde_json::from_str::<Vec<Value>>(value).map_err(StorageError::DeserializeSearchData)?
    } else {
        return Err(StorageError::InvalidLegacyField(key.to_owned()));
    };
    let vector = values
        .into_iter()
        .map(|value| {
            value
                .as_f64()
                .and_then(|value| value.to_string().parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .ok_or_else(|| StorageError::InvalidLegacyField(key.to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(vector))
}

fn compatible_document_status(status: Option<&str>) -> &'static str {
    match status {
        Some("unknown") => "unknown",
        Some("queued") => "queued",
        Some("extracting") => "extracting",
        Some("chunking") => "chunking",
        Some("embedding") => "embedding",
        Some("indexing") => "indexing",
        Some("failed") => "failed",
        _ => "done",
    }
}

fn compatible_relation(relation: &str) -> &'static str {
    match relation {
        "updates" => "updates",
        "derives" => "derives",
        _ => "extends",
    }
}

fn validate_vector(vector: &[f32], dimensions: usize) -> Result<(), StorageError> {
    if vector.len() != dimensions {
        return Err(StorageError::InvalidVectorLength {
            expected: dimensions,
            actual: vector.len(),
        });
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(StorageError::NonFiniteVector);
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 1e-4 {
        return Err(StorageError::InvalidVectorNorm { norm });
    }
    Ok(())
}

fn vector_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn matches_search_options(
    document_id: &str,
    custom_id: Option<&str>,
    container_tags: &[String],
    filepath: Option<&str>,
    metadata: &Map<String, Value>,
    options: &SearchOptions,
) -> bool {
    if options
        .document_id
        .as_deref()
        .is_some_and(|wanted| wanted != document_id && Some(wanted) != custom_id)
    {
        return false;
    }
    if !options.container_tags.is_empty()
        && !options
            .container_tags
            .iter()
            .any(|wanted| container_tags.contains(wanted))
    {
        return false;
    }
    if options.filepath.as_deref().is_some_and(|wanted| {
        if let Some(prefix) = wanted.strip_suffix('/') {
            !filepath.is_some_and(|actual| actual.starts_with(prefix))
        } else {
            filepath != Some(wanted)
        }
    }) {
        return false;
    }
    options
        .filters
        .as_ref()
        .is_none_or(|filter| matches_filter(filter, metadata))
}

struct ResolvedParent {
    id: String,
    relation: String,
    root_memory_id: Option<String>,
    version: i64,
}

fn resolve_memory_parents(
    connection: &Connection,
    org_id: &str,
    container_tag: &str,
    parents: &[MemoryParent],
    temporary_ids: &HashMap<String, String>,
) -> Result<Vec<ResolvedParent>, StorageError> {
    let mut result = Vec::new();
    for parent in parents {
        let id = temporary_ids
            .get(&parent.memory_id)
            .map_or(parent.memory_id.as_str(), String::as_str);
        let resolved = connection
            .query_row(
                "SELECT id, root_memory_id, version FROM memories WHERE id=?1 AND org_id=?2 AND container_tag=?3",
                params![id, org_id, container_tag],
                |row| {
                    Ok(ResolvedParent {
                        id: row.get(0)?,
                        relation: parent.relation.clone(),
                        root_memory_id: row.get(1)?,
                        version: row.get(2)?,
                    })
                },
            )
            .optional()
            .map_err(StorageError::Read)?;
        if let Some(resolved) = resolved {
            result.push(resolved);
        }
    }
    Ok(result)
}

fn find_exact_memory(
    connection: &Connection,
    org_id: &str,
    container_tag: &str,
    content: &str,
) -> Result<Option<MemoryRecord>, StorageError> {
    let wanted = normalized_memory(content);
    let mut statement = connection.prepare(
        "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_latest=1 AND is_forgotten=0 ORDER BY rowid",
    ).map_err(StorageError::Read)?;
    let rows = statement
        .query_map(params![org_id, container_tag], read_memory)
        .map_err(StorageError::Read)?;
    for row in rows {
        let memory = row.map_err(StorageError::Read)?;
        if normalized_memory(&memory.memory) == wanted {
            return Ok(Some(memory));
        }
    }
    Ok(None)
}

fn read_memory_by_id(connection: &Connection, id: &str) -> Result<MemoryRecord, StorageError> {
    connection
        .query_row(
            "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE id=?1",
            [id],
            read_memory,
        )
        .map_err(StorageError::Read)
}

fn read_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let metadata: String = row.get(2)?;
    Ok(MemoryRecord {
        id: row.get(0)?,
        memory: row.get(1)?,
        metadata: parse_json(&metadata, 2)?,
        is_inferred: row.get(3)?,
        is_static: row.get(4)?,
        is_latest: row.get(5)?,
        is_forgotten: row.get(6)?,
        root_memory_id: row.get(7)?,
        parent_memory_id: row.get(8)?,
        version: row.get(9)?,
        forget_after: row.get(10)?,
        forget_reason: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

fn hydrate_memory_hits(
    connection: &Connection,
    hits: &mut [MemorySearchHit],
) -> Result<(), StorageError> {
    if hits.is_empty() {
        return Ok(());
    }
    let ids = hits
        .iter()
        .map(|hit| hit.record.id.clone())
        .collect::<Vec<_>>();
    let placeholders = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    for (parents, owner_column, related_column) in [
        (true, "child_id", "parent_id"),
        (false, "parent_id", "child_id"),
    ] {
        let sql = format!(
            "SELECT memory_relations.{owner_column}, memory_relations.relation, memories.version, memories.content, memories.metadata, memories.updated_at FROM memory_relations JOIN memories ON memories.id=memory_relations.{related_column} WHERE memory_relations.{owner_column} IN ({placeholders}) ORDER BY memories.version, memories.updated_at, memories.id"
        );
        let mut statement = connection.prepare(&sql).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                let metadata: String = row.get(4)?;
                Ok((
                    row.get::<_, String>(0)?,
                    MemoryRelationHit {
                        relation: row.get(1)?,
                        version: row.get(2)?,
                        memory: row.get(3)?,
                        metadata: parse_json(&metadata, 4)?,
                        updated_at: row.get(5)?,
                    },
                ))
            })
            .map_err(StorageError::Read)?;
        for row in rows {
            let (owner, relation) = row.map_err(StorageError::Read)?;
            if let Some(hit) = hits.iter_mut().find(|hit| hit.record.id == owner) {
                if matches!(relation.relation.as_str(), "extends" | "derives") {
                    hit.related.push(relation.clone());
                }
                if parents {
                    hit.parents.push(relation);
                } else {
                    hit.children.push(relation);
                }
            }
        }
    }
    let sql = format!(
        "SELECT memory_sources.memory_id, COALESCE(documents.custom_id, documents.id), documents.title, documents.document_type, documents.metadata, documents.summary, documents.created_at, documents.updated_at FROM memory_sources JOIN documents ON documents.id=memory_sources.document_id WHERE memory_sources.memory_id IN ({placeholders}) ORDER BY memory_sources.created_at, documents.id"
    );
    let mut statement = connection.prepare(&sql).map_err(StorageError::Read)?;
    let rows = statement
        .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
            let metadata: String = row.get(4)?;
            Ok((
                row.get::<_, String>(0)?,
                MemorySourceDocument {
                    id: row.get(1)?,
                    title: row.get(2)?,
                    document_type: row.get(3)?,
                    metadata: parse_json(&metadata, 4)?,
                    summary: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                },
            ))
        })
        .map_err(StorageError::Read)?;
    for row in rows {
        let (owner, document) = row.map_err(StorageError::Read)?;
        if let Some(hit) = hits.iter_mut().find(|hit| hit.record.id == owner) {
            hit.documents.push(document);
        }
    }
    Ok(())
}

fn valid_future_datetime(
    connection: &Connection,
    value: Option<&str>,
) -> Result<Option<String>, StorageError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let valid: bool = connection
        .query_row(
            "SELECT datetime(?1) IS NOT NULL AND datetime(?1) > CURRENT_TIMESTAMP",
            [value],
            |row| row.get(0),
        )
        .map_err(StorageError::Read)?;
    Ok(valid.then(|| value.to_owned()))
}

fn deduplicate_memories(
    memories: Vec<MemoryRecord>,
    limit: usize,
    excluded: &std::collections::HashSet<String>,
) -> Vec<MemoryRecord> {
    let mut seen = excluded.clone();
    memories
        .into_iter()
        .filter(|memory| {
            let normalized = normalized_memory(&memory.memory);
            !normalized.is_empty() && seen.insert(normalized)
        })
        .take(limit)
        .collect()
}

fn normalized_memory(memory: &str) -> String {
    let mut normalized = String::new();
    let mut whitespace = false;
    for character in memory.to_lowercase().chars() {
        if character.is_alphanumeric() || character == '_' {
            normalized.push(character);
            whitespace = false;
        } else if character.is_whitespace() && !whitespace && !normalized.is_empty() {
            normalized.push(' ');
            whitespace = true;
        }
    }
    normalized.trim().to_owned()
}

fn matches_filter(filter: &FilterExpression, metadata: &Map<String, Value>) -> bool {
    match filter {
        FilterExpression::And(filters) => filters
            .iter()
            .all(|filter| matches_filter(filter, metadata)),
        FilterExpression::Or(filters) => filters
            .iter()
            .any(|filter| matches_filter(filter, metadata)),
        FilterExpression::Condition(condition) => {
            let matched = metadata
                .get(&condition.key)
                .is_some_and(|value| matches_condition(condition, value));
            matched != condition.negate
        }
    }
}

fn matches_condition(condition: &FilterCondition, value: &Value) -> bool {
    match condition.kind {
        FilterKind::Metadata => comparable_text(value)
            .is_some_and(|actual| equal_text(&actual, &condition.value, condition.ignore_case)),
        FilterKind::StringContains => comparable_text(value)
            .is_some_and(|actual| contains_text(&actual, &condition.value, condition.ignore_case)),
        FilterKind::ArrayContains => value.as_array().is_some_and(|values| {
            values.iter().any(|value| {
                comparable_text(value).is_some_and(|actual| {
                    equal_text(&actual, &condition.value, condition.ignore_case)
                })
            })
        }),
        FilterKind::Numeric => value.as_f64().is_some_and(|actual| {
            condition.value.parse::<f64>().ok().is_some_and(|wanted| {
                match condition.numeric_operator {
                    NumericOperator::Greater => actual > wanted,
                    NumericOperator::Less => actual < wanted,
                    NumericOperator::GreaterOrEqual => actual >= wanted,
                    NumericOperator::LessOrEqual => actual <= wanted,
                    NumericOperator::Equal => actual.total_cmp(&wanted).is_eq(),
                }
            })
        }),
    }
}

fn comparable_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn equal_text(actual: &str, wanted: &str, ignore_case: bool) -> bool {
    if ignore_case {
        actual.to_lowercase() == wanted.to_lowercase()
    } else {
        actual == wanted
    }
}

fn contains_text(actual: &str, wanted: &str, ignore_case: bool) -> bool {
    if ignore_case {
        actual.to_lowercase().contains(&wanted.to_lowercase())
    } else {
        actual.contains(wanted)
    }
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
    register_vector_functions(connection)?;
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
    validate_existing_version(&tx, current)?;
    for &(version, sql) in MIGRATIONS.iter().filter(|(version, _)| *version > current) {
        tx.execute_batch(sql).map_err(StorageError::Migrate)?;
        tx.execute(
            "INSERT INTO schema_migrations (version) VALUES (?1)",
            [version],
        )
        .map_err(StorageError::Migrate)?;
    }
    validate_current_schema(&tx)?;
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
        "UPDATE documents SET status='queued', updated_at=CURRENT_TIMESTAMP WHERE id IN (SELECT document_id FROM jobs WHERE kind='document' AND status='queued') AND status IN ('extracting','chunking','embedding','indexing')",
        [],
    )
    .map_err(StorageError::Migrate)?;
    tx.execute(
        "UPDATE documents SET status='indexing', updated_at=CURRENT_TIMESTAMP WHERE id IN (SELECT document_id FROM jobs WHERE kind='memory' AND status='queued') AND status IN ('extracting','chunking','embedding','indexing')",
        [],
    )
    .map_err(StorageError::Migrate)?;
    tx.commit().map_err(StorageError::Migrate)?;
    Ok(org_id)
}

#[expect(
    clippy::too_many_lines,
    reason = "exact migration column contracts are intentionally explicit"
)]
fn validate_current_schema(connection: &Connection) -> Result<(), StorageError> {
    validate_columns(
        connection,
        "organizations",
        &["id", "slug", "created_at", "name", "metadata"],
    )?;
    validate_columns(
        connection,
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
            "revision",
            "title",
            "summary",
            "document_type",
            "source",
            "url",
            "user_id",
        ],
    )?;
    validate_columns(
        connection,
        "document_chunks",
        &["id", "document_id", "ordinal", "content", "stable_id"],
    )?;
    validate_columns(
        connection,
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
            "revision",
            "extraction_result",
            "last_error_kind",
        ],
    )?;
    validate_columns(
        connection,
        "chunk_embeddings",
        &["chunk_id", "model_id", "dimensions", "vector", "created_at"],
    )?;
    validate_columns(
        connection,
        "memories",
        &[
            "id",
            "org_id",
            "container_tag",
            "content",
            "metadata",
            "is_inferred",
            "is_static",
            "is_latest",
            "is_forgotten",
            "root_memory_id",
            "parent_memory_id",
            "version",
            "forget_after",
            "forget_reason",
            "created_at",
            "updated_at",
            "source_count",
        ],
    )?;
    validate_columns(
        connection,
        "memory_sources",
        &["memory_id", "document_id", "created_at"],
    )?;
    validate_columns(
        connection,
        "memory_relations",
        &["parent_id", "child_id", "relation", "created_at"],
    )?;
    validate_columns(
        connection,
        "memory_embeddings",
        &[
            "memory_id",
            "model_id",
            "dimensions",
            "vector",
            "active",
            "created_at",
        ],
    )?;
    validate_columns(
        connection,
        "spaces",
        &[
            "id",
            "org_id",
            "container_tag",
            "entity_context",
            "name",
            "description",
            "metadata",
            "profile_buckets",
            "created_at",
            "updated_at",
        ],
    )?;
    validate_columns(
        connection,
        "api_keys",
        &[
            "id",
            "org_id",
            "key_hash",
            "name",
            "enabled",
            "expires_at",
            "created_at",
            "updated_at",
        ],
    )?;
    validate_columns(
        connection,
        "legacy_imports",
        &["source_hash", "imported_at", "report"],
    )
}

fn validate_existing_version(connection: &Connection, version: i64) -> Result<(), StorageError> {
    if (1..5).contains(&version) {
        validate_columns(connection, "organizations", &["id", "slug", "created_at"])?;
    }
    if (1..3).contains(&version) {
        validate_columns(
            connection,
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
            connection,
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
    }
    if version == 2 {
        validate_columns(
            connection,
            "document_chunks",
            &["id", "document_id", "ordinal", "content"],
        )?;
    }
    Ok(())
}

fn register_vector_functions(connection: &Connection) -> Result<(), StorageError> {
    connection
        .create_scalar_function(
            "cosine_similarity",
            3,
            FunctionFlags::SQLITE_DETERMINISTIC | FunctionFlags::SQLITE_INNOCUOUS,
            |context| {
                let left = context.get_raw(0).as_blob()?;
                let right = context.get_raw(1).as_blob()?;
                let dimensions = usize::try_from(context.get::<i64>(2)?)
                    .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))?;
                let expected = dimensions.saturating_mul(std::mem::size_of::<f32>());
                if left.len() != expected || right.len() != expected {
                    return Err(rusqlite::Error::UserFunctionError(
                        format!(
                            "vector byte length mismatch: expected {expected}, found {} and {}",
                            left.len(),
                            right.len()
                        )
                        .into(),
                    ));
                }
                let (dot, left_norm, right_norm) =
                    left.chunks_exact(4).zip(right.chunks_exact(4)).try_fold(
                        (0.0, 0.0, 0.0),
                        |(dot, left_norm, right_norm), (left, right)| {
                            let left =
                                f64::from(f32::from_le_bytes([left[0], left[1], left[2], left[3]]));
                            let right = f64::from(f32::from_le_bytes([
                                right[0], right[1], right[2], right[3],
                            ]));
                            if !left.is_finite() || !right.is_finite() {
                                return Err(rusqlite::Error::UserFunctionError(
                                    "vector contains a non-finite component".into(),
                                ));
                            }
                            Ok((
                                dot + left * right,
                                left_norm + left * left,
                                right_norm + right * right,
                            ))
                        },
                    )?;
                let norm = left_norm.sqrt() * right_norm.sqrt();
                if norm <= f64::EPSILON {
                    return Err(rusqlite::Error::UserFunctionError(
                        "vector has zero norm".into(),
                    ));
                }
                Ok(dot / norm)
            },
        )
        .map_err(StorageError::Configure)
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
    #[error("failed to read legacy export {path}: {source}")]
    ReadLegacyExport {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("legacy export contains invalid JSON at line {line}: {source}")]
    MalformedLegacyExport {
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("legacy export contains an invalid record at line {line}")]
    InvalidLegacyRow { line: usize },
    #[error("legacy export ended without a completion marker; no rows were imported")]
    IncompleteLegacyExport,
    #[error("legacy export is missing required field {0}")]
    MissingLegacyField(String),
    #[error("legacy export field {0} has an unsupported value")]
    InvalidLegacyField(String),
    #[error("failed to serialize document data: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("failed to decode stored search metadata; rebuild the affected document: {0}")]
    DeserializeSearchData(#[source] serde_json::Error),
    #[error("the operating system random source failed; no identifier was generated: {0}")]
    Random(getrandom::Error),
    #[error("embedding dimensions must be greater than zero, found {dimensions}")]
    InvalidVectorDimensions { dimensions: usize },
    #[error("embedding has {actual} components, expected {expected}")]
    InvalidVectorLength { expected: usize, actual: usize },
    #[error("embedding contains a non-finite component; no vector data was written")]
    NonFiniteVector,
    #[error("embedding has invalid L2 norm {norm}; expected a normalized vector")]
    InvalidVectorNorm { norm: f32 },
    #[error("stored embedding contains {actual} bytes, expected {expected}; rebuild this vector")]
    MalformedStoredVector { expected: usize, actual: usize },
    #[error(
        "document {document_id} revision {claimed} is stale; current revision is {current:?}; no searchable data was changed"
    )]
    StaleRevision {
        document_id: String,
        claimed: i64,
        current: Option<i64>,
    },
    #[error("unsupported document processing stage {0}")]
    InvalidJobStage(String),
    #[error("memory proposal is missing a temporary id or content; no memory data was changed")]
    InvalidMemoryProposal,
    #[error(
        "document {0} cannot be used as a memory source because it does not exist in this organization"
    )]
    UnknownMemorySource(String),
    #[error("an independent read connection cannot be opened for an in-memory database")]
    CannotForkInMemory,
    #[error("memory was not found or is already forgotten")]
    MemoryNotFound,
    #[error(
        "existing table {table} has malformed columns; expected [{expected}], found [{actual}]; repair or recreate this unreleased schema-v1 database"
    )]
    MalformedSchema {
        table: String,
        expected: String,
        actual: String,
    },
}

impl StorageError {
    /// Returns whether continuing could expose corrupted or misleading storage state.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        fn fatal_sqlite(error: &rusqlite::Error) -> bool {
            matches!(
                error,
                rusqlite::Error::SqliteFailure(sqlite_error, _)
                    if sqlite_error.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_FUNCTION
            ) || matches!(
                error.sqlite_error_code(),
                Some(
                    rusqlite::ErrorCode::DatabaseCorrupt
                        | rusqlite::ErrorCode::NotADatabase
                        | rusqlite::ErrorCode::SystemIoFailure
                        | rusqlite::ErrorCode::DiskFull
                        | rusqlite::ErrorCode::CannotOpen
                )
            )
        }
        match self {
            Self::Open(_)
            | Self::Configure(_)
            | Self::Migrate(_)
            | Self::MalformedSchema { .. }
            | Self::DeserializeSearchData(_)
            | Self::MalformedStoredVector { .. }
            | Self::Read(rusqlite::Error::UserFunctionError(_)) => true,
            Self::Write(error) | Self::Read(error) => fatal_sqlite(error),
            _ => false,
        }
    }
}
