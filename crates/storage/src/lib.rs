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
use serde::{Deserialize, Serialize};
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
    (
        8,
        include_str!("../../../migrations/0008_v006_persistence.sql"),
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

/// Processing state persisted for a document and its durable job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentState {
    Unknown,
    Queued,
    Extracting,
    Chunking,
    Embedding,
    Indexing,
    Done,
    Failed,
}

impl DocumentState {
    /// Returns the stable value stored in `SQLite` and serialized on the wire.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Queued => "queued",
            Self::Extracting => "extracting",
            Self::Chunking => "chunking",
            Self::Embedding => "embedding",
            Self::Indexing => "indexing",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }

    /// Moves to `next` when the persisted lifecycle permits that edge.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] when the transition is not valid.
    pub fn transition(self, next: Self) -> Result<Self, TransitionError> {
        match (self, next) {
            (Self::Unknown | Self::Failed, Self::Queued)
            | (Self::Queued, Self::Extracting)
            | (Self::Extracting, Self::Chunking)
            | (Self::Chunking, Self::Embedding)
            | (Self::Embedding, Self::Indexing)
            | (Self::Indexing, Self::Done)
            | (
                Self::Queued | Self::Extracting | Self::Chunking | Self::Embedding | Self::Indexing,
                Self::Failed,
            ) => Ok(next),
            _ => Err(TransitionError {
                from: self,
                to: next,
            }),
        }
    }
}

/// A rejected document lifecycle transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("document cannot transition from {from:?} to {to:?}")]
pub struct TransitionError {
    pub from: DocumentState,
    pub to: DocumentState,
}

#[cfg(test)]
mod document_state_tests {
    use super::DocumentState;

    #[test]
    fn persisted_state_uses_stable_snake_case_values() {
        assert_eq!(DocumentState::Embedding.as_str(), "embedding");
        assert_eq!("done".parse(), Ok(DocumentState::Done));
        assert_eq!(
            DocumentState::Embedding.transition(DocumentState::Indexing),
            Ok(DocumentState::Indexing)
        );
        assert!(
            DocumentState::Queued
                .transition(DocumentState::Done)
                .is_err()
        );
        assert!("finished".parse::<DocumentState>().is_err());
    }
}

impl std::str::FromStr for DocumentState {
    type Err = DocumentStateParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "unknown" => Ok(Self::Unknown),
            "queued" => Ok(Self::Queued),
            "extracting" => Ok(Self::Extracting),
            "chunking" => Ok(Self::Chunking),
            "embedding" => Ok(Self::Embedding),
            "indexing" => Ok(Self::Indexing),
            "done" => Ok(Self::Done),
            "failed" => Ok(Self::Failed),
            _ => Err(DocumentStateParseError(value.to_owned())),
        }
    }
}

/// A document state value stored in `SQLite` that is not part of the lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid persisted document state {0:?}")]
pub struct DocumentStateParseError(pub String);

/// Result of an atomic identity/upsert decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpsertResult {
    pub id: String,
    pub status: DocumentState,
    pub enqueued: bool,
}

/// A claimed document job whose processing happens outside the database lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedJob {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub revision: i64,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    And(Vec<FilterExpression>),
    Or(Vec<FilterExpression>),
    Condition(FilterCondition),
}

/// One metadata comparison in a search filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub key: String,
    pub value: String,
    pub kind: FilterKind,
    pub numeric_operator: NumericOperator,
    pub negate: bool,
    pub ignore_case: bool,
}

/// Comparison behavior for a metadata condition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FilterKind {
    Metadata,
    Numeric,
    ArrayContains,
    StringContains,
}

/// Numeric comparison operator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

/// Selects optional context to batch-hydrate for ranked memory hits.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryHydration {
    /// Load parent, child, and related memory summaries.
    pub relations: bool,
    /// Load source-document summaries.
    pub documents: bool,
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
    pub status: DocumentState,
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

mod connection;
mod document_jobs;
mod documents;
mod legacy;
mod memories;
mod memory_jobs;
mod migrations;
mod search;

pub use documents::DocumentChunk;

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
            status: existing.document.status,
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
            status: DocumentState::Queued,
            enqueued,
        });
    }

    if existing.content_hash == hash {
        if metadata_equivalent(&existing.document.metadata, &input.metadata) {
            return Ok(UpsertResult {
                id,
                status: existing.document.status,
                enqueued: false,
            });
        }
        let metadata = merge_metadata(&existing.document.metadata, &input.metadata);
        if status == "indexing" {
            update_full(tx, &id, input, hash, &metadata, "queued")?;
            let enqueued = enqueue_unless_active(tx, &id)?;
            return Ok(UpsertResult {
                id,
                status: DocumentState::Queued,
                enqueued,
            });
        }
        tx.execute(
            "UPDATE documents SET metadata = ?2, updated_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id, json(&metadata)?],
        )
        .map_err(StorageError::Write)?;
        return Ok(UpsertResult {
            id,
            status: existing.document.status,
            enqueued: false,
        });
    }

    let metadata = merge_metadata(&existing.document.metadata, &input.metadata);
    update_full(tx, &id, input, hash, &metadata, "queued")?;
    let enqueued = enqueue_unless_active(tx, &id)?;
    Ok(UpsertResult {
        id,
        status: DocumentState::Queued,
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
        status: DocumentState::Queued,
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
    tx.execute(
        "UPDATE jobs SET status='failed', last_error_kind='stale_revision', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE document_id=?1 AND kind='memory' AND revision<>(SELECT revision FROM documents WHERE id=?1) AND status IN ('queued','extracting','chunking','embedding','indexing')",
        [id],
    )
    .map_err(StorageError::Write)?;
    Ok(())
}

fn mark_memory_job_stale(
    tx: &rusqlite::Transaction<'_>,
    job: &ClaimedMemoryJob,
) -> Result<(), StorageError> {
    tx.execute(
        "UPDATE jobs SET status='failed', last_error_kind='stale_revision', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status IN ('queued','extracting','chunking','embedding','indexing')",
        params![job.id, job.revision],
    )
    .map(|_| ())
    .map_err(StorageError::Write)
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
            status: row.get::<_, String>(4)?.parse().map_err(
                |error: DocumentStateParseError| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                },
            )?,
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

fn active_memory_ids_by_content(
    connection: &Connection,
    org_id: &str,
    container_tag: &str,
) -> Result<HashMap<String, String>, StorageError> {
    let mut statement = connection
        .prepare(
            "SELECT id, content FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_latest=1 AND is_forgotten=0 ORDER BY version DESC, updated_at DESC, id",
        )
        .map_err(StorageError::Read)?;
    let rows = statement
        .query_map(params![org_id, container_tag], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(StorageError::Read)?;
    let mut memories = HashMap::new();
    for row in rows {
        let (id, content) = row.map_err(StorageError::Read)?;
        memories.entry(normalized_memory(&content)).or_insert(id);
    }
    Ok(memories)
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
    hydration: MemoryHydration,
) -> Result<(), StorageError> {
    if hits.is_empty() || (!hydration.relations && !hydration.documents) {
        return Ok(());
    }
    let ids = hits
        .iter()
        .map(|hit| hit.record.id.clone())
        .collect::<Vec<_>>();
    let positions = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect::<HashMap<_, _>>();
    let placeholders = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    if hydration.relations {
        for (parents, owner_column, related_column) in [
            (true, "child_id", "parent_id"),
            (false, "parent_id", "child_id"),
        ] {
            let sql = format!(
                "SELECT owner_id, relation, version, content, metadata, updated_at FROM (SELECT memory_relations.{owner_column} AS owner_id, memory_relations.relation, memories.version, memories.content, memories.metadata, memories.updated_at, ROW_NUMBER() OVER (PARTITION BY memory_relations.{owner_column} ORDER BY memories.version, memories.updated_at, memories.id) AS relation_rank FROM memory_relations JOIN memories ON memories.id=memory_relations.{related_column} WHERE memory_relations.{owner_column} IN ({placeholders})) WHERE relation_rank<=2 ORDER BY version, updated_at"
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
                if let Some(index) = positions.get(&owner) {
                    let hit = &mut hits[*index];
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
    }
    if !hydration.documents {
        return Ok(());
    }
    let sql = format!(
        "SELECT memory_id, document_id, title, document_type, metadata, summary, created_at, updated_at FROM (SELECT memory_sources.memory_id, COALESCE(documents.custom_id, documents.id) AS document_id, documents.title, documents.document_type, documents.metadata, documents.summary, documents.created_at, documents.updated_at, ROW_NUMBER() OVER (PARTITION BY memory_sources.memory_id ORDER BY memory_sources.created_at, documents.id) AS source_rank FROM memory_sources JOIN documents ON documents.id=memory_sources.document_id WHERE memory_sources.memory_id IN ({placeholders})) WHERE source_rank=1"
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
        if let Some(index) = positions.get(&owner) {
            hits[*index].documents.push(document);
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
    memory
        .to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
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
        "UPDATE jobs SET status='failed', last_error_kind='stale_revision', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE kind='memory' AND status='queued' AND NOT EXISTS(SELECT 1 FROM documents WHERE documents.id=jobs.document_id AND documents.revision=jobs.revision)",
        [],
    )
    .map_err(StorageError::Migrate)?;
    tx.execute(
        "UPDATE documents SET status='queued', updated_at=CURRENT_TIMESTAMP WHERE EXISTS(SELECT 1 FROM jobs WHERE jobs.document_id=documents.id AND jobs.revision=documents.revision AND jobs.kind='document' AND jobs.status='queued') AND status IN ('extracting','chunking','embedding','indexing')",
        [],
    )
    .map_err(StorageError::Migrate)?;
    tx.execute(
        "UPDATE documents SET status='indexing', updated_at=CURRENT_TIMESTAMP WHERE EXISTS(SELECT 1 FROM jobs WHERE jobs.document_id=documents.id AND jobs.revision=documents.revision AND jobs.kind='memory' AND jobs.status='queued') AND status IN ('extracting','chunking','embedding','indexing')",
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
    )?;
    validate_columns(
        connection,
        "organization_settings",
        &[
            "org_id",
            "chunk_size",
            "should_llm_filter",
            "filter_prompt",
            "include_items",
            "exclude_items",
            "profile_buckets",
            "created_at",
            "updated_at",
        ],
    )?;
    validate_columns(
        connection,
        "file_blobs",
        &[
            "id",
            "org_id",
            "sha256",
            "content_type",
            "filename",
            "byte_length",
            "bytes",
            "created_at",
        ],
    )?;
    validate_columns(
        connection,
        "content_sources",
        &[
            "document_id",
            "kind",
            "source_url",
            "file_blob_id",
            "content_type",
            "extraction_status",
            "extracted_content",
            "extracted_metadata",
            "error_kind",
            "error_message",
            "created_at",
            "updated_at",
        ],
    )?;
    validate_columns(
        connection,
        "download_tokens",
        &[
            "token_hash",
            "org_id",
            "file_blob_id",
            "expires_at",
            "consumed_at",
            "created_at",
        ],
    )?;
    validate_columns(
        connection,
        "container_tag_merge_jobs",
        &[
            "id",
            "org_id",
            "source_tags",
            "target_tag",
            "status",
            "error_message",
            "created_at",
            "updated_at",
        ],
    )?;
    validate_columns(
        connection,
        "memory_forget_batches",
        &[
            "id",
            "org_id",
            "container_tag",
            "query",
            "dry_run",
            "candidate_count",
            "forgotten_count",
            "created_at",
            "completed_at",
        ],
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
        .map_err(StorageError::Configure)?;
    connection
        .create_scalar_function(
            "matches_metadata_filter",
            2,
            FunctionFlags::SQLITE_DETERMINISTIC | FunctionFlags::SQLITE_INNOCUOUS,
            |context| {
                let metadata =
                    serde_json::from_str::<Map<String, Value>>(&context.get::<String>(0)?)
                        .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))?;
                let filter =
                    serde_json::from_str::<FilterExpression>(&context.get::<String>(1)?)
                        .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))?;
                Ok(matches_filter(&filter, &metadata))
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
            ) || matches!(error, rusqlite::Error::FromSqlConversionFailure(_, _, _))
                || matches!(
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
            | Self::Read(rusqlite::Error::UserFunctionError(_)) => true,
            Self::Write(error) | Self::Read(error) => fatal_sqlite(error),
            _ => false,
        }
    }
}
