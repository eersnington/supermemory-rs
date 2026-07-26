//! Synchronous `PostgreSQL` persistence and atomic document identity decisions.

#![expect(
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    reason = "the storage facade exposes one shared database error contract"
)]

use std::{
    cell::{RefCell, RefMut},
    collections::{HashMap, HashSet},
    path::Path,
};

use bytes::BytesMut;
use postgres::{Client, GenericClient, NoTls, Row};
use postgres_types::{FromSql, IsNull, ToSql, Type, to_sql_checked};
use serde::Serialize;
use serde_json::{Map, Value};
use sha1::{Digest, Sha1};
use thiserror::Error;

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

/// An initialized synchronous `PostgreSQL` connection.
pub struct Storage {
    connection: RefCell<Option<Client>>,
    local_org_id: String,
}

impl Storage {
    /// Connects to `PostgreSQL` using the supplied connection string and initializes a fresh schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let dsn = path
            .as_ref()
            .to_str()
            .ok_or(StorageError::InvalidConnectionString)?;
        let mut connection = Client::connect(dsn, NoTls).map_err(StorageError::Open)?;
        let local_org_id = initialize(&mut connection)?;
        Ok(Self {
            connection: RefCell::new(Some(connection)),
            local_org_id,
        })
    }

    /// Connects using `DATABASE_URL`. Intended for isolated test databases.
    pub fn in_memory() -> Result<Self, StorageError> {
        let dsn = std::env::var("DATABASE_URL").map_err(|_| StorageError::MissingDatabaseUrl)?;
        Self::open(dsn)
    }

    fn client(&self) -> RefMut<'_, Client> {
        RefMut::map(self.connection.borrow_mut(), |connection| {
            connection
                .as_mut()
                .expect("PostgreSQL client is available until Storage is dropped")
        })
    }

    #[must_use]
    pub fn local_organization_id(&self) -> &str {
        &self.local_org_id
    }

    pub fn upsert_document(&mut self, input: UpsertDocument) -> Result<UpsertResult, StorageError> {
        let org = self.local_org_id.clone();
        self.upsert_document_for(&org, input)
    }

    pub fn upsert_document_for(
        &mut self,
        org_id: &str,
        mut input: UpsertDocument,
    ) -> Result<UpsertResult, StorageError> {
        input.content = sanitize_content(&input.content);
        let hash = content_hash(&input.content);
        let mut client = self.client();
        let mut tx = client.transaction().map_err(StorageError::Write)?;
        let existing = if let Some(custom) = input.custom_id.as_deref() {
            find_custom(&mut tx, org_id, custom, &input.container_tags)?
        } else {
            find_duplicate(
                &mut tx,
                org_id,
                &hash,
                &input.metadata,
                &input.container_tags,
            )?
        };
        let result = if let Some(existing) = existing {
            apply_existing(&mut tx, &existing, &input, &hash)?
        } else {
            insert_new(&mut tx, org_id, &input, &hash)?
        };
        tx.commit().map_err(StorageError::Write)?;
        Ok(result)
    }

    pub fn find_document(&self, identifier: &str) -> Result<Option<Document>, StorageError> {
        self.find_document_for(&self.local_org_id, identifier)
    }

    pub fn find_document_for(
        &self,
        org_id: &str,
        identifier: &str,
    ) -> Result<Option<Document>, StorageError> {
        let mut db = self.client();
        if let Some(row) = db
            .query_opt(
                &format!("{SELECT_DOCUMENT} WHERE org_id=$1 AND id=$2 ORDER BY created_at LIMIT 1"),
                &[&org_id, &identifier],
            )
            .map_err(StorageError::Read)?
        {
            return Ok(Some(read_document(&row)?.document));
        }
        db.query_opt(
            &format!(
                "{SELECT_DOCUMENT} WHERE org_id=$1 AND custom_id=$2 ORDER BY created_at LIMIT 1"
            ),
            &[&org_id, &identifier],
        )
        .map_err(StorageError::Read)?
        .map(|r| read_document(&r).map(|d| d.document))
        .transpose()
    }

    pub fn claim_job(&mut self) -> Result<Option<ClaimedJob>, StorageError> {
        let mut db = self.client();
        let mut tx = db.transaction().map_err(StorageError::Write)?;
        let row = tx.query_opt("SELECT j.id,d.id,d.content,j.revision,d.org_id FROM jobs j JOIN documents d ON d.id=j.document_id WHERE j.status='queued' AND j.revision=d.revision AND j.available_at<=now() ORDER BY j.created_at FOR UPDATE OF j SKIP LOCKED LIMIT 1", &[]).map_err(StorageError::Read)?;
        let Some(row) = row else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(None);
        };
        let job = ClaimedJob {
            id: row.get(0),
            document_id: row.get(1),
            content: row.get(2),
            revision: row.get(3),
            organization_id: row.get(4),
        };
        let revision_parameter = job.revision.to_string();
        tx.execute("UPDATE jobs SET status='extracting',attempts=attempts+1,updated_at=now() WHERE id=$1 AND revision=$2::bigint", &[&job.id,&revision_parameter]).map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='extracting',updated_at=now() WHERE id=$1 AND revision=$2::bigint",
            &[&job.document_id, &revision_parameter],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(Some(job))
    }

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
        self.publish_job(job, &embedded, None)
    }

    pub fn complete_embedded_job(
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
        self.publish_job(job, chunks, Some(model_id))
    }

    fn publish_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        model_id: Option<&str>,
    ) -> Result<(), StorageError> {
        let mut db = self.client();
        let mut tx = db.transaction().map_err(StorageError::Write)?;
        let current = tx
            .query_opt(
                "SELECT revision FROM documents WHERE id=$1 FOR UPDATE",
                &[&job.document_id],
            )
            .map_err(StorageError::Read)?
            .map(|r| r.get(0));
        if current != Some(job.revision) {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current,
            });
        }
        let rows = tx
            .query(
                "SELECT ordinal,content,stable_id FROM document_chunks WHERE document_id=$1",
                &[&job.document_id],
            )
            .map_err(StorageError::Read)?;
        let mut existing: HashMap<(i64, String), String> = rows
            .into_iter()
            .map(|r| ((r.get(0), r.get(1)), r.get(2)))
            .collect();
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id=$1",
            &[&job.document_id],
        )
        .map_err(StorageError::Write)?;
        for (ordinal, chunk) in chunks.iter().enumerate() {
            let ordinal = i64::try_from(ordinal).map_err(|_| StorageError::PositionOverflow)?;
            let stable = existing
                .remove(&(ordinal, chunk.content.to_owned()))
                .map_or_else(generate_id, Ok)?;
            let ordinal_parameter = ordinal.to_string();
            let chunk_id: i64 = tx
                .query_one(
                    "UPDATE id_allocator SET next_id=next_id+1 WHERE name='document_chunks' RETURNING next_id",
                    &[],
                )
                .map_err(StorageError::Write)?
                .get(0);
            let chunk_id_parameter = chunk_id.to_string();
            tx.execute("INSERT INTO document_chunks(id,document_id,ordinal,content,stable_id) VALUES($1::bigint,$2,$3::bigint,$4,$5)", &[&chunk_id_parameter,&job.document_id,&ordinal_parameter,&chunk.content,&stable]).map_err(StorageError::Write)?;
            if let Some(model) = model_id {
                let chunk_id_parameter = chunk_id.to_string();
                let vector_parameter = vector_text(chunk.vector);
                tx.execute(
                    "INSERT INTO chunk_embeddings(chunk_id,model_id,vector) VALUES($1::bigint,$2,$3::vector)",
                    &[&chunk_id_parameter, &model, &vector_parameter],
                )
                .map_err(StorageError::Write)?;
            }
        }
        let revision_parameter = job.revision.to_string();
        tx.execute("UPDATE jobs SET status='done',last_error=NULL,updated_at=now() WHERE id=$1 AND revision=$2::bigint", &[&job.id,&revision_parameter]).map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='done',updated_at=now() WHERE id=$1 AND revision=$2::bigint",
            &[&job.document_id, &revision_parameter],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }

    pub fn mark_job_stage(&mut self, job: &ClaimedJob, stage: &str) -> Result<(), StorageError> {
        let expected = match stage {
            "chunking" => "extracting",
            "embedding" => "chunking",
            "indexing" => "embedding",
            _ => return Err(StorageError::InvalidJobStage(stage.to_owned())),
        };
        let mut db = self.client();
        let mut tx = db.transaction().map_err(StorageError::Write)?;
        let revision_parameter = job.revision.to_string();
        let a=tx.execute("UPDATE jobs SET status=$3,updated_at=now() WHERE id=$1 AND revision=$2::bigint AND status=$4", &[&job.id,&revision_parameter,&stage,&expected]).map_err(StorageError::Write)?;
        let b = tx
            .execute(
                "UPDATE documents SET status=$3,updated_at=now() WHERE id=$1 AND revision=$2::bigint",
                &[&job.document_id, &revision_parameter, &stage],
            )
            .map_err(StorageError::Write)?;
        if a != 1 || b != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: None,
            });
        }
        tx.commit().map_err(StorageError::Write)
    }
    pub fn fail_job(&mut self, job: &ClaimedJob, error: &str) -> Result<(), StorageError> {
        let mut db = self.client();
        let mut tx = db.transaction().map_err(StorageError::Write)?;
        let revision_parameter = job.revision.to_string();
        tx.execute("UPDATE jobs SET status='failed',last_error=$3,updated_at=now() WHERE id=$1 AND revision=$2::bigint",&[&job.id,&revision_parameter,&error]).map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='failed',updated_at=now() WHERE id=$1 AND revision=$2::bigint",
            &[&job.document_id, &revision_parameter],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }
    pub fn delete_empty_document(&mut self, job: &ClaimedJob) -> Result<(), StorageError> {
        let revision_parameter = job.revision.to_string();
        self.client().execute("DELETE FROM documents WHERE id=$1 AND revision=$2::bigint AND status IN ('extracting','chunking')",&[&job.document_id,&revision_parameter]).map(|_|()).map_err(StorageError::Write)
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, StorageError> {
        self.search_for(&self.local_org_id, query, limit)
    }
    pub fn search_for(
        &self,
        org_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchHit>, StorageError> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let pattern = format!("%{}%", query.to_lowercase());
        let limit_parameter = limit.to_string();
        let rows=self.client().query("SELECT c.stable_id,d.id,c.content,1.0::float8,c.ordinal,d.custom_id,d.metadata,d.filepath,d.created_at::text,d.updated_at::text,d.content FROM document_chunks c JOIN documents d ON d.id=c.document_id WHERE lower(c.content) LIKE $1 AND d.org_id=$2 AND d.status='done' ORDER BY c.ordinal LIMIT $3::bigint",&[&pattern,&org_id,&limit_parameter]).map_err(StorageError::Read)?;
        rows.iter().map(read_search_hit).collect()
    }

    pub fn search_semantic(
        &self,
        query: &[f32],
        model_id: &str,
        limit: usize,
        threshold: f32,
        options: &SearchOptions,
    ) -> Result<Vec<SearchHit>, StorageError> {
        validate_vector(query, 768)?;
        let org = options
            .organization_id
            .as_deref()
            .unwrap_or(&self.local_org_id);
        let result_limit = limit;
        let candidate_limit = i64::try_from(limit.saturating_mul(10)).unwrap_or(i64::MAX);
        let query_parameter = PgVector(query.to_vec());
        let sql = format!(
            "SELECT c.stable_id,d.id,c.content,e.vector,c.ordinal,d.custom_id,d.metadata,d.filepath,d.created_at::text,d.updated_at::text,d.content,d.container_tags FROM chunk_embeddings e JOIN document_chunks c ON c.id=e.chunk_id JOIN documents d ON d.id=c.document_id WHERE e.model_id=$2 AND d.org_id=$3 AND d.status='done' ORDER BY e.vector<=>$1::vector LIMIT {candidate_limit}"
        );
        let rows = self
            .client()
            .query(&sql, &[&query_parameter, &model_id, &org])
            .map_err(StorageError::Read)?;
        let mut hits = Vec::new();
        for row in rows {
            let metadata = object(row.get(6))?;
            let tags: Vec<String> = serde_json::from_str(&row.get::<_, String>(11))
                .map_err(StorageError::DeserializeSearchData)?;
            if !matches_search_options(
                row.get::<_, String>(1).as_str(),
                row.get::<_, Option<String>>(5).as_deref(),
                &tags,
                row.get::<_, Option<String>>(7).as_deref(),
                &metadata,
                options,
            ) {
                continue;
            }
            let stored: PgVector = row.get(3);
            let score = exact_similarity(query, &stored.0)?;
            if score >= threshold {
                hits.push(read_search_hit_with_score(&row, f64::from(score))?);
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.document_id.cmp(&right.document_id))
                .then_with(|| left.position.cmp(&right.position))
                .then_with(|| left.id.cmp(&right.id))
        });
        hits.truncate(result_limit);
        Ok(hits)
    }

    pub fn reconcile_memories(
        &mut self,
        document_id: &str,
        container_tag: &str,
        proposals: &[MemoryProposal],
        model_id: &str,
        dimensions: usize,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let org = self.local_org_id.clone();
        self.reconcile_memories_for(
            &org,
            document_id,
            container_tag,
            proposals,
            model_id,
            dimensions,
        )
    }
    pub fn reconcile_memories_for(
        &mut self,
        org_id: &str,
        document_id: &str,
        container_tag: &str,
        proposals: &[MemoryProposal],
        model_id: &str,
        dimensions: usize,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        if dimensions != 768 {
            return Err(StorageError::InvalidVectorDimensions { dimensions });
        }
        for p in proposals {
            validate_vector(&p.vector, dimensions)?;
            if p.temporary_id.is_empty() || p.content.trim().is_empty() {
                return Err(StorageError::InvalidMemoryProposal);
            }
        }
        let mut db = self.client();
        let mut tx = db.transaction().map_err(StorageError::Write)?;
        if tx
            .query_opt(
                "SELECT 1 FROM documents WHERE id=$1 AND org_id=$2",
                &[&document_id, &org_id],
            )
            .map_err(StorageError::Read)?
            .is_none()
        {
            return Err(StorageError::UnknownMemorySource(document_id.to_owned()));
        }
        let mut temp = HashMap::new();
        let mut ids = Vec::new();
        for p in proposals {
            if let Some(existing) = find_exact_memory(&mut tx, org_id, container_tag, &p.content)? {
                tx.execute("INSERT INTO memory_sources(memory_id,document_id) VALUES($1,$2) ON CONFLICT DO NOTHING",&[&existing.id,&document_id]).map_err(StorageError::Write)?;
                temp.insert(p.temporary_id.clone(), existing.id.clone());
                ids.push(existing.id);
                continue;
            }
            let parents =
                resolve_memory_parents(&mut tx, org_id, container_tag, &p.parents, &temp)?;
            let primary = parents.first();
            let id = format!("mem_{}", generate_id()?);
            let root = primary.map(|x| x.root_memory_id.clone().unwrap_or_else(|| x.id.clone()));
            let parent = primary.map(|x| x.id.clone());
            let version = primary.map_or(1, |x| x.version + 1);
            let inferred = p.is_inferred || parents.iter().any(|x| x.relation == "derives");
            let forget = valid_future_datetime(&mut tx, p.forget_after.as_deref())?;
            let metadata = json_value(&p.metadata)?;
            let inferred_parameter = inferred.to_string();
            let static_parameter = p.is_static.to_string();
            let version_parameter = version.to_string();
            tx.execute("INSERT INTO memories(id,org_id,container_tag,content,metadata,is_inferred,is_static,root_memory_id,parent_memory_id,version,forget_after,forget_reason) VALUES($1,$2,$3,$4,$5,$6::boolean,$7::boolean,$8,$9,$10::bigint,$11::timestamptz,$12)",&[&id,&org_id,&container_tag,&p.content,&metadata,&inferred_parameter,&static_parameter,&root,&parent,&version_parameter,&forget,&p.forget_reason]).map_err(StorageError::Write)?;
            tx.execute(
                "INSERT INTO memory_sources(memory_id,document_id) VALUES($1,$2)",
                &[&id, &document_id],
            )
            .map_err(StorageError::Write)?;
            for x in &parents {
                tx.execute("INSERT INTO memory_relations(parent_id,child_id,relation) VALUES($1,$2,$3) ON CONFLICT DO NOTHING",&[&x.id,&id,&x.relation]).map_err(StorageError::Write)?;
                if x.relation == "updates" {
                    tx.execute("UPDATE memories SET is_latest=false,is_static=false,updated_at=now() WHERE id=$1",&[&x.id]).map_err(StorageError::Write)?;
                    tx.execute(
                        "UPDATE memory_embeddings SET active=false WHERE memory_id=$1",
                        &[&x.id],
                    )
                    .map_err(StorageError::Write)?;
                }
            }
            tx.execute(
                "INSERT INTO memory_embeddings(memory_id,model_id,vector) VALUES($1,$2,$3)",
                &[&id, &model_id, &PgVector(p.vector.clone())],
            )
            .map_err(StorageError::Write)?;
            temp.insert(p.temporary_id.clone(), id.clone());
            ids.push(id);
        }
        let records = ids
            .iter()
            .map(|id| read_memory_by_id(&mut tx, id))
            .collect::<Result<_, _>>()?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(records)
    }

    pub fn static_profile(&self, tag: &str) -> Result<Vec<MemoryRecord>, StorageError> {
        self.static_profile_for(&self.local_org_id, tag)
    }
    pub fn static_profile_for(
        &self,
        org: &str,
        tag: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.profile(org, tag, true, &HashSet::new())
    }
    pub fn dynamic_profile(
        &self,
        tag: &str,
        statics: &[MemoryRecord],
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.dynamic_profile_for(&self.local_org_id, tag, statics)
    }
    pub fn dynamic_profile_for(
        &self,
        org: &str,
        tag: &str,
        statics: &[MemoryRecord],
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let excluded = statics
            .iter()
            .map(|m| normalized_memory(&m.memory))
            .collect();
        self.profile(org, tag, false, &excluded)
    }
    fn profile(
        &self,
        org: &str,
        tag: &str,
        is_static: bool,
        excluded: &HashSet<String>,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let static_parameter = is_static.to_string();
        let rows=self.client().query("SELECT id,content,metadata,is_inferred,is_static,is_latest,is_forgotten,root_memory_id,parent_memory_id,version,forget_after::text,forget_reason,created_at::text,updated_at::text FROM memories WHERE org_id=$1 AND container_tag=$2 AND is_static=$3::boolean AND is_latest AND NOT is_forgotten AND (forget_after IS NULL OR forget_after>now()) ORDER BY updated_at DESC LIMIT 300",&[&org,&tag,&static_parameter]).map_err(StorageError::Read)?;
        let records = rows.iter().map(read_memory).collect::<Result<_, _>>()?;
        Ok(deduplicate_memories(records, 100, excluded))
    }
    pub fn bucket_profile(
        &self,
        tag: &str,
        bucket: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.bucket_profile_for(&self.local_org_id, tag, bucket)
    }
    pub fn bucket_profile_for(
        &self,
        org: &str,
        tag: &str,
        bucket: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let s = self.static_profile_for(org, tag)?;
        let d = self.dynamic_profile_for(org, tag, &s)?;
        Ok(s.into_iter()
            .chain(d)
            .filter(|m| {
                m.metadata
                    .get("buckets")
                    .and_then(Value::as_array)
                    .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(bucket)))
            })
            .take(100)
            .collect())
    }

    pub fn forget_memory(
        &mut self,
        id: Option<&str>,
        content: Option<&str>,
        tag: &str,
        reason: Option<&str>,
    ) -> Result<String, StorageError> {
        let org = self.local_org_id.clone();
        self.forget_memory_for(&org, id, content, tag, reason)
    }
    pub fn forget_memory_for(
        &mut self,
        org: &str,
        id: Option<&str>,
        content: Option<&str>,
        tag: &str,
        reason: Option<&str>,
    ) -> Result<String, StorageError> {
        let mut db = self.client();
        let mut tx = db.transaction().map_err(StorageError::Write)?;
        let row=if let Some(id)=id{tx.query_opt("SELECT id FROM memories WHERE id=$1 AND org_id=$2 AND container_tag=$3 AND NOT is_forgotten",&[&id,&org,&tag])}else if let Some(content)=content{tx.query_opt("SELECT id FROM memories WHERE content=$1 AND org_id=$2 AND container_tag=$3 AND NOT is_forgotten ORDER BY created_at LIMIT 1",&[&content,&org,&tag])}else{Ok(None)}.map_err(StorageError::Read)?;
        let id: String = row.ok_or(StorageError::MemoryNotFound)?.get(0);
        tx.execute("UPDATE memories SET is_forgotten=true,is_latest=false,is_static=false,forget_after=now(),forget_reason=$2,updated_at=now() WHERE id=$1",&[&id,&reason.unwrap_or("user_requested")]).map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE memory_embeddings SET active=false WHERE memory_id=$1",
            &[&id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(id)
    }

    pub fn search_memories(
        &self,
        q: &[f32],
        model: &str,
        tag: &str,
        limit: usize,
        threshold: f32,
        forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        self.search_memories_for(
            &self.local_org_id,
            q,
            model,
            tag,
            limit,
            threshold,
            forgotten,
        )
    }
    #[expect(clippy::too_many_arguments, reason = "public search contract")]
    pub fn search_memories_for(
        &self,
        org: &str,
        q: &[f32],
        model: &str,
        tag: &str,
        limit: usize,
        threshold: f32,
        forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        validate_vector(q, 768)?;
        let candidate_limit = i64::try_from(limit.saturating_mul(10)).unwrap_or(i64::MAX);
        let query_parameter = PgVector(q.to_vec());
        let forgotten_parameter = forgotten.to_string();
        let sql = format!(
            "SELECT m.id,m.content,m.metadata,m.is_inferred,m.is_static,m.is_latest,m.is_forgotten,m.root_memory_id,m.parent_memory_id,m.version,m.forget_after::text,m.forget_reason,m.created_at::text,m.updated_at::text,e.vector FROM memory_embeddings e JOIN memories m ON m.id=e.memory_id WHERE m.org_id=$2 AND m.container_tag=$3 AND e.model_id=$4 AND (e.active OR $5::boolean) AND (NOT m.is_forgotten OR $5::boolean) ORDER BY e.vector<=>$1::vector LIMIT {candidate_limit}"
        );
        let rows = self
            .client()
            .query(
                &sql,
                &[&query_parameter, &org, &tag, &model, &forgotten_parameter],
            )
            .map_err(StorageError::Read)?;
        let mut hits = Vec::new();
        for row in rows {
            let vector: PgVector = row.get(14);
            let similarity = exact_similarity(q, &vector.0)?;
            if similarity >= threshold {
                hits.push(MemorySearchHit {
                    record: read_memory(&row)?,
                    similarity: f64::from(similarity),
                    parents: Vec::new(),
                    children: Vec::new(),
                    related: Vec::new(),
                    documents: Vec::new(),
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.record.id.cmp(&right.record.id))
        });
        hits.truncate(limit);
        let mut db = self.client();
        for hit in &mut hits {
            let id = &hit.record.id;
            hit.parents = memory_relations(&mut *db, id, true, None)?;
            hit.children = memory_relations(&mut *db, id, false, None)?;
            hit.documents = memory_source_documents(&mut *db, id)?;
        }
        Ok(hits)
    }

    pub fn api_key_hashes(&self) -> Result<Vec<[u8; 32]>, StorageError> {
        Ok(self
            .api_key_identities()?
            .into_iter()
            .map(|x| x.0)
            .collect())
    }
    pub fn api_key_identities(&self) -> Result<Vec<ApiKeyIdentity>, StorageError> {
        let rows=self.client().query("SELECT key_hash,org_id FROM api_keys WHERE enabled AND (expires_at IS NULL OR expires_at>now())",&[]).map_err(StorageError::Read)?;
        rows.into_iter()
            .map(|r| {
                let b: Vec<u8> = r.get(0);
                let hash = b
                    .try_into()
                    .map_err(|b: Vec<u8>| StorageError::MalformedApiKeyHash(b.len()))?;
                Ok((hash, r.get(1)))
            })
            .collect()
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Some(client) = self.connection.get_mut().take() {
            let _ = std::thread::spawn(move || drop(client)).join();
        }
    }
}

/// Stored API-key hash and optional organization.
pub type ApiKeyIdentity = ([u8; 32], Option<String>);

const SELECT_DOCUMENT: &str = "SELECT id,content,content_hash,custom_id,status,container_tags,entity_context,metadata,task_type,filepath,filter_by_metadata,dreaming,created_at::text,updated_at::text FROM documents";

fn json_value<T: Serialize + ?Sized>(v: &T) -> Result<String, StorageError> {
    serde_json::to_string(v).map_err(StorageError::Serialize)
}
fn read_document(r: &Row) -> Result<StoredDocument, StorageError> {
    Ok(StoredDocument {
        document: Document {
            id: r.get(0),
            content: r.get(1),
            custom_id: r.get(3),
            status: r.get(4),
            container_tags: serde_json::from_str(&r.get::<_, String>(5))
                .map_err(StorageError::DeserializeSearchData)?,
            entity_context: r.get(6),
            metadata: object(r.get(7))?,
            task_type: r.get(8),
            filepath: r.get(9),
            filter_by_metadata: object(r.get(10))?,
            dreaming: r.get(11),
            created_at: r.get(12),
            updated_at: r.get(13),
        },
        content_hash: r.get(2),
    })
}
fn object(value: String) -> Result<Map<String, Value>, StorageError> {
    serde_json::from_str(&value).map_err(StorageError::DeserializeSearchData)
}
fn read_search_hit(r: &Row) -> Result<SearchHit, StorageError> {
    read_search_hit_with_score(r, r.get(3))
}

fn read_search_hit_with_score(r: &Row, score: f64) -> Result<SearchHit, StorageError> {
    Ok(SearchHit {
        id: r.get(0),
        document_id: r.get(1),
        chunk: r.get(2),
        score,
        position: usize::try_from(r.get::<_, i64>(4))
            .map_err(|_| StorageError::PositionOverflow)?,
        custom_id: r.get(5),
        metadata: object(r.get(6))?,
        filepath: r.get(7),
        created_at: r.get(8),
        updated_at: r.get(9),
        document_content: r.get(10),
    })
}

fn find_custom<C: GenericClient>(
    db: &mut C,
    org: &str,
    custom: &str,
    tags: &[String],
) -> Result<Option<StoredDocument>, StorageError> {
    let tags = json_value(tags)?;
    db.query_opt(&format!("{SELECT_DOCUMENT} WHERE org_id=$1 AND custom_id=$2 AND container_tags=$3 ORDER BY created_at LIMIT 1"),&[&org,&custom,&tags]).map_err(StorageError::Read)?.map(|r|read_document(&r)).transpose()
}
fn find_duplicate<C: GenericClient>(
    db: &mut C,
    org: &str,
    hash: &str,
    metadata: &Map<String, Value>,
    tags: &[String],
) -> Result<Option<StoredDocument>, StorageError> {
    let rows=db.query(&format!("{SELECT_DOCUMENT} WHERE org_id=$1 AND content_hash=$2 AND status='done' ORDER BY created_at"),&[&org,&hash]).map_err(StorageError::Read)?;
    let wanted = normalized_tags(tags);
    for r in rows {
        let d = read_document(&r)?;
        if normalized_tags(&d.document.container_tags) == wanted
            && metadata_equivalent(&d.document.metadata, metadata)
        {
            return Ok(Some(d));
        }
    }
    Ok(None)
}
fn apply_existing<C: GenericClient>(
    db: &mut C,
    e: &StoredDocument,
    input: &UpsertDocument,
    hash: &str,
) -> Result<UpsertResult, StorageError> {
    let id = e.document.id.clone();
    let status = e.document.status.as_str();
    if matches!(
        status,
        "unknown" | "queued" | "extracting" | "chunking" | "embedding" | "indexing"
    ) {
        return Ok(UpsertResult {
            id,
            status: status.to_owned(),
            enqueued: false,
        });
    }
    if status == "failed" {
        update_full(
            db,
            &id,
            input,
            hash,
            &merge_metadata(&e.document.metadata, &input.metadata),
            "queued",
        )?;
        let enqueued = enqueue(db, &id)?;
        return Ok(UpsertResult {
            id,
            status: "queued".into(),
            enqueued,
        });
    }
    if e.content_hash == hash {
        if metadata_equivalent(&e.document.metadata, &input.metadata) {
            return Ok(UpsertResult {
                id,
                status: "done".into(),
                enqueued: false,
            });
        }
        let metadata = json_value(&merge_metadata(&e.document.metadata, &input.metadata))?;
        db.execute(
            "UPDATE documents SET metadata=$2,updated_at=now() WHERE id=$1",
            &[&id, &metadata],
        )
        .map_err(StorageError::Write)?;
        return Ok(UpsertResult {
            id,
            status: "done".into(),
            enqueued: false,
        });
    }
    update_full(
        db,
        &id,
        input,
        hash,
        &merge_metadata(&e.document.metadata, &input.metadata),
        "queued",
    )?;
    let enqueued = enqueue(db, &id)?;
    Ok(UpsertResult {
        id,
        status: "queued".into(),
        enqueued,
    })
}
fn insert_new<C: GenericClient>(
    db: &mut C,
    org: &str,
    input: &UpsertDocument,
    hash: &str,
) -> Result<UpsertResult, StorageError> {
    let id = generate_id()?;
    db.execute("INSERT INTO documents(id,org_id,content,content_hash,custom_id,status,container_tags,entity_context,metadata,task_type,filepath,filter_by_metadata,dreaming) VALUES($1,$2,$3,$4,$5,'queued',$6,$7,$8,$9,$10,$11,$12)",&[&id,&org,&input.content,&hash,&input.custom_id,&json_value(&input.container_tags)?,&input.entity_context,&json_value(&input.metadata)?,&input.task_type,&input.filepath,&json_value(&input.filter_by_metadata)?,&input.dreaming]).map_err(StorageError::Write)?;
    enqueue(db, &id)?;
    Ok(UpsertResult {
        id,
        status: "queued".into(),
        enqueued: true,
    })
}
fn update_full<C: GenericClient>(
    db: &mut C,
    id: &str,
    i: &UpsertDocument,
    hash: &str,
    m: &Map<String, Value>,
    status: &str,
) -> Result<(), StorageError> {
    db.execute("UPDATE documents SET content=$2,content_hash=$3,custom_id=$4,status=$5,container_tags=$6,entity_context=$7,metadata=$8,task_type=$9,filepath=$10,filter_by_metadata=$11,dreaming=$12,revision=revision+1,updated_at=now() WHERE id=$1",&[&id,&i.content,&hash,&i.custom_id,&status,&json_value(&i.container_tags)?,&i.entity_context,&json_value(m)?,&i.task_type,&i.filepath,&json_value(&i.filter_by_metadata)?,&i.dreaming]).map(|_|()).map_err(StorageError::Write)
}
fn enqueue<C: GenericClient>(db: &mut C, id: &str) -> Result<bool, StorageError> {
    if db.query_opt("SELECT 1 FROM jobs WHERE document_id=$1 AND status IN ('queued','extracting','chunking','embedding','indexing')",&[&id]).map_err(StorageError::Read)?.is_some(){return Ok(false)}
    let revision: i64 = db
        .query_one("SELECT revision FROM documents WHERE id=$1", &[&id])
        .map_err(StorageError::Read)?
        .get(0);
    let revision_parameter = revision.to_string();
    db.execute(
        "INSERT INTO jobs(id,document_id,revision) VALUES($1,$2,$3::bigint)",
        &[&generate_id()?, &id, &revision_parameter],
    )
    .map_err(StorageError::Write)?;
    Ok(true)
}

struct ResolvedParent {
    id: String,
    relation: String,
    root_memory_id: Option<String>,
    version: i64,
}
fn resolve_memory_parents<C: GenericClient>(
    db: &mut C,
    org: &str,
    tag: &str,
    parents: &[MemoryParent],
    temp: &HashMap<String, String>,
) -> Result<Vec<ResolvedParent>, StorageError> {
    let mut out = Vec::new();
    for p in parents {
        let id = temp
            .get(&p.memory_id)
            .map_or(p.memory_id.as_str(), String::as_str);
        if let Some(r)=db.query_opt("SELECT id,root_memory_id,version FROM memories WHERE id=$1 AND org_id=$2 AND container_tag=$3",&[&id,&org,&tag]).map_err(StorageError::Read)?{out.push(ResolvedParent{id:r.get(0),relation:p.relation.clone(),root_memory_id:r.get(1),version:r.get(2)})}
    }
    Ok(out)
}
const SELECT_MEMORY: &str = "SELECT id,content,metadata,is_inferred,is_static,is_latest,is_forgotten,root_memory_id,parent_memory_id,version,forget_after::text,forget_reason,created_at::text,updated_at::text FROM memories";
fn find_exact_memory<C: GenericClient>(
    db: &mut C,
    org: &str,
    tag: &str,
    content: &str,
) -> Result<Option<MemoryRecord>, StorageError> {
    db.query_opt(&format!("{SELECT_MEMORY} WHERE org_id=$1 AND container_tag=$2 AND content=$3 AND is_latest AND NOT is_forgotten ORDER BY created_at LIMIT 1"),&[&org,&tag,&content]).map_err(StorageError::Read)?.map(|r|read_memory(&r)).transpose()
}
fn read_memory_by_id<C: GenericClient>(db: &mut C, id: &str) -> Result<MemoryRecord, StorageError> {
    let r = db
        .query_one(&format!("{SELECT_MEMORY} WHERE id=$1"), &[&id])
        .map_err(StorageError::Read)?;
    read_memory(&r)
}
fn read_memory(r: &Row) -> Result<MemoryRecord, StorageError> {
    Ok(MemoryRecord {
        id: r.get(0),
        memory: r.get(1),
        metadata: object(r.get(2))?,
        is_inferred: r.get(3),
        is_static: r.get(4),
        is_latest: r.get(5),
        is_forgotten: r.get(6),
        root_memory_id: r.get(7),
        parent_memory_id: r.get(8),
        version: r.get(9),
        forget_after: r.get(10),
        forget_reason: r.get(11),
        created_at: r.get(12),
        updated_at: r.get(13),
    })
}
fn memory_relations<C: GenericClient>(
    db: &mut C,
    id: &str,
    parents: bool,
    allowed: Option<&[&str]>,
) -> Result<Vec<MemoryRelationHit>, StorageError> {
    let (join, related) = if parents {
        ("child_id", "parent_id")
    } else {
        ("parent_id", "child_id")
    };
    let rows=db.query(&format!("SELECT r.relation,m.version,m.content,m.metadata,m.updated_at::text FROM memory_relations r JOIN memories m ON m.id=r.{related} WHERE r.{join}=$1 ORDER BY m.version,m.updated_at"),&[&id]).map_err(StorageError::Read)?;
    rows.into_iter()
        .filter_map(|r| {
            let relation: String = r.get(0);
            allowed
                .is_none_or(|a| a.contains(&relation.as_str()))
                .then(|| {
                    Ok(MemoryRelationHit {
                        relation,
                        version: r.get(1),
                        memory: r.get(2),
                        metadata: object(r.get(3))?,
                        updated_at: r.get(4),
                    })
                })
        })
        .collect()
}
fn memory_source_documents<C: GenericClient>(
    db: &mut C,
    id: &str,
) -> Result<Vec<MemorySourceDocument>, StorageError> {
    db.query("SELECT COALESCE(d.custom_id,d.id),d.title,d.document_type,d.metadata,d.summary,d.created_at::text,d.updated_at::text FROM memory_sources s JOIN documents d ON d.id=s.document_id WHERE s.memory_id=$1 ORDER BY s.created_at",&[&id]).map_err(StorageError::Read)?.into_iter().map(|r|Ok(MemorySourceDocument{id:r.get(0),title:r.get(1),document_type:r.get(2),metadata:object(r.get(3))?,summary:r.get(4),created_at:r.get(5),updated_at:r.get(6)})).collect()
}
fn valid_future_datetime<C: GenericClient>(
    db: &mut C,
    v: Option<&str>,
) -> Result<Option<String>, StorageError> {
    let Some(v) = v else { return Ok(None) };
    let valid: bool = db
        .query_one("SELECT $1::timestamptz>now()", &[&v])
        .map_err(StorageError::Read)?
        .get(0);
    Ok(valid.then(|| v.to_owned()))
}

fn vector_text(vector: &[f32]) -> String {
    let mut text = String::from("[");
    for (index, value) in vector.iter().enumerate() {
        if index != 0 {
            text.push(',');
        }
        text.push_str(&value.to_string());
    }
    text.push(']');
    text
}

#[derive(Debug, Clone)]
struct PgVector(Vec<f32>);
impl ToSql for PgVector {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn std::error::Error + Sync + Send>> {
        if ty.name() != "vector" {
            return Err("expected pgvector vector type".into());
        }
        let dimensions = u16::try_from(self.0.len())?;
        out.extend_from_slice(&dimensions.to_be_bytes());
        out.extend_from_slice(&0u16.to_be_bytes());
        for value in &self.0 {
            out.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        Ok(IsNull::No)
    }
    fn accepts(ty: &Type) -> bool {
        ty.name() == "vector"
    }
    to_sql_checked!();
}
impl<'a> FromSql<'a> for PgVector {
    fn from_sql(
        ty: &Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        if ty.name() != "vector" || raw.len() < 4 {
            return Err("invalid pgvector value".into());
        }
        let n = usize::from(u16::from_be_bytes([raw[0], raw[1]]));
        if raw.len() != 4 + n * 4 {
            return Err("invalid pgvector payload length".into());
        }
        let values = raw[4..]
            .chunks_exact(4)
            .map(|b| f32::from_bits(u32::from_be_bytes([b[0], b[1], b[2], b[3]])))
            .collect();
        Ok(Self(values))
    }
    fn accepts(ty: &Type) -> bool {
        ty.name() == "vector"
    }
}

fn initialize(db: &mut Client) -> Result<String, StorageError> {
    db.batch_execute(SCHEMA).map_err(StorageError::Migrate)?;
    let row = db
        .query_opt("SELECT id FROM organizations WHERE slug=$1", &[&LOCAL_SLUG])
        .map_err(StorageError::Migrate)?;
    let id = row.map(|r| r.get(0)).map_or_else(generate_id, Ok)?;
    db.execute(
        "INSERT INTO organizations(id,slug) VALUES($1,$2) ON CONFLICT(slug) DO NOTHING",
        &[&id, &LOCAL_SLUG],
    )
    .map_err(StorageError::Migrate)?;
    let actual: String = db
        .query_one("SELECT id FROM organizations WHERE slug=$1", &[&LOCAL_SLUG])
        .map_err(StorageError::Migrate)?
        .get(0);
    db.batch_execute("UPDATE jobs SET status='queued',available_at=now(),updated_at=now() WHERE status IN ('extracting','chunking','embedding','indexing'); UPDATE documents SET status='queued',updated_at=now() WHERE id IN(SELECT document_id FROM jobs WHERE status='queued') AND status IN ('extracting','chunking','embedding','indexing')").map_err(StorageError::Migrate)?;
    Ok(actual)
}

const SCHEMA: &str = r"
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS organizations(id text PRIMARY KEY,slug text UNIQUE NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS documents(id text PRIMARY KEY,org_id text NOT NULL REFERENCES organizations(id),content text NOT NULL,content_hash text NOT NULL,custom_id text,status text NOT NULL DEFAULT 'queued',container_tags text NOT NULL DEFAULT '[]',entity_context text,metadata text NOT NULL DEFAULT '{}',task_type text NOT NULL,filepath text,filter_by_metadata text NOT NULL DEFAULT '{}',dreaming text NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),revision bigint NOT NULL DEFAULT 1,title text,summary text,document_type text,source text,url text,user_id text);
CREATE INDEX IF NOT EXISTS documents_identity_idx ON documents(org_id,custom_id);
CREATE TABLE IF NOT EXISTS jobs(id text PRIMARY KEY,document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,kind text NOT NULL DEFAULT 'document',status text NOT NULL DEFAULT 'queued',attempts integer NOT NULL DEFAULT 0,available_at timestamptz NOT NULL DEFAULT now(),last_error text,created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),revision bigint NOT NULL);
CREATE TABLE IF NOT EXISTS id_allocator(name text PRIMARY KEY,next_id bigint NOT NULL);
INSERT INTO id_allocator(name,next_id) VALUES('document_chunks',0) ON CONFLICT(name) DO NOTHING;
CREATE TABLE IF NOT EXISTS document_chunks(id bigint PRIMARY KEY,document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,ordinal bigint NOT NULL,content text NOT NULL,stable_id text NOT NULL UNIQUE,UNIQUE(document_id,ordinal));
CREATE TABLE IF NOT EXISTS chunk_embeddings(chunk_id bigint PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,model_id text NOT NULL,vector vector(768) NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_idx ON chunk_embeddings USING hnsw(vector vector_cosine_ops);
CREATE TABLE IF NOT EXISTS memories(id text PRIMARY KEY,org_id text NOT NULL REFERENCES organizations(id),container_tag text NOT NULL,content text NOT NULL,metadata text NOT NULL DEFAULT '{}',is_inferred boolean NOT NULL DEFAULT false,is_static boolean NOT NULL DEFAULT false,is_latest boolean NOT NULL DEFAULT true,is_forgotten boolean NOT NULL DEFAULT false,root_memory_id text REFERENCES memories(id),parent_memory_id text REFERENCES memories(id),version bigint NOT NULL DEFAULT 1,forget_after timestamptz,forget_reason text,created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS memory_embeddings(memory_id text PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,model_id text NOT NULL,vector vector(768) NOT NULL,active boolean NOT NULL DEFAULT true,created_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS memory_embeddings_hnsw_idx ON memory_embeddings USING hnsw(vector vector_cosine_ops);
CREATE TABLE IF NOT EXISTS memory_relations(parent_id text REFERENCES memories(id) ON DELETE CASCADE,child_id text REFERENCES memories(id) ON DELETE CASCADE,relation text NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(parent_id,child_id,relation));
CREATE TABLE IF NOT EXISTS memory_sources(memory_id text REFERENCES memories(id) ON DELETE CASCADE,document_id text REFERENCES documents(id) ON DELETE CASCADE,created_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(memory_id,document_id));
CREATE TABLE IF NOT EXISTS api_keys(id text PRIMARY KEY,org_id text REFERENCES organizations(id),key_hash bytea NOT NULL,name text,enabled boolean NOT NULL DEFAULT true,expires_at timestamptz,created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now());
";

fn validate_vector(v: &[f32], dimensions: usize) -> Result<(), StorageError> {
    if v.len() != dimensions {
        return Err(StorageError::InvalidVectorLength {
            expected: dimensions,
            actual: v.len(),
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(StorageError::NonFiniteVector);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 1e-4 {
        return Err(StorageError::InvalidVectorNorm { norm });
    }
    Ok(())
}
fn exact_similarity(left: &[f32], right: &[f32]) -> Result<f32, StorageError> {
    if left.len() != right.len() {
        return Err(StorageError::InvalidVectorLength {
            expected: left.len(),
            actual: right.len(),
        });
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum())
}

fn matches_search_options(
    document_id: &str,
    custom: Option<&str>,
    tags: &[String],
    filepath: Option<&str>,
    metadata: &Map<String, Value>,
    o: &SearchOptions,
) -> bool {
    if o.document_id
        .as_deref()
        .is_some_and(|x| x != document_id && Some(x) != custom)
    {
        return false;
    }
    if !o.container_tags.is_empty() && !o.container_tags.iter().any(|x| tags.contains(x)) {
        return false;
    }
    if o.filepath.as_deref().is_some_and(|x| {
        x.strip_suffix('/').map_or(filepath != Some(x), |p| {
            !filepath.is_some_and(|f| f.starts_with(p))
        })
    }) {
        return false;
    }
    o.filters
        .as_ref()
        .is_none_or(|f| matches_filter(f, metadata))
}
fn matches_filter(f: &FilterExpression, m: &Map<String, Value>) -> bool {
    match f {
        FilterExpression::And(fs) => fs.iter().all(|f| matches_filter(f, m)),
        FilterExpression::Or(fs) => fs.iter().any(|f| matches_filter(f, m)),
        FilterExpression::Condition(c) => {
            m.get(&c.key).is_some_and(|v| matches_condition(c, v)) != c.negate
        }
    }
}
fn matches_condition(c: &FilterCondition, v: &Value) -> bool {
    match c.kind {
        FilterKind::Metadata => comparable(v).is_some_and(|x| eq(&x, &c.value, c.ignore_case)),
        FilterKind::StringContains => comparable(v).is_some_and(|x| {
            if c.ignore_case {
                x.to_lowercase().contains(&c.value.to_lowercase())
            } else {
                x.contains(&c.value)
            }
        }),
        FilterKind::ArrayContains => v.as_array().is_some_and(|a| {
            a.iter()
                .any(|v| comparable(v).is_some_and(|x| eq(&x, &c.value, c.ignore_case)))
        }),
        FilterKind::Numeric => {
            v.as_f64()
                .zip(c.value.parse().ok())
                .is_some_and(|(a, b)| match c.numeric_operator {
                    NumericOperator::Greater => a > b,
                    NumericOperator::Less => a < b,
                    NumericOperator::GreaterOrEqual => a >= b,
                    NumericOperator::LessOrEqual => a <= b,
                    NumericOperator::Equal => a.total_cmp(&b).is_eq(),
                })
        }
    }
}
fn comparable(v: &Value) -> Option<String> {
    match v {
        Value::String(x) => Some(x.clone()),
        Value::Number(x) => Some(x.to_string()),
        Value::Bool(x) => Some(x.to_string()),
        _ => None,
    }
}
fn eq(a: &str, b: &str, ignore: bool) -> bool {
    if ignore {
        a.to_lowercase() == b.to_lowercase()
    } else {
        a == b
    }
}
fn deduplicate_memories(
    ms: Vec<MemoryRecord>,
    limit: usize,
    excluded: &HashSet<String>,
) -> Vec<MemoryRecord> {
    let mut seen = excluded.clone();
    ms.into_iter()
        .filter(|m| {
            let n = normalized_memory(&m.memory);
            !n.is_empty() && seen.insert(n)
        })
        .take(limit)
        .collect()
}
fn normalized_memory(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
fn merge_metadata(a: &Map<String, Value>, b: &Map<String, Value>) -> Map<String, Value> {
    let mut m = a.clone();
    m.extend(b.clone());
    m
}
fn metadata_equivalent(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    let clean = |m: &Map<String, Value>| {
        m.iter()
            .filter(|(k, _)| !k.starts_with("sm_") && k.as_str() != "commonQuestions")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<Map<_, _>>()
    };
    clean(a) == clean(b)
}
fn normalized_tags(tags: &[String]) -> Vec<&str> {
    let mut v: Vec<_> = tags
        .iter()
        .map(String::as_str)
        .filter(|x| !x.trim().is_empty())
        .collect();
    v.sort_unstable();
    v
}
#[must_use]
pub fn sanitize_content(content: &str) -> String {
    content
        .trim()
        .chars()
        .filter(|c| !matches!(*c as u32,0x00..=0x08|0x0b|0x0c|0x0e..=0x1f|0x7f))
        .collect::<String>()
        .trim()
        .to_owned()
}
#[must_use]
pub fn content_hash(content: &str) -> String {
    format!("{:x}", Sha1::digest(content.as_bytes()))
}
pub fn generate_id() -> Result<String, StorageError> {
    let mut id = String::with_capacity(22);
    let mut random = [0u8; 32];
    while id.len() < 22 {
        getrandom::fill(&mut random).map_err(StorageError::Random)?;
        for byte in random.into_iter().filter(|b| *b < 232) {
            id.push(char::from(BASE58[usize::from(byte % 58)]));
            if id.len() == 22 {
                break;
            }
        }
    }
    Ok(id)
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("failed to connect to PostgreSQL: {0}")]
    Open(#[source] postgres::Error),
    #[error("failed to initialize PostgreSQL schema: {0}")]
    Migrate(#[source] postgres::Error),
    #[error("failed to write PostgreSQL data: {0}")]
    Write(#[source] postgres::Error),
    #[error("failed to read PostgreSQL data: {0}")]
    Read(#[source] postgres::Error),
    #[error("PostgreSQL connection string is not valid UTF-8")]
    InvalidConnectionString,
    #[error("DATABASE_URL is required")]
    MissingDatabaseUrl,
    #[error("failed to serialize JSON: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("failed to decode stored JSON: {0}")]
    DeserializeSearchData(#[source] serde_json::Error),
    #[error("stored JSON is not an object")]
    MalformedJsonObject,
    #[error("API key hash has {0} bytes, expected 32")]
    MalformedApiKeyHash(usize),
    #[error("position exceeds PostgreSQL integer range")]
    PositionOverflow,
    #[error("random source failed: {0}")]
    Random(getrandom::Error),
    #[error("embedding dimensions must be 768, found {dimensions}")]
    InvalidVectorDimensions { dimensions: usize },
    #[error("embedding has {actual} components, expected {expected}")]
    InvalidVectorLength { expected: usize, actual: usize },
    #[error("embedding contains a non-finite component")]
    NonFiniteVector,
    #[error("embedding has invalid L2 norm {norm}")]
    InvalidVectorNorm { norm: f32 },
    #[error("document {document_id} revision {claimed} is stale; current revision is {current:?}")]
    StaleRevision {
        document_id: String,
        claimed: i64,
        current: Option<i64>,
    },
    #[error("unsupported document processing stage {0}")]
    InvalidJobStage(String),
    #[error("invalid memory proposal")]
    InvalidMemoryProposal,
    #[error("unknown memory source document {0}")]
    UnknownMemorySource(String),
    #[error("memory was not found or already forgotten")]
    MemoryNotFound,
}
