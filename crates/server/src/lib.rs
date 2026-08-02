//! HTTP boundary and durable worker lifecycle with sticky fatal health degradation.

mod auth;
mod health;
mod indexing_coordinator;
mod memory_extraction_coordinator;
mod runtime;
mod search_engine;
mod search_projection;
mod ui;
mod writer;
mod routes {
    pub(super) mod documents;
}

use auth::authenticate;
use routes::documents::{create_document, get_document};
use ui::{api_reference, landing_page, openapi};

use health::ServiceHealth;
pub use runtime::ServerRuntime;
use runtime::{StorageReaderError, StorageReaders};

use std::{net::SocketAddr, num::NonZeroUsize, sync::Arc};

use axum::{
    Json, Router,
    extract::{ConnectInfo, Extension, Path, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use storage::{Storage, UpsertDocument};
use subtle::ConstantTimeEq;
use thiserror::Error;
use tokio::net::TcpListener;

type SharedHealth = Arc<ServiceHealth>;

#[derive(Clone, Copy)]
struct FatalApiFailure;

#[derive(Clone)]
struct AppState {
    api_keys: Arc<Vec<([u8; 32], String)>>,
    api_key: Option<String>,
    port: u16,
    writer: writer::StorageWriter,
    readers: StorageReaders,
    embedding_executor: Option<memory_engine::EmbeddingExecutor>,
    local_org_id: String,
    health: SharedHealth,
}

#[derive(Clone)]
struct OrganizationId(String);

/// Builds the complete HTTP application.
pub fn router(api_key: Option<String>, storage: Storage) -> Router {
    let runtime = ServerRuntime::new(storage, None);
    router_with_runtime(api_key, 6767, &runtime)
}

/// Builds the HTTP application with local semantic search enabled.
pub fn router_with_embeddings(
    api_key: Option<String>,
    storage: Storage,
    embeddings: Arc<memory_engine::EmbeddingModel>,
) -> Router {
    let runtime = ServerRuntime::new(storage, Some(embeddings));
    router_with_runtime(api_key, 6767, &runtime)
}

/// Builds a router from an explicitly injected process runtime.
///
/// Tests that also run workers must use this constructor so requests and workers
/// demonstrably share the same writer and embedding executor.
pub fn router_with_runtime(api_key: Option<String>, port: u16, runtime: &ServerRuntime) -> Router {
    router_with_runtime_health(api_key, port, Arc::new(ServiceHealth::default()), runtime)
}

fn router_with_runtime_health(
    api_key: Option<String>,
    port: u16,
    health: SharedHealth,
    runtime: &ServerRuntime,
) -> Router {
    let local_org_id = runtime.local_org_id().to_owned();
    let mut api_keys = runtime
        .api_key_identities()
        .iter()
        .map(|(hash, org)| (*hash, org.clone().unwrap_or_else(|| local_org_id.clone())))
        .collect::<Vec<_>>();
    if let Some(key) = api_key.as_ref() {
        api_keys.push((Sha256::digest(key).into(), local_org_id.clone()));
    }
    let state = AppState {
        writer: runtime.writer(),
        api_keys: Arc::new(api_keys),
        api_key,
        port,
        readers: runtime.readers(),
        embedding_executor: runtime.embeddings(),
        local_org_id,
        health,
    };
    let api = Router::new()
        .route("/documents", post(create_document))
        .route("/documents/{id}", get(get_document))
        .route("/search", post(v3_search))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    let search = Router::new()
        .route("/search", post(search))
        .route("/profile", post(profile))
        .route("/profile/buckets", post(profile_buckets))
        .route("/memories/", delete(forget_memory))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    Router::new()
        .route("/", get(landing_page))
        .route("/health", get(health_handler))
        .route("/v4/openapi", get(openapi))
        .route("/v4/reference", get(api_reference))
        .nest("/v3", api)
        .nest("/v4", search)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            degrade_after_internal_error,
        ))
        .with_state(state)
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Serves the application until the process receives a shutdown signal.
///
/// # Errors
///
/// Returns an error if the address cannot be bound or the HTTP server stops unexpectedly.
pub async fn serve(
    address: SocketAddr,
    api_key: Option<String>,
    storage: Storage,
) -> Result<(), ServerError> {
    serve_with_ready(address, api_key, storage, || {}).await
}

/// Serves the application and invokes `ready` after successfully binding the listener.
///
/// # Errors
/// Returns an error if the address cannot be bound or the HTTP server stops unexpectedly.
pub async fn serve_with_ready(
    address: SocketAddr,
    api_key: Option<String>,
    storage: Storage,
    ready: impl FnOnce(),
) -> Result<(), ServerError> {
    serve_with_services_ready(address, api_key, storage, None, None, ready).await
}

/// Serves with a loaded local embedding model and reports readiness after binding.
///
/// # Errors
/// Returns an error if the address cannot be bound or the HTTP server stops unexpectedly.
pub async fn serve_with_embeddings_ready(
    address: SocketAddr,
    api_key: Option<String>,
    storage: Storage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    ready: impl FnOnce(),
) -> Result<(), ServerError> {
    serve_with_services_ready(address, api_key, storage, embeddings, None, ready).await
}

/// Serves with local embeddings and an optional configured memory provider.
///
/// # Errors
/// Returns an error if the address cannot be bound or the HTTP server stops unexpectedly.
pub async fn serve_with_services_ready(
    address: SocketAddr,
    api_key: Option<String>,
    storage: Storage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    provider: Option<Arc<memory_engine::MemoryProvider>>,
    ready: impl FnOnce(),
) -> Result<(), ServerError> {
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;
    ready();
    let health = Arc::new(ServiceHealth::default());
    let runtime = ServerRuntime::new(storage, embeddings);
    let worker = tokio::spawn(
        indexing_coordinator::IndexingCoordinator::new(
            runtime.writer(),
            runtime.embeddings(),
            provider.as_ref().map(Arc::clone),
        )
        .run(Arc::clone(&health)),
    );
    let memory_worker =
        provider
            .as_ref()
            .zip(runtime.embeddings().as_ref())
            .map(|(provider, executor)| {
                tokio::spawn(
                    memory_extraction_coordinator::MemoryExtractionCoordinator::new(
                        runtime.writer(),
                        runtime.readers(),
                        executor.clone(),
                        Arc::clone(provider),
                    )
                    .run(Arc::clone(&health)),
                )
            });
    let result = axum::serve(
        listener,
        router_with_runtime_health(api_key, address.port(), health, &runtime)
            .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(ServerError::Serve);
    worker.abort();
    if let Some(worker) = memory_worker {
        worker.abort();
    }
    result
}

async fn degrade_after_internal_error(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if state.health.is_degraded()
        && !matches!(
            request.uri().path(),
            "/" | "/health" | "/v4/openapi" | "/v4/reference"
        )
    {
        return ApiError::ServiceDegraded.into_response();
    }
    let response = next.run(request).await;
    if response.extensions().get::<FatalApiFailure>().is_some() {
        state.health.degrade();
    }
    response
}

async fn health_handler(State(state): State<AppState>) -> Response {
    if state.health.is_degraded() {
        let probe = state
            .writer
            .execute(|storage| storage.migration_version())
            .await;
        if matches!(probe, Ok(Ok(_))) {
            state.health.recover();
        }
    }
    if state.health.is_degraded() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(Health { status: "degraded" }),
        )
            .into_response()
    } else {
        Json(Health { status: "ok" }).into_response()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateDocumentRequest {
    content: String,
    container_tag: Option<String>,
    container_tags: Option<Vec<String>>,
    entity_context: Option<String>,
    custom_id: Option<String>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default = "default_task_type")]
    task_type: TaskType,
    filepath: Option<String>,
    #[serde(default)]
    filter_by_metadata: Map<String, Value>,
    #[serde(default = "default_dreaming")]
    dreaming: Dreaming,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum TaskType {
    Memory,
    Superrag,
}
impl TaskType {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Superrag => "superrag",
        }
    }
}
fn default_task_type() -> TaskType {
    TaskType::Memory
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Dreaming {
    Instant,
    Dynamic,
}
impl Dreaming {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Instant => "instant",
            Self::Dynamic => "dynamic",
        }
    }
}
fn default_dreaming() -> Dreaming {
    Dreaming::Dynamic
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
#[expect(clippy::struct_excessive_bools, reason = "v0.0.5 wire contract")]
struct V3SearchRequest {
    q: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default)]
    chunk_threshold: f32,
    #[serde(default)]
    document_threshold: f32,
    container_tag: Option<String>,
    container_tags: Option<Vec<String>>,
    doc_id: Option<String>,
    filters: Option<Value>,
    #[serde(default)]
    include_full_docs: bool,
    #[serde(default)]
    include_summary: bool,
    #[serde(default = "default_true", rename = "onlyMatchingChunks")]
    _only_matching_chunks: bool,
    #[serde(default)]
    rerank: bool,
    #[serde(default)]
    #[serde(rename = "rewriteQuery")]
    _rewrite_query: bool,
    #[serde(rename = "categoriesFilter")]
    _categories_filter: Option<Vec<String>>,
    filepath: Option<String>,
}

const fn default_true() -> bool {
    true
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct V3SearchResponse {
    results: Vec<V3DocumentResult>,
    timing: f64,
    total: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct V3DocumentResult {
    chunks: Vec<V3ChunkResult>,
    created_at: String,
    document_id: String,
    metadata: Option<Map<String, Value>>,
    score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    title: Option<String>,
    updated_at: String,
    #[serde(rename = "type")]
    document_type: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct V3ChunkResult {
    content: String,
    is_relevant: bool,
    score: f64,
    position: usize,
}

/// Runs one read operation on a pooled `SQLite` reader, falling back to the writer in memory.
async fn run_search<T, F>(state: &AppState, run: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&Storage) -> Result<T, ApiError> + Send + 'static,
{
    state
        .readers
        .execute(run)
        .await
        .map_err(|error| match error {
            StorageReaderError::Executor(error) => ApiError::DatabaseExecutor(error),
            StorageReaderError::Unavailable | StorageReaderError::Closed => {
                ApiError::StorageUnavailable
            }
        })?
}

async fn v3_search(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<V3SearchRequest>,
) -> Result<Json<V3SearchResponse>, ApiError> {
    validate_search_basics(&request.q, request.limit, request.chunk_threshold)?;
    if request.rerank {
        return Err(ApiError::RerankingUnavailable);
    }
    if !request.document_threshold.is_finite() || !(0.0..=1.0).contains(&request.document_threshold)
    {
        return Err(ApiError::Validation(
            "documentThreshold must be between 0 and 1",
        ));
    }
    if request.doc_id.as_ref().is_some_and(|id| id.len() > 255) {
        return Err(ApiError::Validation("docId must be at most 255 characters"));
    }
    validate_container_tag(request.container_tag.as_deref())?;
    if let Some(tags) = request.container_tags.as_ref() {
        for tag in tags {
            validate_container_tag(Some(tag))?;
        }
    }
    search_engine::SearchEngine::new(state)
        .search_v3(&organization.0, request)
        .await
        .map(Json)
}

fn metadata_string(metadata: &Map<String, Value>, key: &str) -> Option<String> {
    metadata.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn validate_search_basics(query: &str, limit: usize, threshold: f32) -> Result<(), ApiError> {
    if query.trim().is_empty() {
        return Err(ApiError::Validation("Search query cannot be empty"));
    }
    if !(1..=100).contains(&limit) {
        return Err(ApiError::Validation("limit must be between 1 and 100"));
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(ApiError::Validation(
            "chunkThreshold must be between 0 and 1",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    q: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default = "default_search_threshold")]
    threshold: f32,
    container_tag: Option<String>,
    filters: Option<Value>,
    #[serde(default)]
    include: SearchInclude,
    #[serde(default)]
    rerank: bool,
    #[serde(default)]
    aggregate: bool,
    #[serde(default)]
    #[serde(rename = "rewriteQuery")]
    _rewrite_query: bool,
    #[serde(default)]
    search_mode: SearchMode,
    filepath: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
#[expect(clippy::struct_excessive_bools, reason = "v0.0.5 wire contract")]
struct SearchInclude {
    #[serde(default)]
    documents: bool,
    #[serde(default)]
    summaries: bool,
    #[serde(default)]
    related_memories: bool,
    #[serde(default)]
    forgotten_memories: bool,
    #[serde(default)]
    chunks: bool,
}

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SearchMode {
    #[default]
    Memories,
    Hybrid,
    Documents,
}

const fn default_search_limit() -> usize {
    10
}

const fn default_search_threshold() -> f32 {
    0.6
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResponse {
    results: Vec<SearchResult>,
    timing: f64,
    total: usize,
}

#[derive(Serialize)]
#[serde(untagged)]
enum SearchResult {
    Memory(MemorySearchResult),
    Chunk(ChunkSearchResult),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MemorySearchResult {
    id: String,
    memory: String,
    metadata: Option<Map<String, Value>>,
    updated_at: String,
    similarity: f64,
    version: i64,
    root_memory_id: Option<String>,
    context: EmptyContext,
    documents: Vec<SearchDocument>,
    chunks: Vec<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChunkSearchResult {
    id: String,
    chunk: String,
    metadata: Option<Map<String, Value>>,
    filepath: Option<String>,
    updated_at: String,
    similarity: f64,
    version: u8,
    context: EmptyContext,
    documents: Vec<SearchDocument>,
    chunks: Vec<Value>,
}

#[derive(Serialize)]
struct EmptyContext {
    parents: Vec<Value>,
    children: Vec<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    related: Vec<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchDocument {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    document_type: Option<String>,
    metadata: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    created_at: String,
    updated_at: String,
}

async fn search(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let started = std::time::Instant::now();
    if request.q.trim().is_empty() {
        return Err(ApiError::Validation("q must not be empty"));
    }
    let limit = NonZeroUsize::new(request.limit)
        .filter(|limit| limit.get() <= 100)
        .ok_or(ApiError::Validation("limit must be between 1 and 100"))?;
    if !request.threshold.is_finite() || !(0.0..=1.0).contains(&request.threshold) {
        return Err(ApiError::Validation(
            "threshold must be a finite number between 0 and 1",
        ));
    }
    if request.rerank {
        return Err(ApiError::RerankingUnavailable);
    }
    validate_container_tag(request.container_tag.as_deref())?;
    let filters = request.filters.as_ref().map(parse_filter).transpose()?;
    let mode = if request.include.chunks && request.search_mode == SearchMode::Memories {
        SearchMode::Hybrid
    } else {
        request.search_mode
    };
    let options = storage::SearchOptions {
        organization_id: Some(organization.0.clone()),
        container_tags: vec![
            request
                .container_tag
                .unwrap_or_else(|| "sm_project_default".to_owned()),
        ],
        document_id: None,
        filepath: request.filepath,
        filters,
    };
    let outcome = search_engine::SearchEngine::new(state)
        .search(
            &organization.0,
            search_engine::SearchQuery {
                text: request.q.trim().to_owned(),
                mode,
                limit,
                threshold: request.threshold,
                options,
                include_forgotten: request.include.forgotten_memories,
                include_documents: request.include.documents,
                include_summaries: request.include.summaries,
                include_related: request.include.related_memories,
                aggregate: request.aggregate,
            },
        )
        .await?;
    let total = outcome.results.len();
    Ok(Json(SearchResponse {
        results: outcome.results,
        timing: started.elapsed().as_secs_f64() * 1_000.0,
        total,
    }))
}

fn memory_search_result(
    hit: storage::MemorySearchHit,
    boost: f64,
    include_documents: bool,
    include_summaries: bool,
    include_related: bool,
) -> MemorySearchResult {
    let relations = |relations: Vec<storage::MemoryRelationHit>| {
        if include_related {
            let mut seen = std::collections::HashSet::new();
            relations
                .into_iter()
                .filter(|relation| seen.insert(relation.memory.to_lowercase()))
                .take(2)
                .filter_map(|relation| serde_json::to_value(relation).ok())
                .collect()
        } else {
            Vec::new()
        }
    };
    let parents = relations(hit.parents);
    let children = relations(hit.children);
    let related = relations(hit.related);
    let documents = if include_documents {
        hit.documents
            .into_iter()
            .take(1)
            .map(|document| SearchDocument {
                id: document.id,
                title: document.title,
                document_type: document.document_type,
                metadata: Some(document.metadata),
                summary: include_summaries.then_some(document.summary).flatten(),
                created_at: document.created_at,
                updated_at: document.updated_at,
            })
            .collect()
    } else {
        Vec::new()
    };
    MemorySearchResult {
        id: hit.record.id,
        memory: hit.record.memory,
        metadata: Some(hit.record.metadata),
        updated_at: hit.record.updated_at,
        similarity: (hit.similarity * boost).min(1.0),
        version: hit.record.version,
        root_memory_id: hit.record.root_memory_id,
        context: EmptyContext {
            parents,
            children,
            related,
        },
        documents,
        chunks: Vec::new(),
    }
}

fn chunk_search_result(hit: storage::SearchHit) -> ChunkSearchResult {
    let title = hit
        .metadata
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let document_type = hit
        .metadata
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let document = SearchDocument {
        id: hit.custom_id.unwrap_or_else(|| hit.document_id.clone()),
        title,
        document_type,
        metadata: Some(hit.metadata.clone()),
        summary: None,
        created_at: hit.created_at,
        updated_at: hit.updated_at.clone(),
    };
    ChunkSearchResult {
        id: hit.id,
        chunk: hit.chunk,
        metadata: Some(hit.metadata),
        filepath: hit.filepath,
        updated_at: hit.updated_at,
        similarity: hit.score,
        version: 1,
        context: EmptyContext {
            parents: Vec::new(),
            children: Vec::new(),
            related: Vec::new(),
        },
        documents: vec![document],
        chunks: Vec::new(),
    }
}

fn validate_container_tag(container_tag: Option<&str>) -> Result<(), ApiError> {
    if container_tag.is_some_and(|tag| {
        tag.chars().count() > 100
            || tag.is_empty()
            || !tag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'-'))
    }) {
        return Err(ApiError::Validation(
            "containerTag may only contain up to 100 alphanumeric characters, hyphens, underscores, and colons",
        ));
    }
    Ok(())
}

fn parse_filter(value: &Value) -> Result<storage::FilterExpression, ApiError> {
    let parsed = if let Some(value) = value.as_str() {
        serde_json::from_str(value)
            .map_err(|_| ApiError::Validation("filters must be valid JSON"))?
    } else {
        value.clone()
    };
    let mut conditions = 0;
    parse_filter_at(&parsed, 0, &mut conditions)
}

fn parse_filter_at(
    value: &Value,
    depth: usize,
    conditions: &mut usize,
) -> Result<storage::FilterExpression, ApiError> {
    if depth > 8 {
        return Err(ApiError::Validation(
            "filter structure is too complex; use at most 8 nesting levels",
        ));
    }
    let object = value
        .as_object()
        .ok_or(ApiError::Validation("invalid filter condition structure"))?;
    if let Some(values) = object.get("AND").or_else(|| object.get("OR")) {
        let values = values
            .as_array()
            .ok_or(ApiError::Validation("AND and OR filters must be arrays"))?;
        *conditions = conditions.saturating_add(values.len());
        if values.len() > 200 || *conditions > 200 {
            return Err(ApiError::Validation(
                "too many filter conditions; use at most 200 conditions",
            ));
        }
        let nested = values
            .iter()
            .map(|value| parse_filter_at(value, depth + 1, conditions))
            .collect::<Result<Vec<_>, _>>()?;
        return if object.contains_key("AND") {
            Ok(storage::FilterExpression::And(nested))
        } else {
            Ok(storage::FilterExpression::Or(nested))
        };
    }
    *conditions = conditions.saturating_add(1);
    if *conditions > 200 {
        return Err(ApiError::Validation(
            "too many filter conditions; use at most 200 conditions",
        ));
    }
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or(ApiError::Validation("filter key must be a string"))?;
    let value = object
        .get("value")
        .and_then(Value::as_str)
        .ok_or(ApiError::Validation("filter value must be a string"))?;
    let kind = match object
        .get("filterType")
        .and_then(Value::as_str)
        .unwrap_or("metadata")
    {
        "metadata" => storage::FilterKind::Metadata,
        "numeric" => storage::FilterKind::Numeric,
        "array_contains" => storage::FilterKind::ArrayContains,
        "string_contains" => storage::FilterKind::StringContains,
        _ => return Err(ApiError::Validation("invalid filterType")),
    };
    if matches!(kind, storage::FilterKind::Numeric)
        && (value.trim().is_empty() || value.parse::<f64>().is_err())
    {
        return Err(ApiError::Validation(
            "numeric filter value must be a valid number",
        ));
    }
    let numeric_operator = match object
        .get("numericOperator")
        .and_then(Value::as_str)
        .unwrap_or("=")
    {
        ">" => storage::NumericOperator::Greater,
        "<" => storage::NumericOperator::Less,
        ">=" => storage::NumericOperator::GreaterOrEqual,
        "<=" => storage::NumericOperator::LessOrEqual,
        "=" => storage::NumericOperator::Equal,
        _ => return Err(ApiError::Validation("invalid numericOperator")),
    };
    Ok(storage::FilterExpression::Condition(
        storage::FilterCondition {
            key: key.to_owned(),
            value: value.to_owned(),
            kind,
            numeric_operator,
            negate: boolean_field(object.get("negate"))?,
            ignore_case: boolean_field(object.get("ignoreCase"))?,
        },
    ))
}

fn boolean_field(value: Option<&Value>) -> Result<bool, ApiError> {
    match value {
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::String(value)) if value == "true" => Ok(true),
        Some(Value::String(value)) if value == "false" => Ok(false),
        Some(_) => Err(ApiError::Validation(
            "filter boolean fields must be true or false",
        )),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProfileRequest {
    q: Option<String>,
    container_tag: String,
    #[serde(default = "default_profile_threshold")]
    threshold: f32,
    filters: Option<Value>,
    include: Option<Vec<ProfileSection>>,
    buckets: Option<Vec<String>>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ProfileSection {
    Static,
    Dynamic,
    Buckets,
}

const fn default_profile_threshold() -> f32 {
    0.5
}

#[derive(Serialize)]
struct ProfileResponse {
    profile: Profile,
    #[serde(rename = "searchResults", skip_serializing_if = "Option::is_none")]
    search_results: Option<SearchResponse>,
}

#[derive(Default, Serialize)]
struct Profile {
    #[serde(rename = "static", skip_serializing_if = "Option::is_none")]
    static_memories: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Complete immutable facts; response projection never decorates fact text.
    dynamic: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    buckets: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProfileBucketsRequest {
    container_tag: String,
}

#[derive(Serialize)]
struct ProfileBucketsResponse {
    buckets: Vec<BucketDefinition>,
}

#[derive(Serialize)]
struct BucketDefinition {
    key: &'static str,
    description: &'static str,
}

const PREFERENCES_DESCRIPTION: &str = "Explicit first-person preferences only — things the person directly stated they prefer, like, or dislike (e.g. 'prefers X over Y', 'dislikes Z', 'always uses W'). Must be a direct stated choice, not an observation or inference. Exclude: personality traits, communication styles, behavioral patterns, opinions about products or other people, and anything described as a characteristic rather than a stated preference.";

async fn profile(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<ProfileRequest>,
) -> Result<Json<ProfileResponse>, ApiError> {
    validate_container_tag(Some(&request.container_tag))?;
    if !request.threshold.is_finite() || !(0.0..=1.0).contains(&request.threshold) {
        return Err(ApiError::Validation("threshold must be between 0 and 1"));
    }
    let _filters = request.filters.as_ref().map(parse_filter).transpose()?;
    let include = request.include.unwrap_or_else(|| {
        vec![
            ProfileSection::Static,
            ProfileSection::Dynamic,
            ProfileSection::Buckets,
        ]
    });
    let container_tag = request.container_tag.clone();
    let profile_org_id = organization.0.clone();
    let requested_buckets = request
        .buckets
        .unwrap_or_else(|| vec!["preferences".to_owned()]);
    let want_static = include.contains(&ProfileSection::Static);
    let want_dynamic = include.contains(&ProfileSection::Dynamic);
    let want_buckets = include.contains(&ProfileSection::Buckets);
    let (static_memories, dynamic_memories, buckets) = run_search(&state, move |storage| {
        let static_memories = storage
            .static_profile_for(&profile_org_id, &container_tag)
            .map_err(ApiError::Storage)?;
        let dynamic_memories = storage
            .dynamic_profile_for(&profile_org_id, &container_tag, &static_memories)
            .map_err(ApiError::Storage)?;
        let buckets = if want_buckets {
            requested_buckets
                .iter()
                .map(|bucket| {
                    storage
                        .bucket_profile_for(&profile_org_id, &container_tag, bucket)
                        .map(|memories| (bucket.clone(), memories))
                        .map_err(ApiError::Storage)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        Ok((static_memories, dynamic_memories, buckets))
    })
    .await?;
    let mut bucket_values = Map::new();
    for (bucket, memories) in buckets {
        bucket_values.insert(
            bucket,
            serde_json::json!(
                memories
                    .into_iter()
                    .map(|memory| memory.memory)
                    .collect::<Vec<_>>()
            ),
        );
    }
    let search_results = if let Some(query) = request.q.filter(|query| !query.trim().is_empty()) {
        let outcome = search_engine::SearchEngine::new(state.clone())
            .search_profile(search_engine::ProfileSearchQuery {
                organization_id: organization.0.clone(),
                text: query,
                container_tag: request.container_tag.clone(),
                threshold: request.threshold,
            })
            .await?;
        Some(SearchResponse {
            total: outcome.results.len(),
            results: outcome.results,
            timing: outcome.timing,
        })
    } else {
        None
    };
    Ok(Json(ProfileResponse {
        profile: Profile {
            static_memories: want_static.then(|| {
                static_memories
                    .into_iter()
                    .map(|memory| memory.memory)
                    .collect()
            }),
            dynamic: want_dynamic.then(|| {
                dynamic_memories
                    .into_iter()
                    .map(|memory| memory.memory)
                    .collect()
            }),
            buckets: want_buckets.then_some(bucket_values),
        },
        search_results,
    }))
}

async fn profile_buckets(
    Json(request): Json<ProfileBucketsRequest>,
) -> Result<Json<ProfileBucketsResponse>, ApiError> {
    validate_container_tag(Some(&request.container_tag))?;
    Ok(Json(ProfileBucketsResponse {
        buckets: vec![BucketDefinition {
            key: "preferences",
            description: PREFERENCES_DESCRIPTION,
        }],
    }))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ForgetMemoryRequest {
    id: Option<String>,
    content: Option<String>,
    container_tag: String,
    reason: Option<String>,
}

#[derive(Serialize)]
struct ForgetMemoryResponse {
    id: String,
    forgotten: bool,
}

async fn forget_memory(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<ForgetMemoryRequest>,
) -> Result<Json<ForgetMemoryResponse>, ApiError> {
    if request.id.is_none() && request.content.is_none() {
        return Err(ApiError::Validation("id or content is required"));
    }
    validate_container_tag(Some(&request.container_tag))?;
    let id = state
        .writer
        .forget_memory(
            organization.0,
            request.id,
            request.content,
            request.container_tag,
            request.reason,
        )
        .await
        .map_err(|_| ApiError::StorageUnavailable)?
        .map_err(|error| match error {
            storage::StorageError::MemoryNotFound => ApiError::MemoryNotFound,
            error => ApiError::Storage(error),
        })?;
    Ok(Json(ForgetMemoryResponse {
        id,
        forgotten: true,
    }))
}

/// Test and embedding-free worker façade. Ownership is explicit: callers inject
/// the runtime rather than creating an independent writer or model scheduler.
#[derive(Clone)]
pub struct IndexingWorker {
    coordinator: indexing_coordinator::IndexingCoordinator,
}

impl IndexingWorker {
    #[must_use]
    pub fn with_runtime(runtime: &ServerRuntime) -> Self {
        Self {
            coordinator: indexing_coordinator::IndexingCoordinator::new(
                runtime.writer(),
                runtime.embeddings(),
                None,
            ),
        }
    }

    /// # Errors
    /// Returns a durable indexing failure from the injected coordinator.
    pub async fn process_available(&self) -> Result<bool, WorkerError> {
        self.coordinator.process_available().await
    }
}

fn validate_request(request: &CreateDocumentRequest) -> Result<(), ApiError> {
    if request.content.trim().is_empty() {
        return Err(ApiError::Validation("content must not be empty"));
    }
    if request.container_tag.is_some() && request.container_tags.is_some() {
        return Err(ApiError::Validation(
            "containerTag and containerTags cannot both be provided",
        ));
    }
    if request
        .entity_context
        .as_ref()
        .is_some_and(|value| value.chars().count() > 1500)
    {
        return Err(ApiError::Validation(
            "entityContext must be at most 1500 characters",
        ));
    }
    if let Some(custom_id) = &request.custom_id
        && (custom_id.chars().count() > 100
            || custom_id.is_empty()
            || !custom_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'-')))
    {
        return Err(ApiError::Validation(
            "customId must be 1-100 characters containing only letters, numbers, _, :, or -",
        ));
    }
    if request.filepath.as_deref() == Some("/profile.md") {
        return Err(ApiError::Validation("filepath /profile.md is reserved"));
    }
    validate_metadata(&request.metadata)?;
    validate_metadata(&request.filter_by_metadata)
}

fn validate_metadata(values: &Map<String, Value>) -> Result<(), ApiError> {
    if values.values().all(|value| {
        matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_))
            || matches!(value, Value::Array(items) if items.iter().all(Value::is_string))
    }) {
        Ok(())
    } else {
        Err(ApiError::Validation(
            "metadata values must be strings, numbers, booleans, or arrays of strings",
        ))
    }
}

async fn shutdown_signal() {
    if tokio::signal::ctrl_c().await.is_err() {
        tracing::error!("failed to install process shutdown signal handler");
    }
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
}
#[derive(Serialize)]
struct DocumentResult {
    id: String,
    status: String,
}
#[derive(Serialize)]
struct ErrorBody {
    error: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<&'static str>,
}

struct AuthError;
impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorBody {
                error: "Unauthorized",
                details: Some("A valid Bearer API key is required"),
            }),
        )
            .into_response()
    }
}

#[derive(Debug, Error)]
enum ApiError {
    #[error("invalid document request: {0}")]
    Validation(&'static str),
    #[error("document not found")]
    NotFound,
    #[error("service is degraded after a fatal storage failure")]
    ServiceDegraded,
    #[error("storage lock is unavailable after a previous operation failed")]
    StorageUnavailable,
    #[error("storage operation failed: {0}")]
    Storage(#[source] storage::StorageError),
    #[error("database executor stopped before completing the operation: {0}")]
    DatabaseExecutor(#[source] tokio::task::JoinError),
    #[error("embedding executor failed: {0}")]
    EmbeddingExecutor(#[source] memory_engine::EmbeddingExecutorError),
    #[error("query embedding returned no vector")]
    EmptyEmbedding,
    #[error("reranking is not configured")]
    RerankingUnavailable,
    #[error("memory not found")]
    MemoryNotFound,
}

impl ApiError {
    fn is_fatal(&self) -> bool {
        matches!(self, Self::StorageUnavailable)
            || matches!(self, Self::Storage(error) if error.is_fatal())
            || matches!(self, Self::DatabaseExecutor(error) if error.is_panic())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let fatal = self.is_fatal();
        let mut response = match self {
            Self::Validation(details) => (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "Invalid request",
                    details: Some(details),
                }),
            )
                .into_response(),
            Self::NotFound => (
                StatusCode::NOT_FOUND,
                Json(ErrorBody {
                    error: "Document not found",
                    details: None,
                }),
            )
                .into_response(),
            Self::RerankingUnavailable => (
                StatusCode::NOT_IMPLEMENTED,
                Json(ErrorBody {
                    error: "Reranking is not configured",
                    details: Some("Configure a reranker before using rerank=true"),
                }),
            )
                .into_response(),
            Self::MemoryNotFound => (
                StatusCode::NOT_FOUND,
                Json(ErrorBody {
                    error: "Memory not found",
                    details: None,
                }),
            )
                .into_response(),
            Self::ServiceDegraded => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody {
                    error: "Service degraded",
                    details: None,
                }),
            )
                .into_response(),
            Self::StorageUnavailable
            | Self::Storage(_)
            | Self::DatabaseExecutor(_)
            | Self::EmbeddingExecutor(_)
            | Self::EmptyEmbedding => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: "Internal server error",
                    details: Some("The operation could not be completed; retry the request"),
                }),
            )
                .into_response(),
        };
        if fatal {
            response.extensions_mut().insert(FatalApiFailure);
        }
        response
    }
}

/// Failure while running the HTTP server.
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("failed to bind the HTTP server to {address}: {source}")]
    Bind {
        address: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("the HTTP server stopped unexpectedly: {0}")]
    Serve(#[source] std::io::Error),
}

/// Failure while claiming or completing queued document work.
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("storage lock is unavailable after a previous operation failed")]
    StorageUnavailable,
    #[error("document worker storage operation failed: {0}")]
    Storage(#[source] storage::StorageError),
    #[error("document worker executor stopped before completing the operation: {0}")]
    Executor(#[source] tokio::task::JoinError),
    #[error("document chunking failed: {0}")]
    Chunking(#[source] memory_engine::ChunkingError),
    #[error("document embedding failed: {0}")]
    Embedding(#[source] memory_engine::EmbeddingError),
    #[error("embedding executor failed: {0}")]
    EmbeddingExecutor(#[source] memory_engine::EmbeddingExecutorError),
    #[error("memory extraction failed: {0}")]
    Provider(#[source] memory_engine::ProviderError),
    #[error("cached memory extraction is invalid: {0}")]
    CachedExtraction(#[source] serde_json::Error),
}

impl WorkerError {
    fn is_fatal(&self) -> bool {
        matches!(self, Self::StorageUnavailable)
            || matches!(self, Self::Storage(error) if error.is_fatal())
            || matches!(self, Self::Executor(error) if error.is_panic())
    }
}
