//! HTTP boundary and durable worker lifecycle, including stop-on-fatal health degradation.

use std::{
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

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
use storage::{EmbeddedChunk, Storage, UpsertDocument};
use subtle::ConstantTimeEq;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::time::{Duration, Instant, sleep};

/// Safely shared application storage.
pub type SharedStorage = Arc<Mutex<Storage>>;

type SharedHealth = Arc<ServiceHealth>;

#[derive(Clone, Copy)]
struct FatalApiFailure;

#[derive(Default)]
struct ServiceHealth(AtomicBool);

impl ServiceHealth {
    fn is_degraded(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn degrade(&self) {
        self.0.store(true, Ordering::Release);
    }
}

#[derive(Clone)]
struct AppState {
    api_keys: Arc<Vec<([u8; 32], String)>>,
    api_key: Option<String>,
    port: u16,
    storage: SharedStorage,
    search_connections: Arc<Mutex<Vec<Storage>>>,
    search_permits: Arc<Semaphore>,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    local_org_id: String,
    health: SharedHealth,
}

#[derive(Clone)]
struct OrganizationId(String);

/// Builds the complete HTTP application.
pub fn router(api_key: Option<String>, storage: SharedStorage) -> Router {
    router_with_port(api_key, storage, None, 6767)
}

/// Builds the HTTP application with local semantic search enabled.
pub fn router_with_embeddings(
    api_key: Option<String>,
    storage: SharedStorage,
    embeddings: Arc<memory_engine::EmbeddingModel>,
) -> Router {
    router_with_port(api_key, storage, Some(embeddings), 6767)
}

fn router_with_port(
    api_key: Option<String>,
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    port: u16,
) -> Router {
    router_with_health(
        api_key,
        storage,
        embeddings,
        port,
        Arc::new(ServiceHealth::default()),
    )
}

fn router_with_health(
    api_key: Option<String>,
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    port: u16,
    health: SharedHealth,
) -> Router {
    let local_org_id = storage.lock().map_or_else(
        |_| String::new(),
        |storage| storage.local_organization_id().to_owned(),
    );
    let mut api_keys = Vec::new();
    if let Some(key) = api_key.as_ref() {
        api_keys.push((Sha256::digest(key).into(), local_org_id.clone()));
    }
    let state = AppState {
        api_keys: Arc::new(api_keys),
        api_key,
        port,
        storage,
        search_connections: Arc::new(Mutex::new(Vec::new())),
        search_permits: Arc::new(Semaphore::new(8)),
        embeddings,
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

async fn landing_page(State(state): State<AppState>) -> Response {
    let api_key = state.api_key.as_deref().unwrap_or("sm_your_local_api_key");
    let escaped_key = escape_html(api_key);
    let port = state.port;
    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>supermemory · local</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root{{--bg:#080a0c;--panel:#101317;--line:#242a31;--text:#eef3f6;--muted:#929ba5;--cyan:#55ddeb;--orange:#f79332}}
    *{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 20% 0%,#10242a 0,transparent 34%),var(--bg);color:var(--text);font-family:Inter,sans-serif}}
    main{{max-width:1060px;margin:auto;padding:72px 28px}} .brand{{font:700 clamp(42px,8vw,88px)/.9 'Space Grotesk';letter-spacing:-.065em}} .brand span{{color:var(--orange)}}
    .eyebrow{{color:var(--cyan);font:500 13px 'JetBrains Mono';text-transform:uppercase;letter-spacing:.16em;margin-bottom:20px}} h1{{font:600 clamp(32px,5vw,58px)/1.05 'Space Grotesk';max-width:760px;margin:42px 0 18px}}
    .lede{{max-width:700px;color:var(--muted);font-size:18px;line-height:1.7}} .status{{display:flex;gap:10px;align-items:center;color:#90efb1;margin:32px 0 50px;font:500 14px 'JetBrains Mono'}} .dot{{width:8px;height:8px;border-radius:50%;background:#62e893;box-shadow:0 0 18px #62e893}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}} .card{{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:14px;padding:24px}}
    .card h2{{font:600 19px 'Space Grotesk';margin:0 0 8px}} .card p{{color:var(--muted);line-height:1.55;margin:0 0 18px}} pre{{position:relative;overflow:auto;background:#080a0d;border:1px solid #20262d;border-radius:9px;padding:18px 48px 18px 16px;color:#b9f5ef;font:13px/1.65 'JetBrains Mono'}}
    button{{position:absolute;right:8px;top:8px;border:1px solid #303842;background:#171c21;color:#c9d1d9;border-radius:6px;padding:6px 8px;cursor:pointer}} a{{color:var(--cyan);text-decoration:none}} nav{{display:flex;gap:24px;flex-wrap:wrap;margin-top:38px;padding-top:28px;border-top:1px solid var(--line)}}
  </style>
</head>
<body><main>
  <div class="eyebrow">local · self-hosted · running on this machine</div>
  <div class="brand">supermemory<span>-RS</span></div>
  <h1>Your memory infrastructure, running locally.</h1>
  <p class="lede">Add documents and search your local Supermemory-compatible server. Your local API key is ready to use in SDKs and command-line requests.</p>
  <div class="status"><i class="dot"></i> listening on http://localhost:{port}</div>
  <section class="grid">
    <article class="card"><h2>Add a memory</h2><p>Send text to the document ingestion endpoint.</p><pre><button class="copy-btn">copy</button><code>curl -X POST http://localhost:{port}/v3/documents \
  -H 'Authorization: Bearer {escaped_key}' \
  -H 'Content-Type: application/json' \
  -d '{{"content":"Remember this locally"}}'</code></pre></article>
    <article class="card"><h2>Search</h2><p>Search completed documents through the V4 endpoint.</p><pre><button class="copy-btn">copy</button><code>curl -X POST http://localhost:{port}/v4/search \
  -H 'Authorization: Bearer {escaped_key}' \
  -H 'Content-Type: application/json' \
  -d '{{"q":"remember","searchMode":"documents"}}'</code></pre></article>
    <article class="card"><h2>Local API key</h2><p>Use this key for SDKs or non-loopback clients.</p><pre><button class="copy-btn">copy</button><code>{escaped_key}</code></pre></article>
  </section>
  <nav><a href="/v4/reference">API reference</a><a href="/v4/openapi">OpenAPI document</a><a href="https://supermemory.ai/docs/self-hosting/overview">Self-hosting docs</a><a href="https://github.com/supermemoryai/supermemory">GitHub</a></nav>
</main><script>document.querySelectorAll('.copy-btn').forEach((button)=>button.addEventListener('click',async()=>{{const text=button.parentElement.querySelector('code').textContent;await navigator.clipboard.writeText(text);button.textContent='copied';setTimeout(()=>button.textContent='copy',1200)}}));</script></body></html>"#
    );
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    (headers, Html(html)).into_response()
}

async fn api_reference() -> Html<&'static str> {
    Html(
        r#"<!doctype html><html><head><title>supermemory API reference</title><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><script id="api-reference" data-url="/v4/openapi"></script><script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script></body></html>"#,
    )
}

async fn openapi(State(state): State<AppState>) -> Json<Value> {
    let server_url = format!("http://localhost:{}", state.port);
    Json(serde_json::json!({
        "openapi": "3.1.0",
        "info": { "title": "supermemory local API", "version": "0.0.5-rs" },
        "servers": [{ "url": server_url }],
        "paths": {
            "/health": { "get": { "responses": { "200": { "description": "Healthy" }, "503": { "description": "Degraded after a fatal worker or storage request failure" } } } },
            "/v3/documents": { "post": { "summary": "Add a document", "responses": { "200": { "description": "Queued document" }, "503": { "description": "Service degraded" } } } },
            "/v3/documents/{id}": { "get": { "summary": "Get a document", "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }], "responses": { "200": { "description": "Document" }, "404": { "description": "Not found" } } } },
            "/v3/search": { "post": { "summary": "Search document chunks", "responses": { "200": { "description": "Search results" }, "503": { "description": "Service degraded" } } } },
            "/v4/search": { "post": { "summary": "Search local documents", "responses": { "200": { "description": "Search results" }, "503": { "description": "Service degraded" } } } },
            "/v4/memories/": { "delete": { "summary": "Forget a memory", "responses": { "200": { "description": "Memory forgotten" }, "503": { "description": "Service degraded" } } } }
        }
    }))
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
    storage: SharedStorage,
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
    storage: SharedStorage,
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
    storage: SharedStorage,
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
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    provider: Option<Arc<memory_engine::MemoryProvider>>,
    ready: impl FnOnce(),
) -> Result<(), ServerError> {
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;
    ready();
    let health = Arc::new(ServiceHealth::default());
    let worker = tokio::spawn(worker_loop(
        Arc::clone(&storage),
        embeddings.as_ref().map(Arc::clone),
        provider.as_ref().map(Arc::clone),
        Arc::clone(&health),
    ));
    let memory_worker = provider
        .as_ref()
        .zip(embeddings.as_ref())
        .map(|(provider, embeddings)| {
            tokio::spawn(memory_worker_loop(
                Arc::clone(&storage),
                Arc::clone(embeddings),
                Arc::clone(provider),
                Arc::clone(&health),
            ))
        });
    let result = axum::serve(
        listener,
        router_with_health(api_key, storage, embeddings, address.port(), health)
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

/// Makes fatal request-path failures sticky so subsequent health and guarded APIs return 503.
async fn degrade_after_internal_error(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let response = next.run(request).await;
    if response.extensions().get::<FatalApiFailure>().is_some() {
        state.health.degrade();
    }
    response
}

/// Reports the sticky service health state without exposing internal failure details.
async fn health_handler(State(state): State<AppState>) -> Response {
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

async fn create_document(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<CreateDocumentRequest>,
) -> Result<Json<DocumentResult>, ApiError> {
    if state.health.is_degraded() {
        return Err(ApiError::ServiceDegraded);
    }
    validate_request(&request)?;
    let content = storage::sanitize_content(&request.content);
    if content.is_empty() {
        return Err(ApiError::Validation(
            "content must not be empty after sanitization",
        ));
    }
    let tags = request.container_tag.clone().map_or_else(
        || {
            request
                .container_tags
                .clone()
                .unwrap_or_else(|| vec!["sm_project_default".to_owned()])
        },
        |tag| vec![tag],
    );
    let document = UpsertDocument {
        content,
        custom_id: request.custom_id,
        container_tags: tags,
        entity_context: request.entity_context,
        metadata: request.metadata,
        task_type: request.task_type.as_str().to_owned(),
        filepath: request.filepath,
        filter_by_metadata: request.filter_by_metadata,
        dreaming: request.dreaming.as_str().to_owned(),
    };
    let storage = Arc::clone(&state.storage);
    let result = tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .upsert_document_for(&organization.0, document)
            .map_err(ApiError::Storage)
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??;
    Ok(Json(DocumentResult {
        id: result.id,
        status: result.status,
    }))
}

async fn get_document(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Path(id): Path<String>,
) -> Result<Json<storage::Document>, ApiError> {
    let storage = Arc::clone(&state.storage);
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .find_document_for(&organization.0, &id)
            .map_err(ApiError::Storage)
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??
    .map(Json)
    .ok_or(ApiError::NotFound)
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

async fn v3_search(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<V3SearchRequest>,
) -> Result<Json<V3SearchResponse>, ApiError> {
    if state.health.is_degraded() {
        return Err(ApiError::ServiceDegraded);
    }
    let started = std::time::Instant::now();
    validate_search_basics(&request.q, request.limit, request.chunk_threshold)?;
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
    let filters = request.filters.as_ref().map(parse_filter).transpose()?;
    let tags = request.container_tag.map_or_else(
        || {
            request
                .container_tags
                .unwrap_or_else(|| vec!["sm_project_default".to_owned()])
        },
        |tag| vec![tag],
    );
    let query = request.q.trim().to_owned();
    let query_vector = embed_query(state.embeddings.as_ref(), query.clone()).await?;
    let candidate_limit = if request.rerank {
        request.limit.max((request.limit * 3).min(30))
    } else {
        request.limit
    };
    let options = storage::SearchOptions {
        organization_id: Some(organization.0.clone()),
        container_tags: tags,
        document_id: request.doc_id,
        filepath: request.filepath,
        filters,
    };
    let storage = Arc::clone(&state.storage);
    let connections = Arc::clone(&state.search_connections);
    let permit = Arc::clone(&state.search_permits)
        .acquire_owned()
        .await
        .map_err(|_| ApiError::StorageUnavailable)?;
    let hits = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let connection = connections
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .pop();
        let connection = match connection {
            Some(connection) => connection,
            None => storage
                .lock()
                .map_err(|_| ApiError::StorageUnavailable)?
                .fork()
                .map_err(ApiError::Storage)?,
        };
        let result = query_vector
            .map_or_else(
                || connection.search_for(&organization.0, &query, candidate_limit),
                |vector| {
                    connection.search_semantic(
                        vector.as_slice(),
                        "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                        candidate_limit,
                        candidate_limit.saturating_mul(10).max(100),
                        request.chunk_threshold,
                        &options,
                    )
                },
            )
            .map_err(ApiError::Storage);
        connections
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .push(connection);
        result
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??;
    let results = group_v3_results(
        hits,
        request.limit,
        request.include_summary,
        request.include_full_docs,
    );
    let total = results.iter().map(|result| result.chunks.len()).sum();
    Ok(Json(V3SearchResponse {
        results,
        timing: started.elapsed().as_secs_f64() * 1_000.0,
        total,
    }))
}

fn group_v3_results(
    hits: Vec<storage::SearchHit>,
    limit: usize,
    include_summary: bool,
    include_full_docs: bool,
) -> Vec<V3DocumentResult> {
    let mut results: Vec<V3DocumentResult> = Vec::new();
    for hit in hits.into_iter().take(limit) {
        let chunk = V3ChunkResult {
            content: hit.chunk,
            is_relevant: true,
            score: hit.score,
            position: hit.position,
        };
        if let Some(existing) = results
            .iter_mut()
            .find(|result| result.document_id == hit.document_id)
        {
            existing.score = existing.score.max(hit.score);
            existing.chunks.push(chunk);
            continue;
        }
        let summary = include_summary
            .then(|| metadata_string(&hit.metadata, "summary"))
            .flatten();
        let content = include_full_docs.then_some(hit.document_content);
        results.push(V3DocumentResult {
            chunks: vec![chunk],
            created_at: hit.created_at,
            document_id: hit.document_id,
            metadata: Some(hit.metadata.clone()),
            score: hit.score,
            summary,
            content,
            title: metadata_string(&hit.metadata, "title"),
            updated_at: hit.updated_at,
            document_type: metadata_string(&hit.metadata, "type"),
        });
    }
    results
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

async fn embed_query(
    embeddings: Option<&Arc<memory_engine::EmbeddingModel>>,
    query: String,
) -> Result<Option<memory_engine::EmbeddingVector>, ApiError> {
    let Some(embeddings) = embeddings else {
        return Ok(None);
    };
    let embeddings = Arc::clone(embeddings);
    tokio::task::spawn_blocking(move || embeddings.embed(&[query]))
        .await
        .map_err(ApiError::DatabaseExecutor)?
        .map_err(ApiError::Embedding)?
        .into_iter()
        .next()
        .map(Some)
        .ok_or(ApiError::EmptyEmbedding)
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
    #[serde(default, rename = "documents")]
    _documents: bool,
    #[serde(default, rename = "summaries")]
    _summaries: bool,
    #[serde(default, rename = "relatedMemories")]
    _related_memories: bool,
    #[serde(default)]
    forgotten_memories: bool,
    #[serde(default)]
    chunks: bool,
}

#[derive(Default, Deserialize, PartialEq, Eq)]
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

impl SearchResult {
    fn similarity(&self) -> f64 {
        match self {
            Self::Memory(result) => result.similarity,
            Self::Chunk(result) => result.similarity,
        }
    }
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

#[expect(
    clippy::too_many_lines,
    reason = "v4 search contract validation and orchestration stay together"
)]
async fn search(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if state.health.is_degraded() {
        return Err(ApiError::ServiceDegraded);
    }
    let started = std::time::Instant::now();
    if request.q.trim().is_empty() {
        return Err(ApiError::Validation("q must not be empty"));
    }
    if !(1..=100).contains(&request.limit) {
        return Err(ApiError::Validation("limit must be between 1 and 100"));
    }
    if !request.threshold.is_finite() || !(0.0..=1.0).contains(&request.threshold) {
        return Err(ApiError::Validation(
            "threshold must be a finite number between 0 and 1",
        ));
    }
    if request.aggregate && request.rerank {
        return Err(ApiError::AggregateAndRerank);
    }
    validate_container_tag(request.container_tag.as_deref())?;
    let filters = request.filters.as_ref().map(parse_filter).transpose()?;
    let search_mode = if request.include.chunks && request.search_mode == SearchMode::Memories {
        SearchMode::Hybrid
    } else {
        request.search_mode
    };
    let query = request.q.trim().to_owned();
    let embedding_started = std::time::Instant::now();
    let query_vector = if let Some(embeddings) = state.embeddings.as_ref() {
        let embeddings = Arc::clone(embeddings);
        let value = query.clone();
        Some(
            tokio::task::spawn_blocking(move || embeddings.embed(&[value]))
                .await
                .map_err(ApiError::DatabaseExecutor)?
                .map_err(ApiError::Embedding)?
                .into_iter()
                .next()
                .ok_or(ApiError::EmptyEmbedding)?,
        )
    } else {
        None
    };
    let embedding_elapsed = embedding_started.elapsed();
    let storage = Arc::clone(&state.storage);
    let limit = request.limit;
    let threshold = request.threshold;
    let include_forgotten = request.include.forgotten_memories;
    let memory_mode = search_mode != SearchMode::Documents;
    let document_mode = search_mode != SearchMode::Memories;
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
    let candidate_limit = limit.saturating_mul(if request.aggregate { 5 } else { 3 });
    let connections = Arc::clone(&state.search_connections);
    let permit = Arc::clone(&state.search_permits)
        .acquire_owned()
        .await
        .map_err(|_| ApiError::StorageUnavailable)?;
    let storage_started = std::time::Instant::now();
    let (memory_hits, chunk_hits) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let connection = connections
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .pop();
        let connection = match connection {
            Some(connection) => connection,
            None => storage
                .lock()
                .map_err(|_| ApiError::StorageUnavailable)?
                .fork()
                .map_err(ApiError::Storage)?,
        };
        let result = if let Some(vector) = query_vector {
            let memories = if memory_mode {
                connection
                    .search_memories_for(
                        &organization.0,
                        vector.as_slice(),
                        "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                        options
                            .container_tags
                            .first()
                            .map_or("sm_project_default", String::as_str),
                        candidate_limit,
                        candidate_limit.saturating_mul(10).max(100),
                        threshold,
                        include_forgotten,
                    )
                    .map_err(ApiError::Storage)?
            } else {
                Vec::new()
            };
            let chunks = if document_mode {
                connection
                    .search_semantic(
                        vector.as_slice(),
                        "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                        candidate_limit,
                        candidate_limit.saturating_mul(10).max(100),
                        threshold,
                        &options,
                    )
                    .map_err(ApiError::Storage)?
            } else {
                Vec::new()
            };
            Ok((memories, chunks))
        } else {
            let chunks = if document_mode {
                connection
                    .search_for(&organization.0, &query, candidate_limit)
                    .map_err(ApiError::Storage)?
            } else {
                Vec::new()
            };
            Ok((Vec::new(), chunks))
        };
        connections
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .push(connection);
        result
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??;
    let storage_elapsed = storage_started.elapsed();
    let memory_hit_count = memory_hits.len();
    let chunk_hit_count = chunk_hits.len();
    let memory_boost = if search_mode == SearchMode::Hybrid {
        1.15
    } else {
        1.0
    };
    let mut results: Vec<_> = memory_hits
        .into_iter()
        .map(|hit| SearchResult::Memory(memory_search_result(hit, memory_boost)))
        .chain(
            chunk_hits
                .into_iter()
                .map(|hit| SearchResult::Chunk(chunk_search_result(hit))),
        )
        .collect();
    results.retain(|result| result.similarity() >= f64::from(threshold));
    results.sort_by(|left, right| right.similarity().total_cmp(&left.similarity()));
    results.truncate(limit);
    let total = results.len();
    let total_elapsed = started.elapsed();
    tracing::info!(
        embedding_ms = embedding_elapsed.as_millis(),
        storage_ms = storage_elapsed.as_millis(),
        response_ms = total_elapsed
            .saturating_sub(embedding_elapsed)
            .saturating_sub(storage_elapsed)
            .as_millis(),
        total_ms = total_elapsed.as_millis(),
        memory_hit_count,
        chunk_hit_count,
        result_count = total,
        "search completed"
    );
    Ok(Json(SearchResponse {
        results,
        timing: total_elapsed.as_secs_f64() * 1_000.0,
        total,
    }))
}

fn memory_search_result(hit: storage::MemorySearchHit, boost: f64) -> MemorySearchResult {
    let parents = hit
        .parents
        .into_iter()
        .filter_map(|relation| serde_json::to_value(relation).ok())
        .collect();
    let children = hit
        .children
        .into_iter()
        .filter_map(|relation| serde_json::to_value(relation).ok())
        .collect();
    let related = hit
        .related
        .into_iter()
        .filter_map(|relation| serde_json::to_value(relation).ok())
        .collect();
    let documents = hit
        .documents
        .into_iter()
        .map(|document| SearchDocument {
            id: document.id,
            title: document.title,
            document_type: document.document_type,
            metadata: Some(document.metadata),
            summary: document.summary,
            created_at: document.created_at,
            updated_at: document.updated_at,
        })
        .collect();
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
    let storage = Arc::clone(&state.storage);
    let container_tag = request.container_tag.clone();
    let profile_org_id = organization.0.clone();
    let requested_buckets = request
        .buckets
        .unwrap_or_else(|| vec!["preferences".to_owned()]);
    let want_static = include.contains(&ProfileSection::Static);
    let want_dynamic = include.contains(&ProfileSection::Dynamic);
    let want_buckets = include.contains(&ProfileSection::Buckets);
    let (static_memories, dynamic_memories, buckets) = tokio::task::spawn_blocking(move || {
        let storage = storage.lock().map_err(|_| ApiError::StorageUnavailable)?;
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
    .await
    .map_err(ApiError::DatabaseExecutor)??;
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
        Some(
            profile_search(
                &state,
                &organization.0,
                query,
                &request.container_tag,
                request.threshold,
            )
            .await?,
        )
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
                    .map(|memory| {
                        format!(
                            "[{}] {}",
                            memory.created_at.get(..10).unwrap_or(&memory.created_at),
                            memory.memory
                        )
                    })
                    .collect()
            }),
            buckets: want_buckets.then_some(bucket_values),
        },
        search_results,
    }))
}

async fn profile_search(
    state: &AppState,
    organization_id: &str,
    query: String,
    container_tag: &str,
    threshold: f32,
) -> Result<SearchResponse, ApiError> {
    let started = std::time::Instant::now();
    let Some(vector) = embed_query(state.embeddings.as_ref(), query).await? else {
        return Ok(SearchResponse {
            results: Vec::new(),
            timing: 0.0,
            total: 0,
        });
    };
    let storage = Arc::clone(&state.storage);
    let container_tag = container_tag.to_owned();
    let organization_id = organization_id.to_owned();
    let hits = tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .search_memories_for(
                &organization_id,
                vector.as_slice(),
                "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                &container_tag,
                15,
                150,
                threshold,
                false,
            )
            .map_err(ApiError::Storage)
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??;
    let results = hits
        .into_iter()
        .map(|hit| SearchResult::Memory(memory_search_result(hit, 1.0)))
        .collect::<Vec<_>>();
    let total = results.len();
    Ok(SearchResponse {
        results,
        timing: started.elapsed().as_secs_f64() * 1_000.0,
        total,
    })
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
    if state.health.is_degraded() {
        return Err(ApiError::ServiceDegraded);
    }
    if request.id.is_none() && request.content.is_none() {
        return Err(ApiError::Validation("id or content is required"));
    }
    validate_container_tag(Some(&request.container_tag))?;
    let storage = Arc::clone(&state.storage);
    let id = tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .forget_memory_for(
                &organization.0,
                request.id.as_deref(),
                request.content.as_deref(),
                &request.container_tag,
                request.reason.as_deref(),
            )
            .map_err(|error| match error {
                storage::StorageError::MemoryNotFound => ApiError::MemoryNotFound,
                error => ApiError::Storage(error),
            })
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??;
    Ok(Json(ForgetMemoryResponse {
        id,
        forgotten: true,
    }))
}

/// Processes at most one queued document, returning whether work was claimed.
///
/// # Errors
/// Returns an error if the storage lock is poisoned or a database operation fails.
pub async fn process_next_job(storage: SharedStorage) -> Result<bool, WorkerError> {
    process_next_job_with_embeddings(storage, None).await
}

/// Processes one job through chunking, local embedding, and atomic indexing.
///
/// # Errors
/// Returns an error if any durable stage or model operation fails.
pub async fn process_next_job_with_embeddings(
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
) -> Result<bool, WorkerError> {
    process_next_job_with_services(storage, embeddings, None).await
}

/// Processes one document and optionally extracts and reconciles memories.
///
/// # Errors
/// Returns an error if durable document publication fails.
#[expect(
    clippy::too_many_lines,
    reason = "durable document stages and their timings remain explicit"
)]
pub async fn process_next_job_with_services(
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    provider: Option<Arc<memory_engine::MemoryProvider>>,
) -> Result<bool, WorkerError> {
    let started = Instant::now();
    let claim_started = Instant::now();
    let job = tokio::task::spawn_blocking({
        let storage = Arc::clone(&storage);
        move || {
            storage
                .lock()
                .map_err(|_| WorkerError::StorageUnavailable)?
                .claim_job()
                .map_err(WorkerError::Storage)
        }
    })
    .await
    .map_err(WorkerError::Executor)??;
    let claim_elapsed = claim_started.elapsed();
    let Some(job) = job else {
        return Ok(false);
    };

    let chunking_started = Instant::now();
    mark_job_stage(Arc::clone(&storage), job.clone(), "chunking").await?;
    let chunks = match memory_engine::chunk_text(&job.content, None) {
        Ok(chunks) => chunks,
        Err(error) => {
            let message = error.to_string();
            persist_job_failure(Arc::clone(&storage), job.clone(), message).await?;
            return Err(WorkerError::Chunking(error));
        }
    };
    let chunking_elapsed = chunking_started.elapsed();
    if chunks.is_empty() {
        tokio::task::spawn_blocking(move || {
            storage
                .lock()
                .map_err(|_| WorkerError::StorageUnavailable)?
                .delete_empty_document(&job)
                .map_err(WorkerError::Storage)
        })
        .await
        .map_err(WorkerError::Executor)??;
        return Ok(true);
    }
    if let Some(embeddings) = embeddings {
        let embedding_started = Instant::now();
        mark_job_stage(Arc::clone(&storage), job.clone(), "embedding").await?;
        let values = chunks.clone();
        let document_embeddings = Arc::clone(&embeddings);
        let vectors = match tokio::task::spawn_blocking(move || document_embeddings.embed(&values))
            .await
            .map_err(WorkerError::Executor)?
        {
            Ok(vectors) => vectors,
            Err(error) => {
                let message = error.to_string();
                persist_job_failure(Arc::clone(&storage), job.clone(), message).await?;
                return Err(WorkerError::Embedding(error));
            }
        };
        let embedding_elapsed = embedding_started.elapsed();
        let indexing_started = Instant::now();
        mark_job_stage(Arc::clone(&storage), job.clone(), "indexing").await?;
        let publication_storage = Arc::clone(&storage);
        let publication_job = job.clone();
        let chunk_count = chunks.len();
        tokio::task::spawn_blocking(move || {
            let embedded: Vec<_> = chunks
                .iter()
                .zip(&vectors)
                .map(|(content, vector)| EmbeddedChunk {
                    content,
                    vector: vector.as_slice(),
                })
                .collect();
            let mut storage = publication_storage
                .lock()
                .map_err(|_| WorkerError::StorageUnavailable)?;
            if provider.is_some() {
                storage.complete_embedded_job_with_memory_extraction(
                    &publication_job,
                    &embedded,
                    "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                    memory_engine::BGE_DIMENSIONS,
                )
            } else {
                storage.complete_embedded_job(
                    &publication_job,
                    &embedded,
                    "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                    memory_engine::BGE_DIMENSIONS,
                )
            }
            .map_err(WorkerError::Storage)
        })
        .await
        .map_err(WorkerError::Executor)??;
        let indexing_elapsed = indexing_started.elapsed();
        tracing::info!(
            job_id = %job.id,
            document_id = %job.document_id,
            chunk_count,
            claim_ms = claim_elapsed.as_millis(),
            chunking_ms = chunking_elapsed.as_millis(),
            embedding_ms = embedding_elapsed.as_millis(),
            indexing_ms = indexing_elapsed.as_millis(),
            total_ms = started.elapsed().as_millis(),
            "document job completed"
        );
        return Ok(true);
    }
    let completed_job_id = job.id.clone();
    let completed_document_id = job.document_id.clone();
    let chunk_count = chunks.len();
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .complete_job(&job, &chunks)
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
    tracing::info!(
        job_id = %completed_job_id,
        document_id = %completed_document_id,
        chunk_count,
        claim_ms = claim_elapsed.as_millis(),
        chunking_ms = chunking_elapsed.as_millis(),
        total_ms = started.elapsed().as_millis(),
        "document job completed"
    );
    Ok(true)
}

async fn mark_job_stage(
    storage: SharedStorage,
    job: storage::ClaimedJob,
    stage: &'static str,
) -> Result<(), WorkerError> {
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .mark_job_stage(&job, stage)
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
    Ok(())
}

async fn reconcile_extracted_memories(
    storage: SharedStorage,
    job: &storage::ClaimedMemoryJob,
    candidates: Vec<memory_engine::MemoryCandidate>,
    vectors: Vec<memory_engine::EmbeddingVector>,
) -> Result<(), WorkerError> {
    let document_id = job.document_id.clone();
    let organization_id = job.organization_id.clone();
    let job_revision = job.revision;
    let container_tag = job.container_tag.clone();
    let proposals = candidates
        .into_iter()
        .zip(vectors)
        .map(|(candidate, vector)| {
            let mut metadata = Map::new();
            metadata.insert("buckets".to_owned(), serde_json::json!(candidate.buckets));
            if let Some(temporal) = candidate.temporal_context {
                metadata.insert(
                    "temporalContext".to_owned(),
                    serde_json::to_value(temporal).unwrap_or(Value::Null),
                );
            }
            storage::MemoryProposal {
                temporary_id: candidate.tmp_id,
                content: candidate.memory,
                is_inferred: candidate.is_inferred,
                is_static: candidate.add_to_static_profile,
                metadata,
                parents: candidate
                    .parent_relations
                    .into_iter()
                    .map(|parent| storage::MemoryParent {
                        memory_id: parent.memory_id,
                        relation: match parent.relation {
                            memory_engine::RelationKind::Updates => "updates",
                            memory_engine::RelationKind::Extends => "extends",
                            memory_engine::RelationKind::Derives => "derives",
                        }
                        .to_owned(),
                    })
                    .collect(),
                forget_after: candidate.forget_after,
                forget_reason: candidate.forget_reason,
                vector: vector.as_slice().to_vec(),
            }
        })
        .collect::<Vec<_>>();
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .reconcile_memories_for(
                &organization_id,
                &document_id,
                Some(job_revision),
                &container_tag,
                &proposals,
                "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                memory_engine::BGE_DIMENSIONS,
            )
            .map(|_| ())
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
    Ok(())
}

async fn persist_job_failure(
    storage: SharedStorage,
    job: storage::ClaimedJob,
    message: String,
) -> Result<(), WorkerError> {
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .fail_job(&job, &message)
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
    Ok(())
}

/// Resumes one memory job from its persisted extraction or performs one provider attempt.
#[expect(
    clippy::too_many_lines,
    reason = "durable cache, retry, embedding, and publication transitions remain explicit"
)]
async fn process_next_memory_job(
    storage: SharedStorage,
    embeddings: Arc<memory_engine::EmbeddingModel>,
    provider: Arc<memory_engine::MemoryProvider>,
) -> Result<bool, WorkerError> {
    let started = Instant::now();
    let claim_started = Instant::now();
    let job = tokio::task::spawn_blocking({
        let storage = Arc::clone(&storage);
        move || {
            storage
                .lock()
                .map_err(|_| WorkerError::StorageUnavailable)?
                .claim_memory_job()
                .map_err(WorkerError::Storage)
        }
    })
    .await
    .map_err(WorkerError::Executor)??;
    let claim_elapsed = claim_started.elapsed();
    let Some(job) = job else {
        return Ok(false);
    };
    let mut context_elapsed = Duration::ZERO;
    let mut extraction_elapsed = Duration::ZERO;
    let mut extraction_usage = None;
    let candidates = if let Some(cached) = job.extraction_result.as_deref() {
        match serde_json::from_str(cached) {
            Ok(candidates) => candidates,
            Err(error) => {
                retry_memory_job(
                    Arc::clone(&storage),
                    job.clone(),
                    "invalid_output",
                    None,
                    error.to_string(),
                )
                .await?;
                return Err(WorkerError::CachedExtraction(error));
            }
        }
    } else {
        let context_started = Instant::now();
        let existing = tokio::task::spawn_blocking({
            let storage = Arc::clone(&storage);
            let org = job.organization_id.clone();
            let tag = job.container_tag.clone();
            move || {
                storage
                    .lock()
                    .map_err(|_| WorkerError::StorageUnavailable)?
                    .existing_memories_for_extraction(&org, &tag)
                    .map_err(WorkerError::Storage)
            }
        })
        .await
        .map_err(WorkerError::Executor)??;
        context_elapsed = context_started.elapsed();
        let context: Vec<_> = existing
            .into_iter()
            .map(|memory| (memory.id, memory.content))
            .collect();
        let extraction_started = Instant::now();
        let extraction = provider
            .extract_once_with_usage(&job.content, job.document_date.as_deref(), &context)
            .await;
        extraction_elapsed = extraction_started.elapsed();
        let candidates = match extraction {
            Ok(outcome) => {
                extraction_usage = Some(outcome.usage);
                outcome.memories
            }
            Err(error) => {
                tracing::warn!(
                    job_id = %job.id,
                    document_id = %job.document_id,
                    attempt = job.attempts,
                    context_ms = context_elapsed.as_millis(),
                    extraction_ms = extraction_elapsed.as_millis(),
                    total_ms = started.elapsed().as_millis(),
                    %error,
                    "memory extraction attempt failed"
                );
                let failure = error.failure();
                let retry_delay = extraction_retry_delay(failure, job.attempts);
                retry_memory_job(
                    Arc::clone(&storage),
                    job.clone(),
                    extraction_failure_name(failure),
                    retry_delay,
                    error.to_string(),
                )
                .await?;
                return Err(WorkerError::Provider(error));
            }
        };
        let cached = serde_json::to_string(&candidates).map_err(WorkerError::CachedExtraction)?;
        tokio::task::spawn_blocking({
            let storage = Arc::clone(&storage);
            let job = job.clone();
            move || {
                storage
                    .lock()
                    .map_err(|_| WorkerError::StorageUnavailable)?
                    .cache_memory_extraction(&job, &cached)
                    .map_err(WorkerError::Storage)
            }
        })
        .await
        .map_err(WorkerError::Executor)??;
        candidates
    };
    let candidate_count = candidates.len();
    let mut embedding_elapsed = Duration::ZERO;
    let mut reconciliation_elapsed = Duration::ZERO;
    if !candidates.is_empty() {
        let values = candidates
            .iter()
            .map(|candidate| candidate.memory.clone())
            .collect::<Vec<_>>();
        let embedding_started = Instant::now();
        let embedding_result = tokio::task::spawn_blocking(move || embeddings.embed(&values))
            .await
            .map_err(WorkerError::Executor)?;
        embedding_elapsed = embedding_started.elapsed();
        let vectors = match embedding_result {
            Ok(vectors) => vectors,
            Err(error) => {
                retry_memory_job(
                    Arc::clone(&storage),
                    job.clone(),
                    "embedding",
                    Some(Duration::from_secs(1)),
                    error.to_string(),
                )
                .await?;
                return Err(WorkerError::Embedding(error));
            }
        };
        let reconciliation_started = Instant::now();
        let reconciliation =
            reconcile_extracted_memories(Arc::clone(&storage), &job, candidates, vectors).await;
        reconciliation_elapsed = reconciliation_started.elapsed();
        match reconciliation {
            Ok(()) | Err(WorkerError::Storage(storage::StorageError::StaleRevision { .. })) => {}
            Err(error) => {
                retry_memory_job(
                    Arc::clone(&storage),
                    job.clone(),
                    "reconciliation",
                    Some(Duration::from_secs(1)),
                    error.to_string(),
                )
                .await?;
                return Err(error);
            }
        }
    }
    let completion_started = Instant::now();
    let completed_job_id = job.id.clone();
    let completed_document_id = job.document_id.clone();
    let completed_attempts = job.attempts;
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .complete_memory_job(&job)
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
    let completion_elapsed = completion_started.elapsed();
    tracing::info!(
        job_id = %completed_job_id,
        document_id = %completed_document_id,
        attempt = completed_attempts,
        candidate_count,
        input_tokens = extraction_usage.map_or(0, |usage| usage.input_tokens),
        output_tokens = extraction_usage.map_or(0, |usage| usage.output_tokens),
        total_tokens = extraction_usage.map_or(0, |usage| usage.total_tokens),
        reasoning_tokens = extraction_usage.map_or(0, |usage| usage.reasoning_tokens),
        claim_ms = claim_elapsed.as_millis(),
        context_ms = context_elapsed.as_millis(),
        extraction_ms = extraction_elapsed.as_millis(),
        embedding_ms = embedding_elapsed.as_millis(),
        reconciliation_ms = reconciliation_elapsed.as_millis(),
        completion_ms = completion_elapsed.as_millis(),
        total_ms = started.elapsed().as_millis(),
        "memory job completed"
    );
    Ok(true)
}

/// Maps semantic extraction failures to the durable queue's bounded retry schedule.
fn extraction_retry_delay(
    failure: memory_engine::ExtractionFailure,
    attempt: i32,
) -> Option<Duration> {
    match failure {
        memory_engine::ExtractionFailure::Authentication
        | memory_engine::ExtractionFailure::Configuration => None,
        memory_engine::ExtractionFailure::InvalidOutput if attempt >= 2 => None,
        memory_engine::ExtractionFailure::InvalidOutput => Some(Duration::from_secs(1)),
        memory_engine::ExtractionFailure::RateLimited
        | memory_engine::ExtractionFailure::Transport
        | memory_engine::ExtractionFailure::Provider => {
            Some(Duration::from_secs(1_u64 << attempt.clamp(0, 6)))
        }
    }
}

/// Returns the stable error kind stored with durable memory jobs.
const fn extraction_failure_name(failure: memory_engine::ExtractionFailure) -> &'static str {
    match failure {
        memory_engine::ExtractionFailure::RateLimited => "rate_limited",
        memory_engine::ExtractionFailure::Transport => "transport",
        memory_engine::ExtractionFailure::InvalidOutput => "invalid_output",
        memory_engine::ExtractionFailure::Authentication => "authentication",
        memory_engine::ExtractionFailure::Configuration => "configuration",
        memory_engine::ExtractionFailure::Provider => "provider",
    }
}

/// Persists a retry delay or terminal extraction failure before returning to the worker loop.
async fn retry_memory_job(
    storage: SharedStorage,
    job: storage::ClaimedMemoryJob,
    failure_kind: &'static str,
    retry_delay: Option<Duration>,
    message: String,
) -> Result<(), WorkerError> {
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .retry_memory_job(&job, failure_kind, &message, retry_delay)
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
    Ok(())
}

/// Runs memory jobs until shutdown or a fatal storage failure degrades the service.
async fn memory_worker_loop(
    storage: SharedStorage,
    embeddings: Arc<memory_engine::EmbeddingModel>,
    provider: Arc<memory_engine::MemoryProvider>,
    health: SharedHealth,
) {
    loop {
        match process_next_memory_job(
            Arc::clone(&storage),
            Arc::clone(&embeddings),
            Arc::clone(&provider),
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => sleep(Duration::from_millis(100)).await,
            Err(error) if error.is_fatal() => {
                health.degrade();
                tracing::error!(%error, "memory worker stopped; service degraded");
                break;
            }
            Err(error) => {
                tracing::error!(%error, "memory worker failed; retrying");
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

/// Runs document jobs until shutdown or a fatal storage failure degrades the service.
async fn worker_loop(
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
    provider: Option<Arc<memory_engine::MemoryProvider>>,
    health: SharedHealth,
) {
    loop {
        match process_next_job_with_services(
            Arc::clone(&storage),
            embeddings.as_ref().map(Arc::clone),
            provider.as_ref().map(Arc::clone),
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => sleep(Duration::from_millis(100)).await,
            Err(error) if error.is_fatal() => {
                health.degrade();
                tracing::error!(%error, "document worker stopped; service degraded");
                break;
            }
            Err(error) => {
                tracing::error!(%error, "document worker failed; retrying");
                sleep(Duration::from_secs(1)).await;
            }
        }
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
    if let Some(custom_id) = &request.custom_id {
        if custom_id.chars().count() > 100
            || custom_id.is_empty()
            || !custom_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'-'))
        {
            return Err(ApiError::Validation(
                "customId must be 1-100 characters containing only letters, numbers, _, :, or -",
            ));
        }
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

async fn authenticate(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let supplied = request
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let organization = supplied.and_then(|key| {
        let supplied_hash: [u8; 32] = Sha256::digest(key).into();
        state
            .api_keys
            .iter()
            .find(|(expected, _)| bool::from(supplied_hash.ct_eq(expected)))
            .map(|(_, organization)| organization.clone())
    });
    let has_session_material = request.headers().contains_key(http::header::COOKIE);
    let organization = organization.or_else(|| {
        (supplied.is_none() && !has_session_material && peer.ip().is_loopback())
            .then(|| state.local_org_id.clone())
    });
    if let Some(organization) = organization {
        request
            .extensions_mut()
            .insert(OrganizationId(organization));
        Ok(next.run(request).await)
    } else {
        Err(AuthError)
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
    #[error("service is degraded after a fatal worker failure")]
    ServiceDegraded,
    #[error("storage lock is unavailable after a previous operation failed")]
    StorageUnavailable,
    #[error("storage operation failed: {0}")]
    Storage(#[source] storage::StorageError),
    #[error("database executor stopped before completing the operation: {0}")]
    DatabaseExecutor(#[source] tokio::task::JoinError),
    #[error("query embedding failed: {0}")]
    Embedding(#[source] memory_engine::EmbeddingError),
    #[error("query embedding returned no vector")]
    EmptyEmbedding,
    #[error("cannot aggregate and rerank the same search")]
    AggregateAndRerank,
    #[error("memory not found")]
    MemoryNotFound,
}
impl ApiError {
    /// Returns whether this request failure makes subsequent results unreliable.
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
            Self::AggregateAndRerank => (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "Cannot use both aggregate and rerank simultaneously",
                    details: None,
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
                    details: Some("A fatal worker failure requires a server restart"),
                }),
            )
                .into_response(),
            Self::StorageUnavailable
            | Self::Storage(_)
            | Self::DatabaseExecutor(_)
            | Self::Embedding(_)
            | Self::EmptyEmbedding => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: "Internal server error",
                    details: Some(
                        "The document operation could not be completed; retry the request",
                    ),
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
    #[error("memory extraction failed: {0}")]
    Provider(#[source] memory_engine::ProviderError),
    #[error("cached memory extraction is invalid: {0}")]
    CachedExtraction(#[source] serde_json::Error),
}

impl WorkerError {
    /// Returns whether the worker must stop and degrade service health instead of retrying.
    fn is_fatal(&self) -> bool {
        matches!(self, Self::StorageUnavailable)
            || matches!(self, Self::Storage(error) if error.is_fatal())
            || matches!(self, Self::Executor(error) if error.is_panic())
    }
}
