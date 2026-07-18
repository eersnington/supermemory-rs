//! HTTP routing and authentication boundary.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::{ConnectInfo, Path, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use storage::{EmbeddedChunk, Storage, UpsertDocument};
use subtle::ConstantTimeEq;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::time::{Duration, sleep};

/// Safely shared application storage.
pub type SharedStorage = Arc<Mutex<Storage>>;

#[derive(Clone)]
struct AppState {
    api_key_hash: Option<[u8; 32]>,
    api_key: Option<String>,
    port: u16,
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
}

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
    let state = AppState {
        api_key_hash: api_key.as_ref().map(|key| Sha256::digest(key).into()),
        api_key,
        port,
        storage,
        embeddings,
    };
    let api = Router::new()
        .route("/documents", post(create_document))
        .route("/documents/{id}", get(get_document))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    let search = Router::new()
        .route("/search", post(search))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    Router::new()
        .route("/", get(landing_page))
        .route("/health", get(health))
        .route("/v4/openapi", get(openapi))
        .route("/v4/reference", get(api_reference))
        .nest("/v3", api)
        .nest("/v4", search)
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
  -d '{{"q":"remember"}}'</code></pre></article>
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
            "/health": { "get": { "responses": { "200": { "description": "Healthy" } } } },
            "/v3/documents": { "post": { "summary": "Add a document", "responses": { "200": { "description": "Queued document" } } } },
            "/v3/documents/{id}": { "get": { "summary": "Get a document", "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }], "responses": { "200": { "description": "Document" }, "404": { "description": "Not found" } } } },
            "/v4/search": { "post": { "summary": "Search local documents", "responses": { "200": { "description": "Search results" } } } }
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
    serve_with_embeddings_ready(address, api_key, storage, None, ready).await
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
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;
    ready();
    let worker = tokio::spawn(worker_loop(
        Arc::clone(&storage),
        embeddings.as_ref().map(Arc::clone),
    ));
    let result = axum::serve(
        listener,
        router_with_port(api_key, storage, embeddings, address.port())
            .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(ServerError::Serve);
    worker.abort();
    result
}

async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
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
    Json(request): Json<CreateDocumentRequest>,
) -> Result<Json<DocumentResult>, ApiError> {
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
            .upsert_document(document)
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
    Path(id): Path<String>,
) -> Result<Json<storage::Document>, ApiError> {
    let storage = Arc::clone(&state.storage);
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .find_document(&id)
            .map_err(ApiError::Storage)
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??
    .map(Json)
    .ok_or(ApiError::NotFound)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    q: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default = "default_search_threshold")]
    threshold: f32,
}

const fn default_search_limit() -> usize {
    10
}

const fn default_search_threshold() -> f32 {
    0.6
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<storage::SearchHit>,
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
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
    let query = request.q.trim().to_owned();
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
    let storage = Arc::clone(&state.storage);
    let limit = request.limit;
    let threshold = request.threshold;
    let results = tokio::task::spawn_blocking(move || {
        let storage = storage.lock().map_err(|_| ApiError::StorageUnavailable)?;
        query_vector
            .map_or_else(
                || storage.search(&query, limit),
                |vector| {
                    storage.search_semantic(
                        vector.as_slice(),
                        "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                        limit,
                        threshold,
                    )
                },
            )
            .map_err(ApiError::Storage)
    })
    .await
    .map_err(ApiError::DatabaseExecutor)??;
    Ok(Json(SearchResponse { results }))
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
    let Some(job) = job else {
        return Ok(false);
    };

    mark_job_stage(Arc::clone(&storage), job.clone(), "chunking").await?;
    let chunks = match memory_engine::chunk_text(&job.content, None) {
        Ok(chunks) => chunks,
        Err(error) => {
            let message = error.to_string();
            persist_job_failure(Arc::clone(&storage), job.clone(), message).await?;
            return Err(WorkerError::Chunking(error));
        }
    };
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
        mark_job_stage(Arc::clone(&storage), job.clone(), "embedding").await?;
        let values = chunks.clone();
        let vectors = match tokio::task::spawn_blocking(move || embeddings.embed(&values))
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
        mark_job_stage(Arc::clone(&storage), job.clone(), "indexing").await?;
        return tokio::task::spawn_blocking(move || {
            let embedded: Vec<_> = chunks
                .iter()
                .zip(&vectors)
                .map(|(content, vector)| EmbeddedChunk {
                    content,
                    vector: vector.as_slice(),
                })
                .collect();
            storage
                .lock()
                .map_err(|_| WorkerError::StorageUnavailable)?
                .complete_embedded_job(
                    &job,
                    &embedded,
                    "Xenova/bge-base-en-v1.5:q8:mean:normalized",
                    memory_engine::BGE_DIMENSIONS,
                )
                .map_err(WorkerError::Storage)
        })
        .await
        .map_err(WorkerError::Executor)?
        .map(|()| true);
    }
    tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| WorkerError::StorageUnavailable)?
            .complete_job(&job, &chunks)
            .map_err(WorkerError::Storage)
    })
    .await
    .map_err(WorkerError::Executor)??;
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

async fn worker_loop(
    storage: SharedStorage,
    embeddings: Option<Arc<memory_engine::EmbeddingModel>>,
) {
    loop {
        match process_next_job_with_embeddings(
            Arc::clone(&storage),
            embeddings.as_ref().map(Arc::clone),
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => sleep(Duration::from_millis(100)).await,
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
    request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let supplied = request
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let valid = supplied
        .zip(state.api_key_hash.as_ref())
        .is_some_and(|(key, expected)| {
            let supplied_hash: [u8; 32] = Sha256::digest(key).into();
            bool::from(supplied_hash.ct_eq(expected))
        });
    let has_session_material = request.headers().contains_key(http::header::COOKIE);
    if valid || (supplied.is_none() && !has_session_material && peer.ip().is_loopback()) {
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
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
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
        }
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
}
