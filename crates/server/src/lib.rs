//! HTTP routing and authentication boundary.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::{ConnectInfo, Path, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use storage::{Storage, UpsertDocument};
use subtle::ConstantTimeEq;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::time::{Duration, sleep};

/// Safely shared application storage.
pub type SharedStorage = Arc<Mutex<Storage>>;

#[derive(Clone)]
struct AppState {
    api_key_hash: Option<[u8; 32]>,
    storage: SharedStorage,
}

/// Builds the complete HTTP application.
pub fn router(api_key: Option<String>, storage: SharedStorage) -> Router {
    let state = AppState {
        api_key_hash: api_key.map(|key| Sha256::digest(key).into()),
        storage,
    };
    let api = Router::new()
        .route("/documents", post(create_document))
        .route("/documents/{id}", get(get_document))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    let search = Router::new()
        .route("/search", post(search))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    Router::new()
        .route("/health", get(health))
        .nest("/v3", api)
        .nest("/v4", search)
        .with_state(state)
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
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;
    ready();
    let worker = tokio::spawn(worker_loop(Arc::clone(&storage)));
    let result = axum::serve(
        listener,
        router(api_key, storage).into_make_service_with_connect_info::<SocketAddr>(),
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
}

const fn default_search_limit() -> usize {
    10
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
    let storage = Arc::clone(&state.storage);
    let query = request.q;
    let limit = request.limit;
    let results = tokio::task::spawn_blocking(move || {
        storage
            .lock()
            .map_err(|_| ApiError::StorageUnavailable)?
            .search(query.trim(), limit)
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

async fn worker_loop(storage: SharedStorage) {
    loop {
        match process_next_job(Arc::clone(&storage)).await {
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
            Self::StorageUnavailable | Self::Storage(_) | Self::DatabaseExecutor(_) => (
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
}
