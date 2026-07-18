use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    response::Response,
};
use memory_engine::EmbeddingModel;
use serde_json::{Value, json};
use server::{
    SharedStorage, process_next_job, process_next_job_with_embeddings, router,
    router_with_embeddings,
};
use storage::Storage;
use tower::ServiceExt;

fn app() -> axum::Router {
    let storage: SharedStorage = Arc::new(Mutex::new(Storage::in_memory().expect("storage")));
    router(Some("secret".to_owned()), storage)
}

fn with_peer(mut request: Request<Body>, peer: SocketAddr) -> Request<Body> {
    request
        .extensions_mut()
        .insert(axum::extract::ConnectInfo(peer));
    request
}

fn remote(request: Request<Body>) -> Request<Body> {
    with_peer(request, SocketAddr::from(([192, 0, 2, 1], 1234)))
}

async fn request(method: &str, path: &str, body: Value, authorized: bool) -> Response {
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if authorized {
        builder = builder.header("authorization", "Bearer secret");
    }
    app()
        .oneshot(remote(
            builder.body(Body::from(body.to_string())).expect("request"),
        ))
        .await
        .expect("response")
}

async fn body(response: Response) -> Value {
    serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("JSON body")
}

#[tokio::test]
async fn health_is_public() {
    assert_eq!(
        request("GET", "/health", json!({}), false).await.status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn landing_page_exposes_local_examples_without_caching_the_key() {
    let response = request("GET", "/", json!({}), false).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["cache-control"], "no-store");
    let html = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("HTML");
    assert!(html.contains("supermemory<span>-RS</span>"));
    assert!(html.contains("Bearer secret"));
    assert!(html.contains("/v4/reference"));
}

#[tokio::test]
async fn openapi_and_reference_are_public() {
    let openapi = request("GET", "/v4/openapi", json!({}), false).await;
    assert_eq!(openapi.status(), StatusCode::OK);
    assert_eq!(body(openapi).await["openapi"], "3.1.0");
    assert_eq!(
        request("GET", "/v4/reference", json!({}), false)
            .await
            .status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn post_rejects_missing_credentials_with_details() {
    let response = request("POST", "/v3/documents", json!({"content":"x"}), false).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        body(response).await["details"],
        "A valid Bearer API key is required"
    );
}

#[tokio::test]
async fn get_rejects_invalid_credentials() {
    assert_eq!(
        request("GET", "/v3/documents/x", json!({}), false)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn localhost_without_authentication_material_uses_local_identity() {
    for peer in [
        SocketAddr::from(([127, 0, 0, 1], 1234)),
        SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], 1234)),
    ] {
        let request = Request::post("/v3/documents")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"content":"local"}"#))
            .expect("request");
        assert_eq!(
            app()
                .oneshot(with_peer(request, peer))
                .await
                .expect("response")
                .status(),
            StatusCode::OK
        );
    }
}

#[tokio::test]
async fn server_without_api_key_remains_loopback_only() {
    let storage: SharedStorage = Arc::new(Mutex::new(Storage::in_memory().expect("storage")));
    let app = router(None, storage);
    let local = Request::post("/v3/documents")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"local"}"#))
        .expect("request");
    assert_eq!(
        app.clone()
            .oneshot(with_peer(local, SocketAddr::from(([127, 0, 0, 1], 1234))))
            .await
            .expect("response")
            .status(),
        StatusCode::OK
    );

    let remote_request = Request::post("/v3/documents")
        .header("authorization", "Bearer arbitrary")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"remote"}"#))
        .expect("request");
    assert_eq!(
        app.oneshot(remote(remote_request))
            .await
            .expect("response")
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn spoofed_localhost_from_non_loopback_peer_is_rejected() {
    let request = Request::post("/v3/documents")
        .header("host", "localhost:6767")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"remote"}"#))
        .expect("request");
    assert_eq!(
        app()
            .oneshot(remote(request))
            .await
            .expect("response")
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn localhost_with_invalid_supplied_bearer_is_rejected() {
    let request = Request::post("/v3/documents")
        .header("authorization", "Bearer wrong")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"local"}"#))
        .expect("request");
    assert_eq!(
        app()
            .oneshot(with_peer(
                request,
                SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], 1234))
            ))
            .await
            .expect("response")
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn post_rejects_invalid_fields() {
    for invalid in [
        json!({"content":"  "}),
        json!({"content":"x","containerTag":"a","containerTags":["b"]}),
        json!({"content":"x","filepath":"/profile.md"}),
        json!({"content":"x","metadata":{"bad":null}}),
    ] {
        assert_eq!(
            request("POST", "/v3/documents", invalid, true)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
}

#[tokio::test]
async fn create_persists_and_get_resolves_id_with_defaults() {
    let app = app();
    let create = remote(
        Request::post("/v3/documents")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"content":"  retained  ","customId":"custom:1"}"#,
            ))
            .expect("request"),
    );
    let created = app.clone().oneshot(create).await.expect("response");
    assert_eq!(created.status(), StatusCode::OK);
    let created = body(created).await;
    assert_eq!(created["status"], "queued");
    let id = created["id"].as_str().expect("id");
    let get = remote(
        Request::get(format!("/v3/documents/{id}"))
            .header("authorization", "Bearer secret")
            .body(Body::empty())
            .expect("request"),
    );
    let document = body(app.clone().oneshot(get).await.expect("response")).await;
    assert_eq!(document["content"], "retained");
    assert_eq!(document["containerTags"], json!(["sm_project_default"]));
    let get = remote(
        Request::get("/v3/documents/custom:1")
            .header("authorization", "Bearer secret")
            .body(Body::empty())
            .expect("request"),
    );
    assert_eq!(
        app.oneshot(get).await.expect("response").status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn get_returns_contract_not_found_body() {
    let response = request("GET", "/v3/documents/missing", json!({}), true).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(body(response).await, json!({"error":"Document not found"}));
}

#[tokio::test]
async fn submitted_document_is_processed_and_searchable() {
    let storage: SharedStorage = Arc::new(Mutex::new(Storage::in_memory().expect("storage")));
    let app = router(Some("secret".to_owned()), Arc::clone(&storage));
    let create = remote(
        Request::post("/v3/documents")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"content":"A distinctive kingfisher observation"}"#,
            ))
            .expect("request"),
    );
    let created = body(app.clone().oneshot(create).await.expect("create response")).await;
    let id = created["id"].as_str().expect("document id");

    assert!(process_next_job(storage).await.expect("worker"));

    let search = remote(
        Request::post("/v4/search")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"q":"kingfisher","searchMode":"documents","threshold":0}"#,
            ))
            .expect("request"),
    );
    let response = app.oneshot(search).await.expect("search response");
    assert_eq!(response.status(), StatusCode::OK);
    let searched = body(response).await;
    assert_eq!(
        searched["results"][0]["documents"][0]["id"], id,
        "{searched}"
    );
}

#[tokio::test]
async fn submitted_document_is_embedded_and_semantically_searchable() {
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    let home = PathBuf::from(home).join(".supermemory");
    let model_path = home.join("models/Xenova/bge-base-en-v1.5");
    let runtime_path = home.join(
        "runtime/ort-native/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib",
    );
    if !model_path.exists() || !runtime_path.exists() {
        return;
    }
    let embeddings = Arc::new(
        EmbeddingModel::load(&model_path, &runtime_path).expect("existing model should load"),
    );
    let storage: SharedStorage = Arc::new(Mutex::new(Storage::in_memory().expect("storage")));
    let app = router_with_embeddings(
        Some("secret".to_owned()),
        Arc::clone(&storage),
        Arc::clone(&embeddings),
    );
    for content in [
        "The sky is blue on a clear day.",
        "A database transaction preserves atomicity.",
    ] {
        let create = remote(
            Request::post("/v3/documents")
                .header("authorization", "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "content": content,
                        "metadata": {
                            "topic": if content.contains("sky") { "weather" } else { "database" }
                        }
                    })
                    .to_string(),
                ))
                .expect("request"),
        );
        app.clone().oneshot(create).await.expect("create response");
        assert!(
            process_next_job_with_embeddings(Arc::clone(&storage), Some(Arc::clone(&embeddings)))
                .await
                .expect("semantic worker")
        );
    }

    let v3_search = remote(
        Request::post("/v3/search")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "q": "What color is the sky?",
                    "chunkThreshold": 0,
                    "filters": {
                        "AND": [{"key": "topic", "value": "weather"}]
                    }
                })
                .to_string(),
            ))
            .expect("request"),
    );
    let v3 = body(app.clone().oneshot(v3_search).await.expect("v3 search")).await;
    assert_eq!(v3["results"].as_array().map(Vec::len), Some(1));
    assert_eq!(v3["results"][0]["chunks"][0]["isRelevant"], true);
    assert_eq!(v3["total"], 1);

    let search = remote(
        Request::post("/v4/search")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"q":"What color is the sky?","threshold":0,"searchMode":"documents"}"#,
            ))
            .expect("request"),
    );
    let searched = body(app.oneshot(search).await.expect("search response")).await;
    assert_eq!(
        searched["results"][0]["chunk"],
        "The sky is blue on a clear day."
    );
}

#[tokio::test]
async fn v4_defaults_to_memory_results_instead_of_document_chunks() {
    let response = request("POST", "/v4/search", json!({"q":"anything"}), true).await;
    assert_eq!(response.status(), StatusCode::OK);
    let response = body(response).await;
    assert_eq!(response["results"], json!([]));
    assert_eq!(response["total"], 0);
    assert!(response["timing"].is_number());
}
