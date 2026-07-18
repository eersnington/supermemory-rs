use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    response::Response,
};
use serde_json::{Value, json};
use server::{SharedStorage, process_next_job, router};
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
            .body(Body::from(r#"{"q":"kingfisher"}"#))
            .expect("request"),
    );
    let response = app.oneshot(search).await.expect("search response");
    assert_eq!(response.status(), StatusCode::OK);
    let searched = body(response).await;
    assert_eq!(searched["results"][0]["documentId"], id);
}
