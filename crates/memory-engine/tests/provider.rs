use std::collections::HashMap;

use memory_engine::{MemoryProvider, ProviderConfig, ProviderKind, RelationKind};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

#[test]
fn provider_selection_matches_v005_precedence_and_defaults() {
    let values = HashMap::from([
        ("GEMINI_API_KEY", "gemini-secret"),
        ("OPENAI_API_KEY", "openai-secret"),
        ("OPENAI_MODEL", "local-model"),
    ]);
    let config = ProviderConfig::from_values(|key| values.get(key).map(ToString::to_string))
        .expect("configured provider");
    assert_eq!(config.kind, ProviderKind::OpenAi);
    assert_eq!(config.model, "local-model");
}

#[tokio::test]
async fn openai_compatible_response_is_validated_and_normalized() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut request = vec![0; 16 * 1024];
        let _ = stream.read(&mut request).await.expect("read request");
        let extracted = serde_json::json!({
            "memoriesToAddOrUpdate": [{
                "tmpId": "tmp_1",
                "memory": "The user prefers concise answers.",
                "isInferred": false,
                "addToStaticProfile": true,
                "buckets": null,
                "parentRelations": [
                    {"memoryId":"mem_parent", "relation":"unknown"},
                    {"memoryId":"mem_parent", "relation":"updates"},
                    {"memoryId":"invalid", "relation":"updates"}
                ],
                "temporalContext": null,
                "forgetAfter": null,
                "forgetReason": null
            }]
        });
        let body = serde_json::json!({
            "choices": [{"message": {"content": extracted.to_string()}}]
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.expect("write");
    });
    let provider = MemoryProvider::new(ProviderConfig {
        kind: ProviderKind::OpenAi,
        api_key: "secret".to_owned(),
        model: "fixture".to_owned(),
        base_url: Some(format!("http://{address}")),
    })
    .expect("provider");
    let candidates = provider
        .extract("The user prefers concise answers.", None, &[])
        .await
        .expect("extract");
    server.await.expect("server");
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].buckets.is_empty());
    assert_eq!(candidates[0].parent_relations.len(), 1);
    assert_eq!(
        candidates[0].parent_relations[0].relation,
        RelationKind::Extends
    );
}
