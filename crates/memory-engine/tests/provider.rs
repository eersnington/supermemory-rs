use std::collections::HashMap;

use memory_engine::{MemoryProvider, ProviderConfig, ProviderKind, ProviderModels, RelationKind};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

#[test]
fn provider_selection_uses_precedence_and_toml_models() {
    let values = HashMap::from([
        ("GEMINI_API_KEY", "gemini-secret"),
        ("OPENAI_API_KEY", "openai-secret"),
        ("OPENAI_MODEL", "ignored-environment-model"),
    ]);
    let models = ProviderModels {
        openai: "configured-openai".to_owned(),
        openai_reasoning_effort: "medium".to_owned(),
        anthropic: "configured-anthropic".to_owned(),
        gemini: "configured-gemini".to_owned(),
        groq: "configured-groq".to_owned(),
    };
    let config =
        ProviderConfig::from_values(|key| values.get(key).map(ToString::to_string), &models)
            .expect("configured provider");
    assert_eq!(config.kind, ProviderKind::OpenAi);
    assert_eq!(config.model, "configured-openai");
    assert_eq!(config.reasoning_effort.as_deref(), Some("medium"));
}

#[test]
fn provider_models_use_requested_defaults() {
    let models = ProviderModels::default();
    assert_eq!(models.openai, "gpt-5.6-luna");
    assert_eq!(models.openai_reasoning_effort, "medium");
    assert_eq!(models.gemini, "gemini-3.5-flash");
}

#[tokio::test]
async fn openai_compatible_response_is_validated_and_normalized() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut request = vec![0; 16 * 1024];
        let bytes = stream.read(&mut request).await.expect("read request");
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
            "id": "fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "fixture",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": "submit", "arguments": extracted.to_string()}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.expect("write");
        String::from_utf8(request[..bytes].to_vec()).expect("request text")
    });
    let provider = MemoryProvider::new(ProviderConfig {
        kind: ProviderKind::OpenAi,
        api_key: "secret".to_owned(),
        model: "fixture".to_owned(),
        base_url: Some(format!("http://{address}")),
        reasoning_effort: Some("medium".to_owned()),
    })
    .expect("provider");
    let candidates = provider
        .extract_once("The user prefers concise answers.", None, &[])
        .await
        .expect("extract");
    let request = server.await.expect("server");
    assert!(request.contains(r#""reasoning_effort":"medium""#));
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].buckets.is_empty());
    assert_eq!(candidates[0].parent_relations.len(), 1);
    assert_eq!(
        candidates[0].parent_relations[0].relation,
        RelationKind::Extends
    );
}
