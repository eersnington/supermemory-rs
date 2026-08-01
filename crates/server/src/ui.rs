use super::{
    AppState, HeaderMap, HeaderValue, Html, IntoResponse, Json, Response, State, Value,
    escape_html, header,
};

pub(crate) async fn landing_page(State(state): State<AppState>) -> Response {
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

pub(crate) async fn api_reference() -> Html<&'static str> {
    Html(
        r#"<!doctype html><html><head><title>supermemory API reference</title><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><script id="api-reference" data-url="/v4/openapi"></script><script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script></body></html>"#,
    )
}

pub(crate) async fn openapi(State(state): State<AppState>) -> Json<Value> {
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
