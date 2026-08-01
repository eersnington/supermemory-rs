use super::{
    AppState, HeaderMap, HeaderValue, Html, IntoResponse, Json, Response, State, Value,
    escape_html, header,
};

pub(crate) async fn landing_page(State(state): State<AppState>) -> Response {
    let api_key = state.api_key.as_deref().unwrap_or("sm_your_local_api_key");
    let html = LANDING_PAGE
        .replace("{{PORT}}", &state.port.to_string())
        .replace("{{API_KEY}}", &escape_html(api_key));
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    (headers, Html(html)).into_response()
}

const LANDING_PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>supermemory-RS · local</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root { --accent-start:#1148f7; --accent-mid:#117dff; --accent:#117dff; --rs:#ff8700; --bg:#faf9f4; --surface:#fff; --muted-bg:#f3f1ec; --code:#0f1014; --border:#e3e0db; --border-muted:#f0ede8; --text:#0a0a0a; --secondary:#525252; --muted:#a3a3a3; --code-text:#e6e6e6; --code-muted:#8b8b94; --success:#16a34a; --sans:"Inter",ui-sans-serif,system-ui,sans-serif; --display:"Space Grotesk",var(--sans); --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; --ease:cubic-bezier(.23,1,.32,1); }
* { box-sizing:border-box; } html,body { margin:0; padding:0; } body { min-height:100vh; color:var(--text); background:radial-gradient(1200px 600px at 80% -10%,rgb(17 125 255 / 7%),transparent 60%),radial-gradient(900px 500px at -10% 110%,rgb(205 244 255 / 50%),transparent 60%),var(--bg); font:15px/1.55 var(--sans); -webkit-font-smoothing:antialiased; }
.wrap { max-width:980px; margin:0 auto; padding:56px 28px 96px; } header { display:flex; align-items:center; gap:12px; margin-bottom:48px; } .brand-logo { display:flex; width:28px; height:28px; align-items:center; justify-content:center; color:var(--accent-start); } .brand-logo svg { width:100%; height:100%; } .brand { font:600 17px var(--display); letter-spacing:-.01em; } .brand .rs { color:var(--rs); } .brand .dim { color:var(--muted); font-weight:500; } .status-pill { display:inline-flex; align-items:center; gap:8px; margin-left:auto; padding:6px 12px; border-radius:999px; color:var(--success); background:rgb(22 163 74 / 10%); font-size:12px; font-weight:500; } .status-dot { width:7px; height:7px; border-radius:50%; background:var(--success); box-shadow:0 0 0 4px rgb(22 163 74 / 18%); }
h1 { margin:0 0 16px; font:700 clamp(36px,5vw,56px)/1.05 var(--display); letter-spacing:-.025em; } h1 .gradient { background:linear-gradient(135deg,var(--accent-start),var(--accent-mid)); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; } h1 .rs-gradient { background:linear-gradient(135deg,#ff8700,#ffd08a); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; } .subtitle { max-width:60ch; margin:0 0 32px; color:var(--secondary); font-size:17px; }
.resource-band { display:grid; grid-template-columns:1fr; gap:12px; margin:-8px 0 24px; } @media (min-width:760px) { .resource-band { grid-template-columns:1.35fr 1fr 1fr; } } .resource-link { display:flex; min-height:108px; flex-direction:column; justify-content:space-between; gap:18px; padding:18px; border:1px solid var(--border); border-radius:14px; color:var(--text); background:var(--surface); text-decoration:none; transition:transform 140ms var(--ease),border-color 140ms ease,box-shadow 140ms ease; } .resource-link.primary { border-color:transparent; color:#fff; background:linear-gradient(135deg,var(--accent-start),var(--accent-mid)); box-shadow:0 18px 40px rgb(17 72 247 / 18%); } .resource-label { display:flex; align-items:center; justify-content:space-between; gap:14px; font:600 18px var(--display); letter-spacing:-.01em; } .resource-arrow { color:var(--accent); font-size:22px; line-height:1; } .primary .resource-arrow { color:#ffffffdc; } .resource-desc { margin:0; color:var(--secondary); font-size:13px; line-height:1.45; } .primary .resource-desc { color:#ffffffc7; }
.summary-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-bottom:64px; } .summary-card { padding:14px 16px; border:1px solid var(--border); border-radius:12px; background:var(--surface); } .summary-label,.card-label { margin:0 0 6px; color:var(--muted); font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; } .summary-value { overflow-wrap:anywhere; color:var(--text); font:13px var(--mono); }
.section-grid { display:grid; grid-template-columns:1fr; gap:56px; } @media (min-width:920px) { .section-grid { grid-template-columns:1fr 1fr; gap:48px; } } .section { display:flex; min-width:0; flex-direction:column; } .section-eyebrow { margin:0 0 10px; color:var(--accent); font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; } .section-title { margin:0 0 8px; font:600 24px var(--display); letter-spacing:-.01em; } .section-desc { margin:0 0 20px; color:var(--secondary); font-size:14px; }
.card-block { margin-bottom:14px; padding:18px; border:1px solid var(--border); border-radius:14px; background:var(--surface); } .card-label { margin:0 0 10px; } .copy { position:relative; overflow-x:auto; padding:14px 88px 14px 16px; border:1px solid #1d1e26; border-radius:10px; color:var(--code-text); background:var(--code); font:12.5px var(--mono); } .copy pre { margin:0; font:12.5px/1.55 var(--mono); white-space:pre-wrap; overflow-wrap:anywhere; } .copy-btn { position:absolute; top:10px; right:10px; border:1px solid #2a2b35; border-radius:7px; padding:5px 9px; color:var(--code-muted); background:rgb(255 255 255 / 4%); font:11px var(--sans); cursor:pointer; transition:transform 140ms var(--ease),color 120ms ease,border-color 120ms ease; } .copy-btn:active { transform:scale(.97); } .copy-btn.copied { border-color:rgb(74 222 128 / 40%); color:#4ade80; }
.plugin { overflow:hidden; margin-bottom:10px; border:1px solid var(--border); border-radius:12px; background:var(--surface); transition:border-color 120ms ease; } .plugin[open] { border-color:var(--accent); } .plugin summary { display:flex; align-items:center; gap:14px; padding:14px 16px; cursor:pointer; list-style:none; } .plugin summary::-webkit-details-marker { display:none; } .plugin-logo { display:grid; width:36px; height:36px; flex:0 0 auto; place-items:center; overflow:hidden; border-radius:8px; background:var(--muted-bg); color:#171717; font:600 13px var(--display); } .plugin-meta { display:flex; min-width:0; flex:1; flex-direction:column; } .plugin-name { color:var(--text); font:600 15px var(--display); } .plugin-tagline { margin-top:2px; color:var(--muted); font-size:12px; line-height:1.4; } .plugin-chevron { color:var(--muted); font-size:20px; transition:transform 160ms var(--ease),color 160ms ease; } .plugin[open] .plugin-chevron { color:var(--accent); transform:rotate(90deg); } .steps { margin:0; padding:0 16px 16px; list-style:none; counter-reset:step; } .steps li { position:relative; padding:12px 0 12px 32px; border-top:1px solid var(--border-muted); counter-increment:step; } .steps li::before { position:absolute; top:14px; left:0; display:grid; width:22px; height:22px; place-items:center; border-radius:50%; color:var(--secondary); background:var(--muted-bg); content:counter(step); font-size:11px; font-weight:600; } .step-title { margin-bottom:2px; font-size:14px; font-weight:600; } .step-desc { margin-bottom:10px; color:var(--secondary); font-size:13px; }
.links { display:flex; flex-wrap:wrap; gap:18px; margin-top:40px; padding-top:28px; border-top:1px solid var(--border-muted); font-size:14px; } .links a { padding-bottom:1px; border-bottom:1px dotted var(--border); color:var(--secondary); text-decoration:none; } footer { margin-top:32px; color:var(--muted); font-size:12px; }
button:focus-visible,a:focus-visible,summary:focus-visible { outline:3px solid rgb(17 125 255 / 40%); outline-offset:3px; } @media (hover:hover) and (pointer:fine) { .resource-link:hover { border-color:var(--accent); box-shadow:0 14px 32px rgb(10 10 10 / 8%); transform:translateY(-2px); } .resource-link.primary:hover { box-shadow:0 20px 44px rgb(17 72 247 / 24%); } .copy-btn:hover { border-color:var(--code-muted); color:#fff; } .links a:hover { border-color:var(--accent); color:var(--accent); } } @media (prefers-reduced-motion:reduce) { *,*::before,*::after { transition-duration:0ms!important; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand-logo" aria-hidden="true"><svg fill="none" viewBox="0 0 39.467 32"><path d="M39.126 12.632H24.606V.01h-4.692v13.695c0 1.454.574 2.851 1.595 3.88l11.856 11.958 3.318-3.346-8.758-8.831h11.204v-4.734ZM2.446 5.822l8.757 8.832H0v4.731h14.52v12.623h4.692V18.312c0-1.453-.573-2.847-1.595-3.88L5.764 2.476Z" fill="currentColor"/></svg></div>
    <div class="brand">supermemory<span class="rs">-RS</span> <span class="dim">· local</span></div>
    <div class="status-pill"><span class="status-dot"></span> running</div>
  </header>

  <h1><span class="gradient">Supermemory</span><span class="rs-gradient">-RS</span> is live.</h1>
  <p class="subtitle">supermemory-RS is running on this machine. Rust, SQLite, local embeddings, and search stay local. Configure a provider only when you want memory extraction.</p>

  <nav class="resource-band" aria-label="Documentation and reference">
    <a class="resource-link primary" href="/v4/reference"><span class="resource-label">Supermemory-RS API <span class="resource-arrow">↗</span></span><p class="resource-desc">Browse the endpoints implemented by this running server.</p></a>
    <a class="resource-link" href="https://supermemory.ai/docs/self-hosting/overview" target="_blank" rel="noopener"><span class="resource-label">Supermemory Docs <span class="resource-arrow">↗</span></span><p class="resource-desc">Upstream SDK and self-hosting guides.</p></a>
    <a class="resource-link" href="https://github.com/eersnington/supermemory-rs" target="_blank" rel="noopener"><span class="resource-label">GitHub <span class="resource-arrow">↗</span></span><p class="resource-desc">Source code, compatibility work, and issues.</p></a>
  </nav>

  <div class="summary-row">
    <div class="summary-card"><p class="summary-label">endpoint</p><div class="summary-value">http://localhost:{{PORT}}</div></div>
    <div class="summary-card"><p class="summary-label">runtime</p><div class="summary-value">Rust · Axum · SQLite</div></div>
    <div class="summary-card"><p class="summary-label">embeddings</p><div class="summary-value">local BGE · 768d</div></div>
  </div>

  <div class="section-grid">
    <section class="section">
      <p class="section-eyebrow">build</p>
      <h2 class="section-title">Get started</h2>
      <p class="section-desc">Use supported Supermemory SDK flows and REST endpoints against this local Rust server. Point the client at this base URL and keep your data on your machine.</p>
      <div class="card-block"><p class="card-label">install the SDK</p><div class="copy" id="sdk-install"><pre>npm install supermemory
# or: bun add supermemory · pnpm add supermemory</pre><button class="copy-btn" data-target="sdk-install" type="button">copy</button></div></div>
      <div class="card-block"><p class="card-label">use it</p><div class="copy" id="sdk-snippet"><pre>import Supermemory from "supermemory"

const client = new Supermemory({
  apiKey: "{{API_KEY}}",
  baseURL: "http://localhost:{{PORT}}",
})

await client.add({
  content: "running on supermemory-RS",
})

const results = await client.search.execute({
  q: "what is running locally?",
})</pre><button class="copy-btn" data-target="sdk-snippet" type="button">copy</button></div></div>
      <div class="card-block"><p class="card-label">curl, if you prefer</p><div class="copy" id="curl"><pre>curl -X POST http://localhost:{{PORT}}/v3/documents \
  -H "Authorization: Bearer {{API_KEY}}" \
  -H "Content-Type: application/json" \
  -d '{"content":"supermemory-RS is running locally"}'</pre><button class="copy-btn" data-target="curl" type="button">copy</button></div></div>
    </section>

    <section class="section">
      <p class="section-eyebrow">use</p>
      <h2 class="section-title">Use with your agents</h2>
      <p class="section-desc">Point compatible tools at this local instance. Expand an integration to copy the base URL and key.</p>
      <details class="plugin"><summary><span class="plugin-logo">CC</span><span class="plugin-meta"><span class="plugin-name">Claude Code</span><span class="plugin-tagline">Persistent memory across coding sessions.</span></span><span class="plugin-chevron" aria-hidden="true">›</span></summary><ol class="steps"><li><div class="step-title">Point Claude Code at this server</div><div class="step-desc">Add these variables to your shell profile.</div><div class="copy" id="claude"><pre>export SUPERMEMORY_BASE_URL="http://localhost:{{PORT}}"
export SUPERMEMORY_CC_API_KEY="{{API_KEY}}"</pre><button class="copy-btn" data-target="claude" type="button">copy</button></div></li><li><div class="step-title">Install the plugin</div><div class="copy" id="claude-install"><pre>/plugin marketplace add supermemoryai/claude-supermemory
/plugin install claude-supermemory</pre><button class="copy-btn" data-target="claude-install" type="button">copy</button></div></li></ol></details>
      <details class="plugin"><summary><span class="plugin-logo">OC</span><span class="plugin-meta"><span class="plugin-name">OpenCode</span><span class="plugin-tagline">Search and capture context across coding sessions.</span></span><span class="plugin-chevron" aria-hidden="true">›</span></summary><ol class="steps"><li><div class="step-title">Set the local server</div><div class="copy" id="opencode"><pre>export SUPERMEMORY_BASE_URL="http://localhost:{{PORT}}"
export SUPERMEMORY_API_KEY="{{API_KEY}}"</pre><button class="copy-btn" data-target="opencode" type="button">copy</button></div></li><li><div class="step-title">Install the plugin</div><div class="copy" id="opencode-install"><pre>bunx opencode-supermemory@latest install</pre><button class="copy-btn" data-target="opencode-install" type="button">copy</button></div></li></ol></details>
      <details class="plugin"><summary><span class="plugin-logo">OCl</span><span class="plugin-meta"><span class="plugin-name">OpenClaw</span><span class="plugin-tagline">Local memory for multi-platform agents.</span></span><span class="plugin-chevron" aria-hidden="true">›</span></summary><ol class="steps"><li><div class="step-title">Install the plugin</div><div class="copy" id="openclaw"><pre>openclaw plugins install @supermemory/openclaw-supermemory</pre><button class="copy-btn" data-target="openclaw" type="button">copy</button></div></li><li><div class="step-title">Configure for local</div><div class="copy" id="openclaw-setup"><pre>openclaw supermemory setup
# base url: http://localhost:{{PORT}}
# api key:  {{API_KEY}}</pre><button class="copy-btn" data-target="openclaw-setup" type="button">copy</button></div></li></ol></details>
      <details class="plugin"><summary><span class="plugin-logo">H</span><span class="plugin-meta"><span class="plugin-name">Hermes</span><span class="plugin-tagline">Memory setup for the Hermes agent.</span></span><span class="plugin-chevron" aria-hidden="true">›</span></summary><ol class="steps"><li><div class="step-title">Start the memory wizard</div><div class="copy" id="hermes"><pre>hermes memory setup
# base url: http://localhost:{{PORT}}
# api key:  {{API_KEY}}</pre><button class="copy-btn" data-target="hermes" type="button">copy</button></div></li></ol></details>
      <details class="plugin"><summary><span class="plugin-logo">⌘</span><span class="plugin-meta"><span class="plugin-name">OpenAI Codex</span><span class="plugin-tagline">Persistent memory for the Codex CLI.</span></span><span class="plugin-chevron" aria-hidden="true">›</span></summary><ol class="steps"><li><div class="step-title">Set the local server</div><div class="copy" id="codex"><pre>export SUPERMEMORY_BASE_URL="http://localhost:{{PORT}}"
export SUPERMEMORY_CODEX_API_KEY="{{API_KEY}}"</pre><button class="copy-btn" data-target="codex" type="button">copy</button></div></li><li><div class="step-title">Install the plugin</div><div class="copy" id="codex-install"><pre>npx codex-supermemory@latest install</pre><button class="copy-btn" data-target="codex-install" type="button">copy</button></div></li></ol></details>
    </section>
  </div>

  <div class="links"><a href="/v4/openapi">OpenAPI spec</a><a href="/v4/reference">Supermemory-RS API</a><a href="https://supermemory.ai/docs" target="_blank" rel="noopener">Supermemory Docs ↗</a><a href="https://github.com/eersnington/supermemory-rs" target="_blank" rel="noopener">Supermemory-RS on GitHub ↗</a></div>
  <footer>supermemory-RS · self-hosted Rust, SQLite, and local embeddings.</footer>
</div>
<script>
  document.querySelectorAll(".copy-btn").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault(); event.stopPropagation();
      const target = document.getElementById(button.dataset.target);
      const text = target?.querySelector("pre")?.textContent || "";
      try { await navigator.clipboard.writeText(text.trim()); button.textContent = "copied"; button.classList.add("copied"); setTimeout(() => { button.textContent = "copy"; button.classList.remove("copied"); }, 1400); } catch { button.textContent = "unable to copy"; }
    });
  });
</script>
</body>
</html>"#;

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
