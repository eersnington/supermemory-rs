const baseURL = process.env.SUPERMEMORY_BASE_URL;
const apiKey = process.env.SUPERMEMORY_API_KEY;
const documentCount = Number(process.env.BENCH_DOCUMENTS ?? 50);
const queryCount = Number(process.env.BENCH_QUERIES ?? 100);
if (!baseURL || !apiKey) throw new Error("SUPERMEMORY_BASE_URL and SUPERMEMORY_API_KEY are required");

const headers = { authorization: `Bearer ${apiKey}`, "content-type": "application/json" };
const tag = `bench-${Date.now()}`;
const started = performance.now();
const ids = [];
for (let index = 0; index < documentCount; index += 1) {
  const response = await fetch(`${baseURL}/v3/documents`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      content: `Benchmark document ${index}. The benchmark token is token-${index}. Semantic retrieval should associate this text with measurement ${index}.`,
      customId: `${tag}-${index}`,
      containerTag: tag,
      taskType: "superrag",
    }),
  });
  if (!response.ok) throw new Error(`ingestion failed: ${response.status} ${await response.text()}`);
  ids.push((await response.json()).id);
}
const acceptedAt = performance.now();
for (;;) {
  const statuses = await Promise.all(ids.map(async (id) => {
    const response = await fetch(`${baseURL}/v3/documents/${id}`, { headers });
    if (!response.ok) throw new Error(`status failed: ${response.status}`);
    return (await response.json()).status;
  }));
  if (statuses.every((status) => status === "done")) break;
  if (statuses.some((status) => status === "failed")) throw new Error("a benchmark document failed");
  await Bun.sleep(20);
}
const completedAt = performance.now();

async function distribution(path, body) {
  const values = [];
  for (let index = 0; index < queryCount; index += 1) {
    const before = performance.now();
    const response = await fetch(`${baseURL}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(typeof body === "function" ? body(index) : body),
    });
    await response.arrayBuffer();
    if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
    values.push(performance.now() - before);
  }
  values.sort((left, right) => left - right);
  const percentile = (fraction) => values[Math.ceil(values.length * fraction) - 1];
  return {
    count: values.length,
    min: values[0],
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    max: values.at(-1),
  };
}

const search = await distribution("/v4/search", (index) => ({
  q: `measurement ${index % documentCount}`,
  containerTag: tag,
  searchMode: "documents",
  threshold: 0,
  limit: 10,
}));
const profile = await distribution("/v4/profile", {
  containerTag: tag,
  include: ["static", "dynamic", "buckets"],
});

console.log(JSON.stringify({
  schemaVersion: 1,
  implementation: "supermemory-rs",
  documents: documentCount,
  ingestionDocumentsPerSecond: documentCount / ((acceptedAt - started) / 1000),
  processingDocumentsPerSecond: documentCount / ((completedAt - acceptedAt) / 1000),
  searchLatencyMs: search,
  profileLatencyMs: profile,
}));
