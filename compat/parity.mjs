import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { cases } from "./v0.0.6-cases.mjs";

const args = new Map(process.argv.slice(2).map((value, index, values) => value.startsWith("--") ? [value, values[index + 1]] : []));
const oracle = args.get("--oracle") ?? "http://127.0.0.1:6767";
const candidate = args.get("--candidate");
const acceptOracle = args.has("--accept-oracle");
const fixturePath = resolve("compat/fixtures/v0.0.6/http.json");

if (!candidate && !acceptOracle) throw new Error("pass --candidate <URL>, or use --accept-oracle to capture upstream evidence");

function normalize(value) {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, entry]) => {
      if (/^(id|documentId|memoryId|jobId|mergeId|orgId|userId|createdAt|updatedAt|expiresAt|fileUrl|url)$/i.test(key)) return [key, `<${key}>`];
      if (/^(timing|duration|elapsed)(Ms)?$/i.test(key)) return [key, `<${key}>`];
      return [key, normalize(entry)];
    }));
  }
  return value;
}

async function request(baseUrl, definition, variables) {
  const replace = (value) => typeof value === "string"
    ? value.replaceAll("{documentId}", variables.documentId ?? "missing-parity-v006").replaceAll("{tag}", "parity-v006")
    : Array.isArray(value) ? value.map(replace)
    : value && typeof value === "object" ? Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, replace(entry)]))
    : value;
  const response = await fetch(`${baseUrl}${replace(definition.path)}`, {
    method: definition.method,
    headers: { "content-type": "application/json", ...(process.env.SUPERMEMORY_API_KEY ? { authorization: `Bearer ${process.env.SUPERMEMORY_API_KEY}` } : {}) },
    body: definition.body === undefined ? undefined : JSON.stringify(replace(definition.body)),
  });
  const text = await response.text();
  let body;
  try { body = JSON.parse(text); } catch { body = text; }
  return { status: response.status, contentType: response.headers.get("content-type")?.split(";")[0] ?? null, body: normalize(body), raw: body };
}

async function run(baseUrl) {
  const variables = {};
  const results = {};
  for (const definition of cases) {
    const result = await request(baseUrl, definition, variables);
    if (definition.name === "v3-documents-create" && result.raw && typeof result.raw === "object") variables.documentId = result.raw.id;
    results[definition.name] = { status: result.status, contentType: result.contentType, body: result.body };
  }
  return results;
}

const oracleResults = await run(oracle);
await mkdir(dirname(fixturePath), { recursive: true });
if (acceptOracle) {
  await writeFile(fixturePath, `${JSON.stringify(oracleResults, null, 2)}\n`);
  console.log(`wrote ${fixturePath}`);
}
const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
if (JSON.stringify(oracleResults) !== JSON.stringify(fixture)) {
  throw new Error("upstream behavior differs from the reviewed v0.0.6 fixture; rerun with --accept-oracle after reviewing the difference");
}
if (candidate) {
  const candidateResults = await run(candidate);
  if (JSON.stringify(candidateResults) !== JSON.stringify(fixture)) {
    await writeFile(resolve("compat/fixtures/v0.0.6/candidate-actual.json"), `${JSON.stringify(candidateResults, null, 2)}\n`);
    throw new Error("candidate differs from v0.0.6 fixture; wrote compat/fixtures/v0.0.6/candidate-actual.json");
  }
}
console.log(`v0.0.6 parity passed for ${cases.length} cases`);
