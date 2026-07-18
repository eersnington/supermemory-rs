import Supermemory from "supermemory";

const baseURL = process.env.SUPERMEMORY_BASE_URL;
const apiKey = process.env.SUPERMEMORY_API_KEY;
if (!baseURL || !apiKey) throw new Error("SUPERMEMORY_BASE_URL and SUPERMEMORY_API_KEY are required");

const client = new Supermemory({ apiKey, baseURL });
const tag = `sdk-js-${Date.now()}`;
const added = await client.add({
  content: `JavaScript SDK compatibility memory ${tag}`,
  containerTags: [tag],
  customId: tag,
  taskType: "superrag",
});

let document;
for (let attempt = 0; attempt < 100; attempt += 1) {
  document = await client.documents.get(added.id);
  if (document.status === "done" || document.status === "failed") break;
  await Bun.sleep(50);
}
if (document?.status !== "done") throw new Error(`document did not complete: ${document?.status}`);

const v3 = await client.search.documents({
  q: tag,
  containerTags: [tag],
  limit: 5,
  chunkThreshold: 0,
  includeFullDocs: true,
});
const v4 = await client.search.memories({
  q: tag,
  containerTag: tag,
  limit: 5,
  threshold: 0,
  searchMode: "documents",
});
const profile = await client.profile({ containerTag: tag });

if (v3.total < 1) throw new Error("V3 SDK search returned no relevant chunks");
if (v4.total < 1) throw new Error("V4 SDK search returned no document chunks");
if (!profile.profile) throw new Error("V4 profile response is missing profile");
console.log(JSON.stringify({ sdk: "javascript", added: added.id, v3: v3.total, v4: v4.total }));
