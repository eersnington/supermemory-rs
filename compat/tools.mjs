import { supermemoryTools } from "@supermemory/tools/ai-sdk";
import { createToolCallExecutor, getToolDefinitions } from "@supermemory/tools/openai";

const apiKey = process.env.SUPERMEMORY_API_KEY;
const baseUrl = process.env.SUPERMEMORY_BASE_URL;
if (!apiKey || !baseUrl) throw new Error("SUPERMEMORY_BASE_URL and SUPERMEMORY_API_KEY are required");

const aiTools = supermemoryTools(apiKey, { baseUrl, containerTags: ["tools-smoke"] });
const definitions = getToolDefinitions();
const execute = createToolCallExecutor(apiKey, { baseUrl, containerTags: ["tools-smoke"] });
if (Object.keys(aiTools).length === 0 || definitions.length === 0 || typeof execute !== "function") {
  throw new Error("@supermemory/tools did not construct its supported tool surfaces");
}
console.log(JSON.stringify({ tools: Object.keys(aiTools), definitions: definitions.length }));
