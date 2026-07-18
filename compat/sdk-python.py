import json
import os
import time

from supermemory import Supermemory

base_url = os.environ["SUPERMEMORY_BASE_URL"]
api_key = os.environ["SUPERMEMORY_API_KEY"]
client = Supermemory(api_key=api_key, base_url=base_url)
tag = f"sdk-py-{time.time_ns()}"
added = client.add(
    content=f"Python SDK compatibility memory {tag}",
    container_tags=[tag],
    custom_id=tag,
    task_type="superrag",
)

document = None
for _ in range(100):
    document = client.documents.get(added.id)
    if document.status in ("done", "failed"):
        break
    time.sleep(0.05)
if document is None or document.status != "done":
    raise RuntimeError(f"document did not complete: {getattr(document, 'status', None)}")

v3 = client.search.documents(
    q=tag,
    container_tags=[tag],
    limit=5,
    chunk_threshold=0,
    include_full_docs=True,
)
v4 = client.search.memories(
    q=tag,
    container_tag=tag,
    limit=5,
    threshold=0,
    search_mode="documents",
)
profile = client.profile(container_tag=tag)
if v3.total < 1:
    raise RuntimeError("V3 SDK search returned no relevant chunks")
if v4.total < 1:
    raise RuntimeError("V4 SDK search returned no document chunks")
if profile.profile is None:
    raise RuntimeError("V4 profile response is missing profile")
print(json.dumps({"sdk": "python", "added": added.id, "v3": v3.total, "v4": v4.total}))
