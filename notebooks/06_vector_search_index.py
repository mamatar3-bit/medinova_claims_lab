# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Vector Search index over the policy chunks
# MAGIC * A **Vector Search endpoint** is the compute that serves queries (create once, reuse for many indexes).
# MAGIC * A **Delta Sync index** stays in sync with a Delta table (`silver_policy_chunks`) via Change Data Feed.
# MAGIC * **Managed embeddings**: we give it the text column and an embedding endpoint
# MAGIC   (`databricks-gte-large-en`); Databricks computes the vectors for us.
# MAGIC * `pipeline_type="TRIGGERED"` = sync only when we call `sync()` (cheap; our docs change rarely).
# MAGIC   `CONTINUOUS` = near real-time, costs more.
# MAGIC
# MAGIC In the scheduled job this notebook only **syncs**; the first run creates everything.
# MAGIC (Newer docs may call this product *AI Search*; the API is the same.)

# COMMAND ----------

# MAGIC %pip install -q --no-deps databricks-vectorsearch deprecation
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# 1. Endpoint (create once)
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print("Creating endpoint (can take ~10 minutes)…")

while vsc.get_endpoint(VS_ENDPOINT)["endpoint_status"]["state"] != "ONLINE":
    print("  endpoint state:", vsc.get_endpoint(VS_ENDPOINT)["endpoint_status"]["state"])
    time.sleep(30)
print("Endpoint ONLINE:", VS_ENDPOINT)

# COMMAND ----------

# 2. Index (create once, then just sync)
def index_exists() -> bool:
    try:
        vsc.get_index(VS_ENDPOINT, VS_INDEX)
        return True
    except Exception:
        return False

if not index_exists():
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=SILVER_CHUNKS,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    print("Index created; initial sync started.")
else:
    vsc.get_index(VS_ENDPOINT, VS_INDEX).sync()
    print("Sync triggered for existing index.")

index = vsc.get_index(VS_ENDPOINT, VS_INDEX)
for _ in range(90):                                   # wait up to ~30 minutes
    status = index.describe().get("status", {})
    state = status.get("detailed_state", "")
    if status.get("ready") and state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE"):
        break
    print("  index state:", state)
    time.sleep(20)
print("Index status:", index.describe().get("status", {}).get("detailed_state"))

# COMMAND ----------

# MAGIC %md ## 3. Try a similarity search (this is what the agent's retriever tool will do)

# COMMAND ----------

for q in ["Does an MRI need prior authorization?", "How long do I have to appeal a denied claim?"]:
    res = index.similarity_search(query_text=q, columns=["doc_name", "section", "chunk_text"], num_results=3)
    print(f"\nQ: {q}")
    for doc_name, section, text, score in res["result"]["data_array"]:
        print(f"  {score:.3f}  {doc_name} > {section}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Try hybrid search (keyword + meaning)
# MAGIC Exact codes such as `DN06` are a weak spot for pure embeddings; hybrid search helps.

# COMMAND ----------

res = index.similarity_search(query_text="DN06", columns=["doc_name", "section"], num_results=3,
                              query_type="HYBRID")
display(res["result"]["data_array"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Why does the source table need Change Data Feed enabled?
# MAGIC 2. When would you pick a Direct Vector Access index instead of Delta Sync?
# MAGIC 3. Compare the ANN and HYBRID results for "DN06". Which ranks the denial-codes guide first?
