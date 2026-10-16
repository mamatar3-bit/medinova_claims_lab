# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Deploy the `@champion` agent, query it, monitor it, and run batch AI
# MAGIC * **Deploy** with `databricks.agents.deploy()` → a Model Serving endpoint with tracing,
# MAGIC   inference tables and the Review App for SME feedback.
# MAGIC * **Govern** the endpoint with AI Gateway: usage tracking and rate limits.
# MAGIC * **Monitor** with inference tables / MLflow traces.
# MAGIC * **Batch**: classify thousands of adjuster notes with `ai_query()` in SQL — no endpoint code.
# MAGIC
# MAGIC Model Serving is the deployment path the certification exam covers. Databricks now also offers
# MAGIC Agent Runtime on Databricks Apps (Agent Bricks CLI) — see the stretch goals in the lab guide.
# MAGIC
# MAGIC **Run top to bottom.** This workspace requires scale-to-zero on every served model. The deploy
# MAGIC cell removes old endpoints for this model that were created without it, and if
# MAGIC `agents.deploy()` still fails on scale-to-zero, it creates the endpoint directly.

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.1.3" databricks-agents "databricks-sdk==0.122.0" "protobuf==6.33.5"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# Fail fast if 00_config did not define everything this notebook needs
REQUIRED = ["UC_MODEL_NAME", "EXPERIMENT_NAME", "LLM_ENDPOINT", "SILVER_CLAIMS"]
missing = [name for name in REQUIRED if name not in globals()]
assert not missing, f"Missing from 00_config: {missing}. Re-run the %run ./00_config cell."

# COMMAND ----------

# MAGIC %md ## Deploy the `@champion` version

# COMMAND ----------

import time, mlflow
from mlflow import MlflowClient
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, EndpointTag

mlflow.set_registry_uri("databricks-uc")
exp = mlflow.set_experiment(EXPERIMENT_NAME)          # deployed traces land in this experiment

w = WorkspaceClient()
version = MlflowClient().get_model_version_by_alias(UC_MODEL_NAME, "champion").version
model_short = UC_MODEL_NAME.split(".")[-1]

def _state(x):
    return getattr(x, "value", str(x))

def serves_this_model(ep) -> bool:
    """True if the endpoint is named after this model or serves it."""
    if model_short in ep.name:
        return True
    for cfg in (ep.config, ep.pending_config):
        for e in (getattr(cfg, "served_entities", None) or []):
            if getattr(e, "entity_name", None) == UC_MODEL_NAME:
                return True
    return False

def wait_deleted(name: str):
    for _ in range(60):
        try:
            w.serving_endpoints.get(name)
            time.sleep(10)
        except NotFound:
            print(f"  deleted {name}")
            return

# 1. Find every endpoint for this model (by name or served entity) and delete any that has a
#    served model without scale-to-zero. agents.deploy() updates an existing endpoint rather than
#    replacing it, so one such served model makes every new deploy fail.
for summary in w.serving_endpoints.list():
    try:
        ep = w.serving_endpoints.get(summary.name)
    except NotFound:
        continue
    if not serves_this_model(ep):
        continue
    entities = []
    for cfg in (ep.config, ep.pending_config):
        entities += (getattr(cfg, "served_entities", None) or [])
    print(f"Found endpoint {ep.name}: " +
          (", ".join(f"{e.name} scale_to_zero={e.scale_to_zero_enabled}" for e in entities) or "no served models"))
    if entities and all(e.scale_to_zero_enabled for e in entities):
        continue
    print(f"Deleting {ep.name} (has a served model without scale-to-zero)")
    w.serving_endpoints.delete(ep.name)
    wait_deleted(ep.name)

# 2. Deploy. Prefer agents.deploy (adds the Review App); fall back to creating the endpoint
#    directly with scale-to-zero forced on if agents.deploy still hits the scale-to-zero rule.
REVIEW_APP = None
try:
    deployment = agents.deploy(UC_MODEL_NAME, version, scale_to_zero_enabled=True,
                               tags={"project": "medinova_claims"})
    ENDPOINT = deployment.endpoint_name
    REVIEW_APP = getattr(deployment, "review_app_url", None)
except Exception as e:
    if "Scale to zero" not in str(e):
        raise
    print("agents.deploy did not apply scale-to-zero; creating the endpoint directly instead.")
    ENDPOINT = f"agents_{UC_MODEL_NAME.replace('.', '-')}"[:63]
    try:
        w.serving_endpoints.get(ENDPOINT)
        w.serving_endpoints.delete(ENDPOINT)
        wait_deleted(ENDPOINT)
    except NotFound:
        pass
    w.serving_endpoints.create(
        name=ENDPOINT,
        config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(
            entity_name=UC_MODEL_NAME,
            entity_version=str(version),
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "ENABLE_MLFLOW_TRACING": "true",
                "MLFLOW_EXPERIMENT_ID": exp.experiment_id,
            },
        )]),
        tags=[EndpointTag(key="project", value="medinova_claims")],
    )

print(f"Deploying {UC_MODEL_NAME} v{version} to endpoint {ENDPOINT}")
print("Review App:", REVIEW_APP or "not available on this path; test from the Serving UI or Playground")

# COMMAND ----------

# MAGIC %md ## Wait until the endpoint is ready (can take ~15 minutes)

# COMMAND ----------

ready_ok = False
for _ in range(60):
    try:
        ep = w.serving_endpoints.get(ENDPOINT)
    except NotFound:
        print("  endpoint not visible yet ...")
        time.sleep(30)
        continue
    ready = _state(ep.state.ready)                      # "READY" or "NOT_READY"
    updating = _state(ep.state.config_update)           # "NOT_UPDATING", "IN_PROGRESS", "UPDATE_FAILED"
    if ready == "READY" and updating == "NOT_UPDATING":
        ready_ok = True
        break
    if updating == "UPDATE_FAILED":
        raise Exception(f"Deployment failed: open Serving > {ENDPOINT} > Events to see why")
    print("  endpoint:", ready, updating)
    time.sleep(30)

if not ready_ok:
    raise Exception(f"Endpoint {ENDPOINT} not ready after 30 minutes. Check Serving > {ENDPOINT} > Events.")
print("Endpoint ready:", ENDPOINT)

# COMMAND ----------

# MAGIC %md ## Query the endpoint like an application would

# COMMAND ----------

from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

def extract_text(resp) -> str:
    """Return only the final answer text; skip reasoning blocks from reasoning models."""
    messages = [o for o in resp.get("output", []) if isinstance(o, dict) and o.get("type") == "message"]
    if not messages:
        return str(resp)
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in (content or [])
                     if isinstance(c, dict) and c.get("type") in ("output_text", "text")).strip()

def ask_endpoint(question: str, retries: int = 6) -> str:
    """Scale-to-zero endpoints can take a minute to wake up, so retry on transient errors."""
    for attempt in range(retries):
        try:
            resp = client.predict(endpoint=ENDPOINT,
                                  inputs={"input": [{"role": "user", "content": question}]})
            return extract_text(resp)
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in ("503", "504", "429", "timed out", "Timeout", "scaled to zero"))
            if not transient or attempt == retries - 1:
                raise
            print(f"  endpoint waking up / busy, retrying in 20s ({attempt + 1}/{retries})")
            time.sleep(20)

print(ask_endpoint("Why was claim CLM-100042 denied and how does the member appeal?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Gateway: usage tracking + rate limit
# MAGIC Rate limits protect cost and stop abuse; usage tracking writes token usage to system tables.

# COMMAND ----------

try:
    from databricks.sdk.service.serving import (AiGatewayRateLimit, AiGatewayRateLimitKey,
                                                AiGatewayRateLimitRenewalPeriod, AiGatewayUsageTrackingConfig)
    current = w.serving_endpoints.get(ENDPOINT).ai_gateway
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT,
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        rate_limits=[AiGatewayRateLimit(calls=60, key=AiGatewayRateLimitKey.USER,
                                        renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
        # keep whatever was already configured
        inference_table_config=current.inference_table_config if current else None,
        guardrails=current.guardrails if current else None,
        fallback_config=current.fallback_config if current else None,
    )
    print("AI Gateway configured: 60 calls/minute per user, usage tracking on")
except Exception as e:
    print("AI Gateway not configured here (configure it in the Serving UI instead):",
          str(e).splitlines()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor: what did real users ask, and how did the agent do?
# MAGIC Two places to look:
# MAGIC 1. **MLflow experiment → Traces**: every request with its full trace; add scorers for production monitoring.
# MAGIC 2. **Inference table** (a Delta table in UC): request/response payloads you can query with SQL.

# COMMAND ----------

# 1. Recent traces from the deployed endpoint (always available)
try:
    recent = mlflow.search_traces(max_results=10, order_by=["timestamp_ms DESC"])
    cols = [c for c in ["request_time", "timestamp_ms", "request", "response", "execution_duration", "state"]
            if c in recent.columns]
    display(recent[cols] if cols else recent)
except Exception as e:
    print("Could not read traces:", str(e).splitlines()[0])

# 2. Inference table, if the endpoint has one
cfg = w.serving_endpoints.get(ENDPOINT).ai_gateway
tbl = cfg.inference_table_config if cfg else None
if tbl and tbl.enabled:
    inference_table = f"{tbl.catalog_name}.{tbl.schema_name}.{tbl.table_name_prefix}_payload"
    print("Inference table:", inference_table, "(rows appear a few minutes after traffic)")
    try:
        display(spark.sql(f"""
          SELECT request_time, status_code, execution_duration_ms, request, response
          FROM {inference_table} ORDER BY request_time DESC LIMIT 20"""))
    except Exception as e:
        print("Not populated yet:", str(e).splitlines()[0])
else:
    print("Inference tables not enabled on this endpoint; use the MLflow traces above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference with `ai_query()` — triage adjuster notes
# MAGIC Real-time endpoints are for chat. For "run an LLM over every row", use SQL.

# COMMAND ----------

try:
    display(spark.sql(f"""
    SELECT claim_id, claim_status, adjuster_note,
           upper(trim(ai_query(
             '{LLM_ENDPOINT}',
             concat('Classify this health-insurance adjuster note into exactly one label: ',
                    'APPEAL_LIKELY, NEEDS_PROVIDER_ACTION, NEEDS_MEMBER_ACTION, ROUTINE. ',
                    'Answer with the label only. Note: ', adjuster_note)
           ))) AS triage_label
    FROM {SILVER_CLAIMS}
    WHERE adjuster_note IS NOT NULL AND claim_status IN ('Denied', 'Pending Info')
    LIMIT 50
    """))
except Exception as e:
    print("ai_query failed:", str(e).splitlines()[0])
    print("Check that LLM_ENDPOINT is a chat endpoint you can query and that this compute supports ai_query.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean-up (run only when you are done with the lab)
# MAGIC Deleting the deployment removes the endpoint and its URL.

# COMMAND ----------

# agents.delete_deployment(model_name=UC_MODEL_NAME, model_version=version)
# If the endpoint was created directly (fallback path), delete it with:
# w.serving_endpoints.delete(ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Send 5 questions from the Review App, then find them in the inference table and the traces.
# MAGIC 2. When is `ai_query()` a better choice than calling the serving endpoint in a loop?
# MAGIC 3. Where would you look first if latency doubled after a new version was deployed?
# MAGIC 4. Clean-up: `agents.delete_deployment(model_name=UC_MODEL_NAME, model_version=version)` when done.
