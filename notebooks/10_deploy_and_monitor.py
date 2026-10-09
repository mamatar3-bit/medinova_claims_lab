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

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1.3" databricks-agents databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import time, mlflow
from mlflow import MlflowClient
from databricks import agents

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)          # deployed traces land in this experiment

version = MlflowClient().get_model_version_by_alias(UC_MODEL_NAME, "champion").version
deployment = agents.deploy(UC_MODEL_NAME, version, scale_to_zero_enabled=True,
                           tags={"project": "medinova_claims"})
ENDPOINT = deployment.endpoint_name
print(f"Deploying {UC_MODEL_NAME} v{version} to endpoint {ENDPOINT}")
print("Review App:", getattr(deployment, "review_app_url", "see the Serving UI"))

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

for _ in range(60):                                   # deployments can take ~15 minutes
    ep = w.serving_endpoints.get(ENDPOINT)
    if str(ep.state.ready).endswith("READY") and "NOT_UPDATING" in str(ep.state.config_update):
        break
    print("  endpoint:", ep.state.ready, ep.state.config_update)
    time.sleep(30)
print("Endpoint ready:", ENDPOINT)

# COMMAND ----------

# MAGIC %md ## Query the endpoint like an application would

# COMMAND ----------

from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

resp = client.predict(endpoint=ENDPOINT, inputs={
    "input": [{"role": "user", "content": "Why was claim CLM-100042 denied and how does the member appeal?"}]})
print([o for o in resp["output"] if o.get("type") == "message"][-1]["content"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Gateway: usage tracking + rate limit
# MAGIC Rate limits protect cost and stop abuse; usage tracking writes token usage to system tables.

# COMMAND ----------

from databricks.sdk.service.serving import (AiGatewayRateLimit, AiGatewayRateLimitKey,
                                            AiGatewayRateLimitRenewalPeriod, AiGatewayUsageTrackingConfig)
try:
    current = w.serving_endpoints.get(ENDPOINT).ai_gateway
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT,
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        rate_limits=[AiGatewayRateLimit(calls=60, key=AiGatewayRateLimitKey.USER,
                                        renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
        inference_table_config=current.inference_table_config if current else None,  # keep tables on
    )
    print("AI Gateway configured: 60 calls/minute per user, usage tracking on")
except Exception as e:
    print("Configure AI Gateway in the Serving UI instead:", str(e).splitlines()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor: what did real users ask, and how did the agent do?
# MAGIC Two places to look:
# MAGIC 1. **MLflow experiment → Traces**: every request with its full trace; add scorers for production monitoring.
# MAGIC 2. **Inference table** (a Delta table in UC): request/response payloads you can query with SQL.

# COMMAND ----------

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
    print("Inference tables not enabled on this endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference with `ai_query()` — triage adjuster notes
# MAGIC Real-time endpoints are for chat. For "run an LLM over every row", use SQL.

# COMMAND ----------

display(spark.sql(f"""
SELECT claim_id, claim_status, adjuster_note,
       ai_query(
         '{LLM_ENDPOINT}',
         concat('Classify this health-insurance adjuster note into exactly one label: ',
                'APPEAL_LIKELY, NEEDS_PROVIDER_ACTION, NEEDS_MEMBER_ACTION, ROUTINE. ',
                'Answer with the label only. Note: ', adjuster_note)
       ) AS triage_label
FROM {SILVER_CLAIMS}
WHERE adjuster_note IS NOT NULL AND claim_status IN ('Denied', 'Pending Info')
LIMIT 50
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Send 5 questions from the Review App, then find them in the inference table and the traces.
# MAGIC 2. When is `ai_query()` a better choice than calling the serving endpoint in a loop?
# MAGIC 3. Where would you look first if latency doubled after a new version was deployed?
# MAGIC 4. Clean-up: `agents.delete_deployment(model_name=UC_MODEL_NAME, model_version=version)` when done.
