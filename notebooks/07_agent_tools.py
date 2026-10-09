# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Agent tools as Unity Catalog functions
# MAGIC An agent tool is just a function with a **clear name, description and typed inputs** —
# MAGIC the LLM reads the description to decide when to call it.
# MAGIC
# MAGIC Registering tools as **UC functions** means they are governed like tables:
# MAGIC versioned in the catalog, permissioned with `GRANT EXECUTE`, visible in lineage, and
# MAGIC reusable from SQL, notebooks, Genie and any agent.
# MAGIC
# MAGIC | Tool | Reads | Never returns |
# MAGIC |---|---|---|
# MAGIC | `get_claim_status(claim_id)` | `gold_claim_detail` | names, DOB, diagnosis |
# MAGIC | `get_member_benefits(member_id)` | `gold_member_accumulators` | names, DOB, contact details |
# MAGIC
# MAGIC The third tool, the policy-document retriever, comes from the Vector Search index (notebook 08).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CLAIM_TOOL}(
  claim_id_input STRING COMMENT 'Claim ID in the format CLM-123456'
)
RETURNS TABLE (
  claim_id STRING, member_id STRING, plan_id STRING, claim_status STRING, service_date DATE,
  received_date DATE, last_updated DATE, service_description STRING, requires_prior_auth BOOLEAN,
  provider_name STRING, provider_network STRING, billed_amount DECIMAL(12,2),
  allowed_amount DECIMAL(12,2), paid_amount DECIMAL(12,2), member_responsibility DECIMAL(12,2),
  denial_reason_code STRING, denial_reason STRING, denial_next_step STRING, appeal_deadline DATE
)
COMMENT 'Looks up one health insurance claim by claim ID. Returns status, service, provider, amounts, and for denied claims the denial reason, next step and appeal deadline. Use for any question about a specific claim.'
RETURN
  SELECT claim_id, member_id, plan_id, claim_status, service_date, received_date, last_updated,
         service_description, requires_prior_auth, provider_name, provider_network,
         billed_amount, allowed_amount, paid_amount, member_responsibility,
         denial_reason_code, denial_reason, denial_next_step, appeal_deadline
  FROM {GOLD_CLAIM_DETAIL}
  WHERE claim_id = upper(trim(claim_id_input))
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {BENEFITS_TOOL}(
  member_id_input STRING COMMENT 'Member ID in the format M123456'
)
RETURNS TABLE (
  member_id STRING, plan_name STRING, network_type STRING, deductible DOUBLE,
  deductible_met DOUBLE, deductible_remaining DOUBLE, out_of_pocket_max DOUBLE,
  oop_remaining DOUBLE, paid_claims BIGINT, coverage_start DATE, coverage_end DATE
)
COMMENT 'Returns a member''s plan and their plan-year deductible and out-of-pocket progress. Use when asked how much deductible is met or remaining, or which plan a member has.'
RETURN
  SELECT member_id, plan_name, network_type,
         CAST(deductible AS DOUBLE), CAST(deductible_met AS DOUBLE), CAST(deductible_remaining AS DOUBLE),
         CAST(out_of_pocket_max AS DOUBLE), CAST(oop_remaining AS DOUBLE),
         paid_claims, coverage_start, coverage_end
  FROM {GOLD_ACCUMULATORS}
  WHERE member_id = upper(trim(member_id_input))
""")

# COMMAND ----------

# MAGIC %md ## Test the tools exactly as the agent will call them

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CLAIM_TOOL}('CLM-100042')"))
display(spark.sql(f"SELECT * FROM {BENEFITS_TOOL}('m100007')"))     # lower case on purpose

# COMMAND ----------

# MAGIC %md
# MAGIC ## Permissions
# MAGIC Users (and the deployed agent) need `EXECUTE` on the function, plus read access to the
# MAGIC Gold table it queries. Notebook 08 declares both as *resources* when it logs the agent, so
# MAGIC the serving endpoint gets short-lived credentials for exactly these objects.

# COMMAND ----------

for fn in [CLAIM_TOOL, BENEFITS_TOOL]:
    try:
        spark.sql(f"GRANT EXECUTE ON FUNCTION {fn} TO `account users`")
    except Exception as e:
        print("Grant skipped:", str(e).splitlines()[0])
display(spark.sql(f"SHOW USER FUNCTIONS IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Rewrite the `get_claim_status` description badly (e.g. "gets data"). How might the agent misuse it?
# MAGIC 2. Why return `appeal_deadline` from the tool instead of letting the LLM compute it?
