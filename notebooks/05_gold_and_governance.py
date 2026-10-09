# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · GOLD tables + PHI governance
# MAGIC **Gold** tables answer business questions directly. We build four:
# MAGIC
# MAGIC | Gold table | Who uses it |
# MAGIC |---|---|
# MAGIC | `gold_claims_status_daily` | operations dashboard: volume and dollars by status |
# MAGIC | `gold_denials_by_reason` | denial management: which reasons cost the most |
# MAGIC | `gold_member_accumulators` | member services + the agent: deductible / out-of-pocket progress |
# MAGIC | `gold_claim_detail` | the agent: one readable row per claim, with code descriptions and next steps |
# MAGIC
# MAGIC Then we protect PHI with Unity Catalog **column masks** and a **row filter**.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md ## 1. Gold tables

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD_CLAIMS_DAILY}
COMMENT 'Daily claim counts and dollars by status (from latest claim versions).'
AS SELECT
  date(updated_at) AS as_of_date, claim_status,
  count(*) AS claims, sum(billed_amount) AS billed, sum(paid_amount) AS paid
FROM {SILVER_CLAIMS}
GROUP BY ALL
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD_DENIALS}
COMMENT 'Denied claims by reason code with dollars at stake.'
AS SELECT
  c.denial_reason_code, d.description, d.usually_resolved_by,
  count(*) AS denied_claims, sum(c.billed_amount) AS billed_at_stake,
  round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_of_denials
FROM {SILVER_CLAIMS} c
LEFT JOIN {REF_DENIALS} d USING (denial_reason_code)
WHERE c.claim_status = 'Denied'
GROUP BY ALL
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD_ACCUMULATORS}
COMMENT 'Plan-year 2026 deductible and out-of-pocket progress per member (no PHI).'
AS
WITH spend AS (
  SELECT member_id, sum(member_responsibility) AS member_paid_ytd, count(*) AS paid_claims
  FROM {SILVER_CLAIMS}
  WHERE claim_status = 'Paid' AND year(service_date) = 2026
  GROUP BY member_id
)
SELECT
  m.member_id, m.plan_id, p.plan_name, p.network_type,
  p.deductible, p.out_of_pocket_max,
  coalesce(s.member_paid_ytd, 0)                                    AS member_paid_ytd,
  least(coalesce(s.member_paid_ytd, 0), p.deductible)               AS deductible_met,
  greatest(p.deductible - coalesce(s.member_paid_ytd, 0), 0)        AS deductible_remaining,
  greatest(p.out_of_pocket_max - coalesce(s.member_paid_ytd, 0), 0) AS oop_remaining,
  coalesce(s.paid_claims, 0)                                        AS paid_claims,
  m.coverage_start, m.coverage_end
FROM {SILVER_MEMBERS} m
JOIN {REF_PLANS} p USING (plan_id)
LEFT JOIN spend s USING (member_id)
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD_CLAIM_DETAIL}
COMMENT 'One readable row per claim for member services and the AI agent. No names, DOB or diagnosis.'
AS SELECT
  c.claim_id, c.member_id, m.plan_id, c.claim_status, c.claim_type,
  c.service_date, c.received_date, date(c.updated_at) AS last_updated,
  c.procedure_code, pc.description AS service_description, pc.requires_prior_auth,
  pr.provider_name, pr.specialty AS provider_specialty, pr.network_status AS provider_network,
  c.billed_amount, c.allowed_amount, c.paid_amount, c.member_responsibility,
  c.denial_reason_code, d.description AS denial_reason, d.next_step AS denial_next_step,
  CASE WHEN c.claim_status = 'Denied'
       THEN date_add(date(c.updated_at), 180) END AS appeal_deadline
FROM {SILVER_CLAIMS} c
LEFT JOIN {SILVER_MEMBERS} m USING (member_id)
LEFT JOIN {REF_PROCEDURES} pc USING (procedure_code)
LEFT JOIN {REF_PROVIDERS} pr USING (provider_npi)
LEFT JOIN {REF_DENIALS} d USING (denial_reason_code)
""")

for t in [GOLD_CLAIMS_DAILY, GOLD_DENIALS, GOLD_ACCUMULATORS, GOLD_CLAIM_DETAIL]:
    spark.sql(f"ALTER TABLE {t} SET TAGS ('layer' = 'gold')")

display(spark.table(GOLD_DENIALS).orderBy("denied_claims", ascending=False))
display(spark.sql(f"SELECT * FROM {GOLD_CLAIM_DETAIL} WHERE claim_id = 'CLM-100042'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Column masks — hide PHI from anyone not in `phi_readers`
# MAGIC A mask is a SQL function UC runs **every time** the column is read. The table stays one
# MAGIC table; what you see depends on who you are. (Create the `phi_readers` group in the account
# MAGIC console to test; until then everyone, including you, sees masked values.)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.mask_phi_string(value STRING)
RETURNS STRING
COMMENT 'Shows PHI only to members of {PHI_GROUP}'
RETURN CASE WHEN is_account_group_member('{PHI_GROUP}') THEN value ELSE '***REDACTED***' END
""")
spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.mask_phi_date(value DATE)
RETURNS DATE
RETURN CASE WHEN is_account_group_member('{PHI_GROUP}') THEN value ELSE NULL END
""")

for col in ["first_name", "last_name", "email", "phone"]:
    spark.sql(f"ALTER TABLE {SILVER_MEMBERS} ALTER COLUMN {col} SET MASK {CATALOG}.{SCHEMA}.mask_phi_string")
spark.sql(f"ALTER TABLE {SILVER_MEMBERS} ALTER COLUMN date_of_birth SET MASK {CATALOG}.{SCHEMA}.mask_phi_date")
# Diagnosis codes are PHI too. We leave silver_claims unmasked here because notebook 04 MERGEs into it
# every run; in production, mask it and run the pipeline as a service principal in `phi_readers`.

display(spark.sql(f"SELECT member_id, first_name, last_name, date_of_birth, email FROM {SILVER_MEMBERS} LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Row filter — analysts only see members in their state (demo)
# MAGIC Members of `va_claims_analysts` see Virginia rows only; `phi_readers` see everything.
# MAGIC Left **commented out** so it does not hide rows from you while you work; try it, then drop it.

# COMMAND ----------

# spark.sql(f"""
# CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.state_filter(state STRING)
# RETURNS BOOLEAN
# RETURN is_account_group_member('{PHI_GROUP}')
#     OR (is_account_group_member('va_claims_analysts') AND state = 'VA')
# """)
# spark.sql(f"ALTER TABLE {SILVER_MEMBERS} SET ROW FILTER {CATALOG}.{SCHEMA}.state_filter ON (state)")
# spark.sql(f"ALTER TABLE {SILVER_MEMBERS} DROP ROW FILTER")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Least privilege for the agent
# MAGIC The agent never needs names, DOB or diagnoses, so its tools read **only** the Gold tables
# MAGIC `gold_claim_detail` and `gold_member_accumulators`. That is a governance design decision:
# MAGIC *minimum necessary* access, as HIPAA requires.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Open Catalog Explorer → `gold_claim_detail` → **Lineage**. Which Silver and reference tables feed it?
# MAGIC 2. Why is masking done in Unity Catalog instead of in each notebook or dashboard?
# MAGIC 3. What is the difference between a column mask and a row filter?
