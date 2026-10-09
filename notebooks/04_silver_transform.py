# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · SILVER — clean, validate, deduplicate, chunk
# MAGIC Silver turns raw Bronze rows into **trusted** tables:
# MAGIC
# MAGIC | Problem found in Bronze | Silver fix |
# MAGIC |---|---|
# MAGIC | `' paid '`, `'IN REVIEW'` | normalise status to one of 6 valid values |
# MAGIC | `07/14/2026` and `2026-07-14` mixed | parse both into a real `DATE` |
# MAGIC | amounts stored as text | cast to `DECIMAL(12,2)` |
# MAGIC | exact duplicate rows | drop |
# MAGIC | same claim in several batches | keep the **latest** version per `claim_id` (MERGE / upsert) |
# MAGIC | rows that still break rules | send to a **quarantine** table instead of silently dropping |
# MAGIC
# MAGIC The rules mirror `src/medinova/quality.py`, which is unit-tested in CI (see `tests/`).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md ## 1. Claims: standardise and validate

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW claims_typed AS
SELECT
  trim(claim_id)                                         AS claim_id,
  trim(member_id)                                        AS member_id,
  trim(provider_npi)                                     AS provider_npi,
  claim_type,
  coalesce(try_to_date(service_date, 'yyyy-MM-dd'),
           try_to_date(service_date, 'MM/dd/yyyy'))      AS service_date,
  try_to_date(received_date, 'yyyy-MM-dd')               AS received_date,
  procedure_code,
  diagnosis_code,
  try_cast(billed_amount AS DECIMAL(12,2))               AS billed_amount,
  try_cast(nullif(allowed_amount, '') AS DECIMAL(12,2))  AS allowed_amount,
  try_cast(nullif(paid_amount, '') AS DECIMAL(12,2))     AS paid_amount,
  try_cast(nullif(member_responsibility, '') AS DECIMAL(12,2)) AS member_responsibility,
  CASE lower(regexp_replace(claim_status, '\\\\s+', ''))
    WHEN 'received' THEN 'Received'   WHEN 'inreview' THEN 'In Review'
    WHEN 'pendinginfo' THEN 'Pending Info' WHEN 'approved' THEN 'Approved'
    WHEN 'paid' THEN 'Paid'           WHEN 'denied' THEN 'Denied'
  END                                                    AS claim_status,
  nullif(denial_reason_code, '')                         AS denial_reason_code,
  nullif(adjuster_note, '')                              AS adjuster_note,
  to_timestamp(updated_at)                               AS updated_at,
  _source_file
FROM (SELECT DISTINCT * EXCEPT (_ingested_at, _rescued_data) FROM {BRONZE_CLAIMS})   -- drop exact duplicates
""")

# Data-quality rules: a row is valid only if all are true
spark.sql("""
CREATE OR REPLACE TEMP VIEW claims_checked AS
SELECT *,
  array_compact(array(
    CASE WHEN claim_id IS NULL OR NOT claim_id RLIKE '^CLM-[0-9]{6}$' THEN 'bad_claim_id' END,
    CASE WHEN member_id IS NULL OR NOT member_id RLIKE '^M[0-9]{6}$' THEN 'bad_member_id' END,
    CASE WHEN service_date IS NULL THEN 'bad_service_date' END,
    CASE WHEN claim_status IS NULL THEN 'unknown_status' END,
    CASE WHEN billed_amount IS NULL OR billed_amount < 0 THEN 'bad_billed_amount' END,
    CASE WHEN claim_status = 'Denied' AND denial_reason_code IS NULL THEN 'denied_without_reason' END
  )) AS dq_errors
FROM claims_typed
""")
display(spark.sql("SELECT size(dq_errors) = 0 AS is_valid, count(*) FROM claims_checked GROUP BY 1"))

# COMMAND ----------

# MAGIC %md ## 2. Quarantine bad rows, upsert good rows (MERGE)

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER_CLAIMS} (
  claim_id STRING NOT NULL, member_id STRING, provider_npi STRING, claim_type STRING,
  service_date DATE, received_date DATE, procedure_code STRING, diagnosis_code STRING,
  billed_amount DECIMAL(12,2), allowed_amount DECIMAL(12,2), paid_amount DECIMAL(12,2),
  member_responsibility DECIMAL(12,2), claim_status STRING, denial_reason_code STRING,
  adjuster_note STRING, updated_at TIMESTAMP, _source_file STRING,
  CONSTRAINT silver_claims_pk PRIMARY KEY (claim_id)
) COMMENT 'One row per claim, latest status. Cleaned and validated.'
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {SILVER_CLAIMS}_quarantine AS
SELECT * FROM claims_checked WHERE size(dq_errors) > 0
""")

spark.sql(f"""
MERGE INTO {SILVER_CLAIMS} AS t
USING (
  SELECT * EXCEPT (dq_errors) FROM claims_checked
  WHERE size(dq_errors) = 0
  QUALIFY row_number() OVER (PARTITION BY claim_id ORDER BY updated_at DESC) = 1   -- latest version
) AS s
ON t.claim_id = s.claim_id
WHEN MATCHED AND s.updated_at > t.updated_at THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")
display(spark.sql(f"SELECT claim_status, count(*) n FROM {SILVER_CLAIMS} GROUP BY 1 ORDER BY 2 DESC"))

# COMMAND ----------

# MAGIC %md ## 3. Members: type the columns, keep one row per member

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {SILVER_MEMBERS}
COMMENT 'Members with typed eligibility dates. Contains PHI (name, DOB, email, phone).'
AS SELECT
  member_id, first_name, last_name,
  to_date(date_of_birth) AS date_of_birth, gender, zip_code, city, state, plan_id,
  to_date(coverage_start) AS coverage_start, to_date(coverage_end) AS coverage_end,
  email, phone
FROM {BRONZE_MEMBERS}
QUALIFY row_number() OVER (PARTITION BY member_id ORDER BY _ingested_at DESC) = 1
""")
spark.sql(f"ALTER TABLE {SILVER_MEMBERS} SET TAGS ('layer' = 'silver', 'contains_phi' = 'true')")
for col in ["first_name", "last_name", "date_of_birth", "email", "phone"]:
    spark.sql(f"ALTER TABLE {SILVER_MEMBERS} ALTER COLUMN {col} SET TAGS ('phi' = 'true')")
spark.sql(f"ALTER TABLE {SILVER_CLAIMS} SET TAGS ('layer' = 'silver', 'contains_phi' = 'true')")
spark.sql(f"ALTER TABLE {SILVER_CLAIMS} ALTER COLUMN diagnosis_code SET TAGS ('phi' = 'true')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Policy documents → chunks (for RAG)
# MAGIC We chunk with the unit-tested `medinova.chunking.chunk_markdown` (split on headings, then
# MAGIC on sentences with overlap, each chunk prefixed with "Document > Section").
# MAGIC Eight documents fit easily on the driver; for thousands of PDFs you would use a pandas UDF
# MAGIC or `ai_parse_document` instead.
# MAGIC
# MAGIC **Change Data Feed** must be on: Vector Search's Delta Sync index reads changes from it.

# COMMAND ----------

from medinova.chunking import chunk_markdown

latest_docs = spark.sql(f"""
  SELECT doc_name, raw_text FROM {BRONZE_DOCS}
  QUALIFY row_number() OVER (PARTITION BY doc_name ORDER BY file_modified_at DESC) = 1
""").collect()

chunks = [c.to_dict() for d in latest_docs for c in chunk_markdown(d.doc_name, d.raw_text,
                                                                   max_chars=1000, overlap_chars=150)]
spark.createDataFrame(chunks).createOrReplaceTempView("new_chunks")
print(f"{len(latest_docs)} documents -> {len(chunks)} chunks")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER_CHUNKS} (
  chunk_id STRING NOT NULL, doc_name STRING, section STRING, chunk_index BIGINT, chunk_text STRING,
  CONSTRAINT silver_chunks_pk PRIMARY KEY (chunk_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
COMMENT 'Chunked policy documents; source table for the Vector Search index.'
""")

# Chunk ids are content hashes: unchanged text -> same id -> no re-embedding.
spark.sql(f"""
MERGE INTO {SILVER_CHUNKS} t USING new_chunks s ON t.chunk_id = s.chunk_id
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE THEN DELETE
""")
display(spark.table(SILVER_CHUNKS).orderBy("doc_name", "chunk_index"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Is `silver_claims_quarantine` empty? Land a CSV with a broken row (e.g. claim_id `CLM-1`) and see where it goes.
# MAGIC 2. Why `WHEN MATCHED AND s.updated_at > t.updated_at` rather than always updating?
# MAGIC 3. Change a sentence in a policy doc, re-land it and re-run 03 → 04. How many chunks changed?
