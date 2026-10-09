# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · BRONZE — ingest raw files exactly as they arrived
# MAGIC **The medallion architecture in one line each**
# MAGIC * **Bronze** = raw data as it arrived + ingestion metadata. Never cleaned, append-only. Lets you replay history.
# MAGIC * **Silver** = cleaned, typed, deduplicated, validated and PHI-protected. The trusted "single source of truth".
# MAGIC * **Gold** = business-ready tables: KPIs for dashboards and the curated views the AI agent reads.
# MAGIC
# MAGIC We use **Auto Loader** (`cloudFiles`) so each run picks up *only new files*. Its checkpoint
# MAGIC remembers what was already loaded, so re-running is safe (idempotent).
# MAGIC `trigger(availableNow=True)` = "process everything new, then stop" — streaming code, batch cost.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F

def autoload_csv(source_dir: str, target_table: str, name: str):
    """Incrementally load CSV files into a Bronze table. Every column stays STRING."""
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "false")          # Bronze keeps raw strings
        .option("cloudFiles.schemaLocation", f"{CHECKPOINTS}/{name}_schema")
        .option("cloudFiles.schemaEvolutionMode", "rescue")       # unexpected columns -> _rescued_data
        .load(source_dir)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    query = (stream.writeStream
             .option("checkpointLocation", f"{CHECKPOINTS}/{name}")
             .trigger(availableNow=True)
             .toTable(target_table))
    query.awaitTermination()
    print(f"{target_table}: {spark.table(target_table).count()} rows total")

# COMMAND ----------

# MAGIC %md ## Claims and members (CSV)

# COMMAND ----------

autoload_csv(LANDING_CLAIMS, BRONZE_CLAIMS, "bronze_claims")
autoload_csv(LANDING_MEMBERS, BRONZE_MEMBERS, "bronze_members")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Policy documents (Markdown) → `bronze_policy_docs`
# MAGIC `binaryFile` reads each file whole; we decode the bytes to text. For PDFs you would keep
# MAGIC the bytes here and parse them in Silver (for example with the `ai_parse_document` SQL function).

# COMMAND ----------

docs = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.md")
    .load(LANDING_DOCS)
    .select(
        F.col("path").alias("doc_path"),
        F.regexp_extract("path", r"([^/]+)$", 1).alias("doc_name"),
        F.col("content").cast("string").alias("raw_text"),
        F.col("modificationTime").alias("file_modified_at"),
        F.current_timestamp().alias("_ingested_at"),
    )
)
(docs.writeStream
    .option("checkpointLocation", f"{CHECKPOINTS}/bronze_policy_docs")
    .trigger(availableNow=True)
    .toTable(BRONZE_DOCS)
    .awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference tables
# MAGIC Small, slowly changing lookups (plans, providers, codes) are simply overwritten each run.

# COMMAND ----------

for file_name, table in [("plans.csv", REF_PLANS), ("providers.csv", REF_PROVIDERS),
                         ("procedure_codes.csv", REF_PROCEDURES), ("diagnosis_codes.csv", REF_DIAGNOSES),
                         ("denial_reason_codes.csv", REF_DENIALS)]:
    (spark.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LANDING_REFERENCE}/{file_name}")
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table))
    print(f"{table}: {spark.table(table).count()} rows")

# COMMAND ----------

spark.sql(f"COMMENT ON TABLE {BRONZE_CLAIMS} IS 'Raw claim rows from the claims system, all strings, append-only. Contains PHI.'")
spark.sql(f"COMMENT ON TABLE {BRONZE_MEMBERS} IS 'Raw member eligibility file. Contains PHI.'")
spark.sql(f"ALTER TABLE {BRONZE_CLAIMS} SET TAGS ('layer' = 'bronze', 'contains_phi' = 'true')")
spark.sql(f"ALTER TABLE {BRONZE_MEMBERS} SET TAGS ('layer' = 'bronze', 'contains_phi' = 'true')")

display(spark.sql(f"""
  SELECT _source_file, count(*) AS rows FROM {BRONZE_CLAIMS}
  GROUP BY _source_file ORDER BY _source_file
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look for the messy data on purpose
# MAGIC Bronze keeps problems that Silver must fix. Run these and note what you see.

# COMMAND ----------

display(spark.sql(f"SELECT claim_status, count(*) n FROM {BRONZE_CLAIMS} GROUP BY 1 ORDER BY 2 DESC"))
display(spark.sql(f"SELECT count(*) AS us_style_dates FROM {BRONZE_CLAIMS} WHERE service_date LIKE '%/%'"))
display(spark.sql(f"""
  SELECT claim_id, updated_at, count(*) AS copies FROM {BRONZE_CLAIMS}
  GROUP BY ALL HAVING count(*) > 1 ORDER BY copies DESC LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. Re-run notebook 02, then this one. Why do only the new batch's rows get added?
# MAGIC 2. Why does Bronze keep every column as a string?
# MAGIC 3. You found three kinds of mess above. Which layer fixes them? *(Silver — next notebook.)*
