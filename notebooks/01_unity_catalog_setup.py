# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Unity Catalog setup
# MAGIC Unity Catalog (UC) is the governance layer for **everything** in this lab: files, tables,
# MAGIC functions, vector indexes and the agent model itself.
# MAGIC
# MAGIC **The hierarchy (learn this first):**
# MAGIC ```
# MAGIC Metastore  (one per region, set up by an account admin)
# MAGIC └── Catalog            e.g. workspace            ← top-level container, often one per env/team
# MAGIC     └── Schema         e.g. medinova_claims        ← a "database"
# MAGIC         ├── Volume     raw                       ← governed files (CSV, PDF, Markdown…)
# MAGIC         ├── Table      bronze_claims, silver_…   ← Delta tables
# MAGIC         ├── Function   get_claim_status          ← SQL/Python functions (agent tools!)
# MAGIC         ├── Model      claims_support_agent     ← registered MLflow models
# MAGIC         └── Index      policy_chunks_index         ← Vector Search index
# MAGIC ```
# MAGIC Every object is addressed with a **three-level name**: `catalog.schema.object`.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md ## 1. Where am I? Inspect the metastore and your identity

# COMMAND ----------

display(spark.sql("SELECT current_metastore() AS metastore, current_user() AS me, current_catalog() AS catalog"))
display(spark.sql("SHOW CATALOGS"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Catalog
# MAGIC Creating a catalog needs the `CREATE CATALOG` privilege on the metastore.
# MAGIC In Free Edition or a shared workspace you usually **can't**, so the lab defaults to the
# MAGIC existing `workspace` catalog. If you are an admin, try a dedicated catalog instead:
# MAGIC set the `catalog` widget at the top to e.g. `medinova` and the cell below creates it.

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'MediNova claims learning lab'")
    print(f"Catalog ready: {CATALOG}")
except Exception as e:
    print(f"Could not create catalog {CATALOG} (fine if it already exists and you can use it):\n{e}")
spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md ## 3. Schema and volume

# COMMAND ----------

spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}
  COMMENT 'MediNova claims lab: medallion tables, governed PHI, agent tools and agent'
""")
spark.sql(f"USE SCHEMA {SCHEMA}")

# A *managed* volume: UC decides where the files live and governs access to them.
# (An *external* volume points at a cloud path you already own — needs an external location.)
spark.sql(f"""
  CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}
  COMMENT 'Landing zone for raw files, plus Auto Loader checkpoints'
""")

for p in [LANDING_DOCS, LANDING_MEMBERS, LANDING_CLAIMS, LANDING_REFERENCE, CHECKPOINTS]:
    dbutils.fs.mkdirs(p)

display(spark.sql(f"DESCRIBE SCHEMA EXTENDED {CATALOG}.{SCHEMA}"))
display(spark.sql(f"SHOW VOLUMES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tags — make data discoverable and classifiable
# MAGIC Tags are key/value labels searchable in Catalog Explorer. Governance teams use them to
# MAGIC mark PII, owners, environments and so on.

# COMMAND ----------

spark.sql(f"ALTER SCHEMA {CATALOG}.{SCHEMA} SET TAGS ('project' = 'medinova_claims', 'data_domain' = 'healthcare_claims', 'env' = 'dev')")
display(spark.sql(f"""
  SELECT * FROM {CATALOG}.information_schema.schema_tags
  WHERE schema_name = '{SCHEMA}'
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Permissions (GRANT / REVOKE)
# MAGIC Privileges are **inherited downwards**: a grant on a schema applies to every table in it.
# MAGIC To read a table a user needs all three:
# MAGIC `USE CATALOG` on the catalog + `USE SCHEMA` on the schema + `SELECT` on the table.
# MAGIC
# MAGIC Below we give the built-in `account users` group read access to the schema.
# MAGIC Replace it with a real group (for example `claims-analysts`) in a real project.

# COMMAND ----------

GROUP = "account users"
for stmt in [
    f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{GROUP}`",
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {CATALOG}.{SCHEMA} TO `{GROUP}`",
    f"GRANT READ VOLUME ON VOLUME {CATALOG}.{SCHEMA}.{VOLUME} TO `{GROUP}`",
]:
    try:
        spark.sql(stmt)
        print("OK   ", stmt)
    except Exception as e:
        print("SKIP ", stmt, "->", str(e).splitlines()[0])

display(spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. information_schema — query the catalog itself with SQL
# MAGIC Re-run this cell at the end of the lab to see every object you created.

# COMMAND ----------

display(spark.sql(f"""
  SELECT table_name, table_type, comment
  FROM {CATALOG}.information_schema.tables
  WHERE table_schema = '{SCHEMA}'
  ORDER BY table_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check yourself
# MAGIC 1. What three privileges does a user need to `SELECT` from `workspace.medinova_claims.silver_claims`?
# MAGIC 2. What is the difference between a managed and an external volume?
# MAGIC 3. Where in the UI do you see **lineage**? *(Catalog Explorer → a table → Lineage tab — check it after notebook 05.)*
