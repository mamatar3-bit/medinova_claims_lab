# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Lab configuration — MediNova Health claims lab
# MAGIC Every other notebook runs this one with `%run ./00_config`.
# MAGIC The widgets below become **job parameters** when the bundle deploys the pipeline,
# MAGIC so dev, staging and prod can point at different catalogs without code changes.
# MAGIC
# MAGIC **All data in this lab is synthetic.** Never load real PHI into a learning workspace;
# MAGIC production healthcare workloads need a compliance-enabled workspace and a BAA.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "medinova_claims", "Schema")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-120b", "LLM endpoint")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# Storage (Unity Catalog volume) for raw files and Auto Loader checkpoints
VOLUME = "raw"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
LANDING_DOCS = f"{VOLUME_PATH}/landing/policy_docs"
LANDING_MEMBERS = f"{VOLUME_PATH}/landing/members"
LANDING_CLAIMS = f"{VOLUME_PATH}/landing/claims"
LANDING_REFERENCE = f"{VOLUME_PATH}/landing/reference"
CHECKPOINTS = f"{VOLUME_PATH}/_checkpoints"

# Repo paths (notebooks run from <repo>/notebooks, so the repo root is one level up)
import os, sys
REPO_ROOT = os.path.abspath("..")
SAMPLE_DATA = os.path.join(REPO_ROOT, "sample_data")
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(REPO_ROOT, "src"))      # makes `import medinova` work

# Medallion tables
BRONZE_DOCS = f"{CATALOG}.{SCHEMA}.bronze_policy_docs"
BRONZE_MEMBERS = f"{CATALOG}.{SCHEMA}.bronze_members"
BRONZE_CLAIMS = f"{CATALOG}.{SCHEMA}.bronze_claims"
SILVER_MEMBERS = f"{CATALOG}.{SCHEMA}.silver_members"
SILVER_CLAIMS = f"{CATALOG}.{SCHEMA}.silver_claims"
SILVER_CHUNKS = f"{CATALOG}.{SCHEMA}.silver_policy_chunks"
GOLD_CLAIMS_DAILY = f"{CATALOG}.{SCHEMA}.gold_claims_status_daily"
GOLD_DENIALS = f"{CATALOG}.{SCHEMA}.gold_denials_by_reason"
GOLD_ACCUMULATORS = f"{CATALOG}.{SCHEMA}.gold_member_accumulators"
GOLD_CLAIM_DETAIL = f"{CATALOG}.{SCHEMA}.gold_claim_detail"

# Reference tables (small, static)
REF_PROVIDERS = f"{CATALOG}.{SCHEMA}.ref_providers"
REF_PLANS = f"{CATALOG}.{SCHEMA}.ref_plans"
REF_PROCEDURES = f"{CATALOG}.{SCHEMA}.ref_procedure_codes"
REF_DIAGNOSES = f"{CATALOG}.{SCHEMA}.ref_diagnosis_codes"
REF_DENIALS = f"{CATALOG}.{SCHEMA}.ref_denial_reason_codes"

# Vector Search
VS_ENDPOINT = "medinova_vs_endpoint"
VS_INDEX = f"{CATALOG}.{SCHEMA}.policy_chunks_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"   # Foundation Model API embedding model

# LLM (any chat model listed under Serving > Foundation models in your workspace)
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")

# Unity Catalog functions used as agent tools
CLAIM_TOOL = f"{CATALOG}.{SCHEMA}.get_claim_status"
BENEFITS_TOOL = f"{CATALOG}.{SCHEMA}.get_member_benefits"

# Governance
PHI_GROUP = "phi_readers"          # members of this group see unmasked PHI

# Agent registration
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.claims_support_agent"
EXPERIMENT_NAME = f"/Shared/medinova_claims_agent_{SCHEMA}"

print(f"Catalog.schema : {CATALOG}.{SCHEMA}")
print(f"Volume path    : {VOLUME_PATH}")
print(f"LLM endpoint   : {LLM_ENDPOINT}")
