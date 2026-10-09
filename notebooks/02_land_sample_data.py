# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Land sample data in the volume (the "source system")
# MAGIC This notebook plays the outside world dropping files into your **landing zone**
# MAGIC (`/Volumes/<catalog>/<schema>/raw/landing/...`). It copies files from the repo's
# MAGIC `sample_data/` folder:
# MAGIC
# MAGIC | Run | What lands |
# MAGIC |---|---|
# MAGIC | 1st | 8 policy documents, members, reference tables, **claims batch 1** (608 rows) |
# MAGIC | 2nd, 3rd, 4th | the next claims batch: status updates + 80 new claims each |
# MAGIC | 5th+ | nothing new (all 4 batches landed) |
# MAGIC
# MAGIC So each scheduled job run gives Bronze → Silver → Gold a fresh, *incremental* batch.
# MAGIC Prefer clicking? You can also upload the same files with **Catalog Explorer → volume → Upload**.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os, shutil

def copy_dir(src: str, dst: str, pattern_ext: tuple) -> list:
    os.makedirs(dst, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(src)):
        if name.endswith(pattern_ext) and not os.path.exists(os.path.join(dst, name)):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
            copied.append(name)
    return copied

assert os.path.isdir(SAMPLE_DATA), f"sample_data not found at {SAMPLE_DATA}. Clone the repo into a Git folder."

# Static files: only copied the first time (copy_dir skips files already present)
print("policy docs :", copy_dir(f"{SAMPLE_DATA}/policy_docs", LANDING_DOCS, (".md",)))
print("members     :", copy_dir(f"{SAMPLE_DATA}/members", LANDING_MEMBERS, (".csv",)))
print("reference   :", copy_dir(f"{SAMPLE_DATA}/reference", LANDING_REFERENCE, (".csv",)))

# Claims: land exactly ONE new batch per run
os.makedirs(LANDING_CLAIMS, exist_ok=True)
available = sorted(f for f in os.listdir(f"{SAMPLE_DATA}/claims") if f.endswith(".csv"))
landed = set(os.listdir(LANDING_CLAIMS))
pending = [f for f in available if f not in landed]
if pending:
    shutil.copy(f"{SAMPLE_DATA}/claims/{pending[0]}", f"{LANDING_CLAIMS}/{pending[0]}")
    print(f"claims      : landed {pending[0]}  ({len(pending) - 1} batch(es) still to come)")
else:
    print("claims      : all batches already landed - nothing new this run")

# COMMAND ----------

display(dbutils.fs.ls(LANDING_CLAIMS))
