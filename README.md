# MediNova Claims Lab — Databricks end to end

A hands-on learning project: **Unity Catalog → Medallion (Bronze/Silver/Gold) → Vector Search →
AI agent → evaluation → deployment → CI/CD**, built around one healthcare use case: a
claims-support agent for *MediNova Health*, a fictional health plan.

> All data is synthetic. Never load real PHI into a learning workspace.

## Repo layout

```
databricks.yml                 Asset Bundle: dev / staging / prod targets
resources/
  data_pipeline.job.yml        Job 1: UC setup -> land -> Bronze -> Silver -> Gold + Vector Search sync
  agent_release.job.yml        Job 2: tools -> build agent -> evaluate (quality gate) -> deploy
notebooks/
  00_config.py                 names of everything; widgets = job parameters
  01_unity_catalog_setup.py    catalog, schema, volume, tags, grants, information_schema
  02_land_sample_data.py       copies sample_data into the volume, one claims batch per run
  03_bronze_ingest.py          Auto Loader -> bronze tables
  04_silver_transform.py       clean, validate, quarantine, MERGE, chunk policy docs
  05_gold_and_governance.py    gold tables + column masks / row filter for PHI
  06_vector_search_index.py    endpoint + Delta Sync index, similarity & hybrid search
  07_agent_tools.py            UC SQL functions used as agent tools
  08_build_agent.py            test, trace, log, register agent (@challenger)
  09_evaluate_agent.py         mlflow.genai.evaluate + quality gate (@champion)
  10_deploy_and_monitor.py     agents.deploy, AI Gateway, inference tables, ai_query batch
  agent.py                     the agent itself (LangGraph + MLflow ResponsesAgent)
src/medinova/                  chunking + data-quality helpers (imported by notebooks)
tests/                         pytest unit tests (run in CI, no cluster needed)
sample_data/                   members, providers, claims batches, reference codes, policy docs, eval set
scripts/generate_sample_data.py  regenerates sample_data deterministically
.github/workflows/             ci.yml, deploy-staging.yml, deploy-prod.yml
```

## Quick start (interactive, in the workspace UI)

1. Push this folder to a new GitHub repo (see *Git setup* below).
2. In Databricks: **Workspace → Create → Git folder**, paste the repo URL.
3. Open `notebooks/00_config.py`; set `catalog` (default `workspace`) and `llm_endpoint` to a chat model
   listed under **Serving**.
4. Run notebooks `01` → `10` in order. Run `02` → `03` → `04` → `05` again to process the next claims batch.

## Quick start (as code, with the Databricks CLI)

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
# edit the host lines in databricks.yml first
databricks bundle validate -t dev
databricks bundle deploy   -t dev
databricks bundle run      -t dev medinova_data_pipeline    # run 1-4 times: one claims batch per run
databricks bundle run      -t dev medinova_agent_release
```

## Git setup and the CI/CD loop

```bash
cd medinova_claims_lab
git remote add origin https://github.com/<you>/medinova_claims_lab.git
git push -u origin main
```

GitHub repo settings:

| Setting | Value |
|---|---|
| Secrets (Settings → Secrets and variables → Actions) | `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` (OAuth secret of a Databricks **service principal**) |
| Environments | `staging`, and `production` with **Required reviewers** |
| Branch protection on `main` | require the `CI` checks to pass before merging |

The loop to practise:

1. `git checkout -b feature/better-prompt`
2. Change something (e.g. the system prompt in `notebooks/08_build_agent.py` or a rule in `src/medinova/quality.py`).
3. `pytest -q` locally, commit, push, open a pull request → **CI** runs lint, tests, data reproducibility, `bundle validate`.
4. Merge → **Deploy to staging** deploys the bundle, runs the data pipeline, builds and evaluates the agent.
   If evaluation scores drop below the thresholds in notebook 09, the run fails and nothing is promoted.
5. Publish a GitHub release (e.g. `v1.0.0`) → approve → **Deploy to production**.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
ruff check src tests scripts
python scripts/generate_sample_data.py   # rebuild sample_data
```

## Cost and clean-up

Vector Search endpoints and serving endpoints bill while they exist. When you finish a session:
delete the serving endpoint (Serving UI or `agents.delete_deployment`), delete the Vector Search
endpoint, and `DROP SCHEMA <catalog>.<schema> CASCADE`.
