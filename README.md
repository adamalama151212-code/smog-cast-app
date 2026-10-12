# smog-cast-app

The app of the **smogcast** project (interface built with [Reflex](https://reflex.dev)): will the daily PM10 / PM2.5
limit be exceeded tomorrow in one of the 10 largest cities of Poland, and how certain is that. It reads its data from
the **`gold_secure`** views (Unity Catalog, row- and column-level security) through a **Databricks SQL Warehouse** –
no Spark and no Java in the image.

The project's other repositories: the data pipeline (Databricks, asset bundle) and the infrastructure (Terraform).
The project's working documents (journal, migration plan, handoff notes) are kept outside this repository.

## Structure

| File | What it does |
|---|---|
| `smogcast/app/` | the app's logic, **this repo's own, edited here**: `assistant` (routing, prompts, text-to-SQL, RAG), `forecast`, `guardrails`, `llm`, `rag`, `text`, `gold.py` (warehouse / saved copy), `config.py`, `services.py` (the assistant, the model loaded at start), `conf/app.yaml` |
| `smogcast/app/conf/{cities,thresholds,model,gold_dictionary}.yaml`, `smogcast/core/schema.py` | **the data contract – a copy from the pipeline repo, not to be edited here** (the pipeline produces the gold tables according to these files; the dictionary becomes the descriptions, tags and masks in Unity Catalog). A change in the pipeline → `python tools/copy_logic.py`; checksums in `smogcast/app/COPIED.txt`, `--check` compares with the pipeline |
| `docs/rag/`, `data/rag_sources/`, `data/rag_index/` | **the assistant's knowledge base, all of it here (since 2026-10-11):** documents written for the assistant, the official sources as downloaded (PDF/HTML) and the index built from them (436 passages) |
| `smogcast_app/design.py` | design tokens (colours of both themes, forecast states, typography, spacing, motion) → `assets/design.css` |
| `assets/app.css` | component styles, built on the tokens only |
| `smogcast_app/components/` | `shell.py` (top bar, bottom bar on a phone, footer with the data source and "Refresh data", page header, loading and error states, SQL panel, state badge), `icons.py` (own icons) |
| `smogcast_app/pages/` | `info.py`, `tomorrow.py` (+ `/tomorrow/preview`: a smog day from 2025), `city.py` (`/city/<city>`), `model.py`, `data_quality.py`, `assistant.py` |
| `smogcast_app/components/charts.py` | charts (HTML bars, SVG calibration plot) on the design tokens |
| `smogcast_app/poland_outline.py` | the outline of Poland (Natural Earth, public domain) |
| `tests/` | tests of the logic and the data layer (including the guard of the contract: `test_data_contract_is_unchanged`) |
| `tools/shots.py` | smoke test of every page (dark/light × desktop/phone) + screenshots in `screenshots/` |
| `tools/ask_eval.py` | **control questions for the assistant** on the real model and data: 15 questions, for each one the route, the time, the SQL, the rows, the answer and a check of the expected fact; results in `tools/eval/<label>.json` (outside git); `--compare before after`, `--recheck`, `--only`; it paces its own requests to the model's rate limit |
| `tools/copy_logic.py` | copies the data contract from the pipeline repo and records the checksums (`--check` only compares) |
| `tools/build_index.py` | builds the knowledge-base index from the repo's files (`docker compose --profile tools run --rm index`; `--check` – is it up to date, `--download` – fetches the missing sources) |
| `.github/workflows/ci.yml` | tests on every commit of a pull request (the same libraries as in the image) |
| `.github/workflows/build-index.yml` | manual: builds the knowledge-base index and commits it to the branch it was started on – when `test_the_index_matches_the_documents` fails in `ci`, nothing has to be built locally |
| `.github/workflows/deploy.yml` | **deployment** – manual: Actions → deploy → "Use workflow from" = the branch → `dev`. Builds the image from that branch, pushes it to GitHub Container Registry (tag = commit), signs in to Azure through OIDC (no password) and puts the image on the app in Azure Container Apps; waits until the new version answers. The environment's settings (ids, not secrets) are GitHub Environment variables – listed in the file's header |

## Running locally (Docker, Git Bash)

```bash
. tools/dev-env.sh            # the workspace, the warehouse and a short-lived token (about 1 h, in the shell's memory only)
docker compose up -d app      # frontend http://localhost:3000, backend :8000; code changes reload by themselves
docker compose --profile tools run --rm test                         # tests of the logic and the data (no warehouse)
docker compose --profile tools run --rm shots python tools/shots.py   # smoke test + screenshots in screenshots/
docker compose --profile tools run --rm index                         # knowledge base: build the index after changing documents
# the assistant's control questions (the real model; about 5 min at a limit of 5 requests/min), then a comparison of two runs:
MSYS_NO_PATHCONV=1 docker exec -i -w /app smog-cast-app-app-1 python -u tools/ask_eval.py --label after
MSYS_NO_PATHCONV=1 docker exec -i -w /app smog-cast-app-app-1 python -u tools/ask_eval.py --compare before after
```

Run every `docker compose` command in a shell after `. tools/dev-env.sh` – otherwise compose recreates the container
without a token. The token expires after about 1 h: `. tools/dev-env.sh` again, then `docker compose up -d app`.

Without a warehouse (for example while the Azure account changes): `SMOGCAST_DATA=sample` in `.env` (or in the
shell) – the data comes from the saved copy `tools/sample/*.json` (the 10 `gold_secure` views as of 2026-10-09,
outside git and outside the image); the interface then says that it shows a saved copy. The technical columns are
set to NULL in the copy, as they are for the `analyst` role.

The image is **production by default** (the frontend is built into the image, `reflex run --env prod --single-port`,
one port, 3000); the development server is an override of the command in the `app` service. To check the image "as
hosted":

```bash
docker compose build app                                                  # in the background
docker compose --profile prod up -d app-prod                              # http://localhost:3001, 1 CPU / 2 GiB
docker compose --profile prod run --rm shots-prod python tools/shots.py   # smoke test on that image
docker compose --profile prod rm -sf app-prod                             # stop only the production container
```

## Settings (environment variables; never in git)

Locally the language model and the data source are set in the **`.env`** file next to `docker-compose.yml` –
`docker compose` reads it by itself (the file is in `.gitignore` and `.dockerignore`, so it reaches neither git nor
the image); variables from the shell win over `.env`. The Databricks token does not go into `.env`: it is
short-lived and comes from `. tools/dev-env.sh`. Example:

```
SMOGCAST_DATA=sample
SMOGCAST_LLM_BASE_URL=https://api.cerebras.ai/v1
SMOGCAST_LLM_MODEL=gpt-oss-120b
SMOGCAST_LLM_API_KEY=<key>
DATABRICKS_PROFILE=<Databricks CLI profile>         # only for SMOGCAST_DATA=auto / warehouse
DATABRICKS_HOST=https://adb-<workspace>.azuredatabricks.net
DATABRICKS_WAREHOUSE_ID=<warehouse id>
```

| Variable | Meaning |
|---|---|
| `DATABRICKS_HOST` | the workspace's address (locally in `.env`) |
| `DATABRICKS_WAREHOUSE_ID` | the SQL Warehouse (locally in `.env`) |
| `DATABRICKS_PROFILE` | local only: the Databricks CLI profile that `. tools/dev-env.sh` takes the short-lived token from |
| `DATABRICKS_TOKEN` | locally the author's short-lived token (`. tools/dev-env.sh`); not used by the hosted app |
| `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` | **hosting:** the app's service principal (role `analyst`), OAuth machine-to-machine sign-in – the SQL connector takes the token from `databricks-sdk` and renews it by itself (valid for 1 h). The secret comes from Key Vault, never from a file. When both are set they win over the token; the id alone, without the secret, is not enough |
| `SMOGCAST_LLM_BASE_URL`, `SMOGCAST_LLM_MODEL`, `SMOGCAST_LLM_API_KEY` | the language model (any OpenAI-compatible endpoint; today **Cerebras `gpt-oss-120b`** – free plan, answers in ~1–2 s; Gemini was rejected: 20 free requests a day per model) |
| `SMOGCAST_CATALOG` | the environment's catalog (default `smogcast_dev`); the schema is always `gold_secure` |
| `SMOGCAST_DATA` | `auto` (default: the warehouse if it is configured, the saved copy otherwise; the copy also when the warehouse does not answer), `warehouse`, `sample` |
| `SMOGCAST_BASE_URL` | smoke test only (`tools/shots.py`): the address of the app under test, default `http://localhost:3000`. The same test against the public address: `SMOGCAST_BASE_URL=https://<address> docker compose --profile tools run --rm shots python tools/shots.py` (the `app` container must be running – the test uses its network) |
