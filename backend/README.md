# GEO Search Web Workbench

Internal Web workbench for managing GEO analysis configuration, running the existing script pipeline, browsing analysis CSVs, and reviewing BI metrics.

## Run

One-click startup on Windows:

```powershell
.\start_dashboard.bat
```

This starts the FastAPI backend and serves the static dashboard frontend from the same local service. Open `http://127.0.0.1:8000`.

Manual startup:

```powershell
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Scope

- `brands.yaml` visual editing with version snapshots under `backend/data/runs`.
- Run configuration snapshots with `run_id`.
- Script task execution with logs under `backend/data/tasks`.
- CSV previews from `results/analysis`.
- BI overview generated from existing analysis CSVs.

## Script Management

Scripts are launched through `backend/app/services/task_runner.py`. The runner records status in `backend/data/tasks/tasks.json` and writes one log file per task. The script catalog is intentionally centralized there so Web-visible pipeline steps stay explicit.

## Docker

From the repository root:

```powershell
docker compose up --build -d
```

The compose file mounts runtime state from the host:

- `config/api_keys.yaml`
- `config/brands.yaml`
- `config/knowledge_base.json`
- `questions/`
- `docs/products_db/`
- `results/`
- `backend/data/`
