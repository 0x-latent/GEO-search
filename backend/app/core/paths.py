from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

CONFIG_DIR = ROOT_DIR / "config"
QUESTIONS_DIR = ROOT_DIR / "questions"
RESULTS_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
SCRIPTS_DIR = ROOT_DIR / "scripts"
GEO_DATA_DIR = ROOT_DIR / "data" / "geo_datasets"
GEO_SQLITE_PATH = GEO_DATA_DIR / "geo_answers.sqlite"

DATA_DIR = BACKEND_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
RUNS_DIR = DATA_DIR / "runs"

for path in (DATA_DIR, TASKS_DIR, RUNS_DIR, GEO_DATA_DIR):
    path.mkdir(parents=True, exist_ok=True)
