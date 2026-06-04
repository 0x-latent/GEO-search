# Service Boundaries

The Web backend wraps the existing script pipeline instead of importing and rewriting it all at once.

- `yaml_store.py` owns business configuration reads and writes.
- `run_store.py` owns immutable run configuration snapshots.
- `task_runner.py` owns script execution, status, and logs.
- `analysis_store.py` owns BI and CSV reads from `results/analysis`.

Existing CLI scripts should remain runnable from the command line. When a script gains new options for the Web app, add them as optional argparse parameters so current usage keeps working.

