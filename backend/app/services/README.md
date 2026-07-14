# Service Boundaries

The Web backend wraps the existing script pipeline instead of importing and rewriting it all at once.

- `yaml_store.py` owns business configuration reads and writes (saving brands also syncs the product master).
- `product_master.py` owns the `products` table, synced from `config/brands.yaml` (`brand_999`).
- `job_store.py` owns the self-service pipeline: collect → analyze → extract → verify → import → materialize.
- `question_parser.py` parses uploads and anchors product names to master `product_code`s.
- `sqlite_dashboard.py` owns workbench reads (datasets/overview/samples/splits over materialized tables).
- `insight_store.py` owns business-facing reads (health cards, three-stage journey, trends, evidence chain).
- `source_insight_store.py` owns source/citation reads (domain normalization, catalog classification, multi-product comparison, gaps and answer drill-down).
- `user_config_store.py` owns per-user brands/knowledge-base overrides.

Existing CLI scripts should remain runnable from the command line. When a script gains new options for the Web app, add them as optional argparse parameters so current usage keeps working. Schema lives in `utils/sqlite_schema.py` (single source, lazy migration, shared by CLI and backend startup).
