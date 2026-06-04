# Script Pipeline Notes

The Web workbench treats these scripts as the stable pipeline boundary:

| Step | Script | Web role |
| --- | --- | --- |
| 01 | `01_parse_questions.py` | Parse source question files |
| 02 | `02_expand_questions.py` | Generate variants |
| 03 | `03_query_models.py` | Collect model answers |
| 04 | `04_analyze_results.py` | Produce baseline CSVs |
| 05 | `05_extract_recommendations.py` | Extract recommendation structure |
| 06 | `06_build_knowledge_base.py` | Build `config/knowledge_base.json` |
| 07 | `07_verify_accuracy.py` | Verify Q1/Q2 answers against the knowledge base |
| 08 | `08_generate_report.py` | Generate narrative Markdown report |

Keep CLI compatibility when optimizing scripts. New Web-specific controls should be exposed as optional arguments, with defaults matching current behavior.

