# SQLite 数据管理方案

目标：把现有 8 产品基线数据和三九胃泰/养胃舒专项数据统一导入 SQLite，后续分析从 SQLite 调取，不再依赖 Excel 文件作为运行时数据源。

## 数据库位置

默认数据库：

```text
data/geo_datasets/geo_answers.sqlite
```

## 核心表

| 表 | 作用 |
| --- | --- |
| `datasets` | 数据集登记表。区分 8 产品基线、养胃舒专项等 |
| `products` | 产品主数据 |
| `questions` | 标准问题表 |
| `answers` | 每条 AI 回答事实表 |
| `sources` | 每条回答的引用信源明细 |
| `import_files` | 原始文件导入审计，含路径、hash、大小、修改时间 |
| `external_tables` | Excel/CSV 工作表登记 |
| `external_rows` | Excel/CSV 每行原样 JSON 化保存 |

## 当前数据集

| dataset_id | 内容 |
| --- | --- |
| `baseline_8products_20260423` | 现有 `results/raw` 的 8 产品问答 JSON，以及 `results/analysis` 下的分析 CSV |
| `weitai_yangweishu_20260602` | `questions/三九养胃舒数据源` 下的 AI 回答明细、信源明细、提及推荐率、社媒热度、问题库与选词建议 |

当前已导入状态：

| dataset_id | 标准问题 | 标准回答 | 产品 | 模型 | 外部表 | 外部行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_8products_20260423` | 128 | 4,445 | 8 | 4 | 18 | 86,793 |
| `weitai_yangweishu_20260602` | 143 | 4,290 | 1 | 6 | 214 | 60,033 |

注意：8 产品基线中部分 JSON 文件的 `search_enabled` 字段与文件名不一致。导入时优先使用文件名中的 `_search` / `_nosearch` 判断采集模式，并在 `answers.metadata_json` 中记录 `search_mode_source`。

## 初始化和导入

初始化 schema：

```powershell
.\.venv\Scripts\python.exe scripts\manage_geo_sqlite.py init
```

一次性导入两套数据：

```powershell
.\.venv\Scripts\python.exe scripts\manage_geo_sqlite.py import-all --reset
```

只导入 8 产品基线：

```powershell
.\.venv\Scripts\python.exe scripts\manage_geo_sqlite.py import-baseline --reset
```

只导入养胃舒专项：

```powershell
.\.venv\Scripts\python.exe scripts\manage_geo_sqlite.py import-yangweishu --reset
```

查看库内概况：

```powershell
.\.venv\Scripts\python.exe scripts\manage_geo_sqlite.py summary
```

## 给现有分析脚本使用

短期内，现有 `04/05/07/08` 脚本仍读取 `results/raw` 和 `questions/*.json`。需要分析某个 SQLite 数据集时，先导出成现有结构：

```powershell
.\.venv\Scripts\python.exe scripts\manage_geo_sqlite.py export-raw `
  --dataset-id weitai_yangweishu_20260602 `
  --output-raw-dir data\exports\weitai_yangweishu_20260602\raw `
  --questions-output data\exports\weitai_yangweishu_20260602\imported_questions.json `
  --reset-output
```

后续可以把分析脚本改成直接按 `dataset_id` 查询 SQLite，逐步取消 `results/raw` 中间层。

## 常用 SQL

按数据集看产品、模型、回答量：

```sql
SELECT dataset_id, product_name, model, search_enabled, COUNT(*) AS answers
FROM answers
GROUP BY dataset_id, product_name, model, search_enabled
ORDER BY dataset_id, product_name, model, search_enabled;
```

查询养胃舒各层级问题数量：

```sql
SELECT source_level, scenario, COUNT(DISTINCT question_id) AS questions
FROM questions
WHERE dataset_id = 'weitai_yangweishu_20260602'
GROUP BY source_level, scenario
ORDER BY source_level, scenario;
```

查某个数据集引用最多的域名：

```sql
SELECT domain, COUNT(*) AS refs
FROM sources
WHERE dataset_id = 'weitai_yangweishu_20260602'
GROUP BY domain
ORDER BY refs DESC
LIMIT 20;
```

查外部 Excel 已入库的工作表：

```sql
SELECT dataset_id, table_name, sheet_name, row_count
FROM external_tables
ORDER BY dataset_id, table_name, sheet_name;
```

## 管理原则

- Excel 和 CSV 只作为导入原件与审计来源，不作为分析运行时依赖。
- 新数据必须先进入 `datasets/questions/answers/sources` 标准结构。
- 不能结构化到核心问答表的数据，至少进入 `external_tables/external_rows`，保证可追溯。
- 所有分析必须显式指定 `dataset_id`，避免 8 产品基线和养胃舒专项样本混用。
