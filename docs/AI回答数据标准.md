# AI回答数据标准

本文定义外部采集的 AI 回答数据如何接入当前 GEO 分析框架。核心原则是：所有来源先标准化为同一个“回答事实表”，分析脚本只读取标准层，不直接适配不同 Excel。

## 1. 分层

### 1.1 原始层

保留数据方交付的原始文件，不直接修改。

示例：

- `questions/【养胃舒】明细数据（AI回答）.xlsx`
- `questions/三九养胃舒数据源/...`

### 1.2 标准问题层

每个唯一问题生成一条问题定义，保存为：

- `questions/imported_questions_<dataset_id>.json`

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 问题唯一 ID，规则为 `{product_code}_{level_code}_{hash}` |
| `product` | 是 | 当前报告使用的产品短名，如 `胃泰` |
| `product_code` | 是 | 产品代码，如 `weitai` |
| `category` | 是 | 分析类别，推荐/提及类统一为 `mention_recommend` |
| `level` | 是 | 标准问题层级，必须能被分析方法识别 |
| `question` | 是 | 原始问题文本 |
| `has_brand_name` | 是 | 问题中是否显式出现品牌名/产品名 |
| `is_variant` | 是 | 是否为已有基准问题变体。外部批量问题默认 `false` |
| `variant_of` | 否 | 变体来源问题 ID |
| `source_level` | 否 | 外部文件中的原始层级名称 |
| `scenario` | 否 | 外部文件中的场景名称 |

### 1.3 标准回答层

每一条 AI 回答转成当前框架已支持的 JSON，保存到：

- `results/raw/<model_key>/<question_id>_r<round>_<search|nosearch>.json`

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `question_id` | 是 | 对应标准问题层的 `id` |
| `question_text` | 是 | 问题文本 |
| `product` | 是 | 产品短名 |
| `model` | 是 | 标准模型代码，如 `deepseek`、`qwen`、`doubao` |
| `model_name` | 是 | 展示名称，如 `DeepSeek`、`通义千问` |
| `search_enabled` | 是 | 是否联网或是否带引用信源 |
| `round` | 是 | 查询轮次，整数 |
| `answer` | 是 | AI 原始回答 |
| `timestamp` | 是 | 查询时间，ISO 字符串优先 |
| `sources` | 是 | 引用信源列表，结构为 `[{ "title": "", "url": "" }]` |
| `latency_ms` | 否 | API 响应耗时；外部数据没有可留空 |
| `external_meta` | 否 | 数据集、原始文件、原始行号、场景等审计信息 |

### 1.4 分析输出层

标准回答层生成后，继续使用现有脚本：

- `scripts/04_analyze_results.py`
- `scripts/05_extract_recommendations.py`
- `scripts/07_verify_accuracy.py`（只适用于有知识库校验需求的问题）
- `scripts/08_generate_report.py`

`04_analyze_results.py` 会合并读取 `questions_expanded.json`、`questions_base.json` 和 `questions/imported_questions*.json`。

## 2. 问题层级标准

外部层级需要映射到统一链路阶段。养胃舒当前文件使用以下映射：

| 外部层级 | 标准 level | 链路含义 | 主要分析 |
| --- | --- | --- | --- |
| `解决方案` | `q3_solution` | 用户只有症状/需求，尚未明确用药 | 品类是否出现、品牌是否被自然带出 |
| `泛式吃药` | `q4_medicine` | 用户开始问“吃什么药” | 药品/品类推荐、竞品结构 |
| `中药相关` | `q5_tcm` | 用户限定中药或中成药方向 | 品牌名/通用名推荐、排名、理由 |

后续新增产品或新问题类型时，只新增映射，不改变分析统计口径。需要进入推荐抽取的层级，问题 ID 中必须包含 `_q3_`、`_q4_` 或 `_q5_`，以兼容现有 `05_extract_recommendations.py`。

## 3. 模型标准

模型字段分为“标准代码”和“展示名称”。

| 外部名称 | 标准代码 | 展示名称 |
| --- | --- | --- |
| `deepseek` | `deepseek` | `DeepSeek` |
| `kimi` | `kimi` | `Kimi` |
| `元宝` | `yuanbao` | `元宝` |
| `千问` | `qwen` | `通义千问` |
| `百度AI` | `baidu` | `百度AI` |
| `豆包` | `doubao` | `豆包` |

分析层可以接受任意模型代码；只有调用 API 采集数据时才需要 `config/models.yaml` 支持。

## 4. 联网字段标准

优先要求数据方提供显式字段：

- `联网`：`是` / `否`

如果没有显式字段，导入脚本按以下规则推断：

- `信源数量 > 0` 或 `引用信源` 非空，则 `search_enabled = true`
- 否则 `search_enabled = false`

注意：这只能代表“回答带引用信源”，不一定等同于模型真实联网模式。后续采集模板应补上显式 `联网` 字段。

## 5. 当前养胃舒文件映射

输入文件：

- `questions/【养胃舒】明细数据（AI回答）.xlsx`

输入字段映射：

| Excel 字段 | 标准字段 |
| --- | --- |
| `提问层级` | `source_level`，并映射到 `level` |
| `场景` | `scenario` |
| `问题` | `question_text` / 标准问题层 `question` |
| `查询轮次` | `round` |
| `查询时间` | `timestamp` |
| `AI模型` | `model` / `model_name` |
| `AI回答` | `answer` |
| `回答字数` | 可审计字段，分析时可重新计算 |
| `引用信源` | `sources[].url` |
| `信源数量` | `source_count`，并用于推断 `search_enabled` |

当前文件概况：

- 1 个工作表：`明细数据`
- 4,290 条回答
- 143 个唯一问题
- 6 个模型
- 5 轮重复
- 9 个场景

## 6. 导入命令

先做校验，不写文件：

```powershell
.\.venv\Scripts\python.exe scripts\import_external_answers.py --input "questions\【养胃舒】明细数据（AI回答）.xlsx" --sheet "明细数据" --dry-run
```

确认映射无误后正式导入：

```powershell
.\.venv\Scripts\python.exe scripts\import_external_answers.py --input "questions\【养胃舒】明细数据（AI回答）.xlsx" --sheet "明细数据"
```

正式导入会生成：

- `questions/imported_questions_weitai_yangweishu_external.json`
- `results/raw/<model_key>/*.json`
- `results/imports/weitai_yangweishu_external_manifest.json`

然后运行：

```powershell
.\.venv\Scripts\python.exe scripts\04_analyze_results.py
.\.venv\Scripts\python.exe scripts\05_extract_recommendations.py
```

## 7. 质检规则

导入前必须检查：

- 必填字段完整：层级、场景、问题、轮次、时间、模型、回答。
- 同一数据集内唯一键不重复：`question_id + model + search_enabled + round`。
- 每个唯一问题的轮次是否齐全。
- 每个模型的问题覆盖是否一致。
- 联网字段是否显式存在；没有时只能用信源推断。
- 产品代码、产品短名必须与报告口径一致。

## 8. 后续采集模板建议

后续所有外部采集表建议直接使用以下字段：

| 字段 | 说明 |
| --- | --- |
| `dataset_id` | 数据集 ID |
| `product` | 产品短名 |
| `product_code` | 产品代码 |
| `question_level` | 标准层级，如 `q3_solution` |
| `source_level` | 采集方原始层级 |
| `scenario` | 场景 |
| `question_id` | 若采集前已生成，则直接填 |
| `question` | 问题文本 |
| `model` | 标准模型代码 |
| `model_name` | 模型展示名 |
| `search_enabled` | 是否联网 |
| `round` | 轮次 |
| `timestamp` | 查询时间 |
| `answer` | AI 回答 |
| `sources` | 引用信源 URL，多个用分号分隔 |
| `source_count` | 信源数量 |

这样数据方给到的表可以直接进入标准回答层，分析方法不需要再按产品或文件单独适配。
