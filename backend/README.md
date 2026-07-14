# GEO 洞察平台（Web）

面向业务的 GEO 分析平台：品牌总览（健康卡+环比）、产品详情（消费者链路三阶段：病症→品类→品牌，含趋势与证据链下钻）、自助分析流水线、管理员工作台。

## 架构

- 后端：FastAPI（`backend/app`），session cookie 认证，portal SSO 可选。
- 前端：Vue 3 + Vite + Element Plus + ECharts（`frontend/`），构建产物输出到 `backend/app/static`（该目录是构建产物，不进 git）。
- 数据：统一 SQLite `data/geo_datasets/geo_answers.sqlite`；指标在导入时物化（`manage_geo_sqlite.py materialize`），页面毫秒级响应。
- schema 单一来源：`utils/sqlite_schema.py`（CLI 与后端启动共用，惰性迁移）。

## 本地开发

```powershell
# 终端 1：后端（.venv）
.\start_dashboard.bat -Reload

# 终端 2：前端 dev server（代理 /api 到 8000）
cd frontend
npm install
npm run dev    # http://localhost:5173
```

仅验证生产形态时：`cd frontend && npm run build`，然后只起 uvicorn 访问 `http://127.0.0.1:8000`。

## 自助分析流水线

`backend/app/services/job_store.py` 单 worker 串行执行：
collect(03) → analyze(04) → extract(05, 含负面情感) → verify(07, 准确率) → import → materialize。
任务需关联产品主数据（趋势锚点）与批次日期；同一产品用相同问题集定期提交即可积累趋势。

## 关键 API

- `/api/insight/products`：品牌总览健康卡
- `/api/insight/products/{code}/journey`：三阶段详情
- `/api/insight/products/{code}/trend`：跨批次趋势（同问题集指纹才可比）
- `/api/insight/evidence` + `/api/insight/answers/...`：证据链下钻到 AI 原文
- `/api/insight/sources/options|analysis|answers`：信源分类、任意产品组合分析与域名下钻；域名分类目录位于 `config/source_domains.yaml`
- `/api/sqlite/*`：工作台明细；`/api/jobs*`：任务；`/api/auth/*`、`/api/config/*`

## Docker

```bash
docker compose up --build -d   # 多阶段构建：node 构建前端 → python 运行
```

运行数据与密钥通过 volume 挂载（见 docker-compose.yml），healthcheck 走 `/api/health`。

## 历史数据回刷

```bash
python scripts/backfill_dataset.py --dataset-id <id> --sample 50   # 先抽样对比口径
python scripts/backfill_dataset.py --dataset-id <id>               # 全量重抽+物化
```
