# GEO 洞察平台（Web）

面向业务的 GEO 分析平台：品牌总览（健康卡+环比）、产品详情（消费者链路三阶段：病症→品类→品牌，含趋势与证据链下钻）、自助分析流水线、管理员工作台。

## 架构

- 后端：FastAPI（`backend/app`），仅接受门户 SSO 注入的身份和角色；账户、密码及用户管理由门户负责。
- 前端：Vue 3 + Vite + Element Plus + ECharts（`frontend/`），构建产物输出到 `backend/app/static`（该目录是构建产物，不进 git）。
- 数据：统一 SQLite `data/geo_datasets/geo_answers.sqlite`；指标在导入时物化（`manage_geo_sqlite.py materialize`），页面毫秒级响应。
- schema 单一来源：`utils/sqlite_schema.py`（CLI 与后端启动共用，惰性迁移）。

## 本地开发

后端启动前必须配置门户共享密钥；开发环境也需要由门户网关注入 `X-Portal-User`、`X-Portal-Role` 和 `X-Portal-Secret`。

```powershell
# 示例（请使用实际门户网关的密钥）
$env:GEO_PORTAL_SECRET = "replace-with-portal-secret"

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

运行数据与密钥通过 volume 挂载（见 docker-compose.yml）；启动前必须设置 `GEO_PORTAL_SECRET`，门户网关需使用同一密钥注入 `X-Portal-User` 和 `X-Portal-Role`，healthcheck 走 `/api/health`。

## 2026-09 审查修复后的运行说明

- 调查的自动基线仅取当前数据集同一归属用户的批次；普通用户读取调查时同时检查当前批次和基线权限。管理员仍可显式选择跨用户基线。
- 用户配置仅从门户用户名的 SHA-256 目录读取。旧版清洗用户名目录不会自动读取或删除：管理员应先备份并核对真实归属，再由对应用户在设置页重新保存品牌配置和知识库；不要仅根据旧目录名推断用户。
- 采集、分析、推荐抽取和准确率校验遇到执行失败会停止任务并显示失败，支持重试。采集及推荐抽取复用成功断点，校验阶段重新执行；缺少知识库导致的跳过仍按原逻辑处理。
- 推荐抽取日志升级为版本 2。旧版空缓存无法区分失败与真实空结果，会在下次重跑时重新抽取一次；新版本的成功空结果继续缓存。
- 更新部署时应停止旧审稿 Worker，再启动新 Web 和 Worker，避免旧进程继续按旧规则写入任务状态。应用 `scripts/nginx_geo_location.conf` 后，外部投稿入口及其资源由投稿会话鉴权，其余接口仍由门户鉴权。

### 回归测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## 历史数据回刷命令

```bash
python scripts/backfill_dataset.py --dataset-id <id> --sample 50   # 先抽样对比口径
python scripts/backfill_dataset.py --dataset-id <id>               # 全量重抽+物化
```
