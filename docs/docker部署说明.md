# Docker 部署说明

## 打包原则

镜像只包含代码、依赖和默认配置模板；运行数据和密钥通过 volume 挂载。

**多阶段构建**：第一阶段用 node:20 构建 Vue 前端（`frontend/`），产物拷入
`/app/backend/app/static` 由 FastAPI 托管；第二阶段 python:3.11 运行。
本地不需要装 Node 也能出镜像。

不要打进镜像：

- `config/api_keys.yaml`
- `results/`
- `backend/data/`
- `.venv/`
- `frontend/node_modules/`
- 原始 Excel、压缩包

## 本地构建

```bash
docker build -t geo-search-workbench:latest .
```

## 国内服务器：docker.io 受限时

基础镜像支持 build args 覆盖。在服务器的部署目录放一个
`docker-compose.override.yml`（不进 git），compose 会自动合并：

```yaml
services:
  geo-web:
    build:
      args:
        NODE_IMAGE: docker.m.daocloud.io/library/node:20-alpine
        PYTHON_IMAGE: docker.m.daocloud.io/library/python:3.11-slim
```

或使用 compose：

```bash
docker compose up --build -d
```

访问：

```text
http://localhost:8000
```

## 运行前准备

确保宿主机存在以下文件或目录：

```text
config/api_keys.yaml
config/brands.yaml
config/knowledge_base.json
questions/
docs/products_db/
results/
backend/data/
```

如果没有 API key 文件：

```bash
cp config/api_keys.yaml.example config/api_keys.yaml
```

## 推送阿里云 ACR

示例：

```bash
docker login registry.cn-hangzhou.aliyuncs.com
docker tag geo-search-workbench:latest registry.cn-hangzhou.aliyuncs.com/<namespace>/geo-search-workbench:<tag>
docker push registry.cn-hangzhou.aliyuncs.com/<namespace>/geo-search-workbench:<tag>
```

ECS 上拉取并启动：

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/<namespace>/geo-search-workbench:<tag>
docker compose up -d
```

## 升级部署注意（v2 业务化改版）

- 首次启动会自动迁移 SQLite schema（新增 dataset_products / metrics_* / metric_evidence 表和 datasets 批次列），并从 brands.yaml 同步产品主数据——无需手工操作。
- 老库升级后需要跑一次物化才能出品牌总览/产品详情数据：
  `docker compose exec geo-web python scripts/manage_geo_sqlite.py materialize --dataset-id all`
- compose 已带 healthcheck（`/api/health`，公开路径）。

## 生产注意

当前是单容器 MVP：Web 服务和脚本任务 runner 在同一个容器内。容器重启会中断正在运行的脚本任务（任务会在重启后自动重新入队，03/05 有断点续跑）。

后续多人使用或任务量变大时，应拆成：

- `geo-web`
- `geo-worker`
- `redis`
- `postgres`
- OSS 或独立数据盘保存 `results/`

