# 基础镜像可用 build args 覆盖（国内服务器 docker.io 受限时指向镜像源，
# 见 docs/docker部署说明.md 的 docker-compose.override.yml 示例）
# Vite 8 要求 Node ^20.19 或 >=22.12，用 22 保证余量
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.11-slim

# ---- 前端构建阶段：Vue 3 + Vite 8 + pnpm，产物拷入 FastAPI 静态目录 ----
FROM ${NODE_IMAGE} AS webbuild

WORKDIR /web
# corepack 下载 pnpm 也走国内镜像
ENV COREPACK_NPM_REGISTRY=https://registry.npmmirror.com
RUN corepack enable && corepack prepare pnpm@10.33.0 --activate
# vendor/ 内是共享包 tgz（file: 依赖），必须先于 install 进入镜像
COPY frontend/package.json frontend/pnpm-lock.yaml ./
COPY frontend/vendor ./vendor
RUN pnpm config set registry https://registry.npmmirror.com \
    && pnpm install --frozen-lockfile
COPY frontend/ ./
# vite.config 默认输出到 ../backend/app/static，容器内改为本地 dist
RUN pnpm build --outDir /web/dist --emptyOutDir

# ---- 运行阶段 ----
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY scripts /app/scripts
COPY utils /app/utils
COPY config /app/config
COPY docs /app/docs

# 前端构建产物覆盖静态目录（本地 backend/app/static 是构建产物，不进 git）
COPY --from=webbuild /web/dist /app/backend/app/static

# questions/results 都是活数据,由 compose 挂载进来(服务器上它们软链到
# /data/appdata/geo_search/*)。镜像里只建空目录:
#   1) 打包 context 时指向 context 之外的软链解析不了,COPY 会直接失败;
#   2) 挂载会盖掉镜像内容,COPY 进去的那份本来就是死的
#      —— QUESTIONS_DIR 在 backend/app/core/paths.py 里定义了但全仓库没有使用点。
RUN mkdir -p /app/questions /app/results /app/backend/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
