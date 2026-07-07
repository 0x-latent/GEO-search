# ---- 前端构建阶段：Vue 3 + Vite，产物拷入 FastAPI 静态目录 ----
FROM node:20-slim AS webbuild

WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm config set registry https://registry.npmmirror.com \
    && npm install --no-audit --no-fund
COPY frontend/ ./
# vite.config 默认输出到 ../backend/app/static，容器内改为本地 dist
RUN npm run build -- --outDir /web/dist --emptyOutDir

# ---- 运行阶段 ----
FROM python:3.11-slim

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
COPY questions /app/questions
COPY docs /app/docs

# 前端构建产物覆盖静态目录（本地 backend/app/static 是构建产物，不进 git）
COPY --from=webbuild /web/dist /app/backend/app/static

RUN mkdir -p /app/results /app/backend/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
