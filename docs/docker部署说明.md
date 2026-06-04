# Docker 部署说明

## 打包原则

镜像只包含代码、依赖和默认配置模板；运行数据和密钥通过 volume 挂载。

不要打进镜像：

- `config/api_keys.yaml`
- `results/`
- `backend/data/`
- `.venv/`
- 原始 Excel、压缩包

## 本地构建

```bash
docker build -t geo-search-workbench:latest .
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

## 生产注意

当前是单容器 MVP：Web 服务和脚本任务 runner 在同一个容器内。容器重启会中断正在运行的脚本任务。

后续多人使用或任务量变大时，应拆成：

- `geo-web`
- `geo-worker`
- `redis`
- `postgres`
- OSS 或独立数据盘保存 `results/`

