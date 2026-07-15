# 外部投稿与 AI 审稿运维

生产环境的 Web 与 Worker 使用同一个 ACR 镜像，不在 ECS 上构建：

```env
GEO_IMAGE=crpi-5c3cvh7cf04avg4p.cn-shenzhen.personal.cr.aliyuncs.com/financial-analyzer/geo-search:<commit>
GEO_ARTICLE_REVIEW_CONCURRENCY_MAX=5
GEO_ARTICLE_REVIEW_MEMORY_LIMIT=512m
```

发布顺序：先在管理后台暂停审稿队列，使用 SQLite backup API 备份 `data/geo_datasets/geo_answers.sqlite`（或停止 Web/Worker 后连同 WAL 一起备份），在本机或 CI 构建并推送 ACR；服务器执行 `docker compose pull` 和 `docker compose up -d --no-build --force-recreate`。数据库迁移只新增表和惰性列。

上线后检查 `geo-web` 健康状态、`geo-review-worker` 健康状态，以及“设置 → AI 审稿设置”中的 Worker 心跳。用一篇测试文章完成邀请、上传、AI 审稿、管理员确认和发布 URL 补充的端到端验证后，再恢复队列。

回滚时暂停队列，恢复 SQLite 备份并将 `GEO_IMAGE` 改回上一镜像标签，再 pull/recreate。迁移到更大的单机时，可先将环境上限提高到 `50` 并重建 Worker，再将后台并发动态调整到 `30`。需要多个 Worker 副本前，应先迁移 PostgreSQL/Redis 队列。
