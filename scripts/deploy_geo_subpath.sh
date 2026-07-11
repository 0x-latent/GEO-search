#!/bin/bash
# geo 子路径优化部署脚本（在本地 GEO-search-web 目录执行）
# 前提：fix/geo-subpath 分支代码；服务器 /opt/geo_search 为部署目录。
# 步骤：同步代码 -> 重建镜像 -> 换 nginx 配置（去 sub_filter）-> 平滑重载 -> 冒烟测试
set -e
cd "$(dirname "$0")/.."

echo "== 1/5 同步代码到服务器（不碰 config/questions/results/backend/data 等活数据）=="
rsync -avz --exclude node_modules --exclude __pycache__ --exclude backend/app/static \
  ./frontend ./backend/app ./backend/__init__.py ./scripts ./utils \
  ./requirements.txt ./Dockerfile ./.dockerignore \
  aliyun:/opt/geo_search/

echo "== 2/5 服务器重建镜像，并收回 8001 公网直连（GEO_BIND=127.0.0.1）=="
ssh aliyun 'cd /opt/geo_search && touch .env \
  && (grep -q "^GEO_BIND=" .env && sed -i "s/^GEO_BIND=.*/GEO_BIND=127.0.0.1/" .env || echo "GEO_BIND=127.0.0.1" >> .env) \
  && docker compose build'

echo "== 3/5 备份并替换 nginx /geo/ 配置段 =="
ssh aliyun 'cp /etc/nginx/sites-enabled/aigc-creative-workflow \
  /etc/nginx/sites-available/aigc-creative-workflow.bak-geo-subpath-$(date +%Y%m%d-%H%M)'
scp scripts/nginx_geo_location.conf aliyun:/tmp/nginx_geo_location.conf
# 门户共享密钥不进 git：从服务器现有配置中提取后填入模板
ssh aliyun 'python3 - <<EOF
import re
p = "/etc/nginx/sites-enabled/aigc-creative-workflow"
text = open(p).read()
secret = re.search(r"X-Portal-Secret (\S+);", text).group(1)
new_block = open("/tmp/nginx_geo_location.conf").read().replace("__PORTAL_SECRET__", secret)
# 替换整个 location /geo/ { ... } 块（该块无嵌套花括号）
pattern = re.compile(r"    location /geo/ \{.*?\n    \}\n", re.S)
assert pattern.search(text), "location /geo/ 块未找到"
open(p, "w").write(pattern.sub(new_block, text))
print("nginx 配置已替换")
EOF'

echo "== 4/5 切换：新容器 + nginx 重载 =="
ssh aliyun 'cd /opt/geo_search && docker compose up -d && nginx -t && systemctl reload nginx'

echo "== 5/5 冒烟测试 =="
ssh aliyun '
  sleep 5
  echo "--- 容器健康 ---"
  docker ps --filter name=geo --format "{{.Names}} {{.Status}}"
  echo "--- 端口绑定（应只剩 127.0.0.1）---"
  docker port geo_search-geo-web-1
  echo "--- 直连入口（服务器本机回环）---"
  curl -s -o /dev/null -w "GET /api/health -> %{http_code}\n" http://127.0.0.1:8001/api/health
  curl -s -o /dev/null -w "GET /login.html -> %{http_code}\n" http://127.0.0.1:8001/login.html
  curl -s -o /dev/null -w "GET / 未登录 -> %{http_code} Location=%{redirect_url}\n" http://127.0.0.1:8001/
  echo "--- 子路径入口（无门户 cookie，应 302 到 /portal/login）---"
  curl -s -o /dev/null -w "GET /geo/ -> %{http_code} Location=%{redirect_url}\n" http://127.0.0.1/geo/
'
echo "部署完成。浏览器验证：登录门户后进 /geo/，检查请求都打到 /geo/api/*。"
